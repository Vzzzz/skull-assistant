#!/usr/bin/env bash
# Запуск стенда Skull Assistant в WSL из Git Bash (Windows).
#
#   skull-stand.sh service   флаг + WSL без skull-сервисов и ollama (скачать модели, пакеты)
#   skull-stand.sh normal    снять флаг + WSL, сервисы стартуют
#   skull-stand.sh status    флаг, состояние WSL и юнитов, звук (Windows и pulse WSLg)
#   skull-stand.sh restart [--mode A|B] [--drop-caches] [--fresh]
#                            перезапуск стенда без wsl --shutdown (deploy/stand/skull-restart.sh, T-047) + звук
#   skull-stand.sh stop      остановить WSL целиком (все сервисы гаснут)
#
# Сервисный режим = флаг-файл $HOME/.skull-maintenance (= /mnt/c/Users/<user>/.skull-maintenance
# в WSL). Юниты skull-*, skull.target и ollama имеют ConditionPathExists=!<флаг> и при его
# наличии не стартуют вместе с WSL. Флаг ставится/снимается при выключенном WSL, гонки нет.
set -u
MODE="${1:-status}"
DISTRO="${DISTRO:-Ubuntu-22.04}"
FLAG="$HOME/.skull-maintenance"
PROJECT='~/skull-assistant'

wslq() { wsl.exe -d "$DISTRO" -- bash -lc "$1" 2>&1 | tr -d '\0'; return "${PIPESTATUS[0]}"; }

# I-15: звук WSLg идёт через msrdc.exe в устройство вывода Windows по умолчанию; Bluetooth-гарнитура (A2DP) там
# тормозит аудиодвижок, и pulse WSLg встаёт. Имена сверяются по ASCII-части (Realtek, WH-CH520): кириллица в выводе
# PowerShell зависит от кодовой страницы. Код 3 — устройство вывода не динамики.
winsound() {
  local out rc=0
  out=$(powershell.exe -NoProfile -Command '
    [Console]::OutputEncoding = [Text.Encoding]::UTF8
    [Windows.Media.Devices.MediaDevice,Windows.Media.Devices,ContentType=WindowsRuntime] | Out-Null
    $eps = Get-PnpDevice -Class AudioEndpoint
    function name($id) { $g = ($id -split "#")[2]; ($eps | Where-Object { $_.InstanceId -like "*$g" } | Select-Object -First 1).FriendlyName }
    "OUT|" + (name ([Windows.Media.Devices.MediaDevice]::GetDefaultAudioRenderId(0)))
    "IN|" + (name ([Windows.Media.Devices.MediaDevice]::GetDefaultAudioCaptureId(0)))
    $eps | Where-Object { $_.FriendlyName -like "*WH-CH520 Stereo*" -and $_.Status -eq "OK" } | ForEach-Object { "HEADSET|" + $_.FriendlyName }
    Get-ChildItem "HKCU:\Software\Microsoft\Multimedia\Audio\DefaultEndpoint" -ErrorAction SilentlyContinue | ForEach-Object {
      $p = Get-ItemProperty $_.PSPath
      if ($p."(default)" -like "*msrdc.exe") {
        $v = ($p.PSObject.Properties | Where-Object { $_.Name -notlike "PS*" -and $_.Name -ne "(default)" } | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join "; "
        "PIN|" + $_.PSChildName + ": " + $(if ($v) { $v } else { "ключ есть, устройств в нём нет" })
      }
    }' 2>/dev/null | tr -d '\r')
  local outdev indev
  outdev=$(echo "$out" | sed -n 's/^OUT|//p'); indev=$(echo "$out" | sed -n 's/^IN|//p')
  echo "Windows: вывод по умолчанию — ${outdev:-?}; ввод — ${indev:-?}"
  echo "$out" | sed -n 's/^PIN|/Windows: закрепление msrdc.exe (DefaultEndpoint) — /p'
  if ! echo "$outdev" | grep -q 'Realtek'; then
    echo "ЗВУК WSLg уходит в ${outdev:-?} (I-15): закрепи msrdc.exe за динамиками или отключи устройство"; rc=3
  fi
  echo "$out" | grep -q '^HEADSET|' && echo "ВНИМАНИЕ: подключена гарнитура WH-CH520 (I-15) — переключение на гарнитуру (HFP) закреплением не покрыто; на время стенда лучше отключить"
  return $rc
}


case "$MODE" in
  service)
    : > "$FLAG"
    echo "[service] флаг установлен: $FLAG — skull-* и ollama не стартуют"
    # если WSL уже работал — погасить то, что успело подняться
    wslq 'systemctl --user stop skull.target 2>/dev/null; systemctl is-active --quiet ollama && echo "ollama активен: sudo systemctl stop ollama" || true'
    exec wsl.exe -d "$DISTRO" --cd "$PROJECT"
    ;;
  normal)
    if [ -e "$FLAG" ]; then rm -f "$FLAG"; echo "[normal] флаг снят"; fi
    wslq 'systemctl --user start skull.target 2>/dev/null; systemctl is-active --quiet ollama || echo "ollama не активен: sudo systemctl start ollama"'
    exec wsl.exe -d "$DISTRO" --cd "$PROJECT"
    ;;
  status)
    if [ -e "$FLAG" ]; then echo "флаг сервисного режима: ЕСТЬ (сервисы не стартуют)"; else echo "флаг сервисного режима: нет"; fi
    wsl.exe -l -v 2>&1 | tr -d '\0'
    wslq 'systemctl --user list-units "skull*" --no-pager 2>/dev/null; echo "ollama: $(systemctl is-active ollama 2>/dev/null)"; timeout 10 ollama ps 2>/dev/null'
    winsound; w=$?
    wslq "cd $PROJECT && deploy/stand/skull-restart.sh --check-sound"; p=$?
    if [ $w -ne 0 ] || [ $p -ne 0 ]; then exit 3; fi
    ;;
  restart)
    shift
    winsound; w=$?
    wslq "cd $PROJECT && deploy/stand/skull-restart.sh $*"; r=$?
    [ $r -ne 0 ] && exit $r
    exit $w
    ;;
  stop)
    wsl.exe -t "$DISTRO" && echo "[stop] $DISTRO остановлен"
    ;;
  *)
    echo "usage: $0 {service|normal|status|restart|stop}" >&2; exit 2
    ;;
esac
