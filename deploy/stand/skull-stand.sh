#!/usr/bin/env bash
# Запуск стенда Skull Assistant в WSL из Git Bash (Windows).
#
#   skull-stand.sh service   флаг + WSL без skull-сервисов и ollama (скачать модели, пакеты)
#   skull-stand.sh normal    снять флаг + WSL, сервисы стартуют
#   skull-stand.sh status    флаг, состояние WSL и юнитов
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

wslq() { wsl.exe -d "$DISTRO" -- bash -lc "$1" 2>&1 | tr -d '\0'; }

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
    ;;
  stop)
    wsl.exe -t "$DISTRO" && echo "[stop] $DISTRO остановлен"
    ;;
  *)
    echo "usage: $0 {service|normal|status|stop}" >&2; exit 2
    ;;
esac
