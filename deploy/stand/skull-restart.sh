#!/usr/bin/env bash
# Перезапуск стенда без wsl --shutdown (T-047): все процессы с моделями убиты, модели загружены заново, режим A.
# Запускается внутри WSL; из Git Bash — `deploy/stand/skull-stand.sh restart [флаги]`.
#
#   deploy/stand/skull-restart.sh [--mode A|B] [--drop-caches] [--fresh]
#     --mode B        поднять сразу режим B (mode:current до старта; reconcile диспетчера и llm_server читают его)
#     --drop-caches   сбросить кэш страниц (нужен sudo без пароля; иначе подсказка и дальше)
#     --fresh         очистить dialog:history и dialog:vec (по умолчанию история и память не трогаются)
#     --check-sound   только проверить pulse WSLg (skull-stand.sh status), ничего не перезапуская
#
# Коды выхода: 0 — стенд поднят; 1 — не поднялся за 120 с; 2 — сервисный режим (флаг) или неверные аргументы;
# 3 — стенд поднят, но звук WSLg завис (I-15) и не ожил: нужен wsl --shutdown.
# Звук: pulse живёт в системном дистрибутиве WSLg, но заклинивает его тракт микрофона (rdp-source, `q overrun` в
# /mnt/wslg/pulseaudio.log); когда юниты остановлены и поток записи закрыт, msrdc переподключает rdp-source и pulse
# оживает за ~20 с (T-047, 2026-10-31) — поэтому между stop и start скрипт ждёт pulse до 60 с.
set -u
cd "$(dirname "$0")/../.." || exit 2
MODE=A; DROP=0; FRESH=0; CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift ;;
    --drop-caches) DROP=1 ;;
    --fresh) FRESH=1 ;;
    --check-sound) CHECK=1 ;;
    *) echo "usage: $0 [--mode A|B] [--drop-caches] [--fresh]" >&2; exit 2 ;;
  esac
  shift
done
case "$MODE" in A|B) ;; *) echo "--mode: A или B" >&2; exit 2 ;; esac

FLAG="$(deploy/systemd/maint_flag.sh)"

PY=.venv/bin/python
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'; }
units() { systemctl --user list-dependencies skull.target --plain --no-pager 2>/dev/null | sed -n '2,$p' | tr -d ' '; }
# мёртвы: юниты не active/deactivating (systemd убивает весь cgroup юнита), на GPU никого, llama-server нет и вне юнитов;
# не `pgrep -f` по командной строке — она ловит и чужие шеллы со словами «core.dispatcher» в аргументах
alive() { for u in $(units); do systemctl --user is-active --quiet "$u" && echo "$u"; \
  [ "$(systemctl --user show -p ActiveState --value "$u")" = deactivating ] && echo "$u"; done; pgrep -x llama-server; }
health() { curl -s -m 2 "$1/health" 2>/dev/null | grep -q '"ok"'; }
manifest() { $PY -c "import yaml; p = yaml.safe_load(open('services/brain/manifest.yaml'))['params']; print($1)"; }

# pulse WSLg: тайм-аут и «Connection refused» (очередь сокета полна, главный цикл встал) — одно и то же зависание
sound() {
  local out rc
  out=$(timeout 5 pactl info 2>&1); rc=$?
  if [ $rc -eq 124 ] || echo "$out" | grep -qi 'refused\|connection failure'; then
    echo "звук WSLg завис (I-15): нужен wsl --shutdown"; return 3
  fi
  [ $rc -eq 0 ] && { echo "звук ok"; return 0; }
  echo "pactl: ${out:0:80}"; return 3
}

[ $CHECK = 1 ] && { sound; exit $?; }        # и в сервисном режиме: флаг гасит юниты, не pulse
if [ -e "$FLAG" ]; then echo "сервисный режим ($FLAG): перезапуск не делаю"; exit 2; fi

VRAM0=$(vram); T0=$(date +%s%3N)
systemctl --user stop skull.target
dead=0
for i in $(seq 1 30); do
  if [ "$(gpu_pids)" = 0 ] && [ -z "$(alive)" ]; then dead=1; break; fi
  sleep 1
done
if [ $dead = 0 ]; then
  echo "процессы живы через 30 с: KILL"
  for u in $(units); do systemctl --user kill -s KILL "$u" 2>/dev/null; done
  for i in $(seq 1 10); do
    if [ "$(gpu_pids)" = 0 ] && [ -z "$(alive)" ]; then dead=1; break; fi
    sleep 1
  done
  [ $dead = 0 ] && { echo "не умерли: $(alive | tr '\n' ' ')"; exit 1; }
fi
for m in $(timeout 10 ollama ps 2>/dev/null | awk 'NR > 1 { print $1 }'); do ollama stop "$m"; done
VRAM_MID=$(vram); GPU_MID=$(gpu_pids)
PULSE="звук до старта ok"
if ! sound >/dev/null; then
  for i in $(seq 1 60); do sound >/dev/null && break; sleep 1; done
  sound >/dev/null && PULSE="звук ожил после остановки захвата за $i с" || PULSE="звук не ожил за 60 с"
fi
T1=$(date +%s%3N)

redis-cli set mode:current "$MODE" >/dev/null
redis-cli del mode:transition mode:llm tts:speaking >/dev/null
[ $FRESH = 1 ] && redis-cli del dialog:history dialog:vec >/dev/null && echo "история диалога и память очищены"
if [ $DROP = 1 ]; then
  if sudo -n true 2>/dev/null; then sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; echo "кэш страниц сброшен"
  else echo "--drop-caches: нужен sudo без пароля (sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'), пропускаю"; fi
fi

systemctl --user start skull.target
LLM=$(manifest 'p["host"]'); EMB=$(manifest 'p["memory"]["embed_host"]')
up=0
for i in $(seq 1 120); do
  bad=$(for u in $(units); do systemctl --user is-active --quiet "$u" || echo "$u"; done)
  st=$($PY -m core.dispatcher --status 2>/dev/null | head -1)
  if [ -z "$bad" ] && health "$LLM" && health "$EMB" \
     && echo "$st" | grep -q "mode:current $MODE; transition None; skull-llm holds b'$MODE'"; then up=1; break; fi
  sleep 1
done
T2=$(date +%s%3N)
sec() { printf '%d.%d' $(($1 / 1000)) $(($1 % 1000 / 100)); }
STOP_S=$(sec $((T1 - T0))); START_S=$(sec $((T2 - T1)))
if [ $up = 0 ]; then
  echo "стенд не поднялся за 120 с: неактивны [${bad:-—}]; $st"
  exit 1
fi
SND=$(sound); RC=$?
echo "перезапуск: stop ${STOP_S} с (GPU-процессов после: $GPU_MID), start ${START_S} с; VRAM ${VRAM0} → ${VRAM_MID} → $(vram) МБ; режим $MODE; $PULSE; $SND"
exit $RC
