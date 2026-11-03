#!/usr/bin/env bash
# Подготовка стенда к дублю демо (docs/DEMO.md «Подготовка непосредственно перед дублем»): исполнитель, внутри WSL.
# Из Git Bash: wsl -d Ubuntu-22.04 -- bash -lc 'cd ~/skull-assistant && deploy/stand/demo_prep.sh'
#
#   deploy/stand/demo_prep.sh [--no-warm] [--record]
#     --no-warm   без прогрева режима B (по умолчанию: B и обратно A один раз — первый переход в кадре не холодный)
#     --record    после подготовки — запись шины в research/bench/results/demo_take_<дата>_<время>_bus.json (в фоне)
#
# Шаги: restart --fresh (чистая история и память, режим A) → сеть `ask` → прогрев B → камера и skull-vision →
# сводка (юниты, режим, сеть, история, GPU) → что остаётся владельцу. Коды выхода: 0 — готово; 1 — не готово (что
# именно — в последней строке); 2 — сервисный режим или неверные аргументы; 3 — звук WSLg (I-15): нужен wsl --shutdown.
set -u
cd "$(dirname "$0")/../.." || exit 2
WARM=1; REC=0
for a in "$@"; do
  case "$a" in
    --no-warm) WARM=0 ;;
    --record) REC=1 ;;
    *) echo "usage: $0 [--no-warm] [--record]" >&2; exit 2 ;;
  esac
done
PY=.venv/bin/python
fail() { echo "НЕ ГОТОВО: $*"; exit 1; }
wait_mode() {   # wait_mode A|B timeout_s: mode:current = $1 and no transition in progress
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$2" ]; do
    [ "$(redis-cli get mode:current)" = "$1" ] && [ "$(redis-cli exists mode:transition)" = "0" ] && return 0
    sleep 1
  done
  return 1
}

echo "[1/5] перезапуск с чистой историей (skull-restart.sh --fresh)"
deploy/stand/skull-restart.sh --fresh; rc=$?
[ $rc -ne 0 ] && { echo "skull-restart.sh: код $rc"; exit $rc; }

echo "[2/5] сеть: ask"
$PY services/brain/net.py policy ask

if [ $WARM -eq 1 ]; then
  echo "[3/5] прогрев режима B (B и обратно A)"
  for m in B A; do
    t0=$SECONDS
    $PY -m core.dispatcher --switch $m > /dev/null || fail "переход в $m не принят диспетчером"
    wait_mode $m 180 || fail "переход в $m не завершился за 180 с (journalctl --user -u skull-dispatcher)"
    echo "      режим $m за $((SECONDS - t0)) с"
  done
else
  echo "[3/5] прогрев B пропущен (--no-warm)"
fi

echo "[4/5] камера"
ls /dev/video0 > /dev/null 2>&1 || fail "нет /dev/video0 — камера не проброшена (Windows: usbipd attach --wsl --busid <id>)"
systemctl --user is-active --quiet skull-vision || systemctl --user start skull-vision
sleep 3
systemctl --user is-active --quiet skull-vision || fail "skull-vision не активен (journalctl --user -u skull-vision)"
echo "      /dev/video0 есть, skull-vision активен"

echo "[5/5] сводка"
down=$(systemctl --user list-units 'skull*' --type=service --no-legend --all | awk '$3 != "active" {print $1}')
[ -n "$down" ] && fail "не активны: $down"
mode=$(redis-cli get mode:current); hist=$(redis-cli llen dialog:history)
gpu=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
echo "      юниты активны; режим ${mode:-?}; $($PY services/brain/net.py policy status | cut -d';' -f1); история $hist; GPU ${gpu/,//} МБ"
[ "$mode" = "A" ] || fail "режим $mode, нужен A"
[ "$hist" = "0" ] || fail "история не пуста ($hist)"

if [ $REC -eq 1 ]; then
  out=research/bench/results/demo_take_$(date +%F_%H%M)_bus.json
  nohup $PY deploy/stand/skull_console.py --record "$out" > /dev/null 2>&1 &
  echo "      запись шины: $out (pid $!; остановить: kill $!)"
fi

cat <<'EOF'

ГОТОВО. Остаётся владельцу:
  - гарнитура Bluetooth отключена (I-15); устройство звука Windows — динамики ноутбука;
  - консоль на правой половине экрана, из Git Bash:
      wsl -d Ubuntu-22.04 -- bash -lc 'cd ~/skull-assistant && .venv/bin/python deploy/stand/skull_console.py --demo'
  - запись экрана включена, телефон на штативе, хлопок перед сценой 1.
EOF
