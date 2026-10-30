#!/usr/bin/env bash
# Ручной переход режима стенда из Git Bash (Windows) — T-044, D-030.
#
#   skull-mode.sh A        переход в режим A (разговор: VLM 4B + голос + эмбеддер)
#   skull-mode.sh B        переход в режим B (думающий: 26B-A4B, эксперты на CPU)
#   skull-mode.sh status   mode:current, переход в процессе, юниты по манифестам
#
# Команда уходит в Redis (stream cmd:mode) через `python -m core.dispatcher --switch`; переход делает юнит
# skull-dispatcher, фазы печатаются по мере events:mode. Внутри WSL то же самое:
#   .venv/bin/python -m core.dispatcher --switch B
set -u
MODE="${1:-status}"
DISTRO="${DISTRO:-Ubuntu-22.04}"
PROJECT='~/skull-assistant'

wslq() { wsl.exe -d "$DISTRO" -- bash -lc "cd $PROJECT && $1" 2>&1 | tr -d '\0'; }

case "$MODE" in
  A|B) wslq ".venv/bin/python -m core.dispatcher --switch $MODE --reason skull-mode.sh" ;;
  status) wslq ".venv/bin/python -m core.dispatcher --status" ;;
  *) echo "usage: $0 A|B|status"; exit 2 ;;
esac
