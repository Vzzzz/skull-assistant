#!/usr/bin/env bash
# Подтянуть Windows-зеркало (C:\Users\valer\git\skull-assistant) до состояния WSL-репозитория.
# Запуск из WSL: deploy/sync-win.sh   (после коммита в ~/skull-assistant)
set -euo pipefail
WIN=/mnt/c/Users/valer/git/skull-assistant
if [ -n "$(git -C ~/skull-assistant status --porcelain)" ]; then
  echo "в ~/skull-assistant есть незакоммиченные изменения — сначала commit" >&2; exit 1
fi
git -C "$WIN" pull -q --ff-only origin main
echo "win: $(git -C "$WIN" log --oneline -1)"
