#!/usr/bin/env bash
# Устанавливает skull.target и юниты из этой папки как user-юниты systemd и включает их.
# Идемпотентно. Плейсхолдер @REPO@ заменяется на реальный путь репозитория.
# Сервисный режим (юниты не стартуют): флаг-файл /mnt/c/Users/valer/.skull-maintenance,
# см. deploy/stand/skull-stand.ps1 и README «Сервисный режим».
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DST="$HOME/.config/systemd/user"
mkdir -p "$DST"

for unit in "$HERE"/skull.target "$HERE"/skull-*.service; do
  name="$(basename "$unit")"
  sed "s|@REPO@|$REPO|g" "$unit" > "$DST/$name"
  echo "installed $name"
done

systemctl --user daemon-reload
# юниты теперь WantedBy=skull.target; старые симлинки в default.target.wants убираем
for s in skull-capture skull-wakeword skull-stt skull-brain skull-tts; do
  rm -f "$DST/default.target.wants/$s.service"
done
systemctl --user enable skull.target skull-capture skull-wakeword skull-stt skull-llm skull-embed skull-brain skull-tts
echo "done."
echo "  start all : systemctl --user start skull.target"
echo "  stop all  : systemctl --user stop skull.target"
echo "  status    : systemctl --user list-units 'skull*'"
