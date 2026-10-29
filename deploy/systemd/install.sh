#!/usr/bin/env bash
# Устанавливает skull.target и юниты из этой папки как user-юниты systemd и включает их.
# Идемпотентно. Плейсхолдеры: @REPO@ — путь репозитория, @MAINT_FLAG@ — флаг сервисного режима
# (юниты не стартуют, пока файл существует; путь — deploy/systemd/maint_flag.sh, переопределение SKULL_MAINT_FLAG).
# См. deploy/stand/skull-stand.sh и README «Сервисный режим».
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DST="$HOME/.config/systemd/user"
MAINT_FLAG="$(bash "$HERE/maint_flag.sh")"
mkdir -p "$DST"
echo "maintenance flag: $MAINT_FLAG"

for unit in "$HERE"/skull.target "$HERE"/skull-*.service; do
  name="$(basename "$unit")"
  sed -e "s|@REPO@|$REPO|g" -e "s|@MAINT_FLAG@|$MAINT_FLAG|g" "$unit" > "$DST/$name"
  echo "installed $name"
done

systemctl --user daemon-reload
# юниты теперь WantedBy=skull.target; старые симлинки в default.target.wants убираем
for s in skull-capture skull-wakeword skull-stt skull-brain skull-tts; do
  rm -f "$DST/default.target.wants/$s.service"
done
systemctl --user enable skull.target skull-capture skull-wakeword skull-stt skull-llm skull-embed skull-brain skull-tts skull-vision
echo "done."
echo "  start all : systemctl --user start skull.target"
echo "  stop all  : systemctl --user stop skull.target"
echo "  status    : systemctl --user list-units 'skull*'"
