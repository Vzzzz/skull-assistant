#!/usr/bin/env bash
# Устанавливает юниты из этой папки как user-юниты systemd (symlink) и включает их.
# Идемпотентно. Пути внутри юнитов — абсолютные, под текущее расположение репозитория.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DST="$HOME/.config/systemd/user"
mkdir -p "$DST"

for unit in "$HERE"/skull-*.service; do
  name="$(basename "$unit")"
  # подставляем реальный путь репозитория (в юнитах используется плейсхолдер @REPO@)
  sed "s|@REPO@|$REPO|g" "$unit" > "$DST/$name"
  echo "installed $name"
done

systemctl --user daemon-reload
systemctl --user enable skull-capture skull-wakeword skull-stt skull-brain skull-tts
echo "done. start: systemctl --user start skull-capture skull-wakeword skull-stt skull-brain skull-tts"
