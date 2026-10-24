#!/usr/bin/env bash
# Ставит drop-in с условием сервисного режима на системный ollama.service. Требует sudo.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MAINT_FLAG="$(bash "$HERE/maint_flag.sh")"
TMP="$(mktemp)"; sed "s|@MAINT_FLAG@|$MAINT_FLAG|g" "$HERE/ollama-maintenance.conf" > "$TMP"
sudo install -D -m 644 "$TMP" /etc/systemd/system/ollama.service.d/maintenance.conf; rm -f "$TMP"
sudo systemctl daemon-reload
echo "ollama drop-in installed:"; systemctl cat ollama.service | grep -A1 maintenance.conf
