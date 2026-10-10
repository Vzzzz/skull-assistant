#!/usr/bin/env bash
# Ставит drop-in с условием сервисного режима на системный ollama.service. Требует sudo.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
sudo install -D -m 644 "$HERE/ollama-maintenance.conf" /etc/systemd/system/ollama.service.d/maintenance.conf
sudo systemctl daemon-reload
echo "ollama drop-in installed:"; systemctl cat ollama.service | grep -A1 maintenance.conf
