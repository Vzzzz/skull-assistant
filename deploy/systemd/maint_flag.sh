#!/usr/bin/env bash
# Prints the path of the maintenance flag file (service mode, D-014): when it exists, no skull unit starts.
# Override with SKULL_MAINT_FLAG. Default: on the stand (WSL) — the Windows user's home, so Git Bash can set it
# while WSL is down (deploy/stand/skull-stand.sh); elsewhere (Jetson) — /var/lib/skull/maintenance.
if [ -n "${SKULL_MAINT_FLAG:-}" ]; then echo "$SKULL_MAINT_FLAG"; exit 0; fi
if [ -d /mnt/c/Users ] && command -v cmd.exe >/dev/null 2>&1; then
  WINUSER="$(cd /mnt/c && cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r\n')"
  if [ -n "$WINUSER" ] && [ -d "/mnt/c/Users/$WINUSER" ]; then echo "/mnt/c/Users/$WINUSER/.skull-maintenance"; exit 0; fi
fi
echo /var/lib/skull/maintenance
