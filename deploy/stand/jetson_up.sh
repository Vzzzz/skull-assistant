#!/usr/bin/env bash
# Bring the Jetson link up from WSL after the board is powered on (JETSON_BRINGUP §0a): the forward proxy, the reverse
# tunnel that gives the board internet, the board clock (its RTC does not hold time — no NTP without internet) and a check.
#   deploy/stand/jetson_up.sh        # from WSL; needs `Host jetson` in ~/.ssh/config and NOPASSWD sudo on the board
cd ~/skull-assistant || exit 1
if ! pgrep -f jetson_proxy.py >/dev/null; then
  nohup python3 deploy/stand/jetson_proxy.py --port 3128 >/tmp/jetson_proxy.log 2>&1 &
  sleep 1; echo "proxy started"
else echo "proxy running"; fi
if ! pgrep -f "ssh.*-R 3128:127.0.0.1:3128 jetson" >/dev/null; then
  ssh -fN -o BatchMode=yes -o ExitOnForwardFailure=yes -R 3128:127.0.0.1:3128 jetson && echo "tunnel started"
else echo "tunnel running"; fi
now=$(date -u +%s)
ssh -o BatchMode=yes jetson "sudo -n date -u -s @$now >/dev/null && sudo -n hwclock -w 2>/dev/null; echo 'board clock:' \$(date)
export https_proxy=http://127.0.0.1:3128 http_proxy=http://127.0.0.1:3128
timeout 8 curl -sS -o /dev/null -w 'github via tunnel: %{http_code}\n' https://github.com/ || echo 'no internet via tunnel'
cd ~/llama.cpp 2>/dev/null && git log --oneline -1 | sed 's/^/board llama.cpp: /'"
