#!/usr/bin/env bash
# T-056: memory of one skull unit on the Jetson (unified memory, no nvidia-smi). Run ON the board.
#   bash research/bench/orin_unit_mem.sh <unit> [hold_s=30] [--stop] >> research/bench/results/orin_units_<date>.csv
# Units already running stay up, so the delta is what this unit adds on top of them (bring-up one by one, T-056 step 4).
# Points: before (RAM used mean over 10 s), peak over hold_s after start (tegrastats 500 ms), VmHWM / RssAnon / RssFile
# of the unit's main PID (skull-llm / skull-embed exec into llama-server — same PID), /proc/meminfo Cached before/after;
# --stop: stop the unit and add the third point (RAM used 10 s after the stop). CUDA memory on Orin is nvmap, not RSS:
# a bare cudaFree(0) context is +70-82 MB RAM used with 20 MB RSS (2026-11-02) — compare ram_delta, not RSS, for GPU units.
set -euo pipefail
UNIT="$1"; HOLD="${2:-30}"; STOP="${3:-}"
TMP="$(mktemp -d)"; trap 'sudo pkill -f "tegrastats --interval 500 --logfile $TMP" 2>/dev/null || true; rm -rf "$TMP"' EXIT
ram_used() { sudo timeout "$1" tegrastats --interval 500 | awk '{split($4,a,"/"); s+=a[1]; n++} END{printf "%.0f", s/n}'; }
cached() { awk '/^Cached:/{printf "%.0f", $2/1024}' /proc/meminfo; }
health() {  # start -> /health for the llama-server units, start -> active for the rest
  local url=""
  case "$UNIT" in skull-llm) url=http://127.0.0.1:11437/health ;; skull-embed) url=http://127.0.0.1:11438/health ;; esac
  for _ in $(seq 1 1200); do
    if [ -n "$url" ]; then curl -sf "$url" >/dev/null 2>&1 && return 0
    else systemctl --user is-active -q "$UNIT" && return 0; fi
    sleep 0.1
  done; return 1; }

C0="$(cached)"; R0="$(ram_used 10)"
sudo tegrastats --interval 500 --logfile "$TMP/tg.log" &
T0="$(date +%s.%N)"; systemctl --user start "$UNIT"
health && HS="$(awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')" || HS=timeout
sleep "$HOLD"
PID="$(systemctl --user show -p MainPID --value "$UNIT")"
read -r HWM ANON FILE < <(awk '/^VmHWM/{h=$2} /^RssAnon/{a=$2} /^RssFile/{f=$2} END{printf "%d %d %d\n", h/1024, a/1024, f/1024}' "/proc/$PID/status")
sudo pkill -f "tegrastats --interval 500 --logfile $TMP"; sleep 0.6
RP="$(awk '{split($4,a,"/"); if (a[1]>m) m=a[1]} END{print m}' "$TMP/tg.log")"
TMAX="$(grep -o 'tj@[0-9.]*' "$TMP/tg.log" | cut -d@ -f2 | sort -n | tail -1)"
C1="$(cached)"; RA=""
if [ "$STOP" = "--stop" ]; then systemctl --user stop "$UNIT"; sleep 5; RA="$(ram_used 10)"; fi
# unit,date,ram_before_mb,ram_peak_mb,ram_delta_mb,vmhwm_mb,rss_anon_mb,rss_file_mb,cached_before_mb,cached_after_mb,health_s,tj_max_c,ram_after_stop_mb
echo "$UNIT,$(date +%F_%T),$R0,$RP,$((RP - R0)),$HWM,$ANON,$FILE,$C0,$C1,$HS,$TMAX,$RA"
