#!/usr/bin/env bash
# Замер B: RSS голосового стека (пять user-юнитов) в точке «модель загружена, простой».
# Поднимает skull.target, ждёт загрузки моделей, снимает VmRSS/VmHWM (пик) и CPU%, гасит стенд.
# Запуск из WSL: research/bench/measure_voice_rss.sh [wait_s]
set -u
WAIT="${1:-45}"
OUT="$(dirname "$0")/results/voice_rss_$(date +%F).csv"
systemctl --user start skull.target
sleep "$WAIT"
echo "service,pid,vmrss_mb,vmhwm_mb,cpu_pct_10s,threads" | tee "$OUT"
for s in skull-capture skull-wakeword skull-stt skull-brain skull-tts; do
  pid=$(systemctl --user show -p MainPID --value "$s")
  if [ -z "$pid" ] || [ "$pid" = 0 ]; then echo "$s,0,NA,NA,NA,NA" | tee -a "$OUT"; continue; fi
  t1=$(awk '{print $14+$15}' /proc/$pid/stat); sleep 10; t2=$(awk '{print $14+$15}' /proc/$pid/stat)
  cpu=$(( (t2-t1) * 100 / ($(getconf CLK_TCK) * 10) ))
  rss=$(awk '/VmRSS/ {printf "%d",$2/1024}' /proc/$pid/status)
  hwm=$(awk '/VmHWM/ {printf "%d",$2/1024}' /proc/$pid/status)
  thr=$(awk '/Threads/ {print $2}' /proc/$pid/status)
  echo "$s,$pid,$rss,$hwm,$cpu,$thr" | tee -a "$OUT"
done
echo "redis,$(pgrep -x redis-server | head -1),$(awk '/VmRSS/ {printf "%d",$2/1024}' /proc/$(pgrep -x redis-server | head -1)/status),NA,NA,NA" | tee -a "$OUT"
systemctl --user stop skull.target
echo "csv: $OUT"
