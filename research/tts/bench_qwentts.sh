#!/usr/bin/env bash
# Qwen3-TTS via qwentts.cpp (T-013): memory and latency of the 0.6B CustomVoice talker on CPU and CUDA.
# Starts tts-server (OpenAI-compatible), measures RSS/VmHWM + VRAM after load, then for each line of
# research/eval/tts_lines.txt requests PCM streaming and records time-to-first-audio (first byte), total time,
# audio seconds and real-time factor. PCM is saved as WAV under research/tts/results/qwentts/ for listening (not in git).
# Env: QT_DIR (~/qwentts.cpp), TALKER, CODEC, SPEAKER (ryan), LANG (Russian), BACKENDS ("CUDA0 CPU"), LINES, PORT (8089)
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
QT_DIR="${QT_DIR:-$HOME/qwentts.cpp}"
TALKER="${TALKER:-$ROOT/models/hf/Qwen3-TTS-GGUF/qwen-talker-0.6b-customvoice-Q8_0.gguf}"
CODEC="${CODEC:-$ROOT/models/hf/Qwen3-TTS-GGUF/qwen-tokenizer-12hz-Q8_0.gguf}"
SPEAKER="${SPEAKER:-ryan}"; LANG_="${LANG_:-Russian}"
BACKENDS="${BACKENDS:-CUDA0 CPU}"
LINES="${LINES:-$ROOT/research/eval/tts_lines.txt}"
PORT="${PORT:-8089}"
OUT="${1:-$ROOT/research/tts/results/qwentts_$(date +%F).csv}"
WAVDIR="$ROOT/research/tts/results/qwentts"; mkdir -p "$WAVDIR"
SRV=""; trap 'kill $SRV 2>/dev/null' EXIT
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
rss_of() { awk '/VmRSS/ {printf "%d", $2/1024}' /proc/"$1"/status 2>/dev/null || echo NA; }
hwm_of() { awk '/VmHWM/ {printf "%d", $2/1024}' /proc/"$1"/status 2>/dev/null || echo NA; }

[ -x "$QT_DIR/build/tts-server" ] || { echo "no $QT_DIR/build/tts-server — build qwentts.cpp first (./buildcuda.sh)"; exit 1; }
[ -f "$TALKER" ] && [ -f "$CODEC" ] || { echo "missing GGUF: $TALKER / $CODEC"; exit 1; }
if [ ! -s "$OUT" ]; then echo "backend,talker,speaker,lang,line,chars,ttfa_s,total_s,audio_s,rt,vram_idle_mb,vram_loaded_mb,vram_peak_mb,rss_mb,vmhwm_mb,load_s" > "$OUT"; fi

for BE in $BACKENDS; do
  echo "== backend $BE talker $(basename "$TALKER")"
  V0=$(vram); T0=$(date +%s.%N)
  GGML_BACKEND="$BE" "$QT_DIR/build/tts-server" --model "$TALKER" --codec "$CODEC" --alias qwen3-tts --port "$PORT" >"/tmp/qwentts_${BE}.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 180); do curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "200" && break; kill -0 "$SRV" 2>/dev/null || { echo "tts-server died: tail /tmp/qwentts_${BE}.log"; tail -5 "/tmp/qwentts_${BE}.log"; break; }; sleep 1; done
  LOAD=$(python3 -c "import time; print(f'{time.time()-$T0:.1f}')")
  V1=$(vram); VPEAK=$V1
  # warm-up (first request compiles graphs / fills caches)
  curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' \
    -d "{\"input\":\"Проверка связи.\",\"voice\":\"$SPEAKER\",\"language\":\"$LANG_\",\"response_format\":\"pcm\",\"seed\":42}" -o /dev/null
  n=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue; n=$((n+1))
    body=$(python3 -c "import json,sys; print(json.dumps({'input': sys.argv[1], 'voice': '$SPEAKER', 'language': '$LANG_', 'response_format': 'pcm', 'seed': 42}, ensure_ascii=False))" "$line")
    PCM="$WAVDIR/${BE}_$(basename "$TALKER" .gguf)_$n.pcm"
    read -r TTFA TOTAL BYTES < <(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' -d "$body" -o "$PCM" -w "%{time_starttransfer} %{time_total} %{size_download}")
    v=$(vram); [ "$v" -gt "$VPEAK" ] && VPEAK=$v
    AUD=$(python3 -c "print(f'{$BYTES/(24000*2):.2f}')"); RT=$(python3 -c "print(f'{$TOTAL/max($AUD,0.01):.3f}')")
    python3 - "$PCM" <<'EOF'
import sys, wave
p = sys.argv[1]; data = open(p, "rb").read()
with wave.open(p[:-4] + ".wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(data)
EOF
    rm -f "$PCM"
    echo "$BE,$(basename "$TALKER" .gguf),$SPEAKER,$LANG_,$n,${#line},$TTFA,$TOTAL,$AUD,$RT,$V0,$V1,$VPEAK,$(rss_of "$SRV"),$(hwm_of "$SRV"),$LOAD" | tee -a "$OUT"
  done < "$LINES"
  kill "$SRV"; wait "$SRV" 2>/dev/null; SRV=""; sleep 2
done
echo "csv: $OUT ; wav: $WAVDIR"
