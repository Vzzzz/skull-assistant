#!/usr/bin/env bash
# Stack bench (T-033): memory of the whole mode-A set on the stand, step by step.
#   P0 baseline (no skull units, ollama idle) -> P1 voice units (capture, wakeword, stt, tts; NOT brain)
#   -> P2 llama-server text (IQ4_XS+imatrix @8k, q8_0 KV, flash) -> P3 + mmproj (+ one image request)
#   -> P4 + embedder (llama-server --embeddings, CPU) -> P5 + YOLO probe (onnxruntime, CPU) -> totals.
# Each step: VRAM used (nvidia-smi), host RAM used (free), RSS of the new process, and for models a warm request.
# Run from WSL (~/skull-assistant), stand in NORMAL mode but with ollama idle (`ollama ps` empty) and brain stopped.
# Env: GGUF, MMPROJ, EMBED_GGUF (embeddinggemma GGUF; empty = skip), YOLO_ONNX (empty = skip), CTX (8192), LLAMA_BIN, WAIT (45)
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
GGUF="${GGUF:-$ROOT/models/quant/kostyan-e4b-iq4xs-imx.gguf}"
MMPROJ="${MMPROJ:-$(ls "$ROOT"/models/hf/gemma-4-E4B-it-GGUF/mmproj*.gguf 2>/dev/null | head -1)}"
EMBED_GGUF="${EMBED_GGUF:-$(ollama show embeddinggemma --modelfile 2>/dev/null | awk '/^FROM/ {print $2}' | head -1)}"
YOLO_ONNX="${YOLO_ONNX:-$ROOT/models/vision/yolo11n_640.onnx}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-cuda/bin}"
CTX="${CTX:-8192}"; WAIT="${WAIT:-45}"
PORT=11437; EPORT=11438
OUT="${1:-$ROOT/research/bench/results/stack_$(date +%F).csv}"
PIDS=(); trap 'for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done' EXIT

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
ram_used() { free -m | awk '/^Mem:/ {print $3}'; }
rss_of() { awk '/VmRSS/ {printf "%d", $2/1024}' /proc/"$1"/status 2>/dev/null || echo NA; }
hwm_of() { awk '/VmHWM/ {printf "%d", $2/1024}' /proc/"$1"/status 2>/dev/null || echo NA; }
row() { echo "$1,$2,$3,$4,$5,$6,$7,$8" | tee -a "$OUT"; }

echo "step,component,pid,rss_mb,vmhwm_mb,vram_used_mb,ram_used_mb,note" > "$OUT"
echo "== P0 baseline"; systemctl --user stop skull.target 2>/dev/null; sleep 3
if [ -n "$(ollama ps 2>/dev/null | tail -n +2)" ]; then echo "ollama has models loaded — run: ollama stop <model> for each (ollama ps)"; exit 1; fi
V0=$(vram); R0=$(ram_used)
row P0 baseline - - - "$V0" "$R0" "skull.target stopped, ollama idle"

echo "== P1 voice units"
for s in skull-capture skull-wakeword skull-stt skull-tts; do systemctl --user start "$s"; done
sleep "$WAIT"
for s in skull-capture skull-wakeword skull-stt skull-tts; do
  pid=$(systemctl --user show -p MainPID --value "$s"); row P1 "$s" "$pid" "$(rss_of "$pid")" "$(hwm_of "$pid")" "$(vram)" "$(ram_used)" "after ${WAIT}s"
done
rp=$(pgrep -x redis-server | head -1); row P1 redis "$rp" "$(rss_of "$rp")" "$(hwm_of "$rp")" "$(vram)" "$(ram_used)" ""

start_llama() {  # $1 = extra args string; sets SRV
  "$LLAMA_BIN/llama-server" -m "$GGUF" $1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c "$CTX" -np 1 --reasoning-budget 0 --port "$PORT" --log-disable >/tmp/llama_stack.log 2>&1 &
  SRV=$!; PIDS+=("$SRV")
  for i in $(seq 1 120); do curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break; sleep 1; done
}
chat() { curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$1" >/dev/null; }

echo "== P2 llama-server text"
start_llama ""
chat '{"messages":[{"role":"user","content":"Привет. Скажи одну фразу о погоде за окном."}],"max_tokens":60,"temperature":0.3}'
row P2 llama-server-text "$SRV" "$(rss_of "$SRV")" "$(hwm_of "$SRV")" "$(vram)" "$(ram_used)" "$(basename "$GGUF") ctx=$CTX q8_0 fa, after 1 chat"
kill "$SRV"; wait "$SRV" 2>/dev/null; sleep 3

if [ -n "$MMPROJ" ] && [ -f "$MMPROJ" ]; then
  echo "== P3 llama-server + mmproj"
  start_llama "--mmproj $MMPROJ"
  row P3 llama-server-vision "$SRV" "$(rss_of "$SRV")" "$(hwm_of "$SRV")" "$(vram)" "$(ram_used)" "+mmproj loaded, idle"
  # one image request: 640x480 synthetic PNG (pure python, no PIL)
  PNG=$(python3 - <<'EOF'
import zlib, struct, base64
w, h = 640, 480
raw = b"".join(b"\x00" + bytes(((x * 7 + y * 3) % 256, (x + y) % 256, (x * y) % 256) for x in range(w)) for y in range(h))
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
print(base64.b64encode(png).decode())
EOF
)
  chat "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Что на картинке? Одной фразой.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$PNG\"}}]}],\"max_tokens\":60}"
  row P3 llama-server-vision "$SRV" "$(rss_of "$SRV")" "$(hwm_of "$SRV")" "$(vram)" "$(ram_used)" "after 1 image request 640x480"
  chat '{"messages":[{"role":"user","content":"И ещё одну фразу."}],"max_tokens":40}'
  row P3 llama-server-vision "$SRV" "$(rss_of "$SRV")" "$(hwm_of "$SRV")" "$(vram)" "$(ram_used)" "after text request (steady)"
else
  echo "== P3 skipped: no mmproj (MMPROJ=$MMPROJ)"; start_llama ""
fi

if [ -n "$EMBED_GGUF" ] && [ -f "$EMBED_GGUF" ]; then
  echo "== P4 embedder (CPU)"
  "$LLAMA_BIN/llama-server" -m "$EMBED_GGUF" --embeddings -ngl 0 -c 2048 --port "$EPORT" --log-disable >/tmp/embed_stack.log 2>&1 &
  EMB=$!; PIDS+=("$EMB")
  for i in $(seq 1 60); do curl -s "http://127.0.0.1:$EPORT/health" | grep -q '"ok"' && break; sleep 1; done
  T0=$(date +%s.%N); curl -s "http://127.0.0.1:$EPORT/embedding" -H 'Content-Type: application/json' -d '{"content":"Ключи я переложил в рюкзак, в передний карман."}' >/dev/null; T1=$(date +%s.%N)
  row P4 embedder-cpu "$EMB" "$(rss_of "$EMB")" "$(hwm_of "$EMB")" "$(vram)" "$(ram_used)" "$(basename "$EMBED_GGUF") -ngl 0, 1 embedding $(python3 -c "print(f'{($T1-$T0)*1000:.0f} ms')")"
else
  echo "== P4 skipped: no embedder GGUF"
fi

if [ -n "$YOLO_ONNX" ] && [ -f "$YOLO_ONNX" ]; then
  echo "== P5 YOLO probe (onnxruntime)"
  .venv/bin/python research/vision/yolo_probe.py --model "$YOLO_ONNX" --fps 5 >/tmp/yolo_stack.log 2>&1 &
  YP=$!; PIDS+=("$YP"); sleep 20
  row P5 yolo-onnx "$YP" "$(rss_of "$YP")" "$(hwm_of "$YP")" "$(vram)" "$(ram_used)" "$(tail -1 /tmp/yolo_stack.log | tr ',' ';' | cut -c1-120)"
else
  echo "== P5 skipped: no YOLO ONNX (YOLO_ONNX=$YOLO_ONNX)"
fi

echo "== totals"
TOT=0; for p in "${PIDS[@]}"; do r=$(rss_of "$p"); [ "$r" != NA ] && TOT=$((TOT + r)); done
for s in skull-capture skull-wakeword skull-stt skull-tts; do pid=$(systemctl --user show -p MainPID --value "$s"); r=$(rss_of "$pid"); [ "$r" != NA ] && TOT=$((TOT + r)); done
row P6 totals - "$TOT" - "$(vram)" "$(ram_used)" "sum RSS of all skull processes; vram delta vs P0 = $(( $(vram) - V0 )); ram delta vs P0 = $(( $(ram_used) - R0 ))"
systemctl --user stop skull.target 2>/dev/null
echo "csv: $OUT"
