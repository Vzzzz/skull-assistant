#!/usr/bin/env bash
# T-024 / Q-048: llama-server on one GGUF, same measurements as bench_vlm_mem.sh v2.1 (so the CSV rows are comparable
# with ollama rows), plus the two things ollama cannot do: slot save/restore (freeze/thaw for M8) and n_probs (Q-034).
#
#   bash bench_llamaserver.sh bench  out.csv   # peak VRAM / RSS / host RAM / load / warm TTFT / prefill / tok/s at CTXS
#   bash bench_llamaserver.sh slots            # fill ~8k context, save slot, restart, restore: times and file size
#   bash bench_llamaserver.sh probs            # n_probs on a yes/no question: first-token distribution + latency
#
# Env: GGUF (default models/quant/kostyan-e4b-iq4xs-imx.gguf), MMPROJ (optional: path to mmproj gguf -> VRAM with vision),
#      LLAMA_BIN (default ~/llama.cpp/build-cuda/bin), CTXS, KV (q8_0), FA (on), PORT (11436).
# CSV: runtime,model,num_ctx,kv,flash,vram_idle_mb,vram_peak_mb,vram_delta_mb,rss_server_mb,ram_delta_mb,load_s,ttft_warm_s,prompt_tok_s,tok_s
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="${1:-bench}"; OUT="${2:-$ROOT/research/bench/results/llamaserver_$(date +%F).csv}"
GGUF="${GGUF:-$ROOT/models/quant/kostyan-e4b-iq4xs-imx.gguf}"
MMPROJ="${MMPROJ:-}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-cuda/bin}"
CTXS="${CTXS:-2048 8192 16384}"; KV="${KV:-q8_0}"; FA="${FA:-on}"; PORT="${PORT:-11436}"
HOST="http://127.0.0.1:$PORT"
SLOTS="$ROOT/models/quant/slots"; mkdir -p "$SLOTS"
PROMPT="Ты — череп на столе. Ответь одной фразой: что ты видишь перед собой прямо сейчас и почему это скучно?"
PROMPT_WARM="Второй вопрос, не связанный с первым. $(for i in 1 2 3 4 5 6; do printf '%s ' "$PROMPT"; done)Ответь одним предложением."
[ -x "$LLAMA_BIN/llama-server" ] || { echo "нет $LLAMA_BIN/llama-server — собрать с -DGGML_CUDA=ON (см. TASKS T-024)"; exit 1; }
[ -s "$GGUF" ] || { echo "нет $GGUF"; exit 1; }

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
ram_used()  { awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{printf "%d", (t-a)/1024}' /proc/meminfo; }
jstr()      { python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'; }
# A stale llama-server on the port (left by an aborted run on 2026-10-19) silently served every request at ctx 2048.
# "[l]lama-server" — the bracket keeps pkill from matching this script's own command line.
pkill -f "[l]lama-server -m .*--port $PORT" 2>/dev/null && { echo "убит старый llama-server на :$PORT"; sleep 2; }
SRV=""; trap 'kill $SRV 2>/dev/null' EXIT
# Gemma 4 instruct answers EOS to a raw prompt: every prompt goes through the chat template (/apply-template).
# The template enables thinking by default (<|think|> in the system turn) -> server runs with --reasoning-budget 0
# and the tag is stripped from the templated prompt as a belt-and-braces measure.
tpl() {  # $1 user text, $2 optional system text -> templated prompt string
  local body; body=$(python3 -c 'import json,sys; m=[]
if sys.argv[2]: m.append({"role":"system","content":sys.argv[2]})
m.append({"role":"user","content":sys.argv[1]}); print(json.dumps({"messages":m}))' "$1" "${2:-}")
  curl -s "$HOST/apply-template" -d "$body" | python3 -c 'import json,sys; p=json.load(sys.stdin)["prompt"]; print(p.replace("<|think|>\n",""), end="")'
}
start_server() {  # $1 = ctx; sets SRV (pid) and LOAD (seconds to /health). Must NOT be called in $(...): subshell loses SRV.
  local extra=(); [ -n "$MMPROJ" ] && extra+=(--mmproj "$MMPROJ")
  "$LLAMA_BIN/llama-server" -m "$GGUF" "${extra[@]}" -ngl 99 -fa "$FA" -ctk "$KV" -ctv "$KV" -c "$1" -np 1 \
      --reasoning-budget 0 --port "$PORT" --slot-save-path "$SLOTS" --log-disable >/tmp/llamaserver_bench.log 2>&1 &
  SRV=$!
  local t0; t0=$(date +%s.%N)
  for i in $(seq 1 600); do curl -sf "$HOST/health" >/dev/null 2>&1 && break; sleep 0.2; done
  LOAD=$(python3 -c "print(f'{$(date +%s.%N)-$t0:.2f}')")
}
stop_server() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 1; }
completion() {  # $1 user text (templated here), $2 extra json fields
  local p; p=$(tpl "$1")
  curl -s "$HOST/completion" -d "{\"prompt\": $(printf '%s' "$p" | jstr), \"n_predict\": 64, \"temperature\": 0, \"cache_prompt\": false ${2:+, $2}}"
}

case "$MODE" in
bench)
  [ -s "$OUT" ] || echo "runtime,model,num_ctx,kv,flash,vram_idle_mb,vram_peak_mb,vram_delta_mb,rss_server_mb,ram_delta_mb,load_s,ttft_warm_s,prompt_tok_s,tok_s" > "$OUT"
  for CTX in $CTXS; do
    IDLE=$(vram_used); RAM_IDLE=$(ram_used)
    PEAKF=$(mktemp); echo 0 > "$PEAKF"
    ( while :; do v=$(vram_used); p=$(cat "$PEAKF"); [ "$v" -gt "$p" ] && echo "$v" > "$PEAKF"; sleep 0.1; done ) & POLL=$!
    start_server "$CTX"
    completion "$PROMPT" >/dev/null                      # cold: graph warm-up
    RESP=$(completion "$PROMPT_WARM")                     # warm, different prompt -> honest prefill
    RSS=$(awk '/VmRSS/ {printf "%d", $2/1024}' /proc/$SRV/status 2>/dev/null || echo NA)
    RAM_DELTA=$(( $(ram_used) - RAM_IDLE ))
    sleep 0.3; kill $POLL 2>/dev/null; wait $POLL 2>/dev/null; PEAK=$(cat "$PEAKF"); rm -f "$PEAKF"
    read -r TTFT PTOKS TOKS <<<"$(printf '%s' "$RESP" | python3 -c '
import json,sys
try:
    t=json.load(sys.stdin)["timings"]
    a=t["prompt_ms"]/1000; b=t["prompt_per_second"]; c=t["predicted_per_second"]
    print(f"{a:.2f} {b:.0f} {c:.1f}")
except Exception: print("ERR ERR ERR")')"
    echo "llama-server,$(basename "$GGUF" .gguf)${MMPROJ:++mmproj},$CTX,$KV,$FA,$IDLE,$PEAK,$((PEAK-IDLE)),$RSS,$RAM_DELTA,$LOAD,$TTFT,$PTOKS,$TOKS" | tee -a "$OUT"
    stop_server
  done
  echo "готово: $OUT (строки ollama для того же GGUF — bench_vlm_mem.sh с MODELS=<имя в ollama>)" ;;

slots)
  CTX=8192; start_server "$CTX"; echo "load: $LOAD s"
  # fill the context with ~8k tokens of in-domain text (calibration file), then save the slot
  FILL=$(tpl "$(head -c 24000 "$ROOT/research/quant/calib_ru.txt")

Прочитай и жди вопроса.")
  curl -s "$HOST/completion" -d "{\"prompt\": $(printf '%s' "$FILL" | jstr), \"n_predict\": 8, \"cache_prompt\": true}" | python3 -c 'import json,sys;d=json.load(sys.stdin);print("filled tokens:",d["timings"]["prompt_n"])'
  curl -s "$HOST/slots/0?action=save" -d '{"filename":"brain8k.bin"}' | python3 -c 'import json,sys;d=json.load(sys.stdin);print("save:",d.get("n_saved"),"tokens,",d.get("timings",{}).get("save_ms"),"ms")'
  ls -l "$SLOTS/brain8k.bin" | awk '{printf "slot file: %.1f MB\n",$5/1e6}'
  stop_server
  start_server "$CTX"; echo "reload: $LOAD s"
  curl -s "$HOST/slots/0?action=restore" -d '{"filename":"brain8k.bin"}' | python3 -c 'import json,sys;d=json.load(sys.stdin);print("restore:",d.get("n_restored"),"tokens,",d.get("timings",{}).get("restore_ms"),"ms")'
  # continue the same prefix: prompt cache must hit -> prompt_n small, TTFT small
  curl -s "$HOST/completion" -d "{\"prompt\": $(printf '%s' "$FILL" | jstr), \"n_predict\": 32, \"cache_prompt\": true}" | python3 -c 'import json,sys;d=json.load(sys.stdin);t=d["timings"];print("after restore: prompt_n",t["prompt_n"],"(cache_n",t.get("cache_n"),") prompt_ms",round(t["prompt_ms"]),"-> thaw = restore_ms + this")'
  stop_server ;;

probs)
  start_server 4096; echo "load: $LOAD s"
  Q=$(tpl "Реплика Валерия: «Костян, ты там?» Это обращение к тебе? Ответь одним словом: да или нет." "$(cat "$ROOT/persona/system_prompt_rules.md")")
  T0=$(date +%s.%N)
  curl -s "$HOST/completion" -d "{\"prompt\": $(printf '%s' "$Q" | jstr), \"n_predict\": 1, \"temperature\": 0, \"n_probs\": 8}" | python3 -c '
import json,sys
d=json.load(sys.stdin); t=d["timings"]
print("prompt_n", t["prompt_n"], "prompt_ms", round(t["prompt_ms"]), "| first token:", repr(d["content"]))
for p in d.get("completion_probabilities", [])[:1]:
    for c in p.get("top_probs", p.get("probs", []))[:8]:
        tok = c.get("token", c.get("tok_str")); pr = c.get("prob", 0)
        print(f"  {tok!r:>12}  {pr:.3f}")'
  python3 -c "print(f'wall: {$(date +%s.%N)-$T0:.2f} s')"
  stop_server ;;
*) echo "usage: $0 bench|slots|probs [out.csv]"; exit 2 ;;
esac
