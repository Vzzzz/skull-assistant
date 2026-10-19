#!/usr/bin/env bash
# Перезамер памяти VLM под ollama: модель × num_ctx × тип KV × flash-attention.
# Запускается в WSL. Поднимает ОТДЕЛЬНЫЙ ollama на порту 11435 с нужными env,
# чтобы не трогать системный юнит и не требовать sudo.
# Пишет CSV (формат v2, 2026-10-19): model,num_ctx,kv,flash,vram_idle_mb,vram_peak_mb,vram_delta_mb,ollama_ps_size,ollama_ps_proc,
#   rss_runner_mb,ram_delta_mb,load_s,ttft_warm_s,prompt_tok_s,tok_s
# ram_delta_mb — рост занятой host RAM (MemTotal−MemAvailable) при загруженной модели: для MoE с CPU-offload это главная цифра.
# load_s — первый запрос (загрузка + префилл); ttft_warm_s / prompt_tok_s / tok_s — второй запрос с тёплой моделью (T-006).
# vram_delta = peak - idle (реальная цена модели+KV+контекста на GPU); ollama_ps_size — ОЦЕНКА ollama, сравнивать с delta.
# ollama_ps_proc: "100%GPU" — всё на GPU; "NN%/MM%CPU/GPU" — часть слоёв на CPU, VRAM-цифра занижена.
#
# Использование: bash bench_vlm_mem.sh [out.csv]
# Перед запуском: убедиться, что системный ollama ничего не держит (ollama ps пуст),
# иначе цифры VRAM смешаются. skull-brain можно не останавливать — он молчит без транскриптов.

set -u
OUT="${1:-vlm_mem_$(date +%Y%m%d_%H%M).csv}"
PORT=11435
export OLLAMA_HOST="127.0.0.1:$PORT"
# модели системного сервиса (ollama pull кладёт их туда); у второго сервера иначе будет 404
export OLLAMA_MODELS="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
MODELS="${MODELS:-gemma3:4b gemma4:e2b gemma4:e4b}"
CTXS="${CTXS:-2048 4096 8192 16384}"
KVS="${KVS:-f16 q8_0}"
FLASHES="${FLASHES:-1 0}"
PROMPT="Ты череп. Ответь одним коротким предложением: сколько будет 17 умножить на 23? Сначала выкладка."

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
ram_used()  { awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{printf "%d", (t-a)/1024}' /proc/meminfo; }
# gen_req MODEL CTX [PROMPT]. Второй (тёплый) запрос обязан идти ДРУГИМ промптом: одинаковый попадает в кэш
# префикса ollama, префилл считается для 1–5 токенов и prompt_tok_s превращается в шум (2026-10-19: 32 880 ток/с у ministral).
gen_req()   { local P="${3:-$PROMPT}"; curl -s "http://$OLLAMA_HOST/api/generate" -d "{
      \"model\": \"$1\", \"prompt\": $(printf '%s' "$P" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'),
      \"stream\": false, \"keep_alive\": \"5m\",
      \"options\": {\"num_ctx\": $2, \"num_predict\": 64, \"temperature\": 0}
    }"; }
# промпт для замера префилла: другое начало (мимо кэша) + ~400 токенов текста
PROMPT_WARM="Второй вопрос, не связанный с первым. $(for i in 1 2 3 4 5 6; do printf '%s ' "$PROMPT"; done)Ответь одним предложением."

echo "model,num_ctx,kv,flash,vram_idle_mb,vram_peak_mb,vram_delta_mb,ollama_ps_size,ollama_ps_proc,rss_runner_mb,ram_delta_mb,load_s,ttft_warm_s,prompt_tok_s,tok_s" > "$OUT"

for FLASH in $FLASHES; do
for KV in $KVS; do
  # q8_0 KV в ollama требует flash attention; без него пара бессмысленна
  if [ "$KV" != "f16" ] && [ "$FLASH" = "0" ]; then continue; fi

  echo "### server: flash=$FLASH kv=$KV" >&2
  OLLAMA_FLASH_ATTENTION=$FLASH OLLAMA_KV_CACHE_TYPE=$KV OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=0 \
    ollama serve >/tmp/ollama_bench_${FLASH}_${KV}.log 2>&1 &
  SRV=$!
  for i in $(seq 1 30); do curl -s "http://$OLLAMA_HOST/" >/dev/null && break; sleep 0.5; done
  sleep 1
  IDLE=$(vram_used); RAM_IDLE=$(ram_used)

  for M in $MODELS; do
  for CTX in $CTXS; do
    echo "--- $M ctx=$CTX" >&2
    # фон: опрос пика VRAM каждые 100 мс
    PEAKF=$(mktemp); echo 0 > "$PEAKF"
    ( while :; do v=$(vram_used); p=$(cat "$PEAKF"); [ "$v" -gt "$p" ] && echo "$v" > "$PEAKF"; sleep 0.1; done ) &
    POLL=$!

    RESP_COLD=$(gen_req "$M" "$CTX")          # загрузка + префилл + генерация
    RESP=$(gen_req "$M" "$CTX" "$PROMPT_WARM")  # тёплая модель, другой промпт ~400 ток.: честный префилл, TTFT и tok/s

    PS_LINE=$(ollama ps 2>/dev/null | awk -v m="$M" '$1==m')
    PS_SIZE=$(echo "$PS_LINE" | awk '{print $3$4}')
    PS_PROC=$(echo "$PS_LINE" | awk '{print $5$6}')
    RUNNER_PID=$(pgrep -P "$SRV" | head -1)   # раннер — дочерний процесс сервера
    RSS=$( [ -n "${RUNNER_PID:-}" ] && awk '/VmRSS/ {printf "%d", $2/1024}' /proc/$RUNNER_PID/status 2>/dev/null || echo NA )
    RAM_DELTA=$(( $(ram_used) - RAM_IDLE ))
    sleep 0.3
    kill $POLL 2>/dev/null; wait $POLL 2>/dev/null
    PEAK=$(cat "$PEAKF"); rm -f "$PEAKF"

    LOAD=$(printf '%s' "$RESP_COLD" | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    v=(d.get("load_duration",0)+d.get("prompt_eval_duration",0))/1e9
    print(f"{v:.2f}")
except Exception: print("ERR")')
    read -r TTFT PTOKS TOKS <<<"$(printf '%s' "$RESP" | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    pc=d.get("prompt_eval_count",0); pe=d.get("prompt_eval_duration",1)/1e9
    ev=d.get("eval_count",0); ed=d.get("eval_duration",1)/1e9
    print(f"{pe:.2f} {pc/pe if pe else 0:.0f} {ev/ed if ed else 0:.1f}")
except Exception as e:
    print("ERR ERR ERR")')"

    echo "$M,$CTX,$KV,$FLASH,$IDLE,$PEAK,$((PEAK-IDLE)),${PS_SIZE:-NA},${PS_PROC:-NA},$RSS,$RAM_DELTA,$LOAD,$TTFT,$PTOKS,$TOKS" | tee -a "$OUT"

    # выгрузить модель, убить CUDA-контекст между прогонами (правило протокола)
    curl -s "http://$OLLAMA_HOST/api/generate" -d "{\"model\":\"$M\",\"keep_alive\":0}" >/dev/null
    sleep 2
  done
  done

  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  sleep 2
done
done

echo "готово: $OUT" >&2
echo "примечание: gemma4:e2b/e4b — сравнить vram_peak с размером файла (7.2/9.6 ГБ); разница = что осталось на CPU (PLE через mmap)." >&2
echo "MoE с CPU-offload: смотреть ollama_ps_proc (split) и ram_delta_mb; prompt_tok_s — скорость префилла, главный налог режима B." >&2
