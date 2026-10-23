#!/usr/bin/env bash
# Загрузка кандидатов для замеров. Запуск из WSL (или на Orin) в сервисном режиме стенда:
#   deploy/fetch_models.sh stand     # кандидаты режима A + текстовые альтернативы (~25 ГБ)
#   deploy/fetch_models.sh tts       # TTS/omni-кандидаты с Hugging Face (~15 ГБ, нужен huggingface_hub)
#   deploy/fetch_models.sh modeb     # режим B: MoE со стримингом экспертов и MTP (~40 ГБ; на Orin с NVMe)
#   deploy/fetch_models.sh coder     # qwen3-coder 30b-a3b (режим B, ~19 ГБ)
#   deploy/fetch_models.sh prune     # удалить лишние квантизации HF-репо, скачанные до появления --include
#   deploy/fetch_models.sh all
# Требуется запущенный ollama (sudo systemctl start ollama). HF: .venv/bin/pip install -U huggingface_hub (CLI: hf)
set -u
HF_DIR="${HF_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models/hf}"
what="${1:-stand}"

# ollama pull resumes a partial download, so on a flaky connection we just retry (2026-10-17: 26b-a4b took
# several attempts by hand). PULL_TRIES=0 disables retries.
PULL_TRIES="${PULL_TRIES:-6}"
pull() {
  for m in "$@"; do
    echo "=== ollama pull $m"
    n=0
    until ollama pull "$m" 2>&1 | tr '\r' '\n' | grep -vE '^pulling [0-9a-f]+.*%|^$' | tail -2; test "${PIPESTATUS[0]}" -eq 0; do
      n=$((n+1)); [ "$n" -ge "$PULL_TRIES" ] && { echo "!!! $m: giving up after $n tries"; break; }
      echo "... retry $n/$PULL_TRIES in 15s"; sleep 15
    done
  done
}
# CLI Hugging Face: в huggingface_hub >= 1.0 команда называется hf (huggingface-cli устарел); у нас она в .venv/bin
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -x "$REPO_DIR/.venv/bin/hf" ]; then HF_CLI="$REPO_DIR/.venv/bin/hf"
elif command -v hf >/dev/null; then HF_CLI=hf
elif command -v huggingface-cli >/dev/null; then HF_CLI=huggingface-cli
else HF_CLI=""; fi
hf()   { for r in "$@"; do echo "=== hf download $r"; "$HF_CLI" download "$r" --local-dir "$HF_DIR/${r#*/}" 2>&1 | tail -1; done; }
# hf_inc REPO PATTERN... — only the listed files. Without --include a GGUF repo pulls every quantization
# (2026-10-16: Qwen3-TTS-GGUF 64 GB, MiniCPM-o 75 GB for ~9 GB actually needed).
# Flaky link + large file (2026-10-19, 16 GB BF16): the Xet backend of `hf download` does NOT resume after a drop.
# Either HF_HUB_DISABLE_XET=1 (plain HTTP, resumes via .incomplete) or fetch the single file with aria2c/wget -c:
#   aria2c -c -x 8 -s 8 -k 1M --max-tries=0 -o <file> https://huggingface.co/<repo>/resolve/main/<file>
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"   # 2026-10-23: with =1 downloads do not start on the owner's WSL; keep Xet on, resume big files with aria2c
hf_inc() { r="$1"; shift; echo "=== hf download $r ($*)"; "$HF_CLI" download "$r" --local-dir "$HF_DIR/${r#*/}" --include "$@" 2>&1 | tail -1; }

case "$what" in
  stand|all)
    # Режим A: мультимодальные 2–4B (кандидаты M5). QAT-варианты — официальные int4-чекпойнты (ROADMAP M9).
    pull gemma4:e2b-it-qat gemma4:e4b-it-qat minicpm-v4.6:q4
    # Текстовые альтернативы (без зрения): сравнить качество персоны/JSON при том же бюджете
    pull ministral-3:3b ministral-3:8b phi4-mini:3.8b smollm2:1.7b
    # MiniCPM5-2B есть только как GGUF на HF (ollama умеет hf.co/…); тег квантизации проверить на странице репо
    pull hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M || echo "MiniCPM5-2B: уточнить имя GGUF-файла на huggingface.co/openbmb/MiniCPM5-2B-GGUF"
    ;;&
  tts|all)
    [ -n "$HF_CLI" ] || { echo "нет CLI Hugging Face: .venv/bin/pip install -U huggingface_hub (команда hf)"; exit 1; }
    "$HF_CLI" auth whoami >/dev/null 2>&1 || echo "hf не аутентифицирован: $HF_CLI auth login (нужно для gated-репозиториев)"
    mkdir -p "$HF_DIR"
    # Qwen3-TTS: русский, стриминг; 0.6B — кандидат в рантайм (GPU) / через qwentts.cpp (CPU); 1.7B-VoiceDesign — офлайн-учитель (голос по описанию)
    hf Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign Qwen/Qwen3-TTS-Tokenizer-12Hz
    # GGUF only for the CPU runtime candidate (0.6B CustomVoice) + tokenizer; the 1.7B teacher runs from safetensors on GPU
    hf_inc Serveurperso/Qwen3-TTS-GGUF "qwen-talker-0.6b-customvoice-Q4_K_M.gguf" "qwen-talker-0.6b-customvoice-Q8_0.gguf" "qwen-tokenizer-12hz-Q8_0.gguf" "*.md" "*.json"
    # Chatterbox Multilingual (0.5B, ru, контроль эмоций); ONNX-сборка — для CPU-теста
    hf ResembleAI/chatterbox-multilingual onnx-community/chatterbox-multilingual-ONNX
    # OpenAudio S1-mini (0.5B, ru) — CC-BY-NC-SA, gated: нужен hf auth login и принятые условия
    hf fishaudio/openaudio-s1-mini || echo "openaudio-s1-mini: gated — hf auth login + принять условия на странице модели"
    # MiniCPM-o 4.5 GGUF (9B omni, full-duplex речь) — «голос внутри VLM», тест на стенде; на Orin — отдельно
    # Only Q4_K_M of the LLM + the F16 sub-models it needs (audio/vision/tts/token2wav); other quants and CoreML are not used
    hf_inc openbmb/MiniCPM-o-4_5-gguf "MiniCPM-o-4_5-Q4_K_M.gguf" "audio/*.gguf" "vision/*.gguf" "tts/*.gguf" "token2wav-gguf/*" "*.md" "*.json"
    # Эмоции в речи (фоновый режим, AMBIENT_MODE.md): emotion2vec+ large (~300M), через funasr/modelscope
    hf emotion2vec/emotion2vec_plus_large emotion2vec/emotion2vec_plus_base
    ;;&
  prune)
    # Remove the variants fetched before hf_inc existed (2026-10-13..15); keeps what the tts section above would download.
    cd "$HF_DIR" || exit 1
    du -sh Qwen3-TTS-GGUF MiniCPM-o-4_5-gguf 2>/dev/null
    find Qwen3-TTS-GGUF -maxdepth 1 -name '*.gguf' \
      ! -name 'qwen-talker-0.6b-customvoice-Q4_K_M.gguf' ! -name 'qwen-talker-0.6b-customvoice-Q8_0.gguf' ! -name 'qwen-tokenizer-12hz-Q8_0.gguf' \
      -print -delete
    find MiniCPM-o-4_5-gguf -maxdepth 1 -name '*.gguf' ! -name 'MiniCPM-o-4_5-Q4_K_M.gguf' -print -delete
    rm -rf MiniCPM-o-4_5-gguf/vision/*.mlmodelc Qwen3-TTS-GGUF/.cache MiniCPM-o-4_5-gguf/.cache
    du -sh Qwen3-TTS-GGUF MiniCPM-o-4_5-gguf
    ;;
  modeb|all)
    # Режим B: MoE со стримингом экспертов. -mtp- = multi-token prediction (встроенная спекуляция, P-015).
    # Точные теги библиотеки (проверено 2026-10-17): суффикс q4_K_M, сокращённых "-q4" не существует.
    # gemma4 26b-a4b: 17 ГБ модель + 1.2 ГБ проектор + 0.5 ГБ draft; qwen3.6 35b-a3b: ~20–23 ГБ.
    pull gemma4:26b-a4b-it-q4_K_M gemma4:26b-a4b-it-mtp-q4_K_M qwen3.6:35b-a3b-q4_K_M qwen3.6:35b-a3b-mtp-q4_K_M
    ;;&
  coder|all)
    # Coder-линейка «для любопытства» (владелец, 2026-10-16): tool-calling / JSON / агентский режим (U-002).
    # Маленьких coder у Qwen в 2026 нет (младшие — Qwen2.5-Coder 2024, старое поколение не берём).
    # qwen3.6 coding-вариант той же 35b-a3b (с MTP) и qwen3-coder 30b-a3b — оба класс режима B, ~20 ГБ каждый.
    pull qwen3.6:35b-a3b-coding-mtp-q4_K_M qwen3-coder:30b
    # Qwen3-Coder-Next 80B-A3B (~45 ГБ Q4) — класс режима C («трофей»), только на Orin с NVMe:
    #   ollama pull hf.co/unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M
    ;;
esac
echo "готово: $what"; ollama list 2>/dev/null | head -30
