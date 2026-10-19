#!/usr/bin/env bash
# Q-047 / T-023: own quantization of gemma-4-E4B-it from the ggml-org BF16 GGUF with an imatrix built on Russian
# in-domain text, then three ollama models to compare against the official gemma4:e4b-it-qat on CTX-1 and bench:
#   kostyan-e4b-q4km-imx   Q4_K_M with imatrix
#   kostyan-e4b-iq4xs-imx  IQ4_XS with imatrix (smaller)
#   kostyan-e4b-q4km       Q4_K_M without imatrix (separates the imatrix effect from the QAT effect)
# Runs in WSL from the repo root. CPU-only llama.cpp build is enough (quantize is CPU; imatrix on CPU ~10-30 min).
# Needs: research/quant/calib_ru.txt (make_calib.py), ~20 GB free, hf auth (ggml-org repo is public).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QD="$ROOT/models/quant"; mkdir -p "$QD"
LLAMA="${LLAMA:-$HOME/llama.cpp}"
CALIB="$ROOT/research/quant/calib_ru.txt"
BF16="$QD/gemma-4-E4B-it-BF16.gguf"
HF_CLI="$ROOT/.venv/bin/hf"
[ -s "$CALIB" ] || { echo "нет $CALIB — сначала research/quant/make_calib.py"; exit 1; }

# 1. llama.cpp (CPU build; add -DGGML_CUDA=ON if a CUDA toolkit is installed)
if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
  command -v cmake >/dev/null || { echo "нужен cmake: sudo apt install -y cmake build-essential"; exit 1; }
  [ -d "$LLAMA" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
  cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA="${GGML_CUDA:-OFF}" -DLLAMA_CURL=OFF >/dev/null
  cmake --build "$LLAMA/build" -j"$(nproc)" --target llama-quantize llama-imatrix llama-cli
fi
BIN="$LLAMA/build/bin"; echo "llama.cpp: $(cd "$LLAMA" && git rev-parse --short HEAD)"

# 2. BF16 GGUF (16 GB) — no HF conversion step needed
[ -s "$BF16" ] || "$HF_CLI" download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-BF16.gguf --local-dir "$QD"

# 3. imatrix on the Russian calibration text (chunks of 512 tokens)
IMX="$QD/imatrix-e4b-ru.gguf"
if [ ! -s "$IMX" ]; then
  /usr/bin/time -f "imatrix: %e s, %M KB maxrss" "$BIN/llama-imatrix" -m "$BF16" -f "$CALIB" -o "$IMX" --chunk 512 -t "$(nproc)" ${GGML_CUDA:+-ngl 99}
fi

# 4. three quantizations
q() { local out="$QD/$1.gguf"; shift
  [ -s "$out" ] && { echo "есть $out"; return; }
  /usr/bin/time -f "$out: %e s" "$BIN/llama-quantize" "$@" "$BF16" "$out" "$QTYPE" "$(nproc)"; }
QTYPE=Q4_K_M q kostyan-e4b-q4km-imx --imatrix "$IMX"
QTYPE=IQ4_XS q kostyan-e4b-iq4xs-imx --imatrix "$IMX"
QTYPE=Q4_K_M q kostyan-e4b-q4km
ls -la "$QD"/*.gguf

# 5. ollama models (text-only: mmproj not attached — CTX-1 and bench are text; vision later)
for n in kostyan-e4b-q4km-imx kostyan-e4b-iq4xs-imx kostyan-e4b-q4km; do
  printf 'FROM %s/%s.gguf\n' "$QD" "$n" > "$QD/Modelfile.$n"
  ollama create "$n" -f "$QD/Modelfile.$n"
done
ollama list | grep kostyan-e4b
echo "дальше: bench и CTX-1 (T-023 шаги 4–5)"
