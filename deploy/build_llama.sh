#!/usr/bin/env bash
# Canonical llama.cpp build of the prod (T-059 step 6, D-033): our fork with the cache_reuse patch for multimodal
# prompts without media, at a pinned commit; two builds — CUDA (skull-llm, VLM on the GPU) and CPU (skull-embed: the CUDA
# build creates a 176 MB context even at -ngl 0, D-024). After the merge upstream: back to master, the patch is dropped.
#   bash deploy/build_llama.sh [dir=~/llama.cpp-fork] [cuda_arch: 86 stand | 87 Orin] [jobs]
# Orin: jobs 4 with the stack down (-j2 peak 1.7 GB RAM, -j4 ~21 min; never next to a resident mode A — I-24).
# Units use the build through SKULL_LLAMA_DIR (services/brain/llm_server.py), set in a skull-llm / skull-embed drop-in.
set -euo pipefail
URL=https://github.com/Vzzzz/llama.cpp
BRANCH=server-cache-reuse-mtmd
COMMIT=2d417888196b018358876771a0d2dd1cb5abd709      # on master 9309bf1 (2026-11-03)
DIR="${1:-$HOME/llama.cpp-fork}"
ARCH="${2:-$([ -e /etc/nv_tegra_release ] && echo 87 || echo 86)}"
JOBS="${3:-$(nproc)}"
export PATH=/usr/local/cuda/bin:$PATH

[ -d "$DIR/.git" ] || git clone --branch "$BRANCH" "$URL" "$DIR"
git -C "$DIR" fetch --depth 50 origin "$BRANCH"
git -C "$DIR" checkout -q "$COMMIT"
echo "llama.cpp: $(git -C "$DIR" log --oneline -1), CUDA arch $ARCH, -j$JOBS"

T0=$(date +%s)
cmake -S "$DIR" -B "$DIR/build-cuda" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$DIR/build-cuda" -j"$JOBS" --target llama-server llama-cli llama-bench
T1=$(date +%s); echo "CUDA build: $((T1 - T0)) s"
cmake -S "$DIR" -B "$DIR/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$DIR/build" -j"$JOBS" --target llama-server
echo "CPU build: $(( $(date +%s) - T1 )) s"
"$DIR/build-cuda/bin/llama-server" --version 2>&1 | tail -2
