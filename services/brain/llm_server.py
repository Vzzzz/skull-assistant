"""Launch a llama-server for brain from manifest params (units skull-llm and skull-embed).

  llm_server.py llm    VLM + projector on GPU (D-024), -c = params.num_ctx; the model by Redis mode:current (T-044):
                       manifest modes.<mode> overrides gguf / mmproj / num_ctx and adds extra_args (B: 26B-A4B)
  llm_server.py embed  embedding model on the CPU build (no CUDA context), for the memory
"""
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import redis
import yaml

ROOT = Path(__file__).resolve().parents[2]
M = yaml.safe_load(Path(__file__).with_name("manifest.yaml").read_text(encoding="utf-8"))
P = M["params"]


def path(p):
    return str(ROOT / p)


def current_mode():
    try:
        return (redis.Redis().get("mode:current") or b"A").decode()
    except redis.RedisError:
        return "A"


def args(role, mode="A"):
    if role == "llm":
        port = urlparse(P["host"]).port
        o = {**P, **{k: v for k, v in ((M.get("modes") or {}).get(mode) or {}).items() if k != "budget"}}
        return [os.path.expanduser(P["bin"]), "-m", path(o["gguf"])] + (["--mmproj", path(o["mmproj"])] if o.get("mmproj") else []) + [
                "-ngl", "99", "-c", str(o["num_ctx"]), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                "--slot-save-path", path(P["slot_save_path"]), "--host", "127.0.0.1", "--port", str(port),
                # no host-RAM prompt cache / SWA checkpoints: with the defaults (8192 MiB, 32 per slot) RSS kept growing
                # (10.5 GB after three CTX-1 runs @2048); without them it plateaus at ~3.55 GB, prefill not slower
                "--cache-ram", "0", "--ctx-checkpoints", "0",
                # per-layer embedding tables of Gemma E-models read from disk on demand, not kept resident
                "--lazy-mode", "on"] + [str(x) for x in o.get("extra_args", [])]
    if role == "embed":
        m = P["memory"]
        port = urlparse(m["embed_host"]).port
        # whole exchange in one batch: embeddings need n_ubatch >= input tokens (default 512 rejects long replies)
        return [os.path.expanduser(m["embed_bin"]), "-m", path(m["embed_gguf"]), "--embedding",
                "-c", "2048", "-b", "2048", "-ub", "2048", "--host", "127.0.0.1", "--port", str(port)]
    raise SystemExit(f"unknown role {role!r}: llm | embed")


if __name__ == "__main__":
    role = sys.argv[1] if len(sys.argv) > 1 else "llm"
    mode = current_mode() if role == "llm" else "A"
    a = args(role, mode)
    if role == "llm":
        redis.Redis().set("mode:llm", mode)      # which model this server holds: the dispatcher's reconcile checks it
    print(f"llm_server: mode {mode}:", " ".join(a), flush=True)
    os.execv(a[0], a)
