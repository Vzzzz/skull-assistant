"""Launch a llama-server for brain from manifest params (units skull-llm and skull-embed).

  llm_server.py llm    VLM + projector on GPU (D-024), -c = params.num_ctx
  llm_server.py embed  embedding model on the CPU build (no CUDA context), for the memory
"""
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[2]
P = yaml.safe_load(Path(__file__).with_name("manifest.yaml").read_text(encoding="utf-8"))["params"]


def path(p):
    return str(ROOT / p)


def args(role):
    if role == "llm":
        port = urlparse(P["host"]).port
        return [os.path.expanduser(P["bin"]), "-m", path(P["gguf"]), "--mmproj", path(P["mmproj"]),
                "-ngl", "99", "-c", str(P["num_ctx"]), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                "--slot-save-path", path(P["slot_save_path"]), "--host", "127.0.0.1", "--port", str(port)]
    if role == "embed":
        m = P["memory"]
        port = urlparse(m["embed_host"]).port
        # whole exchange in one batch: embeddings need n_ubatch >= input tokens (default 512 rejects long replies)
        return [os.path.expanduser(m["embed_bin"]), "-m", path(m["embed_gguf"]), "--embedding",
                "-c", "2048", "-b", "2048", "-ub", "2048", "--host", "127.0.0.1", "--port", str(port)]
    raise SystemExit(f"unknown role {role!r}: llm | embed")


if __name__ == "__main__":
    a = args(sys.argv[1] if len(sys.argv) > 1 else "llm")
    print("llm_server:", " ".join(a), flush=True)
    os.execv(a[0], a)
