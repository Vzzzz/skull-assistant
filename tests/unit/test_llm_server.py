"""services/brain/llm_server.py: the per-machine switches of the unit environment (T-059, D-033) — SKULL_LLAMA_DIR moves
both llama-server binaries to another build, SKULL_LLM_EXTRA_ARGS adds args to skull-llm in mode A only; without them
the command line is the one of the manifest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services/brain"))
import llm_server  # noqa: E402


def test_no_env_is_the_manifest(monkeypatch):
    monkeypatch.delenv("SKULL_LLAMA_DIR", raising=False)
    monkeypatch.delenv("SKULL_LLM_EXTRA_ARGS", raising=False)
    a = llm_server.args("llm", "A")
    assert a[0].endswith("/llama.cpp/build-cuda/bin/llama-server")
    assert a[-2:] == ["--lazy-mode", "on"]
    assert llm_server.args("embed")[0].endswith("/llama.cpp/build/bin/llama-server")


def test_llama_dir_moves_both_binaries(monkeypatch):
    monkeypatch.setenv("SKULL_LLAMA_DIR", "~/llama.cpp-fork")
    assert llm_server.args("llm", "A")[0].endswith("/llama.cpp-fork/build-cuda/bin/llama-server")
    assert llm_server.args("embed")[0].endswith("/llama.cpp-fork/build/bin/llama-server")


def test_extra_args_mode_a_only(monkeypatch):
    monkeypatch.setenv("SKULL_LLM_EXTRA_ARGS", "--cache-reuse 256 --swa-full")
    assert llm_server.args("llm", "A")[-3:] == ["--cache-reuse", "256", "--swa-full"]
    b = llm_server.args("llm", "B")
    assert "--cache-reuse" not in b and "--swa-full" not in b
