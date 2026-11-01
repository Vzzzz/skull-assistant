"""MTP draft for mode B on the stand (T-052): does it fit, and what does it give?

Starts llama-server (a build with --spec-type draft-mtp, e.g. a worktree of llama.cpp master) on the mode-B model with
the production args (services/brain/llm_server.py: -ngl 99 -c 8192 -fa on, q8_0 KV, no mmproj) per configuration, and
asks the three questions of the T-046 gate after a real history (the A / B exchanges of the gate, ~3k tokens), each
3 times, prompt cache off (every TTFT is a full prefill). Per configuration: VRAM peak (nvidia-smi, polled), RSS peak of
the server, TTFT, generation tok/s, draft tokens accepted / drafted (llama-server timings).

  LLAMA_BIN=~/llama.cpp-master/build-cuda/bin .venv/bin/python research/bench/bench_draft.py [--quick]
The stand must be down (EXECUTOR §2: systemctl --user stop skull.target, GPU empty). -> results/mtp_b_<date>.csv + .json
"""
import argparse
import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("LLAMA_BIN", Path.home() / "llama.cpp-master/build-cuda/bin")) / "llama-server"
GGUF = ROOT / "models/hf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-mtp-Q4_K_M.gguf"
DRAFT = ROOT / "models/hf/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
PORT = 11439
HOST = f"http://127.0.0.1:{PORT}"
BASE = ["-m", str(GGUF), "-ngl", "99", "-c", "8192", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "0",
        "--ctx-checkpoints", "0", "--host", "127.0.0.1", "--port", str(PORT), "--reasoning-budget", "0"]
CONFIGS = [  # name, args
    ("no draft, -ncmoe 24", ["-ncmoe", "24"]),
    ("draft GPU, -ncmoe 25", ["-ncmoe", "25", "--spec-type", "draft-mtp", "-md", str(DRAFT), "-ngld", "99"]),
    ("draft GPU, -ncmoe 26", ["-ncmoe", "26", "--spec-type", "draft-mtp", "-md", str(DRAFT), "-ngld", "99"]),
    ("draft CPU, -ncmoe 24", ["-ncmoe", "24", "--spec-type", "draft-mtp", "-md", str(DRAFT), "-ngld", "0"]),
]
QUESTIONS = ["Что ты помнишь про моего брата?", "Почему зимой рано темнеет?",
             "Сколько нужно человек, чтобы передвинуть трёхтонный каменный обелиск в пустыне?"]
OUT = ROOT / "research/bench/results"


def history():
    """System = persona; the gate's owner turns with the skull's replies (parts 1 and 2 of 2026-10-31)."""
    msgs = [{"role": "system", "content": (ROOT / "persona/system_prompt.md").read_text(encoding="utf-8").strip()}]
    for name in ("mode_gate_2026-10-31_bus.json", "mode_gate_2026-10-31_2_bus.json"):
        bus = json.loads((OUT / name).read_text(encoding="utf-8"))
        for i, e in enumerate(bus):
            if e["s"] != "events:transcript" or e["f"].get("speaker") not in ("owner", "uncertain"):
                continue
            rep = next((json.loads(b["f"]["meta"]).get("reply") for b in bus[i + 1:i + 6]
                        if b["s"] == "events:reply" and "meta" in b["f"]), None)
            if rep:
                msgs += [{"role": "user", "content": e["f"]["text"]}, {"role": "assistant", "content": rep}]
    return msgs


def vram():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.split()[0])


def hwm(pid):
    try:
        return int(Path(f"/proc/{pid}/status").read_text().split("VmHWM:")[1].split()[0]) // 1024
    except (OSError, IndexError):
        return None


def run_config(name, args, msgs, repeats, draft_max=None):
    extra = ["--spec-draft-n-max", str(draft_max)] if draft_max else []
    log = open(OUT / "mtp_b_server.log", "a")
    log.write(f"\n===== {name} {' '.join(extra)}\n"); log.flush()
    v0 = vram()
    p = subprocess.Popen([str(BIN), *BASE, *args, *extra], stdout=log, stderr=subprocess.STDOUT)
    peak, stop = [v0], threading.Event()

    def poll():
        while not stop.is_set():
            try:
                peak.append(vram())
            except Exception:
                pass
            time.sleep(0.3)
    threading.Thread(target=poll, daemon=True).start()
    rows = []
    try:
        t0 = time.time()
        while time.time() - t0 < 180:
            if p.poll() is not None:
                return [{"config": name, "error": f"server exited {p.returncode} (see mtp_b_server.log)"}]
            try:
                if requests.get(f"{HOST}/health", timeout=2).json().get("status") == "ok":
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        load_s = round(time.time() - t0, 1)
        requests.post(f"{HOST}/v1/chat/completions", json={"messages": msgs[:3] + [{"role": "user", "content": "Привет"}],
                      "max_tokens": 8}, timeout=300)                  # warm-up
        for q in QUESTIONS:
            for k in range(repeats):
                body = {"messages": msgs + [{"role": "user", "content": q}], "max_tokens": 300, "temperature": 0.3,
                        "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
                r = requests.post(f"{HOST}/v1/chat/completions", json=body, timeout=600).json()
                t = r.get("timings", {})
                rows.append({"config": name, "draft_max": draft_max or "", "question": q, "repeat": k + 1, "load_s": load_s,
                             "prompt_tokens": t.get("prompt_n"), "ttft_s": round(t.get("prompt_ms", 0) / 1000, 2),
                             "gen_tokens": t.get("predicted_n"), "tok_s": round(t.get("predicted_per_second", 0), 1),
                             "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
                             "vram_peak_mb": max(peak), "vram_idle_mb": v0, "rss_peak_mb": hwm(p.pid)})
                print(f"  {name} {extra} | {q[:24]}… #{k + 1}: ttft {rows[-1]['ttft_s']} s, {rows[-1]['tok_s']} tok/s, "
                      f"draft {rows[-1]['draft_accepted']}/{rows[-1]['draft_n']}, VRAM {max(peak)}", flush=True)
    finally:
        stop.set()
        p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
        time.sleep(3)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 repeat, configs only")
    a = ap.parse_args()
    if vram() > 500:
        raise SystemExit(f"GPU busy ({vram()} MB): stop the stand first (systemctl --user stop skull.target)")
    msgs = history()
    print(f"history: {len(msgs) - 1} messages; llama-server {BIN}", flush=True)
    reps = 1 if a.quick else 3
    rows = []
    for name, args in CONFIGS:
        rows += run_config(name, args, msgs, reps)
    drafted = [r for r in rows if r.get("draft_n") and "error" not in r]
    if drafted and not a.quick:   # --spec-draft-n-max on the best draft configuration
        best = max({r["config"] for r in drafted},
                   key=lambda c: sum(r["tok_s"] for r in drafted if r["config"] == c) / sum(1 for r in drafted if r["config"] == c))
        args = dict(CONFIGS)[best]
        for n in (2, 4, 8):
            rows += run_config(best, args, msgs, 1, draft_max=n)
    stamp = time.strftime("%Y-%m-%d")
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(OUT / f"mtp_b_{stamp}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("->", OUT / f"mtp_b_{stamp}.csv")


if __name__ == "__main__":
    main()
