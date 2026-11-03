"""cache_reuse on the multimodal llama-server (T-059, P-033): before / after the fork's patch.

Starts llama-server with the production args of skull-llm (services/brain/llm_server.py, mode A: E4B IQ4_XS @8192,
q8_0 KV, --cache-ram 0, --ctx-checkpoints 0, --lazy-mode on) plus --cache-reuse N, per configuration (binary, mmproj
on/off, --swa-full on/off), on its own port. Two scenarios:

  edit    a ~3k-token prompt A (persona + calibration text), then B = A without one paragraph in the middle, then
          C = B without another one; 3 rounds, the slots are erased between rounds. Per request:
          timings.prompt_n (tokens actually evaluated), cache_n, prompt_ms; the answer at temperature 0 for equality.
  window  CTX-1 (research/eval/ctx_probe.py --backend llamaserver --ctx 2048 --trim tokens): 60 turns, the token
          ledger drops the oldest exchanges -> the prefix changes right after the system prompt (D-026). TTFT of the
          turns with a shift vs without.

Per configuration also: the reuse / SWA lines of the server log, VRAM peak (nvidia-smi; on Jetson — nvmap total).

  .venv/bin/python research/bench/bench_cache_reuse.py --bin ~/llama.cpp-master/build-cuda/bin --tag before
  .venv/bin/python research/bench/bench_cache_reuse.py --bin ~/llama.cpp-mtmd/build-cuda/bin --tag after
The stand must be down (EXECUTOR §2). -> results/llamacpp_cache_reuse_<tag>_<date>.json (+ ctx_probe JSON per config)
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services/brain"))
import llm_server  # noqa: E402

PORT = 11440
HOST = f"http://127.0.0.1:{PORT}"
OUT = ROOT / "research/bench/results"
CALIB = ROOT / "research/quant/calib_ru.txt"
CONFIGS = {  # name -> (mmproj, swa_full)
    "mmproj+swa-full": (True, True),
    "mmproj": (True, False),
    "text+swa-full": (False, True),
    "text": (False, False),
}
WARN = re.compile(r"cache.reuse|cache_reuse|kv_cache.*size|KV self size|llama_kv_cache.*MiB", re.I)


def vram():
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if out.stdout.split() and out.stdout.split()[0].isdigit():
            return int(out.stdout.split()[0])
    # Jetson (unified memory; nvidia-smi of JetPack 6 prints [N/A]): GPU allocations of all processes from nvmap, MB (T-056)
    out = subprocess.run(["sudo", "-n", "cat", "/sys/kernel/debug/nvmap/iovmm/clients"], capture_output=True, text=True)
    return int(out.stdout.split("total")[-1].split()[0].rstrip("K")) // 1024


def server_args(bin_dir, mmproj, swa_full, reuse):
    a = llm_server.args("llm", "A")
    a[0] = str(Path(bin_dir).expanduser() / "llama-server")
    a[a.index("--port") + 1] = str(PORT)
    if not mmproj and "--mmproj" in a:
        i = a.index("--mmproj")
        del a[i:i + 2]
    return a + ["--cache-reuse", str(reuse)] + (["--swa-full"] if swa_full else [])


class Server:
    def __init__(self, args, log_path):
        self.args, self.log_path = args, log_path

    def __enter__(self):
        self.log = open(self.log_path, "w")
        self.v0 = vram()
        self.p = subprocess.Popen(self.args, stdout=self.log, stderr=subprocess.STDOUT)
        self.peak, self.stop = [self.v0], threading.Event()

        def poll():
            while not self.stop.is_set():
                try:
                    self.peak.append(vram())
                except Exception:
                    pass
                time.sleep(0.3)
        threading.Thread(target=poll, daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < 180:
            if self.p.poll() is not None:
                raise RuntimeError(f"server exited {self.p.returncode}, see {self.log_path}")
            try:
                if requests.get(f"{HOST}/health", timeout=2).json().get("status") == "ok":
                    return self
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError("server not healthy in 180 s")

    def __exit__(self, *exc):
        self.stop.set()
        self.p.terminate()
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()
        time.sleep(3)

    def log_lines(self):
        return [l.rstrip() for l in Path(self.log_path).read_text(errors="replace").splitlines() if WARN.search(l)]


def tokens(text):
    return len(requests.post(f"{HOST}/tokenize", json={"content": text}, timeout=30).json()["tokens"])


def build_prompts(target=3000):
    """A = persona + calibration paragraphs up to ~target tokens; B = A without a middle paragraph; C = B without
    another one. Returns texts and the token counts of the removed paragraphs."""
    persona = (ROOT / "persona/system_prompt.md").read_text(encoding="utf-8").strip()
    lines = [l.strip() for l in CALIB.read_text(encoding="utf-8").splitlines() if l.strip()]
    paras = [" ".join(lines[i:i + 6]) for i in range(0, len(lines), 6)]     # ~150-token paragraphs
    body = []
    while tokens(persona + "\n\n" + "\n\n".join(body)) < target and len(body) < len(paras):
        body.append(paras[len(body)])
    mid, mid2 = len(body) // 2, len(body) // 2 + 2
    join = lambda ps: persona + "\n\n" + "\n\n".join(ps) + "\n\nКороче, Костян, что скажешь?"  # noqa: E731
    a = join(body)
    b = join(body[:mid] + body[mid + 1:])
    c = join([p for i, p in enumerate(body) if i not in (mid, mid2)])
    return a, b, c


def complete(prompt):
    r = requests.post(f"{HOST}/completion", json={"prompt": prompt, "n_predict": 24, "temperature": 0, "seed": 42,
                                                  "cache_prompt": True}, timeout=300).json()
    t = r.get("timings", {})
    return {"prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": round(t.get("prompt_ms", 0), 1),
            "content": r.get("content", "")}


def scenario_edit(rounds=3):
    a, b, c = build_prompts()
    sizes = {k: tokens(v) for k, v in (("A", a), ("B", b), ("C", c))}
    rows = []
    for k in range(rounds):
        for slot in requests.get(f"{HOST}/slots", timeout=10).json():      # every round starts from empty slots
            requests.post(f"{HOST}/slots/{slot['id']}?action=erase", timeout=10)
        for name, p in (("A", a), ("B", b), ("C", c)):
            rows.append({"round": k + 1, "prompt": name, "n_tokens": sizes[name], **complete(p)})
            print(f"    edit #{k + 1} {name}: n={sizes[name]} prompt_n={rows[-1]['prompt_n']} cache_n={rows[-1]['cache_n']} "
                  f"{rows[-1]['prompt_ms']} ms", flush=True)
    return {"sizes": sizes, "rows": rows}


def scenario_window(label):
    cmd = [sys.executable, str(ROOT / "research/eval/ctx_probe.py"), "--backend", "llamaserver", "--host", HOST,
           "--model", label, "--ctx", "2048", "--trim", "tokens"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    res = ROOT / "research/eval/results"
    log = json.loads((res / f"ctx_probe_{label}_trim_2048.json").read_text(encoding="utf-8"))["log"]
    for f in [res / f"ctx_probe_{label}_trim_2048.json", *res.glob(f"ctx_probe_{label}_*.md")]:
        f.unlink()                             # the per-turn rows below are all we keep
    return [{"turn": e.get("turn"), "window_turns": e.get("window_turns"), "shift": e.get("ctx_overflow_dropped", 0) > 0,
             "ttft_s": e.get("ttft_s"), "prompt_tokens": e.get("prompt_eval_count")} for e in log]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="dir with llama-server")
    ap.add_argument("--tag", required=True, help="before | after")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--reuse", type=int, default=256)
    ap.add_argument("--scenarios", nargs="+", default=["edit", "window"], choices=["edit", "window"])
    a = ap.parse_args()
    if vram() > 500:
        raise SystemExit(f"GPU busy ({vram()} MB): stop the stand first (systemctl --user stop skull.target)")
    stamp = time.strftime("%Y-%m-%d")
    res = {"bin": a.bin, "tag": a.tag, "reuse": a.reuse, "configs": {}}
    for name in a.configs:
        mmproj, swa_full = CONFIGS[name]
        args = server_args(a.bin, mmproj, swa_full, a.reuse)
        print(f"== {a.tag} / {name}: {' '.join(args[1:])}", flush=True)
        log_path = OUT / f"llamacpp_cache_reuse_{a.tag}_{name}.log"
        r = {"args": args[1:]}
        with Server(args, log_path) as s:
            if "edit" in a.scenarios:
                r["edit"] = scenario_edit()
            if "window" in a.scenarios:
                r["window"] = scenario_window(f"reuse-{a.tag}-{name}")
            r["vram_idle_mb"], r["vram_peak_mb"] = s.v0, max(s.peak)
        r["log"] = s.log_lines()
        res["configs"][name] = r
        print(f"   VRAM peak {r['vram_peak_mb']} MB; log: {r['log'][-3:]}", flush=True)
    out = OUT / f"llamacpp_cache_reuse_{a.tag}_{stamp}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
