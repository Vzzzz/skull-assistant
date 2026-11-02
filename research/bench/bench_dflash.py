"""T-055: the DFlash drafter of Nemotron 3.5 Bolt E4B in llama.cpp (a build with --spec-type draft-dflash, e.g. the
master worktree of T-052). 10 persona turns of CTX-1 (judge_persona) with the production persona prompt, one by one
(no history), temperature 0.3, max 300 tokens; per configuration: tok/s, drafted / accepted tokens (AL = accepted per
draft step is not exposed — the acceptance rate is), reply length.

  LLAMA_BIN=~/llama.cpp-master/build-cuda/bin .venv/bin/python research/bench/bench_dflash.py
-> research/bench/results/bolt_dflash_<date>.json
"""
import json, os, statistics as st, subprocess, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("LLAMA_BIN", Path.home() / "llama.cpp-master/build-cuda/bin")) / "llama-server"
DF = ROOT / "models/hf/nemotron-bolt-e4b/Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf"
MODELS = {"bolt-q4km": ROOT / "models/hf/nemotron-bolt-e4b/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf",
          "bolt-iq4xs": ROOT / "models/quant/bolt-e4b-iq4xs-imx.gguf"}
PORT = 11443
PERSONA = (ROOT / "persona/system_prompt.md").read_text(encoding="utf-8").strip()
TURNS = [json.loads(l)["text"] for l in (ROOT / "research/eval/ctx_probe_dialog.jsonl").read_text(encoding="utf-8").splitlines()
         if json.loads(l).get("judge_persona")][:10]


def run(name, gguf, draft):
    args = [str(BIN), "-m", str(gguf), "-ngl", "99", "-c", "8192", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
            "--port", str(PORT), "--reasoning-budget", "0"] + (["--spec-type", "draft-dflash", "-md", str(DF), "-ngld", "99"] if draft else [])
    log = open(ROOT / "research/bench/results/bolt_dflash_server.log", "a")
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        for _ in range(120):
            if p.poll() is not None:
                return {"config": name, "draft": draft, "error": f"server exited {p.returncode}"}
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).json().get("status") == "ok":
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        for t in TURNS:
            body = {"messages": [{"role": "system", "content": PERSONA}, {"role": "user", "content": t}], "max_tokens": 300,
                    "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}}
            j = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, timeout=300).json()
            tm = j.get("timings", {})
            rows.append({"tok_s": tm.get("predicted_per_second"), "n": tm.get("predicted_n"), "draft_n": tm.get("draft_n"),
                         "draft_accepted": tm.get("draft_n_accepted"), "reply": j["choices"][0]["message"]["content"]})
    finally:
        p.terminate(); p.wait(30); time.sleep(2)
    dn = sum(r["draft_n"] or 0 for r in rows); da = sum(r["draft_accepted"] or 0 for r in rows)
    return {"config": name, "draft": draft, "tok_s_median": round(st.median(r["tok_s"] for r in rows), 1),
            "tokens_mean": round(st.mean(r["n"] for r in rows), 1), "draft_n": dn, "draft_accepted": da,
            "accept_rate": round(da / dn, 2) if dn else None, "rows": rows}


def main():
    out = []
    for name, g in MODELS.items():
        for draft in (False, True):
            res = run(name, g, draft)
            out.append(res)
            print({k: v for k, v in res.items() if k != "rows"}, flush=True)
    p = ROOT / f"research/bench/results/bolt_dflash_{time.strftime('%Y-%m-%d')}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
