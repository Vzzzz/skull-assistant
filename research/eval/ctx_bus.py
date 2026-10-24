"""CTX-1 through the bus: feed the dialog to a running skull-brain (events:transcript -> events:reply).

Unlike ctx_probe.py this does not call the model itself: it tests the service as deployed (manifest params,
token window, router, memory, reply filter). Per-turn details come from the `meta` field brain puts on
events:reply; metrics are the same as ctx_probe.metrics. The history is reset (/reset) before the run.

  .venv/bin/python research/eval/ctx_bus.py --tag run1     # brain's num_ctx is read from llama-server /props
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import redis
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/eval"))
from ctx_probe import DIALOG, OUT, metrics  # noqa: E402


def say(r, text, timeout=300):
    last = r.xrevrange("events:reply", count=1)
    last_id = last[0][0] if last else "0"
    r.xadd("events:transcript", {"text": text, "ts": f"{time.time():.3f}"})
    resp = r.xread({"events:reply": last_id}, count=1, block=timeout * 1000)
    if not resp:
        raise SystemExit(f"no reply in {timeout}s to {text!r}")
    f = resp[0][1][0][1]
    return f[b"text"].decode(), json.loads(f.get(b"meta", b"{}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. run1")
    args = ap.parse_args()
    params = yaml.safe_load((ROOT / "services/brain/manifest.yaml").read_text(encoding="utf-8"))["params"]
    ctx = requests.get(f"{params['host']}/props", timeout=5).json()["default_generation_settings"]["n_ctx"]
    model = Path(params["gguf"]).stem
    rows = [json.loads(l) for l in DIALOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = redis.Redis(socket_timeout=None)
    r.xadd("events:transcript", {"text": "/reset"})
    time.sleep(1)
    log = []
    for row in rows:
        t0 = time.perf_counter()
        reply, meta = say(r, row["text"])
        e = {**row, "reply": reply, "think": False, "thinking_chars": 0, "ctx_overflow_dropped": 0,
             "window_turns": meta.get("window_turns"), "prompt_eval_count": meta.get("prompt_tokens", 0),
             "eval_count": meta.get("completion_tokens", 0), "ttft_s": meta.get("ttft_s", 0), "tok_s": meta.get("tok_s"),
             "wall_s": round(time.perf_counter() - t0, 2), "done_reason": meta.get("finish_reason"),
             "max_tokens": meta.get("max_tokens")}
        if "route" in meta:
            e["route"] = meta["route"]
        if "mem_hits" in meta:
            note, raw = meta.get("note", ""), meta.get("raw", reply)
            low = raw.lower()
            e["mem_hits"] = meta["mem_hits"]
            e["mem_note_chars"] = len(note)
            # raw-model defects as in ctx_probe: copied note / trailing aside (the filter then removes them)
            e["mem_note_copied"] = bool(note) and (note[:40] in raw or "ты помнишь" in low or "из памяти" in low)
            e["mem_aside"] = bool(note) and bool(re.search(r"\n\s*\([^)]{15,}\)\s*$", raw))
            e["guard"] = meta.get("guard", [])
            if "raw" in meta:
                e["reply_raw"] = meta["raw"]
        log.append(e)
        rt = f" route={meta['route']['label']}" if "route" in meta else ""
        mh = f" mem={[h['turn'] for h in meta.get('mem_hits', [])]}" if meta.get("mem_hits") else ""
        print(f"[bus ctx={ctx}] t{row['turn']:02d} pe={e['prompt_eval_count']:5d} ec={e['eval_count']:4d} "
              f"win={e['window_turns']}{rt}{mh} {reply[:70]!r}", flush=True)
    m = metrics(log, ctx)
    tag = f"bus_{model}" + (f"_{args.tag}" if args.tag else "")
    path = OUT / f"ctx_probe_{tag}_{ctx}.json"
    path.write_text(json.dumps({"model": model, "ctx": ctx, "backend": "bus", "params": params,
                                "persona_file": params["persona"], "metrics": m, "judge": None, "log": log},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"== bus {model} ctx={ctx}: recall={m['recall_rate']} far={m['recall_far_rate']} math={m['math_acc']} "
          f"retrieval={m.get('mem_retrieval_recall')} copied={m.get('mem_note_copied')} asides={m.get('mem_asides')} "
          f"guard={m.get('guard_fired')} router={m.get('router_acc')} cut={m['cut_by_length']} "
          f"L2={m['longgen'].get('L2_hedged')} wall={m['wall_total_s']}s -> {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
