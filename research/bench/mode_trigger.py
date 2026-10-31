"""Mode B by voice through the bus (T-046 step 5): trigger phrase -> «Сейчас.» -> switch -> the deferred question in B;
turns in B (llm calls per turn); the exit phrase -> «Всё, я обратно.»; the idle exit; skull-tts CPU / RSS while the
transition sound plays. Turns are published as the owner's (speaker owner, trigger wakeword), so the voice id and the
mic are not involved; the sink is muted while it runs (pactl), the owner's dialog:history / dialog:vec are restored.

  .venv/bin/python research/bench/mode_trigger.py [--runs 3] [--idle]   # --idle: wait for the idle exit too
JSON -> research/bench/results/mode_trigger_<date>.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import redis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/bench/results"
r = redis.Redis(socket_timeout=None)
SAVED = ("dialog:history", "dialog:vec")
SEED = ["Кот опять спал на клавиатуре. Его зовут Бублик.", "Ключи я переложил в передний карман рюкзака.",
        "Завтра в половине двенадцатого придёт Дима, с собакой."]
TRIGGER = "Костян, подумай как следует: где мои ключи и кто завтра придёт?"
IN_B = ["А почему кот любит спать именно на клавиатуре?", "Ладно. А сколько лет живут коты?"]


def last_id(stream):
    x = r.xrevrange(stream, count=1)
    return x[0][0] if x else "0"


def publish(text):
    r.xadd("events:transcript", {"text": text, "trigger": "wakeword", "speaker": "owner", "ts": f"{time.time():.3f}"})


def collect(t0, until, timeout=240, rep_from=None, mode_from=None):
    """Read events:reply and events:mode from the ids given until `until(events)` -> list of (t - t0, stream, fields)."""
    ids = {"events:reply": rep_from, "events:mode": mode_from}
    out, t_end = [], time.time() + timeout
    while time.time() < t_end:
        ev = r.xread(ids, count=50, block=1000)
        for stream, items in ev or []:
            for i, f in items:
                ids[stream.decode()] = i
                out.append((round(time.time() - t0, 2), stream.decode(), {k.decode(): v.decode() for k, v in f.items()}))
        if until(out):
            return out
    raise SystemExit(f"timeout; got {[(t, s, f.get('phase') or f.get('text', '')[:30]) for t, s, f in out]}")


def finals(events):
    return [(t, f) for t, s, f in events if s == "events:reply" and f.get("final", "1") == "1"]


def proc(unit):
    pid = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", unit], capture_output=True, text=True).stdout.strip()
    st = Path(f"/proc/{pid}/stat").read_text().split()
    s = Path(f"/proc/{pid}/status").read_text()
    return {"cpu_s": (int(st[13]) + int(st[14])) / 100, "VmRSS": int(s.split("VmRSS:")[1].split()[0]) // 1024,
            "VmHWM": int(s.split("VmHWM:")[1].split()[0]) // 1024}


def turn(text):
    t0, rep, mode = time.time(), last_id("events:reply"), last_id("events:mode")
    publish(text)
    ev = collect(t0, lambda e: len(finals(e)) >= 1, rep_from=rep, mode_from=mode)
    first = next(t for t, s, f in ev if s == "events:reply")
    meta = json.loads(finals(ev)[0][1].get("meta", "{}"))
    return {"text": text, "first_chunk_s": first, "final_s": finals(ev)[0][0], "reply": meta.get("reply"),
            "mode": meta.get("mode"), "llm_calls": meta.get("llm_calls"), "prompt_tokens": meta.get("prompt_tokens")}


def enter(text):
    """Trigger -> «Сейчас.» -> begin..done -> the deferred answer (final)."""
    tts0 = proc("skull-tts")
    t0, rep, mode = time.time(), last_id("events:reply"), last_id("events:mode")
    publish(text)
    ev = collect(t0, lambda e: len(finals(e)) >= 2, rep_from=rep, mode_from=mode)
    tts1 = proc("skull-tts")
    ph = {f["phase"]: t for t, s, f in ev if s == "events:mode"}
    reason = next((f.get("reason") for t, s, f in ev if s == "events:mode" and f.get("phase") == "begin"), None)
    replies = [(t, f) for t, s, f in ev if s == "events:reply"]
    wait_line = replies[0]
    answer_first = next(t for t, f in replies if f.get("reply_id") != wait_line[1].get("reply_id"))
    meta = json.loads(finals(ev)[-1][1].get("meta", "{}"))
    span = (ph.get("done", 0) - ph.get("begin", 0)) or 1
    return {"text": text, "wait_line": wait_line[1].get("text"), "wait_line_s": wait_line[0], "begin_s": ph.get("begin"),
            "done_s": ph.get("done"), "reason": reason, "answer_first_chunk_s": answer_first, "answer_final_s": finals(ev)[-1][0],
            "answer": meta.get("reply"), "mode": meta.get("mode"), "llm_calls": meta.get("llm_calls"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "tts_cpu_pct_during_switch": round(100 * (tts1["cpu_s"] - tts0["cpu_s"]) / max(answer_first, 1e-3), 1),
            "tts_rss": tts1["VmRSS"], "tts_hwm": tts1["VmHWM"], "switch_s": round(span, 2)}


def leave(text):
    t0, rep, mode = time.time(), last_id("events:reply"), last_id("events:mode")
    publish(text)
    ev = collect(t0, lambda e: len(finals(e)) >= 1 and any(s == "events:mode" and f.get("phase") == "done" for _, s, f in e),
                 rep_from=rep, mode_from=mode)
    ph = {f["phase"]: t for t, s, f in ev if s == "events:mode"}
    back = finals(ev)[0]
    return {"text": text, "back_line": back[1].get("text"), "back_line_s": back[0], "begin_s": ph.get("begin"),
            "done_s": ph.get("done"), "reason": next((f.get("reason") for t, s, f in ev if f.get("phase") == "begin"), None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--idle", action="store_true", help="enter B without a question and wait for the idle exit")
    a = ap.parse_args()
    if (r.get("mode:current") or b"A") != b"A":
        raise SystemExit("start in mode A")
    saved = {k: r.dump(k) for k in SAVED}
    subprocess.run(["timeout", "5", "pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"])
    res = {"enter": [], "in_b": [], "leave": []}
    try:
        r.xadd("events:transcript", {"text": "/reset"})
        time.sleep(1)
        for s in SEED:
            turn(s)
        for i in range(a.runs):
            e = enter(TRIGGER)
            res["enter"].append(e)
            print(f"enter {i + 1}: «{e['wait_line']}» +{e['wait_line_s']} s, switch {e['begin_s']}..{e['done_s']} s, "
                  f"answer first chunk +{e['answer_first_chunk_s']} s ({e['prompt_tokens']} tok, calls {e['llm_calls']}), "
                  f"tts CPU {e['tts_cpu_pct_during_switch']} %: {str(e['answer'])[:80]!r}", flush=True)
            for q in IN_B:
                t = turn(q)
                res["in_b"].append(t)
                print(f"  in B: {t['first_chunk_s']} s, calls {t['llm_calls']}, mode {t['mode']}: {str(t['reply'])[:60]!r}", flush=True)
            lv = leave("Хватит думать")
            res["leave"].append(lv)
            print(f"leave {i + 1}: «{lv['back_line']}» +{lv['back_line_s']} s (switch {lv['begin_s']}..{lv['done_s']} s, {lv['reason']})", flush=True)
            res.setdefault("in_a", []).append(turn("А сейчас ты кто, обычный?"))
            print(f"  in A: calls {res['in_a'][-1]['llm_calls']}, mode {res['in_a'][-1]['mode']}", flush=True)
        if a.idle:
            t0, rep, mode = time.time(), last_id("events:reply"), last_id("events:mode")
            publish("Подумай хорошенько")
            ev = collect(t0, lambda e: any(s == "events:mode" and f.get("phase") == "done" and f.get("to") == "A"
                                           for _, s, f in e), timeout=900, rep_from=rep, mode_from=mode)
            res["idle"] = [(t, s, f.get("phase") or f.get("text"), f.get("reason")) for t, s, f in ev]
            print("idle:", res["idle"], flush=True)
    finally:
        subprocess.run(["timeout", "5", "pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"])
        for k, v in saved.items():
            r.delete(k)
            if v:
                r.restore(k, 0, v)
        subprocess.run(["systemctl", "--user", "restart", "skull-brain"])   # reload memory from the restored history
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"mode_trigger_{time.strftime('%Y-%m-%d_%H%M')}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
