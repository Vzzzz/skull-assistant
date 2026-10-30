"""Mode switch A <-> B on the stand (T-044 step 5): phases, memory, mic continuity, CTX-1 through a switch.

Needs the running skull-dispatcher and brain; TTS out of the way (`systemctl --user disable --now skull-tts`: a
disabled unit is not managed by the dispatcher, so it is not started again). The owner's dialog:history /
dialog:vec are saved before and restored after (CTX-1 resets the history).

  .venv/bin/python research/bench/mode_switch.py series [--n 10]   # A->B, B->A ... each followed by a fixed question
  .venv/bin/python research/bench/mode_switch.py restore [--n 2]   # A->B->A without a turn between: slot restore path
  .venv/bin/python research/bench/mode_switch.py ctx [--no-switch] # CTX-1: turns 1-30 in A, 31-60 in B, 5 control
                                                                   # questions back in A (or all in A: the baseline)
  .venv/bin/python research/bench/mode_switch.py thaw               # thaw_budget_s of the services without one: stop,
        # start, time to ready (embed: /health; capture: a new audio:mic entry; stt / tts / vision: their ready line)
Per switch: the phases from events:mode, VRAM (nvidia-smi every 200 ms) before / peak / after, skull-llm VmHWM and
RssAnon before it is stopped, the first reply chunk after the question, the longest gap in audio:mic and the
wake word's NRestarts. JSON -> research/bench/results/mode_switch_<date>_<what>.json.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import redis

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/eval"))
from ctx_bus import say  # noqa: E402
from ctx_probe import DIALOG, any_kw, metrics  # noqa: E402

OUT = ROOT / "research/bench/results"
r = redis.Redis(socket_timeout=None)
QUESTION = "Сколько будет семь умножить на восемь?"
CONTROL = [  # anchors of the first half (turns 1-30), asked again after B -> A
    ("C1", "A1", "Как зовут моего кота?", ["Бублик"]),
    ("C2", "A2", "С кем у меня завтра встреча и во сколько?", ["11:30", "половине двенадцатого", "пол двенадцатого", "11.30"]),
    ("C3", "A3u", "Где сейчас мои ключи?", ["рюкзак"]),
    ("C4", "A4", "Что мне нельзя класть в еду?", ["кинз"]),
    ("C5", "A5", "Какой код от домофона?", ["4712", "четыре семь один два", "47 12"]),
]
SAVED = ("dialog:history", "dialog:vec")


class Monitor:
    """VRAM every 200 ms (nvidia-smi) and the age of the newest audio:mic entry every 100 ms."""

    def __init__(self):
        self.vram, self.mic_gap, self.stop = [], [], False
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "200"],
                                  stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self.read_vram, daemon=True).start()
        threading.Thread(target=self.watch_mic, daemon=True).start()

    def read_vram(self):
        for line in self.p.stdout:
            if line.strip().isdigit():
                self.vram.append((time.time(), int(line)))

    def watch_mic(self):
        rr = redis.Redis()
        while not self.stop:
            last = rr.xinfo_stream("audio:mic")["last-generated-id"].decode()
            self.mic_gap.append((time.time(), time.time() - int(last.split("-")[0]) / 1000))
            time.sleep(0.1)

    def vram_between(self, t0, t1):
        v = [m for t, m in self.vram if t0 <= t <= t1]
        return max(v) if v else None

    def vram_at(self, t):
        v = [m for tt, m in self.vram if tt <= t]
        return v[-1] if v else None

    def mic_max_gap(self, t0, t1):
        g = [a for t, a in self.mic_gap if t0 <= t <= t1]
        return round(max(g), 2) if g else None

    def close(self):
        self.stop = True
        self.p.kill()


def unit_pid(unit):
    return int(subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", unit],
                              capture_output=True, text=True).stdout.strip() or 0)


def proc_mem(pid):
    try:
        s = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return {}
    return {k: int(s.split(k + ":")[1].split()[0]) // 1024 for k in ("VmHWM", "VmRSS", "RssAnon") if k + ":" in s}


def nrestarts(unit):
    return int(subprocess.run(["systemctl", "--user", "show", "-p", "NRestarts", "--value", unit],
                              capture_output=True, text=True).stdout.strip() or 0)


def switch(dst, mon, reason):
    """Ask the dispatcher, collect its events for this request -> dict."""
    llm_pid = unit_pid("skull-llm")
    mem_before = proc_mem(llm_pid)
    last = r.xinfo_stream("events:mode")["last-generated-id"] if r.exists("events:mode") else "0-0"
    req = f"ms{int(time.time() * 1000) % 10**8}"
    t0 = time.time()
    r.xadd("cmd:mode", {"mode": dst, "reason": reason, "id": req})
    phases, failed = [], False
    while True:
        ev = r.xread({"events:mode": last}, count=50, block=180000)
        if not ev:
            raise SystemExit("no events from the dispatcher in 180 s")
        for last, f in ev[0][1]:
            f = {k.decode(): v.decode() for k, v in f.items()}
            if not f.get("id", "").startswith(req):
                continue
            phases.append(f)
            failed |= f["phase"] == "failed"
        if phases and phases[-1]["phase"] in ("done", "noop") or (phases and phases[-1]["phase"] == "failed" and phases[-1]["id"].endswith("-rb")):
            break
    t1 = time.time()
    return {"to": dst, "t0": t0, "t1": t1, "total_s": round(t1 - t0, 2), "failed": failed, "phases": phases,
            "llm_mem_before_stop": mem_before, "vram_before": mon.vram_at(t0), "vram_peak": mon.vram_between(t0, t1),
            "vram_after": mon.vram_at(t1)}


def ask(text, mon):
    """A turn through the bus; the time to the first reply chunk and to the final one."""
    last = r.xrevrange("events:reply", count=1)
    last_id = last[0][0] if last else "0"
    t0 = time.time()
    r.xadd("events:transcript", {"text": text, "ts": f"{t0:.3f}"})
    first = None
    while True:
        resp = r.xread({"events:reply": last_id}, count=50, block=300000)
        if not resp:
            raise SystemExit(f"no reply to {text!r}")
        for last_id, f in resp[0][1]:
            first = first or time.time()
            if f.get(b"final", b"1") == b"1":
                meta = json.loads(f.get(b"meta", b"{}"))
                return {"text": text, "reply": meta.get("reply", f[b"text"].decode()), "first_chunk_s": round(first - t0, 2),
                        "final_s": round(time.time() - t0, 2), "ttft_s": meta.get("ttft_s"),
                        "prompt_tokens": meta.get("prompt_tokens"), "vram_peak": mon.vram_between(t0, time.time())}


def series(n, mon, with_turns=True):
    out = []
    for i in range(n):
        dst = "B" if (r.get("mode:current") or b"A").decode() == "A" else "A"
        s = switch(dst, mon, f"bench {i + 1}/{n}")
        if with_turns:
            s["turn"] = ask(QUESTION, mon)
            s["vram_peak_mode"] = mon.vram_between(s["t1"], time.time())
            s["llm_mem_after_turn"] = proc_mem(unit_pid("skull-llm"))
        s["mic_max_gap_s"] = mon.mic_max_gap(s["t0"], time.time())
        out.append(s)
        ph = {p["phase"] + (":" + p["service"] if p.get("service") else ""): p.get("dt_ms") for p in s["phases"]}
        print(f"{i + 1:2d} -> {dst}: {s['total_s']:5.1f} s, VRAM {s['vram_before']} / peak {s['vram_peak']} / after "
              f"{s['vram_after']}, mic gap {s['mic_max_gap_s']} s"
              + (f", first chunk {s['turn']['first_chunk_s']} s ({s['turn']['prompt_tokens']} tok)" if with_turns else "")
              + f"\n     {ph}", flush=True)
    return out


def ctx(mon, no_switch):
    rows = [json.loads(l) for l in DIALOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    if (r.get("mode:current") or b"A").decode() != "A":
        switch("A", mon, "ctx: start in A")
    r.xadd("events:transcript", {"text": "/reset"})
    time.sleep(1)
    log, switches = [], []
    for row in rows:
        if row["turn"] == 31 and not no_switch:
            switches.append(switch("B", mon, "ctx: turn 31"))
        t0 = time.time()
        reply, meta = say(r, row["text"])
        e = {**row, "reply": reply, "mode": (r.get("mode:current") or b"A").decode(), "wall_s": round(time.time() - t0, 2),
             "prompt_eval_count": meta.get("prompt_tokens", 0), "eval_count": meta.get("completion_tokens", 0),
             "ttft_s": meta.get("ttft_s", 0), "tok_s": meta.get("tok_s"), "window_turns": meta.get("window_turns"), "think": False,
             "thinking_chars": 0, "ctx_overflow_dropped": 0, "done_reason": meta.get("finish_reason"),
             "max_tokens": meta.get("max_tokens")}
        if "route" in meta:
            e["route"] = meta["route"]
        log.append(e)
        print(f"t{row['turn']:02d} [{e['mode']}] {e['wall_s']:5.1f} s pe={e['prompt_eval_count']:5d} {reply[:70]!r}", flush=True)
    if not no_switch:
        switches.append(switch("A", mon, "ctx: control"))
    control = []
    for cid, anchor, text, kws in CONTROL:
        t0 = time.time()
        turn = ask(text, mon)
        reply = turn["reply"]
        hit = any_kw(reply, kws)
        control.append({"id": cid, "anchor": anchor, "text": text, "reply": reply, "hit": hit,
                        "wall_s": round(time.time() - t0, 2), "first_chunk_s": turn["first_chunk_s"],
                        "prompt_tokens": turn["prompt_tokens"]})
        print(f"{cid} {anchor} {'HIT ' if hit else 'miss'} {time.time() - t0:5.1f} s {reply[:70]!r}", flush=True)
    restore_check = None
    if not no_switch:   # A -> B -> A with no turn between: the A slot is restored (history unchanged) — same question again
        switches.append(switch("B", mon, "ctx: restore check"))
        switches.append(switch("A", mon, "ctx: restore check"))
        restore_check = ask(CONTROL[0][2], mon)
        print(f"after restore: first chunk {restore_check['first_chunk_s']} s ({restore_check['prompt_tokens']} tok)", flush=True)
    raw = OUT / f"mode_switch_{time.strftime('%Y-%m-%d_%H%M')}_ctx_raw.json"   # in case the metrics fail
    raw.write_text(json.dumps({"log": log, "control": control, "switches": switches, "restore_check": restore_check},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    m = metrics(log, 8192)
    return {"metrics": m, "log": log, "control": control, "control_recall": sum(c["hit"] for c in control) / len(control),
            "switches": switches, "restore_check": restore_check}


READY = {  # unit -> how it says it is ready (no healthcheck in the manifest)
    "skull-stt": "waiting for wake events", "skull-tts": "dmitri ready", "skull-vision": "gaze on",
}


def thaw_times(units=("skull-capture", "skull-stt", "skull-tts", "skull-embed", "skull-vision"), runs=3):
    out = {}
    for u in units:
        was_active = subprocess.run(["systemctl", "--user", "is-active", u], capture_output=True, text=True).stdout.strip() == "active"
        times = []
        for _ in range(runs):
            subprocess.run(["systemctl", "--user", "stop", u])
            time.sleep(1)
            mic0 = r.xinfo_stream("audio:mic")["last-generated-id"]
            since = time.strftime("%Y-%m-%d %H:%M:%S")
            t0 = time.time()
            subprocess.run(["systemctl", "--user", "start", u])
            while time.time() - t0 < 60:
                if u == "skull-capture":
                    ok = r.xinfo_stream("audio:mic")["last-generated-id"] != mic0
                elif u == "skull-embed":
                    ok = subprocess.run(["curl", "-sf", "http://127.0.0.1:11438/health"], capture_output=True).returncode == 0
                else:
                    ok = READY[u] in subprocess.run(["journalctl", "--user", "-u", u, "--since", since, "--no-pager", "-o", "cat"],
                                                    capture_output=True, text=True).stdout
                if ok:
                    break
                time.sleep(0.05)
            times.append(round(time.time() - t0, 2) if ok else None)
        if not was_active:                     # vision / tts are off on the stand: leave them as found
            subprocess.run(["systemctl", "--user", "stop", u])
        out[u] = times
        print(f"{u}: {times} s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["series", "restore", "ctx", "thaw"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--no-switch", action="store_true", help="ctx: all in A (the baseline)")
    a = ap.parse_args()
    if subprocess.run(["systemctl", "--user", "is-active", "skull-tts"], capture_output=True, text=True).stdout.strip() == "active":
        raise SystemExit("skull-tts is active: systemctl --user disable --now skull-tts (re-enable afterwards)")
    saved = {k: r.dump(k) for k in SAVED}
    wake0 = nrestarts("skull-wakeword")
    mon = Monitor()
    time.sleep(1)
    t_start = time.time()
    try:
        if a.what == "series":
            res = {"switches": series(a.n, mon)}
        elif a.what == "thaw":
            res = {"thaw_s": thaw_times()}
        elif a.what == "restore":
            res = {"switches": series(2 * a.n, mon, with_turns=False)}
            res["turn_after"] = ask(QUESTION, mon)
        else:
            res = ctx(mon, a.no_switch)
    finally:
        mon.close()
        for k, v in saved.items():                    # the owner's conversation back
            r.delete(k)
            if v:
                r.restore(k, 0, v)
        # brain keeps the associative memory in-process (built from dialog:history at start): reload it
        subprocess.run(["systemctl", "--user", "restart", "skull-brain"])
    res.update({"wakeword_restarts": nrestarts("skull-wakeword") - wake0, "mic_max_gap_s": mon.mic_max_gap(t_start, time.time()),
                "units_after": subprocess.run(["systemctl", "--user", "list-units", "skull*", "--no-pager", "--plain", "--no-legend"],
                                              capture_output=True, text=True).stdout})
    tag = a.what + ("_noswitch" if a.no_switch else "")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"mode_switch_{time.strftime('%Y-%m-%d_%H%M')}_{tag}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wake word restarts {res['wakeword_restarts']}, mic max gap {res['mic_max_gap_s']} s -> {path.relative_to(ROOT)}")
    print(res["units_after"])


if __name__ == "__main__":
    main()
