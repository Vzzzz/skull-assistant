#!/usr/bin/env python3
"""Demo and debug console (T-058): the bus and the resources in one terminal — what the skull hears, decides, says.

  .venv/bin/python deploy/stand/skull_console.py [--demo] [--record bus.json]      # live, from Redis
  .venv/bin/python deploy/stand/skull_console.py --replay research/bench/results/mode_gate_2026-10-31_2_bus.json \
        [--speed 4] [--from 20:35:00] [--demo]                                       # a dump of the bus
  ssh -t jetson 'cd skull-assistant && .venv/bin/python deploy/stand/skull_console.py --demo'

Header: host, mode (A/B and the switch phase), network policy, the ENGAGED window, tts:speaking, memory (stand:
nvidia-smi; device: tegrastats; else /proc/meminfo), the microphone level. Feed: one line per event, and per owner turn
the latency by stage: end of the segment (VAD) -> transcript -> decision -> first fragment of the answer -> first sound
-> end of speech (events:tts from services/tts/speaker.py). --demo: wider lines, service fields off (score / why stay).
Replay reads the dump format of research/bench/gate_log ({t, s, f} as a JSON list); --record writes the same.
No curses: ANSI colours and a scroll region under a fixed header. stdlib + redis only.
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from array import array
from pathlib import Path

STREAMS = ["events:wakeword", "events:transcript", "events:ambient", "events:reply", "events:tts", "events:mode",
           "cmd:mode", "events:net", "events:vision"]
COLOR = {"events:wakeword": 90, "events:transcript": 36, "events:ambient": 33, "events:reply": 32, "events:tts": 90,
         "events:mode": 35, "cmd:mode": 35, "events:net": 34, "events:vision": 94, "latency": 37}
TAG = {"events:wakeword": "VAD ", "events:transcript": "STT ", "events:ambient": "GATE", "events:reply": "SAY ",
       "events:tts": "TTS ", "events:mode": "MODE", "cmd:mode": "CMD ", "events:net": "NET ", "events:vision": "LOOK",
       "latency": "  ⏱ "}
HEADER_LINES = 3


def clock(t):
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 10) % 10}"


def num(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


class Turn:
    """One owner turn, keyed by the transcript's ts (brain and TTS copy it): the times of its stages."""
    STAGES = ("segment_end", "transcript", "decision", "first_fragment", "first_sound", "speech_end")

    def __init__(self, ts):
        self.ts, self.t = ts, {}

    def mark(self, stage, t):
        self.t.setdefault(stage, t)

    def line(self):
        """«VAD→STT 0.21 · →gate 0.02 · →1st 0.66 · →sound 0.45 · speech 2.1 s» over the stages present, or None."""
        names = {"transcript": "→STT", "decision": "→gate", "first_fragment": "→1st", "first_sound": "→sound"}
        parts, prev = [], None
        for st in self.STAGES[:-1]:
            if st in self.t:
                if prev is not None and st in names:
                    parts.append(f"{names[st]} {self.t[st] - self.t[prev]:.2f}")
                prev = st
        if "speech_end" in self.t and "first_sound" in self.t:
            parts.append(f"speech {self.t['speech_end'] - self.t['first_sound']:.1f} s")
        start = self.t.get("segment_end", self.t.get("transcript"))
        if start is not None and "first_sound" in self.t:
            parts.append(f"total to sound {self.t['first_sound'] - start:.2f} s")
        elif start is not None and "first_fragment" in self.t:
            parts.append(f"total to 1st fragment {self.t['first_fragment'] - start:.2f} s")
        return " · ".join(parts) if len(parts) >= 2 else None


class Console:
    """Event -> feed lines; keeps the state shown in the header. Pure: no Redis, no terminal (tested on a dump)."""

    def __init__(self, demo=False, width=100, expect_tts=False):
        """expect_tts: events:tts will come (live; a dump that has them) — a turn closes at the end of speech, not at
        the final message of the reply."""
        self.demo, self.width, self.expect_tts = demo, width, expect_tts
        self.state = {"mode": "A", "transition": "", "net": "?", "speaking": False, "engaged_s": None, "frame": False,
                      "mem": "", "vu": 0.0}
        self.turns, self.done, self.last_wake_end, self.reply_turn = {}, [], None, {}

    # -- helpers
    def _turn(self, ts):
        if not ts:
            return None
        if ts not in self.turns:
            self.turns[ts] = Turn(ts)
        return self.turns[ts]

    def _cut(self, s):
        s = re.sub(r"\s+", " ", str(s))
        return s if len(s) <= self.width - 22 else s[:self.width - 23] + "…"

    def _line(self, t, s, text):
        text = text if s == "latency" else self._cut(text)     # the latency line is the point: never cut
        return f"\x1b[{COLOR.get(s, 37)}m{clock(t)} {TAG.get(s, s[:4])} {text}\x1b[0m"

    # -- one event
    def feed(self, t, s, f):
        """-> list of feed lines (possibly empty) for event (t: seconds, s: stream, f: fields as str -> str)."""
        f = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.items()}
        k = (s.decode() if isinstance(s, bytes) else s)
        fn = getattr(self, "_" + k.replace(":", "_"), None)
        return [x for x in (fn(t, f) if fn else [self._line(t, k, json.dumps(f, ensure_ascii=False))]) if x]

    def _events_wakeword(self, t, f):
        if f.get("word") == "speech":
            self.last_wake_end = num(f.get("end_ts"), t)
            return [] if self.demo else [self._line(t, "events:wakeword", f"speech {f.get('score', '?')} s")]
        return [self._line(t, "events:wakeword", f"wake word {f.get('word')} {f.get('score', '')}")]

    def _events_transcript(self, t, f):
        turn = self._turn(f.get("ts"))
        if turn:
            if self.last_wake_end and 0 <= (num(f.get("ts"), t) - self.last_wake_end) < 5:
                turn.mark("segment_end", self.last_wake_end)
            turn.mark("transcript", t)
        spk = f.get("speaker", "?")
        extra = "" if self.demo or f.get("deferred") else f" seg {f.get('seg_s', '?')} s pause {f.get('pause_s', '?')}"
        tag = " (deferred)" if f.get("deferred") else ""
        return [self._line(t, "events:transcript", f"{spk} {f.get('spk_score', '')}{extra}{tag} «{f.get('text', '')}»")]

    def _events_ambient(self, t, f):
        turn = self._turn(f.get("ts")) if f.get("text") else None
        if turn:
            turn.mark("decision", t)
        if f.get("decision") in ("gaze", "gaze_drop"):
            return [self._line(t, "events:ambient", f"{f['decision']} — {f.get('why', '')} «{f.get('reply', '')}»")]
        extra = "" if self.demo else f" [{f.get('state', '')}, {f.get('speaker', '')}{', p_aside ' + f['p_aside'] if f.get('p_aside') else ''}]"
        return [self._line(t, "events:ambient", f"{f.get('decision', '?')} — {f.get('why', '')}{extra}")]

    def _events_reply(self, t, f):
        out, ts, rid = [], f.get("ts"), f.get("reply_id")
        turn = self.turns.get(ts)
        if turn and f.get("text"):
            turn.mark("first_fragment", t)
        if rid and ts:
            self.reply_turn[rid] = ts
        if f.get("text"):
            out.append(self._line(t, "events:reply", f"«{f['text']}»"))
        if "meta" in f:
            try:
                m = json.loads(f["meta"])
            except ValueError:
                m = {}
            parts = []
            if m.get("route"):
                r = m["route"]
                parts.append(f"route {r.get('label')}" + (f" {max((r.get('probs') or {}).values() or [0]):.2f}" if r.get("probs") else ""))
            if m.get("mode") in ("A", "B"):
                parts.append(f"mode {m['mode']}")
            elif m.get("mode") or "turn" not in m and m.get("reply"):
                parts.append("fixed line")              # brain's constants (mode switch, net, transition hold)
            if m.get("ambient"):
                parts.append(m["ambient"])
            if m.get("mem_hits"):
                parts.append(f"mem {len(m['mem_hits'])} ({max(h.get('score', 0) for h in m['mem_hits']):.2f})")
            if m.get("calc"):
                parts.append(m["calc"])
            if m.get("net"):
                parts.append(f"net {m['net'] if isinstance(m['net'], str) else ', '.join(m['net'])}")
            if not self.demo:
                for key, label in (("prompt_tokens", "prompt"), ("window_turns", "window"), ("completion_tokens", "out")):
                    if m.get(key) is not None:
                        parts.append(f"{label} {m[key]}")
                if m.get("llm_calls"):
                    parts.append("calls " + ",".join(m["llm_calls"]))
            if parts:
                out.append(self._line(t, "events:reply", "· " + " · ".join(parts)))
            if turn and not self.expect_tts:
                out += self._close(turn, t)
        return out

    def _events_tts(self, t, f):
        turn = self.turns.get(f.get("ts") or self.reply_turn.get(f.get("reply_id")))
        if not turn:
            return []
        if f.get("phase") == "first_audio":
            turn.mark("first_sound", num(f.get("t"), t))
            return [] if self.demo else [self._line(t, "events:tts", f"first audio {f.get('reply_id')}")]
        if f.get("phase") == "end":
            turn.mark("speech_end", num(f.get("t"), t))
            return self._close(turn, t)
        return []

    def _close(self, turn, t):
        line = turn.line()
        self.turns.pop(turn.ts, None)
        if line:
            self.done.append(turn)
            return [self._line(t, "latency", line)]
        return []

    def _events_mode(self, t, f):
        ph = f.get("phase", "")
        if ph == "begin":
            self.state["transition"] = f"{f.get('from')}→{f.get('to')}"
        elif ph in ("done", "failed", "noop"):
            self.state["transition"] = ""
            if ph == "done":
                self.state["mode"] = f.get("to", self.state["mode"])
        if self.demo and ph not in ("begin", "done", "failed"):
            return []
        dt = f" {f['dt_ms']} ms" if f.get("dt_ms") else ""
        return [self._line(t, "events:mode", f"{f.get('from')}→{f.get('to')} {ph} {f.get('service', '')}{dt} (t {f.get('t_ms', '?')} ms)"
                           + (f" {f['reason']}" if f.get("reason") and ph == "begin" else ""))]

    def _cmd_mode(self, t, f):
        return [self._line(t, "cmd:mode", f"→ {f.get('mode')} ({f.get('reason', '')})")]

    def _events_net(self, t, f):
        return [self._line(t, "events:net", f"{f.get('phase')}" + (f", {f['n']} results" if f.get("n") else ""))]

    def _events_vision(self, t, f):
        self.state["frame"] = f.get("frame") == "1"
        return [self._line(t, "events:vision", f"{f.get('kind')} {f.get('dur_s', '')} s yaw {f.get('yaw', '?')}"
                           + (" + frame" if f.get("frame") == "1" else ""))]

    # -- header
    def header(self, host=""):
        st = self.state
        mode = st["mode"] + (f" ({st['transition']})" if st["transition"] else "")
        eng = "—" if st["engaged_s"] is None else ("open" if st["engaged_s"] == math.inf else f"{st['engaged_s']:.0f} s")
        spk = "\x1b[32mspeaking\x1b[0m" if st["speaking"] else "quiet"
        line1 = (f"\x1b[1m{host or socket.gethostname()}\x1b[0m  mode \x1b[35m{mode}\x1b[0m  net {st['net']}  "
                 f"ENGAGED {eng}  {spk}  {st['mem']}")
        bar = int(min(1.0, st["vu"] / 4000) * 40)
        line2 = "mic " + "\x1b[32m" + "█" * bar + "\x1b[0m" + "·" * (40 - bar) + f" {st['vu']:5.0f}"
        return [line1, line2, "─" * min(self.width, shutil.get_terminal_size((100, 30)).columns)]


# -- sources

def resources():
    """«GPU 3812 MB» on the stand, «RAM 3456/7620 MB GR3D 45%» on the device, else RAM from /proc/meminfo."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=2).stdout.split(",")
        if len(out) == 2:
            return f"GPU {int(out[0])}/{int(out[1])} MB"
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        p = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE, text=True)
        line = p.stdout.readline(); p.kill()
        ram = re.search(r"RAM (\d+)/(\d+)MB", line); gr = re.search(r"GR3D_FREQ (\d+)%", line)
        if ram:
            return f"RAM {ram.group(1)}/{ram.group(2)} MB" + (f" GR3D {gr.group(1)}%" if gr else "")
    except OSError:
        pass
    m = dict(re.findall(r"(\w+):\s+(\d+)", Path("/proc/meminfo").read_text()))
    return f"RAM {(int(m['MemTotal']) - int(m['MemAvailable'])) // 1024}/{int(m['MemTotal']) // 1024} MB"


def live(con, rec, draw):
    import redis
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from core.ambient import engaged_until
    r = redis.Redis(socket_timeout=None)
    # the last id now, not "$": "$" re-sent every round would only see what arrives during that round's block —
    # events published while the console polls keys or draws would be lost
    ids = {s: ((r.xrevrange(s, count=1) or [[b"0-0"]])[0][0]) for s in STREAMS}
    next_keys = next_res = next_mic = 0.0
    while True:
        now = time.time()
        if now >= next_mic:
            x = r.xrevrange("audio:mic", count=1)
            if x and b"pcm" in x[0][1]:
                a = array("h"); a.frombytes(x[0][1][b"pcm"][: len(x[0][1][b"pcm"]) // 2 * 2])
                con.state["vu"] = math.sqrt(sum(v * v for v in a) / max(len(a), 1))
            next_mic = now + 0.1
        if now >= next_keys:
            con.state["mode"] = (r.get("mode:current") or b"A").decode()
            con.state["net"] = (r.get("net:policy") or b"manifest").decode()
            con.state["speaking"] = bool(r.exists("tts:speaking"))
            eu = engaged_until(r)
            con.state["engaged_s"] = math.inf if eu == math.inf else (eu - now if eu > now else None)
            con.state["frame"] = bool(r.exists("vision:frame"))
            next_keys = now + 0.5
        if now >= next_res:
            con.state["mem"] = resources(); next_res = now + 2.0
        for s, items in r.xread(ids, block=100, count=50) or []:
            s = s.decode()
            for i, f in items:
                ids[s] = i
                t = int(i.decode().split("-")[0]) / 1000
                f = {k.decode(): v.decode(errors="replace") for k, v in f.items()}
                if rec:
                    rec.write({"t": round(t, 3), "s": s, "f": f})
                draw(con.feed(t, s, f))
        draw([])


def load_dump(path):
    """The dump as a list; a recording killed before close() lacks the closing bracket."""
    text = Path(path).read_text(encoding="utf-8").rstrip()
    return json.loads(text if text.endswith("]") else text.rstrip(",") + "]")


def replay(con, path, speed, start, draw):
    events = load_dump(path)
    con.expect_tts = con.expect_tts or any(e["s"] == "events:tts" for e in events)
    if start:
        events = [e for e in events if time.strftime("%H:%M:%S", time.localtime(e["t"])) >= start]
    prev = None
    for e in events:
        if prev is not None and speed > 0:
            time.sleep(max(0.0, min(5.0, (e["t"] - prev) / speed)))
        prev = e["t"]
        draw(con.feed(e["t"], e["s"], e["f"]))


class Recorder:
    """Writes the dump as a JSON list, closed on exit (a killed run leaves `[…,` — replay accepts it)."""

    def __init__(self, path):
        self.fh, self.n = open(path, "w", encoding="utf-8"), 0
        self.fh.write("[\n")

    def write(self, ev):
        self.fh.write((",\n" if self.n else "") + json.dumps(ev, ensure_ascii=False)); self.fh.flush(); self.n += 1

    def close(self):
        self.fh.write("\n]\n"); self.fh.close()


class Screen:
    """A fixed header at the top, the feed scrolling under it (ANSI scroll region; plain print when not a tty)."""

    def __init__(self, con):
        self.con, self.tty, self.last = con, sys.stdout.isatty(), 0.0
        if self.tty:
            rows = shutil.get_terminal_size((100, 30)).lines
            sys.stdout.write(f"\x1b[2J\x1b[{HEADER_LINES + 1};{rows}r\x1b[{rows};1H")

    def __call__(self, lines):
        for l in lines:
            print(l)
        if self.tty and (lines or time.time() - self.last > 0.5):
            hdr = self.con.header()
            sys.stdout.write("\x1b7" + "".join(f"\x1b[{i + 1};1H\x1b[2K{h}" for i, h in enumerate(hdr)) + "\x1b8")
            self.last = time.time()
        sys.stdout.flush()

    def close(self):
        if self.tty:
            sys.stdout.write("\x1b[r\n"); sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", action="store_true", help="wider lines, service fields off")
    ap.add_argument("--replay", type=Path, help="a dump {t, s, f} (JSON list)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed; 0 = no pauses")
    ap.add_argument("--from", dest="start", help="replay from HH:MM:SS")
    ap.add_argument("--record", type=Path, help="write the live bus to this dump")
    a = ap.parse_args()
    con = Console(demo=a.demo, width=100 if a.demo else max(100, shutil.get_terminal_size((120, 30)).columns),
                  expect_tts=not a.replay)
    screen = Screen(con)
    rec = Recorder(a.record) if a.record else None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))   # kill closes the dump too
    try:
        if a.replay:
            replay(con, a.replay, a.speed, a.start, screen)
        else:
            live(con, rec, screen)
    except KeyboardInterrupt:
        pass
    finally:
        screen.close()
        if rec:
            rec.close()
        if con.done:
            print(f"turns with latency: {len(con.done)}")


if __name__ == "__main__":
    main()
