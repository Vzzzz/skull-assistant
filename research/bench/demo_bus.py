"""The 8 scenes of docs/DEMO.md through the bus, without the owner (T-058 follow-up, demo prep).

Turns are published on events:transcript the way skull-stt does (trigger speech, speaker / spk_score / seg_s / bcast),
in the recording order of DEMO.md: 1 address -> 2 background -> 3 memory -> 4 calc -> 5 net («да» and «нет» branches) ->
6 mode B (A -> B -> A by the phrases) -> 7 a look (events:vision gaze + the frame in vision:frame; skull-vision is
stopped for it so the room's real looks do not take the turn; ok = the line is by the frame) -> 8 the end.
Before the scenes: /reset (as `restart --fresh`), net policy ask, mode B warmed once (core.dispatcher --switch B, then A).
The bus is recorded (all streams the console reads) and replayed through the console in --demo per scene.

  .venv/bin/python research/bench/demo_bus.py [--no-warm] [--scenes 7 8 --tag _78] [--frame <jpg>]
The stand must be up in mode A; the skull speaks through the speaker. History and memory are backed up and restored;
skull-stt and skull-vision are stopped for the run (the room stays out) and started again. A scene is ok by the gate's
decision on that very turn (events:ambient with its text), not by any reply that comes.
-> results/demo_bus_<date>_bus.json (dump), demo_bus_<date>.md (console lines per scene)
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import redis

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/stand"))
import skull_console as sc  # noqa: E402
sys.path.insert(0, str(ROOT))
from core.ambient import engaged_until  # noqa: E402

OUT = ROOT / "research/bench/results"
r = redis.Redis()
OWNER = {"speaker": "owner", "spk_score": "0.62"}
OTHER = {"speaker": "other", "spk_score": "0.08", "bcast": "1"}
MURMUR = {"speaker": "uncertain", "spk_score": "0.38"}

# (scene, text, voice, seg_s, expect): expect — 'reply' (wait for a final reply), 'silent' (a decision, no reply),
# 'ask' (the net question), 'B' / 'A' (a mode switch done), 'gaze'
SCENES = [
    (1, "Костян, ты тут?", OWNER, 1.2, "reply"),
    (1, "Чем занят?", OWNER, 0.9, "reply"),
    (2, "…и вот тогда, говорит он, мы и поняли, что рынок перегрет, а инвесторы просто не замечают сигналов", OTHER, 5.5, "silent"),
    (2, "Знаете, в девяностые всё было иначе, никто не думал о долгосрочных последствиях", OTHER, 4.8, "silent"),
    (2, "так, это надо переделать", MURMUR, 1.6, "silent"),
    (3, "Костян, я купил синий чайник, он свистит громче старого", OWNER, 3.1, "reply"),
    (3, "Ладно, а что ты думаешь про понедельники?", OWNER, 2.0, "reply"),     # chatter, not world questions (DEMO)
    (3, "Как тебе моя футболка?", OWNER, 1.3, "reply"),
    (3, "Ты вообще спишь когда-нибудь?", OWNER, 1.5, "reply"),
    (3, "Напомни, какого цвета чайник?", OWNER, 1.7, "reply"),
    (4, "Костян, сколько будет семнадцать умножить на двадцать три?", OWNER, 2.6, "reply"),
    (4, "Если выйду в восемь сорок и ехать час тридцать пять, во сколько приеду?", OWNER, 3.4, "reply"),
    (5, "Костян, какой город столица Австралии?", OWNER, 2.0, "ask"),
    (5, "Да", OWNER, 0.6, "reply"),
    (5, "Костян, какая сегодня погода?", OWNER, 1.5, "ask"),
    (5, "Нет", OWNER, 0.6, "reply"),
    (6, "Костян, подумай как следует: что ты помнишь про чайник и зачем он свистит?", OWNER, 4.2, "B"),
    (6, "Хватит думать", OWNER, 1.1, "A"),
    (7, None, None, None, "gaze"),
    (8, "Костян, спасибо, на сегодня всё", OWNER, 1.8, "reply"),
]


def last(stream):
    x = r.xrevrange(stream, count=1)
    return x[0][0] if x else b"0-0"


def publish(text, voice, seg_s):
    now = time.time()                  # no events:wakeword: skull-stt would cut audio by it
    r.xadd("events:transcript", {"text": text, "lang": "ru", "ts": f"{now:.3f}", "trigger": "speech", "pause_s": "0.85",
                                 "seg_s": f"{seg_s:.2f}", **voice})


def quiet(timeout):
    """Wait until the skull stops speaking (tts:speaking gone for 1.5 s) or the timeout."""
    t0, since = time.time(), None
    while time.time() - t0 < timeout:
        if r.exists("tts:speaking") or r.exists("mode:transition"):
            since = None
        elif since is None:
            since = time.time()
        elif time.time() - since > 1.5:
            return True
        time.sleep(0.2)
    return False


def wait_for(expect, ids, timeout, text=None):
    """Read the streams from `ids` until the expectation is met -> (ok, what was seen). The gate's decision on this very
    turn (events:ambient with its text) decides: `silent` wants silent / ack, `reply` and `ask` fail on silent / ack."""
    t0, seen = time.time(), []
    while time.time() - t0 < timeout:
        for s, items in r.xread(ids, block=500) or []:
            for i, f in items:
                ids[s] = i
                f = {k.decode(): v.decode(errors="replace") for k, v in f.items()}
                seen.append((s.decode(), f))
                # brain normalizes the text (time words -> 08:40): match by the head of the turn
                if s == b"events:ambient" and text and f.get("text", "")[:10] == text[:10]:
                    quiet_ok = f.get("decision") in ("silent", "ack")
                    if expect == "silent" or (expect in ("reply", "ask") and quiet_ok):
                        return expect == "silent" and quiet_ok, seen
                if expect in ("reply", "ask") and s == b"events:reply" and f.get("final", "1") == "1" and "meta" in f:
                    meta = json.loads(f["meta"])
                    if meta.get("ambient") == "ack" or (expect == "ask") != (meta.get("net") == "ask"):
                        continue
                    return True, seen
                if expect in ("A", "B") and s == b"events:mode" and f.get("phase") == "done" and f.get("to") == expect:
                    return True, seen
                if expect == "gaze" and s == b"events:ambient" and f.get("decision") == "gaze":
                    return f.get("frame") == "1", seen          # T-060 (9): the first look after /reset is by the frame
    return False, seen


def run_scenes(frame, only=None):
    marks = []
    for scene, text, voice, seg_s, expect in SCENES:
        if only and scene not in only:
            continue
        quiet(60)
        # the room stays out: a mode switch brings the residents back, so the mic and the camera go off before each turn
        subprocess.run(["systemctl", "--user", "stop", "skull-stt", "skull-vision"], check=False)
        ids = {s: last(s) for s in ("events:reply", "events:ambient", "events:mode")}
        t = time.time()
        if expect == "gaze":
            # a look inside the ENGAGED window gets no line by design — wait for it to close, as the owner must after
            # «Всё, я обратно.»
            t_w = time.time()
            while engaged_until(r) > time.time() and time.time() - t_w < 60:
                time.sleep(0.5)
            print(f"  (window closed after {time.time() - t_w:.0f} s)", flush=True)
            t = time.time()
            if frame:
                r.set("vision:frame", Path(frame).read_bytes(), ex=10)
            r.xadd("events:vision", {"kind": "gaze", "dur_s": "1.2", "frame": "1" if frame else "0", "ts": f"{t:.3f}"})
        else:
            publish(text, voice, seg_s)
        timeout = 150 if expect in ("A", "B") else 60
        if expect == "B":       # the deferred question is answered in B after the switch: wait for that reply too
            ok, seen = wait_for("B", ids, timeout)
            ok2, seen2 = wait_for("reply", ids, 90)       # the deferred question: no ambient decision, its reply
            ok, seen = ok and ok2, seen + seen2
        else:
            ok, seen = wait_for(expect, ids, timeout, text)
        print(f"scene {scene} {text or 'gaze'!r:.60} -> {expect} {'ok' if ok else 'TIMEOUT'} ({time.time() - t:.1f} s)", flush=True)
        marks.append({"scene": scene, "text": text, "expect": expect, "ok": ok, "t": round(t, 3)})
    quiet(60)
    return marks


def render(dump, marks, stamp):
    """The console in --demo over the dump, its lines grouped by scene (a scene: from its turn to the next one)."""
    con = sc.Console(demo=True, expect_tts=True)
    lines = [[] for _ in marks]
    for e in dump:
        k = max((i for i, m in enumerate(marks) if e["t"] >= m["t"] - 0.5), default=None)
        out = con.feed(e["t"], e["s"], e["f"])
        if k is not None:
            lines[k] += [re.sub(r"\x1b\[[0-9;]*m", "", x) for x in out]
    md = [f"# Демо по шине (8 сцен DEMO.md) · {stamp}", "", "Консоль `--demo` на дампе прогона, строки по сценам "
          "(`research/bench/demo_bus.py`; без владельца, голоса заданы полями шины).", ""]
    for m, ls in zip(marks, lines):
        md += [f"## Сцена {m['scene']}: {m['text'] or 'взгляд'} — ожидание `{m['expect']}`: "
               f"{'да' if m['ok'] else '**нет (таймаут)**'}", "", "```", *ls, "```", ""]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-warm", action="store_true", help="skip the B warm-up")
    ap.add_argument("--scenes", type=int, nargs="+", help="run only these scenes (e.g. 7 8)")
    ap.add_argument("--render", metavar="DUMP", help="only re-render the .md of a dump (its _marks.json next to it)")
    ap.add_argument("--tag", default="", help="suffix of the output names")
    ap.add_argument("--frame", default=str(ROOT / "research/vision/results/cam_probe_last.jpg"))
    a = ap.parse_args()
    if a.render:
        dump_path = Path(a.render)
        marks = json.loads(Path(str(dump_path).replace("_bus.json", "_marks.json")).read_text(encoding="utf-8"))
        md_path = Path(str(dump_path).replace("_bus.json", ".md"))
        md_path.write_text(render(sc.load_dump(dump_path), marks, md_path.stem.split("_")[2]), encoding="utf-8")
        print("->", md_path)
        return
    if (r.get("mode:current") or b"A") != b"A":
        raise SystemExit("mode must be A")
    stamp = time.strftime("%Y-%m-%d")
    backup = {k: r.dump(k) for k in ("dialog:history", "dialog:vec")}
    net_before = r.get("net:policy")
    rec_path = OUT / f"demo_bus_{stamp}{a.tag}_bus.json"
    rec = subprocess.Popen([sys.executable, str(ROOT / "deploy/stand/skull_console.py"), "--record", str(rec_path)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        r.set("net:policy", "ask")
        if not a.no_warm:
            for m in ("B", "A"):
                t0 = time.time()
                subprocess.run([sys.executable, "-m", "core.dispatcher", "--switch", m], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
                while (r.get("mode:current") or b"").decode() != m or r.exists("mode:transition"):
                    time.sleep(0.5)
                print(f"warm-up: mode {m} in {time.time() - t0:.1f} s", flush=True)
        r.xadd("events:transcript", {"text": "/reset"})
        time.sleep(1)
        marks = run_scenes(a.frame if a.frame and Path(a.frame).exists() else None, a.scenes)
    finally:
        time.sleep(2)
        rec.terminate()
        rec.wait(10)
        subprocess.run(["systemctl", "--user", "start", "skull-stt"], check=False)
        if (r.get("mode:current") or b"A") == b"A":
            subprocess.run(["systemctl", "--user", "start", "skull-vision"], check=False)
        for k, v in backup.items():          # the owner's history and memory back
            r.delete(k)
            if v:
                r.restore(k, 0, v)
        if net_before:
            r.set("net:policy", net_before)
        else:
            r.delete("net:policy")
    dump = sc.load_dump(rec_path)
    rec_path.write_text(json.dumps(dump, ensure_ascii=False, indent=0), encoding="utf-8")
    (OUT / f"demo_bus_{stamp}{a.tag}_marks.json").write_text(json.dumps(marks, ensure_ascii=False, indent=1), encoding="utf-8")
    md = render(dump, marks, stamp)
    out = OUT / f"demo_bus_{stamp}{a.tag}.md"
    out.write_text(md, encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
