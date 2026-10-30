"""Gaze calibration on the stand (T-031 steps 1 / 1b): does «looking at the skull» separate from the rest?

Frames are read from the camera at the service rate (services/vision manifest) and measured exactly as the service does
(services/vision/gaze.measure: YuNet nose offset, yaw / pitch; Face Mesh iris_h / iris_v / iris_vl / open); every frame's
features are logged with its phase. Frames are not stored. CSV -> research/vision/results/ (numbers only).

  .venv/bin/python research/vision/gaze_calib.py --natural 600   # step 1b recalibration: 10 min of work with natural
        # looks about once a minute; the owner holds Right Ctrl while looking (look_keys.ps1: global, any window in focus)
  .venv/bin/python research/vision/gaze_calib.py --marks 1500    # gate: look marks only (the service has the camera),
        # then the triggers in events:vision against them; --gate <marks csv> repeats that analysis later
  .venv/bin/python research/vision/gaze_calib.py --work 300 --looks 10    # step 1b, first try: 5 min of work, then
        # 10 looks on the console's command «СМОТРИ НА ЧЕРЕП» (looks on command turned out unlike natural ones)
  .venv/bin/python research/vision/gaze_calib.py --static skull --seconds 15     # one state, said in the chat
  .venv/bin/python research/vision/gaze_calib.py --report <csv> [<csv> ...]       # over saved recordings

Report: per feature, «skull» frames against everything else (work, between the looks, monitor): medians, p10 / p90,
robust spread (1.4826 x MAD) and the gap between the medians; criterion (core, 2026-10-29): gap >= 2 x the larger
spread. Bands are fixed in advance — [p5, p95] of the «skull» frames — for single features and their combinations; per
band: the share of frames inside it, and the service's dwell logic replayed over the log (triggers during work, looks
caught out of N). A recording with look marks (--natural): «skull» = frames inside the marked looks, «work» = the
rest except GUARD_S around each look; bands for iris_h + nose, gaze_h, gaze_h + nose; per band set: gap / spread, shares,
false triggers and the longest work run in the bands; dwell 1.0 / 1.25 / 1.5 s -> looks caught / false / margin.
"""
import argparse
import csv
import importlib.util
import math
import random
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import redis

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gaze", ROOT / "services/vision/gaze.py")
gaze = importlib.util.module_from_spec(spec); spec.loader.exec_module(gaze)   # measure() etc., no service loop
P, G = gaze.P, gaze.G
OUT = Path(__file__).resolve().parent / "results"
KEYS = ["phase", "look", "t", "settled", "face", "score", "nose", "yaw", "pitch", "iris_h", "iris_v", "iris_vl", "open",
        "presence", "gaze_h", "ms"]
FEATURES = ["iris_h", "iris_v", "iris_vl", "nose", "yaw", "pitch", "gaze_h"]
COMBOS = [("iris_h", "iris_v"), ("iris_h", "iris_vl"), ("iris_h", "nose"), ("iris_h", "iris_v", "nose")]
LOOK_S, SETTLE_S = 4.0, 1.2           # a look on command: held 4 s, the first 1.2 s the eyes are still moving
GUARD_S = 1.0                         # marked recordings: frames this close to a look are neither «skull» nor «work»
SETS = [("iris_h", "nose"), ("gaze_h",), ("gaze_h", "nose")]
DWELLS = (1.0, 1.25, 1.5)
MARK = ("key_down", "key_up", "session")


class Marks:
    """Right Ctrl held = looking at the skull (look_keys.ps1 on the Windows side through WSL interop)."""

    def __init__(self, seconds):
        ps1 = subprocess.run(["wslpath", "-w", str(Path(__file__).with_name("look_keys.ps1"))],
                             capture_output=True, text=True).stdout.strip()
        self.p = subprocess.Popen(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1,
                                   str(seconds + 10)], stdout=subprocess.PIPE, text=True)
        if self.p.stdout.readline().strip() != "ready":
            raise SystemExit("look_keys.ps1 did not start")
        self.rows, self.n = [], 0
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        for line in self.p.stdout:
            now, line = time.time(), line.strip()
            if line == "down":
                self.n += 1
                print(f"  взгляд {self.n}: отмечен", flush=True)
            if line in ("down", "up"):
                self.rows.append({"phase": "key_" + line, "look": self.n, "t": round(now, 3)})

    def close(self):
        self.p.kill()
        return self.rows


def open_camera():
    cap = cv2.VideoCapture(P.get("device", 0), cv2.CAP_V4L2)
    if P.get("fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*P["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, P["width"]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, P["height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def record(cap, det, sess, rows, phase, seconds, look=0, settle_s=0.0, tick=None):
    """Measure frames at the service rate for `seconds`, appending rows of `phase`."""
    period = 1.0 / G.get("fps", 5)
    t_start = time.time(); t_end = t_start + seconds; next_tick = t_start
    while time.time() < t_end:
        t0 = time.time()
        if tick and t0 >= next_tick:
            tick(t_end - t0); next_tick += 30
        ok, fr = cap.read()
        if not ok:
            continue
        m = gaze.measure(det, sess, fr, fr.shape[1], fr.shape[0])
        now = time.time()
        row = {"phase": phase, "look": look, "t": round(now, 3), "settled": int(now >= t_start + settle_s),
               "face": int(bool(m)), "ms": round(1000 * (now - t0))}
        row.update({k: round(v, 4) for k, v in m.items()})
        rows.append(row)
        time.sleep(max(0.0, period - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=float, help="seconds of ordinary work, then --looks looks on command")
    ap.add_argument("--looks", type=int, default=10)
    ap.add_argument("--static", choices=["skull", "monitor", "work"], help="record one state for --seconds")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--natural", type=float, help="seconds of work with natural looks marked by holding Right Ctrl")
    ap.add_argument("--marks", type=float, help="seconds of look marks only (gate: the service has the camera)")
    ap.add_argument("--gate", help="marks CSV: triggers in events:vision against the marked looks")
    ap.add_argument("--report", nargs="+", help="CSV files of earlier recordings")
    a = ap.parse_args()
    if a.report:
        return report(load(a.report))
    if a.gate:
        return gate(load([a.gate]))
    if a.marks:
        return record_marks(a.marks)
    cap = open_camera()
    ok, fr = cap.read()
    if not ok:
        raise SystemExit("no frame")
    det = cv2.FaceDetectorYN.create(str(ROOT / G["model"]), "", (fr.shape[1], fr.shape[0]), G["score_min"], 0.3, 5000)
    sess = gaze.mesh_session(ROOT / G["mesh_model"])
    rows = []
    if a.static:
        record(cap, det, sess, rows, a.static, a.seconds)
        tag = a.static
    elif a.natural:
        marks = Marks(a.natural)
        print(f"\n=== Работа как обычно, {a.natural / 60:.0f} мин; на череп смотреть естественно, примерно раз в минуту. "
              "Пока смотришь — держи правый Ctrl. ===", flush=True)
        t0 = time.time()
        record(cap, det, sess, rows, "natural", a.natural,
               tick=lambda left: print(f"  осталось {left / 60:.1f} мин, отмечено взглядов: {marks.n}", flush=True))
        rows += marks.close() + [{"phase": "session", "t": round(t0, 3)}, {"phase": "session", "t": round(time.time(), 3)}]
        rows.sort(key=lambda x: x["t"])
        print(f"\a=== Всё. Взглядов отмечено: {marks.n} ===", flush=True)
        tag = "natural"
    else:
        if a.work:
            print(f"\n=== Работа как обычно, {a.work / 60:.0f} мин. На череп НЕ смотреть. ===", flush=True)
            record(cap, det, sess, rows, "work", a.work,
                   tick=lambda left: print(f"  работа: осталось {left / 60:.1f} мин", flush=True))
        print(f"\a\n=== Взгляды: {a.looks} раз. По команде СМОТРИ — на череп, глазами, пока не появится ОТБОЙ. "
              "Между ними — в монитор / консоль. ===", flush=True)
        for k in range(1, a.looks + 1):
            record(cap, det, sess, rows, "between", random.uniform(5, 8), look=k, settle_s=1.5)
            print(f"\a  >>> СМОТРИ НА ЧЕРЕП ({k}/{a.looks})", flush=True)
            record(cap, det, sess, rows, "skull", LOOK_S, look=k, settle_s=SETTLE_S)
            print("  <<< ОТБОЙ", flush=True)
        print("\a=== Всё. ===", flush=True)
        tag = "work"
    OUT.mkdir(exist_ok=True)
    out = OUT / f"gaze_1b_{time.strftime('%Y-%m-%d_%H%M%S')}_{tag}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=KEYS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    ms = [x["ms"] for x in rows if "ms" in x]
    print(f"{len(rows)} frames -> {out}; measure ms median {np.median(ms):.0f} p95 {np.percentile(ms, 95):.0f}")
    report(rows)


def load(files):
    rows = []
    for f in files:
        for x in csv.DictReader(open(f, encoding="utf-8")):
            x = {k: v for k, v in x.items() if v not in (None, "")}
            if "geo" in x:                   # step 1 recordings: the nose offset was «geo»
                x["nose"] = x.pop("geo")
            for k in ("t", *FEATURES, "open", "score"):
                if k in x:
                    x[k] = float(x[k])
            x["settled"], x["face"], x["look"] = int(x.get("settled", 1)), int(x.get("face", 0)), int(x.get("look", x.get("cycle", 0)))
            rows.append(x)
    return sorted(rows, key=lambda x: x["t"])


def stats(v):
    v = np.asarray(v)
    med = float(np.median(v))
    return med, float(np.percentile(v, 10)), float(np.percentile(v, 90)), 1.4826 * float(np.median(np.abs(v - med)))


def replay(rows, bands, dwell=None):
    """The service's dwell logic over the log -> trigger rows."""
    dwell = dwell or G["dwell_s"]
    since, armed, last_event, last_in, out = None, True, -1e9, -1e9, []
    for x in rows:
        if x["phase"] in MARK:
            continue
        now = x["t"]
        if not gaze.in_bands(x, bands):
            if now - last_in > G.get("gap_s", 0.4):
                since, armed = None, True
        else:
            since, last_in = since or now, now
            if armed and now - since >= dwell and now - last_event >= G["min_interval_s"]:
                out.append(x); armed, last_event = False, now
    return out


def report(rows):
    if any(x["phase"] == "key_down" for x in rows):
        return report_natural(rows)
    pos = [x for x in rows if x["phase"] == "skull" and x["settled"]]
    neg = [x for x in rows if x["phase"] != "skull" and x["settled"]]
    phases = sorted({x["phase"] for x in rows})
    print(f"\nframes: " + ", ".join(f"{p} {sum(1 for x in rows if x['phase'] == p)} "
                                    f"(face {sum(1 for x in rows if x['phase'] == p and x['face'])}, "
                                    f"eyes {sum(1 for x in rows if x['phase'] == p and 'iris_h' in x)})" for p in phases))
    print("\nfeature   | state    | median  p10 .. p90       | spread | gap   | gap/spread")
    for k in FEATURES:
        s = [x[k] for x in pos if k in x]
        if not s:
            continue
        ms, *_, ss = stats(s)
        for p in phases:
            v = [x[k] for x in rows if x["phase"] == p and x["settled"] and k in x]
            if not v:
                continue
            m, p10, p90, sd = stats(v)
            gap = "" if p == "skull" else f"{abs(m - ms):5.2f} | {abs(m - ms) / max(ss, sd, 1e-9):4.1f}{'  ok' if abs(m - ms) >= 2 * max(ss, sd) else ''}"
            print(f"{k:<9} | {p:<8} | {m:+7.3f} {p10:+7.3f} .. {p90:+7.3f} | {sd:6.3f} | {gap}")
    if not pos:
        return
    looks = sorted({x["look"] for x in rows if x["phase"] == "skull"})
    print(f"\nbands = [p5, p95] of the settled «skull» frames; dwell {G['dwell_s']} s, gap {G.get('gap_s', 0.4)} s, "
          f"min interval {G['min_interval_s']} s replayed over the whole log")
    print("features              | band(s)                          | skull in | other in | triggers: work / between / looks caught")
    for combo in [(k,) for k in FEATURES] + COMBOS:
        if not all(any(k in x for x in pos) for k in combo):
            continue
        bands = {k: (float(np.percentile([x[k] for x in pos if k in x], 5)),
                     float(np.percentile([x[k] for x in pos if k in x], 95))) for k in combo}
        s_in = sum(gaze.in_bands(x, bands) for x in pos) / len(pos)
        o_in = sum(gaze.in_bands(x, bands) for x in neg) / max(len(neg), 1)
        trig = replay(rows, bands)
        by = {p: sum(1 for x in trig if x["phase"] == p) for p in ("work", "between", "skull", "monitor")}
        caught = len({x["look"] for x in trig if x["phase"] == "skull"})
        bs = " ".join(f"{k} [{lo:+.2f},{hi:+.2f}]" for k, (lo, hi) in bands.items())
        print(f"{'+'.join(combo):<21} | {bs:<32} | {s_in:8.2f} | {o_in:8.2f} | {by['work']} / {by['between']} / "
              f"{caught} of {len(looks)}" + (f" (monitor {by['monitor']})" if by["monitor"] else ""))


def looks_of(rows):
    looks, start = [], None
    for x in rows:
        if x["phase"] == "key_down":
            start = x["t"]
        elif x["phase"] == "key_up" and start is not None:
            looks.append((start, x["t"])); start = None
    return looks


def label(rows, looks):
    """Frames -> state: skull (inside a marked look), edge (within GUARD_S of one), work."""
    frames = [x for x in rows if x["phase"] not in MARK]
    for x in frames:
        x["state"] = "work"
        for a, b in looks:
            if a <= x["t"] <= b:
                x["state"] = "skull"; break
            if a - GUARD_S <= x["t"] <= b + GUARD_S:
                x["state"] = "edge"
    return frames


def longest_work_run(frames, bands):
    """The longest stretch of work frames inside the bands, dropouts <= gap_s allowed; a look breaks it."""
    best, since, last_in = 0.0, None, -1e9
    for x in frames:
        if x["state"] != "work":
            since = None; continue
        if gaze.in_bands(x, bands):
            since, last_in = since or x["t"], x["t"]
            best = max(best, x["t"] - since)
        elif x["t"] - last_in > G.get("gap_s", 0.4):
            since = None
    return best


def score_triggers(times, looks):
    """-> (looks caught, false triggers): a trigger inside a look (+0.5 s) catches it; outside every look +- GUARD_S
    it is false."""
    caught = {i for t in times for i, (a, b) in enumerate(looks) if a <= t <= b + 0.5}
    false = [t for t in times if not any(a - GUARD_S <= t <= b + GUARD_S for a, b in looks)]
    return len(caught), false


def report_natural(rows):
    looks = looks_of(rows)
    frames = label(rows, looks)
    dur = [b - a for a, b in looks]
    pos = [x for x in frames if x["state"] == "skull"]
    work = [x for x in frames if x["state"] == "work"]
    print(f"\nlooks marked {len(looks)}: duration median {np.median(dur):.1f} s [{min(dur):.1f}..{max(dur):.1f}]; "
          f"frames skull {len(pos)} (eyes {sum('iris_h' in x for x in pos)}), work {len(work)} "
          f"(eyes {sum('iris_h' in x for x in work)}), edge {sum(x['state'] == 'edge' for x in frames)}")
    print("\nfeature | skull: median p10 .. p90, spread | work: median p10 .. p90, spread | gap / spread")
    gs = {}
    for k in ("iris_h", "iris_v", "nose", "yaw", "pitch", "gaze_h"):
        s, w = [x[k] for x in pos if k in x], [x[k] for x in work if k in x]
        if not s or not w:
            continue
        ms_, s10, s90, ss = stats(s); mw, w10, w90, sw = stats(w)
        gs[k] = abs(ms_ - mw) / max(ss, sw, 1e-9)
        print(f"{k:<7} | {ms_:+8.3f} {s10:+8.3f} .. {s90:+8.3f}, {ss:6.3f} | {mw:+8.3f} {w10:+8.3f} .. {w90:+8.3f}, {sw:6.3f} | "
              f"{gs[k]:4.1f}{' ok' if gs[k] >= 2 else ''}")
    print(f"\nbands = [p5, p95] of the marked-look frames; dwell {G['dwell_s']} s, gap {G.get('gap_s', 0.4)} s, "
          f"min interval {G['min_interval_s']} s")
    print("set            | bands                                   | gap/spread  | skull in | work in | false | longest work run | caught")
    out = {}
    for combo in SETS:
        if not all(any(k in x for x in pos) for k in combo):
            continue
        bands = {k: (float(np.percentile([x[k] for x in pos if k in x], 5)),
                     float(np.percentile([x[k] for x in pos if k in x], 95))) for k in combo}
        s_in = sum(gaze.in_bands(x, bands) for x in pos) / len(pos)
        w_in = sum(gaze.in_bands(x, bands) for x in work) / max(len(work), 1)
        run = longest_work_run(frames, bands)
        caught, false = score_triggers([x["t"] for x in replay(frames, bands)], looks)
        out[combo] = (bands, run)
        print(f"{'+'.join(combo):<14} | {' '.join(f'{k} [{lo:+.2f},{hi:+.2f}]' for k, (lo, hi) in bands.items()):<39} | "
              f"{' '.join(f'{gs[k]:.1f}' for k in combo):<11} | {s_in:8.2f} | {w_in:7.2f} | {len(false):5d} | "
              f"{run:14.1f} s | {caught} of {len(looks)}")
    print("\ndwell -> looks caught / false / margin (dwell / longest work run; >= 2 wanted)")
    for combo, (bands, run) in out.items():
        cells = []
        need = max(0.25, math.ceil(2 * run / 0.25) * 0.25)      # the margin rule: dwell >= 2 x the longest work run
        for d in sorted(set(DWELLS) | {need}):
            caught, false = score_triggers([x["t"] for x in replay(frames, bands, d)], looks)
            cells.append(f"{d:.2f} s{' (2x run)' if d == need else ''}: {caught}/{len(looks)}, {len(false)} false, "
                         f"x{d / max(run, 1e-9):.1f}")
        print(f"{'+'.join(combo):<14} | " + " | ".join(cells))


def record_marks(seconds):
    marks = Marks(seconds)
    print(f"\n=== Отметки взглядов, {seconds / 60:.0f} мин: пока смотришь на череп — держи правый Ctrl. ===", flush=True)
    t0 = time.time()
    while time.time() < t0 + seconds:
        time.sleep(1)
    rows = marks.close() + [{"phase": "session", "t": round(t0, 3)}, {"phase": "session", "t": round(time.time(), 3)}]
    OUT.mkdir(exist_ok=True)
    out = OUT / f"gaze_1b_{time.strftime('%Y-%m-%d_%H%M%S')}_marks.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["phase", "look", "t"], extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"{marks.n} looks -> {out}")
    gate(load([out]))


def gate(rows):
    """Triggers the service published (events:vision) during the marks session against the marked looks."""
    looks = looks_of(rows)
    t0, t1 = min(x["t"] for x in rows), max(x["t"] for x in rows)
    ev = redis.Redis().xrange("events:vision", min=int(t0 * 1000), max=int(t1 * 1000))
    times = [float(f[b"ts"]) for _, f in ev if f.get(b"kind") == b"gaze"]
    caught, false = score_triggers(times, looks)
    first = looks[0][0] if looks else t1
    print(f"session {(t1 - t0) / 60:.1f} min; before the first look ({(first - t0) / 60:.1f} min): "
          f"{sum(t < first - GUARD_S for t in times)} triggers")
    print(f"looks marked {len(looks)}, caught {caught}; false triggers {len(false)}"
          + (": " + ", ".join(time.strftime('%H:%M:%S', time.localtime(t)) for t in false) if false else ""))
    for i, (a, b) in enumerate(looks, 1):
        hit = [t for t in times if a <= t <= b + 0.5]
        print(f"  look {i}: {time.strftime('%H:%M:%S', time.localtime(a))} {b - a:.1f} s -> "
              + (f"trigger +{hit[0] - a:.1f} s" if hit else "missed"))


if __name__ == "__main__":
    main()
