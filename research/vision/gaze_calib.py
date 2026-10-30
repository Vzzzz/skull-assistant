"""Gaze calibration on the stand (T-031 steps 1 / 1b): does «looking at the skull» separate from the rest?

Frames are read from the camera at the service rate (services/vision manifest) and measured exactly as the service does
(services/vision/gaze.measure: YuNet nose offset, yaw / pitch; Face Mesh iris_h / iris_v / iris_vl / open); every frame's
features are logged with its phase. Frames are not stored. CSV -> research/vision/results/ (numbers only).

  .venv/bin/python research/vision/gaze_calib.py --work 300 --looks 10    # step 1b, the owner runs it in a terminal:
        # 5 min of ordinary work (no looks at the skull), then 10 looks on the console's command «СМОТРИ НА ЧЕРЕП»
  .venv/bin/python research/vision/gaze_calib.py --static skull --seconds 15     # one state, said in the chat
  .venv/bin/python research/vision/gaze_calib.py --report <csv> [<csv> ...]       # over saved recordings

Report: per feature, «skull» frames against everything else (work, between the looks, monitor): medians, p10 / p90,
robust spread (1.4826 x MAD) and the gap between the medians; criterion (core, 2026-10-29): gap >= 2 x the larger
spread. Bands are fixed in advance — [p5, p95] of the «skull» frames — for single features and their combinations; per
band: the share of frames inside it, and the service's dwell logic replayed over the log (triggers during work, looks
caught out of N).
"""
import argparse
import csv
import importlib.util
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gaze", ROOT / "services/vision/gaze.py")
gaze = importlib.util.module_from_spec(spec); spec.loader.exec_module(gaze)   # measure() etc., no service loop
P, G = gaze.P, gaze.G
OUT = Path(__file__).resolve().parent / "results"
KEYS = ["phase", "look", "t", "settled", "face", "score", "nose", "yaw", "pitch", "iris_h", "iris_v", "iris_vl", "open",
        "presence", "ms"]
FEATURES = ["iris_h", "iris_v", "iris_vl", "nose", "yaw", "pitch"]
COMBOS = [("iris_h", "iris_v"), ("iris_h", "iris_vl"), ("iris_h", "nose"), ("iris_h", "iris_v", "nose")]
LOOK_S, SETTLE_S = 4.0, 1.2           # a look on command: held 4 s, the first 1.2 s the eyes are still moving


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
    ap.add_argument("--report", nargs="+", help="CSV files of earlier recordings")
    a = ap.parse_args()
    if a.report:
        return report(load(a.report))
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
    ms = [x["ms"] for x in rows]
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
            x["settled"], x["face"], x["look"] = int(x["settled"]), int(x["face"]), int(x.get("look", x.get("cycle", 0)))
            rows.append(x)
    return sorted(rows, key=lambda x: x["t"])


def stats(v):
    v = np.asarray(v)
    med = float(np.median(v))
    return med, float(np.percentile(v, 10)), float(np.percentile(v, 90)), 1.4826 * float(np.median(np.abs(v - med)))


def replay(rows, bands):
    """The service's dwell logic over the log -> trigger times."""
    since, armed, last_event, last_in, out = None, True, -1e9, -1e9, []
    for x in rows:
        now = x["t"]
        if not gaze.in_bands(x, bands):
            if now - last_in > G.get("gap_s", 0.4):
                since, armed = None, True
        else:
            since, last_in = since or now, now
            if armed and now - since >= G["dwell_s"] and now - last_event >= G["min_interval_s"]:
                out.append(x); armed, last_event = False, now
    return out


def report(rows):
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


if __name__ == "__main__":
    main()
