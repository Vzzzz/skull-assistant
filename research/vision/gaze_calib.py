"""Gaze calibration on the stand (T-031 step 1): does «looking at the skull» separate from «looking at the monitor»?

The skull itself gives the commands through TTS (events:reply): «Смотри на меня» / «Смотри в монитор», `--phase` s
each, `--cycles` times; meanwhile the camera (services/vision manifest) is read at its rate and every frame's
yaw / pitch (services/vision/gaze.head_pose) and the nose offset between the eyes are logged with the phase. The first
second of each phase is skipped (the head is turning). Prints per-phase distributions, the share of frames each
threshold pair calls «facing», and replays the dwell logic over the log. CSV -> research/vision/results/ (numbers only).

  .venv/bin/python research/vision/gaze_calib.py [--cycles 5] [--phase 4]
  .venv/bin/python research/vision/gaze_calib.py --static skull --seconds 15     # no TTS: one state, said in the chat
  .venv/bin/python research/vision/gaze_calib.py --report <csv> [<csv> ...]       # distributions over saved recordings
The WSLg sound server hangs on the stand (I-15) — the spoken commands then come late; --static does without them.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import redis
import yaml

ROOT = Path(__file__).resolve().parents[2]
ns = {"__file__": str(ROOT / "services/vision/gaze.py")}
exec((ROOT / "services/vision/gaze.py").read_text(encoding="utf-8").split("def main():")[0]
     .replace("r = redis.Redis()", "r = None"), ns)       # FACE_3D and head_pose without starting the service
head_pose = ns["head_pose"]
P = yaml.safe_load((ROOT / "services/vision/manifest.yaml").read_text(encoding="utf-8"))["params"]
OUT = Path(__file__).resolve().parent / "results"
SAY = {"skull": "Смотри на меня.", "monitor": "Смотри в монитор."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--phase", type=float, default=4.0)
    ap.add_argument("--static", choices=["skull", "monitor"], help="record one state for --seconds, without TTS")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--score-min", type=float, help="YuNet threshold (default: manifest)")
    ap.add_argument("--report", nargs="+", help="CSV files of earlier recordings")
    a = ap.parse_args()
    if a.report:
        rows = [dict(x) for f in a.report for x in csv.DictReader(open(f, encoding="utf-8"))]
        for x in rows:
            x["settled"], x["face"] = int(x["settled"]), int(x["face"])
            for k in ("yaw", "pitch", "geo"):
                x[k] = float(x[k]) if x.get(k) else None
        return report(rows)
    r = redis.Redis()
    cap = cv2.VideoCapture(P.get("device", 0), cv2.CAP_V4L2)
    if P.get("fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*P["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, P["width"]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, P["height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    det = cv2.FaceDetectorYN.create(str(ROOT / P["gaze"]["model"]), "", (P["width"], P["height"]),
                                    a.score_min or P["gaze"]["score_min"], 0.3, 5000)
    rows = []
    plan = [(0, a.static)] if a.static else [(c, ph) for c in range(a.cycles) for ph in ("skull", "monitor")]
    for c, phase in plan:
            if not a.static:
                r.xadd("events:reply", {"text": SAY[phase], "final": 1, "ts": f"{time.time():.3f}"})
            t_end = time.time() + (a.seconds if a.static else a.phase)
            t_from = time.time() + (0 if a.static else 2.0)   # speech ~1 s + turning the head ~1 s
            while time.time() < t_end:
                ok, fr = cap.read()
                if not ok:
                    continue
                now = time.time()
                _, faces = det.detect(fr)
                row = {"cycle": c, "phase": phase, "t": round(now, 3), "settled": int(now >= t_from), "face": 0}
                if faces is not None and len(faces):
                    f = max(faces, key=lambda x: x[2] * x[3]); lm = f[4:14].reshape(5, 2)
                    pose = head_pose(lm, P["width"], P["height"])
                    if pose:
                        geo = (lm[2][0] - (lm[0][0] + lm[1][0]) / 2) / max(np.linalg.norm(lm[0] - lm[1]), 1)
                        row.update(face=1, yaw=round(pose[0], 1), pitch=round(pose[1], 1), geo=round(float(geo), 3), score=round(float(f[14]), 2))
                rows.append(row)
                time.sleep(0.15)
    if not a.static:
        r.xadd("events:reply", {"text": "Всё.", "final": 1, "ts": f"{time.time():.3f}"})
    OUT.mkdir(exist_ok=True)
    out = OUT / f"gaze_calib_{time.strftime('%Y-%m-%d_%H%M%S')}{'_' + a.static if a.static else ''}.csv"
    keys = ["cycle", "phase", "t", "settled", "face", "yaw", "pitch", "geo", "score"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} frames -> {out}")
    report(rows)


def report(rows):
    for phase in ("skull", "monitor"):
        s = [x for x in rows if x["phase"] == phase and x["settled"]]
        f = [x for x in s if x["face"]]
        if not f:
            print(f"{phase}: {len(s)} frames, no face"); continue
        y, p, g = (np.array([x[k] for x in f]) for k in ("yaw", "pitch", "geo"))
        print(f"{phase:<8} frames {len(s)}, face {len(f)}; yaw median {np.median(y):+.0f} [p10 {np.percentile(y, 10):+.0f}, p90 {np.percentile(y, 90):+.0f}], "
              f"pitch median {np.median(p):+.0f} [p10 {np.percentile(p, 10):+.0f}, p90 {np.percentile(p, 90):+.0f}], "
              f"nose offset median {np.median(g):+.2f} [p10 {np.percentile(g, 10):+.2f}, p90 {np.percentile(g, 90):+.2f}]")
    print("share of settled frames called «facing» (skull / monitor), by |yaw|, |pitch| limits:")
    for ym in (10, 15, 20, 25, 30):
        for pm in (15, 20, 30):
            share = []
            for phase in ("skull", "monitor"):
                s = [x for x in rows if x["phase"] == phase and x["settled"]]
                share.append(sum(1 for x in s if x["face"] and abs(x["yaw"]) <= ym and abs(x["pitch"]) <= pm) / max(len(s), 1))
            print(f"  yaw <= {ym:2d}, pitch <= {pm:2d}: {share[0]:.2f} / {share[1]:.2f}")


if __name__ == "__main__":
    main()
