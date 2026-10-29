"""Camera on the stand (T-031 step 0): is there a frame, at what resolution, how often and how fast.

  .venv/bin/python research/vision/cam_probe.py [--dev 0] [--seconds 10] [--mjpg]

Opens /dev/video<dev> through V4L2 at 640x480, reads frames for --seconds and prints: the negotiated size and FOURCC,
frames per second, read() time per frame (median / p95 / max) and the age of the frame at read time when the driver
gives a timestamp (CAP_PROP_POS_MSEC, monotonic clock). Glass-to-glass latency needs a visible clock in the frame —
not measured here. Saves the last frame to research/vision/results/cam_probe_last.jpg (not in git: the room).
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--mjpg", action="store_true", help="ask for MJPG (USB 2 cameras give 30 fps at 640x480 only so)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(a.dev, cv2.CAP_V4L2)
    if a.mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"/dev/video{a.dev}: not opened")
    ok, frame = cap.read()
    open_s = time.perf_counter() - t0
    if not ok:
        raise SystemExit(f"/dev/video{a.dev}: opened, no frame")
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    print(f"/dev/video{a.dev}: {frame.shape[1]}x{frame.shape[0]} {fourcc}, first frame after {open_s:.2f} s, "
          f"driver fps {cap.get(cv2.CAP_PROP_FPS):.0f}")
    reads, ages, n, t_end = [], [], 0, time.perf_counter() + a.seconds
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        t1 = time.perf_counter()
        ok, frame = cap.read()
        reads.append(time.perf_counter() - t1)
        if not ok:
            print("  read failed"); break
        n += 1
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ts > 0:
            ages.append(time.clock_gettime(time.CLOCK_MONOTONIC) * 1000 - ts)
    dur = time.perf_counter() - t_start
    r = np.array(reads) * 1000
    print(f"{n} frames in {dur:.1f} s -> {n / dur:.1f} fps; read() ms median {np.median(r):.1f} p95 {np.percentile(r, 95):.1f} max {r.max():.1f}")
    if ages:
        g = np.array(ages)
        print(f"frame age at read (driver timestamp, monotonic) ms median {np.median(g):.0f} p95 {np.percentile(g, 95):.0f}")
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / "cam_probe_last.jpg"), frame)
    print(f"last frame -> {OUT / 'cam_probe_last.jpg'}; brightness mean {frame.mean():.0f}")
    cap.release()


if __name__ == "__main__":
    main()
