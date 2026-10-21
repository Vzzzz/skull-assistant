"""YOLO memory/latency probe for the stack bench (docs/TASKS.md T-033, M7-lite T-031).

Runs a YOLO ONNX model (default: yolo11n 640, models/vision/yolo11n_640.onnx) in a loop on a synthetic frame
at a target fps and prints ms/frame and RSS. No OpenCV/PIL: input is a random uint8 frame normalised in numpy.
Kept alive until killed (so the bench can sample its RSS), or --frames N for a bounded run.

Usage:
    .venv/bin/python research/vision/yolo_probe.py --model models/vision/yolo11n_640.onnx --fps 5 --frames 50
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) // 1024
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/vision/yolo11n_640.onnx")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--frames", type=int, default=0, help="0 = run until killed")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    avail = ort.get_available_providers()
    if args.provider == "cuda" or (args.provider == "auto" and "CUDAExecutionProvider" in avail):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    t0 = time.perf_counter()
    sess = ort.InferenceSession(args.model, so, providers=providers)
    load_s = time.perf_counter() - t0
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]  # e.g. [1,3,640,640]
    h, w = shape[2], shape[3]
    print(f"model={os.path.basename(args.model)} providers={sess.get_providers()} input={shape} load={load_s:.2f}s rss={rss_mb()}MB", flush=True)

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    period = 1.0 / args.fps if args.fps > 0 else 0
    times = []
    n = 0
    while args.frames == 0 or n < args.frames:
        t1 = time.perf_counter()
        x = frame.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = sess.run(None, {inp.name: x})
        dt = time.perf_counter() - t1
        times.append(dt)
        n += 1
        if n % 25 == 0 or n == args.frames:
            arr = np.array(times[-25:]) * 1000
            print(f"frames={n} ms/frame median={np.median(arr):.1f} p95={np.percentile(arr, 95):.1f} rss={rss_mb()}MB out={[o.shape for o in out]}", flush=True)
        if period:
            time.sleep(max(0.0, period - dt))


if __name__ == "__main__":
    main()
