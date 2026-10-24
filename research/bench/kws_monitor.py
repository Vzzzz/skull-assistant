"""Passive wake-word monitor: why does "hey jarvis" need 2-4 attempts? (2026-10-24)

Reads audio:mic alongside the running skull-wakeword (own model instance, same frames, nothing is published),
splits the stream into speech segments by energy and prints for each: duration, mean RMS, the peak wake score
and when it came relative to the segment end, and whether skull-wakeword fired (events:wakeword) or was muted
by tts:speaking. Misses with peaks 0.2-0.5 point at the threshold, peaks near 0 at the model or the level.

  .venv/bin/python research/bench/kws_monitor.py --csv research/bench/results/kws_monitor_<date>.csv
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import redis
from openwakeword.model import Model

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models/wakeword"
FRAME_S = 480 / 16000
SPEECH_RMS, GAP_FRAMES, MIN_FRAMES = 300, 10, 8   # stt's silence threshold; 0.3 s gap ends a segment; 0.24 s minimum
TAIL_FRAMES = 15                                   # keep scoring 0.45 s after the segment: the score lags the word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    model = Model(wakeword_models=[str(MODELS / "hey_jarvis_v0.1.onnx")],
                  melspec_model_path=str(MODELS / "melspectrogram.onnx"),
                  embedding_model_path=str(MODELS / "embedding_model.onnx"), inference_framework="onnx")
    r = redis.Redis(socket_timeout=None)
    out = None
    if args.csv:
        f = open(args.csv, "a", newline="")
        out = csv.writer(f)
        if f.tell() == 0:
            out.writerow(["t", "dur_s", "rms_mean", "peak", "peak_after_end_s", "fired", "muted"])
    last_id, wake_last = "$", (r.xrevrange("events:wakeword", count=1) or [[b"0"]])[0][0]
    seg, gap, tail = None, 0, 0
    print("kws_monitor: listening; say 'hey jarvis' with pauses. Ctrl-C to stop.", flush=True)
    while True:
        resp = r.xread({"audio:mic": last_id}, count=10, block=1000)
        if not resp:
            continue
        for msg_id, fields in resp[0][1]:
            last_id = msg_id
            pcm = np.frombuffer(fields[b"pcm"], dtype=np.int16)
            score = float(next(iter(model.predict(pcm).values())))
            rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
            if seg is None:
                if rms >= SPEECH_RMS:
                    seg = {"t0": time.time(), "n": 0, "rms": [], "peak": score, "peak_i": 0, "end_i": None,
                           "muted": bool(r.exists("tts:speaking"))}
                    gap = tail = 0
                else:
                    continue
            i = seg["n"]
            seg["n"] += 1
            if score > seg["peak"]:
                seg["peak"], seg["peak_i"] = score, i
            if seg["end_i"] is None:
                if rms >= SPEECH_RMS:
                    seg["rms"].append(rms); gap = 0
                else:
                    gap += 1
                    if gap >= GAP_FRAMES:
                        seg["end_i"] = i - gap + 1
                continue
            tail += 1
            if tail < TAIL_FRAMES:
                continue
            if len(seg["rms"]) >= MIN_FRAMES:
                new = r.xread({"events:wakeword": wake_last}, count=10)   # no block: returns at once
                fired = bool(new)
                if new:
                    wake_last = new[0][1][-1][0]
                row = [time.strftime("%H:%M:%S", time.localtime(seg["t0"])), round(seg["end_i"] * FRAME_S, 2),
                       int(np.mean(seg["rms"])), round(seg["peak"], 3),
                       round((seg["peak_i"] - seg["end_i"]) * FRAME_S, 2), int(fired), int(seg["muted"])]
                print(f"{row[0]}  {row[1]:4.2f}s  rms {row[2]:5d}  peak {row[3]:.3f}  at end{row[4]:+.2f}s  "
                      f"{'WAKE' if fired else '----'}{'  (tts muted)' if seg['muted'] else ''}", flush=True)
                if out:
                    out.writerow(row); f.flush()
            seg = None


if __name__ == "__main__":
    main()
