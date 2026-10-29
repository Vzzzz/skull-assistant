"""vision/gaze: «looking at the skull» as a trigger (T-031, D-029: UI element of the ambient mode — addressing without
a word and without the name).

Camera at the skull's eyes -> YuNet (OpenCV FaceDetectorYN, CPU) at `fps` -> the largest face, its 5 landmarks ->
solvePnP against a generic 3D face -> yaw / pitch (logged), and the nose offset between the eyes (nose x minus eye
midpoint x, in eye distances: 0 — the face turned straight at the camera). «Looking at the skull» is a calibrated band of
the nose offset, not «facing the camera»: with the skull beside the monitor the owner turns the eyes, not the head
(stand 2026-10-29: skull -0.30 [-0.38..-0.21], monitor -1.02 [-1.28..-0.61], yaw -45° against -64° — research/vision/
gaze_calib.py). solvePnP on 5 nearly coplanar points also jumps between mirrored solutions frame to frame.
In the band without a break for dwell_s -> events:vision {kind: gaze, dur_s, nose, yaw, pitch, score, ts} once; the next event needs the face to
turn away (or vanish) first and min_interval_s since the last one. tts:speaking mutes it (the owner looks at the skull
while it answers). Frames are not stored.

  .venv/bin/python services/vision/gaze.py [--show]     # --show: per-frame yaw / pitch / facing in the log
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import redis
import yaml

ROOT = Path(__file__).resolve().parents[2]
P = yaml.safe_load(Path(__file__).with_name("manifest.yaml").read_text(encoding="utf-8"))["params"]
G = P["gaze"]
r = redis.Redis()

# generic face, model units (x right, y up, z out of the face), the 5 YuNet landmarks in YuNet's order:
# right eye, left eye (the subject's), nose tip, right / left mouth corner
FACE_3D = np.array([(-165.0, 170.0, -135.0), (165.0, 170.0, -135.0), (0.0, 0.0, 0.0),
                    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])


def head_pose(lm, w, h):
    """(yaw, pitch) in degrees from 5 landmarks; 0/0 — facing the camera. Pinhole camera with f = image width."""
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(FACE_3D, lm.astype(np.float64), cam, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    # angles of the face normal (model +z, out of the face) in the camera frame — no Euler ambiguity: 0/0 when the
    # normal points straight at the camera (camera z looks forward, so then n = (0, 0, -1))
    n = cv2.Rodrigues(rvec)[0] @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.atan2(n[0], -n[2])), math.degrees(math.atan2(-n[1], -n[2]))


def nose_offset(lm):
    """Nose tip x minus the eye midpoint x, in eye distances (YuNet landmarks: right eye, left eye, nose, ...)."""
    return float((lm[2][0] - (lm[0][0] + lm[1][0]) / 2) / max(np.linalg.norm(lm[0] - lm[1]), 1.0))


def main():
    cv2.setNumThreads(P.get("threads", 1))   # OpenCV's default thread pool spins: 71 % of a core -> ~20 % with one thread
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    cap = cv2.VideoCapture(P.get("device", 0), cv2.CAP_V4L2)
    if P.get("fourcc"):     # MJPG: raw YUYV 640x480@30 (~18 MB/s) arrives broken over usbip (green frames, T-031 step 0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*P["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, P.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, P.get("height", 480))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"vision: no frame from /dev/video{P.get('device', 0)}")
    h, w = frame.shape[:2]
    det = cv2.FaceDetectorYN.create(str(ROOT / G["model"]), "", (w, h), G.get("score_min", 0.8), 0.3, 5000)
    period = 1.0 / G.get("fps", 5)
    since, armed, last_event, last_in = None, True, 0.0, 0.0
    gap_s = G.get("gap_s", 0.4)     # a look flickers out of the band for a frame or two (live 2026-10-29): not a break
    print(f"vision: gaze on {w}x{h} at {G.get('fps', 5)} fps, looking at the skull = nose offset in [{G['nose_min']}, {G['nose_max']}], "
          f"dwell {G['dwell_s']} s", flush=True)
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.5); continue
        _, faces = det.detect(frame)
        pose = nose = None
        if faces is not None and len(faces):
            f = max(faces, key=lambda x: x[2] * x[3])                    # the largest face: the nearest person
            lm = f[4:14].reshape(5, 2)
            pose = head_pose(lm, w, h)
            nose = nose_offset(lm)
        facing = nose is not None and G["nose_min"] <= nose <= G["nose_max"]
        now = time.time()
        if a.show:
            print(f"  {'face' if pose else '----'} " + (f"nose {nose:+.2f} yaw {pose[0]:+5.1f} pitch {pose[1]:+5.1f} score {f[14]:.2f}" if pose else "")
                  + (" FACING" if facing else "") + f"  {1000 * (now - t0):.0f} ms", flush=True)
        if not facing:
            if now - last_in > gap_s:
                since, armed = None, True
        else:
            since, last_in = since or now, now
            dur = now - since
            if armed and dur >= G["dwell_s"] and now - last_event >= G["min_interval_s"] and not r.exists("tts:speaking"):
                r.xadd("events:vision", {"kind": "gaze", "dur_s": f"{dur:.1f}", "nose": f"{nose:.2f}", "yaw": f"{pose[0]:.1f}",
                                         "pitch": f"{pose[1]:.1f}", "score": f"{f[14]:.2f}", "ts": f"{now:.3f}"},
                       maxlen=1000, approximate=True)
                print(f"GAZE {dur:.1f} s nose {nose:+.2f} yaw {pose[0]:+.1f} pitch {pose[1]:+.1f}", flush=True)
                armed, last_event = False, now
        time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
