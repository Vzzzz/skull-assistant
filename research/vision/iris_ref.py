"""Iris features of services/vision/gaze.py (YuNet box -> own crop -> Face Mesh ONNX) against MediaPipe's own pipeline
(FaceLandmarker task: its detector, its ROI, its tflite) on the same live frames (T-031 step 1b). Frames are not stored.

MediaPipe stays out of the prod .venv (the wheel imports matplotlib, wants opencv-contrib) — a separate venv:
  python3 -m venv ~/venvs/skull-mediapipe-ref && ~/venvs/skull-mediapipe-ref/bin/pip install --no-deps mediapipe && \
  ~/venvs/skull-mediapipe-ref/bin/pip install absl-py certifi flatbuffers matplotlib numpy==2.2.6 \
      opencv-python-headless==5.0.0.93 onnxruntime redis pyyaml
  ~/venvs/skull-mediapipe-ref/bin/python research/vision/iris_ref.py --task <face_landmarker.task> [--seconds 20]
"""
import argparse
import importlib.util
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gaze", ROOT / "services/vision/gaze.py")
gaze = importlib.util.module_from_spec(spec); spec.loader.exec_module(gaze)
spec = importlib.util.spec_from_file_location("calib", ROOT / "research/vision/gaze_calib.py")
calib = importlib.util.module_from_spec(spec); spec.loader.exec_module(calib)


def mp_features(res, w, h):
    """The same eye features from MediaPipe's landmarks (normalised -> pixels; no crop rotation, so the eye axis is
    taken per eye exactly as in gaze.iris_features)."""
    if not res.face_landmarks:
        return None
    pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]])
    feats = []
    for a, b, up, lo, c in gaze.EYES:
        a, b, up, lo, c = pts[a], pts[b], pts[up], pts[lo], pts[c]
        width = float(np.linalg.norm(b - a)); u = (b - a) / width; n = np.array([-u[1], u[0]])
        feats.append(((c - a) @ u / width, (c - (a + b) / 2) @ n / width))
    return np.mean(feats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()
    cap = calib.open_camera()
    ok, fr = cap.read()
    h, w = fr.shape[:2]
    det = cv2.FaceDetectorYN.create(str(ROOT / gaze.G["model"]), "", (w, h), gaze.G["score_min"], 0.3, 5000)
    sess = gaze.mesh_session(ROOT / gaze.G["mesh_model"])
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=a.task), running_mode=vision.RunningMode.IMAGE, num_faces=1))
    ours, ref, t_end = [], [], time.time() + a.seconds
    while time.time() < t_end:
        ok, fr = cap.read()
        if not ok:
            continue
        m = gaze.measure(det, sess, fr, w, h)
        f = mp_features(fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))), w, h)
        if "iris_h" in m and f is not None:
            ours.append((m["iris_h"], m["iris_v"])); ref.append(f)
        time.sleep(0.15)
    ours, ref = np.array(ours), np.array(ref)
    print(f"{len(ours)} frames with both")
    for i, k in enumerate(("iris_h", "iris_v")):
        d = ours[:, i] - ref[:, i]
        print(f"{k}: ours {np.median(ours[:, i]):+.3f} [{ours[:, i].min():+.3f}..{ours[:, i].max():+.3f}], "
              f"mediapipe {np.median(ref[:, i]):+.3f}; diff median {np.median(d):+.3f}, |diff| p90 "
              f"{np.percentile(np.abs(d), 90):.3f}, corr {np.corrcoef(ours[:, i], ref[:, i])[0, 1]:.2f}")


if __name__ == "__main__":
    main()
