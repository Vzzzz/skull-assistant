"""vision/gaze: «looking at the skull» as a trigger (T-031, D-029: UI element of the ambient mode — addressing without
a word and without the name).

Camera at the skull's eyes -> YuNet (OpenCV FaceDetectorYN, CPU) at `fps` -> the largest face, its 5 landmarks ->
  * head: solvePnP against a generic 3D face -> yaw / pitch, and the nose offset between the eyes (nose x minus eye
    midpoint x, in eye distances: 0 — the face turned straight at the camera);
  * eyes (step 1b): MediaPipe Face Mesh with irises (478 points, exported to ONNX — research/vision/export_face_mesh.sh)
    on a 256x256 crop around the YuNet box, rotated so the eyes are level -> iris centre against the eye corners
    (iris_h: 0 — at the image-left corner, 1 — at the image-right one) and below the corner line (iris_v, in eye widths),
    averaged over both eyes; eye openness drops blinks.
«Looking at the skull» is a calibrated band per feature (`gaze.bands`, all must hold), not «facing the camera»: with the
skull beside the monitor the owner turns the eyes, not the head. Step 1 used the nose offset alone; the working posture
(reading, the messenger on the right) lay inside its band (live gate 2026-10-29: 31 false triggers in 10 min) — hence the
irises (research/vision/gaze_calib.py --work). In the bands for dwell_s (dropouts <= gap_s allowed) -> events:vision
{kind: gaze, dur_s, iris_h, iris_v, nose, yaw, pitch, score, ts} once; the next event needs the gaze to leave the bands
(or the face to vanish) first and min_interval_s since the last one. tts:speaking mutes it (the owner looks at the skull
while it answers). Frames are not stored.

  .venv/bin/python services/vision/gaze.py [--show]     # --show: per-frame features in the log
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
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

# Face Mesh indices per eye, in image order: (image-left corner, image-right corner, upper lid, lower lid, iris centre).
# The subject's right eye is on the image left: outer corner 33, inner 133; the left eye: inner 362, outer 263.
EYES = ((33, 133, 159, 145, 468), (362, 263, 386, 374, 473))
MESH_SIZE = 256
MESH_SCALE = 1.5        # crop side = 1.5 x the larger side of the face box (MediaPipe's detection-to-ROI scale)


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


def mesh_session(path, threads=1):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def mesh_crop(frame, box, lm):
    """256x256 RGB float crop around the YuNet box, rotated so the eye line is level (as MediaPipe's ROI)."""
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    side = max(bw, bh) * MESH_SCALE
    ang = math.degrees(math.atan2(lm[1][1] - lm[0][1], lm[1][0] - lm[0][0]))
    m = cv2.getRotationMatrix2D((cx, cy), ang, MESH_SIZE / side)
    m[:, 2] += (MESH_SIZE / 2 - cx, MESH_SIZE / 2 - cy)
    crop = cv2.warpAffine(frame, m, (MESH_SIZE, MESH_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0


def iris_features(sess, frame, box, lm):
    """{iris_h, iris_v, iris_vl, open, presence} over both eyes, or None when the mesh does not see a face.
    iris_h — iris centre along the corner-to-corner line (0 image-left corner, 1 image-right one); iris_v — below the
    corner line, in eye widths; iris_vl — between the lids (0 upper, 1 lower); open — lid gap in eye widths."""
    out = sess.run(None, {sess.get_inputs()[0].name: mesh_crop(frame, box, lm)})
    presence = 1.0 / (1.0 + math.exp(-float(np.ravel(out[1])[0])))
    if presence < G.get("presence_min", 0.5):
        return None
    pts = out[0].reshape(-1, 3)[:, :2]
    feats = []
    for a, b, up, lo, c in EYES:
        a, b, up, lo, c = pts[a], pts[b], pts[up], pts[lo], pts[c]
        width = max(float(np.linalg.norm(b - a)), 1e-3)
        u = (b - a) / width
        n = np.array([-u[1], u[0]])          # image y points down: n is «down» across the eye
        gap = float((lo - up) @ n)
        feats.append((float((c - a) @ u) / width, float((c - (a + b) / 2) @ n) / width,
                      float((c - up) @ n) / max(gap, 1e-3), gap / width))
    f = np.mean(feats, axis=0)
    return {"iris_h": float(f[0]), "iris_v": float(f[1]), "iris_vl": float(f[2]), "open": float(f[3]),
            "presence": presence}


def measure(det, sess, frame, w, h):
    """Per-frame features of the largest face: {} without a face."""
    _, faces = det.detect(frame)
    if faces is None or not len(faces):
        return {}
    f = max(faces, key=lambda x: x[2] * x[3])                    # the largest face: the nearest person
    lm = f[4:14].reshape(5, 2)
    m = {"score": float(f[14]), "nose": nose_offset(lm)}
    pose = head_pose(lm, w, h)
    if pose:
        m["yaw"], m["pitch"] = pose
    if sess is not None:
        eyes = iris_features(sess, frame, f[:4], lm)
        if eyes and eyes["open"] >= G.get("open_min", 0.0):    # a blink: no eye feature this frame
            m.update(eyes)
    return m


def in_bands(m, bands):
    return all(k in m and lo <= m[k] <= hi for k, (lo, hi) in bands.items())


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
    sess = mesh_session(ROOT / G["mesh_model"], P.get("threads", 1)) if G.get("mesh_model") else None
    bands = {k: tuple(v) for k, v in G["bands"].items()}
    period = 1.0 / G.get("fps", 5)
    since, armed, last_event, last_in = None, True, 0.0, 0.0
    gap_s = G.get("gap_s", 0.4)     # a look flickers out of the band for a frame or two (live 2026-10-29): not a break
    print(f"vision: gaze on {w}x{h} at {G.get('fps', 5)} fps, looking at the skull = {bands}, dwell {G['dwell_s']} s",
          flush=True)
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.5); continue
        m = measure(det, sess, frame, w, h)
        facing = in_bands(m, bands)
        now = time.time()
        if a.show:
            print("  " + (" ".join(f"{k} {v:+.2f}" for k, v in m.items()) if m else "----")
                  + (" FACING" if facing else "") + f"  {1000 * (now - t0):.0f} ms", flush=True)
        if not facing:
            if now - last_in > gap_s:
                since, armed = None, True
        else:
            since, last_in = since or now, now
            dur = now - since
            if armed and dur >= G["dwell_s"] and now - last_event >= G["min_interval_s"] and not r.exists("tts:speaking"):
                ev = {k: f"{m[k]:.2f}" for k in ("iris_h", "iris_v", "nose", "score") if k in m}
                ev.update({k: f"{m[k]:.1f}" for k in ("yaw", "pitch") if k in m})
                r.xadd("events:vision", {"kind": "gaze", "dur_s": f"{dur:.1f}", **ev, "ts": f"{now:.3f}"},
                       maxlen=1000, approximate=True)
                print(f"GAZE {dur:.1f} s " + " ".join(f"{k} {v}" for k, v in ev.items()), flush=True)
                armed, last_event = False, now
        time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
