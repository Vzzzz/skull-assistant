#!/usr/bin/env bash
# MediaPipe Face Landmarker (478 points, irises 468..477) -> ONNX for services/vision (T-031 step 1b).
#
# The mediapipe wheel (1.2.0) imports matplotlib at import time and wants opencv-contrib (clashes with our headless
# cv2), so it stays out of the prod .venv: the prod service runs the exported landmark model with onnxruntime.
# The conversion needs TensorFlow (tf2onnx) — a throwaway venv outside the repo, no torch.
#
#   bash research/vision/export_face_mesh.sh     # -> models/vision/face_landmarks_478.onnx
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
V=${EXPORT_VENV:-$HOME/venvs/skull-face-mesh-export}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

[ -x "$V/bin/python" ] || python3 -m venv "$V"
"$V/bin/pip" install -q tensorflow-cpu tf2onnx onnxruntime

curl -sSL -o "$W/face_landmarker.task" \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extract('face_landmarks_detector.tflite', sys.argv[2])" \
  "$W/face_landmarker.task" "$W"

"$V/bin/python" -m tf2onnx.convert --tflite "$W/face_landmarks_detector.tflite" \
  --output "$REPO/models/vision/face_landmarks_478.onnx" --opset 17

# the export must match the tflite: landmarks within 1e-3 px on random input
"$V/bin/python" - "$W/face_landmarks_detector.tflite" "$REPO/models/vision/face_landmarks_478.onnx" <<'EOF'
import sys
import numpy as np, tensorflow as tf, onnxruntime as ort
it = tf.lite.Interpreter(model_path=sys.argv[1]); it.allocate_tensors()
s = ort.InferenceSession(sys.argv[2], providers=["CPUExecutionProvider"])
x = np.random.default_rng(0).random((1, 256, 256, 3), dtype=np.float32)
it.set_tensor(it.get_input_details()[0]["index"], x); it.invoke()
ref = it.get_tensor(it.get_output_details()[0]["index"]).reshape(-1)
out = s.run(None, {s.get_inputs()[0].name: x})[0].reshape(-1)
d = float(np.abs(ref - out).max())
print(f"landmarks: max abs diff tflite vs onnx {d:.5f} px")
sys.exit(0 if d < 1e-3 else 1)
EOF
sha256sum "$REPO/models/vision/face_landmarks_478.onnx"
