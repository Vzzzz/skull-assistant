"""Voice ID data and enrollment (T-039, D-029). Audio stays in models/voice/ (not in git: it is the owner's voice).

  record  --label enroll_A --seconds 30   3-s segments with speech from audio:mic -> models/voice/data/<label>/NNN.wav
  enroll  --label enroll_A [--model M]    mean L2-normalised embedding of the label's segments -> models/voice/owner.npy

The mic path is the same as in production (skull-capture -> audio:mic); stop skull-wakeword/stt/brain/tts while
recording so nothing answers. A segment is 3 s of voiced frames (pauses skipped).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import redis
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
VOICE = ROOT / "models/voice"
DATA = VOICE / "data"
RATE, FRAME, SEG_FRAMES = 16000, 480, 100          # 100 x 30 ms = 3 s
DEFAULT_MODEL = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"


def extractor(model):
    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(VOICE / model), num_threads=1)
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def embed(ex, audio):
    st = ex.create_stream()
    st.accept_waveform(RATE, audio)
    st.input_finished()
    e = np.asarray(ex.compute(st), dtype=np.float32)
    return e / max(np.linalg.norm(e), 1e-8)


def record(label, seconds, voiced_rms, hangover=3):
    """Collects voiced 30-ms frames (RMS >= voiced_rms, plus `hangover` frames after each voiced one so word ends are
    not cut) until 3 s of speech, then writes a segment: pauses do not count, natural speech with gaps still fills it."""
    out = DATA / label
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob("*.wav")))
    r = redis.Redis(socket_timeout=None)
    last_id, frames, kept, since_voiced = "$", [], 0, hangover + 1
    t_end = time.time() + seconds
    print(f"record {label}: {seconds} s, speak now", flush=True)
    while time.time() < t_end:
        resp = r.xread({"audio:mic": last_id}, count=20, block=1000)
        if not resp:
            continue
        for msg_id, fields in resp[0][1]:
            last_id = msg_id
            pcm = np.frombuffer(fields[b"pcm"], dtype=np.int16)
            since_voiced = 0 if np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) >= voiced_rms else since_voiced + 1
            if since_voiced > hangover:
                continue
            frames.append(pcm)
            if len(frames) >= SEG_FRAMES:
                audio = np.concatenate(frames).astype(np.float32) / 32768.0
                sf.write(out / f"{n:03d}.wav", audio, RATE)
                print(f"  {label}/{n:03d}.wav  rms {int(np.sqrt(np.mean(audio ** 2)) * 32768)}", flush=True)
                n += 1; kept += 1; frames = []
    print(f"record {label}: kept {kept} segments ({n} in {out.relative_to(ROOT)})", flush=True)


def owner_vector(ex, label):
    files = sorted((DATA / label).glob("*.wav"))
    if not files:
        raise SystemExit(f"no segments in {DATA / label}")
    v = np.mean([embed(ex, sf.read(f, dtype="float32")[0]) for f in files], axis=0)
    return v / np.linalg.norm(v), len(files)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--label", required=True)
    rec.add_argument("--seconds", type=float, default=30)
    rec.add_argument("--voiced-rms", type=float, default=200, help="frame RMS (int16) counted as voiced")
    en = sub.add_parser("enroll")
    en.add_argument("--label", default="enroll_A")
    en.add_argument("--model", default=DEFAULT_MODEL)
    en.add_argument("--out", default=str(VOICE / "owner.npy"))
    a = ap.parse_args()
    if a.cmd == "record":
        record(a.label, a.seconds, a.voiced_rms)
    else:
        v, n = owner_vector(extractor(a.model), a.label)
        np.save(a.out, v)
        print(f"enroll: {n} segments of {a.label} with {a.model} -> {a.out} (dim {v.size})")


if __name__ == "__main__":
    sys.exit(main())
