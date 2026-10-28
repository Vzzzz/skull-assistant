"""Voice id by sliding windows on a broadcast background (T-038 2в), offline — before the live check.

Negatives: the broadcast recordings (podcast from a phone through the stand mic, models/voice/data/broadcast) glued
into 12-s chunks — what the VAD detector sends on a continuous podcast. Positives: the owner's phrases
(research/stt/data/owner) laid over such a chunk at 4 s, podcast at gain 1.0 (both recorded through the same mic: the
real ratio) and 0.5. Scored by services/stt/voiceid.VoiceID with the production params (manifest of skull-stt).

  .venv/bin/python research/voice/bench_windows.py [--win 1.5]
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services/stt"))
from voiceid import VoiceID

CHUNK, AT = 12 * 16000, 4 * 16000


def frames(a):
    a = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    return [a[i:i + 480] for i in range(0, len(a) // 480 * 480, 480)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=1.5)
    ap.add_argument("--hop", type=float, default=0, help="window step, s (0 = win / 2)")
    a = ap.parse_args()
    params = yaml.safe_load((ROOT / "services/stt/manifest.yaml").read_text(encoding="utf-8"))["params"]["voice_id"]
    vid = VoiceID(ROOT, {**params, "win_s": a.win, "win_hop_s": a.hop or a.win / 2})
    pod = np.concatenate([sf.read(f, dtype="float32")[0] for f in sorted(glob.glob(str(ROOT / "models/voice/data/broadcast/*.wav")))])
    chunks = [pod[i:i + CHUNK] for i in range(0, len(pod) - CHUNK + 1, CHUNK)]
    print(f"win {a.win} s, threshold {vid.threshold} / high {vid.threshold_high}; {len(chunks)} podcast chunks of 12 s")
    neg = [vid(frames(c)) for c in chunks]
    for i, r in enumerate(neg):
        print(f"  neg {i:2d}: {r['speaker']:<9} seg {r['spk_score'] or '-':>6} win {r.get('spk_win', '-'):>6} {r.get('spk_region', '')} {r['spk_ms']} ms")
    print(f"podcast: best window max {max(float(r.get('spk_win') or 0) for r in neg):.3f}")
    print(f"podcast chunks -> owner {sum(r['speaker'] == 'owner' for r in neg)}, uncertain {sum(r['speaker'] == 'uncertain' for r in neg)} of {len(neg)}")
    owners = sorted(glob.glob(str(ROOT / "research/stt/data/owner/*.wav")))
    for gain in (1.0, 0.5):
        res = []
        for k, f in enumerate(owners):
            o = sf.read(f, dtype="float32")[0]
            c = chunks[k % len(chunks)] * gain
            c[AT:AT + len(o)] += o
            r = vid(frames(c))
            truth = f"{AT / 16000:.2f}-{(AT + len(o)) / 16000:.2f}"
            res.append(r)
            print(f"  pos {Path(f).name} podcast x{gain}: {r['speaker']:<9} seg {r['spk_score'] or '-':>6} win {r.get('spk_win', '-'):>6} "
                  f"region {r.get('spk_region', '-'):<11} truth {truth}")
        print(f"owner over podcast x{gain}: {np.mean([r['spk_ms'] and float(r['spk_ms']) for r in res]):.0f} ms per chunk")
        print(f"owner over podcast x{gain} -> owner {sum(r['speaker'] == 'owner' for r in res)}, "
              f"uncertain {sum(r['speaker'] == 'uncertain' for r in res)} of {len(res)}")


if __name__ == "__main__":
    main()
