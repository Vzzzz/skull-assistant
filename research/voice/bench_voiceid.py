"""Voice ID EER (T-039): owner vs broadcast/Piper through the stand microphone, for each candidate model.

Owner vector = mean embedding of the enrollment label; every segment of the positive and negative labels is scored
by cosine to it. Reports EER, the threshold at FAR 5 % (and FRR there), per-label score ranges, ms per 3-s segment —
for all segments and for segments with speech (median frame RMS >= --min-median-rms, the same gate as in STT):
the recorder also keeps fan noise bursts and keyboard clicks (+3 hangover frames), which are not anybody's voice.

  .venv/bin/python research/voice/bench_voiceid.py      # labels from models/voice/data/, defaults below
Outputs: research/voice/results/voiceid_<date>.md and voiceid_scores_<date>.csv
"""
import argparse
import csv
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enroll import DATA, ROOT, embed, extractor  # noqa: E402

MODELS = ["3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx", "wespeaker_en_voxceleb_resnet34.onnx"]
OUT = ROOT / "research/voice/results"


def median_rms(audio):
    """Median 30-ms frame RMS on the int16 scale."""
    fr = audio[: len(audio) // 480 * 480].reshape(-1, 480)
    return float(np.median(np.sqrt((fr ** 2).mean(axis=1)) * 32768))


def eer(pos, neg):
    """Equal error rate and its threshold over all candidate thresholds."""
    best = (1.0, 0.0)
    for t in sorted(set(pos) | set(neg)):
        far = sum(s >= t for s in neg) / len(neg)
        frr = sum(s < t for s in pos) / len(pos)
        if abs(far - frr) < abs(best[0] - best[1]) or best == (1.0, 0.0):
            best, thr = (far, frr), t
    return (best[0] + best[1]) / 2, thr


def thr_at_far(pos, neg, far_max=0.05):
    """Lowest threshold with FAR <= far_max; returns (threshold, FRR there)."""
    for t in sorted(set(pos) | set(neg)):
        if sum(s >= t for s in neg) / len(neg) <= far_max:
            return t, sum(s < t for s in pos) / len(pos)
    return max(neg) + 1e-6, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enroll", default="enroll_A")
    ap.add_argument("--pos", nargs="+", default=["enroll_B", "owner_normal", "owner_loud", "owner_far"])
    ap.add_argument("--neg", nargs="+", default=["broadcast", "piper"])
    ap.add_argument("--min-median-rms", type=float, default=150)
    a = ap.parse_args()
    files = {lab: sorted((DATA / lab).glob("*.wav")) for lab in a.enroll.split() + a.pos + a.neg}
    for lab, fs in files.items():
        print(f"{lab}: {len(fs)} segments")
    OUT.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    rows, md = [], [f"# Voice ID: владелец против эфира и Piper ({today}, стенд, микрофон ноутбука через `audio:mic`)", "",
                    f"Enrollment: `{a.enroll}` ({len(files[a.enroll])} × 3 с). Позитивы: {', '.join(f'`{l}` {len(files[l])}' for l in a.pos)}. "
                    f"Негативы: {', '.join(f'`{l}` {len(files[l])}' for l in a.neg)}. Косинус к среднему вектору; 1 поток ORT.", "",
                    "| Модель | dim | Сегменты | EER | порог EER | порог @ FAR 5 % | FRR @ FAR 5 % | мс / сегмент 3 с (средн. / p95) |",
                    "|---|---|---|---|---|---|---|---|"]
    per_label = []
    for model in MODELS:
        ex = extractor(model)
        enr = [embed(ex, sf.read(f, dtype="float32")[0]) for f in files[a.enroll]]
        owner = np.mean(enr, axis=0); owner /= np.linalg.norm(owner)
        scores, ms = {}, []
        for lab in a.pos + a.neg:
            scores[lab] = []
            for f in files[lab]:
                audio = sf.read(f, dtype="float32")[0]
                t0 = time.perf_counter()
                e = embed(ex, audio)
                ms.append((time.perf_counter() - t0) * 1000)
                s = float(e @ owner)
                speech = median_rms(audio) >= a.min_median_rms
                scores[lab].append((s, speech))
                rows.append([model, lab, f.name, round(s, 4), int(speech)])
        p95 = sorted(ms)[int(0.95 * (len(ms) - 1))]
        for gated in (False, True):
            pos = [s for l in a.pos for s, ok in scores[l] if ok or not gated]
            neg = [s for l in a.neg for s, ok in scores[l] if ok or not gated]
            e_rate, e_thr = eer(pos, neg)
            t5, frr5 = thr_at_far(pos, neg)
            what = f"речь (медиана ≥ {a.min_median_rms:.0f}): {len(pos)}+ / {len(neg)}−" if gated else f"все: {len(pos)}+ / {len(neg)}−"
            md.append(f"| {model} | {ex.dim} | {what} | {e_rate:.3f} | {e_thr:.3f} | {t5:.3f} | {frr5:.3f} | {statistics.mean(ms):.0f} / {p95:.0f} |")
            print(f"{model} [{what}]: EER {e_rate:.3f} @ {e_thr:.3f}; thr@FAR5% {t5:.3f} (FRR {frr5:.3f}); {statistics.mean(ms):.0f} ms/seg", flush=True)
        for lab in a.pos + a.neg:
            sc = [s for s, ok in scores[lab] if ok]
            per_label.append(f"| {model.split('_')[0]} | `{lab}` | {'+' if lab in a.pos else '−'} | {len(sc)} из {len(scores[lab])} | "
                             f"{statistics.mean(sc):.3f} | {min(sc):.3f} | {max(sc):.3f} |")
    md += ["", "Косинус по меткам (только сегменты с речью):", "",
           "| Модель | Метка | ± | n | косинус средн. | мин | макс |", "|---|---|---|---|---|---|---|"] + per_label
    (OUT / f"voiceid_{today}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(OUT / f"voiceid_scores_{today}.csv", "w", newline="") as f:
        csv.writer(f).writerows([["model", "label", "file", "score", "speech"]] + rows)
    print(f"-> {(OUT / f'voiceid_{today}.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
