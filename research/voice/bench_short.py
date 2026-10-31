"""Voice id on short phrases (T-050, I-19): how far down the owner still separates from the negatives.

Every recording is reduced to its voiced 30-ms frames (services/stt/voiceid.VoiceID.voiced — the production gate) and
cut into consecutive pieces of 0.5 / 0.75 / 1.0 / 1.5 s of speech (and the whole); each piece is embedded by CAM++ and
compared with the enrollment vector models/voice/owner.npy. Positives: the owner's recordings not used for the
enrollment (owner.npy = enroll_A): enroll_B, owner_normal, owner_loud, owner_far (models/voice/data, 3 s each) and the
owner's phrases of the STT set (research/stt/data/owner). Negatives: broadcast (a podcast from a phone through the stand
mic) and piper (the skull's own voice). Audio is not in git (the owner's voice).

  .venv/bin/python research/voice/bench_short.py
-> research/voice/results/voiceid_short_<date>.md + .csv
"""
import csv
import glob
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services/stt"))
from voiceid import VoiceID

DATA = ROOT / "models/voice/data"
POS = {"enroll_B": DATA / "enroll_B", "owner_normal": DATA / "owner_normal", "owner_loud": DATA / "owner_loud",
       "owner_far": DATA / "owner_far", "stt_owner": ROOT / "research/stt/data/owner"}
NEG = {"broadcast": DATA / "broadcast", "piper": DATA / "piper"}
LENGTHS = [0.5, 0.75, 1.0, 1.5, None]           # None = the whole recording
GATE = {r["file"]: r["speech_gate"] == "1"      # the production speech gate on the whole 3-s recording (T-039): a
        for r in csv.DictReader(open(DATA / "../../../research/voice/data/labels.csv", encoding="utf-8"))}   # failed one is `unknown`
OUT = ROOT / "research/voice/results"


def frames(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    assert sr == 16000, path
    a = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    return [a[i:i + 480] for i in range(0, len(a) // 480 * 480, 480)]


def cosine(vid, voiced):
    st = vid.ex.create_stream()
    st.accept_waveform(16000, np.concatenate(voiced).astype(np.float32) / 32768.0)
    st.input_finished()
    e = np.asarray(vid.ex.compute(st), dtype=np.float32)
    return float(e @ vid.owner / max(np.linalg.norm(e), 1e-8))


def pieces(voiced, length):
    if length is None:
        return [voiced] if voiced else []
    n = int(round(length / 0.03))
    return [voiced[i:i + n] for i in range(0, len(voiced) - n + 1, n)]


def main():
    params = yaml.safe_load((ROOT / "services/stt/manifest.yaml").read_text(encoding="utf-8"))["params"]["voice_id"]
    vid = VoiceID(ROOT, params)
    rows = []
    for group, kind in [(POS, "owner"), (NEG, "neg")]:
        for label, d in group.items():
            for f in sorted(glob.glob(str(d / "*.wav"))):
                if not GATE.get(f"{label}/{Path(f).name}", True):
                    continue
                v = vid.voiced(frames(f))
                for L in LENGTHS:
                    for k, p in enumerate(pieces(v, L)):
                        if np.median([np.sqrt(np.mean(x.astype(np.float32) ** 2)) for x in p]) < vid.voiced_rms:
                            continue        # score() would say `unknown`: a breath / pause tail, not a short phrase
                        rows.append({"file": str(Path(f).relative_to(ROOT)), "label": label, "kind": kind,
                                     "length": "whole" if L is None else L, "piece": k,
                                     "speech_s": round(len(p) * 0.03, 2), "cos": round(cosine(vid, p), 4)})
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    with open(OUT / f"voiceid_short_{stamp}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def table(sel, title):
        out = [f"**{title}**", "", "| Длина речи | Владелец: n, медиана, 5-й перц., мин | Негатив: n, 95-й перц. (FAR 5 %), макс | Запас: 5-й перц. владельца − FAR 5 % | Владелец ≥ 0.29 / ≥ 0.45 / ≥ FAR 5 % | Негатив ≥ 0.29 / ≥ 0.45 |",
               "|---|---|---|---|---|---|"]
        for L in LENGTHS:
            key = "whole" if L is None else L
            o = np.array([r["cos"] for r in rows if r["kind"] == "owner" and r["length"] == key and sel(r)])
            n = np.array([r["cos"] for r in rows if r["kind"] == "neg" and r["length"] == key and sel(r)])
            far5, o5 = float(np.percentile(n, 95)), float(np.percentile(o, 5))
            out.append(f"| {key if L is None else f'{L} с'} | {len(o)}, {np.median(o):.3f}, {o5:.3f}, {o.min():.3f} | "
                       f"{len(n)}, {far5:.3f}, {n.max():.3f} | {o5 - far5:+.3f} | "
                       f"{(o >= 0.29).mean():.0%} / {(o >= 0.45).mean():.0%} / {(o >= far5).mean():.0%} | "
                       f"{(n >= 0.29).mean():.1%} / {(n >= 0.45).mean():.1%} |")
        return out
    head = table(lambda r: r["piece"] == 0, "Первые L с речи каждой записи") + [""] + table(lambda r: True, "Все куски по L с подряд")
    lines = ["| Длина речи | Владелец: n, медиана, 5-й перц., мин | Негатив: n, 95-й перц. (FAR 5 %), макс | Запас: 5-й перц. владельца − FAR 5 % | Владелец ≥ 0.29 / ≥ 0.45 | Негатив ≥ 0.29 |",
             "|---|---|---|---|---|---|"]
    summary = {}
    for L in LENGTHS:
        key = "whole" if L is None else L
        o = np.array([r["cos"] for r in rows if r["kind"] == "owner" and r["length"] == key])
        n = np.array([r["cos"] for r in rows if r["kind"] == "neg" and r["length"] == key])
        far5, o5 = float(np.percentile(n, 95)), float(np.percentile(o, 5))
        summary[key] = (o, n, far5, o5)
        lines.append(f"| {key if L is None else f'{L} с'} | {len(o)}, {np.median(o):.3f}, {o5:.3f}, {o.min():.3f} | "
                     f"{len(n)}, {far5:.3f}, {n.max():.3f} | {o5 - far5:+.3f} | "
                     f"{(o >= 0.29).mean():.0%} / {(o >= 0.45).mean():.0%} | {(n >= 0.29).mean():.1%} |")
    by_label = ["| Набор | " + " | ".join("целиком" if L is None else f"{L} с" for L in LENGTHS) + " |",
                "|---|" + "---|" * len(LENGTHS)]
    for label in list(POS) + list(NEG):
        cells = []
        for L in LENGTHS:
            key = "whole" if L is None else L
            c = np.array([r["cos"] for r in rows if r["label"] == label and r["length"] == key])
            cells.append(f"{np.median(c):.3f} ({c.min():.3f}–{c.max():.3f})" if len(c) else "—")
        by_label.append(f"| {label} | " + " | ".join(cells) + " |")
    text = "\n".join(head) + "\n\nМедиана (мин–макс) косинуса по наборам:\n\n" + "\n".join(by_label) + "\n"
    (OUT / f"voiceid_short_{stamp}.table.md").write_text(text, encoding="utf-8")   # pasted into the summary voiceid_short_<date>.md
    print(text)


if __name__ == "__main__":
    main()
