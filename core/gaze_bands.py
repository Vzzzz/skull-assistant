"""Calibrated bands for «looking at the skull» (T-031): which per-frame gaze features separate the marked looks from
work, and their bands. No camera or model here — numbers in, numbers out (research/vision/gaze_calib.py prints the
report over recordings, services/vision/manifest.yaml keeps the chosen bands).

Per feature: band = [p5, p95] of the look frames; spread = robust spread (1.4826 x MAD) of looks or work, whichever is
larger; gap = |median of looks - median of work|. Rule (core, 2026-10-29 / step 1г): a feature qualifies when
gap >= RATIO x spread; the bands are the qualifying features with the largest gap / spread, at most MAX_N.
"""
import numpy as np

RATIO, MAX_N = 2.0, 3


def spread(v):
    v = np.asarray(v, dtype=float)
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def feature_stats(looks, work, keys):
    """looks / work: lists of frame dicts -> {key: {band, med_look, med_work, spread, gap, ratio}} for keys present."""
    out = {}
    for k in keys:
        s = [x[k] for x in looks if k in x]
        w = [x[k] for x in work if k in x]
        if len(s) < 2 or len(w) < 2:
            continue
        sp = max(spread(s), spread(w), 1e-9)
        gap = abs(float(np.median(s)) - float(np.median(w)))
        out[k] = {"band": (float(np.percentile(s, 5)), float(np.percentile(s, 95))), "med_look": float(np.median(s)),
                  "med_work": float(np.median(w)), "spread": sp, "gap": gap, "ratio": gap / sp}
    return out


def select_bands(stats, ratio=RATIO, max_n=MAX_N):
    """-> {key: (lo, hi)}: the qualifying features (gap >= ratio x spread), best gap / spread first, at most max_n."""
    ok = sorted((k for k, s in stats.items() if s["ratio"] >= ratio), key=lambda k: -stats[k]["ratio"])
    return {k: stats[k]["band"] for k in ok[:max_n]}
