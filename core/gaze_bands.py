"""Calibrated bands for «looking at the skull» (T-031): which per-frame gaze features separate the marked looks from
work, and their bands. No camera or model here — numbers in, numbers out (research/vision/gaze_calib.py prints the
report over recordings, services/vision/manifest.yaml keeps the chosen bands).

Per feature: band = [p5, p95] of the look frames; spread = robust spread (1.4826 x MAD) of looks or work, whichever is
larger; gap = |median of looks - median of work|.

Method (core, step 1г, 2026-11-01): **candidates by the gap, the set by the replay.**
  1. Candidates: every feature with gap >= RATIO x spread (select_bands: best gap / spread first, at most MAX_N).
  2. The set of 1-3 candidates is confirmed by replaying the service's dwell logic over the recording: looks caught
     >= 8 of 10 and 0 false triggers over all the work; on a tie the fewer features win.
The gap alone is not enough: on the 30.10 recording it picked iris_v + iris_h (5 of 10 looks caught), the replay —
iris_h + nose (8 of 10, 0 false): every extra [p5, p95] band keeps only ~90 % of the look frames, and the dwell needs
them in a row.
"""
from itertools import combinations

import numpy as np

RATIO, MAX_N = 2.0, 3
MIN_CAUGHT = 0.8                  # looks caught by the replay: >= 8 of 10


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


def candidates(stats, ratio=RATIO):
    """Every feature with gap >= ratio x spread, best gap / spread first."""
    return sorted((k for k, s in stats.items() if s["ratio"] >= ratio), key=lambda k: -stats[k]["ratio"])


def choose_by_replay(stats, evaluate, keys=None, max_n=MAX_N, min_caught=MIN_CAUGHT):
    """The set of 1..max_n features (from `keys`, by default the candidates) that passes the replay — evaluate(bands) ->
    (caught, looks, false): caught / looks >= min_caught and 0 false; fewer features win, then more looks caught, then
    the larger smallest gap / spread. -> (bands, (caught, looks, false)) or (None, None) when no set passes."""
    keys = candidates(stats) if keys is None else [k for k in keys if k in stats]
    best = None
    for n in range(1, max_n + 1):
        for combo in combinations(keys, n):
            bands = {k: stats[k]["band"] for k in combo}
            caught, looks, false = evaluate(bands)
            if false or caught < min_caught * looks:
                continue
            rank = (n, -caught, -min(stats[k]["ratio"] for k in combo))
            if best is None or rank < best[0]:
                best = (rank, bands, (caught, looks, false))
    return (best[1], best[2]) if best else (None, None)
