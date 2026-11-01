"""core/gaze_bands.py: bands by the rule «gap >= 2 x spread, at most three» (T-031 step 1г)."""
import numpy as np
import pytest

from core.gaze_bands import feature_stats, select_bands, spread

rng = np.random.default_rng(1)


def frames(n, **dists):
    return [{k: float(rng.normal(m, s)) for k, (m, s) in dists.items()} for _ in range(n)]


def test_spread_is_robust():
    assert spread([0, 0, 0, 0, 100]) == 0.0
    assert spread([1, 2, 3, 4, 5]) == pytest.approx(1.4826)


def test_select_by_gap_over_spread():
    looks = frames(200, a=(1.0, 0.1), b=(0.0, 0.1), c=(0.5, 0.1), d=(0.0, 0.1))
    work = frames(800, a=(0.0, 0.1), b=(0.3, 0.1), c=(0.0, 0.1), d=(0.0, 0.1))
    st = feature_stats(looks, work, ["a", "b", "c", "d", "missing"])
    assert set(st) == {"a", "b", "c", "d"}
    assert st["a"]["ratio"] > st["c"]["ratio"] > st["b"]["ratio"]
    bands = select_bands(st)
    assert list(bands) == ["a", "c", "b"]                  # all three qualify, best first; d (no gap) does not
    lo, hi = bands["a"]
    assert lo < 1.0 < hi and hi - lo < 0.5                  # [p5, p95] of the looks


def test_at_most_three_and_none_qualifying():
    looks = frames(200, **{k: (1.0, 0.1) for k in "abcde"})
    work = frames(200, **{k: (0.0, 0.1) for k in "abcde"})
    assert len(select_bands(feature_stats(looks, work, list("abcde")))) == 3
    same = frames(200, a=(0.0, 0.1))
    assert select_bands(feature_stats(same, frames(200, a=(0.0, 0.1)), ["a"])) == {}


def test_spread_is_the_larger_of_looks_and_work():
    looks = frames(300, a=(1.0, 0.05))
    work = frames(300, a=(0.0, 0.6))                        # wide work: gap 1.0 < 2 x 0.6
    st = feature_stats(looks, work, ["a"])
    assert st["a"]["spread"] == pytest.approx(0.6, rel=0.2)
    assert select_bands(st) == {}
