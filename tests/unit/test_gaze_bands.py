"""core/gaze_bands.py: bands by the rule «gap >= 2 x spread, at most three» (T-031 step 1г)."""
import numpy as np
import pytest

from core.gaze_bands import candidates, choose_by_replay, feature_stats, select_bands, spread

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


def fake_stats(**ratios):
    return {k: {"band": (0.0, 1.0), "ratio": r} for k, r in ratios.items()}


def test_candidates_order():
    assert candidates(fake_stats(a=2.1, b=0.8, c=2.3)) == ["c", "a"]


def test_replay_picks_the_passing_set_with_fewest_features():
    st = fake_stats(yaw=8.7, gaze_h=4.4, nose=2.7)
    table = {("yaw",): (9, 12, 1), ("gaze_h",): (11, 12, 0), ("nose",): (12, 12, 3), ("gaze_h", "yaw"): (11, 12, 0)}
    bands, res = choose_by_replay(st, lambda b: table.get(tuple(sorted(b)), (12, 12, 0)))
    assert list(bands) == ["gaze_h"] and res == (11, 12, 0)


def test_replay_none_passes_like_30_10():
    """30.10: candidates iris_v + iris_h only; iris_h — 1 false, iris_v — 2, both — 5 of 10: no set passes; among all
    features the replay finds iris_h + nose (8 of 10, 0 false)."""
    st = fake_stats(iris_h=2.1, iris_v=2.3, nose=0.8)
    table = {("iris_h",): (8, 10, 1), ("iris_v",): (7, 10, 2), ("iris_h", "iris_v"): (5, 10, 2), ("nose",): (7, 10, 9),
             ("iris_h", "nose"): (8, 10, 0), ("iris_v", "nose"): (6, 10, 0), ("iris_h", "iris_v", "nose"): (5, 10, 0)}
    ev = lambda b: table[tuple(sorted(b))]
    assert choose_by_replay(st, ev) == (None, None)
    bands, res = choose_by_replay(st, ev, keys=list(st))
    assert set(bands) == {"iris_h", "nose"} and res == (8, 10, 0)
