"""services/stt/voiceid.py without models: the class from the segment and from windows (T-038 2в, T-049). The
embedding is stubbed — score() and windows() return given cosines."""
from pathlib import Path

import numpy as np
import pytest

from services.stt.voiceid import VoiceID

PARAMS = {"threshold": 0.29, "threshold_high": 0.45, "voiced_rms": 300, "min_speech_s": 0.5, "win_s": 1.5,
          "win_min_s": 2.0, "win_hop_s": 0.5, "threshold_high_bcast": 0.40, "bcast_s": 30,
          "model": "none.onnx", "owner": "none.npy"}


def voiceid(seg, wins):
    v = VoiceID(Path("/nonexistent"), PARAMS)
    v.ex = object()                                    # «a model is loaded»
    v.score = lambda frames, min_s: seg
    v.windows = lambda frames: [(i * 0.5, i * 0.5 + 1.5, s) for i, s in enumerate(wins)]
    return v


FRAMES = [np.zeros(480, dtype=np.int16)] * 100        # 3 s of 30-ms frames


@pytest.mark.parametrize("seg, wins, speaker, region", [
    (0.405, [0.38, 0.449, 0.40], "uncertain", False),  # T-049 gate: same class, a little higher -> no region
    (0.425, [0.445, 0.41], "uncertain", False),
    (0.22, [0.20, 0.47, 0.30], "owner", True),         # 2в: the owner's phrase inside the podcast — the window finds it
    (0.22, [0.20, 0.35, 0.21], "uncertain", True),     # other -> uncertain: found by the window
    (0.40, [0.39, 0.48, 0.41], "owner", True),         # uncertain -> owner
    (None, [0.31, 0.33], "uncertain", True),           # unknown as a whole
    (0.25, [0.20, 0.28], "other", False),
])
def test_window_region_only_when_the_class_rises(seg, wins, speaker, region):
    out = voiceid(seg, wins)(FRAMES, now=1000.0)
    assert out["speaker"] == speaker
    assert ("spk_region" in out) is region


def loud(seconds):
    return [np.full(480, 1000, dtype=np.int16)] * int(round(seconds / 0.03))


@pytest.mark.parametrize("speech_s, bcast_after", [
    (0.6, False),        # T-050 live: the owner's «Нет» at 3 m, `other` 0.24 — not a broadcast
    (1.5, True),         # a foreign voice with enough speech: broadcast for bcast_s
])
def test_short_other_does_not_mark_a_broadcast(speech_s, bcast_after):
    v = voiceid(0.20, [])
    assert v(loud(speech_s), now=1000.0)["speaker"] == "other"
    assert ("bcast" in v(loud(0.9), now=1010.0)) is bcast_after
