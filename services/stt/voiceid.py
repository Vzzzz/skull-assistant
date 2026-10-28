"""Speaker verification for STT (T-039, D-029): was the phrase said by the owner?

Same preprocessing as research/voice (the threshold was measured on it): only voiced 30-ms frames
(RMS >= voiced_rms, +3 hangover frames), CAM++ embedding (sherpa-onnx, CPU, 1 thread), cosine to the owner's
enrollment vector: owner >= threshold_high > uncertain >= threshold > other (T-039 answer 1). Too little speech
(or mostly noise), no model or no enrollment -> 'unknown', never 'other'.

Windows (T-038 2в): on a continuous broadcast the owner's phrase sits inside a 12-s segment of the podcast and the
whole-segment embedding is the podcast's (`other` 0.20–0.24 on «Костян, где ключи оставил?», 2026-10-28). Segments
from win_min_s on that are not `owner` as a whole are also scored in sliding windows of win_s (step win_hop_s); the class
is taken from the best of segment and windows, and when a window lifts it, `region` = the run of windows at or above
`threshold` around the best one (STT cuts the text to it). Offline (research/voice/bench_windows.py): owner phrases over
a podcast at the real level — owner 0/10 by the segment, 8/10 by 1.5-s windows with a 0.5-s step; podcast chunks 0/13.
Broadcast (experiment, core's answer 3 to the 2в report): after a segment that is `other` as a whole, for bcast_s the
owner threshold of window scores and of short phrases without windows is threshold_high_bcast (live: owner over the
podcast 0.42–0.51, podcast windows <= 0.36; family voices reach 0.418 — to be re-measured before it stays); `bcast`
goes into the event either way.
"""
import time

import numpy as np

HANGOVER = 3


class VoiceID:
    def __init__(self, root, params):
        self.threshold = params["threshold"]
        self.threshold_high = params.get("threshold_high", self.threshold)   # owner >= high > uncertain >= threshold
        self.voiced_rms = params["voiced_rms"]
        self.min_speech_s = params["min_speech_s"]
        self.win_s, self.win_min_s = params.get("win_s", 1.5), params.get("win_min_s", 3.0)   # 0 = no windows
        self.win_hop_s = params.get("win_hop_s", self.win_s / 2)
        self.high_bcast, self.bcast_s = params.get("threshold_high_bcast"), params.get("bcast_s", 30)   # None = off
        self.last_other = -1e9
        model, owner = root / params["model"], root / params["owner"]
        self.ex = self.owner = None
        if model.exists() and owner.exists():
            import sherpa_onnx
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=1)
            self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            self.owner = np.load(owner).astype(np.float32)
        self.state = f"{model.name}, threshold {self.threshold}" if self.ex else f"off (no {model.name if not model.exists() else owner.name})"

    def voiced(self, frames):
        out, since = [], HANGOVER + 1
        for f in frames:
            since = 0 if np.sqrt(np.mean(f.astype(np.float32) ** 2)) >= self.voiced_rms else since + 1
            if since <= HANGOVER:
                out.append(f)
        return out

    def score(self, frames, min_speech_s):
        """Cosine to the owner over the voiced frames, or None when there is too little speech (gate of the measurement:
        enough speech, and mostly speech rather than noise bursts with their hangover)."""
        v = self.voiced(frames)
        if not v or len(v) * len(frames[0]) / 16000 < min_speech_s:
            return None
        if np.median([np.sqrt(np.mean(f.astype(np.float32) ** 2)) for f in v]) < self.voiced_rms:
            return None
        st = self.ex.create_stream()
        st.accept_waveform(16000, np.concatenate(v).astype(np.float32) / 32768.0)
        st.input_finished()
        e = np.asarray(self.ex.compute(st), dtype=np.float32)
        return float(e @ self.owner / max(np.linalg.norm(e), 1e-8))

    def label(self, score, high=None):
        return "owner" if score >= (high or self.threshold_high) else "uncertain" if score >= self.threshold else "other"

    def windows(self, frames):
        """[(start_s, end_s, score)] of sliding windows; a window needs half of its length voiced."""
        fs = len(frames[0]) / 16000
        n, hop = max(1, round(self.win_s / fs)), max(1, round(self.win_hop_s / fs))
        out = []
        for i in range(0, max(1, len(frames) - n + 1), hop):
            s = self.score(frames[i:i + n], self.win_s / 2)
            if s is not None:
                out.append((i * fs, (i + n) * fs, s))
        return out

    def __call__(self, frames, now=None):
        """frames: int16 30-ms chunks of the phrase. Returns event fields speaker / spk_score / spk_ms, and with windows
        spk_win (best window score) and spk_region ("start-end" seconds, only when the best window is above the segment
        and not `other`: the class then comes from the window)."""
        if self.ex is None:
            return {"speaker": "unknown", "spk_score": "", "spk_ms": ""}
        t0, now = time.perf_counter(), time.time() if now is None else now
        bcast = self.high_bcast is not None and now - self.last_other <= self.bcast_s
        high = self.high_bcast if bcast else None
        windowed = self.win_s and len(frames) * len(frames[0]) / 16000 >= self.win_min_s
        score = self.score(frames, self.min_speech_s)
        out = {"speaker": "unknown" if score is None else self.label(score, None if windowed else high),
               "spk_score": "" if score is None else f"{score:.3f}"}
        if score is not None and self.label(score) == "other":
            self.last_other = now                   # a foreign voice as a whole: broadcast / guests (2в)
        if bcast:
            out["bcast"] = "1"
        if windowed and out["speaker"] != "owner":
            w = self.windows(frames)
            if w:
                k = max(range(len(w)), key=lambda j: w[j][2])
                out["spk_win"] = f"{w[k][2]:.3f}"
                if self.label(w[k][2], high) != "other" and (score is None or w[k][2] > score):
                    a = b = k                       # the run of windows at or above `threshold` around the best one
                    while a > 0 and w[a - 1][2] >= self.threshold and w[a - 1][1] >= w[a][0]:
                        a -= 1
                    while b < len(w) - 1 and w[b + 1][2] >= self.threshold and w[b + 1][0] <= w[b][1]:
                        b += 1
                    out.update(speaker=self.label(w[k][2], high), spk_region=f"{w[a][0]:.2f}-{w[b][1]:.2f}")
        out["spk_ms"] = f"{(time.perf_counter() - t0) * 1000:.0f}"
        return out
