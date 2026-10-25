"""Speaker verification for STT (T-039, D-029): was the phrase said by the owner?

Same preprocessing as research/voice (the threshold was measured on it): only voiced 30-ms frames
(RMS >= voiced_rms, +3 hangover frames), CAM++ embedding (sherpa-onnx, CPU, 1 thread), cosine to the owner's
enrollment vector. Too little speech (or mostly noise), no model or no enrollment -> 'unknown', never 'other'.
"""
import time

import numpy as np

HANGOVER = 3


class VoiceID:
    def __init__(self, root, params):
        self.threshold = params["threshold"]
        self.voiced_rms = params["voiced_rms"]
        self.min_speech_s = params["min_speech_s"]
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

    def __call__(self, frames):
        """frames: int16 30-ms chunks of the phrase. Returns event fields speaker / spk_score / spk_ms."""
        if self.ex is None:
            return {"speaker": "unknown", "spk_score": "", "spk_ms": ""}
        t0 = time.perf_counter()
        v = self.voiced(frames)
        rms = [np.sqrt(np.mean(f.astype(np.float32) ** 2)) for f in v]
        # gate of the measurement: enough speech, and mostly speech rather than noise bursts with their hangover
        if len(v) * len(frames[0]) / 16000 < self.min_speech_s or np.median(rms) < self.voiced_rms:
            return {"speaker": "unknown", "spk_score": "", "spk_ms": ""}
        audio = np.concatenate(v).astype(np.float32) / 32768.0
        st = self.ex.create_stream()
        st.accept_waveform(16000, audio)
        st.input_finished()
        e = np.asarray(self.ex.compute(st), dtype=np.float32)
        score = float(e @ self.owner / max(np.linalg.norm(e), 1e-8))
        return {"speaker": "owner" if score >= self.threshold else "other", "spk_score": f"{score:.3f}",
                "spk_ms": f"{(time.perf_counter() - t0) * 1000:.0f}"}
