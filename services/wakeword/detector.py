"""Listening trigger -> events:wakeword (manifest params.trigger).

  kws     openWakeWord "hey jarvis" (score > 0.5)
  speech  T-038 stage 2b (ambient): speech segments by Silero VAD (sherpa-onnx, CPU) — split at pauses inside
          continuous speech, at most max_segment_s; one event per segment with its first and last frame (audio_id,
          end_id), so STT gets the phrase from its start (the KWS trigger loses ~0.3 s, I-14) and the end by VAD (I-03).
          Replaces the adaptive RMS threshold of stage 2: a podcast held it «in speech» for 10–18 min.
          tts:speaking mutes it. openWakeWord is not loaded.
"""
import collections
import time
from pathlib import Path

import numpy as np
import redis
import yaml

r = redis.Redis()
ROOT = Path(__file__).resolve().parents[2]
RATE = 16000
P = yaml.safe_load(Path(__file__).with_name("manifest.yaml").read_text(encoding="utf-8")).get("params", {})
TRIGGER = P.get("trigger", "kws")


def run_kws():
    from openwakeword.model import Model
    MODELS = "/home/valeriy/skull-assistant/models/wakeword"
    model = Model(
        wakeword_models=[f"{MODELS}/hey_jarvis_v0.1.onnx"],
        melspec_model_path=f"{MODELS}/melspectrogram.onnx",
        embedding_model_path=f"{MODELS}/embedding_model.onnx",
        inference_framework="onnx",
    )
    THRESHOLD, COOLDOWN_FRAMES = 0.5, 40        # ~1.2 с тишины после срабатывания
    last_id, cooldown = '$', 0
    print("listening for 'hey jarvis'...", flush=True)
    while True:
        resp = r.xread({'audio:mic': last_id}, count=10, block=1000)
        if not resp:
            continue
        for msg_id, fields in resp[0][1]:
            last_id = msg_id
            pcm = np.frombuffer(fields[b'pcm'], dtype=np.int16)
            scores = model.predict(pcm)
            name, score = next(iter(scores.items()))
            if cooldown > 0:
                cooldown -= 1
            elif score > THRESHOLD:
                if r.exists('tts:speaking'):
                    continue
                r.xadd('events:wakeword', {'word': 'hey_jarvis',
                                           'score': f"{score:.3f}",
                                           'audio_id': msg_id})
                print(f"WAKE! score={score:.3f}", flush=True)
                cooldown = COOLDOWN_FRAMES


def run_speech():
    """Segments by Silero VAD (sherpa-onnx): the event goes out at the end of a segment, with its first and last frame,
    so STT reads exactly that stretch from audio:mic. vad:onset = wall time of the latest start of speech (STT measures
    the pause after a phrase by it). A segment that overlapped tts:speaking is dropped (the skull's own voice).
    Silero's max_speech_duration is not a hard limit (podcast: segments of 16–19 s), so speech running longer than
    max_segment_s is cut here: the piece goes out and the rest of the VAD segment starts at the cut."""
    import sherpa_onnx
    S = P.get("speech", {})
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(ROOT / S.get("vad_model", "models/vad/silero_vad.onnx"))
    cfg.silero_vad.threshold = S.get("threshold", 0.5)
    cfg.silero_vad.min_silence_duration = S.get("min_silence_s", 0.5)
    cfg.silero_vad.min_speech_duration = S.get("min_speech_s", 0.25)
    cfg.silero_vad.max_speech_duration = S.get("max_segment_s", 12)
    cfg.sample_rate, cfg.num_threads = RATE, 1
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
    frames = collections.deque(maxlen=2000)    # (first sample, msg_id) of recent frames, ~60 s like audio:mic
    max_len = int(S.get("max_segment_s", 12) * RATE)
    n, cut, speaking, muted, last_id = 0, 0, False, False, '$'
    print(f"listening for speech: Silero VAD threshold {cfg.silero_vad.threshold}, silence {cfg.silero_vad.min_silence_duration} s, "
          f"segment <= {cfg.silero_vad.max_speech_duration} s", flush=True)

    def frame_at(sample):
        return next((mid for off, mid in reversed(frames) if off <= sample), frames[0][1])

    def emit(start, end, why=""):
        if muted:
            print(f"SPEECH {(end - start) / RATE:.1f}s dropped (tts:speaking)", flush=True)
            return
        r.xadd('events:wakeword', {'word': 'speech', 'score': f"{(end - start) / RATE:.1f}", 'audio_id': frame_at(start),
                                   'end_id': frame_at(end - 1), 'end_ts': f"{time.time() - (n - end) / RATE:.3f}"})
        print(f"SPEECH {(end - start) / RATE:.1f}s{why}", flush=True)

    while True:
        resp = r.xread({'audio:mic': last_id}, count=10, block=1000)
        if not resp:
            continue
        for msg_id, fields in resp[0][1]:
            last_id = msg_id
            pcm = np.frombuffer(fields[b'pcm'], dtype=np.int16)
            frames.append((n, msg_id))
            n += len(pcm)
            tts = bool(r.exists('tts:speaking'))
            vad.accept_waveform(pcm.astype(np.float32) / 32768.0)
            if vad.is_speech_detected() and not speaking:
                r.set('vad:onset', f"{time.time():.3f}")
                muted = tts             # only the skull speaking *during* the segment mutes it (not before: live 2b)
            speaking = vad.is_speech_detected()
            muted = muted or (speaking and tts)
            if speaking and (n // len(pcm)) % 16 == 0:   # current_segment copies the whole segment: every 0.5 s, not
                cur = vad.current_segment                 # every frame (8.6 % of a core on a podcast vs 1.0 %)
                start = max(cur.start, cut)
                if cur.start + len(cur.samples) - start >= max_len:
                    cut = cur.start + len(cur.samples)
                    emit(start, cut, " (cut)")
                    muted = tts
            while not vad.empty():
                seg = vad.front
                start, end = max(seg.start, cut), seg.start + len(seg.samples)
                vad.pop()
                if end - start >= 0.3 * RATE:       # what is left after a cut may be a tail
                    emit(start, end)
                muted = speaking and tts


if __name__ == "__main__":
    run_speech() if TRIGGER == "speech" else run_kws()
