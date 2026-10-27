"""Listening trigger -> events:wakeword (manifest params.trigger).

  kws     openWakeWord "hey jarvis" (score > 0.5)
  speech  T-038 stage 2 (ambient): start of speech by an adaptive RMS threshold — max(floor, factor x median RMS
          of the background over the last window_s), start_frames loud frames in a row, end after end_silence_s
          below end_ratio x threshold (hysteresis), not more often than min_interval_s; tts:speaking mutes it.
          audio_id points preroll_frames before the first loud frame, so STT records the phrase from its start
          (the KWS trigger loses ~0.3 s, kws_monitor_2026-10-24.md). openWakeWord is not loaded.
"""
import collections
import time
from pathlib import Path

import numpy as np
import redis
import yaml

r = redis.Redis()
P = yaml.safe_load(Path(__file__).with_name("manifest.yaml").read_text(encoding="utf-8")).get("params", {})
TRIGGER = P.get("trigger", "kws")


def rms(pcm):
    return float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))


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
    S = P.get("speech", {})
    frame_s = 0.03
    bg = collections.deque(maxlen=int(S.get("window_s", 10) / frame_s))      # background RMS, outside speech only
    floor, factor = S.get("floor_rms", 100), S.get("factor", 2.0)
    start_frames, preroll = S.get("start_frames", 3), S.get("preroll_frames", 5)
    end_frames = int(S.get("end_silence_s", 0.9) / frame_s)
    end_ratio, min_interval = S.get("end_ratio", 0.8), S.get("min_interval_s", 2.0)
    ids = collections.deque(maxlen=start_frames + preroll + 1)
    in_speech, loud, quiet, last_fire, last_id = False, 0, 0, 0.0, '$'
    print(f"listening for speech: threshold max({floor}, {factor} x median of {S.get('window_s', 10)} s)", flush=True)
    while True:
        resp = r.xread({'audio:mic': last_id}, count=10, block=1000)
        if not resp:
            continue
        for msg_id, fields in resp[0][1]:
            last_id = msg_id
            ids.append(msg_id)
            level = rms(np.frombuffer(fields[b'pcm'], dtype=np.int16))
            thr = max(floor, factor * float(np.median(bg))) if len(bg) >= 30 else floor * factor
            if in_speech:
                quiet = quiet + 1 if level < end_ratio * thr else 0
                if quiet >= end_frames:
                    in_speech, loud = False, 0
                continue
            bg.append(level)
            loud = loud + 1 if level >= thr else 0
            if loud < start_frames:
                continue
            in_speech, quiet = True, 0          # speech from here on, whether or not it fires
            now = time.time()
            if r.exists('tts:speaking') or now - last_fire < min_interval:
                continue
            last_fire = now
            r.xadd('events:wakeword', {'word': 'speech', 'score': f"{level:.0f}", 'audio_id': ids[0],
                                       'rms_thr': f"{thr:.0f}"})
            print(f"SPEECH rms={level:.0f} thr={thr:.0f}", flush=True)


if __name__ == "__main__":
    run_speech() if TRIGGER == "speech" else run_kws()
