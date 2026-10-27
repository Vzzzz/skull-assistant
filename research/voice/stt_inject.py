"""Feed recorded segments to a running skull-stt through the bus (T-039 step 5: RSS after N segments, speaker field).

For each WAV: a fake events:wakeword pointing at the current end of audio:mic, then the WAV as 30-ms frames and
1.2 s of silence so STT ends the phrase; waits for events:transcript and prints speaker / spk_score / spk_ms.
Stop skull-capture and skull-wakeword first: live mic frames would interleave with the injected ones.

  .venv/bin/python research/voice/stt_inject.py models/voice/data/owner_normal/00[0-9].wav models/voice/data/broadcast/00[0-9].wav
  .venv/bin/python research/voice/stt_inject.py --word=speech research/stt/data/owner/*.wav    # ambient trigger (T-038)
"""
import sys
import time

import numpy as np
import redis
import soundfile as sf

r = redis.Redis(socket_timeout=None)
args = sys.argv[1:]
word = "inject"
if args and args[0].startswith("--word="):   # --word=speech: the ambient trigger (T-038), STT measures the pause
    word, args = args[0].split("=", 1)[1], args[1:]
for path in args:
    a = (sf.read(path, dtype="float32")[0] * 32768).astype(np.int16)
    last = r.xrevrange("audio:mic", count=1)
    audio_id = last[0][0] if last else b"0-0"
    tr_last = (r.xrevrange("events:transcript", count=1) or [[b"0-0"]])[0][0]
    r.xadd("events:wakeword", {"word": word, "score": "1.0", "audio_id": audio_id, **({"rms_thr": "150"} if word == "speech" else {})})
    time.sleep(0.2)
    for i in range(0, len(a) // 480 * 480, 480):
        r.xadd("audio:mic", {"pcm": a[i:i + 480].tobytes()})
        time.sleep(0.03)
    for _ in range(40):
        r.xadd("audio:mic", {"pcm": np.zeros(480, dtype=np.int16).tobytes()})
        time.sleep(0.03)
    resp = r.xread({"events:transcript": tr_last}, count=1, block=15000)
    if not resp:
        print(f"{path}: no transcript (empty text?)", flush=True)
        continue
    f = {k.decode(): v.decode() for k, v in resp[0][1][0][1].items()}
    print(f"{path}: {f.get('speaker')} {f.get('spk_score')} {f.get('spk_ms')}ms pause {f.get('pause_s', '-')} | {f.get('text', '')[:60]}", flush=True)
