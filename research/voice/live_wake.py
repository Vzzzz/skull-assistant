"""Live voice-id check without the wake word (T-039): issue a fake events:wakeword at the current end of audio:mic,
so skull-stt records whatever sounds next (podcast from a phone, Piper through the speakers), and print the speaker
fields of the transcript. --play WAV plays a file through the default sink right after the wake (Piper self-echo).

  .venv/bin/python research/voice/live_wake.py --n 10 --gap 6          # e.g. a podcast playing in the room
  .venv/bin/python research/voice/live_wake.py --n 1 --play fx00.wav   # Piper phrase, tts:speaking not set
"""
import argparse
import subprocess
import time

import redis

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--gap", type=float, default=6.0)
ap.add_argument("--play", nargs="*", default=[])
a = ap.parse_args()
r = redis.Redis(socket_timeout=None)
for i in range(a.n):
    last = r.xrevrange("audio:mic", count=1)
    tr_last = (r.xrevrange("events:transcript", count=1) or [[b"0-0"]])[0][0]
    r.xadd("events:wakeword", {"word": "live-check", "score": "1.0", "audio_id": last[0][0] if last else b"0-0"})
    if a.play:
        subprocess.Popen(["paplay", a.play[i % len(a.play)]])
    resp = r.xread({"events:transcript": tr_last}, count=1, block=int((a.gap + 10) * 1000))
    if resp:
        f = {k.decode(): v.decode() for k, v in resp[0][1][0][1].items()}
        print(f"{i + 1:2d} {f.get('speaker')} {f.get('spk_score')} {f.get('spk_ms')}ms | {f.get('text', '')[:70]}", flush=True)
    else:
        print(f"{i + 1:2d} no transcript", flush=True)
    time.sleep(a.gap if not resp else 1.0)
