"""tts: events:reply -> speech, sentence by sentence (T-003).

main loop (events:reply) -> text queue -> synth thread (Piper into memory, no files) -> PCM queue -> player thread
(soundcard on PulseAudio; PortAudio/sounddevice sees no devices under WSLg). The next sentence is synthesized while
the previous one plays. tts:speaking is set on the first chunk in the player and cleared when nothing is queued,
nothing is being synthesized and the audio handed to the card has had time to play. Text arrives already in speech form
(brain applies core/speech.for_speech per sentence). No sox effects (preset from research/tts/fx.py — later).
"""
import queue
import threading
import time

import redis
import soundcard as sc
from piper import PiperVoice

r = redis.Redis(socket_timeout=None)
VOICE = "/home/valeriy/skull-assistant/models/tts/ru_RU-dmitri-medium.onnx"
voice = PiperVoice.load(VOICE)
RATE = voice.config.sample_rate

texts = queue.Queue()        # (reply_id, seq, text, ts)
pcm = queue.Queue()          # (reply_id, seq, float32 mono chunk, ts)
pending = 0                  # sentences taken by the main loop and not yet fully synthesized
lock = threading.Lock()


def synth():
    global pending
    while True:
        reply_id, seq, text, ts = texts.get()
        try:
            for chunk in voice.synthesize(text):
                pcm.put((reply_id, seq, chunk.audio_float_array, ts))
        except Exception as e:      # one bad sentence must not stop the voice
            print(f"tts: synth failed on {text[:40]!r}: {e}", flush=True)
        finally:
            with lock:
                pending -= 1


def play():
    speaking, first, last, until = False, set(), None, 0.0   # until: when the audio handed to the card ends
    with sc.default_speaker().player(samplerate=RATE, channels=1) as player:
        while True:
            try:
                reply_id, seq, audio, ts = pcm.get(timeout=0.05)
            except queue.Empty:
                with lock:
                    idle = pending == 0 and pcm.empty()
                if speaking and idle and time.time() >= until:   # play() returns once the data is buffered
                    r.delete('tts:speaking')
                    speaking = False
                    print(f"tts: quiet after {last}", flush=True)   # inside a reply this is an audible gap
                continue
            if not speaking:
                r.set('tts:speaking', 1)
                speaking = True
            if reply_id not in first:
                first.add(reply_id)
                if ts:
                    print(f"tts: first audio +{time.time() - ts:.2f}s", flush=True)
            until = max(until, time.time()) + len(audio) / RATE
            player.play(audio)      # returns when PulseAudio has buffered the chunk, not when it has been heard
            last = f"{reply_id} #{seq}"


threading.Thread(target=synth, daemon=True).start()
threading.Thread(target=play, daemon=True).start()

last_ev = '$'
print(f"speaker: dmitri ready ({RATE} Hz), waiting for replies...", flush=True)
while True:
    ev = r.xread({'events:reply': last_ev}, count=10, block=5000)
    if not ev:
        continue
    for ev_id, fields in ev[0][1]:
        last_ev = ev_id
        text = fields[b'text'].decode().strip()
        if not text:                 # e.g. the empty final message that only closes a reply
            continue
        reply_id = fields.get(b'reply_id', ev_id).decode()   # a message without reply_id is a reply of its own
        seq = int(fields.get(b'seq', b'0'))
        ts = float(fields[b'ts']) if b'ts' in fields else None
        print(f"speaker: [{reply_id} #{seq}] {text[:60]}", flush=True)
        with lock:
            pending += 1
        texts.put((reply_id, seq, text, ts))
