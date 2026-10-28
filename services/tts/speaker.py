"""tts: events:reply -> speech, sentence by sentence (T-003).

main loop (events:reply) -> text queue -> synth thread (Piper into memory, no files) -> PCM queue -> player thread
(soundcard on PulseAudio; PortAudio/sounddevice sees no devices under WSLg). The next sentence is synthesized while
the previous one plays. tts:speaking is set on the first chunk in the player and cleared when nothing is queued,
nothing is being synthesized and the audio handed to the card has had time to play. Text arrives already in speech form
(brain applies core/speech.for_speech per sentence). The skull's timbre (sox pitch/reverb, as before T-003) is applied
per sentence through a pipe on raw PCM, no files; FX = [] turns it off (preset from research/tts/fx.py — later).
Ambient (T-038 stage 2b): tts:speaking stays up TAIL_S after the sound (room echo of the last word), and the expected
end of each reply's playback goes to the hash tts:end (core/ambient.py counts the ENGAGED window from it).
"""
import queue
import subprocess
import threading
import time

import numpy as np
import redis
import soundcard as sc
from piper import PiperVoice

r = redis.Redis(socket_timeout=None)
VOICE = "/home/valeriy/skull-assistant/models/tts/ru_RU-dmitri-medium.onnx"
voice = PiperVoice.load(VOICE)
RATE = voice.config.sample_rate
FX = ["pitch", "-300", "reverb", "30", "50", "100"]   # the voice before T-003 (sox on files)
BLOCK = RATE // 5                                    # 0.2-s pieces for the player
TAIL_S = 1.0                                         # tts:speaking after the sound: the echo is not a new phrase
                                                     # (0.5 let «…что хочешь» through as a new segment, 2в music)

texts = queue.Queue()        # (reply_id, seq, text, ts)
pcm = queue.Queue()          # (reply_id, seq, float32 mono chunk, ts)
pending = 0                  # sentences taken by the main loop and not yet fully synthesized
lock = threading.Lock()


def fx(audio):
    """sox on raw int16 mono through stdin/stdout; returns (int16 array, seconds spent)."""
    t0 = time.perf_counter()
    out = subprocess.run(["sox", "-t", "raw", "-r", str(RATE), "-e", "signed", "-b", "16", "-c", "1", "-",
                          "-t", "raw", "-r", str(RATE), "-e", "signed", "-b", "16", "-c", "1", "-", *FX],
                         input=audio.tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16), time.perf_counter() - t0


def synth():
    global pending
    while True:
        reply_id, seq, text, ts = texts.get()
        if text is None:            # end of a reply: passes the player in order, after its last chunk
            pcm.put((reply_id, seq, None, ts))
            continue
        try:
            t0 = time.perf_counter()
            audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text)])
            print(f"tts: synth {(time.perf_counter() - t0) * 1000:.0f} ms for {len(audio) / RATE:.1f} s (#{seq})", flush=True)
            if FX:
                audio, fx_s = fx(audio)
                print(f"tts: sox {fx_s * 1000:.0f} ms for {len(audio) / RATE:.1f} s (#{seq})", flush=True)
            audio = audio.astype(np.float32) / 32768.0
            for i in range(0, len(audio), BLOCK):
                pcm.put((reply_id, seq, audio[i:i + BLOCK], ts))
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
                if speaking and idle and time.time() >= until + TAIL_S:   # play() returns once the data is buffered
                    r.delete('tts:speaking')
                    speaking = False
                    print(f"tts: quiet after {last}", flush=True)   # inside a reply this is an audible gap
                continue
            if audio is None:       # the reply's last chunk has been handed to the card: it ends at `until`
                r.hset('tts:end', reply_id, f"{max(until, time.time()):.3f}")
                r.expire('tts:end', 3600)
                continue
            if not speaking:
                r.set('tts:speaking', 1)
                speaking = True
            if reply_id not in first:
                first.add(reply_id)
                if ts:
                    print(f"tts: first audio +{time.time() - ts:.2f}s", flush=True)
            elif speaking and time.time() - until > 0.03:   # the card ran dry inside a reply: audible gap
                print(f"tts: gap {(time.time() - until) * 1000:.0f} ms before {reply_id} #{seq} "
                      f"(chunk {len(audio) / RATE:.2f} s, queued {pcm.qsize()})", flush=True)
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
        reply_id = fields.get(b'reply_id', ev_id).decode()   # a message without reply_id is a reply of its own
        seq = int(fields.get(b'seq', b'0'))
        ts = float(fields[b'ts']) if b'ts' in fields else None
        if text:                     # the final message may be empty: it only closes the reply
            print(f"speaker: [{reply_id} #{seq}] {text[:60]}", flush=True)
            with lock:
                pending += 1
            texts.put((reply_id, seq, text, ts))
        if fields.get(b'final') == b'1' or b'reply_id' not in fields:
            texts.put((reply_id, seq, None, ts))
