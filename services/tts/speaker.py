import redis, subprocess, sys, wave, io
from pathlib import Path
from piper import PiperVoice

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.speech import for_speech

r = redis.Redis(socket_timeout=None)
VOICE = "/home/valeriy/skull-assistant/models/tts/ru_RU-dmitri-medium.onnx"
voice = PiperVoice.load(VOICE)

def speak(text: str):
    r.set('tts:speaking', 1)
    try:
        with wave.open('/tmp/tts_out.wav', 'wb') as w:
            voice.synthesize_wav(text, w)
        subprocess.run(['sox', '/tmp/tts_out.wav', '/tmp/tts_fx.wav',
                        'pitch', '-300', 'reverb', '30', '50', '100'])
        subprocess.run(['paplay', '/tmp/tts_fx.wav'])
    finally:
        r.delete('tts:speaking')

last_ev = '$'
print("speaker: dmitri ready, waiting for replies...")
while True:
    ev = r.xread({'events:reply': last_ev}, count=1, block=5000)
    if not ev:
        continue
    ev_id, fields = ev[0][1][0]
    print(f"speaker: got reply: {fields[b'text'][:50]}")
    last_ev = ev_id
    speak(for_speech(fields[b'text'].decode()))   # markdown/LaTeX/emoji are not pronounced
