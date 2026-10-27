import sys
import time
from pathlib import Path
import redis, numpy as np, yaml
import sherpa_onnx
from voiceid import VoiceID

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.ambient import engaged_until, needs_pause
r = redis.Redis(socket_timeout=None)
P = yaml.safe_load(Path(__file__).with_name("manifest.yaml").read_text(encoding="utf-8"))["params"]
# GigaAM v3 CTC with punctuation and digits (D-004, T-040/T-041); same ORT runtime as voice id
STT_DIR = ROOT / P["stt"]["model"]
model = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=str(STT_DIR / "model.int8.onnx"), tokens=str(STT_DIR / "tokens.txt"),
                                                   num_threads=P["stt"]["num_threads"])
print(f"stt: {STT_DIR.name}, {P['stt']['num_threads']} threads")
voice_id = VoiceID(ROOT, P["voice_id"])
print(f"stt: voice id {voice_id.state}")

RATE = 16000
SILENCE_RMS = 300          # порог тишины для int16, подберём по факту
SILENCE_FRAMES = 30        # ~0.9 с тишины = конец фразы
MAX_FRAMES = 400           # потолок ~12 с, защита от вечной записи
PAUSE_MAX_S = 2.0          # ambient: how long the pause after a phrase is measured (duck mode, T-038)


def segment(start_id, end_id):
    """Frames of a VAD segment (trigger speech, T-038 stage 2b): the detector sends its first and last frame."""
    return [np.frombuffer(f[b'pcm'], dtype=np.int16) for _, f in r.xrange('audio:mic', start_id, end_id)]


def wait_pause(end_ts):
    """Pause after the phrase for the duck pass: until the next start of speech (vad:onset from the detector), at most
    PAUSE_MAX_S after the end of the phrase. The wait is the price: only when the duck needs it (core/ambient.needs_pause)."""
    while True:
        onset = float(r.get('vad:onset') or 0)
        if onset > end_ts:
            return min(onset - end_ts, PAUSE_MAX_S)
        if time.time() >= end_ts + PAUSE_MAX_S:
            return PAUSE_MAX_S
        time.sleep(0.05)


def process(chunks, trigger, ts, end_ts):
    """Transcribe + voice id -> events:transcript. Ambient (end_ts given): pause_s is how long the speaker stayed quiet
    after the phrase — waited for only when the duck pass needs it; otherwise the silence VAD already saw."""
    if not chunks:
        return
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    import soundfile as sf                          # dump for listening
    sf.write('/tmp/stt_last.wav', audio, RATE)
    t0 = time.perf_counter()
    st = model.create_stream()                      # offline CTC: the whole segment at once
    st.accept_waveform(RATE, audio)
    model.decode_stream(st)
    text = st.result.text.strip()
    stt_s = time.perf_counter() - t0
    print(f"stt: transcribe {stt_s:.2f}s for {len(audio)/RATE:.1f}s audio")
    spk = voice_id(chunks)                          # whose voice (T-039); ~40 ms on the voiced frames
    if not text:
        print("stt: empty")
        return
    pause_s, wait = SILENCE_FRAMES * 0.03, ""
    if end_ts is not None:
        if needs_pause(text, spk['speaker'], time.time() < engaged_until(r)):
            t1 = time.perf_counter()
            pause_s = wait_pause(end_ts)
            wait = f"{time.perf_counter() - t1:.2f}"
        else:
            pause_s = min(time.time() - end_ts, PAUSE_MAX_S)
    r.xadd('events:transcript', {'text': text, 'lang': 'ru',     # GigaAM is Russian-only
                                 'ts': f"{ts:.3f}", 'stt_s': f"{stt_s:.3f}", 'trigger': trigger,
                                 'pause_s': f"{pause_s:.2f}", 'pause_wait': wait, **spk})
    print(f"stt: [{trigger} {spk['speaker']} {spk['spk_score']} {spk['spk_ms']}ms pause {pause_s:.1f}s"
          f"{' waited ' + wait + 's' if wait else ''}] {text}")


last_ev = '$'
print("stt: waiting for wake events...")
while True:
    ev = r.xread({'events:wakeword': last_ev}, count=1, block=5000)
    if not ev:
        continue
    ev_id, fields = ev[0][1][0]
    last_ev = ev_id
    audio_id = fields[b'audio_id'].decode()
    trigger = fields.get(b'word', b'').decode()
    if b'end_id' in fields:                         # VAD segment: the phrase is already in audio:mic
        chunks = segment(audio_id, fields[b'end_id'].decode())
        end_ts = float(fields[b'end_ts'])
        ts = time.time()                            # end of phrase as the system knows it (latency chain, T-003)
        process(chunks, trigger, ts, end_ts)
        continue
    print("stt: wake received, recording...")

    chunks, silent, last_id = [], 0, audio_id
    speech_started = False
    START_TIMEOUT = 170            # ~5 c на то, чтобы начать говорить
    while len(chunks) < MAX_FRAMES:
        resp = r.xread({'audio:mic': last_id}, count=10, block=1000)
        if not resp:
            continue
        for msg_id, fields in resp[0][1]:
            last_id = msg_id
            frame = np.frombuffer(fields[b'pcm'], dtype=np.int16)
            chunks.append(frame)
            rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
            if rms >= SILENCE_RMS:
                speech_started = True
                silent = 0
            else:
                silent += 1
        if not speech_started:
            if len(chunks) > START_TIMEOUT:
                chunks = []        # никто так и не заговорил — сброс
                break
            continue               # тишину до речи не считаем концом
        if silent >= SILENCE_FRAMES:
            break

    ts = time.time()                                # end of phrase, for latency chain (T-003)
    if not chunks:                                  # nobody spoke after the wake word
        print("stt: no speech after wake, skipped")
        continue
    process(chunks[:-SILENCE_FRAMES], trigger, ts, None)   # without the silent tail
