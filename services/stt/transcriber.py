import redis, numpy as np
from faster_whisper import WhisperModel

r = redis.Redis(socket_timeout=None)
model = WhisperModel("base", device="cpu", compute_type="int8")

RATE = 16000
SILENCE_RMS = 300          # порог тишины для int16, подберём по факту
SILENCE_FRAMES = 30        # ~0.9 с тишины = конец фразы
MAX_FRAMES = 400           # потолок ~12 с, защита от вечной записи

last_ev = '$'
print("stt: waiting for wake events...")
while True:
    ev = r.xread({'events:wakeword': last_ev}, count=1, block=5000)
    if not ev:
        continue
    ev_id, fields = ev[0][1][0]
    last_ev = ev_id
    audio_id = fields[b'audio_id'].decode()
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

    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    audio = audio[:-SILENCE_FRAMES * 480]          # 1) срезать хвост тишины
    import soundfile as sf                          # 2) дамп для прослушки
    sf.write('/tmp/stt_last.wav', audio, RATE)
    segments, info = model.transcribe(
        audio, language=None, vad_filter=True,
        condition_on_previous_text=False,           # 3) профилактика галлюцинаций
    )
    segments, info = model.transcribe(audio, language=None, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    if text:
        r.xadd('events:transcript', {'text': text, 'lang': info.language})
        print(f"stt: [{info.language}] {text}")
    else:
        print("stt: empty")
