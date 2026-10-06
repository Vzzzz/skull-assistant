import soundcard as sc, redis, numpy as np, time

r = redis.Redis()
RATE, CHUNK = 16000, 480              # 30 мс кадры
STREAM, MAXLEN = 'audio:mic', 2000    # кольцевой буфер ~60 c
mic = sc.default_microphone()

n, t0 = 0, time.time()
with mic.recorder(samplerate=RATE, channels=1) as rec:
    print(f"capture: {mic.name} -> {STREAM}")
    while True:
        data = rec.record(numframes=CHUNK)
        pcm = (data[:, 0] * 32767).astype(np.int16).tobytes()
        r.xadd(STREAM, {'pcm': pcm, 'rate': RATE}, maxlen=MAXLEN, approximate=True)
        n += 1
        if n % 100 == 0:
            print(f"{n} frames, {n*0.03/(time.time()-t0):.2f}x realtime")
