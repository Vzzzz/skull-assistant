import redis, numpy as np
from openwakeword.model import Model

r = redis.Redis()
# model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

MODELS = "/home/valeriy/skull-assistant/models/wakeword"
model = Model(
    wakeword_models=[f"{MODELS}/hey_jarvis_v0.1.onnx"],
    melspec_model_path=f"{MODELS}/melspectrogram.onnx",
    embedding_model_path=f"{MODELS}/embedding_model.onnx",
    inference_framework="onnx",
)

THRESHOLD, COOLDOWN_FRAMES = 0.5, 40        # ~1.2 с тишины после срабатывания
last_id, cooldown = '$', 0

print("listening for 'hey jarvis'...")
while True:
    resp = r.xread({'audio:mic': last_id}, count=10, block=1000)
    if not resp:
        continue
    for msg_id, fields in resp[0][1]:
        last_id = msg_id
        pcm = np.frombuffer(fields[b'pcm'], dtype=np.int16)
        # score = model.predict(pcm)['hey_jarvis']
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
            print(f"WAKE! score={score:.3f}")
            cooldown = COOLDOWN_FRAMES
