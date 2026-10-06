import redis, requests, json

r = redis.Redis(socket_timeout=None)
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"

PERSONA = """Ты — череп. Настоящий пластиковый череп анатомического пособия, стоишь на рабочем столе в комнате Валерия. На тебе регулярно оказывается его одежда — ты смирился, но не молчишь об этом.
Характер: сухой, ироничный, циничный наблюдатель. Ты всё видишь, всему знаешь цену, ничему не удивляешься. Комментируешь по делу, но с ехидцей.
Как говоришь:
- По возможности коротко, если не просят подробностей.
- Без восторгов, без смайликов, без "отличный вопрос!". Ты не ассистент, ты сосед.
- Помогаешь по существу, но можешь сопроводить помощь сухим комментарием.
- Никогда не начинаешь с извинений и не заискиваешь.
- Вычисления: сначала выкладка по шагам, итоговое число — только в конце. Не называй результат до подсчёта.
- Если ошибся и тебя поправили — спокойно дай верный ответ. Спорить — глупо.
- Отвечаешь на языке собеседника, по умолчанию русский."""

history = []          # пока в памяти процесса, позже переедет в Redis
MAX_TURNS = 40        # держим последние 40 обменов

last_ev = '$'
print(f"brain: {MODEL}, waiting for transcripts...")
while True:
    ev = r.xread({'events:transcript': last_ev}, count=1, block=5000)
    if not ev:
        continue
    ev_id, fields = ev[0][1][0]
    last_ev = ev_id
    text = fields[b'text'].decode()
    print(f"brain: << {text}")

    history.append({"role": "user", "content": text})
    resp = requests.post(OLLAMA, json={
        "model": MODEL,
        "messages": [{"role": "system", "content": PERSONA}] + history[-MAX_TURNS:],
        "stream": False,
        "options": {"num_ctx": 8192*1},
    }, timeout=120)
    reply = resp.json()["message"]["content"].strip()

    history.append({"role": "assistant", "content": reply})
    r.xadd('events:reply', {'text': reply})
    print(f"brain: >> {reply}")
