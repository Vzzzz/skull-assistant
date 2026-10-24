import json, os, sys, time
from pathlib import Path
import redis, requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.memory import clean_reply

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

HISTORY = "dialog:history"   # Redis list, one JSON element per exchange: turn, ts, user, reply, raw, tokens
SYS_CALIB = "dialog:sys_tokens"   # {"chars", "tokens"}: measured system prompt size, survives restarts
NUM_CTX = int(os.environ.get("BRAIN_NUM_CTX", 8192))   # T-016: from the manifest
MAX_TOKENS = 700              # reply budget, reserved in the window
NOTE_RESERVE = 250            # room for the memory note (T-016), reserved in the window
CHARS_PER_TOKEN = 2.3         # Cyrillic estimate for text not yet seen by the model


def est_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN)


def build_window(history, sys_tokens, est_new):
    """Newest exchanges that fit into NUM_CTX together with the system prompt, the new turn and the reserves.
    Older exchanges stay in Redis (source for the memory beyond the window, T-016)."""
    window, used = [], sys_tokens
    for ex in reversed(history):
        if used + ex["tokens"] + est_new + NOTE_RESERVE + MAX_TOKENS > NUM_CTX:
            break
        window.insert(0, ex)
        used += ex["tokens"]
    return window, used


calib = json.loads(r.get(SYS_CALIB) or "{}")
# measured size if the prompt is unchanged, else the estimate until a call with an empty window calibrates it
sys_tokens = calib["tokens"] if calib.get("chars") == len(PERSONA) else est_tokens(PERSONA)
last_ev = '$'
print(f"brain: {MODEL} num_ctx={NUM_CTX} system~{sys_tokens} tokens, {r.llen(HISTORY)} exchanges in {HISTORY}, waiting for transcripts...")
while True:
    ev = r.xread({'events:transcript': last_ev}, count=1, block=5000)
    if not ev:
        continue
    ev_id, fields = ev[0][1][0]
    last_ev = ev_id
    text = fields[b'text'].decode()
    print(f"brain: << {text}")
    if text.strip() == "/reset":
        r.delete(HISTORY)
        print("brain: history reset")
        continue

    history = [json.loads(x) for x in r.lrange(HISTORY, 0, -1)]   # Redis is the only copy
    est_new = est_tokens(text) + 8
    window, used = build_window(history, sys_tokens, est_new)
    print(f"brain: window {len(window)} exchanges, ~{used + est_new} tokens ({len(history)} stored)")
    messages = [{"role": "system", "content": PERSONA}]
    for ex in window:
        messages += [{"role": "user", "content": ex["user"]}, {"role": "assistant", "content": ex["reply"]}]
    messages.append({"role": "user", "content": text})
    resp = requests.post(OLLAMA, json={
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "num_predict": MAX_TOKENS},
    }, timeout=120)
    data = resp.json()
    raw = data["message"]["content"].strip()
    reply, flags = clean_reply(raw)
    if flags:
        print(f"brain: filter {sorted(flags)}, raw: {raw!r}")

    # token ledger: this exchange's share of later prompts = its user turn (prompt minus what was already there)
    # plus its reply; with an empty window the system prompt size is calibrated from the estimate of the turn
    pe, ec = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
    if not window and pe:
        sys_tokens = max(0, pe - est_new)
        r.set(SYS_CALIB, json.dumps({"chars": len(PERSONA), "tokens": sys_tokens}))
    in_window = sum(ex["tokens"] for ex in window)
    tokens = max(1, pe - sys_tokens - in_window) + ec if pe else est_new + est_tokens(raw)
    turn = history[-1]["turn"] + 1 if history else 1
    ts = float(fields[b'ts']) if b'ts' in fields else time.time()
    r.rpush(HISTORY, json.dumps({"turn": turn, "ts": ts, "user": text, "reply": reply, "raw": raw,
                                 "tokens": tokens}, ensure_ascii=False))
    r.xadd('events:reply', {'text': reply})
    print(f"brain: >> {reply}")
    print(f"brain: turn {turn} prompt={pe} reply={ec} exchange~{tokens} tokens")
