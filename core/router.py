"""Turn-type router (docs/COGNITION_4B.md E-3, D-026). Ported from research/memory/router.py.

One extra call of the same model per turn scored with top_logprobs of the first token (decide pass, P-018).
The router question goes into the LAST user message so the KV prefix (persona + history) stays cached;
the question itself is never kept in the history.
"""
import math
import re
import time

import requests

LABELS = ("chat", "memory", "calc", "explain", "world", "command")

ROUTER_PROMPT = """Служебный вопрос, не отвечай собеседнику. Определи тип его последней реплики и ответь ОДНИМ словом из списка:
chat — болтовня, замечание, рассказ о своих делах, реакция на тебя;
memory — спрашивает о том, что собеседник говорил тебе раньше (его кот, его ключи, его встречи, его планы), или о том, что ты сам раньше сказал или считал — даже если этого нет в видимой части разговора;
calc — просит посчитать: числа, деньги, проценты, время;
explain — просит подробно рассказать или объяснить что-то общее (как устроено, как работает);
world — спрашивает о фактах, которых собеседник тебе не сообщал и которые не касаются его жизни: новости, наука, погода, незнакомые люди, приборы;
command — просит запомнить, напомнить, сделать что-то или изменить твоё поведение.
Реплика собеседника: «{text}»
Тип (одно слово):"""

_LABEL_RE = re.compile("|".join(LABELS), re.I)
# e2b-it-qat answers with the description instead of the label: map the description phrases back to labels
_DESC = [("chat", ("болтовн", "замечан", "рассказ о сво", "реакция на тебя")),
         ("memory", ("уже говорил", "сам раньше", "в этом разговоре")),
         ("calc", ("посчит", "числа", "деньги", "процент")),
         ("explain", ("подробно", "объясн", "как устроен")),
         ("world", ("вне этой комнаты", "новости", "наука", "погода", "прибор")),
         ("command", ("запомнить", "напомнить", "сделать что", "изменить тво"))]


def parse_label(raw: str):
    m = _LABEL_RE.search(raw or "")
    if m:
        return m.group(0).lower()
    low = (raw or "").lower()
    for lab, keys in _DESC:
        if any(k in low for k in keys):
            return lab
    return None


def route_llamaserver(host, messages, text, timeout=120):
    """messages: persona + history (without the current turn). Returns dict(label, raw, probs, s):
    label + distribution over LABELS from top_logprobs of the first generated token."""
    body = {"messages": messages + [{"role": "user", "content": ROUTER_PROMPT.format(text=text)}],
            "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 10,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    r = requests.post(f"{host}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    raw = (ch["message"].get("content") or "").strip()
    probs = {}
    try:
        for alt in ch["logprobs"]["content"][0]["top_logprobs"]:
            lab = parse_label(alt["token"])
            if lab:
                probs[lab] = probs.get(lab, 0.0) + math.exp(alt["logprob"])
    except (KeyError, IndexError, TypeError):
        pass
    label = max(probs, key=probs.get) if probs else parse_label(raw)
    return {"label": label, "raw": raw[:40], "probs": {k: round(v, 3) for k, v in probs.items()},
            "s": round(time.perf_counter() - t0, 3)}


# what each label changes in the main call: reply budget, memory retrieval, instruction prefix
POLICY = {
    "chat":    {"max_tokens": 300, "memory": False},
    "memory":  {"max_tokens": 300, "memory": True},
    "calc":    {"max_tokens": 500, "memory": False, "prefix": "Считай по шагам, все числа пиши цифрами, итог — в конце."},
    "explain": {"max_tokens": 900, "memory": False},
    # world keeps memory on: at 2k the router labelled "как зовут моего кота?" as world when the cat had fallen out of
    # the window (T-028: 4 probes -> "Не знаю."); with retrieval the note arrives and the hedge applies only if memory is empty
    "world":   {"max_tokens": 300, "memory": True,
                "prefix": "Если собеседник спрашивает о том, чего ты не видел, не знаешь и не помнишь, — скажи, что не знаешь, одной фразой в своём тоне, не выдумывай. Если он ничего не спрашивает — отвечай как обычно."},
    "command": {"max_tokens": 200, "memory": True},
}
