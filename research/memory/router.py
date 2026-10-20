"""Turn-type router (docs/COGNITION_4B.md E-3, docs/MEMORY_TRACK.md).

One extra call of the same model per turn: "what kind of turn is this?", answered with a single label.
Through ollama the label is generated (<=4 tokens, think off) — a surrogate for the decide pass;
through llama-server the same prompt can be scored with top_logprobs (route_llamaserver) to get a
distribution instead of a word. The router question goes into the LAST user message so the KV prefix
(persona + history) stays cached; the question itself is never kept in the history.
"""
import re
import time

import requests

LABELS = ("chat", "memory", "calc", "explain", "world", "command")

ROUTER_PROMPT = """Служебный вопрос, не отвечай собеседнику. Определи тип его последней реплики и ответь ОДНИМ словом из списка:
chat — болтовня, замечание, рассказ о своих делах, реакция на тебя;
memory — спрашивает о том, что уже говорилось в этом разговоре, или о том, что ты сам раньше сказал или считал;
calc — просит посчитать: числа, деньги, проценты, время;
explain — просит подробно рассказать или объяснить что-то общее (как устроено, как работает);
world — спрашивает о фактах вне этой комнаты и этого разговора: новости, наука, погода, люди, приборы;
command — просит запомнить, напомнить, сделать что-то или изменить твоё поведение.
Реплика собеседника: «{text}»
Тип (одно слово):"""

_LABEL_RE = re.compile("|".join(LABELS), re.I)
# e2b-it-qat answers with the description instead of the label ("реакция на тебя", "Рассказ о своих делах"):
# map the description phrases back to labels (first match wins; checked after the English labels)
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


def route_ollama(host, model, messages, text, num_ctx, timeout=120):
    """messages: persona + history (without the current turn). Returns dict(label, raw, s)."""
    body = {"model": model, "stream": False, "think": False, "keep_alive": "10m",
            "messages": messages + [{"role": "user", "content": ROUTER_PROMPT.format(text=text)}],
            "options": {"num_ctx": num_ctx, "temperature": 0.0, "seed": 1, "num_predict": 6}}
    t0 = time.perf_counter()
    r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    if r.status_code == 400 and "think" in r.text.lower():
        body.pop("think")
        r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    raw = (r.json()["message"].get("content") or "").strip()
    return {"label": parse_label(raw), "raw": raw[:40], "s": round(time.perf_counter() - t0, 3)}


def route_llamaserver(host, messages, text, timeout=120):
    """Same question through llama-server /v1/chat/completions with logprobs: returns label + distribution
    over LABELS from top_logprobs of the first generated token (decide pass, P-018)."""
    body = {"messages": messages + [{"role": "user", "content": ROUTER_PROMPT.format(text=text)}],
            "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 10,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    r = requests.post(f"{host}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    ch = d["choices"][0]
    raw = (ch["message"].get("content") or "").strip()
    probs = {}
    try:
        for alt in ch["logprobs"]["content"][0]["top_logprobs"]:
            lab = parse_label(alt["token"])
            if lab:
                probs[lab] = probs.get(lab, 0.0) + float(__import__("math").exp(alt["logprob"]))
    except (KeyError, IndexError, TypeError):
        pass
    label = max(probs, key=probs.get) if probs else parse_label(raw)
    return {"label": label, "raw": raw[:40], "probs": {k: round(v, 3) for k, v in probs.items()},
            "s": round(time.perf_counter() - t0, 3)}


# what each label changes in the main call (used by ctx_probe --router apply)
POLICY = {
    "chat":    {"num_predict": 300, "memory": False},
    "memory":  {"num_predict": 300, "memory": True},
    "calc":    {"num_predict": 500, "memory": True, "prefix": "Считай по шагам, все числа пиши цифрами, итог — в конце."},
    "explain": {"num_predict": 900, "memory": False},
    "world":   {"num_predict": 300, "memory": False, "prefix": "Если ты этого не знаешь или не видел — так и скажи одной фразой, не выдумывай."},
    "command": {"num_predict": 200, "memory": True},
}
