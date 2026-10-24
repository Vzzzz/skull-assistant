"""Turn-type router (docs/COGNITION_4B.md E-3, D-026). Ported from research/memory/router.py.

One extra call of the same model per turn scored with top_logprobs of the first token (decide pass, P-018).
The router question goes into the LAST user message so the KV prefix (persona + history) stays cached;
the question itself is never kept in the history.
"""
import math
import re
import time

import requests

LABELS = ("chat", "memory", "calc", "explain", "world", "command", "aside")

ROUTER_PROMPT = """Служебный вопрос, не отвечай собеседнику. Определи тип его последней реплики и ответь ОДНИМ словом из списка:
chat — болтовня, замечание, рассказ о своих делах, реакция на тебя;
memory — спрашивает о том, что собеседник говорил тебе раньше (его кот, его ключи, его встречи, его планы), или о том, что ты сам раньше сказал или считал — даже если этого нет в видимой части разговора;
calc — просит посчитать: числа, деньги, проценты, время;
explain — просит подробно рассказать или объяснить что-то общее (как устроено, как работает);
world — спрашивает о фактах, которых собеседник тебе не сообщал и которые не касаются его жизни: новости, наука, погода, незнакомые люди, приборы;
command — просит запомнить, напомнить, сделать что-то или изменить твоё поведение;
aside — сказано не тебе: бормочет себе под нос, говорит по телефону или с другим человеком в комнате, читает вслух, поёт; речь из телевизора/радио/подкаста/созвона.
Реплика собеседника: «{text}»
Тип (одно слово):"""

_LABEL_RE = re.compile("|".join(LABELS), re.I)
# e2b-it-qat answers with the description instead of the label: map the description phrases back to labels
_DESC = [("chat", ("болтовн", "замечан", "рассказ о сво", "реакция на тебя")),
         ("memory", ("уже говорил", "сам раньше", "в этом разговоре")),
         ("calc", ("посчит", "числа", "деньги", "процент")),
         ("explain", ("подробно", "объясн", "как устроен")),
         ("world", ("вне этой комнаты", "новости", "наука", "погода", "прибор")),
         ("command", ("запомнить", "напомнить", "сделать что", "изменить тво")),
         ("aside", ("не тебе", "под нос", "по телефону", "с другим человеком", "читает вслух"))]


def parse_label(raw: str):
    m = _LABEL_RE.search(raw or "")
    if m:
        return m.group(0).lower()
    low = (raw or "").lower()
    for lab, keys in _DESC:
        if any(k in low for k in keys):
            return lab
    return None


def _probs(top_logprobs):
    """Distribution over LABELS from the first generated token's alternatives."""
    probs = {}
    for alt in top_logprobs:
        lab = parse_label(alt["token"])
        if lab:
            probs[lab] = probs.get(lab, 0.0) + math.exp(alt["logprob"])
    return probs


def route_ollama(host, model, messages, text, num_ctx, timeout=120):
    """Same question through ollama /api/chat (research: models that only ollama runs, e.g. gemma4:e2b-it-qat);
    logprobs of the first token when the server returns them, else the generated word."""
    body = {"model": model, "stream": False, "think": False, "keep_alive": "10m", "logprobs": True, "top_logprobs": 10,
            "messages": messages + [{"role": "user", "content": ROUTER_PROMPT.format(text=text)}],
            "options": {"num_ctx": num_ctx, "temperature": 0.0, "seed": 1, "num_predict": 6}}
    t0 = time.perf_counter()
    r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    if r.status_code == 400 and "think" in r.text.lower():
        body.pop("think")
        r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    raw = (d["message"].get("content") or "").strip()
    try:
        probs = _probs(d["logprobs"][0]["top_logprobs"])
    except (KeyError, IndexError, TypeError):
        probs = {}
    if sum(probs.values()) < 0.5:
        probs = {}   # the first token is not a label (e2b answers with the description: «реакция на тебя»)
    label = max(probs, key=probs.get) if probs else parse_label(raw)
    return {"label": label, "raw": raw[:40], "probs": {k: round(v, 3) for k, v in probs.items()},
            "s": round(time.perf_counter() - t0, 3)}


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
    try:
        probs = _probs(ch["logprobs"]["content"][0]["top_logprobs"])
    except (KeyError, IndexError, TypeError):
        probs = {}
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
    # the window (T-028: 4 probes -> "Не знаю."); with retrieval the note arrives and the hedge applies only if memory is empty.
    # The hedge is narrowed to what the model cannot know (owner, 2026-10-24): the router also labels general knowledge
    # as world ("Почему небо голубое?" 0.81), and the old wording turned it into "Извини, я не знаю."
    "world":   {"max_tokens": 300, "memory": True,
                "prefix": "Если это общеизвестный факт или общие знания (наука, история, как устроен мир) — ответь как обычно. "
                          "Если собеседник спрашивает о том, чего ты не можешь знать или о чём не слышал (сегодняшние новости и погода, "
                          "конкретный человек, прибор или событие, о котором ты не знаешь), — скажи, что не знаешь, одной фразой в своём тоне, "
                          "не выдумывай. Если он ничего не спрашивает — отвечай как обычно."},
    "command": {"max_tokens": 200, "memory": True},
    # said to someone else (T-038, ambient-lite): no reply, nothing stored. max_tokens only keeps services/brain
    # working until it honours `respond` (T-038 stage 2)
    "aside":   {"max_tokens": 300, "memory": False, "respond": False},
}


# guards on the router's answer (T-037): a label is taken only when the model is confident in it, and `calc`
# only when the turn has a number — «Как меня зовут?» right after a multiplication came out calc 0.42 / chat 0.27,
# and the calc prefix made the model redo the previous sum before answering
MIN_PROB, MIN_MARGIN = 0.45, 1.3
# aside (T-038): staying silent when addressed is worse than answering a remark not meant for the skull — the label
# needs p >= ASIDE_MIN_PROB, and is never taken when the turn has the skull's name in it
ASIDE_MIN_PROB = 0.6
_NUMBER = re.compile(
    r"\d|\b(?:ноль|нул\w*|один|одн\w*|два|две|двух|двум\w*|три|тр[её]х|четыр\w*|пят\w*|шест\w*|семь|семи|семер\w*|семьдесят\w*|восем\w*|"
    r"восьм\w*|девят\w*|десят\w*|\w+надцат\w*|двадцат\w*|тридцат\w*|сорок\w*|девяност\w*|сто|ста|сотн\w*|\w+сот|"
    r"тысяч\w*|миллион\w*|процент\w*|половин\w*|четверт\w*|полтор\w*|дюжин\w*)\b", re.I)


def has_number(text: str) -> bool:
    return bool(_NUMBER.search(text.replace("ё", "е").replace("Ё", "Е")))


def decide(route, text):
    """Final label after the guards: returns (label, reason) where reason is None when the router's label stands,
    'name' / 'aside-low' (aside with the name «Костян» in the turn / aside below ASIDE_MIN_PROB or without a distribution),
    'low-confidence' (top prob < MIN_PROB or < MIN_MARGIN x second) or 'calc-no-number'; all fall back to chat."""
    label, probs = route.get("label"), route.get("probs") or {}
    if label == "aside":
        if "костян" in text.lower():
            return "chat", "name"
        if probs.get("aside", 0.0) < ASIDE_MIN_PROB:
            return "chat", "aside-low"
    if label and probs:
        top = sorted(probs.values(), reverse=True) + [0.0]
        if top[0] < MIN_PROB or top[0] < MIN_MARGIN * top[1]:
            return "chat", "low-confidence"
    if label == "calc" and not has_number(text):
        return "chat", "calc-no-number"
    return label, None
