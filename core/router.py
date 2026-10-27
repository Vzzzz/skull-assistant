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


# duck mode (T-038 stage 1b): the owner is alone at the desk thinking aloud — every turn is "to the skull", the question
# is whether to speak now. A separate pass, not an 8th label; the pause after the turn is part of the question
# (in the live system it comes from VAD, P-030)
DUCK_LABELS = ("respond", "ack", "listen")
DUCK_PROMPT = """Служебный вопрос, не отвечай собеседнику. Собеседник один, думает вслух за работой, ты рядом. Его реплика: «{text}». После неё пауза {pause_s} с. Что сделать:
respond — ответить по существу (вопрос, просьба, обращение, законченная мысль, на которую ждут реакции);
ack — коротко показать, что слушаешь;
listen — молчать (середина мысли, счёт, чтение вслух, обращение к коту, междометие).
Одно слово:"""
_DUCK_RE = re.compile("|".join(DUCK_LABELS), re.I)
# imperative addressed to «ты» that asks the skull for something (closed list: «слезь», «брысь» to the cat must stay listen)
_IMPERATIVE = re.compile(r"\b(?:скажи|расскажи|подскажи|посчитай|прикинь|напомни|запомни|объясни|ответь|глянь|посмотри|"
                         r"послушай|слушай|проверь|найди|помоги|угадай|представь|давай|включи|выключи|"
                         r"умножь|раздели|сложи|вычти|переведи|сравни)\b", re.I)


def _duck_probs(top_logprobs):
    probs = {}
    for alt in top_logprobs:
        m = _DUCK_RE.search(alt["token"])
        if m:
            lab = m.group(0).lower()
            probs[lab] = probs.get(lab, 0.0) + math.exp(alt["logprob"])
    return probs


def _duck_result(raw, probs, t0):
    if sum(probs.values()) < 0.5:
        probs = {}
    m = _DUCK_RE.search(raw or "")
    label = max(probs, key=probs.get) if probs else (m.group(0).lower() if m else None)
    return {"label": label, "raw": (raw or "")[:40], "probs": {k: round(v, 3) for k, v in probs.items()},
            "s": round(time.perf_counter() - t0, 3)}


def duck_llamaserver(host, messages, text, pause_s, timeout=120):
    body = {"messages": messages + [{"role": "user", "content": DUCK_PROMPT.format(text=text, pause_s=pause_s)}],
            "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 10,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    r = requests.post(f"{host}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    try:
        probs = _duck_probs(ch["logprobs"]["content"][0]["top_logprobs"])
    except (KeyError, IndexError, TypeError):
        probs = {}
    return _duck_result((ch["message"].get("content") or "").strip(), probs, t0)


def duck_ollama(host, model, messages, text, pause_s, num_ctx, timeout=120):
    body = {"model": model, "stream": False, "think": False, "keep_alive": "10m", "logprobs": True, "top_logprobs": 10,
            "messages": messages + [{"role": "user", "content": DUCK_PROMPT.format(text=text, pause_s=pause_s)}],
            "options": {"num_ctx": num_ctx, "temperature": 0.0, "seed": 1, "num_predict": 6}}
    t0 = time.perf_counter()
    r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    try:
        probs = _duck_probs(d["logprobs"][0]["top_logprobs"])
    except (KeyError, IndexError, TypeError):
        probs = {}
    return _duck_result((d["message"].get("content") or "").strip(), probs, t0)


def decide_duck(route, text):
    """Final duck label: (label, reason). The skull's name -> respond ('name'); a question mark or an imperative to «ты»
    -> never listen ('question': the best of respond/ack); a low-confidence label (thresholds as decide()) -> respond
    ('low-confidence'): missing a question is worse than an extra reply."""
    label, probs = route.get("label"), route.get("probs") or {}
    if "костян" in text.lower():
        return "respond", (None if label == "respond" else "name")
    if label == "listen" and is_question(text):
        alt = max(("respond", "ack"), key=lambda k: probs.get(k, 0.0))
        return alt, "question"
    if label and probs:
        top = sorted(probs.values(), reverse=True) + [0.0]
        if top[0] < MIN_PROB or top[0] < MIN_MARGIN * top[1]:
            return "respond", (None if label == "respond" else "low-confidence")
    return label or "respond", (None if label else "no-label")


# duck mode, binary question (T-038 1b.4): the 3-way question collapsed into `ack` (e4b 34/37); the model now only
# says whether an answer is expected right now, in Russian one-token words; «угу» (ack) is a rule over the pause
DUCK_PROMPT_BIN = """Служебный вопрос, не отвечай собеседнику. Собеседник один за компьютером и говорит при тебе. Ждёт ли он от тебя ответа прямо сейчас?
да — вопрос, просьба, обращение к тебе, законченная мысль, на которую естественно отреагировать;
нет — бормочет под нос, считает, читает вслух, говорит коту, мысль не закончена.
Его реплика: «{text}». После реплики пауза {pause_s} с. Ответ одним словом:"""
_YESNO = {"да": "да", "нет": "нет"}
ACK_MIN_PAUSE_S, ACK_MIN_WORDS = 1.5, 3


def _yesno_probs(top_logprobs):
    probs = {}
    for alt in top_logprobs:
        w = _YESNO.get(alt["token"].strip().lower())
        if w:
            probs[w] = probs.get(w, 0.0) + math.exp(alt["logprob"])
    return probs


def _yesno_word(raw):
    m = re.match(r"\W*(да|нет)\b", (raw or "").lower())
    return m.group(1) if m else None


def _bin_result(raw, probs, t0):
    if sum(probs.values()) < 0.5:
        probs = {}
    return {"ans_tok": max(probs, key=probs.get) if probs else None, "ans_word": _yesno_word(raw), "raw": (raw or "")[:40],
            "probs": {k: round(v, 3) for k, v in probs.items()}, "s": round(time.perf_counter() - t0, 3)}


def duck_bin_llamaserver(host, messages, text, pause_s, timeout=120):
    """Answer by the first token's top_logprobs (ans_tok) and by the generated word, 3 tokens (ans_word)."""
    body = {"messages": messages + [{"role": "user", "content": DUCK_PROMPT_BIN.format(text=text, pause_s=pause_s)}],
            "max_tokens": 3, "temperature": 0.0, "logprobs": True, "top_logprobs": 10,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    r = requests.post(f"{host}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    try:
        probs = _yesno_probs(ch["logprobs"]["content"][0]["top_logprobs"])
    except (KeyError, IndexError, TypeError):
        probs = {}
    return _bin_result((ch["message"].get("content") or "").strip(), probs, t0)


def duck_bin_ollama(host, model, messages, text, pause_s, num_ctx, timeout=120):
    body = {"model": model, "stream": False, "think": False, "keep_alive": "10m", "logprobs": True, "top_logprobs": 10,
            "messages": messages + [{"role": "user", "content": DUCK_PROMPT_BIN.format(text=text, pause_s=pause_s)}],
            "options": {"num_ctx": num_ctx, "temperature": 0.0, "seed": 1, "num_predict": 3}}
    t0 = time.perf_counter()
    r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    try:
        probs = _yesno_probs(d["logprobs"][0]["top_logprobs"])
    except (KeyError, IndexError, TypeError):
        probs = {}
    return _bin_result((d["message"].get("content") or "").strip(), probs, t0)


def finished(text):
    """A completed remark: at least ACK_MIN_WORDS words (the pause is checked by the caller). Not by the final full stop:
    punct-STT often leaves it out, and `ack` never fired on the live test (T-038 stage 2b, answer 4)."""
    return len(re.findall(r"\w+", text)) >= ACK_MIN_WORDS


def duck_from_no(text, pause_s):
    return "ack" if pause_s >= ACK_MIN_PAUSE_S and finished(text) else "listen"


# a question by its words, not by «?» (T-038 stage 2, step 7a): punct-STT puts «?» on 1 of 6 owner questions (T-041).
# The word must open the phrase (after a name, «а», «ну», «слушай», «кстати») — «я думаю, что…» is not a question
QUESTION_WORDS = ("сколько", "где", "как", "какой", "какая", "какое", "какие", "что", "чем", "когда", "почему",
                  "зачем", "кто", "куда", "откуда", "напомни", "скажи", "подскажи", "можешь", "помнишь", "видел", "знаешь",
                  "который", "которая", "которое", "чей", "чья", "чьё")
_LEAD = {"а", "и", "ну", "слушай", "кстати", "так", "эй", "ты", "не"}   # «не помнишь, …», «не знаешь, …»


def is_question(text):
    words = re.findall(r"[а-яё-]+", text.lower())
    while words and (words[0] in _LEAD or words[0].startswith("костян")):
        words = words[1:]
    return ("?" in text or bool(words and words[0] in QUESTION_WORDS) or words[1:2] == ["ли"]   # «есть ли …», «знаешь ли»
            or bool(_IMPERATIVE.search(text)))


def decide_duck_bin(ans, probs, text, pause_s):
    """(label, reason) from a да/нет answer: да -> respond; нет -> ack by the pause rule, else listen. Guards: the name,
    a question (is_question: words, «?», an imperative to «ты») -> respond; low confidence (thresholds as decide()) -> ack ('low-confidence');
    no answer at all -> ack ('no-label'). probs may be empty (answer by the generated word: no confidence check)."""
    if "костян" in text.lower():
        return "respond", (None if ans == "да" else "name")
    if ans != "да" and pause_s >= ACK_MIN_PAUSE_S and is_question(text):
        # a question that waits for an answer: the asker keeps the pause; muttered self-questions run on (P-030)
        return "respond", "question"
    if ans and probs:
        top = sorted(probs.values(), reverse=True) + [0.0]
        if top[0] < MIN_PROB or top[0] < MIN_MARGIN * top[1]:
            return "ack", "low-confidence"
    if ans is None:
        return "ack", "no-label"
    return ("respond" if ans == "да" else duck_from_no(text, pause_s)), None
