"""core/router.py: the rule half of the router — label parsing, logprob distributions, guards, the question detector,
the world prefix and the duck decision. Expectations come from the cards and the core's answers (T-037, T-038, T-042)
and from research/eval/ctx_probe_ambient.jsonl; what only the model decides is not tested."""
import json
import math
from pathlib import Path

import pytest

from core.ambient import has_name
from core.router import (POLICY, WORLD_SOFT, _duck_probs, _probs, _yesno_probs, decide, decide_duck_bin, duck_from_no,
                         finished, has_number, has_question_word, is_question, parse_label, policy_prefix)

AMBIENT = Path(__file__).resolve().parents[2] / "research/eval/ctx_probe_ambient.jsonl"


def alts(**p):
    """top_logprobs as llama-server returns them: [{"token", "logprob"}]."""
    return [{"token": t, "logprob": math.log(v)} for t, v in p.items()]


# --- labels and distributions

@pytest.mark.parametrize("raw, label", [
    ("chat", "chat"), (" Memory", "memory"), ("calc.", "calc"), ("aside", "aside"),
    ("реакция на тебя", "chat"),            # e2b answers with the description, not the label
    ("просит посчитать", "calc"), ("сказано не тебе", "aside"), ("", None), ("??", None),
])
def test_parse_label(raw, label):
    assert parse_label(raw) == label


def test_probs_sums_label_tokens_and_drops_others():
    p = _probs(alts(chat=0.6, **{" chat": 0.1, "calc": 0.2, "xyz": 0.05}))
    assert p == pytest.approx({"chat": 0.7, "calc": 0.2})


def test_duck_and_yesno_probs():
    assert _duck_probs(alts(respond=0.5, ack=0.3, listen=0.1)) == pytest.approx({"respond": 0.5, "ack": 0.3, "listen": 0.1})
    assert _yesno_probs(alts(**{"да": 0.6, " Да": 0.1, "нет": 0.2, "ну": 0.05})) == pytest.approx({"да": 0.7, "нет": 0.2})


# --- guards on the router's answer (T-037, T-038)

def route(label, **probs):
    return {"label": label, "probs": probs}


@pytest.mark.parametrize("r, text, expect", [
    (route("aside", aside=0.95), "Костян, это не тебе", ("chat", "name")),      # the name is never aside
    (route("aside", aside=0.5, chat=0.3), "да-да, слушаю", ("chat", "aside-low")),
    (route("aside", aside=0.95), "да-да, слушаю", ("aside", None)),
    (route("chat", chat=0.40, memory=0.30), "ну и ладно", ("chat", "low-confidence")),   # top < 0.45
    (route("memory", memory=0.50, chat=0.40), "а где ключи", ("chat", "low-confidence")),  # margin < 1.3x
    (route("calc", calc=0.9), "Как меня зовут?", ("chat", "calc-no-number")),  # T-037: calc only with a number
    (route("calc", calc=0.9), "Сколько будет семнадцать на три?", ("calc", None)),
    (route("explain", explain=0.8, chat=0.1), "Как устроен замок?", ("explain", None)),
])
def test_decide(r, text, expect):
    assert decide(r, text) == expect


@pytest.mark.parametrize("text, has", [
    ("семнадцать умножить на двадцать три", True), ("15%", True), ("половина пиццы", True), ("сорок два", True),
    ("Как меня зовут?", False), ("Привет, я дома", False),
])
def test_has_number(text, has):
    assert has_number(text) is has


# --- the world prefix (T-042 step 4a)

@pytest.mark.parametrize("label, text, prefix", [
    ("world", "Ты слышал новости про запуск ракеты", POLICY["world"]["prefix"]),   # perception, no question word
    ("world", "Ты слышал, из чего готовят хумус", WORLD_SOFT),       # «из чего» makes it a question: soft
    ("world", "Почему небо голубое", WORLD_SOFT),                      # knowledge, no perception word
    ("world", "В каком году была Бородинская битва", WORLD_SOFT),
    ("calc", "Сколько будет 2 плюс 2", POLICY["calc"]["prefix"]),
    ("chat", "Сегодня я видел кота", ""),                               # only world carries the hedge
])
def test_policy_prefix(label, text, prefix):
    assert policy_prefix(label, text) == prefix


def test_full_world_prefix_refuses_in_character():
    """T-042, core's answer 2: not a bare «Не знаю.» — say what you see or remember instead."""
    assert "не одним словом: скажи, что видишь или помнишь вместо этого" in POLICY["world"]["prefix"]
    assert "не одним словом" not in WORLD_SOFT


# --- the question detector: words anywhere, imperatives, «?» only with «ты», «ж/же» (T-038 2в, T-042)

@pytest.mark.parametrize("text", [
    "В каком году была битва при Ватерлоо",            # word after a preposition (T-042: was a statement)
    "Из чего готовят хумус",
    "Если ля 440, то сколько герц фа",                  # word mid-phrase
    "Костян, где мои ключи",
    "А-а, что посоветуешь на ужин",                     # filler before the word (2в)
    "Повтори, пожалуйста", "Продолжай", "Давай ещё раз",  # imperatives (4a)
    "Ты меня слышишь?",                                 # «?» with «ты»
    "Не помнишь, куда я дел зарядку",
    "Напомни про батарейки",
    "Что же ты молчишь?",                               # «же» — but «?» with «ты» still asks
])
def test_is_question(text):
    assert is_question(text)


@pytest.mark.parametrize("text", [
    "Пора заканчивать и идти гулять",
    "Да ё-моё, опять?",                                 # «?» alone is not a question (2в, answer 3)
    "Бублик, слезь со стола",                           # imperative to the cat is not in the list
    "Ох, ё-моё, что ж за хрень-то такая?",              # «ж» after the question word: rhetorical (T-042 answer 1)
    "Какого ж чёрта",
    "Как же это надоело",
    "Какая-то хрень опять",                             # «-то» forms are not question words
])
def test_not_question(text):
    assert not is_question(text)


def test_question_word_forms():
    assert has_question_word("а в каком году")
    assert has_question_word("о чём речь")
    assert not has_question_word("что-нибудь купить")
    assert not has_question_word("что ж поделать")


def test_ambient_dataset_questions():
    """Every turn the dataset says must be answered as a question is one by words — or carries the name."""
    rows = [json.loads(l) for l in AMBIENT.read_text(encoding="utf-8").splitlines() if l.strip()]
    must = [r for r in rows if r.get("duck_must") == "question"]
    assert len(must) >= 10
    missed = [r["text"] for r in must if not (is_question(r["text"]) or has_name(r["text"]))]
    assert missed == []


# --- the duck: «угу» by the pause rule, binary answer guards (T-038 2б–2в)

@pytest.mark.parametrize("text, ok", [("ну вот", False), ("пора идти гулять", True), ("Бублик, брысь", False)])
def test_finished(text, ok):
    assert finished(text) is ok


def test_duck_from_no():
    assert duck_from_no("пора идти гулять", 2.0) == "ack"
    assert duck_from_no("пора идти гулять", 1.0) == "listen"        # pause < 1.5 s
    assert duck_from_no("ну вот", 3.0) == "listen"                  # < 3 words


@pytest.mark.parametrize("ans, probs, text, pause, expect", [
    ("нет", {"нет": 0.9}, "Костян, ну что там", 0.5, ("respond", "name")),
    ("нет", {"нет": 0.9}, "Где мои ключи", 2.0, ("respond", "question")),
    ("нет", {"нет": 0.9}, "Где мои ключи", 0.8, ("listen", None)),        # a muttered question runs on
    ("да", {"да": 0.5, "нет": 0.45}, "так, посмотрим", 2.0, ("ack", "low-confidence")),
    (None, {}, "так, посмотрим", 2.0, ("ack", "no-label")),
    ("да", {"да": 0.9}, "закончил отчёт наконец", 2.0, ("respond", None)),
    ("нет", {"нет": 0.9}, "закончил отчёт наконец", 2.0, ("ack", None)),  # нет + pause + 3 words -> «угу»
])
def test_decide_duck_bin(ans, probs, text, pause, expect):
    assert decide_duck_bin(ans, probs, text, pause) == expect
