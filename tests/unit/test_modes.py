"""Mode B by voice (T-046): the trigger / exit phrases from the brain manifest, the deferred question, the B policy
(no router, WORLD_SOFT, memory on) and the ambient rules in B (window always open, no duck, owner's voice only)."""
from pathlib import Path

import fakeredis
import pytest
import yaml

from core.ambient import Ambient, engaged_until, open_window
from core.modes import deferred_question, find_phrase, imperative, stem
from core.router import POLICY, POLICY_B, WORLD_SOFT

MODES = yaml.safe_load((Path(__file__).resolve().parents[2] / "services/brain/manifest.yaml").read_text(encoding="utf-8"))["params"]["modes"]
ENTER, EXIT = MODES["enter"], MODES["exit"]


def test_manifest_phrases_and_timeout():
    assert {"подумай как следует", "подумай хорошенько", "включи мозги", "подумай серьёзно"} <= set(ENTER)
    assert set(EXIT) == {"хватит думать", "достаточно думать", "возвращайся", "вернись"}   # no bare «достаточно»
    assert MODES["idle_exit_s"] == 300


@pytest.mark.parametrize("word, s", [("подумай", "подум"), ("мозги", "моз"), ("как", "как"), ("Серьёзно", "серьез")])
def test_stem(word, s):
    assert stem(word) == s


@pytest.mark.parametrize("word, forms", [
    ("подумай", {"подумай", "подумайте"}), ("Включи", {"включи", "включите"}),
    ("возвращайся", {"возвращайся", "возвращайтесь"}), ("вернись", {"вернись", "вернитесь"}),
])
def test_imperative(word, forms):
    assert imperative(word) == forms


@pytest.mark.parametrize("text, phrase, question", [
    ("Костян, подумай как следует: почему небо синее?", "подумай как следует", "почему небо синее?"),
    ("Почему небо синее? Подумай как следует.", "подумай как следует", "Почему небо синее?"),   # the question first
    ("Костян, включи мозг и скажи, сколько стоит Луна", "включи мозги", "скажи, сколько стоит Луна"),
    ("Подумайте хорошенько", "подумай хорошенько", ""),                    # no question: «Ну, думаю. Спрашивай.»
    ("Подумайте как следует", "подумай как следует", ""),                  # the plural imperative
    ("Костян, ну, подумай серьезно", "подумай серьёзно", ""),             # ё / е, fillers and the name are no question
    ("Костян, а почему трава зелёная, подумай серьёзно", "подумай серьёзно", "почему трава зелёная"),
])
def test_enter_phrase_and_deferred_question(text, phrase, question):
    span = find_phrase(text, ENTER)
    assert span and span[0] == phrase
    assert deferred_question(text, span) == question


@pytest.mark.parametrize("text", [
    "Подумай, где ключи",                      # «подумай» alone (gate: no switch without the whole phrase)
    "Я подумаю об этом завтра",
    "Включи свет",
    "Как следует покрасить забор",
    "Надо подумать, как следует покрасить",    # the infinitive: only the imperative switches (core, 2026-11-01)
    "Я подумал хорошенько и решил",
    "Включил мозги наконец",
])
def test_no_enter_phrase(text):
    assert find_phrase(text, ENTER) is None


@pytest.mark.parametrize("text, found", [
    ("Хватит думать", True), ("Ну всё, возвращайся", True), ("Достаточно думать, спасибо", True), ("Вернись", True),
    ("Вернитесь уже", True), ("Думать хватит", False), ("Соли достаточно", False), ("Достаточно, спасибо", False),
    ("Верни мне ключи", False), ("Он вернулся вчера", False), ("Возвращайте книги вовремя", False),
])
def test_exit_phrase(text, found):
    assert (find_phrase(text, EXIT) is not None) is found


def test_policy_b_is_explain_with_soft_hedge_and_memory():
    assert POLICY_B == {"max_tokens": POLICY["explain"]["max_tokens"], "memory": True, "prefix": WORLD_SOFT}


# --- ambient in B

@pytest.fixture
def r():
    return fakeredis.FakeRedis()


def test_window_open_all_the_time_in_b(r):
    open_window(r, "rid", 30, 100.0)
    r.hset("tts:end", "rid", 101.0)
    assert engaged_until(r) == 131.0
    r.set("mode:current", "B")
    assert engaged_until(r) == float("inf")
    r.set("mode:current", "A")                  # back in A: the usual window again
    assert engaged_until(r) == 131.0


def never(*a):
    raise AssertionError("the duck is not asked in mode B")


def test_b_answers_owner_statement_without_the_duck():
    a = Ambient(engaged_s=30)
    assert a.decide("закончил отчёт наконец сегодня", "owner", 0.4, 0.0, 100.0, True, never, thinking=True)[:2] == ("respond", "mode B")


def test_b_addressee_is_the_owner_or_uncertain_voice():
    """T-046 gate: the owner's short turns in B came `uncertain` (0.30-0.39) and were left unanswered; B is the window,
    so uncertain counts as the owner there, unknown / other / a window or broadcast class do not."""
    a = Ambient(engaged_s=30)
    assert a.decide("А зачем людям чайники", "uncertain", 0.4, 0.0, 100.0, True, never, thinking=True)[0] == "respond"
    assert a.decide("ну и что теперь делать", "uncertain", 0.4, 0.0, 105.0, True, never, thinking=True)[0] == "respond"
    assert a.decide("ну и что теперь делать", "uncertain", 0.4, 0.0, 106.0, True, never, lifted=True, thinking=True)[0] == "silent"
    assert a.decide("ну и что теперь делать", "uncertain", 0.4, 0.0, 107.0, True, never, bcast=True, thinking=True)[0] == "silent"
    assert a.decide("ну и что теперь делать", "unknown", 0.4, 0.0, 110.0, True, never, thinking=True)[0] == "silent"
    assert a.decide("ну и что теперь делать", "other", 0.4, 0.0, 115.0, True, never, thinking=True)[0] == "silent"
    assert a.decide("Костян, ну и что", "uncertain", 0.4, 0.0, 120.0, True, never, thinking=True)[0] == "respond"
    assert a.decide("Алло, да, слушаю", "owner", 0.4, 0.0, 130.0, True, never, thinking=True)[0] == "silent"   # the call rule stays
