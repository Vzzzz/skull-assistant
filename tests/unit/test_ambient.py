"""core/ambient.py on fakeredis: the listening policy without a wake word (D-029, T-038 2б–2в) and the gaze window
(T-031). Ambient is built with the brain manifest's numbers, as services/brain/responder.py builds it; the duck pass
is a stub returning a fixed p(«да») and counting its calls (the model is not tested here).

research/eval/sim_addressee_fsm.py is an older offline FSM over stage-1 logs (block changes as pauses), not this policy:
its scenarios are restated here against core.ambient instead of copied."""
from pathlib import Path

import fakeredis
import pytest
import yaml

from core.ambient import OPEN_KEY, TTS_END, Ambient, engaged_until, gaze_skip, has_name, needs_pause, open_window

AMB = yaml.safe_load((Path(__file__).resolve().parents[2] / "services/brain/manifest.yaml").read_text(encoding="utf-8"))["params"]["ambient"]
STATEMENT = "закончил отчёт наконец сегодня"          # 4 words, no question, no name


def brain_ambient():
    return Ambient(engaged_s=AMB["engaged_s"], busy_pause_s=AMB["busy_pause_s"], veto_p=AMB["veto_aside_p"],
                   engaged_yes_p=AMB["engaged_yes_p"], ack_gap_s=AMB["ack_gap_s"], engaged_ack_p=AMB["engaged_ack_p"],
                   idle_yes_p=AMB["idle_yes_p"], idle_ack_p=AMB["idle_ack_p"])


class Duck:
    def __init__(self, p_yes):
        self.p_yes, self.calls = p_yes, 0

    def __call__(self, text, pause_s):
        self.calls += 1
        if self.p_yes is None:
            return {"ans_tok": None, "probs": {}}
        return {"ans_tok": "да" if self.p_yes >= 0.5 else "нет", "probs": {"да": self.p_yes, "нет": 1 - self.p_yes}}


@pytest.fixture
def r():
    return fakeredis.FakeRedis()


def test_manifest_numbers_are_the_accepted_ones():
    """Window after an answer 30 s (2в replay), after a gaze line 15 s (T-031), BUSY ends after 20 s of the owner's
    silence, «угу» not more often than once in 20 s, the duck's bars 0.7 / 0.3 (T-042 4a)."""
    assert (AMB["engaged_s"], AMB["gaze_window_s"], AMB["busy_pause_s"], AMB["ack_gap_s"]) == (30, 15, 20, 20)
    assert (AMB["engaged_yes_p"], AMB["engaged_ack_p"], AMB["idle_yes_p"], AMB["idle_ack_p"]) == (0.7, 0.3, 0.7, 0.3)


# --- IDLE

def test_name_answers_and_opens_window_without_the_duck(r):
    a, duck = brain_ambient(), Duck(0.0)
    assert a.decide("Костян, привет", "owner", 0.5, 0.0, 100.0, False, duck)[::2] == ("respond", True)
    assert duck.calls == 0
    open_window(r, "rid1", AMB["engaged_s"], 100.0)
    assert engaged_until(r) == 130.0                                 # no tts:end yet: from the answer
    r.set("tts:speaking", 1)
    assert engaged_until(r) == float("inf")                          # open while it plays
    r.delete("tts:speaking")
    r.hset(TTS_END, "rid1", 104.5)
    assert engaged_until(r) == 134.5                                 # 30 s from the end of playback


def test_question_answers_without_the_duck():
    a, duck = brain_ambient(), Duck(0.0)
    assert a.decide("А где мои ключи", "owner", 0.3, 0.0, 100.0, False, duck)[:2] == ("respond", "question")
    assert duck.calls == 0


@pytest.mark.parametrize("p_yes, pause, expect", [
    (0.8, 2.0, ("respond", True)),       # T-042 4a: the duck's «да» >= 0.7, owner, pause >= 1.5 s -> an answer
    (0.8, 1.0, ("ack", False)),          # same «да», no pause -> only «угу»
    (0.5, 2.0, ("ack", False)),          # 0.3..0.7 -> «угу»
    (0.1, 2.0, ("ack", False)),          # «нет»: «угу» by the pause rule (finished thought, pause)
    (0.1, 0.8, ("silent", False)),
])
def test_idle_statement_by_the_duck(p_yes, pause, expect):
    out = brain_ambient().decide(STATEMENT, "owner", pause, 0.0, 100.0, False, Duck(p_yes))
    assert (out[0], out[2]) == expect


def test_ack_not_more_often_than_gap():
    a = brain_ambient()
    assert a.decide(STATEMENT, "owner", 2.0, 0.0, 100.0, False, Duck(0.5))[0] == "ack"
    assert a.decide(STATEMENT, "owner", 2.0, 0.0, 110.0, False, Duck(0.5))[0] == "silent"
    assert a.decide(STATEMENT, "owner", 2.0, 0.0, 121.0, False, Duck(0.5))[0] == "ack"


def test_idle_uncertain_and_unknown_are_silent():
    a = brain_ambient()
    assert a.decide(STATEMENT, "uncertain", 2.0, 0.0, 100.0, False, Duck(0.9))[0] == "silent"
    assert a.decide("Где ключи", "unknown", 2.0, 0.0, 200.0, False, Duck(0.9))[0] == "silent"
    assert a.decide("Где ключи", "other", 2.0, 0.0, 300.0, False, Duck(0.9))[0] == "silent"


# --- ENGAGED

@pytest.mark.parametrize("p_yes, expect", [(0.75, ("respond", True)), (0.5, ("ack", False)), (0.1, ("ack", False))])
def test_engaged_statement_extends_only_when_answered(p_yes, expect):
    out = brain_ambient().decide(STATEMENT, "owner", 2.0, 0.0, 100.0, True, Duck(p_yes))
    assert (out[0], out[2]) == expect


def test_engaged_uncertain_counts_as_owner_but_unknown_is_silent():
    a = brain_ambient()
    assert a.decide("А почему так", "uncertain", 0.5, 0.0, 100.0, True, Duck(0.0))[0] == "respond"
    assert a.decide("А почему так", "unknown", 0.5, 0.0, 120.0, True, Duck(0.9))[0] == "silent"   # 2в: also in the window


def test_uncertain_from_a_window_or_broadcast_needs_the_name():
    a = brain_ambient()
    assert a.decide("А почему так", "uncertain", 0.5, 0.0, 100.0, True, Duck(0.9), lifted=True)[0] == "silent"
    assert a.decide("А почему так", "uncertain", 0.5, 0.0, 110.0, True, Duck(0.9), bcast=True)[0] == "silent"
    assert a.decide("Костян, а почему", "uncertain", 0.5, 0.0, 120.0, True, Duck(0.9), bcast=True)[0] == "respond"


def test_short_unknown_right_after_owner_is_uncertain():
    a = brain_ambient()
    a.decide(STATEMENT, "owner", 0.5, 0.0, 100.0, True, Duck(0.9))
    out = a.decide("Ты меня слышишь?", "unknown", 0.5, 0.0, 105.0, True, Duck(0.0), seg_s=1.0)
    assert out[0] == "respond"                                       # T-042 4a: <1.5 s within 10 s after the owner
    out = a.decide("Ты меня слышишь?", "unknown", 0.5, 0.0, 130.0, True, Duck(0.0), seg_s=1.0)
    assert out[0] == "silent"                                        # 25 s later: just unknown


# --- BUSY

def test_veto_makes_busy_until_owner_pause_or_name():
    a = brain_ambient()
    assert a.decide("да, да, я тебя слушаю, говори", "owner", 0.5, 0.95, 100.0, False, Duck(0.9))[0] == "silent"
    assert a.busy
    assert a.decide(STATEMENT, "owner", 2.0, 0.0, 105.0, False, Duck(0.9))[1] == "busy"
    assert a.decide(STATEMENT, "other", 2.0, 0.0, 124.0, False, Duck(0.9))[0] == "silent"   # a podcast is no pause
    assert a.decide(STATEMENT, "owner", 2.0, 0.0, 126.0, False, Duck(0.9))[0] == "respond"  # 21 s of the owner's silence
    assert not a.busy


def test_name_ends_busy_at_once():
    a = brain_ambient()
    a.decide("Алло, да, слушаю", "owner", 0.5, 0.1, 100.0, False, Duck(0.9))
    assert a.busy
    assert a.decide("Костян, всё, я закончил", "owner", 0.5, 0.0, 102.0, False, Duck(0.9))[0] == "respond"


def test_owner_question_is_not_vetoed():
    """2в: a question in the owner's voice is answered even at p_aside 0.9 («а что он сказал про подкаст?»)."""
    a = brain_ambient()
    assert a.decide("А что он там сказал про налоги", "owner", 0.5, 0.92, 100.0, False, Duck(0.0))[0] == "respond"
    assert not a.busy


def test_broadcast_veto_in_uncertain_voice_does_not_make_busy():
    a = brain_ambient()
    assert a.decide("В эфире новости часа", "uncertain", 0.5, 0.97, 100.0, False, Duck(0.0))[0] == "silent"
    assert not a.busy


# --- the gaze window (T-031): a gaze line opens 15 s, a look never extends an open window

def test_gaze_line_opens_15_s(r):
    open_window(r, "gaze1", AMB["gaze_window_s"], 100.0)
    r.hset(TTS_END, "gaze1", 101.0)
    assert engaged_until(r) == 116.0


def test_look_inside_window_is_ignored_not_extended(r):
    open_window(r, "rid1", AMB["engaged_s"], 100.0)
    before = r.get(OPEN_KEY)
    assert gaze_skip(r, 110.0, False, -1e9, AMB["gaze_cooldown_s"]) == "engaged"
    assert r.get(OPEN_KEY) == before


@pytest.mark.parametrize("setup, now, busy, last, why", [
    ({}, 100.0, False, -1e9, None),
    ({"tts:speaking": 1}, 100.0, False, -1e9, "speaking"),
    ({"mode:transition": "{}"}, 100.0, False, -1e9, "mode transition"),
    ({}, 100.0, True, -1e9, "busy"),
    ({}, 100.0, False, 80.0, "cooldown (20 s ago)"),
])
def test_gaze_skip_reasons(r, setup, now, busy, last, why):
    for k, v in setup.items():
        r.set(k, v)
    assert gaze_skip(r, now, busy, last, AMB["gaze_cooldown_s"]) == why


# --- STT waits for the pause only when the duck will use it

@pytest.mark.parametrize("text, speaker, engaged, wait", [
    (STATEMENT, "owner", False, True), ("Где ключи", "owner", False, False), (STATEMENT, "owner", True, False),
    ("Костян, ну", "owner", False, False), (STATEMENT, "other", False, False),
])
def test_needs_pause(text, speaker, engaged, wait):
    assert needs_pause(text, speaker, engaged) is wait


def test_has_name():
    assert has_name("Костян, ты где") and has_name("КОСТЯНЧИК") and not has_name("череп на столе")
