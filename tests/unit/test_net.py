"""services/brain/net.py (T-051) without the network: Wikipedia / SearXNG parsing on saved JSON fixtures, errors -> [],
the excerpts note, the gate (off by default, the one-shot phrase), the memory note."""
import json
from pathlib import Path

import fakeredis
import pytest
import requests

from services.brain import net
from services.brain.net import Net, excerpts, note_for_memory, personal, query_of, search

FIX = Path(__file__).resolve().parents[1] / "fixtures/net"


class Resp:
    def __init__(self, data=None, bad=False):
        self.data, self.bad = data, bad

    def json(self):
        if self.bad:
            raise ValueError("not JSON")       # what a bot check / an HTML error page gives
        return self.data


class Session:
    """requests.Session stand-in: answers by the request's parameters, records every call."""

    def __init__(self, answers=None, exc=None):
        self.headers, self.calls, self.answers, self.exc = {}, [], answers or {}, exc

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        if self.exc:
            raise self.exc
        key = "wiki" if params.get("generator") else "searxng"
        return self.answers[key]


def wiki():
    return Session({"wiki": Resp(json.loads((FIX / "wiki_elbrus.json").read_text()))})


def test_wikipedia_search_and_extracts():
    s = wiki()
    res = search("высота Эльбруса", session=s, timeout_s=4)
    assert [x["title"] for x in res] == ["Эльбрус", "Приют на седловине Эльбруса", "Кавказские горы"]
    assert "5642" in res[0]["snippet"] and res[0]["url"] == "https://ru.wikipedia.org/wiki/Эльбрус"
    assert len(s.calls) == 1 and s.calls[0][2] == 4             # one request: generator=search + extracts
    ua = s.headers["User-Agent"]
    assert "skull-assistant" in ua and "github.com" in ua          # Wikimedia UA policy: a contact (429 without one)


def test_searxng_interface():
    s = Session({"searxng": Resp(json.loads((FIX / "searxng_elbrus.json").read_text()))})
    res = search("высота Эльбруса", backend="searxng", url="http://searx.local:8888/", session=s)
    assert len(res) == 2 and res[1]["url"] == "https://example.org/elbrus"   # the result without text is dropped
    assert s.calls[0][0] == "http://searx.local:8888/search" and s.calls[0][1]["format"] == "json"


@pytest.mark.parametrize("session", [Session(exc=requests.Timeout()), Session(exc=requests.ConnectionError()),
                                     Session({"wiki": Resp(bad=True)}), Session({"wiki": Resp({"error": {}})})])
def test_errors_give_nothing(session):
    assert search("высота Эльбруса", session=session) == []


def test_empty_query_sends_nothing():
    s = wiki()
    assert search("  ", session=s) == [] and s.calls == []


@pytest.mark.parametrize("text, q", [
    ("Кто изобрёл гравиметр?", "изобрёл гравиметр"),
    ("Костян, а какая высота Эльбруса?", "высота Эльбруса"),
    ("Сколько длятся сутки на Марсе", "длятся сутки Марсе"),
    ("Что ты знаешь о гравиметре Тюленева 1987 года", "гравиметре Тюленева 1987 года"),
])
def test_query_of(text, q):
    assert query_of(text) == q


def test_excerpts_note():
    res = search("высота Эльбруса", session=wiki())
    note = excerpts(res)
    assert note.startswith("Выдержки из сети (Википедия):\n1) Эльбрус: ") and "(ru.wikipedia.org)" in note
    assert len(note) <= net.MAX_CHARS + 60
    assert excerpts([]) == ""


def test_memory_note():
    res = search("высота Эльбруса", session=wiki())
    assert note_for_memory("Какая высота Эльбруса?", res, "Пять тысяч шестьсот сорок два метра. Высоко.") == \
        "проверял в сети: Какая высота Эльбруса? → Пять тысяч шестьсот сорок два метра. (ru.wikipedia.org)"
    assert note_for_memory("Погода сегодня?", [], "Сеть молчит.") == "проверял в сети: Погода сегодня? → не нашёл"


@pytest.fixture
def r():
    return fakeredis.FakeRedis()


# --- the policy (T-054): always | ask | never; Redis net:policy over the manifest; the old `enabled` migrates

@pytest.mark.parametrize("params, redis_value, expect", [
    ({"policy": "always"}, None, ("always", "manifest")),
    ({"policy": "ask"}, b"never", ("never", "redis")),          # Redis overrides without a restart
    ({"policy": "never"}, b"bogus", ("never", "manifest")),     # an unknown Redis value is ignored
    ({"enabled": True}, None, ("always", "manifest (enabled, migrated)")),
    ({"enabled": False}, None, ("ask", "manifest (enabled, migrated)")),
    ({}, None, ("ask", "default")),
])
def test_policy_source(r, params, redis_value, expect):
    if redis_value:
        r.set(net.POLICY_KEY, redis_value)
    assert Net(params, r).policy() == expect


@pytest.mark.parametrize("policy, world, personal_q, consent, act", [
    ("always", True, False, None, "search"),
    ("always", True, True, None, "offline"),       # about the owner: from the dialog, not a reference book
    ("always", False, False, None, "offline"),
    ("always", True, False, "yes", "search"),
    ("ask", True, False, None, "ask"),             # «Глянуть в сети?»
    ("ask", True, False, "yes", "search"),         # the owner said yes / the one-shot phrase
    ("ask", True, False, "no", "offline"),
    ("ask", True, True, None, "offline"),
    ("ask", False, False, None, "offline"),
    ("ask", True, True, "yes", "search"),          # asked explicitly about himself — his call
    ("never", True, False, None, "offline"),
    ("never", True, False, "yes", "refuse"),       # «Мне в сеть нельзя.»
    ("never", False, True, None, "offline"),
])
def test_decide(r, policy, world, personal_q, consent, act):
    assert Net({"policy": policy}, r).decide(world, personal_q, consent) == act


def test_one_shot_phrase_is_consent(r):
    n = Net({"policy": "ask"}, r)
    assert n.once_phrase("Костян, погугли")
    assert n.decide(True) == "search" and n.decide(True) == "ask"    # one question, then asking again
    assert n.once_phrase("Проверь в сети, кто изобрёл гравиметр")
    assert not n.once_phrase("Я проверил в сети вчера")              # not the imperative: no


def test_never_sends_nothing(r, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no outgoing request under policy never")
    monkeypatch.setattr(requests.Session, "get", boom)
    before = net.REQUESTS
    n = Net({"policy": "never"}, r)
    assert not n.once_phrase("Погугли") or not r.get(net.ONCE_KEY)   # the phrase arms nothing
    assert n.search("Какая высота Эльбруса?") == []
    r.set(net.POLICY_KEY, "never")
    assert Net({"policy": "always"}, r).search("Кто открыл пенициллин?") == []   # Redis never beats the manifest
    assert net.REQUESTS == before


def test_requests_counted():
    before = net.REQUESTS
    search("высота Эльбруса", session=wiki())
    assert net.REQUESTS == before + 1


@pytest.mark.parametrize("text, answer", [
    ("Да", "yes"), ("Давай", "yes"), ("Ага, глянь", "yes"), ("Посмотри", "yes"), ("Конечно", "yes"),
    ("Нет", "no"), ("Не надо", "no"), ("Да не надо", "no"), ("Сам скажи", "no"), ("Неа", "no"),
    ("Ну не знаю", None), ("Хм", None),
])
def test_yesno(text, answer):
    assert net.yesno(text) == answer


@pytest.mark.parametrize("text, own", [
    ("Какого цвета мой новый чайник?", True), ("Что ты помнишь про моего брата?", True), ("Где у меня ключи?", True),
    ("Почему чайники свистят", False), ("Какая высота Эльбруса?", False), ("При какой погоде комфортнее бегать", False),
])
def test_personal_questions_stay_offline(text, own):
    assert personal(text) is own
