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


def test_off_by_default_no_request(r, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no outgoing request while the net is off")
    monkeypatch.setattr(requests.Session, "get", boom)
    n = Net({"enabled": False}, r)
    assert not n.allowed()
    assert not Net({}, r).allowed()


def test_one_shot_phrase(r):
    n = Net({"enabled": False}, r)
    assert n.once_phrase("Костян, погугли")
    assert n.once_phrase("Проверь в сети, кто изобрёл гравиметр")
    assert not n.once_phrase("Я проверил в сети вчера")        # not the imperative: no
    assert n.allowed() and not n.allowed()                     # one request, then off again


def test_enabled_always_allowed(r):
    assert Net({"enabled": True}, r).allowed()


@pytest.mark.parametrize("text, own", [
    ("Какого цвета мой новый чайник?", True), ("Что ты помнишь про моего брата?", True), ("Где у меня ключи?", True),
    ("Почему чайники свистят", False), ("Какая высота Эльбруса?", False), ("При какой погоде комфортнее бегать", False),
])
def test_personal_questions_stay_offline(text, own):
    assert personal(text) is own
