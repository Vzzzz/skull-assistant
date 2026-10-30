"""core/memory.py: the reply guard (copied note, trailing asides, bare «Не знаю.» — T-030, T-042), tokens / stems,
the note format v6 with the time order (T-036) and AssocMemory retrieval with a stand-in embedder (fixed vectors)."""
import numpy as np
import pytest
import requests

import core.memory as memory
from core.memory import NOTE_PREFIX, AssocMemory, clean_reply, format_note, stems, tokens

# --- the guard


def test_clean_reply_strips_copied_note():
    note = "(из памяти: он говорил «ключи в рюкзаке»)"
    out, flags = clean_reply("В рюкзаке. " + note, note)
    assert out == "В рюкзаке." and "copied" in flags


def test_clean_reply_strips_trailing_aside():
    out, flags = clean_reply("Бублик спит.\n(Я не стану напоминать про кота, это не моё дело.)")
    assert out == "Бублик спит." and flags == {"aside"}


def test_clean_reply_keeps_text_when_everything_would_go():
    out, flags = clean_reply("(из памяти: что-то)", "(из памяти: что-то)")
    assert out == "(из памяти: что-то)" and "copied" in flags


@pytest.mark.parametrize("reply, dunno", [
    ("Не знаю.", True), ("Я не знаю…", True), ("Без понятия!", True), ("понятия не имею", True),
    ("Не знаю. Спроси у кота.", False), ("Бублик.", False), ("Не знаю, где ключи, но куртка на мне.", False),
])
def test_bare_dunno_is_flagged(reply, dunno):
    """T-042 (core's answer 2): a bare «Не знаю.» is said but not stored — brain skips history and memory on 'dunno'."""
    out, flags = clean_reply(reply)
    assert out == reply.strip() and ("dunno" in flags) is dunno


# --- tokens and stems

def test_tokens_drop_stopwords_and_short():
    assert tokens("Я положил ключи в карман, а ты?") == ["положил", "ключи", "карман"]
    assert tokens("Ёлка") == ["елка"]


def test_stems_cut_to_prefix():
    assert stems("ключи ключей ключ") == {"ключи", "ключе", "ключ"}
    assert stems("переложил в рюкзак", n=4) == {"пере", "рюкз"}


# --- the note (v6, T-036): time order with markers, length limits

def rec(turn, user, assistant=""):
    return (1.0, {"turn": turn, "user": user, "assistant": assistant})


def test_format_note_empty():
    assert format_note([]) == ""


def test_format_note_single_has_no_markers():
    note = format_note([rec(6, "Ключи в куртке", "Запомнил.")])
    assert note == NOTE_PREFIX + "он говорил «Ключи в куртке», ты отвечал «Запомнил.»)"


def test_format_note_chronology():
    hits = [rec(25, "Переложил ключи в рюкзак"), rec(6, "Ключи в куртке"), rec(14, "Купить батарейки")]
    note = format_note(hits)
    assert note.index("сначала он говорил «Ключи в куртке»") < note.index("потом он говорил «Купить батарейки»") \
        < note.index("позже всего он говорил «Переложил ключи в рюкзак»")


def test_format_note_limits():
    note = format_note([rec(1, "у" * 300, "о" * 300)], max_reply_chars=80, max_user_chars=140)
    assert "«" + "у" * 140 + "…»" in note and "«" + "о" * 80 + "…»" in note


# --- retrieval with a stand-in embedder

TOPICS = {"ключ": 0, "кот": 1, "бублик": 1, "встреч": 2, "дим": 2}


def fake_embed(host, texts, timeout=60):
    """One axis per topic the text mentions; nothing -> a fourth axis. Normalised like the real one."""
    out = []
    for t in texts:
        v = np.zeros(4, dtype=np.float32)
        for word, axis in TOPICS.items():
            if word in t.lower():
                v[axis] = 1.0
        if not v.any():
            v[3] = 1.0
        out.append(v / np.linalg.norm(v))
    return np.asarray(out)


@pytest.fixture
def mem(monkeypatch):
    monkeypatch.setattr(memory, "embed", fake_embed)
    m = AssocMemory("http://stub", mode="hybrid")
    m.add(1, "Кот опять спал на клавиатуре, его зовут Бублик", "Бублик молодец.")
    m.add(2, "Ключи положил в карман синей куртки", "Запомнил.")
    m.add(3, "Завтра встреча с Димой в половине двенадцатого", "Хорошо.")
    return m


def test_retrieve_hybrid_finds_the_topic(mem):
    hits = mem.retrieve("Где мои ключи?", k=1)
    assert hits[0][1]["turn"] == 2


def test_retrieve_excludes_turns_in_window(mem):
    assert all(h[1]["turn"] != 2 for h in mem.retrieve("Где мои ключи?", k=3, exclude_turns=(2,)))


def test_retrieve_nothing_outside_window_skips_embedding(mem, monkeypatch):
    def fail(*a, **k):
        raise AssertionError("embedding called with nothing to find")
    monkeypatch.setattr(memory, "embed", fail)
    assert mem.retrieve("Где ключи?", exclude_turns=(1, 2, 3)) == []


def test_retrieve_kw_mode_needs_no_embedder(monkeypatch):
    monkeypatch.setattr(memory, "embed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no embedder in kw")))
    m = AssocMemory("http://stub", mode="kw")
    m.add(1, "Код домофона четыре семь один два")
    m.add(2, "Кот спал на клавиатуре")
    assert m.retrieve("какой код от домофона", k=1)[0][1]["turn"] == 1


def test_add_with_failing_embedder_leaves_no_record(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("embedder down (mode B)")
    monkeypatch.setattr(memory, "embed", down)
    m = AssocMemory("http://stub", mode="hybrid")
    with pytest.raises(requests.RequestException):
        m.add(1, "Ключи в рюкзаке")
    assert m.records == [] and m.vecs is None
