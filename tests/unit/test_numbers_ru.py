"""core/numbers_ru.py (T-015): minimal normalization of transcripts and number words for the calc extractor."""
import json
from pathlib import Path

import pytest

from core.numbers_ru import normalize, parse_number, to_digits

ROOT = Path(__file__).resolve().parents[2]
CTX = [json.loads(l) for l in (ROOT / "research/eval/ctx_probe_dialog.jsonl").read_text(encoding="utf-8").splitlines()
       if json.loads(l).get("block") in ("math", "probe")]

OWN = [   # own cases (21): what changes and what does not
    ("Завтра в половине двенадцатого придёт Дима", "Завтра в 11:30 придёт Дима"),
    ("Встреча без четверти восемь", "Встреча 07:45"),
    ("Без десяти девять выхожу", "08:50 выхожу"),
    ("Начало в четверть девятого", "Начало в 08:15"),
    ("Я выехал в восемь сорок", "Я выехал в 08:40"),
    ("Будильник на семь тридцать", "Будильник на 07:30"),
    ("Приду в восемь ноль пять", "Приду в 08:05"),
    ("Приду к восьми", "Приду к восьми"),                                    # oblique case of the hour: not parsed (minimal)
    ("Поезд в двадцать три пятнадцать", "Поезд в 23:15"),
    ("Пятнадцать процентов от 2 400", "15 процентов от 2 400"),
    ("Двадцать три плюс сто сорок пять", "23 плюс сто сорок пять"),
    ("Тысяча девятьсот восемьдесят четыре — хороший год", "1984 — хороший год"),
    ("Код 4 7 1 2, запомни", "Код 4712, запомни"),
    ("Код от домофона четыре семь один два", "Код от домофона 4712"),
    ("Ехать час тридцать пять минут", "Ехать час тридцать пять минут"),      # a duration: left as is
    ("Позвони в два", "Позвони в два"),                                      # an hour without minutes: left
    ("У меня два кота и одна собака", "У меня два кота и одна собака"),
    ("В 8:40 выезжаю", "В 8:40 выезжаю"),                                    # digits from STT: untouched
    ("Цена 4 7.5 и 1", "Цена 4 7.5 и 1"),                                    # not a code: a decimal inside
    ("Сколько будет семнадцать умножить на двадцать три?", "Сколько будет семнадцать умножить на двадцать три?"),
    ("Как зовут моего кота?", "Как зовут моего кота?"),
]


@pytest.mark.parametrize("text, expect", OWN)
def test_normalize(text, expect):
    assert normalize(text) == expect


@pytest.mark.parametrize("x", CTX + [{"text": t} for t, _ in OWN], ids=lambda x: x["text"][:30])
def test_idempotent(x):
    n = normalize(x["text"])
    assert normalize(n) == n


@pytest.mark.parametrize("x", CTX, ids=lambda x: x["text"][:30])
def test_ctx_lines_change_only_numbers(x):
    """Every word that is not a number survives normalization (the CTX-1 math / probe turns)."""
    words = lambda t: {w.strip("?.,!").lower() for w in t.split()} - {""}
    gone = words(x["text"]) - words(normalize(x["text"]))
    assert all(w in {"восемь", "сорок", "в"} or w.isdigit() for w in gone), gone


@pytest.mark.parametrize("words, value", [
    ("семнадцать", 17), ("двадцать три", 23), ("две тысячи четыреста", 2400), ("сто сорок пять", 145),
    ("тысяча", 1000), ("тысяча девятьсот восемьдесят четыре", 1984), ("сто двадцать тысяч триста", 120300),
])
def test_parse_number(words, value):
    assert parse_number(words.split())[0] == value


def test_to_digits():
    assert to_digits("Вещь стоит две тысячи четыреста, скидка пятнадцать процентов") == "Вещь стоит 2400, скидка 15 процентов"
    assert to_digits("четыре семь") == "4 7"                                 # two digits, not 4 + 7 = 11
