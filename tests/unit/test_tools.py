"""services/brain/tools.py (T-015): the calc tool — ast arithmetic, no eval; the expression from a transcript."""
import json
from pathlib import Path

import pytest

from services.brain.tools import calc, extract, fmt, note

ROOT = Path(__file__).resolve().parents[2]
MATH = [json.loads(l) for l in (ROOT / "research/eval/ctx_probe_dialog.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(l).get("block") == "math"]


@pytest.mark.parametrize("expr, value", [("17*23", 391), ("2+2*2", 6), ("(15 + 5) / 4", 5), ("2**10", 1024),
                                          ("7 // 2", 3), ("7 % 3", 1), ("round(2.567, 2)", 2.57), ("abs(-3)", 3),
                                          ("2,5*2", 5.0), ("17 × 23", 391)])
def test_calc(expr, value):
    assert calc(expr) == value


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "9**9999", "1/0", "a+1", "[1]*3",
                                  "(lambda: 1)()", "True+1", "10**20*10**20"])
def test_calc_refuses(expr):
    assert calc(expr) is None


@pytest.mark.parametrize("x", MATH, ids=lambda x: x["math_id"])
def test_ctx_math(x):
    """CTX-1 math M1-M3 in words: 391, 2040, 10:15."""
    r = extract(x["text"])
    assert r and r[1] == x["expect"]["answer"]


@pytest.mark.parametrize("text, result", [
    ("Сколько будет 13% от 3 700", "481"),
    ("Пятнадцать процентов от 2 400", "360"),
    ("Сколько будет 47 умножить на 21", "987"),
    ("Сто двадцать разделить на восемь", "15"),
    ("Выйду в 8:40, ехать час тридцать пять, когда буду?", "10:15"),
    ("Во сколько приеду, если выйду в 8:40 и ехать 1:35", "10:15"),
    ("Выезжаю в 9:15, дорога 40 минут", "9:55"),
    ("Через полтора часа после 18:30", "20:00"),
    ("Ехать в 10:00, в пути 2 часа 15 минут", "12:15"),
    ("В 23:30 выезжаю, ехать 2 часа", "1:30"),
])
def test_extract(text, result):
    assert extract(text)[1] == result


@pytest.mark.parametrize("text", ["Как дела?", "Купил два кота", "Сколько будет 120 разделить на 0", "Дима придёт в 11:30",
                                  "Код 4712"])
def test_no_computation(text):
    assert note(text) == ""


def test_note_and_fmt():
    assert note("Сколько будет семнадцать умножить на двадцать три?") == "(вычислено: 17 × 23 = 391)"
    assert fmt(2.5) == "2,5" and fmt(391.0) == "391"
