"""core/speech.py for_speech: what Piper must not read literally — LaTeX, markdown (tables too), pictographs — units
after a number with Russian agreement (T-003 step 6) and dates as ordinals (T-042). Idempotent: brain applies it per sentence, research tools again."""
import pytest

from core.speech import for_speech

CASES = [
    ("$17 \\times 20 = 340$", "17 умножить на 20 = 340."),                    # inline LaTeX from gemma-4
    ("\\frac{1}{2} стакана", "1 разделить на 2 стакана."),
    ("$$a^2 + b^2$$", "a в квадрате + b в квадрате."),
    ("**жирный** и *курсив*", "жирный и курсив."),
    ("# Заголовок\n- пункт один\n- пункт два", "Заголовок. пункт один. пункт два."),
    ("[ссылка](http://x.ru)", "ссылка."),
    ("Код: `print(1)`", "Код: print(1)."),
    ("17 * 3", "17 умножить на 3."), ("2 × 3", "2 умножить на 3."),
    ("12 : 4 = 3", "12 разделить на 4 = 3."),
    ("Выехал в 08:40", "Выехал в 08:40."),                                    # a time is not a division
    ("≈ 5", "примерно 5."), ("~ 5 минут", "примерно 5 минут."),
    ("Готово ✅", "Готово."), ("→ дальше", "дальше."),                         # pictographs are spelled out by espeak
    ("строка без точки\nвторая", "строка без точки. вторая."),                # lines become sentences
]

UNITS = [
    ("1 ₽", "1 рубль."), ("11 руб.", "11 рублей."), ("3 л молока", "3 литра молока."), ("1 л", "1 литр."),
    ("2,5 л", "2,5 литра."),                                                  # a fraction: genitive singular
    ("5 кг", "5 килограммов."), ("12 кг", "12 килограммов."), ("21 км", "21 километр."), ("111 км", "111 километров."),
    ("15%", "15 процентов."), ("22°", "22 градуса."), ("-5 °C", "-5 градусов."),
]


DATES = [  # T-042, core's answer 5: the day before a month in the genitive is an ordinal in the genitive
    ("Встреча 21 декабря в 11:30.", "Встреча двадцать первого декабря в 11:30."),
    ("1 января 2027 года", "первого января 2027 года."),                    # the year stays digits
    ("3-го марта", "третьего марта."), ("12 июня", "двенадцатого июня."), ("20 февраля", "двадцатого февраля."),
    ("30 апреля и 31 мая", "тридцатого апреля и тридцать первого мая."),
    ("32 мая", "32 мая."),                                                   # not a day
    ("5 кг и 22 декабря", "5 килограммов и двадцать второго декабря."),
]

TABLES = [  # T-045, core's answer 3: rule lines out, cells through commas, rows as sentences
    ("| a | b |\n|---|---|\n| 1 | 2 |", "a, b. 1, 2."),
    ("Итог:\n| Имя | Возраст |\n|:---|---:|\n| Бублик | 3 |", "Итог: Имя, Возраст. Бублик, 3."),
]


@pytest.mark.parametrize("text, spoken", DATES)
def test_dates(text, spoken):
    assert for_speech(text) == spoken


@pytest.mark.parametrize("text, spoken", TABLES)
def test_tables(text, spoken):
    assert for_speech(text) == spoken


@pytest.mark.parametrize("text, spoken", CASES)
def test_markup(text, spoken):
    assert for_speech(text) == spoken


@pytest.mark.parametrize("text, spoken", UNITS)
def test_units(text, spoken):
    assert for_speech(text) == spoken


@pytest.mark.parametrize("text", [t for t, _ in CASES + UNITS + DATES + TABLES] + ["Ответ: **42** ✅\n\n- итог $x^2$"])
def test_idempotent(text):
    once = for_speech(text)
    assert for_speech(once) == once


def test_nothing_unreadable_left():
    out = for_speech("**Итог:** $\\sqrt{16} \\cdot 2 = 8$ ✅ #готово")
    assert not set(out) & set("*$\\#✅")
