"""Text for the speech synthesizer: what the model writes is not what should be pronounced.

Piper (espeak-ng) reads markup literally: "*" -> «звёздочка», "$" -> «доллар», "\\times" -> «бэкслэш таймз»,
"✅" -> «белая галочка», "#" -> «решётка», "^" -> «крышка». gemma-4 answers with markdown and inline LaTeX
($17 \\times 20 = 340$), so the reply is rewritten for TTS only; history and memory keep the model's text.
"""
import re
import unicodedata

_LATEX_CMD = {
    "times": " умножить на ", "cdot": " умножить на ", "div": " разделить на ", "approx": " примерно ",
    "pm": " плюс-минус ", "le": " меньше или равно ", "leq": " меньше или равно ", "ge": " больше или равно ",
    "geq": " больше или равно ", "neq": " не равно ", "to": " — ", "rightarrow": " — ", "Rightarrow": " — ",
    "infty": " бесконечность ", "%": "%",
}
_MATH_SPANS = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)
_BRACED = r"\{([^{}]*)\}"


def _latex(s: str) -> str:
    s = re.sub(r"\\frac" + _BRACED + _BRACED, r"\1 разделить на \2", s)
    s = re.sub(r"\\sqrt" + _BRACED, r"корень из \1", s)
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf|operatorname)\s*" + _BRACED, r" \1 ", s)
    s = re.sub(r"\^\{?2\}?", " в квадрате", s)
    s = re.sub(r"\^\{?3\}?", " в кубе", s)
    s = re.sub(r"\\(%|[A-Za-z]+)", lambda m: _LATEX_CMD.get(m.group(1), " "), s)
    s = re.sub(r"\\[,;:! ]", " ", s)
    return s.replace("{", "").replace("}", "")


def _markdown(s: str) -> str:
    s = re.sub(r"```.*?```", " ", s, flags=re.S)                    # code blocks are not for reading aloud
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)                 # [text](url)
    s = re.sub(r"(\*\*|__)(.+?)\1", r"\2", s, flags=re.S)          # bold
    s = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"\1", s)   # italic
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.M)            # headings
    s = re.sub(r"^\s*[-*•+]\s+", "", s, flags=re.M)                # bullets
    s = re.sub(r"^\s*>\s?", "", s, flags=re.M)                     # quotes
    s = re.sub(r"^\s*\|?\s*:?-{3,}.*$", "", s, flags=re.M)         # table rule lines
    return s.replace("|", ", ")


# units after a number (T-003 step 6): Piper swallows «₽», reads «3 л» as «три эль» and «15%» as «пятнадцать процент»
_UNITS = {"₽": ("рубль", "рубля", "рублей"), "руб": ("рубль", "рубля", "рублей"), "л": ("литр", "литра", "литров"),
          "кг": ("килограмм", "килограмма", "килограммов"), "км": ("километр", "километра", "километров"),
          "°C": ("градус", "градуса", "градусов"), "%": ("процент", "процента", "процентов")}
_UNIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(₽|руб\.?|кг|км|л|°\s*[CС]|°|%)(?![A-Za-zА-Яа-яЁё])")


def _plural(number: str, forms):
    """Russian agreement: 1 рубль, 2 рубля, 5 рублей; a fraction takes the genitive singular (2,5 литра)."""
    if "." in number or "," in number:
        return forms[1]
    n = int(number)
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def _units(s: str) -> str:
    def repl(m):
        unit = m.group(2).rstrip(".").replace(" ", "").replace("С", "C")
        forms = _UNITS.get(unit) or _UNITS["°C"]          # a bare «°» after a number is a temperature
        return f"{m.group(1)} {_plural(m.group(1), forms)}"
    return _UNIT_RE.sub(repl, s)


def for_speech(text: str) -> str:
    s = _MATH_SPANS.sub(lambda m: _latex(next(g for g in m.groups() if g is not None)), text)
    s = _latex(s) if "\\" in s else s
    s = _markdown(s)
    s = _units(s)
    s = re.sub(r"(\d)\s*[*×xх]\s*(?=\d)", r"\1 умножить на ", s)    # 17 * 3, 17 × 3, 17x3
    s = re.sub(r"(\d)\s*[÷:]\s*(?=\d+\s*=)", r"\1 разделить на ", s)  # 12 : 4 = 3 (not 08:40)
    s = s.replace("×", " умножить на ").replace("÷", " разделить на ").replace("≈", " примерно ")
    s = re.sub(r"~\s*(?=\d)", "примерно ", s)
    s = s.replace("^", " в степени ")
    # emoji, check marks, arrows and other pictographs: espeak spells out their Unicode names
    s = "".join(ch for ch in s if (unicodedata.category(ch) not in ("So", "Sk", "Cs", "Co") or ch in "°№")
                and not ("\u2190" <= ch <= "\u21ff") and ch not in "*#$\\~_\u200d\ufe0f")
    # a line without final punctuation would run into the next one: end it as a sentence
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    s = " ".join(ln if ln[-1] in ".!?…:;," else ln + "." for ln in lines)
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+([.,!?;:])", r"\1", s)).strip()
