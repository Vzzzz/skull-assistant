"""Russian numbers in transcripts (T-015, D-016 p.3, D-025 lever 12).

The production STT (GigaAM v3 punct, T-041) already writes most numbers as digits, so the text that brain and the memory
see gets only a minimal normalization (`normalize`):
  * clock time in words -> HH:MM: «в половине двенадцатого» -> «в 11:30», «без четверти восемь» -> «07:45», «четверть
    девятого» -> «08:15», «в восемь сорок» -> «в 08:40» (hour + minutes only after a time cue: в / к / до / после / с /
    около / на, so «ехать час тридцать пять» stays — that is a duration);
  * a numeral at the start of the phrase -> digits: «Пятнадцать процентов от 2 400» -> «15 процентов от 2 400»;
  * a code dictated digit by digit: >= 3 single digits in a row -> one number («4 7 1 2» / «четыре семь один два» -> «4712»).
Nothing else is touched; normalize(normalize(x)) == normalize(x).

`to_digits` turns every run of number words into digits — only for extracting an expression for the calc tool
(services/brain/tools.py), never for the text the model or the memory sees.
"""
import re

UNITS = {"ноль": 0, "один": 1, "одна": 1, "одно": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
         "шесть": 6, "семь": 7, "восемь": 8, "девять": 9}
TEENS = {"десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
         "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19}
TENS = {"двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70,
        "восемьдесят": 80, "девяносто": 90}
HUNDREDS = {"сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500, "шестьсот": 600,
            "семьсот": 700, "восемьсот": 800, "девятьсот": 900}
SCALES = {"тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "тысячу": 1000,
          "миллион": 10 ** 6, "миллиона": 10 ** 6, "миллионов": 10 ** 6}
# hour in the genitive ordinal: «в половине двенадцатого» = 11:30, «без четверти восемь» uses the cardinal
ORD_GEN = {"первого": 1, "второго": 2, "третьего": 3, "четвёртого": 4, "четвертого": 4, "пятого": 5, "шестого": 6,
           "седьмого": 7, "восьмого": 8, "девятого": 9, "десятого": 10, "одиннадцатого": 11, "двенадцатого": 12}
HOUR_WORDS = {**UNITS, **TEENS, "двадцать": 20, "час": 1}
TIME_CUES = {"в", "к", "до", "после", "с", "около", "на", "со"}

_WORD = re.compile(r"[А-Яа-яЁё]+|\d+(?:[.,:]\d+)?|\S")


def _norm(w):
    return w.lower().replace("ё", "е") if w else w


def _lex(w):
    return {k.replace("ё", "е"): v for k, v in w.items()}


UNITS, TEENS, TENS, HUNDREDS, SCALES, ORD_GEN, HOUR_WORDS = map(_lex, (UNITS, TEENS, TENS, HUNDREDS, SCALES, ORD_GEN, HOUR_WORDS))
NUMBER_WORDS = set(UNITS) | set(TEENS) | set(TENS) | set(HUNDREDS) | set(SCALES)


def parse_number(words, i=0):
    """Number words from words[i] on (lower case, ё -> е) -> (value, next index), or (None, i) when there is none.
    «две тысячи четыреста» -> 2400; «двадцать три» -> 23; «тысяча» -> 1000."""
    total, cur, j, seen = 0, 0, i, False
    while j < len(words):
        w = words[j]
        if w in HUNDREDS and cur % 1000 < 100:
            cur += HUNDREDS[w]
        elif w in TENS and cur % 100 == 0:
            cur += TENS[w]
        elif w in TEENS and cur % 100 == 0:
            cur += TEENS[w]
        elif w in UNITS and cur % 10 == 0 and not (seen and cur % 100 == 0 and cur and w == "ноль"):
            if seen and cur == 0 and total == 0:
                break                       # «ноль ноль» is two numbers
            cur += UNITS[w]
        elif w in SCALES and (cur or not seen):
            total += (cur or 1) * SCALES[w]
            cur = 0
        else:
            break
        seen, j = True, j + 1
        if w in UNITS and j < len(words) and words[j] in UNITS:
            break                           # «четыре семь» — two digits of a code, not one number
    return (total + cur, j) if seen else (None, i)


def _tokens(text):
    return [(m.start(), m.end(), m.group(0)) for m in _WORD.finditer(text)]


def _replace(text, spans):
    """spans: (start, end, replacement), non-overlapping."""
    out, pos = [], 0
    for s, e, rep in sorted(spans):
        out.append(text[pos:s]); out.append(rep); pos = e
    return "".join(out) + text[pos:]


def to_digits(text):
    """Every run of number words -> digits (for the calc extractor only)."""
    toks = _tokens(text)
    words = [_norm(t[2]) for t in toks]
    spans, i = [], 0
    while i < len(toks):
        val, j = parse_number(words, i) if words[i] in NUMBER_WORDS else (None, i)
        if val is None:
            i += 1
            continue
        spans.append((toks[i][0], toks[j - 1][1], str(val)))
        i = j
    return _replace(text, spans)


def _hhmm(h, m):
    return f"{h % 24:02d}:{m:02d}"


def _times(text):
    toks = _tokens(text)
    w = [_norm(t[2]) for t in toks]
    spans, i = [], 0
    while i < len(toks):
        # «в половине двенадцатого», «половина девятого»
        if w[i] in ("половине", "половина", "пол") and i + 1 < len(w) and w[i + 1] in ORD_GEN:
            spans.append((toks[i][0], toks[i + 1][1], _hhmm(ORD_GEN[w[i + 1]] - 1, 30))); i += 2; continue
        # «четверть девятого»
        if w[i] == "четверть" and i + 1 < len(w) and w[i + 1] in ORD_GEN:
            spans.append((toks[i][0], toks[i + 1][1], _hhmm(ORD_GEN[w[i + 1]] - 1, 15))); i += 2; continue
        # «без четверти восемь», «без десяти девять»
        if w[i] == "без" and i + 2 < len(w):
            if w[i + 1] == "четверти":
                mins, k = 15, i + 2
            else:
                mins, k = {"пяти": 5, "десяти": 10, "двадцати": 20}.get(w[i + 1]), i + 2
            if mins and k < len(w) and w[k] in HOUR_WORDS and HOUR_WORDS[w[k]] <= 12:
                spans.append((toks[i][0], toks[k][1], _hhmm(HOUR_WORDS[w[k]] - 1, 60 - mins))); i = k + 1; continue
        # «в восемь сорок», «к девяти» is left alone (no minutes)
        if w[i] in TIME_CUES and i + 2 < len(w) and w[i + 1] in HOUR_WORDS:
            h, j = parse_number(w, i + 1) if w[i + 1] != "час" else (1, i + 2)
            if h is not None and h <= 23 and j < len(w) and (w[j] in TENS or w[j] in TEENS or w[j] == "ноль"):
                m, k = parse_number(w, j) if w[j] != "ноль" else (None, j)
                if w[j] == "ноль" and j + 1 < len(w) and w[j + 1] in UNITS:     # «восемь ноль пять» -> 08:05
                    m, k = UNITS[w[j + 1]], j + 2
                if m is not None and m < 60:
                    spans.append((toks[i + 1][0], toks[k - 1][1], _hhmm(h, m))); i = k; continue
        i += 1
    return _replace(text, spans)


def _leading(text):
    toks = _tokens(text)
    if not toks or _norm(toks[0][2]) not in NUMBER_WORDS:
        return text
    words = [_norm(t[2]) for t in toks]
    val, j = parse_number(words, 0)
    if val is None:
        return text
    return text[:toks[0][0]] + str(val) + text[toks[j - 1][1]:]


_CODE = re.compile(r"(?<![\d.,:])\d(?:\s+\d){2,}(?!\d|[.,:]\d)")   # «4 7 1 2,» — not «4 7 1.5»


def _word_codes(text):
    """>= 3 single-digit number words in a row (a dictated code) -> digits."""
    toks = _tokens(text)
    w = [_norm(t[2]) for t in toks]
    spans, i = [], 0
    while i < len(toks):
        j = i
        while j < len(toks) and w[j] in UNITS:
            j += 1
        if j - i >= 3:
            spans.append((toks[i][0], toks[j - 1][1], "".join(str(UNITS[x]) for x in w[i:j])))
        i = max(j, i + 1)
    return _replace(text, spans)


def normalize(text):
    """Minimal normalization of a transcript (see the module doc); idempotent."""
    text = _times(text)
    text = _word_codes(text)
    text = _leading(text)
    return _CODE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), text)
