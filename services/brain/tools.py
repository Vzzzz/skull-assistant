"""The calc tool (T-015, D-016 p.3): the skull does not count in its head — brain computes and gives the model the result
as a note before its answer: «(вычислено: 17 × 23 = 391)». One model call, as with the memory note (T-036).

calc(expr)        arithmetic through `ast` — numbers, + - * / // % **, brackets, round / abs; never eval; None on error
extract(text)     a transcript (router label `calc`) -> (shown expression, result) or None:
                  percent of («15 % от 2400»), a discount («стоит 2400, скидка 15 %» -> 2040), clock + duration («выехал
                  в 8:40, ехать час тридцать пять» -> 10:15), plain arithmetic with operator words («17 умножить на 23»)
note(text)        -> «(вычислено: … = …)» or ""
Number words are turned into digits on a copy (core.numbers_ru.to_digits); the text the model sees is not changed.
"""
import ast
import operator
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.numbers_ru import normalize, to_digits

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_FUNCS = {"round": round, "abs": abs}
MAX_POW, MAX_ABS = 100, 1e15


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > MAX_POW:
            raise ValueError("power too large")
        v = _OPS[type(node.op)](a, b)
        if abs(v) > MAX_ABS:
            raise ValueError("too large")
        return v
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords and 1 <= len(node.args) <= 2):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"not allowed: {ast.dump(node)[:40]}")


def fmt(v):
    """391.0 -> «391», 2.5 -> «2,5» (Russian decimal comma), at most 2 decimals."""
    if isinstance(v, float):
        v = round(v, 2)
        if v == int(v):
            v = int(v)
    return str(v).replace(".", ",")


def calc(expr):
    """Arithmetic expression -> result (int / float) or None. «17*23» -> 391; «__import__('os')» -> None."""
    try:
        expr = re.sub(r"(?<=\d),(?=\d)", ".", expr.replace("×", "*").replace("÷", "/"))   # 2,5 -> 2.5; round(x, 2) stays
        return _eval(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
        return None


NUM = r"\d+(?:[.,]\d+)?"
_PCT_OF = re.compile(rf"({NUM})\s*(?:%|процент\w*)\s*(?:от|из)\s*({NUM})", re.I)
_DISCOUNT = re.compile(rf"скидк\w*\s*(?:в\s*)?({NUM})\s*(?:%|процент\w*)", re.I)
_PRICE = re.compile(rf"(?:стоит|цена|за)\s*({NUM})", re.I)
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_CUE = re.compile(r"(?:ехать|еду|едем|дорога|займ\w*|через|идти|лететь|добираться|в пути|потрачу|уйд[её]т)", re.I)
_HOURS = re.compile(rf"\s*(?:(полтора)\s*час\w*|(час)\b|({NUM})\s*час\w*)", re.I)
_MINS = re.compile(rf"\s*(?:и\s*)?(\d+)\s*(мин\w*)?", re.I)
_HMM = re.compile(r"\s*(\d{1,2})\s*:\s*(\d{2})\b")
_OPWORDS = [(r"умножить\s+на|умножить|помножить\s+на|×|\*|\bх\b|\bx\b", "*"), (r"разделить\s+на|делить\s+на|поделить\s+на|÷|/", "/"),
            (r"плюс|\+|прибавить", "+"), (r"минус|отнять|вычесть", "-"), (r"в\s+степени", "**")]
_SHOW = {"*": "×", "/": ":", "+": "+", "-": "−", "**": "^"}
# T-060 (6): the imperative, with the operand order of the phrase («вычти 5 из 20» = 20 − 5); applied before _OPWORDS
_IMPERATIVE = [(rf"(?:умножь|помножь)\s*({NUM})\s*на\s*({NUM})", r"\1 * \2"),
               (rf"(?:раздели|подели)\s*({NUM})\s*на\s*({NUM})", r"\1 / \2"),
               (rf"(?:прибавь|добавь)\s*({NUM})\s*к\s*({NUM})", r"\2 + \1"),
               (rf"\bк\s*({NUM})\s*(?:прибавь|добавь)\s*({NUM})", r"\1 + \2"),
               (rf"(?:отними|вычти)\s*({NUM})\s*(?:из|от)\s*({NUM})", r"\2 - \1"),
               (rf"\b(?:из|от)\s*({NUM})\s*(?:отними|вычти)\s*({NUM})", r"\1 - \2"),
               (r"\b(?:посчитай|сосчитай|подсчитай|вычисли)\b", " ")]


def _num(s):
    return float(s.replace(",", ".")) if ("," in s or "." in s) else int(s)


def _duration(t):
    """-> minutes after a duration cue, or None: «ехать 1:35», «ехать час 35 минут», «в пути 2 часа 15 минут»,
    «через полтора часа», «дорога 40 минут»."""
    for c in _CUE.finditer(t):
        rest = t[c.end():c.end() + 40]
        if (m := _HMM.match(rest)):
            return int(m.group(1)) * 60 + int(m.group(2))
        h, pos = None, 0
        if (m := _HOURS.match(rest)):
            h = 1.5 if m.group(1) else 1 if m.group(2) else _num(m.group(3))
            pos = m.end()
        mins = None
        if (m := _MINS.match(rest, pos)):
            if h is not None or m.group(2):           # after hours: minutes; alone: only with «мин»
                mins = int(m.group(1))
        total = (h or 0) * 60 + (mins or 0)
        if total:
            return int(total)
    return None


def extract(text):
    """-> (expression as shown, result as text) or None when no computation is found."""
    t = to_digits(normalize(text))
    t = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", t)                # «2 400» -> «2400»
    if (m := _PCT_OF.search(t)):
        p, base = _num(m.group(1)), _num(m.group(2))
        return f"{fmt(p)} % от {fmt(base)}", fmt(base * p / 100)
    if (d := _DISCOUNT.search(t)) and (pr := _PRICE.search(t)):
        p, price = _num(d.group(1)), _num(pr.group(1))
        return f"{fmt(price)} − {fmt(p)} %", fmt(price * (1 - p / 100))
    if (c := _CLOCK.search(t)) and (mins := _duration(t[:c.start()] + " " + t[c.end():])):
        start = int(c.group(1)) * 60 + int(c.group(2))
        end = (start + mins) % (24 * 60)
        return f"{c.group(1)}:{c.group(2)} + {mins // 60}:{mins % 60:02d}", f"{end // 60}:{end % 60:02d}"
    e = t.lower().replace("ё", "е")
    for pat, rep in _IMPERATIVE:
        e = re.sub(pat, rep, e)
    for pat, op in _OPWORDS:
        e = re.sub(pat, f" {op} ", e)
    best = None
    for m in re.finditer(rf"\(?\s*{NUM}(?:\s*(?:\*\*|[-+*/])\s*\(?\s*{NUM}\s*\)?)+", e):
        expr = re.sub(r"\s+", "", m.group(0)).replace(",", ".")
        if (v := calc(expr)) is not None and (best is None or len(expr) > len(best[0])):
            best = (expr, v)
    if best:
        shown = re.sub(r"\*\*|[-+*/]", lambda o: f" {_SHOW[o.group(0)]} ", best[0]).replace(".", ",")
        return shown, fmt(best[1])
    return None


def note(text):
    r = extract(text)
    return f"(вычислено: {r[0]} = {r[1]})" if r else ""
