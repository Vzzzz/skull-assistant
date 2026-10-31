"""Voice control of the thinking mode (T-046): the owner's phrase sends the skull to mode B and back (D-030 dispatcher).

A phrase from the manifest (`params.modes.enter` / `exit`) matches anywhere in the turn: its first word whole, in the
imperative («подумай» / «подумайте», «возвращайся» / «возвращайтесь»), the rest by word stems, like the question
detector: «Костян, подумайте как следует: почему небо синее?» contains «подумай как следует», «Надо подумать, как следует
покрасить» does not (a false switch costs 20-30 s of silence, a missed one only a repeat). The rest of the turn is the
deferred question, asked to model B once the switch is done; the part after the phrase wins, else the part before it
(«Почему небо синее? Подумай как следует.»), the name and fillers not counted.
"""
import re

_WORD = re.compile(r"[а-яёa-z]+", re.I)
_EMPTY = {"костян", "ну", "а", "и", "так", "слушай", "давай", "пожалуйста", "эй"}


def _norm(w):
    return w.lower().replace("ё", "е")


def stem(word):
    """A crude Russian stem: drop up to two final letters, keep at least three («подумай» -> «подум», «мозги» -> «моз»)."""
    w = _norm(word)
    return w[:max(3, len(w) - 2)]


def imperative(word):
    """The singular and plural imperative of a phrase's first word: «подумай» -> {подумай, подумайте},
    «возвращайся» -> {возвращайся, возвращайтесь}; other words («хватит», «достаточно») get «-те» too, harmlessly."""
    w = _norm(word)
    return {w, w[:-2] + "тесь"} if w.endswith(("ся", "сь")) else {w, w + "те"}


def find_phrase(text, phrases):
    """-> (phrase, start, end) of the first phrase found in consecutive words of the text, else None: the first word
    in one of its imperative forms, the others opening with their stems."""
    words = [(m.start(), m.end(), _norm(m.group(0))) for m in _WORD.finditer(text)]
    for phrase in phrases:
        first, *rest = _WORD.findall(phrase)
        forms, stems = imperative(first), [stem(w) for w in rest]
        for i in range(len(words) - len(stems)):
            if words[i][2] in forms and all(words[i + 1 + j][2].startswith(s) for j, s in enumerate(stems)):
                return phrase, words[i][0], words[i + len(stems)][1]
    return None


def _content(part):
    words = [w for w in _WORD.findall(part) if _norm(w) not in _EMPTY and not _norm(w).startswith("костян")]
    return len(words) >= 2


def _trim(part):
    """Leading name, fillers and punctuation off: «и скажи, сколько…» -> «скажи, сколько…»."""
    part = part.strip(" \t:;,.!—-–")
    while (m := re.match(r"(\w+)[\s,]*", part)) and (_norm(m.group(1)) in _EMPTY or _norm(m.group(1)).startswith("костян")):
        part = part[m.end():]
    return part.strip(" \t:;,.!—-–")


def deferred_question(text, span):
    """The rest of the turn around the matched phrase, or '' when it carries nothing to ask."""
    _, start, end = span
    after, before = _trim(text[end:]), _trim(text[:start])
    if _content(after):
        return after
    return before if _content(before) else ""
