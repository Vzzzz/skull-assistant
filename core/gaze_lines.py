"""Lines on a silent look (T-031 step 1в, T-053): the repeat check over the whole line and the message with a frame.

Repeat (core's answer to the 1г gate): the first word was too coarse — «Что в планах?» was dropped after «Что задумал?»
and both attempts went silent. Now the stems of all the words of the line are compared with each of the last RECENT
lines: a repeat when the shared stems cover at least OVERLAP of the shorter line («Устал.» = «Устал.»; «Твой стол всё
такой же.» against «Твоя куртка всё ещё здесь.» is 2 of 5 — not a repeat).
"""
import base64
import re

from core.modes import stem

RECENT, OVERLAP = 5, 0.6
_WORD = re.compile(r"[а-яёa-z0-9]+", re.I)


def stems(line):
    return {stem(w) for w in _WORD.findall(line)}


def is_repeat(line, recent, overlap=OVERLAP):
    """True when `line` shares >= overlap of the shorter line's stems with one of `recent`."""
    s = stems(line)
    for x in recent:
        t = stems(x)
        if s and t and len(s & t) / min(len(s), len(t)) >= overlap:
            return True
    return False


def frame_message(text, jpeg):
    """One user message for llama-server /v1/chat/completions: the text and the frame as a data URL (mmproj needed)."""
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    return {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]}
