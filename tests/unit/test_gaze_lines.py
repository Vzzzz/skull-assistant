"""core/gaze_lines.py (T-053): the repeat check over the whole line (core's answer to the 1г gate) and the message
with the look's frame for llama-server."""
import base64

import pytest

from core.gaze_lines import frame_message, is_repeat, stems, turn_circle

RECENT = ["Твоя куртка всё ещё здесь.", "Устал.", "Что задумал?"]


@pytest.mark.parametrize("line, repeat", [
    ("Устал.", True),                          # the gate: dropped twice and went silent — still a repeat
    ("Опять устал.", True),
    ("Что в планах?", False),                  # the gate: dropped by the first word «Что» — no longer
    ("Твой стол всё такой же.", False),        # 2 of 5 stems shared with the jacket line
    ("Твоя куртка всё ещё тут.", True),
    ("Холодно сегодня.", False),
])
def test_is_repeat(line, repeat):
    assert is_repeat(line, RECENT) is repeat


def test_no_recent_no_repeat():
    assert not is_repeat("Устал.", [])
    assert not is_repeat("", RECENT)


def test_stems():
    assert stems("Что задумал?") == {"что", "задум"}


def test_frame_message():
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"
    m = frame_message("Одна фраза о том, что видно.", jpeg)
    assert m["role"] == "user"
    text, image = m["content"]
    assert text == {"type": "text", "text": "Одна фраза о том, что видно."}
    url = image["image_url"]["url"]
    assert image["type"] == "image_url" and url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == jpeg


def test_turn_circle_frame_first_and_every_other():
    """T-060 (9): with a frame — frame, ask, frame, complain (gaze:turn from 0 after /reset); without — ask, complain."""
    c = turn_circle(True, "F", "ask", "complain")
    assert [c[k % len(c)] for k in range(6)] == ["F", "ask", "F", "complain", "F", "ask"]
    c = turn_circle(False, "F", "ask", "complain")
    assert [c[k % len(c)] for k in range(3)] == ["ask", "complain", "ask"]
