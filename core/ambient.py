"""Listening without a wake word (D-029, T-038 stage 2b): whose voice + conversation state + the duck pass -> respond / ack / silent.

Pure policy: time, the ENGAGED flag and the duck pass come from the caller, so services/brain/responder.py and the
replay in research/eval/ambient_bus.py run the same rules. Changes after the live test of 2026-10-27 (core's answers):
  - ENGAGED opens and is extended only by an answer to the name or to a question / request (not by the duck's «да» on a
    statement: one such «да» used to open a window in which everything was answered — 7 of 14 mutters);
  - a question / request by words (core.router.is_question) -> respond at once, in IDLE too (no pause wait, answer 5);
  - in ENGAGED nothing else is answered unconditionally: the duck with a lower bar (p(«да») >= engaged_yes_p);
    an `uncertain` voice counts as the owner's inside the window, outside it stays silent;
  - in IDLE the duck's «да» on a statement is `ack`, not an answer.
The window counts from the end of playback of the answer that opened it (engaged_until), not from the end of generation.
"""
import json
import re

from core.router import MIN_MARGIN, MIN_PROB, duck_from_no, is_question

PHONE = re.compile(r"\bалл[оё]\b", re.I)
OPEN_KEY = "ambient:open"      # Redis: {"reply_id", "t", "s"} of the answer that opened / extended ENGAGED
TTS_END = "tts:end"            # Redis hash reply_id -> expected end of its playback (services/tts/speaker.py)


def has_name(text):
    return "костян" in text.lower()


def engaged_until(r):
    """End of the ENGAGED window: engaged_s after the opening answer has been played (while it plays: open)."""
    raw = r.get(OPEN_KEY)
    if not raw:
        return 0.0
    o = json.loads(raw)
    if r.exists("tts:speaking"):
        return float("inf")
    end = r.hget(TTS_END, o["reply_id"])
    return (float(end) if end else o["t"]) + o["s"]


def needs_pause(text, speaker, engaged):
    """STT waits for the pause after a phrase only when the duck pass will use it: the owner's voice, idle, no name,
    no question word (other / uncertain voices outside the window are silent whatever the pause)."""
    return speaker == "owner" and not engaged and not has_name(text) and not is_question(text)


class Ambient:
    def __init__(self, engaged_s=20, busy_pause_s=20, veto_p=0.9, engaged_yes_p=0.3):
        self.engaged_s, self.busy_pause_s, self.veto_p, self.engaged_yes_p = engaged_s, busy_pause_s, veto_p, engaged_yes_p
        self.busy, self.last_utt = False, 0.0

    def reset(self):
        self.busy, self.last_utt = False, 0.0

    def decide(self, text, speaker, pause_s, p_aside, now, engaged, duck):
        """Returns (decision, why, opens): decision respond | ack | silent; opens — the answer opens or extends ENGAGED.
        duck(text, pause_s) -> {"ans_tok", "probs"} (core.router.duck_bin_llamaserver), called only when needed."""
        name = has_name(text)
        gap, self.last_utt = now - self.last_utt, now
        if self.busy and (name or gap >= self.busy_pause_s):
            self.busy = False
        if speaker == "other":
            return "silent", "other voice", False
        if p_aside >= self.veto_p and not name:
            self.busy = True
            return "silent", f"veto aside {p_aside:.2f}", False
        if PHONE.search(text) and not name:      # a call starts: the router does not see it in the text (T-038 stage 1)
            self.busy = True
            return "silent", "veto phone", False
        if self.busy:
            return "silent", "busy", False
        if name:
            return "respond", "name", True
        if speaker in ("uncertain", "unknown") and not engaged:
            return "silent", f"{speaker} voice", False
        if speaker == "unknown" and len(re.findall(r"\w+", text)) < 2:
            return "silent", "unknown fragment", False   # no embedding = too little voice: «Д», «С.» from the room
        if is_question(text):   # by words (step 7a); the duck often says «нет» to a bare question in a fresh conversation
            return "respond", "question", True
        dk = duck(text, pause_s)
        ans, probs = dk.get("ans_tok"), dk.get("probs") or {}
        tag = f"duck {ans} {probs}"
        if engaged:
            if probs.get("да", 0.0) >= self.engaged_yes_p:
                return "respond", f"engaged {tag}", False
            return self._quiet(text, pause_s, f"engaged {tag}")
        if ans is None:
            return "ack", f"{tag} no-label", False
        top = sorted(probs.values(), reverse=True) + [0.0]
        if top[0] < MIN_PROB or top[0] < MIN_MARGIN * top[1]:
            return "ack", f"{tag} low-confidence", False
        if ans == "да":
            return "ack", f"statement {tag}", False
        return self._quiet(text, pause_s, tag)

    @staticmethod
    def _quiet(text, pause_s, tag):
        return ("ack" if duck_from_no(text, pause_s) == "ack" else "silent"), tag, False
