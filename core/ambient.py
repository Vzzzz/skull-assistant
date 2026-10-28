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
2в (core's answers 2026-10-28): any answer with content inside the window extends it; `unknown` is silent in the window
too; «угу» not more often than once in ack_gap_s; a voice found only by a window inside a segment of broadcast
(`lifted`, services/stt/voiceid.py) counts as `uncertain` only together with the name.
2в, answers to the live report (2026-10-29): BUSY ends after busy_pause_s without the *owner's* voice (a podcast never
leaves a 20-s gap between segments); the text veto `aside` does not apply to a question in the owner's voice (a
question «about the podcast» was vetoed at 0.90); while a broadcast plays (`bcast`, STT: an `other` segment in the last
30 s) an `uncertain` voice counts only with the name, also inside the window. A statement inside the window (no name,
no question) is answered only at p(«да») >= engaged_yes_p (0.7), 0.3..0.7 is «угу» («Так, это надо переделать» at 0.827
was answered as the window used to let 0.3 through, music 2в).
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
    def __init__(self, engaged_s=20, busy_pause_s=20, veto_p=0.9, engaged_yes_p=0.7, ack_gap_s=20, engaged_ack_p=0.3):
        self.engaged_s, self.busy_pause_s, self.veto_p, self.engaged_yes_p = engaged_s, busy_pause_s, veto_p, engaged_yes_p
        self.ack_gap_s, self.engaged_ack_p = ack_gap_s, engaged_ack_p
        self.busy, self.last_own, self.last_ack = False, 0.0, -1e9

    def reset(self):
        self.busy, self.last_own, self.last_ack = False, 0.0, -1e9

    def decide(self, text, speaker, pause_s, p_aside, now, engaged, duck, lifted=False, bcast=False):
        """Returns (decision, why, opens): decision respond | ack | silent; opens — the answer opens or extends ENGAGED.
        duck(text, pause_s) -> {"ans_tok", "probs"} (core.router.duck_bin_llamaserver), called only when needed.
        lifted — the voice class came from a window inside the segment, not from the whole segment."""
        decision, why, opens = self._decide(text, speaker, pause_s, p_aside, now, engaged, duck, lifted, bcast)
        if decision == "ack":
            if now - self.last_ack < self.ack_gap_s:
                return "silent", f"{why} (ack {now - self.last_ack:.0f} s ago)", False
            self.last_ack = now
        return decision, why, opens

    def _decide(self, text, speaker, pause_s, p_aside, now, engaged, duck, lifted, bcast):
        name = has_name(text)
        gap = now - self.last_own                # the owner's silence: other voices do not count (2в)
        if speaker in ("owner", "uncertain"):
            self.last_own = now
        if self.busy and (name or gap >= self.busy_pause_s):
            self.busy = False
        if speaker == "other":
            return "silent", "other voice", False
        own_question = speaker == "owner" and not lifted and is_question(text)   # not vetoed by the text (2в)
        if p_aside >= self.veto_p and not name and not own_question:
            self.busy = self.busy or speaker == "owner"   # only the owner's own call / aside means «busy»: a podcast
            return "silent", f"veto aside {p_aside:.2f}", False   # line in an uncertain voice is just silent (2в live)
        if PHONE.search(text) and not name:      # a call starts: the router does not see it in the text (T-038 stage 1)
            self.busy = True
            return "silent", "veto phone", False
        if self.busy:
            return "silent", "busy", False
        if name:
            return "respond", "name", True
        if speaker == "unknown":                 # no embedding: too little voice, or mostly noise / broadcast
            return "silent", "unknown voice", False
        if speaker == "uncertain" and (lifted or bcast or not engaged):
            return "silent", "uncertain voice" + (" (window)" if lifted else " (broadcast)" if bcast else ""), False
        if is_question(text):   # by words (step 7a); the duck often says «нет» to a bare question in a fresh conversation
            return "respond", "question", True
        dk = duck(text, pause_s)
        ans, probs = dk.get("ans_tok"), dk.get("probs") or {}
        tag = f"duck {ans} {probs}"
        if engaged:
            if probs.get("да", 0.0) >= self.engaged_yes_p:
                return "respond", f"engaged {tag}", True
            if probs.get("да", 0.0) >= self.engaged_ack_p:
                return "ack", f"engaged {tag}", False
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
