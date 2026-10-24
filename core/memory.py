"""Dialog memory helpers shared by services (docs/MEMORY_TRACK.md, D-026).

Ported from research/memory/assoc_memory.py (T-002: reply filter; T-016: associative memory).
"""
import re

NOTE_PREFIX = "(из памяти: "

_ASIDE = re.compile(r"\s*\((?:из памяти|ты помнишь|не надо|не буду|не жди|я не стану|примерно столько)[^)]*\)\s*$", re.I | re.S)
_ANY_TRAILING_PAREN = re.compile(r"\n\s*\([^)]{15,}\)\s*$", re.S)


def clean_reply(reply: str, note: str = ""):
    """Deterministic guard (brain-side): strip a copied memory note and trailing bracketed asides.
    Applied before TTS, before the history and before the memory. Returns (clean, flags) where flags is
    a set of what fired: 'copied', 'aside'."""
    flags = set()
    out = reply
    if note:
        for probe in (note, note.rstrip(")"), note.rstrip(").")):
            if probe and probe in out:
                out = out.replace(probe, ""); flags.add("copied")
        low = out.lower()
        for marker in ("[ты помнишь", "ты помнишь:", "(из памяти", "из памяти:"):
            i = low.find(marker)
            if i >= 0:
                # drop from the marker to the end of that paragraph
                j = out.find("\n\n", i)
                out = (out[:i] + (out[j:] if j >= 0 else "")); low = out.lower(); flags.add("copied")
    m = _ASIDE.search(out) or _ANY_TRAILING_PAREN.search(out)
    if m:
        out = out[:m.start()]; flags.add("aside")
    out = out.strip()
    return (out if out else reply.strip()), flags
