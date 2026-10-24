"""Dialog memory shared by services (docs/MEMORY_TRACK.md, D-026).

Ported from research/memory/assoc_memory.py (T-002: reply filter; T-016: associative memory).
A record is one exchange (the user's turn and the filtered reply). Retrieval is hybrid: cosine of embeddings
from a llama-server started with --embedding (OpenAI /v1/embeddings) plus keyword overlap with crude prefix
stemming for Russian. Records live in the brain's Redis history; vectors are cached by the caller.
"""
import re
import time

import numpy as np
import requests

NOTE_PREFIX = "(из памяти: "
# one line for the system prompt when memory is on (v5/v6, T-030)
SYSTEM_RULE = ("Иногда после реплики собеседника в круглых скобках стоит пометка «из памяти» — то, что ты сам помнишь "
               "из этого разговора. Это не слова собеседника. Пользуйся этим молча: не повторяй текст в скобках, "
               "не говори «из памяти», не обсуждай, откуда ты это знаешь, и не добавляй своих замечаний в скобках.")

_WORD = re.compile(r"[а-яёa-z0-9]+")
STOP = set("и а но не ни да нет ты я он она мы вы они это то что как у в на с о от до за по из к ну же ли бы уже ещё еще мне тебе меня тебя его её ее их там тут вот так все всё про для или если когда где кто чем чём".split())

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


def tokens(text: str):
    return [w for w in _WORD.findall(text.lower().replace("ё", "е")) if w not in STOP and len(w) > 1]


def stems(text: str, n=5):
    """Poor man's Russian stemming: 5-char prefix (ключи/ключей/ключ -> 'ключи'/'ключе'/'ключ' -> cut to 5)."""
    return {w[:n] for w in tokens(text)}


def embed(host, texts, timeout=60):
    r = requests.post(f"{host}/v1/embeddings", json={"input": texts}, timeout=timeout)
    r.raise_for_status()
    data = sorted(r.json()["data"], key=lambda d: d["index"])
    vecs = np.asarray([d["embedding"] for d in data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-8)


def format_note(hits, max_reply_chars=80, max_user_chars=140):
    """Text appended after the user's turn (v6, T-036): records in time order with markers, otherwise the model
    picks the stale fact («ключи в куртке») over the update («переложил в рюкзак»)."""
    if not hits:
        return ""
    recs = sorted((h[1] for h in hits), key=lambda r: r["turn"])
    parts = []
    n = len(recs)
    for i, rec in enumerate(recs):
        u = rec["user"].replace("\n", " ")
        u = (u[:max_user_chars] + "…") if len(u) > max_user_chars else u
        a = rec["assistant"].replace("\n", " ")
        a = (a[:max_reply_chars] + "…") if len(a) > max_reply_chars else a
        mark = ""
        if n > 1:
            mark = "сначала " if i == 0 else ("позже всего " if i == n - 1 else "потом ")
        parts.append(f"{mark}он говорил «{u}»" + (f", ты отвечал «{a}»" if a else ""))
    return NOTE_PREFIX + "; ".join(parts) + ")"


class AssocMemory:
    def __init__(self, embed_host, mode="hybrid", kw_weight=0.5):
        """mode: 'embed' (cosine only), 'kw' (keyword overlap only), 'hybrid' (cos + kw_weight * overlap)."""
        self.embed_host, self.mode, self.kw_weight = embed_host, mode, kw_weight
        self.records = []      # dicts: turn, user, assistant, stems
        self.vecs = None       # (n, d)
        self.embed_s = []

    @staticmethod
    def text(user, assistant, max_chars=4000):
        """Record text for the embedder; capped below embeddinggemma's 2048-token context (~2.3 chars/token)."""
        t = f"Собеседник: {user}\nКостян: {assistant}" if assistant else f"Собеседник: {user}"
        return t[:max_chars]

    def _embed(self, text):
        t0 = time.perf_counter()
        v = embed(self.embed_host, [text])
        self.embed_s.append(time.perf_counter() - t0)
        return v

    def add(self, turn, user, assistant="", vec=None):
        """vec: cached embedding of text(user, assistant), or None to compute it. Returns the vector (1, d) or None."""
        text = self.text(user, assistant)
        v = None
        if self.mode != "kw":   # embed first: a failed call must not leave a record without a vector
            v = self._embed(text) if vec is None else np.asarray(vec, dtype=np.float32).reshape(1, -1)
            self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])
        self.records.append({"turn": turn, "user": user, "assistant": assistant, "stems": stems(text)})
        return v

    def retrieve(self, query, k=3, exclude_turns=(), min_score=0.0):
        """Top-k records for the query. Returns list of (score, record, cos, kw) sorted by score desc."""
        excl = set(exclude_turns)
        if not any(rec["turn"] not in excl for rec in self.records):
            return []          # nothing outside the window: skip the embedding call
        n = len(self.records)
        cos = np.zeros(n, dtype=np.float32)
        if self.mode != "kw":
            cos = self.vecs @ self._embed(query)[0]
        kw = np.zeros(n, dtype=np.float32)
        if self.mode != "embed":
            qs = stems(query)
            for i, rec in enumerate(self.records):
                if qs and rec["stems"]:
                    kw[i] = len(qs & rec["stems"]) / len(qs)   # fraction of query stems found in the record
        score = cos if self.mode == "embed" else kw if self.mode == "kw" else cos + self.kw_weight * kw
        out = []
        for i in np.argsort(-score):
            rec = self.records[i]
            if rec["turn"] in excl or score[i] < min_score:
                continue
            out.append((float(score[i]), rec, float(cos[i]), float(kw[i])))
            if len(out) >= k:
                break
        return out
