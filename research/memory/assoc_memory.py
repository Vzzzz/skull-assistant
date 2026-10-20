"""Associative memory prototype (docs/MEMORY_TRACK.md, MEM-1).

A record is one exchange: what the user said and what the assistant answered. Retrieval is hybrid:
cosine similarity of embeddings (ollama /api/embed, CPU-sized model) plus keyword overlap with crude
prefix stemming for Russian. Nothing here is persistent: the prototype lives for one dialog; the
service version (M6/M10) will put the same records into SQLite + sqlite-vec.
"""
import re
import time

import numpy as np
import requests

_WORD = re.compile(r"[а-яёa-z0-9]+")
STOP = set("и а но не ни да нет ты я он она мы вы они это то что как у в на с о от до за по из к ну же ли бы уже ещё еще мне тебе меня тебя его её ее их там тут вот так все всё про для или если когда где кто чем чём".split())


def tokens(text: str):
    return [w for w in _WORD.findall(text.lower().replace("ё", "е")) if w not in STOP and len(w) > 1]


def stems(text: str, n=5):
    """Poor man's Russian stemming: 5-char prefix (ключи/ключей/ключ -> 'ключи'/'ключе'/'ключ' -> cut to 5)."""
    return {w[:n] for w in tokens(text)}


def embed(host, model, texts, timeout=120):
    r = requests.post(f"{host}/api/embed", json={"model": model, "input": texts}, timeout=timeout)
    r.raise_for_status()
    vecs = np.asarray(r.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-8)


class AssocMemory:
    def __init__(self, host="http://127.0.0.1:11434", embed_model="embeddinggemma", mode="hybrid", kw_weight=0.5):
        """mode: 'embed' (cosine only), 'kw' (keyword overlap only), 'hybrid' (cos + kw_weight * jaccard)."""
        self.host, self.embed_model, self.mode, self.kw_weight = host, embed_model, mode, kw_weight
        self.records = []      # dicts: turn, user, assistant, text, stems
        self.vecs = None       # (n, d)
        self.embed_calls_s = []

    def _text(self, user, assistant):
        return f"Собеседник: {user}\nКостян: {assistant}" if assistant else f"Собеседник: {user}"

    def add(self, turn, user, assistant=""):
        text = self._text(user, assistant)
        rec = {"turn": turn, "user": user, "assistant": assistant, "text": text, "stems": stems(text)}
        self.records.append(rec)
        if self.mode != "kw":
            t0 = time.perf_counter()
            v = embed(self.host, self.embed_model, [text])
            self.embed_calls_s.append(time.perf_counter() - t0)
            self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])
        return rec

    def retrieve(self, query, k=3, exclude_turns=(), min_score=0.0):
        """Top-k records for the query. Returns list of (score, record, cos, kw) sorted by score desc."""
        if not self.records:
            return []
        n = len(self.records)
        cos = np.zeros(n, dtype=np.float32)
        if self.mode != "kw":
            t0 = time.perf_counter()
            q = embed(self.host, self.embed_model, [query])[0]
            self.embed_calls_s.append(time.perf_counter() - t0)
            cos = self.vecs @ q
        kw = np.zeros(n, dtype=np.float32)
        if self.mode != "embed":
            qs = stems(query)
            for i, rec in enumerate(self.records):
                if qs and rec["stems"]:
                    kw[i] = len(qs & rec["stems"]) / len(qs)   # fraction of query stems found in the record
        if self.mode == "embed":
            score = cos
        elif self.mode == "kw":
            score = kw
        else:
            score = cos + self.kw_weight * kw
        order = np.argsort(-score)
        out = []
        excl = set(exclude_turns)
        for i in order:
            rec = self.records[i]
            if rec["turn"] in excl or score[i] < min_score:
                continue
            out.append((float(score[i]), rec, float(cos[i]), float(kw[i])))
            if len(out) >= k:
                break
        return out

    @staticmethod
    def format_note(hits, max_reply_chars=160):
        """Text injected in front of the user's turn. Kept short and literal: the model should quote, not reason."""
        if not hits:
            return ""
        lines = ["[Память: из этого разговора раньше]"]
        for _, rec, _, _ in hits:
            a = rec["assistant"].replace("\n", " ")
            a = (a[:max_reply_chars] + "…") if len(a) > max_reply_chars else a
            lines.append(f"- (ход {rec['turn']}) собеседник: «{rec['user']}»" + (f" — ты ответил: «{a}»" if a else ""))
        lines.append("[Конец памяти]")
        return "\n".join(lines)
