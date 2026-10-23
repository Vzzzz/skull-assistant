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

    # One line for the system prompt when memory is on. v1 (T-026) put a transcript-like block into the user turn and
    # the model treated it as something the user pasted ("ты мне выгрузил архив") — persona −0.4. v2: the block is
    # phrased as the model's own recollection and the system prompt says so once.
    # v3 (T-028): the note goes AFTER the user's text in round brackets — v2 put it first and the model sometimes
    # copied the whole bracket as the beginning of its reply (4/60 and 1/60 turns at 2k); k=2 and shorter quotes.
    # v4 (T-029): no brackets at all — v3's "(из памяти: …)" made the model add its own asides in brackets
    # ("(Не надо проверять мою память.)") on half of the memory turns. A plain trailing sentence instead.
    # v5 (T-030): v3 placement/format is back — "(из памяти: …)" after the user's text was the only variant the model
    # never copied (T-028: 0/100 turns); v4's plain sentence was copied on half of the turns and polluted the memory
    # records. The remaining v3 defect — the model's own trailing asides in brackets — is removed deterministically
    # by clean_reply() below, as services/brain must do before TTS and before storing the exchange.
    SYSTEM_RULE = ("Иногда после реплики собеседника в круглых скобках стоит пометка «из памяти» — то, что ты сам помнишь "
                   "из этого разговора. Это не слова собеседника. Пользуйся этим молча: не повторяй текст в скобках, "
                   "не говори «из памяти», не обсуждай, откуда ты это знаешь, и не добавляй своих замечаний в скобках.")
    NOTE_PREFIX = "(из памяти: "

    _ASIDE = re.compile(r"\s*\((?:из памяти|ты помнишь|не надо|не буду|не жди|я не стану|примерно столько)[^)]*\)\s*$", re.I | re.S)
    _ANY_TRAILING_PAREN = re.compile(r"\n\s*\([^)]{15,}\)\s*$", re.S)

    @staticmethod
    def clean_reply(reply: str, note: str = ""):
        """Deterministic guard (brain-side): strip a copied memory note and trailing bracketed asides.
        Returns (clean, flags) where flags is a set of what fired: 'copied', 'aside'."""
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
        m = AssocMemory._ASIDE.search(out) or AssocMemory._ANY_TRAILING_PAREN.search(out)
        if m:
            out = out[:m.start()]; flags.add("aside")
        out = out.strip()
        return (out if out else reply.strip()), flags

    @staticmethod
    def format_note(hits, max_reply_chars=80, max_user_chars=140, chronological=True):
        """Text appended after the user's turn (v3/v5). Kept short and literal: the model should use it, not discuss it.
        v6 (T-036): records in time order with markers — T-030 showed the model picking the stale fact («ключи в куртке»)
        when the update («переложил в рюкзак») was listed without any notion of which came later."""
        if not hits:
            return ""
        recs = [h[1] for h in hits]
        if chronological:
            recs = sorted(recs, key=lambda r: r["turn"])
        parts = []
        n = len(recs)
        for i, rec in enumerate(recs):
            u = rec["user"].replace("\n", " ")
            u = (u[:max_user_chars] + "…") if len(u) > max_user_chars else u
            a = rec["assistant"].replace("\n", " ")
            a = (a[:max_reply_chars] + "…") if len(a) > max_reply_chars else a
            mark = ""
            if chronological and n > 1:
                mark = "сначала " if i == 0 else ("позже всего " if i == n - 1 else "потом ")
            parts.append(f"{mark}он говорил «{u}»" + (f", ты отвечал «{a}»" if a else ""))
        return AssocMemory.NOTE_PREFIX + "; ".join(parts) + ")"
