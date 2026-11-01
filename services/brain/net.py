"""Checking in the net (T-051, U-002, D-031): off by default, on by config or once by the owner's phrase.

search(q)        -> [{title, snippet, url}] (k=3), [] on a timeout or any error; one log line per outgoing request
excerpts(res)    -> «Выдержки из сети (…): 1) Заголовок: текст (домен) …», at most MAX_CHARS (~600 tokens)
query_of(text)   -> the search words of a question (question words, the name, fillers dropped)
personal(text)   -> a question about the owner himself (мой / мне / у меня…): never to the net
Gate             -> Net(params, redis): the policy (T-054) `always | ask | never` — Redis net:policy over the manifest's
                    params.net.policy (switch without a restart: skull-stand.sh net …), the old `enabled` migrates
                    (true -> always, false -> ask); Net.decide() -> search / ask / offline / refuse; a one-shot phrase
                    («проверь в сети», «погугли» — by stems, like core/modes.py) is consent in advance; under `never` no
                    request leaves Net at all (REQUESTS counts the outgoing ones)

  .venv/bin/python services/brain/net.py policy [always|ask|never|status]   # the switch (skull-stand.sh net …)

Backends (manifest params.net.backend): `wikipedia` — ru.wikipedia.org API, search -> top-3 -> intro extracts (core,
2026-11-01: a reference book, not the internet — weather, prices, news are not there and the skull says so);
`searxng` — the JSON API of an instance at params.net.url (interface only; no instance on the stand). DuckDuckGo HTML
answers the stand with a bot check (I-22) and is not used.
"""
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.modes import find_phrase

# Wikimedia UA policy: name + a contact; without one the API answered HTTP 429 after ~14 requests a minute (T-051)
UA = "skull-assistant/0.1 (https://github.com/Vzzzz/skull-assistant; offline voice assistant research) python-requests"
WIKI_API = "https://ru.wikipedia.org/w/api.php"
MAX_CHARS, PER_ITEM = 2000, 600
ONCE_KEY, ONCE_S = "net:once", 60
POLICY_KEY, POLICIES = "net:policy", ("always", "ask", "never")
ASK_S = 15                    # `ask`: the owner's yes / no to «Глянуть в сети?» within this, else offline
REQUESTS = 0                  # outgoing requests of this process (the `never` test: stays 0)
_YES = ["да", "давай", "глянь", "посмотри", "ага", "угу", "конечно", "валяй", "проверь", "поищи", "загляни", "можно"]
_NO = ["нет", "не надо", "не нужно", "сам", "неа", "не стоит", "обойдёмся", "отбой"]
PHRASES = ["проверь в сети", "посмотри в интернете", "погугли", "поищи в интернете", "проверь в интернете"]
_STOP = {"кто", "что", "чего", "чем", "как", "какой", "какая", "какое", "какие", "каков", "сколько", "где", "куда",
         "когда", "почему", "зачем", "ли", "ты", "вы", "знаешь", "скажи", "расскажи", "подскажи", "а", "ну", "и", "же",
         "вроде", "его", "её", "их", "у", "нас", "меня", "мне", "это", "такое", "такой", "сейчас", "вообще", "там",
         "костян", "слушай", "интересно", "помнишь", "о", "об", "про", "в", "на", "был", "была", "было"}
_WORD = re.compile(r"[А-Яа-яЁёA-Za-z0-9-]+")


def log(msg):
    print(f"net: {msg}", flush=True)


def query_of(text):
    """«Кто изобрёл гравиметр?» -> «изобрёл гравиметр»; empty when nothing is left."""
    words = [w for w in _WORD.findall(text) if w.lower().replace("ё", "е") not in _STOP and not w.lower().startswith("костян")]
    return " ".join(words)


_PERSONAL = re.compile(r"\b(?:я|мне|меня|мной|мой|моя|моё|мое|мои|моего|моей|моему|моим|моих|мою|нам|наш|наша|наше|наши)\b", re.I)


def personal(text):
    """A question about the owner himself («какого цвета мой новый чайник?») is answered from the dialog and the memory,
    never from a reference book (T-051: the gate's B question went to the net and the answer would be «не нашёл»)."""
    return bool(_PERSONAL.search(text))


def _domain(url):
    return urlparse(url).netloc or url


def _json(resp):
    """The body as JSON; a non-JSON answer (an HTML error / rate-limit page) -> ValueError with the HTTP status."""
    try:
        return resp.json()
    except ValueError:
        raise ValueError(f"HTTP {getattr(resp, 'status_code', '?')}, not JSON") from None


def _wikipedia(q, k, timeout_s, session):
    """One request: search as a generator + intro extracts of the found pages (half the calls of search-then-extracts —
    the rate limit counts requests)."""
    d = _json(session.get(WIKI_API, params={"action": "query", "generator": "search", "gsrsearch": q, "gsrlimit": k,
                                            "prop": "extracts", "exintro": 1, "explaintext": 1, "exchars": 1200,
                                            "exlimit": k, "format": "json", "utf8": 1}, timeout=timeout_s))
    pages = sorted(((d.get("query") or {}).get("pages") or {}).values(), key=lambda p: p.get("index", 99))
    return [{"title": p["title"], "snippet": p["extract"].strip(), "url": "https://ru.wikipedia.org/wiki/" + p["title"].replace(" ", "_")}
            for p in pages if (p.get("extract") or "").strip()]


def _searxng(q, k, timeout_s, session, url):
    d = _json(session.get(url.rstrip("/") + "/search", params={"q": q, "format": "json", "language": "ru"}, timeout=timeout_s))
    return [{"title": x.get("title", ""), "snippet": x.get("content", ""), "url": x.get("url", "")}
            for x in d.get("results", [])[:k] if x.get("content")]


def search(q, k=3, timeout_s=4.0, backend="wikipedia", url=None, session=None):
    """-> [{title, snippet, url}]; [] on a timeout, an HTTP / JSON error or an empty query (never raises)."""
    if not q.strip():
        return []
    global REQUESTS
    session = session or requests.Session()
    session.headers.update({"User-Agent": UA})
    t0 = time.time()
    REQUESTS += 1
    try:
        res = _searxng(q, k, timeout_s, session, url) if backend == "searxng" else _wikipedia(q, k, timeout_s, session)
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        log(f"{backend} {q!r} -> error {type(e).__name__}: {str(e)[:80]} ({time.time() - t0:.2f} s)")
        return []
    log(f"{backend} {q!r} -> {len(res)} ({time.time() - t0:.2f} s)")
    return res


def excerpts(results, source="Википедия"):
    """The note for the model: numbered excerpts with their domain, at most MAX_CHARS."""
    parts, used = [], 0
    for i, x in enumerate(results, 1):
        text = re.sub(r"\s+", " ", x["snippet"])[:PER_ITEM].rsplit(" ", 1)[0] if len(x["snippet"]) > PER_ITEM else x["snippet"]
        item = f"{i}) {x['title']}: {text} ({_domain(x['url'])})"
        if used + len(item) > MAX_CHARS:
            break
        parts.append(item)
        used += len(item)
    return f"Выдержки из сети ({source}):\n" + "\n".join(parts) if parts else ""


def note_for_memory(question, results, reply):
    """«проверял в сети: <вопрос> → <суть> (<домен>)» — the memory record's source (T-036 format)."""
    if not results:
        return f"проверял в сети: {question} → не нашёл"
    gist = re.split(r"(?<=[.!?])\s", reply.strip(), maxsplit=1)[0][:120]
    return f"проверял в сети: {question} → {gist} ({_domain(results[0]['url'])})"


def yesno(text):
    """The owner's answer to «Глянуть в сети?» by rules: 'yes' / 'no' / None (unclear -> the model's yes/no probs)."""
    t = " " + re.sub(r"[^\w\s]", " ", text.lower().replace("ё", "е")) + " "
    no = any(f" {w.replace('ё', 'е')} " in t for w in _NO)
    yes = any(f" {w} " in t for w in _YES)
    return "no" if no else "yes" if yes else None


class Net:
    """The gate (D-031, T-054): the policy decides; under `never` nothing leaves, whatever the caller does."""

    def __init__(self, params, r):
        self.p, self.r = params or {}, r

    def policy(self):
        """-> (policy, source): Redis net:policy, else the manifest's policy, else the old `enabled`, else ask."""
        v = self.r.get(POLICY_KEY)
        v = v.decode() if isinstance(v, bytes) else v
        if v in POLICIES:
            return v, "redis"
        if self.p.get("policy") in POLICIES:
            return self.p["policy"], "manifest"
        if "enabled" in self.p:
            return ("always" if self.p["enabled"] else "ask"), "manifest (enabled, migrated)"
        return "ask", "default"

    def once_phrase(self, text):
        """The owner asked to check in the net: consent for this or the next question (ONCE_S). -> the span or None."""
        span = find_phrase(text, self.p.get("phrases", PHRASES))
        if span and self.policy()[0] != "never":
            self.r.set(ONCE_KEY, 1, ex=self.p.get("once_s", ONCE_S))
        return span

    def decide(self, world, personal_q=False, consent=None):
        """world — a question about the world (router `world` in A, a question in B); personal_q — about the owner;
        consent — 'yes' / 'no' from the owner (an answer to «Глянуть в сети?» or the one-shot phrase), None — none.
        -> 'search' | 'ask' | 'offline' | 'refuse' ('refuse': under never the owner asked for the net explicitly)."""
        pol = self.policy()[0]
        if consent is None and self.r.get(ONCE_KEY):
            consent = "yes"
            self.r.delete(ONCE_KEY)
        if pol == "never":
            return "refuse" if consent == "yes" else "offline"
        if consent == "no" or (personal_q and consent != "yes"):
            return "offline"
        if consent == "yes":
            return "search"
        if not world:
            return "offline"
        return "search" if pol == "always" else "ask"

    def search(self, question):
        if self.policy()[0] == "never":           # closed at the gate itself, not only in the caller
            log(f"blocked by policy never: {question[:40]!r}")
            return []
        return search(query_of(question), k=self.p.get("k", 3), timeout_s=self.p.get("timeout_s", 4.0),
                      backend=self.p.get("backend", "wikipedia"), url=self.p.get("url"))


def main():
    """policy [always|ask|never|status]: write / show the policy (Redis net:policy over the manifest)."""
    import redis
    import yaml
    params = yaml.safe_load((Path(__file__).with_name("manifest.yaml")).read_text(encoding="utf-8"))["params"].get("net") or {}
    r = redis.Redis()
    arg = sys.argv[2] if len(sys.argv) > 2 else "status"
    if arg in POLICIES:
        r.set(POLICY_KEY, arg)
    elif arg == "manifest":
        r.delete(POLICY_KEY)
    elif arg != "status":
        raise SystemExit("usage: net.py policy always|ask|never|manifest|status")
    pol, src = Net(params, r).policy()
    print(f"net policy: {pol} (from {src}); manifest: {params.get('policy', params.get('enabled', '-'))}; "
          f"redis {POLICY_KEY}: {(r.get(POLICY_KEY) or b'-').decode()}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "policy":
    main()
