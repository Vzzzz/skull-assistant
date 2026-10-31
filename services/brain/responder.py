"""brain: events:transcript -> events:reply through llama-server (D-024).

Per turn (order as in research/eval/ctx_probe.py run_dialog): router -> token window -> memory retrieval among
exchanges outside the window -> message (policy prefix + raw turn + memory note) -> reply filter -> history/memory.
Everything tunable comes from manifest.yaml `params`.
"""
import json, re, sys, time, uuid
from pathlib import Path

import numpy as np
import redis, requests, yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.ambient import OPEN_KEY, Ambient, engaged_until, gaze_skip, has_name, open_window
from core.memory import SYSTEM_RULE, AssocMemory, clean_reply, format_note
from core.modes import deferred_question, find_phrase
from core.speech import for_speech
from core.router import POLICY, POLICY_B, decide, duck_bin_llamaserver, policy_prefix, route_llamaserver

P = yaml.safe_load(Path(__file__).with_name("manifest.yaml").read_text(encoding="utf-8"))["params"]
HOST = P["host"]
MODEL = Path(P["gguf"]).stem
NUM_CTX = P["num_ctx"]
MEM = P.get("memory")         # None = no memory beyond the window
PERSONA = (ROOT / P["persona"]).read_text(encoding="utf-8").strip()
SYSTEM = PERSONA + ("\n\n" + SYSTEM_RULE if MEM else "")

r = redis.Redis(socket_timeout=None)
HISTORY = "dialog:history"    # Redis list, one JSON element per exchange: turn, ts, user, reply, raw, tokens
VECS = "dialog:vec"           # Redis hash turn -> float32 embedding of the exchange (memory cache)
SYS_CALIB = "dialog:sys_tokens"   # {"chars", "tokens"}: measured system prompt size, survives restarts
MAX_TOKENS = 700              # reply budget when the router is off
NOTE_RESERVE = 250            # room for the memory note, reserved in the window
CHARS_PER_TOKEN = 2.3         # Cyrillic estimate for text not yet seen by the model
AMB = P.get("ambient", {})    # T-038 stage 2b: listening without a wake word (trigger: speech in services/wakeword)
amb = Ambient(engaged_s=AMB.get("engaged_s", 20), busy_pause_s=AMB.get("busy_pause_s", 20),
              veto_p=AMB.get("veto_aside_p", 0.9), engaged_yes_p=AMB.get("engaged_yes_p", 0.7),
              ack_gap_s=AMB.get("ack_gap_s", 20), engaged_ack_p=AMB.get("engaged_ack_p", 0.3),
              idle_yes_p=AMB.get("idle_yes_p", 0.7), idle_ack_p=AMB.get("idle_ack_p", 0.3))
ACKS = ("Угу.", "М-м.", "Ага.")
acks = 0
# T-031 step 2: a silent look at the skull (services/vision, events:vision {kind: gaze})
GAZE_PROMPT = ("Служебно, не отвечай на это как на реплику: собеседник молча смотрит на тебя уже две секунды. "
               "Скажи одно короткое, в своём тоне.")
GAZE_BAN = (" Не говори, что на тебя смотрят; не используй слова с корнями «смотр», «гляд», «наблюда».")
GAZE_TURNS = (" Спроси у него что-нибудь.", " Пожалуйся на что-нибудь.", " Заметь что-нибудь вокруг.")   # one per look, in turn
GAZE_RECENT = " Твои последние реплики на такой взгляд: {}. Не повторяй их начало и форму."
GAZE_ROOTS = re.compile(r"смотр|гляд|наблюда", re.IGNORECASE)   # the ban, checked: the 4B copies the situation into the line
GAZE_TURN = "gaze:turn"         # Redis counter: which of GAZE_TURNS comes next
GAZE_COOLDOWN_S = AMB.get("gaze_cooldown_s", 30)
# T-044: while the dispatcher swaps the model (Redis mode:transition) llama-server is down or loading: one fixed line
# in character to an address (the name / the wake word), never the model; overheard speech is ignored, no duck
HOLD = "Погоди, думаю."
TRANSITION = "mode:transition"
TRANSITION_TTL_S = 300          # core/dispatcher.py: a switch that has begun ends (done / failed) within this
GAZE_WINDOW_S = AMB.get("gaze_window_s", 15)
GAZE_LINES = "gaze:lines"       # Redis list, newest first: the last gaze lines (the prompt and the repeat check)
last_gaze = 0.0
# T-046: mode B by the owner's voice. The trigger phrase sends cmd:mode B (core/dispatcher.py); the rest of the turn is
# asked to model B after events:mode {phase: done} — re-published on events:transcript with `deferred` (it skips the
# transition check and the trigger). In B: no router, no duck (POLICY_B, Ambient thinking); back by the exit phrase or
# after idle_exit_s without the owner's turns
MODES = P.get("modes", {})
LINES = MODES.get("lines", {})
pending = None                  # {"id", "to", "question", "voice", "t", "seen"}: a switch brain asked for and waits on
PENDING_ACK_S = 20              # no event for it this long (the dispatcher is down / restarting): give up, say so
calls = []                      # this turn's requests to llama-server (router / duck / chat): one per turn in B (T-046)
last_owner_b = time.time()      # the owner's last turn in B (the idle exit counts from it)


def est_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN)


def wait_ready(host, what):
    for i in range(240):
        try:
            if requests.get(f"{host}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        if i % 15 == 0:
            print(f"brain: waiting for {what} at {host}...")
        time.sleep(1)
    raise SystemExit(f"brain: {what} at {host} not ready")


def build_window(history, sys_tokens, need):
    """Newest exchanges that fit into NUM_CTX together with the system prompt and `need` (new turn + reserves).
    Older exchanges stay in Redis: they are what the memory retrieves from."""
    window, used = [], sys_tokens
    for ex in reversed(history):
        if used + ex["tokens"] + need > NUM_CTX:
            break
        window.insert(0, ex)
        used += ex["tokens"]
    return window, used


def to_messages(window):
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in window:
        messages += [{"role": "user", "content": ex["user"]}, {"role": "assistant", "content": ex["reply"]}]
    return messages


def chat_stream(messages, max_tokens, tail):
    """Streams the reply from llama-server (SSE): yields text deltas; `tail` gets finish_reason, usage, timings
    (they come in the last chunks: usage/timings in a chunk with empty choices)."""
    resp = requests.post(f"{HOST}/v1/chat/completions", json={
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": P["temperature"],
        "chat_template_kwargs": {"enable_thinking": P["think"]},   # else "Thinking Process:" arrives in the stream
        "stream": True,
        "stream_options": {"include_usage": True},
    }, timeout=120, stream=True)
    resp.raise_for_status()
    resp.encoding = "utf-8"     # text/event-stream comes without a charset: requests would decode it as latin-1
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            break
        d = json.loads(line[6:])
        tail.update({k: d[k] for k in ("usage", "timings") if k in d})
        for ch in d.get("choices", []):
            if ch.get("finish_reason"):
                tail["finish_reason"] = ch["finish_reason"]
            delta = (ch.get("delta") or {}).get("content")
            if delta:
                yield delta


# sentence end: punctuation (+ closing quote/bracket) followed by whitespace, or a line break (calc steps come as lines)
SENTENCE_END = re.compile(r'[.!?…]+["»)]?\s+|\n+')
LIST_MARK = re.compile(r"(?:\d+|[a-zа-я])[.)]")


def say(reply_id, seq, sentence, note, ts, final=False, meta=None):
    """Publishes one sentence for TTS: memory guard and speech form applied here, per sentence.
    Returns True if something was published."""
    text = sentence.strip()
    if text:
        clean, flags = clean_reply(text, note)
        flags.discard("dunno")                          # «Не знаю.» is said; only the history skips it
        text = "" if flags and clean == text else for_speech(clean)   # guard removed everything -> nothing to say
    if not text and not final:
        return False
    fields = {"text": text, "reply_id": reply_id, "seq": seq, "final": int(final), "ts": f"{ts:.3f}"}
    if meta is not None:
        fields["meta"] = json.dumps(meta, ensure_ascii=False)
    r.xadd("events:reply", fields)
    return True


def ambient_gate(text, fields, route, window, thinking=False):
    """Answer or not, for a turn heard without a wake word: core/ambient.py (rules, D-029). Returns
    (decision, why, opens, engaged): decision respond | ack | silent; opens — the answer opens / extends ENGAGED."""
    speaker = fields.get(b'speaker', b'unknown').decode()
    pause = float(fields.get(b'pause_s', b'2.0'))
    p_aside = ((route or {}).get("probs") or {}).get("aside", 0.0)
    engaged = time.time() < engaged_until(r)

    def duck(t, pause_s):
        calls.append("duck")
        try:
            return duck_bin_llamaserver(HOST, to_messages(window), t, pause_s)
        except requests.RequestException as e:
            print(f"brain: duck pass failed: {e}")
            return {"ans_tok": None, "probs": {}}
    lifted = bool(fields.get(b'spk_region'))      # the voice was found by a window inside a broadcast segment (2в)
    bcast = fields.get(b'bcast') == b'1'          # STT heard a foreign voice in the last 30 s
    seg_s = float(fields[b'seg_s']) if fields.get(b'seg_s') else None
    return (*amb.decide(text, speaker, pause, p_aside, time.time(), engaged, duck, lifted, bcast, seg_s, thinking), engaged)


def mode_now():
    return (r.get("mode:current") or b"A").decode()


def fixed_line(key, ts=None, kind="mode"):
    """A constant line in character (never the model); returns its reply_id."""
    rid = uuid.uuid4().hex[:12]
    say(rid, 0, LINES[key], "", ts or time.time(), final=True, meta={"reply": LINES[key], "mode": kind})
    print(f"brain: >> {LINES[key]} ({kind})")
    return rid


def ask_mode(dst, reason, question="", voice=True):
    global pending
    req = "br" + uuid.uuid4().hex[:6]
    pending = {"id": req, "to": dst, "question": question, "voice": voice, "t": time.time(), "seen": False}
    r.xadd("cmd:mode", {"mode": dst, "reason": reason, "id": req})
    print(f"brain: cmd:mode {dst} ({reason}){' deferred: ' + repr(question) if question else ''}")


def on_mode_event(f):
    """events:mode: finish what brain asked for (the deferred question, the fixed lines); note entries to B."""
    global pending, last_owner_b
    f = {k.decode(): v.decode() for k, v in f.items()}
    if f.get("phase") == "done" and f.get("to") == "B":
        last_owner_b = time.time()                 # also when B was entered by hand (skull-mode.sh)
    if not pending or f.get("id") != pending["id"]:
        return
    pending["seen"] = True
    if f["phase"] == "failed":
        fixed_line("failed")
        pending = None
    elif f["phase"] in ("done", "noop"):
        p_, pending = pending, None
        if p_["to"] == "B":
            if p_["question"]:
                r.xadd("events:transcript", {"text": p_["question"], "trigger": "wakeword", "speaker": "owner",
                                             "deferred": "1", "ts": f"{time.time():.3f}"})
            else:
                fixed_line("ready")
        elif p_["voice"]:
            rid = fixed_line("back")
            open_window(r, rid, amb.engaged_s, time.time())   # back in A: the usual window, counted from this line


def first_word(line):
    return re.sub(r"[^\w-]", "", (line.split() or [""])[0].lower())


def on_gaze(fields):
    """A silent look (T-031 steps 2-3): in IDLE, outside the skull's speech and not during a call (BUSY), one short line
    in character, not more often than GAZE_COOLDOWN_S, kept out of the history and the memory (like «угу»). The line
    opens the window for GAZE_WINDOW_S: a look plus «Что?» invites an answer. A look does not extend an open window
    (core's answer 2: false looks kept it open). The prompt bans saying it is being looked at (and the roots смотр- /
    гляд- / наблюда-), turns the request in a circle (ask / complain / notice something around) and lists the last 3
    gaze lines. A line starting like one of them, or with a banned root, is generated once more; then silence — silence
    beats the fourth «Смотри.» (core's answers 4 of 2026-10-29 and 3 of 2026-10-30)."""
    global last_gaze
    now = time.time()
    why = gaze_skip(r, now, amb.busy, last_gaze, GAZE_COOLDOWN_S, TRANSITION)
    if why:
        print(f"brain: gaze {fields.get(b'dur_s', b'?').decode()} s -> no line ({why})")
        return
    last_gaze = now
    history = [json.loads(x) for x in r.lrange(HISTORY, 0, -1)]
    window, _ = build_window(history, sys_tokens, 200)
    tail = {}
    recent = [x.decode() for x in r.lrange(GAZE_LINES, 0, 2)]
    turn = GAZE_TURNS[(r.incr(GAZE_TURN) - 1) % len(GAZE_TURNS)]
    prompt = GAZE_PROMPT + GAZE_BAN + turn + (GAZE_RECENT.format(", ".join(f"«{x}»" for x in recent)) if recent else "")
    starts = {first_word(x) for x in recent}
    for attempt in (1, 2):
        raw = "".join(chat_stream(to_messages(window) + [{"role": "user", "content": prompt}], 40, tail)).strip()
        line, _ = clean_reply(raw, "")
        why = ("empty" if not line else "banned root" if GAZE_ROOTS.search(line)
               else "repeat" if first_word(line) in starts else None)
        if not why:
            break
        print(f"brain: gaze -> {line!r} dropped ({why}, attempt {attempt}; recent {recent})")
        r.xadd("events:ambient", {"text": "", "decision": "gaze_drop", "why": why, "state": "idle",
                                  "ts": f"{now:.3f}", "reply": line}, maxlen=5000, approximate=True)
    if why:
        return
    r.lpush(GAZE_LINES, line); r.ltrim(GAZE_LINES, 0, 2)
    reply_id = uuid.uuid4().hex[:12]
    say(reply_id, 0, line, "", now, final=True, meta={"reply": line, "ambient": "gaze"})
    open_window(r, reply_id, GAZE_WINDOW_S, time.time())
    r.xadd("events:ambient", {"text": "", "decision": "gaze", "why": f"look {fields.get(b'dur_s', b'?').decode()} s",
                              "state": "idle", "ts": f"{now:.3f}", "reply": line}, maxlen=5000, approximate=True)
    print(f"brain: gaze -> {line!r} ({turn.strip()}; {time.time() - now:.2f} s)")


def load_memory(history):
    mem = AssocMemory(MEM["embed_host"], mode=MEM["mode"])
    for ex in history:
        cached = r.hget(VECS, ex["turn"])
        try:
            v = mem.add(ex["turn"], ex["user"], ex["reply"], vec=np.frombuffer(cached, dtype=np.float32) if cached else None)
        except requests.RequestException as e:
            print(f"brain: memory skips turn {ex['turn']}: {e}")
            continue
        if v is not None and not cached:
            r.hset(VECS, ex["turn"], v.astype(np.float32).tobytes())
    return mem


wait_ready(HOST, "llama-server")
server_ctx = requests.get(f"{HOST}/props", timeout=5).json().get("default_generation_settings", {}).get("n_ctx")
if server_ctx and server_ctx != NUM_CTX:
    print(f"brain: WARNING manifest num_ctx={NUM_CTX}, server n_ctx={server_ctx}; using the server's")
    NUM_CTX = server_ctx
print(f"brain: model={MODEL} num_ctx={NUM_CTX} host={HOST}")
print(f"brain: persona {P['persona']} ({len(PERSONA)} chars), router={'on' if P.get('router') else 'off'}, "
      f"memory={'k=%d %s' % (MEM['k'], MEM['mode']) if MEM else 'off'}")
mem = None
if MEM:
    wait_ready(MEM["embed_host"], "embedder")
    mem = load_memory([json.loads(x) for x in r.lrange(HISTORY, 0, -1)])
calib = json.loads(r.get(SYS_CALIB) or "{}")
# measured size if the prompt is unchanged, else the estimate until a call with an empty window calibrates it
sys_tokens = calib["tokens"] if calib.get("chars") == len(SYSTEM) else est_tokens(SYSTEM)
last_ev, last_vis, last_mode = '$', '$', '$'
print(f"brain: system~{sys_tokens} tokens, {r.llen(HISTORY)} exchanges in {HISTORY}, waiting for transcripts...")
while True:
    ev = r.xread({'events:transcript': last_ev, 'events:vision': last_vis, 'events:mode': last_mode}, count=1, block=5000)
    if pending and time.time() - pending["t"] > (TRANSITION_TTL_S if pending["seen"] else PENDING_ACK_S):
        print(f"brain: no answer to cmd:mode {pending['to']} ({pending['id']}) — given up")
        if pending["voice"]:
            fixed_line("failed")
        pending = None
    if (MODES and not pending and mode_now() == "B" and not r.exists(TRANSITION)
            and time.time() - last_owner_b > MODES.get("idle_exit_s", 300)):
        ask_mode("A", "idle", voice=False)        # the owner has gone quiet: back to A, silently
    if not ev:
        continue
    stream, ((ev_id, fields),) = ev[0]
    if stream == b'events:mode':
        last_mode = ev_id
        on_mode_event(fields)
        continue
    if stream == b'events:vision':
        last_vis = ev_id
        if fields.get(b'kind') == b'gaze':
            try:
                on_gaze(fields)
            except requests.RequestException as e:
                print(f"brain: gaze line failed: {e}")
        continue
    last_ev = ev_id
    text = fields[b'text'].decode()
    ts = float(fields[b'ts']) if b'ts' in fields else time.time()
    print(f"brain: << {text}")
    if text.strip() == "/reset":
        r.delete(HISTORY, VECS)
        mem = AssocMemory(MEM["embed_host"], mode=MEM["mode"]) if MEM else None
        amb.reset()                                    # a new conversation starts idle
        r.delete(OPEN_KEY)
        print("brain: history reset")
        continue
    deferred = fields.get(b'deferred') == b'1'
    if (r.exists(TRANSITION) or pending) and not deferred:
        if fields.get(b'trigger') != b'speech' or has_name(text):
            say(uuid.uuid4().hex[:12], 0, HOLD, "", ts, final=True, meta={"reply": HOLD, "mode": "transition"})
        print(f"brain: mode transition -> {'hold line' if fields.get(b'trigger') != b'speech' or has_name(text) else 'ignored'}")
        continue
    mode = mode_now()
    thinking = mode == "B"
    owner = fields.get(b'speaker') == b'owner'
    if MODES and owner and not deferred:
        span = find_phrase(text, MODES["enter"]) if mode == "A" else find_phrase(text, MODES["exit"])
        if span and mode == "A":
            fixed_line("wait", ts)
            ask_mode("B", f"voice:{span[0]}", deferred_question(text, span))
            continue
        if span:
            ask_mode("A", f"voice:{span[0]}")
            continue
    if thinking and owner:
        last_owner_b = time.time()
    t0 = time.perf_counter()
    calls.clear()

    history = [json.loads(x) for x in r.lrange(HISTORY, 0, -1)]   # Redis is the only copy
    est_new = est_tokens(text) + 8
    route, policy, label = None, None, None
    if thinking:                  # T-046: no router in B — one prefill per turn
        label, policy = "explain", POLICY_B
    elif P.get("router"):
        # same window as an ordinary chat turn, so the main call reuses the KV prefix
        window, _ = build_window(history, sys_tokens, est_new + POLICY["chat"]["max_tokens"] + NOTE_RESERVE)
        try:
            calls.append("router")
            route = route_llamaserver(HOST, to_messages(window), text)
            label, reason = decide(route, text)
            if reason:
                route.update(label_raw=route["label"], label=label, guard=reason)
        except requests.RequestException as e:
            print(f"brain: router failed: {e}")
        label = (route or {}).get("label") or "chat"
        policy = POLICY.get(label, POLICY["chat"])
    if fields.get(b'trigger') == b'speech':
        # ambient: who speaks, conversation state and the duck pass decide; the router label only shapes the answer
        decision, why, opens, engaged = ambient_gate(text, fields, route, window if P.get("router") and not thinking else [],
                                                     thinking)
        st = "busy" if amb.busy else "engaged" if engaged else "idle"
        print(f"brain: ambient {decision} ({why}) state={st} speaker={fields.get(b'speaker', b'-').decode()} "
              f"pause={fields.get(b'pause_s', b'-').decode()}")
        r.xadd("events:ambient", {"text": text, "decision": decision, "why": why, "state": st, "ts": f"{ts:.3f}",
                                  "speaker": fields.get(b'speaker', b'').decode(), "spk_score": fields.get(b'spk_score', b'').decode(),
                                  "pause_s": fields.get(b'pause_s', b'').decode(), "pause_wait": fields.get(b'pause_wait', b'').decode(),
                                  "spk_win": fields.get(b'spk_win', b'').decode(), "spk_region": fields.get(b'spk_region', b'').decode(),
                                  "text_full": fields.get(b'text_full', b'').decode(), "bcast": fields.get(b'bcast', b'').decode(),
                                  "p_aside": f"{((route or {}).get('probs') or {}).get('aside', 0.0):.3f}"},
               maxlen=5000, approximate=True)
        if decision == "silent":       # not answered: nothing in the history or the memory
            continue
        if decision == "ack":          # «угу»: a reaction without content; kept in the history as an exchange
            ack = ACKS[acks % len(ACKS)]
            acks += 1
            say(uuid.uuid4().hex[:12], 0, ack, "", ts, final=True, meta={"reply": ack, "ambient": "ack"})
            turn = history[-1]["turn"] + 1 if history else 1
            r.rpush(HISTORY, json.dumps({"turn": turn, "ts": ts, "user": text, "reply": ack, "raw": ack,
                                         "tokens": est_new + 4}, ensure_ascii=False))
            print(f"brain: >> {ack}")
            continue
        if policy is not None and policy.get("respond") is False:
            label, policy = "chat", POLICY["chat"]    # an aside label below the veto threshold does not silence
    max_tokens = policy["max_tokens"] if policy else MAX_TOKENS
    prefix = (policy.get("prefix", "") if thinking else            # B: WORLD_SOFT always
              policy_prefix(label, text) if policy else "")          # world: hedge only for perception words (T-042)
    use_mem = mem is not None and (policy is None or policy["memory"])
    extra = est_tokens(prefix) + (NOTE_RESERVE if use_mem else 0)
    window, used = build_window(history, sys_tokens, est_new + extra + max_tokens)
    print(f"brain: window {len(window)} exchanges, ~{used + est_new + extra} tokens ({len(history)} stored)"
          + (f", route={route['label']} {route['s']}s" if route else "")
          + (f" ({route['label_raw']} {route['guard']})" if route and route.get("guard") else ""))

    hits, note = [], ""
    if use_mem:
        try:
            hits = mem.retrieve(text, k=MEM["k"], exclude_turns={ex["turn"] for ex in window})
        except requests.RequestException as e:
            print(f"brain: memory retrieval failed: {e}")
        note = format_note(hits)
        if hits:
            print(f"brain: memory {[h[1]['turn'] for h in hits]}")
    sent = (prefix + "\n" + text if prefix else text) + ("\n\n" + note if note else "")
    # stream: sentences go to TTS as soon as they are complete (T-003); the whole reply is stored once at the end
    reply_id, seq, buf, parts, tail = uuid.uuid4().hex[:12], 0, "", [], {}
    calls.append("chat")
    for delta in chat_stream(to_messages(window) + [{"role": "user", "content": sent}], max_tokens, tail):
        buf += delta
        parts.append(delta)
        pos = 0
        while (m := SENTENCE_END.search(buf, pos)):
            if LIST_MARK.fullmatch(buf[:m.end()].strip()):   # «1. » is a list number, not a sentence: keep it with the next one
                pos = m.end()
                continue
            sentence, buf, pos = buf[:m.end()], buf[m.end():], 0
            if say(reply_id, seq, sentence, note, ts):
                if seq == 0:
                    print(f"brain: first sentence +{time.perf_counter() - t0:.2f}s")
                seq += 1
    raw = "".join(parts).strip()
    reply, flags = clean_reply(raw, note)
    if flags:
        print(f"brain: filter {sorted(flags)}, raw: {raw!r}")

    # token ledger: this exchange's share of later prompts = its raw user turn (prompt minus system, window, policy
    # prefix and note, none of which is kept) plus its reply; with an empty window the system prompt is calibrated
    usage, timings = tail.get("usage", {}), tail.get("timings", {})
    pe, ec = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    not_kept = est_tokens(prefix) + est_tokens(note)
    if not window and pe:
        sys_tokens = max(0, pe - est_new - not_kept)
        r.set(SYS_CALIB, json.dumps({"chars": len(SYSTEM), "tokens": sys_tokens}))
    in_window = sum(ex["tokens"] for ex in window)
    tokens = max(1, pe - sys_tokens - in_window - not_kept) + ec if pe else est_new + est_tokens(raw)
    turn = history[-1]["turn"] + 1 if history else 1
    if "dunno" not in flags:      # a bare «Не знаю.» is said, not kept (T-042): nothing in the history or the memory
        r.rpush(HISTORY, json.dumps({"turn": turn, "ts": ts, "user": text, "reply": reply, "raw": raw,
                                     "tokens": tokens}, ensure_ascii=False))
    if mem is not None and "dunno" not in flags:
        try:
            v = mem.add(turn, text, reply)
            if v is not None:
                r.hset(VECS, turn, v.astype(np.float32).tobytes())
        except requests.RequestException as e:
            print(f"brain: memory add failed: {e}")
    wall = time.perf_counter() - t0
    # per-turn details for logs and research/eval/ctx_bus.py, on the final message of the reply (TTS reads `text`)
    meta = {"turn": turn, "window_turns": len(window), "prompt_tokens": pe, "completion_tokens": ec,
            "ttft_s": round(timings.get("prompt_ms", 0) / 1000, 3),
            "tok_s": round(timings["predicted_per_second"], 1) if timings.get("predicted_per_second") and ec >= 5 else None,
            "finish_reason": tail.get("finish_reason"), "wall_s": round(wall, 2), "max_tokens": max_tokens,
            "reply": reply, "sentences": seq + 1, "mode": mode, "llm_calls": list(calls)}
    if route:
        meta["route"] = route
    if mem is not None:
        meta.update({"mem_hits": [{"turn": h[1]["turn"], "score": round(h[0], 3), "cos": round(h[2], 3),
                                   "kw": round(h[3], 3)} for h in hits], "note": note, "guard": sorted(flags)})
    if flags:
        meta["raw"] = raw
    say(reply_id, seq, buf, note, ts, final=True, meta=meta)   # the rest of the text (may be empty) closes the reply
    print(f"brain: >> {reply}")
    print(f"brain: turn {turn} [{mode}] prompt={pe} reply={ec} exchange~{tokens} tokens {wall:.1f}s llm_calls={calls}")
    if fields.get(b'trigger') == b'speech' and opens:
        # the conversation is on: the window runs engaged_s after this answer has been played (tts:end)
        open_window(r, reply_id, amb.engaged_s, time.time())
