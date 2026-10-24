"""brain: events:transcript -> events:reply through llama-server (D-024).

Per turn (order as in research/eval/ctx_probe.py run_dialog): router -> token window -> memory retrieval among
exchanges outside the window -> message (policy prefix + raw turn + memory note) -> reply filter -> history/memory.
Everything tunable comes from manifest.yaml `params`.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import redis, requests, yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.memory import SYSTEM_RULE, AssocMemory, clean_reply, format_note
from core.router import POLICY, decide, route_llamaserver

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


def chat(messages, max_tokens):
    resp = requests.post(f"{HOST}/v1/chat/completions", json={
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": P["temperature"],
        "chat_template_kwargs": {"enable_thinking": P["think"]},
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()


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
last_ev = '$'
print(f"brain: system~{sys_tokens} tokens, {r.llen(HISTORY)} exchanges in {HISTORY}, waiting for transcripts...")
while True:
    ev = r.xread({'events:transcript': last_ev}, count=1, block=5000)
    if not ev:
        continue
    ev_id, fields = ev[0][1][0]
    last_ev = ev_id
    text = fields[b'text'].decode()
    ts = float(fields[b'ts']) if b'ts' in fields else time.time()
    print(f"brain: << {text}")
    if text.strip() == "/reset":
        r.delete(HISTORY, VECS)
        mem = AssocMemory(MEM["embed_host"], mode=MEM["mode"]) if MEM else None
        print("brain: history reset")
        continue
    t0 = time.perf_counter()

    history = [json.loads(x) for x in r.lrange(HISTORY, 0, -1)]   # Redis is the only copy
    est_new = est_tokens(text) + 8
    route, policy = None, None
    if P.get("router"):
        # same window as an ordinary chat turn, so the main call reuses the KV prefix
        window, _ = build_window(history, sys_tokens, est_new + POLICY["chat"]["max_tokens"] + NOTE_RESERVE)
        try:
            route = route_llamaserver(HOST, to_messages(window), text)
            label, reason = decide(route, text)
            if reason:
                route.update(label_raw=route["label"], label=label, guard=reason)
        except requests.RequestException as e:
            print(f"brain: router failed: {e}")
        policy = POLICY.get((route or {}).get("label") or "chat", POLICY["chat"])
    max_tokens = policy["max_tokens"] if policy else MAX_TOKENS
    prefix = policy.get("prefix", "") if policy else ""
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
    data = chat(to_messages(window) + [{"role": "user", "content": sent}], max_tokens)
    choice = data["choices"][0]
    raw = (choice["message"].get("content") or "").strip()
    reply, flags = clean_reply(raw, note)
    if flags:
        print(f"brain: filter {sorted(flags)}, raw: {raw!r}")

    # token ledger: this exchange's share of later prompts = its raw user turn (prompt minus system, window, policy
    # prefix and note, none of which is kept) plus its reply; with an empty window the system prompt is calibrated
    usage, timings = data.get("usage", {}), data.get("timings", {})
    pe, ec = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    not_kept = est_tokens(prefix) + est_tokens(note)
    if not window and pe:
        sys_tokens = max(0, pe - est_new - not_kept)
        r.set(SYS_CALIB, json.dumps({"chars": len(SYSTEM), "tokens": sys_tokens}))
    in_window = sum(ex["tokens"] for ex in window)
    tokens = max(1, pe - sys_tokens - in_window - not_kept) + ec if pe else est_new + est_tokens(raw)
    turn = history[-1]["turn"] + 1 if history else 1
    r.rpush(HISTORY, json.dumps({"turn": turn, "ts": ts, "user": text, "reply": reply, "raw": raw,
                                 "tokens": tokens}, ensure_ascii=False))
    if mem is not None:
        try:
            v = mem.add(turn, text, reply)
            if v is not None:
                r.hset(VECS, turn, v.astype(np.float32).tobytes())
        except requests.RequestException as e:
            print(f"brain: memory add failed: {e}")
    wall = time.perf_counter() - t0
    # per-turn details for logs and research/eval/ctx_bus.py (TTS reads only `text`)
    meta = {"turn": turn, "window_turns": len(window), "prompt_tokens": pe, "completion_tokens": ec,
            "ttft_s": round(timings.get("prompt_ms", 0) / 1000, 3),
            "tok_s": round(timings["predicted_per_second"], 1) if timings.get("predicted_per_second") and ec >= 5 else None,
            "finish_reason": choice.get("finish_reason"), "wall_s": round(wall, 2), "max_tokens": max_tokens}
    if route:
        meta["route"] = route
    if mem is not None:
        meta.update({"mem_hits": [{"turn": h[1]["turn"], "score": round(h[0], 3), "cos": round(h[2], 3),
                                   "kw": round(h[3], 3)} for h in hits], "note": note, "guard": sorted(flags)})
    if flags:
        meta["raw"] = raw
    r.xadd('events:reply', {'text': reply, 'ts': f"{ts:.3f}", 'meta': json.dumps(meta, ensure_ascii=False)})
    print(f"brain: >> {reply}")
    print(f"brain: turn {turn} prompt={pe} reply={ec} exchange~{tokens} tokens {wall:.1f}s")
