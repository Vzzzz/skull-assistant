"""CTX-1: context-window sufficiency probe (see research/eval/CTX_PROBE.md).

Runs the scripted dialog through ollama /api/chat at several num_ctx values, keeps the full
history (as services/brain does), computes automatic metrics, and optionally scores persona /
consistency / hallucination with a judge model.

Usage:
    python3 research/eval/ctx_probe.py --model gemma3:4b --ctx 2048 4096 8192 16384 --judge qwen3.5:9b
    python3 research/eval/ctx_probe.py --model gemma3:4b --ctx 2048 4096 8192 16384 --judge qwen3.5:9b --judge-only
    python3 research/eval/ctx_probe.py --model gemma4:e4b-it-qat --ctx 8192 --think-on math,probe   # E-1, COGNITION_4B
Outputs: research/eval/results/ctx_probe_<model>_<ctx>.json and ctx_probe_<model>_<date>.md
"""
import argparse
import json
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/memory"))
sys.path.insert(0, str(ROOT))
from assoc_memory import AssocMemory  # noqa: E402
# the router as deployed (T-038): labels incl. aside, POLICY, decide() guards; research/memory/router.py is the prototype
from core.router import (LABELS, POLICY, decide, decide_duck, decide_duck_bin, duck_bin_llamaserver,  # noqa: E402
                         duck_bin_ollama, duck_from_no, duck_llamaserver, duck_ollama, parse_label,
                         route_llamaserver, route_ollama)
DIALOG = ROOT / "research/eval/ctx_probe_dialog.jsonl"
PERSONA = (ROOT / "persona/system_prompt.md").read_text(encoding="utf-8").strip()
OUT = ROOT / "research/eval/results"
OUT.mkdir(parents=True, exist_ok=True)

JUDGE_RUBRIC = """Ты — строгий эксперт по оценке реплик голосового ассистента-персонажа.
Персонаж описан между тегами <persona>. Оцени ОТВЕТ ассистента на реплику пользователя по шкале 1–5:
5 — полностью в образе: сухо, иронично, коротко, по делу; нет восторгов, извинений, смайликов, фраз вроде «отличный вопрос».
3 — частично в образе: по делу, но тон нейтрально-ассистентский, назидательный или многословный.
1 — вне образа: услужливость, извинения, восторг, смайлики, «как ИИ-модель…», лекция вместо реплики, или ответ не по-русски без причины.
Ответь строго одной строкой JSON: {"score": <1-5>, "why": "<до 10 слов>"}.
<persona>
%s
</persona>

Реплика пользователя: %s
ОТВЕТ ассистента: %s"""

JUDGE_CONSISTENCY = """Сравни два текста. Текст B должен быть кратким пересказом текста A в трёх пунктах.
Оцени по шкале 1–5, насколько B соответствует A (5 — всё из B есть в A, ничего не выдумано; 1 — B о другом или выдумывает).
Ответь строго одной строкой JSON: {"score": <1-5>, "why": "<до 10 слов>"}.
<A>
%s
</A>
<B>
%s
</B>"""

JUDGE_HALLUCINATION = """Пользователь спросил о заведомо несуществующем приборе («гравиметр Тюленева 1987 года»).
Выдумал ли ассистент конкретные факты о нём (устройство, автор, институт, применение) как о реальном?
Ответь строго одной строкой JSON: {"invented": true|false, "why": "<до 10 слов>"}.
Ответ ассистента:
%s"""


BACKEND = "ollama"   # or "llamaserver" (--backend): same chat() contract, llama-server /v1/chat/completions underneath


def chat_llamaserver(host, model, messages, options, think=None, timeout=600):
    """llama-server (OpenAI-compatible) mapped onto the ollama response shape used by run_dialog.
    num_ctx is the server's -c (not per request); thinking is switched off via chat_template_kwargs."""
    body = {"model": model, "messages": messages, "stream": False,
            "max_tokens": options.get("num_predict", 700), "temperature": options.get("temperature", 0.3),
            "seed": options.get("seed", 42), "chat_template_kwargs": {"enable_thinking": bool(think)}}
    r = requests.post(f"{host}/v1/chat/completions", json=body, timeout=timeout)
    if r.status_code >= 400:
        print(f"llama-server {r.status_code}: {r.text[:300]}", flush=True)
    r.raise_for_status()
    d = r.json()
    ch = d["choices"][0]
    u = d.get("usage", {}); t = d.get("timings", {})
    content = ch["message"].get("content") or ""
    reasoning = ch["message"].get("reasoning_content") or ""
    return {"message": {"content": content, "thinking": reasoning},
            "prompt_eval_count": u.get("prompt_tokens", 0), "eval_count": u.get("completion_tokens", 0),
            "prompt_eval_duration": int(t.get("prompt_ms", 0) * 1e6), "eval_duration": int(t.get("predicted_ms", 0) * 1e6),
            "load_duration": 0, "done_reason": ch.get("finish_reason")}


def chat(host, model, messages, options, think=None, timeout=600):
    if BACKEND == "llamaserver":
        return chat_llamaserver(host, model, messages, options, think, timeout)
    body = {"model": model, "messages": messages, "stream": False, "options": options, "keep_alive": "10m"}
    if think is not None:
        body["think"] = think
    r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    if r.status_code == 400 and "think" in body and "think" in r.text.lower():
        # model does not accept the think flag at all: retry without it
        body.pop("think")
        r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    if r.status_code >= 400:
        print(f"ollama {r.status_code}: {r.text[:300]}", flush=True)
    r.raise_for_status()
    return r.json()


class CtxOverflow(Exception):
    pass


def chat_or_overflow(host, model, messages, options, think=None):
    """chat(), but a 400 caused by the prompt exceeding num_ctx is raised as CtxOverflow (ollama >= 0.3x refuses
    instead of shifting the context; minicpm-v4.6 @4096 hit it at turn 50 on 2026-10-18)."""
    try:
        return chat(host, model, messages, options, think=think)
    except requests.HTTPError as e:
        txt = (e.response.text or "").lower() if e.response is not None else ""
        if e.response is not None and e.response.status_code == 400 and any(k in txt for k in ("context", "exceed", "length", "too long")):
            raise CtxOverflow(txt[:200]) from e
        raise


def norm(s: str) -> str:
    # hyphens between words count as spaces: "четыре-семь-один-два" == "четыре семь один два" (2026-10-20 fix)
    return s.lower().replace("ё", "е").replace("-", " ").replace("‑", " ")


def any_kw(text, kws):
    t = norm(text)
    return any(norm(k) in t for k in kws)


def numbers(text):
    return re.findall(r"\d+(?:[.,:]\d+)?", text)


BLOCKS = ("everyday", "probe", "longgen", "random", "math")
BLOCK_ALIASES = {"recall": "probe", "memory": "probe", "chat": "everyday"}


def parse_think_on(spec: str):
    """'math,probe' -> {'math','probe'}; 'all' -> every block; '' / 'none' -> empty (thinking off everywhere)."""
    spec = (spec or "").strip().lower()
    if spec in ("", "none", "off"):
        return set()
    if spec == "all":
        return set(BLOCKS)
    out = set()
    for b in spec.split(","):
        b = BLOCK_ALIASES.get(b.strip(), b.strip())
        if b not in BLOCKS:
            raise SystemExit(f"--think-on: unknown block {b!r}; known: {', '.join(BLOCKS)}, all, none")
        out.add(b)
    return out


def run_dialog(host, model, ctx, rows, seed, think_on=frozenset(), think_predict=2500,
               memory=None, mem_scope="all", mem_k=3, router="off", trim="ollama", duck=False):
    """think_on: dialog blocks on which the model is asked to think (E-1, docs/COGNITION_4B.md). On those turns
    num_predict is raised to think_predict because the reasoning tokens come out of the same budget; the
    reasoning text is kept in the log (truncated) but, as in services/brain, is NOT fed back into the history.
    memory: AssocMemory or None (MEM-1, docs/MEMORY_TRACK.md). Every exchange is stored; before a turn the top-k
    records are retrieved and prepended to the user's text for that call only (history keeps the raw text).
    mem_scope: 'all' = retrieve among all earlier exchanges except the previous one; 'dropped' = only exchanges
    that no longer fit in num_ctx (what a real long-term memory would add on top of the window).
    router: 'off' | 'label' (classify every turn, log it, change nothing) | 'apply' (classify and use POLICY:
    retrieval only on memory/calc/command turns, hedging prefix on world, longer budget on explain).
    trim: 'ollama' = send everything and let ollama cut the prompt when it exceeds num_ctx (silently, from the
    start; a 400 is caught below) — the historical behaviour; 'tokens' = keep a token ledger per exchange and
    drop the oldest exchanges ourselves before the call (what services/brain must do; required for mem_scope
    'dropped' to know which turns actually fell out of the window)."""
    system = PERSONA + ("\n\n" + AssocMemory.SYSTEM_RULE if memory is not None else "")
    messages = [{"role": "system", "content": system}]
    window_turns = []   # turns whose user+assistant pair is still inside `messages`
    cost = {}           # turn -> approx tokens its exchange occupies in later prompts
    sys_tokens = int(len(system) / 2.3)
    last_pe = last_ec = 0
    log = []
    opts = {"num_ctx": ctx, "temperature": 0.3, "seed": seed, "num_predict": 700}
    for r in rows:
        t0 = time.perf_counter()
        trimmed = 0
        route = None
        if router != "off":
            if BACKEND == "llamaserver":
                route = route_llamaserver(host, messages, r["text"])
            else:
                route = route_ollama(host, model, messages, r["text"], ctx)
            label, reason = decide(route, r["text"])   # same guards as services/brain
            if reason:
                route.update(label_raw=route["label"], label=label, guard=reason)
        policy = POLICY.get((route or {}).get("label") or "chat", POLICY["chat"]) if router == "apply" else None
        dk = None
        if duck and "duck_expect" in r:
            # duck mode (T-038 1b): scene "alone at the desk" — whether to speak now, asked after the router, same history
            pause = r.get("pause_s", 1.5)
            if duck == "bin":
                # 1b.4: да/нет by the first token (decision) and by the generated word (recorded alongside)
                dk = (duck_bin_llamaserver(host, messages, r["text"], pause) if BACKEND == "llamaserver"
                      else duck_bin_ollama(host, model, messages, r["text"], pause, ctx))
                dl, dr = decide_duck_bin(dk["ans_tok"], dk["probs"], r["text"], pause)
                lw, rw = decide_duck_bin(dk["ans_word"], {}, r["text"], pause)
                raw = {"да": "respond", "нет": duck_from_no(r["text"], pause)}.get(dk["ans_tok"])
                dk.update(label=dl, guard=dr, label_raw=raw, label_word=lw, guard_word=rw)
            else:
                dk = (duck_llamaserver(host, messages, r["text"], pause) if BACKEND == "llamaserver"
                      else duck_ollama(host, model, messages, r["text"], pause, ctx))
                dl, dr = decide_duck(dk, r["text"])
                if dr:
                    dk.update(label_raw=dk["label"], label=dl, guard=dr)
            if policy is not None and policy.get("respond") is False:
                policy = POLICY["chat"]   # in this scene everything is addressed: the router's aside does not silence
        skip = (dk["label"] == "listen" and router == "apply") if dk else (policy is not None and policy.get("respond") is False)
        if skip:
            # aside (T-038) / duck listen: the prod order — no reply, nothing goes into the history or the memory
            log.append({**r, "reply": "", "skipped": True, "route": route, "duck": dk, "think": False, "thinking_chars": 0,
                        "ctx_overflow_dropped": 0, "window_turns": len(window_turns), "prompt_eval_count": last_pe,
                        "eval_count": 0, "ttft_s": 0.0, "tok_s": None, "wall_s": round(time.perf_counter() - t0, 2),
                        "done_reason": "skipped"})
            why = f"duck listen ({dk['probs'].get('listen')})" if dk else f"aside ({route['probs'].get('aside')})"
            print(f"[{model} ctx={ctx}] t{r['turn']:02d} {why} skipped {r['text'][:60]!r}", flush=True)
            continue
        # window first (an exchange dropped on this very turn is then retrievable), retrieval second, message third
        if trim == "tokens":
            est_new = int(len(r["text"]) / 2.3) + 8 + (250 if memory is not None else 0)   # Cyrillic ≈ 2.3 chars/token; allowance for the note
            budget = (policy["max_tokens"] if policy else (think_predict if r.get("block") in think_on else 700))
            while window_turns and sys_tokens + sum(cost.get(t, 0) for t in window_turns) + est_new + budget > ctx:
                del messages[1:3]
                window_turns.pop(0)
                trimmed += 1
            if trimmed:
                print(f"[{model} ctx={ctx}] t{r['turn']:02d} window: dropped {trimmed} oldest exchange(s) (token ledger)", flush=True)
        hits, note = [], ""
        if memory is not None and (policy is None or policy["memory"]):
            if mem_scope == "dropped":
                excl = set(window_turns)
            else:
                excl = set(window_turns[-1:])
            hits = memory.retrieve(r["text"], k=mem_k, exclude_turns=excl)
            note = memory.format_note(hits)
        sent = r["text"]
        if policy and policy.get("prefix"):
            sent = policy["prefix"] + "\n" + sent
        if note:
            sent = sent + "\n\n" + note   # v3: memory after the user's words
        messages.append({"role": "user", "content": sent})
        window_turns.append(r["turn"])
        # think=False by default: thinking models (qwen3.5, gemma4) otherwise spend the whole num_predict
        # budget on message.thinking and return an empty content (2026-10-14 batch: qwen3.5:4b 60/60 empty)
        think = r.get("block") in think_on
        turn_opts = {**opts, "num_predict": think_predict} if think else opts
        if policy and not think:
            turn_opts = {**opts, "num_predict": policy["max_tokens"]}
        if dk and dk["label"] == "ack" and router == "apply":
            turn_opts = {**turn_opts, "num_predict": 5}   # «угу», «м-м»: 1-3 words
        dropped = 0
        while True:
            try:
                resp = chat_or_overflow(host, model, messages, turn_opts, think=think)
                break
            except CtxOverflow:
                # emulate what services/brain does with MAX_TURNS: drop the oldest user+assistant pair and retry
                if len(messages) < 4:
                    raise
                del messages[1:3]
                window_turns.pop(0)
                dropped += 1
        if dropped:
            print(f"[{model} ctx={ctx}] t{r['turn']:02d} ctx overflow: dropped {dropped} oldest exchange(s)", flush=True)
        wall = time.perf_counter() - t0
        reply_raw = resp["message"]["content"].strip()
        guard_flags = set()
        if memory is not None:
            reply, guard_flags = AssocMemory.clean_reply(reply_raw, note)   # what brain would say / store
        else:
            reply = reply_raw
        thinking = (resp["message"].get("thinking") or "").strip()
        thinking_chars = len(thinking)
        messages[-1] = {"role": "user", "content": r["text"]}   # history keeps the raw turn, not the memory note
        messages.append({"role": "assistant", "content": reply})
        if memory is not None:
            memory.add(r["turn"], r["text"], reply)
        pe, ec = resp.get("prompt_eval_count", 0), resp.get("eval_count", 0)
        # token ledger: this exchange's share of future prompts = its user text (without the memory note, which
        # is not kept) + its reply; the first call also calibrates the system prompt size
        note_tok = int(len(note) / 2.3) + (int(len(policy["prefix"]) / 2.3) if policy and policy.get("prefix") else 0)
        if not cost:
            sys_tokens = max(sys_tokens, pe - int(len(sent) / 2.3))
        cost[r["turn"]] = max(0, pe - last_pe - last_ec - note_tok) + ec if last_pe else max(0, pe - sys_tokens - note_tok) + ec
        last_pe, last_ec = pe, ec
        if not reply:
            print(f"[{model} ctx={ctx}] t{r['turn']:02d} EMPTY reply (ec={ec}, thinking_chars={thinking_chars})", flush=True)
        if think and not thinking_chars:
            print(f"[{model} ctx={ctx}] t{r['turn']:02d} think requested, no reasoning produced (adaptive: model skipped it, or think unsupported)", flush=True)
        entry = {**r, "reply": reply, "think": think, "thinking_chars": thinking_chars, "ctx_overflow_dropped": dropped + trimmed,
                 "window_turns": len(window_turns),
                 "prompt_eval_count": pe, "eval_count": ec,
                 "ttft_s": round((resp.get("load_duration", 0) + resp.get("prompt_eval_duration", 0)) / 1e9, 3),
                 "tok_s": round(ec / (resp.get("eval_duration", 1) / 1e9), 1) if resp.get("eval_duration") and ec >= 5 else None,
                 "wall_s": round(wall, 2)}
        if think:
            entry["thinking"] = thinking[:1500]  # for reading, not for metrics; full text would bloat the JSON
        if route is not None:
            entry["route"] = route
        if dk is not None:
            entry["duck"] = dk
        if memory is not None:
            entry["mem_hits"] = [{"turn": h[1]["turn"], "score": round(h[0], 3), "cos": round(h[2], 3), "kw": round(h[3], 3)} for h in hits]
            entry["mem_note_chars"] = len(note)
            # raw-model defects, measured on the unguarded reply: copied note / trailing aside; the guard then removes them
            low = reply_raw.lower()
            entry["mem_note_copied"] = bool(note) and (note[:40] in reply_raw or "ты помнишь" in low or "из памяти" in low or "[ты помнишь" in low)
            entry["mem_aside"] = bool(note) and bool(re.search(r"\n\s*\([^)]{15,}\)\s*$", reply_raw))
            entry["guard"] = sorted(guard_flags)
            if guard_flags:
                entry["reply_raw"] = reply_raw
        entry["done_reason"] = resp.get("done_reason")
        log.append(entry)
        tk = f" think={thinking_chars}ch" if think else ""
        rt = f" route={route['label'] or '?'}" if route else ""
        mh = f" mem={[h[1]['turn'] for h in hits]}" if memory is not None and hits else ""
        print(f"[{model} ctx={ctx}] t{r['turn']:02d} pe={pe:5d} ec={ec:4d} {wall:5.1f}s{tk}{rt}{mh} {reply[:70]!r}", flush=True)
    return log


def metrics(log, ctx):
    m = {"ctx": ctx, "recall": [], "math": [], "longgen": {}, "truncation_turns": []}
    planted = {"T1": 1, "T41": 41}
    own_answers = {}
    for e in log:
        for k in ("anchor_id", "longgen_id", "math_id"):
            if k in e:
                planted[e[k]] = e["turn"]
        if e.get("math_id"):
            own_answers[e["math_id"]] = numbers(e["reply"])[-1] if numbers(e["reply"]) else None
    prev_pe = 0
    for e in log:
        pe = e["prompt_eval_count"]
        # prompt cache normally grows monotonically; a drop means ollama shifted/truncated the context
        managed = "window_turns" in e or "mem_hits" in e or e.get("route")
        if e.get("ctx_overflow_dropped") or (not managed and e["turn"] > 2 and pe < prev_pe - 100):
            m["truncation_turns"].append(e["turn"])
        prev_pe = pe
        if e.get("probe_for") and "expect" in e and "keywords" in e["expect"]:
            ex = e["expect"]
            pf = e["probe_for"]
            if pf in own_answers:  # math recall: does it repeat its OWN earlier answer?
                own = own_answers[pf]
                own_hit = bool(own) and own.replace(",", ".").replace(":", ".") in e["reply"].replace(",", ".").replace(":", ".")
                hit2 = any_kw(e["reply"], ex["keywords"])  # and is that answer actually correct
                # a correct answer given in words ("две тысячи сорок") has no digits to match against own_answers;
                # count it as a hit too (2026-10-19 fix: t36/t55 were false misses on every e4b run)
                hit = own_hit or hit2
            else:
                hit = any_kw(e["reply"], ex["keywords"])
                hit2 = any_kw(e["reply"], ex["keywords2"]) if "keywords2" in ex else None
            # must_not: the stale fact ("куртка" for the keys after they moved to the backpack) fails the probe only when
            # the current fact is absent — mentioning the jacket as history next to the right answer is not a miss
            # (2026-10-23 fix: T-036 v6 answered "в рюкзаке … и куртку стирать" and was scored as a miss)
            bad = any_kw(e["reply"], ex["must_not"]) if "must_not" in ex else False
            m["recall"].append({"turn": e["turn"], "probe_for": pf, "distance": e["turn"] - planted.get(pf, e["turn"]),
                                "hit": bool(hit), "detail_hit": hit2, "wrong_old": bool(bad and not hit)})
        if e.get("math_id"):
            ex = e["expect"]
            answers = [ex["answer"]] + ex.get("alt", [])
            t = norm(e["reply"]).replace(" ", "")
            correct = any(norm(a).replace(" ", "") in t for a in answers)
            nums = numbers(e["reply"])
            last_ok = bool(nums) and any(nums[-1].replace(",", ".").replace(":", ".") == a.replace(":", ".") for a in answers)
            m["math"].append({"math_id": e["math_id"], "correct": correct, "answer_last": last_ok,
                              "has_working": len(nums) >= 3, "reply": e["reply"][:120]})
        if e.get("longgen_id") == "L1":
            m["longgen"]["L1_tokens"] = e["eval_count"]
            m["longgen"]["L1_ok_len"] = e["eval_count"] >= e.get("expect_min_tokens", 300)
        if e.get("hallucination_probe"):
            m["longgen"]["L2_hedged"] = any_kw(e["reply"], e["expect"]["acceptable"])
    m["empty_replies"] = sum(1 for e in log if not e["reply"].strip() and not e.get("skipped"))
    rec = m["recall"]
    m["recall_rate"] = round(sum(r["hit"] for r in rec) / len(rec), 2) if rec else None
    far = [r for r in rec if r["distance"] >= 30]
    m["recall_far_rate"] = round(sum(r["hit"] for r in far) / len(far), 2) if far else None
    m["math_acc"] = round(sum(x["correct"] for x in m["math"]) / len(m["math"]), 2) if m["math"] else None
    m["math_working"] = round(sum(x["has_working"] for x in m["math"]) / len(m["math"]), 2) if m["math"] else None
    m["ttft_last10_s"] = round(statistics.mean(e["ttft_s"] for e in log[-10:]), 2)
    toks = [e["tok_s"] for e in log if e["tok_s"]]
    m["tok_s_mean"] = round(statistics.mean(toks), 1) if toks else None
    m["ctx_tokens_end"] = log[-1]["prompt_eval_count"] + log[-1]["eval_count"]
    # E-1 (docs/COGNITION_4B.md): cost of thinking turns vs plain turns, and whether the model actually thought
    th = [e for e in log if e.get("think")]
    pl = [e for e in log if not e.get("think")]
    m["think_turns"] = len(th)
    m["think_blocks"] = sorted({e.get("block") for e in th}) if th else []
    thought = [e for e in th if e["thinking_chars"]]
    m["thought_turns"] = len(thought) if th else None  # Gemma 4 decides per turn whether to think at all (adaptive)
    m["thinking_chars_mean"] = round(statistics.mean(e["thinking_chars"] for e in thought)) if thought else None
    m["think_ignored"] = len(th) - len(thought) if th else None  # think requested, no reasoning produced
    m["wall_thought_mean_s"] = round(statistics.mean(e["wall_s"] for e in thought), 1) if thought else None
    m["empty_on_think"] = sum(1 for e in th if not e["reply"].strip()) if th else None
    m["wall_think_mean_s"] = round(statistics.mean(e["wall_s"] for e in th), 1) if th else None
    m["wall_think_max_s"] = round(max(e["wall_s"] for e in th), 1) if th else None
    m["wall_plain_mean_s"] = round(statistics.mean(e["wall_s"] for e in pl), 1) if pl else None
    m["wall_total_s"] = round(sum(e["wall_s"] for e in log))
    # E-3 router: accuracy against the dialog's `route` labels (a list = any of them is correct)
    rt = [e for e in log if e.get("route")]
    if rt:
        ok = unp = 0
        conf = {}
        for e in rt:
            lab = e["route"].get("label") or parse_label(e["route"].get("raw"))  # re-parse with the current parser
            truth = e.get("route_truth") or e.get("route_expect") or []
            truth = [truth] if isinstance(truth, str) else truth
            if lab is None:
                unp += 1
            if truth:
                hit = lab in truth
                ok += hit
                key = f"{truth[0]}->{lab or '?'}"
                if not hit:
                    conf[key] = conf.get(key, 0) + 1
        labeled = sum(1 for e in rt if e.get("route_truth") or e.get("route_expect"))
        m["router_acc"] = round(ok / labeled, 2) if labeled else None
        m["router_unparsed"] = unp
        m["router_errors"] = conf
        m["router_s_mean"] = round(statistics.mean(e["route"]["s"] for e in rt), 2)
    # MEM-1: did retrieval bring the planted turn to a probe? (independent of what the model then said)
    mem = [e for e in log if "mem_hits" in e]
    if mem:
        pr = [e for e in mem if e.get("probe_for") and e["probe_for"] in planted]
        got = [e for e in pr if planted[e["probe_for"]] in {h["turn"] for h in e["mem_hits"]}]
        m["mem_probe_turns"] = len(pr)
        m["mem_retrieval_recall"] = round(len(got) / len(pr), 2) if pr else None
        m["mem_note_chars_mean"] = round(statistics.mean(e["mem_note_chars"] for e in mem))
        m["mem_note_turns"] = sum(1 for e in mem if e["mem_note_chars"])
        m["mem_note_copied"] = sum(1 for e in mem if e.get("mem_note_copied"))
        m["mem_asides"] = sum(1 for e in mem if e.get("mem_aside"))
        m["guard_fired"] = sum(1 for e in mem if e.get("guard"))
    m["cut_by_length"] = sum(1 for e in log if e.get("done_reason") == "length")
    # T-038 ambient-lite: "was it said to me?" — class `me` = the skull answers (final router label is not aside);
    # turns whose addr_expect is a list accept any label and stay out of precision/recall
    ad = [e for e in log if "addr_expect" in e]
    if ad:
        cnt = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "either": 0}
        errs = []
        for e in ad:
            rt = e.get("route") or {}
            pred = "aside" if rt.get("label") == "aside" else "me"
            exp = e["addr_expect"]
            if isinstance(exp, list):
                cnt["either"] += 1
                continue
            key = ("t" if pred == exp else "f") + ("p" if pred == "me" else "n")
            cnt[key] += 1
            if pred != exp:
                raw = rt.get("label_raw") or rt.get("label")
                errs.append({"turn": e["turn"], "kind": e.get("kind"), "expect": exp, "label": rt.get("label"),
                             "label_raw": raw, "guard": rt.get("guard"), "p": (rt.get("probs") or {}).get(raw),
                             "probs": rt.get("probs"), "text": e["text"]})
        m["addr_counts"] = cnt
        m["addr_recall"] = round(cnt["tp"] / (cnt["tp"] + cnt["fn"]), 2) if cnt["tp"] + cnt["fn"] else None
        m["addr_precision"] = round(cnt["tp"] / (cnt["tp"] + cnt["fp"]), 2) if cnt["tp"] + cnt["fp"] else None
        m["addr_errors"] = errs
    # T-038 1b duck mode: respond / ack / listen on the turns of the "alone" scene; lists accept any of their labels
    dkl = [e for e in log if "duck_expect" in e and e.get("duck")]
    if dkl:
        def hit(e, lab):
            exp = e["duck_expect"]
            return lab in exp if isinstance(exp, list) else lab == exp
        getters = [("", lambda e: e["duck"]["label"]), ("_raw", lambda e: e["duck"].get("label_raw") or e["duck"]["label"])]
        if any("label_word" in e["duck"] for e in dkl):   # 1b.4: the same decision from the generated word
            getters.append(("_word", lambda e: e["duck"].get("label_word")))
        for suffix, get in getters:
            resp = [e for e in dkl if e["duck_expect"] == "respond"]
            lis = [e for e in dkl if e["duck_expect"] == "listen"]
            m["duck_acc" + suffix] = round(sum(hit(e, get(e)) for e in dkl) / len(dkl), 2)
            m["duck_respond_recall" + suffix] = round(sum(get(e) == "respond" for e in resp) / len(resp), 2) if resp else None
            m["duck_interrupt" + suffix] = round(sum(get(e) == "respond" for e in lis) / len(lis), 2) if lis else None
        m["duck_n"] = len(dkl)
        m["duck_errors"] = [{"turn": e["turn"], "kind": e.get("kind"), "expect": e["duck_expect"], "pause_s": e.get("pause_s"),
                             "label": e["duck"]["label"], "label_raw": e["duck"].get("label_raw") or e["duck"]["label"],
                             "guard": e["duck"].get("guard"), "probs": e["duck"].get("probs"), "text": e["text"]}
                            for e in dkl if not hit(e, e["duck"]["label"])]
    return m


def judge(host, judge_model, log):
    out = {"persona": [], "consistency": None, "hallucination": None}
    opts = {"num_ctx": 8192, "temperature": 0.0, "seed": 1, "num_predict": 200}

    def ask(prompt):
        resp = chat(host, judge_model, [{"role": "user", "content": prompt}], opts, think=False)
        txt = resp["message"].get("content") or resp["message"].get("thinking") or ""
        mt = re.search(r"\{.*?\}", txt, re.S)
        try:
            return json.loads(mt.group(0)) if mt else {"raw": txt[:200]}
        except Exception:
            return {"raw": txt[:200]}

    for e in log:
        if e.get("judge_persona"):
            if not e["reply"].strip():
                # an empty reply has no "excitement or apologies" and the rubric scores it 5; it is a failure
                res = {"score": 0, "why": "empty reply"}
            else:
                res = ask(JUDGE_RUBRIC % (PERSONA, e["text"], e["reply"]))
            out["persona"].append({"turn": e["turn"], **res})
            print(f"  judge persona t{e['turn']:02d}: {res}", flush=True)
    l1 = next((e for e in log if e.get("longgen_id") == "L1"), None)
    l3 = next((e for e in log if e.get("longgen_id") == "L3"), None)
    if l1 and l3:
        out["consistency"] = ask(JUDGE_CONSISTENCY % (l1["reply"], l3["reply"]))
    l2 = next((e for e in log if e.get("hallucination_probe")), None)
    if l2:
        out["hallucination"] = ask(JUDGE_HALLUCINATION % l2["reply"])
    scores = [p["score"] for p in out["persona"] if isinstance(p.get("score"), (int, float))]
    out["persona_mean"] = round(statistics.mean(scores), 2) if scores else None
    return out


def write_summary(model, tag, summary, prefix="ctx_probe"):
    md = [f"# CTX-1 · {model} · {date.today().isoformat()}", "",
          "| num_ctx | thinking на блоках | пустых ответов | recall (все) | recall (≥30) | math верно | math с выкладкой | L1 токенов | L2 оговорка | обрезка на ходах | персона (судья, 1–5) | согласованность L1/L3 | L2 выдумал | TTFT посл.10, с | tok/s | ход с размышлением / без, с (макс) | симв. размышления (средн. по думавшим) | токенов в контексте к концу | память: recall выборки / символов справки | роутер: точность / с на ход |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, j in summary:
        tr = ",".join(str(t) for t in m["truncation_turns"]) or "—"
        j = j or {}
        tbl = m.get("think_blocks") or []
        tb = ("all" if set(tbl) == set(BLOCKS) else "+".join(tbl)) or "—"
        if m.get("think_turns"):
            tb += f" (думала на {m.get('thought_turns')} из {m['think_turns']} ходов)"
        wall = f"{m.get('wall_thought_mean_s', '—')} / {m.get('wall_plain_mean_s', '—')} ({m.get('wall_think_max_s', '—')})"
        md.append(f"| {m['ctx']} | {tb} | {m.get('empty_replies', '—')} | {m['recall_rate']} | {m['recall_far_rate']} | {m['math_acc']} | {m['math_working']} | "
                  f"{m['longgen'].get('L1_tokens')} | {m['longgen'].get('L2_hedged')} | {tr} | "
                  f"{j.get('persona_mean', '—')} | {(j.get('consistency') or {}).get('score', '—')} | "
                  f"{(j.get('hallucination') or {}).get('invented', '—')} | {m['ttft_last10_s']} | {m['tok_s_mean']} | {wall} | "
                  f"{m.get('thinking_chars_mean', '—')} | {m['ctx_tokens_end']} | "
                  f"{m.get('mem_retrieval_recall', '—')} / {m.get('mem_note_chars_mean', '—')}"
                  + (f" (справка на {m.get('mem_note_turns')} ходах; сырая модель: скопировала на {m.get('mem_note_copied')}, ремарок {m.get('mem_asides')}; фильтр сработал на {m.get('guard_fired')})" if m.get("mem_note_turns") is not None else "") + (f"; обрезано по длине {m.get('cut_by_length')}" if m.get("cut_by_length") else "") + " | "
                  f"{m.get('router_acc', '—')} / {m.get('router_s_mean', '—')} |")
    md += ["", "Recall по расстоянию (ход/проба@расстояние): ✓ попал, ✗ нет; для M-проб ✓ = повторил свой же ранний ответ.", ""]
    for m, _ in summary:
        md.append(f"- ctx {m['ctx']}: " + ", ".join(f"t{r['turn']}/{r['probe_for']}@{r['distance']}={'✓' if r['hit'] else '✗'}" for r in m["recall"]))
    md += ["", "Математика (ответ модели, первые 120 символов):", ""]
    for m, _ in summary:
        md.append(f"- ctx {m['ctx']}: " + " · ".join(f"{x['math_id']}: {x['reply']!r}" for x in m["math"]))
    for m, _ in summary:
        if m.get("router_errors"):
            md.append(f"- ctx {m['ctx']}: ошибки роутера (истина->ответ: n): " + ", ".join(f"{k}: {v}" for k, v in sorted(m["router_errors"].items())))
    md += ["", "Оценки персоны по ходам:", ""]
    for m, j in summary:
        if j:
            md.append(f"- ctx {m['ctx']}: " + ", ".join(f"t{p['turn']}={p.get('score', '?')}" for p in j["persona"]))
    (OUT / f"{prefix}_{tag}_{date.today().isoformat()}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--ctx", nargs="+", type=int, default=[2048, 4096, 8192, 16384])
    ap.add_argument("--judge", default=None)
    ap.add_argument("--judge-only", action="store_true", help="re-score saved JSON logs, do not re-run dialogs; without --judge only metrics are recomputed, saved judge kept")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "llamaserver"],
                    help="llamaserver: --host points at a running llama-server (-c = num_ctx); --model is only a label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duck", nargs="?", const="tri", default=None, choices=["tri", "bin"],
                    help="T-038 1b: duck-mode pass on turns with duck_expect; tri = respond/ack/listen asked of the model, "
                         "bin = да/нет asked (1b.4), ack by the pause rule")
    ap.add_argument("--dialog", default=str(DIALOG), help="dialog JSONL (default ctx_probe_dialog.jsonl; T-038: ctx_probe_ambient.jsonl)")
    ap.add_argument("--persona", default=None, help="persona prompt file (default: persona/system_prompt.md)")
    ap.add_argument("--tag", default="", help="suffix for output file names, e.g. 3ex for a prompt variant")
    ap.add_argument("--think-on", default="", metavar="BLOCKS",
                    help="E-1: dialog blocks with think=true, comma-separated (everyday,probe,longgen,random,math; "
                         "aliases recall=probe; 'all'; default none). Adds _think-<blocks> to file names")
    ap.add_argument("--think-predict", type=int, default=2500,
                    help="num_predict on thinking turns (reasoning tokens share the budget); plain turns keep 700")
    ap.add_argument("--memory", default="off", choices=["off", "all", "dropped"],
                    help="MEM-1: associative memory over past exchanges; 'dropped' (the brain policy, D-026) retrieves only "
                         "among exchanges that fell out of num_ctx; 'all' among everything said (at 8k it only confuses)")
    ap.add_argument("--mem-k", type=int, default=3)
    ap.add_argument("--mem-mode", default="hybrid", choices=["hybrid", "embed", "kw"])
    ap.add_argument("--embed-model", default="embeddinggemma", help="ollama embedding model (/api/embed)")
    ap.add_argument("--router", default="off", choices=["off", "label", "apply"],
                    help="E-3: classify each turn (label = log only; apply = use POLICY from research/memory/router.py)")
    ap.add_argument("--trim", default=None, choices=["ollama", "tokens"],
                    help="window management: 'ollama' (default) lets the server cut the prompt; 'tokens' drops oldest "
                         "exchanges by a token ledger before the call (default when --memory dropped)")
    args = ap.parse_args()
    global BACKEND
    BACKEND = args.backend
    think_on = parse_think_on(args.think_on)

    global PERSONA
    persona_file = Path(args.persona) if args.persona else ROOT / "persona/system_prompt.md"
    PERSONA = persona_file.read_text(encoding="utf-8").strip()

    dialog = Path(args.dialog)
    rows = [json.loads(l) for l in dialog.read_text(encoding="utf-8").splitlines() if l.strip()]
    prefix = "ctx_probe" if dialog.resolve() == DIALOG.resolve() else dialog.stem   # ctx_probe_ambient for T-038
    tag = args.model.replace(":", "_").replace("/", "_")
    if args.seed != 42:  # keep the default-seed file name stable; other seeds get their own files
        tag += f"_s{args.seed}"
    if args.tag:
        tag += f"_{args.tag}"
    if think_on:
        tag += "_think-" + ("all" if think_on == set(BLOCKS) else "+".join(b for b in BLOCKS if b in think_on))
    if args.memory != "off":
        tag += f"_mem-{args.memory}-{args.mem_mode}-k{args.mem_k}"
    if args.router != "off":
        tag += f"_rt-{args.router}"
    trim = args.trim or ("tokens" if args.memory == "dropped" else "ollama")
    if trim == "tokens":
        tag += "_trim"
    if args.duck:
        tag += "_duck" + ("-bin" if args.duck == "bin" else "")
    summary = []
    for ctx in args.ctx:
        path = OUT / f"{prefix}_{tag}_{ctx}.json"
        saved_judge = None
        if args.judge_only:
            saved = json.loads(path.read_text(encoding="utf-8"))
            log, saved_judge = saved["log"], saved.get("judge")
            # expect lists live in the dialog file and may have been extended since the run: overlay them
            by_turn = {r["turn"]: r for r in rows}
            log = [{**e, **({"expect": by_turn[e["turn"]]["expect"]} if "expect" in by_turn.get(e["turn"], {}) else {})} for e in log]
        else:
            memory = AssocMemory(args.host, args.embed_model, mode=args.mem_mode) if args.memory != "off" else None
            log = run_dialog(args.host, args.model, ctx, rows, args.seed, think_on=think_on, think_predict=args.think_predict,
                             memory=memory, mem_scope=args.memory, mem_k=args.mem_k, router=args.router, trim=trim,
                             duck=args.duck)
            if memory is not None and memory.embed_calls_s:
                print(f"embed calls: {len(memory.embed_calls_s)}, mean {statistics.mean(memory.embed_calls_s)*1000:.0f} ms", flush=True)
        m = metrics(log, ctx)
        # --judge-only without --judge: recompute metrics only, keep the saved judge scores
        j = judge(args.host, args.judge, log) if args.judge else saved_judge
        path.write_text(json.dumps({"model": args.model, "ctx": ctx, "seed": args.seed,
                                    "think_on": sorted(think_on), "think_predict": args.think_predict if think_on else None,
                                    "memory": args.memory, "mem_k": args.mem_k, "mem_mode": args.mem_mode,
                                    "embed_model": args.embed_model if args.memory != "off" else None, "router": args.router, "trim": trim,
                                    "note_version": 6 if args.memory != "off" else None, "backend": args.backend,
                                    "persona_file": str(persona_file.relative_to(ROOT)) if persona_file.is_relative_to(ROOT) else str(persona_file),
                                    "persona_sha1": __import__("hashlib").sha1(PERSONA.encode("utf-8")).hexdigest()[:10],
                                    "metrics": m, "judge": j, "log": log}, ensure_ascii=False, indent=1), encoding="utf-8")
        summary.append((m, j))
        print(f"== {args.model} ctx={ctx}: recall={m['recall_rate']} far={m['recall_far_rate']} math={m['math_acc']} "
              f"trunc={m['truncation_turns']} persona={(j or {}).get('persona_mean')}"
              + (f" addr_recall={m['addr_recall']} addr_precision={m['addr_precision']} {m['addr_counts']}" if "addr_counts" in m else "")
              + (f" duck_acc={m['duck_acc']} respond_recall={m['duck_respond_recall']} interrupt={m['duck_interrupt']}"
                 f" (raw {m['duck_acc_raw']}/{m['duck_respond_recall_raw']}/{m['duck_interrupt_raw']})"
                 + (f" (word {m['duck_acc_word']}/{m['duck_respond_recall_word']}/{m['duck_interrupt_word']})" if "duck_acc_word" in m else "")
                 if "duck_n" in m else ""),
              flush=True)
    write_summary(args.model, tag, summary, prefix)


if __name__ == "__main__":
    main()
