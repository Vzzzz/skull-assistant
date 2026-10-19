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
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
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


def chat(host, model, messages, options, think=None, timeout=600):
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
    return s.lower().replace("ё", "е")


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


def run_dialog(host, model, ctx, rows, seed, think_on=frozenset(), think_predict=2500):
    """think_on: dialog blocks on which the model is asked to think (E-1, docs/COGNITION_4B.md). On those turns
    num_predict is raised to think_predict because the reasoning tokens come out of the same budget; the
    reasoning text is kept in the log (truncated) but, as in services/brain, is NOT fed back into the history."""
    messages = [{"role": "system", "content": PERSONA}]
    log = []
    opts = {"num_ctx": ctx, "temperature": 0.3, "seed": seed, "num_predict": 700}
    for r in rows:
        messages.append({"role": "user", "content": r["text"]})
        t0 = time.perf_counter()
        # think=False by default: thinking models (qwen3.5, gemma4) otherwise spend the whole num_predict
        # budget on message.thinking and return an empty content (2026-10-14 batch: qwen3.5:4b 60/60 empty)
        think = r.get("block") in think_on
        turn_opts = {**opts, "num_predict": think_predict} if think else opts
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
                dropped += 1
        if dropped:
            print(f"[{model} ctx={ctx}] t{r['turn']:02d} ctx overflow: dropped {dropped} oldest exchange(s)", flush=True)
        wall = time.perf_counter() - t0
        reply = resp["message"]["content"].strip()
        thinking = (resp["message"].get("thinking") or "").strip()
        thinking_chars = len(thinking)
        messages.append({"role": "assistant", "content": reply})
        pe, ec = resp.get("prompt_eval_count", 0), resp.get("eval_count", 0)
        if not reply:
            print(f"[{model} ctx={ctx}] t{r['turn']:02d} EMPTY reply (ec={ec}, thinking_chars={thinking_chars})", flush=True)
        if think and not thinking_chars:
            print(f"[{model} ctx={ctx}] t{r['turn']:02d} think requested but no thinking returned (model/ollama ignores think)", flush=True)
        entry = {**r, "reply": reply, "think": think, "thinking_chars": thinking_chars, "ctx_overflow_dropped": dropped,
                 "prompt_eval_count": pe, "eval_count": ec,
                 "ttft_s": round((resp.get("load_duration", 0) + resp.get("prompt_eval_duration", 0)) / 1e9, 3),
                 "tok_s": round(ec / (resp.get("eval_duration", 1) / 1e9), 1) if resp.get("eval_duration") and ec >= 5 else None,
                 "wall_s": round(wall, 2)}
        if think:
            entry["thinking"] = thinking[:1500]  # for reading, not for metrics; full text would bloat the JSON
        log.append(entry)
        tk = f" think={thinking_chars}ch" if think else ""
        print(f"[{model} ctx={ctx}] t{r['turn']:02d} pe={pe:5d} ec={ec:4d} {wall:5.1f}s{tk} {reply[:70]!r}", flush=True)
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
        if (e["turn"] > 2 and pe < prev_pe - 100) or e.get("ctx_overflow_dropped"):
            m["truncation_turns"].append(e["turn"])
        prev_pe = pe
        if e.get("probe_for") and "expect" in e and "keywords" in e["expect"]:
            ex = e["expect"]
            pf = e["probe_for"]
            if pf in own_answers:  # math recall: does it repeat its OWN earlier answer?
                own = own_answers[pf]
                hit = bool(own) and own.replace(",", ".").replace(":", ".") in e["reply"].replace(",", ".").replace(":", ".")
                hit2 = any_kw(e["reply"], ex["keywords"])  # and is that answer actually correct
            else:
                hit = any_kw(e["reply"], ex["keywords"])
                hit2 = any_kw(e["reply"], ex["keywords2"]) if "keywords2" in ex else None
            bad = any_kw(e["reply"], ex["must_not"]) if "must_not" in ex else False
            m["recall"].append({"turn": e["turn"], "probe_for": pf, "distance": e["turn"] - planted.get(pf, e["turn"]),
                                "hit": bool(hit and not bad), "detail_hit": hit2, "wrong_old": bad})
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
    m["empty_replies"] = sum(1 for e in log if not e["reply"].strip())
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
    m["thinking_chars_mean"] = round(statistics.mean(e["thinking_chars"] for e in th)) if th else None
    m["think_ignored"] = sum(1 for e in th if not e["thinking_chars"]) if th else None
    m["empty_on_think"] = sum(1 for e in th if not e["reply"].strip()) if th else None
    m["wall_think_mean_s"] = round(statistics.mean(e["wall_s"] for e in th), 1) if th else None
    m["wall_think_max_s"] = round(max(e["wall_s"] for e in th), 1) if th else None
    m["wall_plain_mean_s"] = round(statistics.mean(e["wall_s"] for e in pl), 1) if pl else None
    m["wall_total_s"] = round(sum(e["wall_s"] for e in log))
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


def write_summary(model, tag, summary):
    md = [f"# CTX-1 · {model} · {date.today().isoformat()}", "",
          "| num_ctx | thinking на блоках | пустых ответов | recall (все) | recall (≥30) | math верно | math с выкладкой | L1 токенов | L2 оговорка | обрезка на ходах | персона (судья, 1–5) | согласованность L1/L3 | L2 выдумал | TTFT посл.10, с | tok/s | ход с think / без, с (макс) | симв. размышления | токенов в контексте к концу |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, j in summary:
        tr = ",".join(str(t) for t in m["truncation_turns"]) or "—"
        j = j or {}
        tb = "+".join(m.get("think_blocks") or []) or "—"
        if m.get("think_turns"):
            tb += f" ({m['think_turns']} ходов" + (f", флаг проигнорирован на {m['think_ignored']}" if m.get("think_ignored") else "") + ")"
        wall = f"{m.get('wall_think_mean_s', '—')} / {m.get('wall_plain_mean_s', '—')} ({m.get('wall_think_max_s', '—')})"
        md.append(f"| {m['ctx']} | {tb} | {m.get('empty_replies', '—')} | {m['recall_rate']} | {m['recall_far_rate']} | {m['math_acc']} | {m['math_working']} | "
                  f"{m['longgen'].get('L1_tokens')} | {m['longgen'].get('L2_hedged')} | {tr} | "
                  f"{j.get('persona_mean', '—')} | {(j.get('consistency') or {}).get('score', '—')} | "
                  f"{(j.get('hallucination') or {}).get('invented', '—')} | {m['ttft_last10_s']} | {m['tok_s_mean']} | {wall} | "
                  f"{m.get('thinking_chars_mean', '—')} | {m['ctx_tokens_end']} |")
    md += ["", "Recall по расстоянию (ход/проба@расстояние): ✓ попал, ✗ нет; для M-проб ✓ = повторил свой же ранний ответ.", ""]
    for m, _ in summary:
        md.append(f"- ctx {m['ctx']}: " + ", ".join(f"t{r['turn']}/{r['probe_for']}@{r['distance']}={'✓' if r['hit'] else '✗'}" for r in m["recall"]))
    md += ["", "Математика (ответ модели, первые 120 символов):", ""]
    for m, _ in summary:
        md.append(f"- ctx {m['ctx']}: " + " · ".join(f"{x['math_id']}: {x['reply']!r}" for x in m["math"]))
    md += ["", "Оценки персоны по ходам:", ""]
    for m, j in summary:
        if j:
            md.append(f"- ctx {m['ctx']}: " + ", ".join(f"t{p['turn']}={p.get('score', '?')}" for p in j["persona"]))
    (OUT / f"ctx_probe_{tag}_{date.today().isoformat()}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--ctx", nargs="+", type=int, default=[2048, 4096, 8192, 16384])
    ap.add_argument("--judge", default=None)
    ap.add_argument("--judge-only", action="store_true", help="re-score saved JSON logs, do not re-run dialogs")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--persona", default=None, help="persona prompt file (default: persona/system_prompt.md)")
    ap.add_argument("--tag", default="", help="suffix for output file names, e.g. 3ex for a prompt variant")
    ap.add_argument("--think-on", default="", metavar="BLOCKS",
                    help="E-1: dialog blocks with think=true, comma-separated (everyday,probe,longgen,random,math; "
                         "aliases recall=probe; 'all'; default none). Adds _think-<blocks> to file names")
    ap.add_argument("--think-predict", type=int, default=2500,
                    help="num_predict on thinking turns (reasoning tokens share the budget); plain turns keep 700")
    args = ap.parse_args()
    think_on = parse_think_on(args.think_on)

    global PERSONA
    persona_file = Path(args.persona) if args.persona else ROOT / "persona/system_prompt.md"
    PERSONA = persona_file.read_text(encoding="utf-8").strip()

    rows = [json.loads(l) for l in DIALOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    tag = args.model.replace(":", "_").replace("/", "_")
    if args.seed != 42:  # keep the default-seed file name stable; other seeds get their own files
        tag += f"_s{args.seed}"
    if args.tag:
        tag += f"_{args.tag}"
    if think_on:
        tag += "_think-" + ("all" if think_on == set(BLOCKS) else "+".join(b for b in BLOCKS if b in think_on))
    summary = []
    for ctx in args.ctx:
        path = OUT / f"ctx_probe_{tag}_{ctx}.json"
        if args.judge_only:
            log = json.loads(path.read_text(encoding="utf-8"))["log"]
        else:
            log = run_dialog(args.host, args.model, ctx, rows, args.seed, think_on=think_on, think_predict=args.think_predict)
        m = metrics(log, ctx)
        j = judge(args.host, args.judge, log) if args.judge else None
        path.write_text(json.dumps({"model": args.model, "ctx": ctx, "seed": args.seed,
                                    "think_on": sorted(think_on), "think_predict": args.think_predict if think_on else None,
                                    "persona_file": str(persona_file.relative_to(ROOT)) if persona_file.is_relative_to(ROOT) else str(persona_file),
                                    "persona_sha1": __import__("hashlib").sha1(PERSONA.encode("utf-8")).hexdigest()[:10],
                                    "metrics": m, "judge": j, "log": log}, ensure_ascii=False, indent=1), encoding="utf-8")
        summary.append((m, j))
        print(f"== {args.model} ctx={ctx}: recall={m['recall_rate']} far={m['recall_far_rate']} math={m['math_acc']} "
              f"trunc={m['truncation_turns']} persona={(j or {}).get('persona_mean')}", flush=True)
    write_summary(args.model, tag, summary)


if __name__ == "__main__":
    main()
