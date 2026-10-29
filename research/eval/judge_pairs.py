"""Pairwise blinded persona judge (Q-053, P-019): instead of absolute 1-5 scores (where qwen3.5:9b is unreliable, D-017)
the judge compares two replies to the same user turn and says which is more in character. Order is randomised and
every pair is asked twice with the order swapped (position-bias check).

Usage:
    python research/eval/judge_pairs.py --judge qwen3.5:9b --pairs label=A.json:B.json [label=...] [--expect label=A|B|tie ...]
Each pair: all `judge_persona` turns of the two CTX-1 logs (same dialog). Output: results/judge_pairs_<date>.csv + .md
with per-pair wins, position-bias rate, and agreement with --expect (the core's verdicts from the decision log).

--expect-score (T-043, P-039 LLM-as-a-Verifier): the same pair, but the answer is one letter A..T on a scale
(A — variant 1 clearly better, K — equal, T — variant 2 clearly better); the score is the expectation over the
first token's top_logprobs (20; Cyrillic look-alikes count as the Latin letter), φ(A) = -1, φ(K) = 0, φ(T) = +1,
renormalised over the letters found. Both orders; a turn is a tie when |E| < TIE in both, a flip when the sign of E
(in the A/B frame) changes between the orders by more than TIE. Output: judge_pairs_<date>_<judge>_expect.csv/.md
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/eval/results"
PERSONA = (ROOT / "persona/system_prompt.md").read_text(encoding="utf-8").strip()

PROMPT = """Ты — строгий эксперт по оценке реплик голосового ассистента-персонажа. Персонаж описан между тегами <persona>.
Собеседник сказал реплику, ассистент ответил двумя вариантами. Какой вариант БОЛЬШЕ в образе персонажа: суше, ироничнее, короче, по делу, без услужливости, извинений, восторгов, лекций и смайликов; по-русски?
Ответь строго одной строкой JSON: {{"better": 1}} или {{"better": 2}} или {{"better": 0}} если они равны.
<persona>
{persona}
</persona>

Реплика собеседника: {user}
Вариант 1: {a}
Вариант 2: {b}"""


PROMPT_SCALE = """Ты — строгий эксперт по оценке реплик голосового ассистента-персонажа. Персонаж описан между тегами <persona>.
Собеседник сказал реплику, ассистент ответил двумя вариантами. Какой вариант БОЛЬШЕ в образе персонажа: суше, ироничнее, короче, по делу, без услужливости, извинений, восторгов, лекций и смайликов; по-русски?
Ответь ОДНОЙ латинской буквой от A до T по шкале: A — вариант 1 явно лучше; K — варианты равны; T — вариант 2 явно лучше; буквы между ними — степень перевеса.
<persona>
{persona}
</persona>

Реплика собеседника: {user}
Вариант 1: {a}
Вариант 2: {b}"""
LETTERS = "ABCDEFGHIJKLMNOPQRST"
PHI = {c: (i - 10) / (10 if i <= 10 else 9) for i, c in enumerate(LETTERS)}   # A -1, K 0, T +1
_LOOKALIKE = str.maketrans("АВСЕКМНОРТХ", "ABCEKMHOPTX")
TIE = 0.1


def ask_scale(host, model, prompt, timeout=300):
    """Expectation of φ over the first token's top_logprobs: (E, letter mass, top letter); E > 0 — variant 2 better."""
    import math
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "think": False,
            "keep_alive": "10m", "logprobs": True, "top_logprobs": 20,
            "options": {"num_ctx": 8192, "temperature": 0.0, "seed": 1, "num_predict": 1}}
    r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    probs = {}
    for alt in (r.json().get("logprobs") or [{}])[0].get("top_logprobs", []):
        c = alt["token"].strip().upper().translate(_LOOKALIKE)
        if len(c) == 1 and c in PHI:
            probs[c] = probs.get(c, 0.0) + math.exp(alt["logprob"])
    mass = sum(probs.values())
    if mass < 0.5:
        return None, round(mass, 3), None
    return sum(p * PHI[c] for c, p in probs.items()) / mass, round(mass, 3), max(probs, key=probs.get)


def ask(host, model, prompt, timeout=300):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "think": False,
            "keep_alive": "10m", "options": {"num_ctx": 8192, "temperature": 0.0, "seed": 1, "num_predict": 30}}
    r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    if r.status_code == 400 and "think" in r.text.lower():
        body.pop("think"); r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    txt = r.json()["message"].get("content") or ""
    m = re.search(r'"better"\s*:\s*([012])', txt)
    return (int(m.group(1)) if m else None), txt[:60]


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    return {e["turn"]: e for e in d["log"]}, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--judge", default="qwen3.5:9b")
    ap.add_argument("--pairs", nargs="+", required=True, help="label=A.json:B.json (paths relative to results/ or absolute)")
    ap.add_argument("--expect", nargs="*", default=[], help="label=A|B|tie — the core's verdict for agreement stats")
    ap.add_argument("--expect-score", action="store_true", help="T-043: letter scale A..T, expectation over logprobs")
    args = ap.parse_args()
    expect = dict(x.split("=", 1) for x in args.expect)
    rng = random.Random(7)
    rows = []
    summary = []
    for spec in args.pairs:
        label, paths = spec.split("=", 1)
        pa, pb = paths.split(":", 1)
        pa = Path(pa) if Path(pa).is_absolute() else OUT / pa
        pb = Path(pb) if Path(pb).is_absolute() else OUT / pb
        A, dA = load(pa); B, dB = load(pb)
        turns = [t for t, e in A.items() if e.get("judge_persona") and t in B]
        wins = {"A": 0, "B": 0, "tie": 0}; flips = 0; unparsed = 0
        t0 = time.perf_counter()
        for t in turns:
            ra, rb = A[t]["reply"].strip(), B[t]["reply"].strip()
            if not ra or not rb:
                continue
            votes, scores, extra = [], [], {}
            for order in (0, 1) if rng.random() < 0.5 else (1, 0):
                x, y = (ra, rb) if order == 0 else (rb, ra)
                if args.expect_score:
                    e, mass, top = ask_scale(args.host, args.judge, PROMPT_SCALE.format(persona=PERSONA, user=A[t]["text"], a=x, b=y))
                    extra.setdefault("mass", []).append(mass); extra.setdefault("top", []).append(top)
                    if e is None:
                        unparsed += 1; scores.append(None); continue
                    scores.append(round(e if order == 0 else -e, 3))   # A/B frame: > 0 — B better
                    continue
                v, raw = ask(args.host, args.judge, PROMPT.format(persona=PERSONA, user=A[t]["text"], a=x, b=y))
                if v is None:
                    unparsed += 1; votes.append(None); continue
                # map back to A/B
                votes.append("tie" if v == 0 else ("A" if (v == 1) == (order == 0) else "B"))
            if args.expect_score:
                s1, s2 = scores
                if s1 is None or s2 is None:
                    verdict = "unparsed"
                elif abs(s1) < TIE and abs(s2) < TIE:
                    verdict = "tie"
                elif (s1 > 0) != (s2 > 0) and abs(s1 - s2) > TIE and min(abs(s1), abs(s2)) >= TIE:
                    verdict = "flip"; flips += 1
                else:
                    m = (s1 + s2) / 2
                    verdict = "tie" if abs(m) < TIE else ("B" if m > 0 else "A")
                v1, v2 = s1, s2
            else:
                v1, v2 = votes
                if v1 is None or v2 is None:
                    verdict = "unparsed"
                elif v1 == v2:
                    verdict = v1
                else:
                    verdict = "flip"; flips += 1
            if verdict in wins:
                wins[verdict] += 1
            rows.append({"pair": label, "turn": t, "vote_order1": v1, "vote_order2": v2, "verdict": verdict,
                         **({"mass": extra.get("mass"), "top": extra.get("top")} if args.expect_score else {}),
                         "reply_a": ra[:120], "reply_b": rb[:120]})
            print(f"[{label}] t{t:02d} {v1}/{v2} -> {verdict}" + (f" mass {extra.get('mass')} top {extra.get('top')}" if args.expect_score else ""), flush=True)
        n = len(turns)
        decided = wins["A"] + wins["B"]
        pref = "A" if wins["A"] > wins["B"] else ("B" if wins["B"] > wins["A"] else "tie")
        exp = expect.get(label)
        agree = (exp == pref) if exp else None
        es = [(r["vote_order1"] + r["vote_order2"]) / 2 for r in rows if r["pair"] == label and r["verdict"] not in ("unparsed",)
              and args.expect_score]
        summary.append({"pair": label, "A": pa.name, "B": pb.name, "turns": n, "wins_A": wins["A"], "wins_B": wins["B"],
                        "mean_e": round(sum(es) / len(es), 3) if es else None,
                        "ties": wins["tie"], "flips": flips, "unparsed": unparsed, "flip_rate": round(flips / n, 2) if n else None,
                        "pref": pref, "expect": exp, "agree": agree, "s": round(time.perf_counter() - t0)})
        print(f"== {label}: A {wins['A']} / B {wins['B']} / tie {wins['tie']} / flip {flips} / unparsed {unparsed} -> {pref} (expect {exp})", flush=True)
    stamp = date.today().isoformat() + "_" + re.sub(r"[^a-z0-9]+", "-", args.judge.lower()) + ("_expect" if args.expect_score else "")
    with open(OUT / f"judge_pairs_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = [f"# Парный слепой судья ({args.judge}{', матожидание по логпробам A–T' if args.expect_score else ''}) · {stamp}", "",
          "| пара | A | B | ходов | A | B | ничья | перевороты | нераспознано | вердикт | ожидание ядра | совпало |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        md.append(f"| {s['pair']} | {s['A']} | {s['B']} | {s['turns']} | {s['wins_A']} | {s['wins_B']} | {s['ties']} | {s['flips']} ({s['flip_rate']}) | {s['unparsed']} | {s['pref']} | {s['expect'] or '—'} | {'' if s['agree'] is None else ('да' if s['agree'] else 'нет')} |")
    agreed = [s for s in summary if s["agree"] is not None]
    if agreed:
        md += ["", f"Совпадение с вердиктами ядра: {sum(s['agree'] for s in agreed)} / {len(agreed)}; средняя доля переворотов (позиционный сдвиг): {round(sum(s['flip_rate'] or 0 for s in summary) / len(summary), 2)}"]
    if args.expect_score:
        md += ["", "Средняя оценка E по паре (> 0 — лучше B): " + ", ".join(f"{s['pair']} {s['mean_e']}" for s in summary),
               f"Время: {sum(s['s'] for s in summary)} с на {sum(s['turns'] for s in summary) * 2} запросов"]
    (OUT / f"judge_pairs_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
