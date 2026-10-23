"""Pairwise blinded persona judge (Q-053, P-019): instead of absolute 1-5 scores (where qwen3.5:9b is unreliable, D-017)
the judge compares two replies to the same user turn and says which is more in character. Order is randomised and
every pair is asked twice with the order swapped (position-bias check).

Usage:
    python research/eval/judge_pairs.py --judge qwen3.5:9b --pairs label=A.json:B.json [label=...] [--expect label=A|B|tie ...]
Each pair: all `judge_persona` turns of the two CTX-1 logs (same dialog). Output: results/judge_pairs_<date>.csv + .md
with per-pair wins, position-bias rate, and agreement with --expect (the core's verdicts from the decision log).
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
            votes = []
            for order in (0, 1) if rng.random() < 0.5 else (1, 0):
                x, y = (ra, rb) if order == 0 else (rb, ra)
                v, raw = ask(args.host, args.judge, PROMPT.format(persona=PERSONA, user=A[t]["text"], a=x, b=y))
                if v is None:
                    unparsed += 1; votes.append(None); continue
                # map back to A/B
                votes.append("tie" if v == 0 else ("A" if (v == 1) == (order == 0) else "B"))
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
                         "reply_a": ra[:120], "reply_b": rb[:120]})
            print(f"[{label}] t{t:02d} {v1}/{v2} -> {verdict}", flush=True)
        n = len(turns)
        decided = wins["A"] + wins["B"]
        pref = "A" if wins["A"] > wins["B"] else ("B" if wins["B"] > wins["A"] else "tie")
        exp = expect.get(label)
        agree = (exp == pref) if exp else None
        summary.append({"pair": label, "A": pa.name, "B": pb.name, "turns": n, "wins_A": wins["A"], "wins_B": wins["B"],
                        "ties": wins["tie"], "flips": flips, "unparsed": unparsed, "flip_rate": round(flips / n, 2) if n else None,
                        "pref": pref, "expect": exp, "agree": agree, "s": round(time.perf_counter() - t0)})
        print(f"== {label}: A {wins['A']} / B {wins['B']} / tie {wins['tie']} / flip {flips} / unparsed {unparsed} -> {pref} (expect {exp})", flush=True)
    stamp = date.today().isoformat() + "_" + re.sub(r"[^a-z0-9]+", "-", args.judge.lower())
    with open(OUT / f"judge_pairs_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = [f"# Парный слепой судья ({args.judge}) · {stamp}", "",
          "| пара | A | B | ходов | A | B | ничья | перевороты | нераспознано | вердикт | ожидание ядра | совпало |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        md.append(f"| {s['pair']} | {s['A']} | {s['B']} | {s['turns']} | {s['wins_A']} | {s['wins_B']} | {s['ties']} | {s['flips']} ({s['flip_rate']}) | {s['unparsed']} | {s['pref']} | {s['expect'] or '—'} | {'' if s['agree'] is None else ('да' if s['agree'] else 'нет')} |")
    agreed = [s for s in summary if s["agree"] is not None]
    if agreed:
        md += ["", f"Совпадение с вердиктами ядра: {sum(s['agree'] for s in agreed)} / {len(agreed)}; средняя доля переворотов (позиционный сдвиг): {round(sum(s['flip_rate'] or 0 for s in summary) / len(summary), 2)}"]
    (OUT / f"judge_pairs_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
