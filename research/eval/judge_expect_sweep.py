"""Re-read a --expect-score CSV of judge_pairs.py (T-043) with other tie thresholds, and compare the agreement with the
core's verdicts against constant judges («always A» / «always tie»): with few decided turns the agreement may be prior.

  .venv/bin/python research/eval/judge_expect_sweep.py research/eval/results/judge_pairs_<date>_<judge>_expect.csv
"""
import csv
import sys
from collections import defaultdict

EXPECT = {"e4b_base_vs_thinkall_s42": "A", "e4b_base_vs_thinkall_s7": "A", "e2b_base_vs_thinkall_s42": "B",
          "e2b_base_vs_thinkall_s7": "tie", "qat_vs_q4km_noimx": "A", "qat_vs_iq4xs_imx": "A", "base2k_vs_mem_v1": "A",
          "base2k_vs_mem_router_v5": "B", "e4b_vs_e2b_v3": "A"}   # the core's verdicts (D-021…D-026), as on 2026-10-23


def verdict(s1, s2, tie):
    if abs(s1) < tie and abs(s2) < tie:
        return "tie"
    if (s1 > 0) != (s2 > 0) and abs(s1 - s2) > tie and min(abs(s1), abs(s2)) >= tie:
        return "flip"
    m = (s1 + s2) / 2
    return "tie" if abs(m) < tie else ("B" if m > 0 else "A")


rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["verdict"] != "unparsed"]
for tie in (0.1, 0.05, 0.02, 0.0):
    wins, flips = defaultdict(lambda: {"A": 0, "B": 0, "tie": 0}), defaultdict(int)
    for r in rows:
        v = verdict(float(r["vote_order1"]), float(r["vote_order2"]), tie)
        if v == "flip":
            flips[r["pair"]] += 1
        else:
            wins[r["pair"]][v] += 1
    agree, decided = 0, 0
    for p, e in EXPECT.items():
        w = wins[p]
        pref = "A" if w["A"] > w["B"] else "B" if w["B"] > w["A"] else "tie"
        agree += pref == e
        decided += w["A"] + w["B"]
    n = sum(1 for r in rows)
    print(f"tie {tie:<4}: agree {agree}/9, flips {sum(flips.values()) / n:.2f}, decided turns {decided}/{n}")
print(f"constant judges: always A {sum(e == 'A' for e in EXPECT.values())}/9, always tie {sum(e == 'tie' for e in EXPECT.values())}/9")
