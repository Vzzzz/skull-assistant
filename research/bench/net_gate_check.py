"""T-051: would brain go to the net on the turns of the T-046 gate (part 2, mode_gate_2026-10-31_2_bus.json) with the
net on? Same order as services/brain/responder.py: mode phrases first (an entry / exit phrase never reaches the net),
then the policy — router label `world` in A, a question in B, never a question about the owner (net.personal). The
router runs on llama-server (mode A) with the gate's history.  .venv/bin/python research/bench/net_gate_check.py"""
import json, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "services/brain"))
from core.memory import SYSTEM_RULE
from core.modes import find_phrase
from core.numbers_ru import normalize
from core.router import decide, is_question, route_llamaserver
from net import personal

P = yaml.safe_load((ROOT / "services/brain/manifest.yaml").read_text())["params"]
SYSTEM = (ROOT / P["persona"]).read_text().strip() + "\n\n" + SYSTEM_RULE
bus = json.loads((ROOT / "research/bench/results/mode_gate_2026-10-31_2_bus.json").read_text())
hist, rows, mode = [], [], "A"
for i, e in enumerate(bus):
    if e["s"] == "events:mode" and e["f"].get("phase") == "done":
        mode = e["f"].get("to", mode)
    if e["s"] != "events:transcript" or e["f"].get("speaker") not in ("owner", "uncertain"):
        continue
    t = normalize(e["f"]["text"])
    rep = next((json.loads(b["f"]["meta"]).get("reply") for b in bus[i + 1:i + 6] if b["s"] == "events:reply" and "meta" in b["f"]), None)
    if find_phrase(t, P["modes"]["enter" if mode == "A" else "exit"]) and not e["f"].get("deferred"):
        rows.append((mode, "mode phrase", False, t))
    elif mode == "A":
        msgs = [{"role": "system", "content": SYSTEM}] + [m for u, a in hist[-6:] for m in ({"role": "user", "content": u}, {"role": "assistant", "content": a})]
        label, _ = decide(route_llamaserver(P["host"], msgs, t), t)
        rows.append((mode, label, label == "world" and not personal(t), t))
    else:
        rows.append((mode, "B question" if is_question(t) else "B", is_question(t) and not personal(t), t))
    if rep:
        hist.append((t, rep))
print(f"{len(rows)} owner turns; to the net: {sum(r[2] for r in rows)}")
for m, l, go, t in rows:
    print(f"  {'NET ' if go else '    '}{m} {l:<11} | {t[:80]}")
