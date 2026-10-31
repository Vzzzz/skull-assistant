# -*- coding: utf-8 -*-
"""historical: stage 1 text-only automaton; superseded by core/ambient.py + tests/unit/test_ambient.py
(kept as the negative result of D-029: addressee by text alone; Windows paths left as they were).

Offline simulation over T-038 stage-1 JSON logs: does a sticky conversational state (FSM-lite) beat per-turn labels?
Uses the router probabilities already in the logs; `block` change = proxy for a pause (stage 2 will have real time)."""
import json, sys, glob, os
NAMES = ("костян", "череп")
files = sorted(glob.glob(r"C:\Users\valer\git\skull-assistant\research\eval\results\ctx_probe_ambient_*_rt-apply_8192.json"))


def pr(decisions, log):
    tp = fp = fn = 0
    for t, d in zip(log, decisions):
        e = t.get("addr_expect")
        if isinstance(e, list):
            continue
        if d and e == "me": tp += 1
        elif d and e == "aside": fp += 1
        elif not d and e == "me": fn += 1
    return tp / max(1, tp + fp), tp / max(1, tp + fn), tp, fp, fn


def has_name(t): return any(n in t["text"].lower() for n in NAMES)


def sim(log, variant):
    out = []; state = "IDLE"; prev_block = None
    for t in log:
        p = (t.get("route") or {}).get("probs") or {}
        aside = p.get("aside", 0.0)
        name = has_name(t)
        if variant == "current":               # what the executor measured: aside only if p>=0.6 and no name
            out.append(not (aside >= 0.6 and not name)); continue
        if variant == "name-only":
            out.append(name); continue
        # FSM-lite: IDLE (not talking to me lately) / ENGAGED (dialogue is on) / BUSY (phone, guest, broadcast)
        if prev_block is not None and t["block"] != prev_block and state == "BUSY":
            state = "IDLE"                        # pause between scenes ends a busy episode
        prev_block = t["block"]
        if name:
            respond = True; state = "ENGAGED"
        elif state == "BUSY":
            respond = False
        elif aside >= 0.9:
            respond = False; state = "BUSY"
        elif state == "ENGAGED":
            respond = aside < 0.6
            if not respond: state = "IDLE"
        else:                                     # IDLE: answer only a confident, non-aside turn
            top = max(p.values()) if p else 0.0
            respond = aside < 0.1 and top >= (0.9 if variant == "fsm" else 0.5)
            if respond: state = "ENGAGED"
        out.append(respond)
    return out


for f in files:
    d = json.load(open(f, encoding="utf-8")); log = d["log"]
    print(os.path.basename(f))
    for v in ("current", "name-only", "fsm", "fsm-loose"):
        prec, rec, tp, fp, fn = pr(sim(log, v), log)
        print(f"  {v:10} precision {prec:.2f} recall {rec:.2f}  tp {tp} fp {fp} fn {fn}")
    dec = sim(log, "fsm")
    errs = [(t["turn"], t.get("kind"), "FP" if r else "FN") for t, r in zip(log, dec)
            if not isinstance(t.get("addr_expect"), list) and ((r and t["addr_expect"] == "aside") or (not r and t["addr_expect"] == "me"))]
    print("  fsm errors:", errs)
