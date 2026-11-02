# -*- coding: utf-8 -*-
"""Token usage of the project's Claude Code sessions: sums `message.usage` over transcript jsonl files in the given
directories (~/.claude/projects/<project>), deduped by message id; prints totals, per model and per day.
  python research/bench/claude_usage.py ~/.claude/projects/<project-dir> [...]
Snapshot 2026-11-02 (first commit 2026-10-07; research sessions from 2026-09-24): researcher (Windows) + executor (WSL,
from 2026-10-24) — output 3.64M tokens, cache writes 31.2M, cache reads 1.14B, uncached input 35k, ~3000 assistant turns."""
import glob, io, json, os, sys
from collections import defaultdict

dirs = sys.argv[1:]
seen = {}
per_model = defaultdict(lambda: defaultdict(int))
per_day = defaultdict(lambda: defaultdict(int))
files = 0
for d in dirs:
    for p in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
        files += 1
        with io.open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if '"usage"' not in line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                m = o.get("message") or {}
                u = m.get("usage")
                if not u or m.get("role") != "assistant":
                    continue
                mid = m.get("id") or o.get("uuid")
                key = (p, mid)
                ts = (o.get("timestamp") or "")[:10]
                model = m.get("model") or "?"
                seen[key] = (model, ts, u)
tot = defaultdict(int)
for (p, mid), (model, ts, u) in seen.items():
    for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        v = int(u.get(k) or 0)
        per_model[model][k] += v; per_day[ts][k] += v; tot[k] += v
    per_model[model]["msgs"] += 1; per_day[ts]["msgs"] += 1
print("files", files, "assistant messages", len(seen))
print("TOTAL", dict(tot))
for model, u in sorted(per_model.items(), key=lambda x: -sum(v for k, v in x[1].items() if k != "msgs")):
    print("MODEL", model, dict(u))
for day in sorted(per_day):
    u = per_day[day]
    print("DAY", day, "msgs", u["msgs"], "in", u["input_tokens"], "out", u["output_tokens"], "cache_w", u["cache_creation_input_tokens"], "cache_r", u["cache_read_input_tokens"])
