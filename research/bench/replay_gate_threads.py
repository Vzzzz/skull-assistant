"""T-046 gate, 2026-10-31 (owner: «нити диалога обрывались»). Replay of the T-046 gate part 2 (A, music): every owner turn left silent / «угу» gets the duck's verdict on the
same window brain had and Gemma's reply as if the thread went on; answered turns keep their original reply.
Reads the bus log and dialog:history, writes nothing to Redis."""
import json, sys, time
from pathlib import Path
import redis, requests, yaml
sys.path.insert(0, str(Path.home() / "skull-assistant"))
from core.memory import SYSTEM_RULE
from core.router import POLICY, duck_bin_llamaserver

ROOT = Path.home() / "skull-assistant"
P = yaml.safe_load((ROOT / "services/brain/manifest.yaml").read_text())["params"]
SYSTEM = (ROOT / P["persona"]).read_text().strip() + "\n\n" + SYSTEM_RULE
HOST, T0, T1 = P["host"], time.mktime(time.strptime("2026-10-31 20:35:00", "%Y-%m-%d %H:%M:%S")), time.mktime(time.strptime("2026-10-31 20:43:30", "%Y-%m-%d %H:%M:%S"))
r = redis.Redis()
history = [json.loads(x) for x in r.lrange("dialog:history", 0, -1)]
base = [ex for ex in history if ex["ts"] < T0]

bus = [json.loads(l) for l in open(Path.home() / "skull-gate/gate2.jsonl")]
turns = []                       # (ts, text, speaker, pause, decision, why, reply)
for i, e in enumerate(bus):
    f = e["f"]
    if e["s"] != "events:transcript" or not (T0 <= e["t"] <= T1) or f.get("deferred"):
        continue
    amb = next((b["f"] for b in bus[i + 1:i + 6] if b["s"] == "events:ambient" and b["f"].get("text") == f["text"]), {})
    rep = next((json.loads(b["f"]["meta"]).get("reply") for b in bus[i + 1:i + 8]
                if b["s"] == "events:reply" and "meta" in b["f"] and b["t"] - e["t"] < 40), None)
    turns.append((e["t"], f["text"], f.get("speaker"), float(f.get("pause_s") or 2), amb.get("decision", "respond"), amb.get("why", ""), rep))

def messages(win):
    m = [{"role": "system", "content": SYSTEM}]
    for u, a in win:
        m += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return m

def gemma(win, text):
    body = {"messages": messages(win) + [{"role": "user", "content": text}], "max_tokens": POLICY["chat"]["max_tokens"],
            "temperature": P["temperature"], "chat_template_kwargs": {"enable_thinking": False}}
    return requests.post(f"{HOST}/v1/chat/completions", json=body, timeout=120).json()["choices"][0]["message"]["content"].strip()

win = [(ex["user"], ex["reply"]) for ex in base][-12:]
for ts, text, spk, pause, dec, why, rep in turns:
    t = time.strftime("%H:%M:%S", time.localtime(ts))
    dropped = dec in ("silent", "ack") and spk != "unknown"
    if not dropped:
        print(f"{t} Владелец: {text}\n         Костян: {rep}" if rep else f"{t} [{spk}, без ответа: {why}] {text}")
        if rep:
            win.append((text, rep))
        continue
    d = duck_bin_llamaserver(HOST, messages(win), text, pause)
    p = (d.get("probs") or {}).get("да", 0.0)
    verdict = "продолжать (≥ 0.7)" if p >= 0.7 else "«угу» (0.3–0.7)" if p >= 0.3 else "молчать (< 0.3)"
    g = gemma(win, text)
    print(f"{t} Владелец: {text}\n         [на гейте: {dec} — {why}; повтор фильтра: да {p:.2f} → {verdict}]\n         Gemma, если продолжить: {g}")
    win.append((text, g))
