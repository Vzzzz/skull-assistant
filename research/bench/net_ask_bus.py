"""T-054 measurement through the bus, policy `ask`: a world question as the owner's turn -> «Глянуть в сети?» -> the
owner's «Да» / «Нет» as a second turn -> the answer (by the excerpts or offline). Latency question -> the ask line, and
the owner's answer -> the first fragment of the reply. Sink muted; dialog:history / dialog:vec restored.

  .venv/bin/python research/bench/net_ask_bus.py      -> research/bench/results/net_ask_<date>.json
"""
import json, subprocess, time
from pathlib import Path
import redis

ROOT = Path(__file__).resolve().parents[2]
CASES = [("Какой город столица Австралии?", "Да"), ("Кто открыл пенициллин?", "Давай, глянь"),
         ("Какая высота Эльбруса?", "Да"), ("В каком году основан Санкт-Петербург?", "Ага"),
         ("Кто написал роман Мастер и Маргарита?", "Посмотри"),
         ("Сколько спутников у Марса?", "Нет"), ("При какой температуре плавится железо?", "Не надо"),
         ("Какая сегодня погода в Москве?", "Нет"), ("Кто изобрёл телефон?", "Сам скажи"), ("Какая самая длинная река в мире?", "Нет")]
r = redis.Redis(socket_timeout=None)


def replies_after(last, until_turn=True, timeout=60, ts=None):
    """-> (last id, [(t, text, meta)]) until a reply with a turn (the model's answer) or a matching `ts` net line."""
    out, t0 = [], time.time()
    while time.time() - t0 < timeout:
        for _, items in r.xread({"events:reply": last}, block=1000) or []:
            for last, f in items:
                f = {k.decode(): v.decode() for k, v in f.items()}
                if ts and f.get("ts") != ts:
                    continue
                m = json.loads(f["meta"]) if "meta" in f else {}
                out.append((round(time.time(), 2), f.get("text", ""), m))
                if (until_turn and "turn" in m) or (not until_turn and m.get("net") == "ask"):
                    return last, out
    return last, out


def main():
    saved = {k: r.dump(k) for k in ("dialog:history", "dialog:vec")}
    subprocess.run(["timeout", "5", "pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"])
    res = []
    try:
        for q, ans in CASES:
            last = (r.xrevrange("events:reply", count=1) or [[b"0"]])[0][0]
            t0 = time.time(); ts = f"{t0:.3f}"
            r.xadd("events:transcript", {"text": q, "trigger": "wakeword", "speaker": "owner", "ts": ts})
            last, got = replies_after(last, until_turn=False, ts=ts)
            ask_s = round(got[-1][0] - t0, 2) if got and got[-1][2].get("net") == "ask" else None
            time.sleep(1.0)
            t1 = time.time()
            r.xadd("events:transcript", {"text": ans, "trigger": "wakeword", "speaker": "owner", "ts": f"{t1:.3f}"})
            last, got2 = replies_after(last)
            first = next((t for t, txt, m in got2 if txt and m.get("net") not in ("look", "empty")), None)
            final = got2[-1][2] if got2 else {}
            row = {"q": q, "answer": ans, "ask_line_s": ask_s, "lines": [txt for _, txt, m in got2 if m.get("net")],
                   "first_fragment_s": round(first - t1, 2) if first else None, "net": final.get("net"),
                   "reply": final.get("reply"), "calls": final.get("llm_calls")}
            res.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            time.sleep(2)
    finally:
        subprocess.run(["timeout", "5", "pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"])
        for k, v in saved.items():
            r.delete(k)
            if v:
                r.restore(k, 0, v)
    out = ROOT / f"research/bench/results/net_ask_{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
