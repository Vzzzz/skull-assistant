"""T-055: does the brain model follow the policy prefixes? 10 owner turns through the bus (trigger wakeword): world
(the full prefix with a perception word, the soft one otherwise), calc (the «(вычислено: …)» note), net (the excerpts
note, net:policy=always for the run). Prints the label, the prefix kind / note, the reply; compliance is judged by hand.
Sink is not used (skull-tts must be down); dialog:history / dialog:vec restored after.

  .venv/bin/python research/eval/prefix_bus.py --tag <model>   -> research/eval/results/prefix_bus_<tag>_<date>.json
"""
import argparse, json, time
from pathlib import Path
import redis

ROOT = Path(__file__).resolve().parents[2]
TURNS = [("world", "Что сейчас за окном, видишь?"), ("world", "Какая погода сегодня на улице?"),
         ("world", "Почему небо голубое?"), ("world", "Кто такой Тюленев, который гравиметр сделал?"),
         ("calc", "Сколько будет 47 умножить на 21?"), ("calc", "Пятнадцать процентов от 2 400 — это сколько?"),
         ("calc", "Выйду в 8:40, ехать час тридцать пять, когда буду?"),
         ("net", "Какая высота Эльбруса?"), ("net", "Кто открыл пенициллин?"), ("net", "Какой город столица Австралии?")]
r = redis.Redis(socket_timeout=None)


def ask(text):
    last = (r.xrevrange("events:reply", count=1) or [[b"0"]])[0][0]
    ts = f"{time.time():.3f}"
    r.xadd("events:transcript", {"text": text, "trigger": "wakeword", "speaker": "owner", "ts": ts})
    t0 = time.time()
    while time.time() - t0 < 120:
        for _, items in r.xread({"events:reply": last}, block=1000) or []:
            for last, f in items:
                f = {k.decode(): v.decode() for k, v in f.items()}
                if f.get("ts") != ts or "meta" not in f:
                    continue
                m = json.loads(f["meta"])
                if "turn" in m:
                    return m
    return {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
    saved = {k: r.dump(k) for k in ("dialog:history", "dialog:vec")}
    old = r.get("net:policy")
    out = []
    try:
        r.xadd("events:transcript", {"text": "/reset"}); time.sleep(1)
        for kind, text in TURNS:
            r.set("net:policy", "always" if kind == "net" else "never")
            m = ask(text)
            row = {"kind": kind, "text": text, "label": (m.get("route") or {}).get("label"), "calc": m.get("calc"),
                   "net": m.get("net"), "reply": m.get("reply"), "completion_tokens": m.get("completion_tokens")}
            out.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        r.set("net:policy", old) if old else r.delete("net:policy")
        for k, v in saved.items():
            r.delete(k)
            if v:
                r.restore(k, 0, v)
    p = ROOT / f"research/eval/results/prefix_bus_{a.tag}_{time.strftime('%Y-%m-%d')}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
