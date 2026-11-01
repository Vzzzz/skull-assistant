"""T-051 measurement through the bus: 12 world questions as the owner's turns (trigger wakeword), the net off or on by
the brain manifest (the caller restarts skull-brain); per question: the lines in order, «Сейчас гляну.» and the first
fragment of the answer after the turn, the answer, whether it holds the expected keyword, the net results' titles.
Sink muted; dialog:history / dialog:vec restored after.

  .venv/bin/python research/bench/net_bus.py off|on    -> research/bench/results/net_<date>_<tag>.json
"""
import json, subprocess, sys, time
from pathlib import Path
import redis

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = [   # (text, keywords of a correct answer; empty = no fact to check, read by hand)
    ("А что ты знаешь о гравиметре Тюленева тысяча девятьсот восемьдесят седьмого года? Его вроде на кафедре у нас стоял.", []),  # t17
    ("Oh wait, what's the weather like on Mars today?", []),                                                                   # t24
    ("Какая высота Эльбруса?", ["5642", "5 642", "пять тысяч шестьсот сорок"]),
    ("Какой город столица Австралии?", ["Канберр"]),
    ("В каком году основан Санкт-Петербург?", ["1703", "тысяча семьсот третьем", "тысяча семьсот третий"]),
    ("Кто написал роман Мастер и Маргарита?", ["Булгаков"]),
    ("Сколько спутников у Марса и как они называются?", ["Фобос", "Деймос"]),
    ("При какой температуре плавится железо?", ["1538", "1 538", "тысяча пятьсот тридцать восемь", "1539"]),
    ("Кто открыл пенициллин?", ["Флеминг"]),
    ("Какая сегодня погода в Москве?", []),          # not for a reference book: «в сети не нашёл» expected
    ("Какой сегодня курс доллара?", []),
    ("Что сегодня главное в новостях?", []),
]
r = redis.Redis(socket_timeout=None)


def ask(text):
    last = (r.xrevrange("events:reply", count=1) or [[b"0"]])[0][0]
    t0 = time.time()
    r.xadd("events:transcript", {"text": text, "trigger": "wakeword", "speaker": "owner", "ts": f"{t0:.3f}"})   # brain copies ts into its replies
    lines, first_answer, look, out = [], None, None, None
    ts = f"{t0:.3f}"
    while time.time() - t0 < 120 and out is None:
        for _, items in r.xread({"events:reply": last}, block=1000) or []:
            for last, f in items:
                f = {k.decode(): v.decode() for k, v in f.items()}
                if f.get("ts") != ts:          # a reply to another turn (the room, a family voice): not ours
                    continue
                m = json.loads(f["meta"]) if "meta" in f else {}
                if f.get("text") and not m.get("net") in ("look", "empty") and first_answer is None and "meta" not in f:
                    first_answer = round(time.time() - t0, 2)
                if m.get("net") == "look":
                    look = round(time.time() - t0, 2)
                if m.get("net") in ("look", "empty"):
                    lines.append(m["reply"])
                if "turn" in m:
                    out = m
                    first_answer = first_answer or round(time.time() - t0, 2)
    return {"look_s": look, "first_answer_s": first_answer, "lines": lines, "reply": out and out.get("reply"),
            "net": out and out.get("net"), "label": out and (out.get("route") or {}).get("label"), "calls": out and out.get("llm_calls")}


def main():
    tag = sys.argv[1]
    saved = {k: r.dump(k) for k in ("dialog:history", "dialog:vec")}
    subprocess.run(["timeout", "5", "pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"])
    res = []
    try:
        for text, keys in QUESTIONS:
            a = ask(text)
            a.update(text=text, keys=keys, correct=(any(k.lower() in (a["reply"] or "").lower() for k in keys) if keys else None))
            res.append(a)
            print(json.dumps(a, ensure_ascii=False), flush=True)
            time.sleep(2)
    finally:
        subprocess.run(["timeout", "5", "pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"])
        for k, v in saved.items():
            r.delete(k)
            if v:
                r.restore(k, 0, v)
    out = ROOT / f"research/bench/results/net_{time.strftime('%Y-%m-%d')}_{tag}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
