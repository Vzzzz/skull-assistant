"""Build a Russian in-domain calibration text for llama-imatrix (Q-047).

The teacher (gemma4 26B-A4B, persona prompt v3) writes short everyday dialogs between Valeriy and the skull on
topics that do NOT appear in the CTX-1 eval dialog; the output doubles as a seed of synthetic user lines for T-020.

Usage (stand, system ollama running):
    .venv/bin/python research/quant/make_calib.py [--n 60] [--model gemma4:26b-a4b-it-mtp-q4_K_M] [--out research/quant/calib_ru.txt]
"""
import argparse
import json
import random
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
PERSONA = (ROOT / "persona/system_prompt.md").read_text(encoding="utf-8").strip()
EVAL = ROOT / "research/eval/ctx_probe_dialog.jsonl"
# anchors and themes of the eval dialog: the calibration text must not touch them (leak into imatrix statistics)
EVAL_WORDS = {"бублик", "дим", "домофон", "4712", "кинз", "батарейк", "весов", "гравиметр", "тюленев", "замок", "замк",
              "марс", "сверл", "дрел", "сериал", "рюкзак", "куртк"}
TOPICS = [
    "сломался чайник", "пришёл счёт за свет больше обычного", "дождь третий день", "забыл пароль от роутера",
    "сосед завёл попугая", "надо выбрать подарок маме", "болит спина от стула", "кончился кофе", "посылка опаздывает",
    "решил начать бегать по утрам", "принтер не печатает", "в холодильнике только горчица", "позвонили из банка",
    "пропал интернет", "купил растение и не знаю, как поливать", "зимняя резина ещё в кладовке", "утренняя пробка",
    "выбор между двумя ноутбуками", "соседка просит помочь с полкой", "отпуск через две недели, билетов нет",
    "перегорела лампочка в коридоре", "хочу научиться готовить плов", "давно не звонил родителям",
    "пришло письмо из налоговой", "кот соседа орёт по ночам", "ремонт в подъезде", "новая серия любимой игры",
    "забыл зонт на работе", "в квартире холодно, отопление не дали", "старый телефон тормозит",
]
SYSTEM_GEN = (PERSONA + "\n\nСейчас ты пишешь сценку: короткий диалог из 4–6 реплик между Валерием и тобой на заданную тему. "
              "Формат строго: строки «Валерий: …» и «Костян: …», без ремарок, без заголовков. Реплики Костяна — в твоём характере, короткие.")


def gen(host, model, topic, seed):
    body = {"model": model, "stream": False, "think": False, "keep_alive": "10m",
            "messages": [{"role": "system", "content": SYSTEM_GEN},
                         {"role": "user", "content": f"Тема: {topic}. Напиши сценку."}],
            "options": {"temperature": 0.8, "seed": seed, "num_predict": 400, "num_ctx": 4096}}
    r = requests.post(f"{host}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def leaks(text):
    t = text.lower().replace("ё", "е")
    return [w for w in EVAL_WORDS if w in t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--model", default="gemma4:26b-a4b-it-mtp-q4_K_M")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--out", default=str(ROOT / "research/quant/calib_ru.txt"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    eval_lines = {json.loads(l)["text"].lower() for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()}
    out, dropped = [], 0
    for i in range(args.n):
        topic = TOPICS[i % len(TOPICS)]
        scene = gen(args.host, args.model, topic, args.seed * 1000 + i)
        scene = re.sub(r"\*[^*]*\*", "", scene).strip()                      # strip stage directions if any
        if leaks(scene) or any(l in scene.lower() for l in eval_lines):
            dropped += 1
            print(f"[{i:02d}] {topic}: dropped (eval leak: {leaks(scene)})", flush=True)
            continue
        out.append(f"# {topic}\n{scene}\n")
        print(f"[{i:02d}] {topic}: {len(scene)} chars", flush=True)
    text = PERSONA + "\n\n" + "\n".join(out)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"\n{len(out)} scenes kept, {dropped} dropped, {len(text)} chars -> {args.out}")


if __name__ == "__main__":
    main()
