"""Ambient gate through the bus without a microphone (T-038 stage 2, check (c)): transcripts with trigger=speech,
a given speaker and pause go to events:transcript, brain's decisions are read from events:ambient.

  .venv/bin/python research/eval/ambient_bus.py --set types     # 10 turns of each type, each from a reset (IDLE)
  .venv/bin/python research/eval/ambient_bus.py --set sequence  # one dialog: ENGAGED / BUSY transitions
Stop skull-tts or mute the sink: answers are spoken.
"""
import argparse
import json
import time

import redis

r = redis.Redis(socket_timeout=None)
DIRECT = ["Сколько будет сорок один на двадцать семь?", "Где я оставил ключи, не видел?", "Напомни мне завтра позвонить маме.",
          "Что ты думаешь про мой новый проект", "Как работает холодильник", "Слушай, какая завтра погода",
          "Почему небо голубое", "Подскажи, сколько стоит видеокарта", "Ты помнишь, как зовут моего кота",
          "Расскажи что-нибудь интересное про космос"]
MUTTER = ["Так, где у меня тут зарядка лежала…", "Ну и бардак. Ладно, потом.", "Почему оно не собирается… опять этот линкер.",
          "Так. Undefined reference. Понятно, ничего не понятно.", "Сорок восемь на три… сто сорок четыре.",
          "Ключи, кошелёк, зарядка. Три вещи.", "Двести граммов муки, два яйца, стакан молока.", "Ой, горячо. Чёрт.",
          "Где телефон… А, вот.", "Бублик, слезь со стола."]
NAME = ["Костян, ты тут?", "Костян, как дела", "Слушай, Костян, сколько времени", "Костян, напомни про встречу",
        "Костян, ты опять молчишь", "Костян, что скажешь", "Эй, Костян", "Костян, посчитай пятнадцать процентов от тысячи",
        "Костян, как зовут кота", "Костян, доброе утро"]
BROADCAST = ["В центре города сегодня перекрыли движение из-за ремонта теплосети.",
             "Курс доллара на завтра установлен на уровне восьмидесяти двух рублей.",
             "Завтра в регионе ожидается облачная погода с прояснениями.", "Вы когда-нибудь задумывались, сколько весит облако?",
             "И у нас на линии Константин из Новосибирска.", "Оставайтесь с нами, после рекламы — главное.",
             "Сегодня мы поговорим о том, как правильно выбирать ноутбук.", "Как вы думаете, кто выиграет в этом сезоне?",
             "Спасибо, что слушаете наш подкаст.", "А теперь новости спорта."]
VETO = ["Алло, да, слушаю.", "Да, мам, привет. Да, всё нормально.", "Алло? Плохо слышно, перезвоню.",
        "Привет, Дим, да, я на созвоне.", "Так, коллеги, слышно меня?", "Алло, это доставка?",
        "Да, я дома, приезжай.", "Ага, через пять минут выйду.", "Да-да, я тебя слушаю.", "Алло, добрый день."]
# (turns, speaker, pause, expected decisions)
TYPES = {"direct": (DIRECT, "owner", 2.0, {"respond"}), "mutter": (MUTTER, "owner", 0.9, {"silent", "ack"}),
         "name_uncertain": (NAME, "uncertain", 1.5, {"respond"}), "uncertain": (DIRECT, "uncertain", 2.0, {"silent"}),
         "other": (BROADCAST, "other", 2.0, {"silent"}), "veto": (VETO, "owner", 1.5, {"silent"})}
SEQUENCE = [("Костян, привет", "owner", 1.5, "respond"), ("Ну и погода сегодня.", "owner", 2.0, "respond"),   # engaged
            ("Алло, да, слушаю.", "owner", 1.0, "silent"), ("Да, буду через час.", "owner", 1.0, "silent"),      # veto, busy
            ("Костян, ты тут?", "owner", 1.5, "respond")]                                                       # name clears busy


def send(text, speaker, pause):
    last = (r.xrevrange("events:ambient", count=1) or [[b"0-0"]])[0][0]
    r.xadd("events:transcript", {"text": text, "ts": f"{time.time():.3f}", "trigger": "speech", "speaker": speaker,
                                 "spk_score": "", "pause_s": f"{pause:.2f}"})
    resp = r.xread({"events:ambient": last}, count=1, block=120000)
    f = {k.decode(): v.decode() for k, v in resp[0][1][0][1].items()}
    if f["decision"] == "respond":    # let the answer finish before the next turn
        lr = (r.xrevrange("events:reply", count=1) or [[b"0-0"]])[0][0]
        while True:
            rr = r.xread({"events:reply": lr}, count=20, block=120000)
            done = False
            for mid, ff in rr[0][1]:
                lr = mid
                done = done or ff.get(b"final") == b"1"
            if done:
                break
    return f


def reset():
    r.xadd("events:transcript", {"text": "/reset"})
    time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["types", "sequence"], default="types")
    a = ap.parse_args()
    rows = []
    if a.set == "types":
        for kind, (turns, spk, pause, ok) in TYPES.items():
            hits = 0
            for t in turns:
                reset()
                f = send(t, spk, pause)
                hit = f["decision"] in ok
                hits += hit
                rows.append({"kind": kind, "text": t, **f, "ok": hit})
                print(f"{kind:<14} {'OK ' if hit else 'BAD'} {f['decision']:<8} {f['why'][:60]:<60} | {t}", flush=True)
            print(f"== {kind}: {hits}/{len(turns)}", flush=True)
    else:
        reset()
        for t, spk, pause, exp in SEQUENCE:
            f = send(t, spk, pause)
            rows.append({"text": t, **f, "expect": exp, "ok": f["decision"] == exp})
            print(f"{'OK ' if f['decision'] == exp else 'BAD'} {f['decision']:<8} state={f['state']:<8} {f['why'][:50]:<50} | {t}", flush=True)
    print(json.dumps({"ok": sum(x["ok"] for x in rows), "n": len(rows)}))


if __name__ == "__main__":
    main()
