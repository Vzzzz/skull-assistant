"""Ambient gate through the bus without a microphone (T-038 stage 2, check (c)): transcripts with trigger=speech,
a given speaker and pause go to events:transcript, brain's decisions are read from events:ambient.

  .venv/bin/python research/eval/ambient_bus.py --set types     # 10 turns of each type, each from a reset (IDLE)
  .venv/bin/python research/eval/ambient_bus.py --set sequence  # one dialog: ENGAGED / BUSY transitions
Stop skull-tts or mute the sink: answers are spoken.

Replay of a live evening (T-038 stage 2b) — new rules on the same turns, without the microphone and without brain:
  ambient_bus.py --dump --since "2026-10-27 18:52" --until "2026-10-27 23:03" --out research/eval/data/ambient_2026-10-27.jsonl
      transcripts from the skull-stt journal (text, speaker, score, pause) + the decisions of that evening (events:ambient)
      + replies and their playback end (events:reply, dialog:history, «tts: quiet after» in the skull-tts journal);
      private (family voices, owner's speech): research/eval/data/ is not in git; --block requests tags every turn
      (T-042: 13 knowledge questions of 2026-10-28 17:33–17:38)
  ambient_bus.py --replay research/eval/data/ambient_2026-10-27.jsonl [--blocks requests,mutter,phone,podcast]
      core/ambient.py on every turn in order, the router and the duck pass on llama-server (skull-llm, :11437) with the
      history the new rules would have built (replies of that evening where they exist); time from the journal.
      Writes research/eval/results/ambient_replay_<date>[_<tag>].json. Brain may stay up: the replay does not use the bus.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import redis

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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
# stage 2b (core/ambient.py): the window opens on the name / a question, an uncertain voice counts inside it
SEQUENCE = [("Костян, привет", "owner", 1.5, "respond"), ("А сколько сейчас времени", "owner", 0.7, "respond"),   # engaged
            ("Который час в Токио", "uncertain", 0.7, "respond"),                                           # uncertain inside
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


# blocks of the live test on 2026-10-27 (research/bench/results/ambient_2026-10-27.md) by wall clock; what is a mistake
BLOCKS_1027 = [("household", "18:52:00", "22:17:53"), ("requests", "22:17:53", "22:21:30"), ("mutter", "22:21:30", "22:28:15"),
               ("phone", "22:28:15", "22:29:12"), ("podcast", "22:29:12", "23:03:00")]
STT_LINE = re.compile(r"stt: \[speech (\S+) (\S*)(?: win \S+(?: @\S+)?)? ?\S*ms pause ([\d.]+)s[^\]]*\] (.*?)(?:  \(full: .*\))?$")


def journal(unit, since, until):
    out = subprocess.run(["journalctl", "--user", "-u", unit, "--since", since, "--until", until, "-o", "json", "--no-pager"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        d = json.loads(line)
        msg = d.get("MESSAGE")
        if isinstance(msg, str):
            yield int(d["__REALTIME_TIMESTAMP"]) / 1e6, msg


def dump(since, until, out, block=None):
    stt = [(t, m) for t, m in journal("skull-stt", since, until) if STT_LINE.match(m)]
    speech = [(t, float(mm.group(1))) for t, m in journal("skull-wakeword", since, until)
              if (mm := re.match(r"SPEECH ([\d.]+)s(?!.*dropped)", m))]   # VAD segment lengths (trigger speech)
    quiet = {}                                     # reply_id -> when the player went quiet after it
    for t, m in journal("skull-tts", since, until):
        mm = re.match(r"tts: quiet after (\w+) #", m)
        if mm:
            quiet[mm.group(1)] = t
    lo, hi = stt[0][0] - 5 if stt else 0, stt[-1][0] + 60 if stt else 0
    amb = [{k.decode(): v.decode() for k, v in f.items()} for _, f in r.xrange("events:ambient", int(lo * 1000), int(hi * 1000))]
    replies = {}                                   # utterance ts -> reply_id (brain copies the transcript ts into the reply)
    for _, f in r.xrange("events:reply", int(lo * 1000), int(hi * 1000)):
        replies.setdefault(f[b"ts"].decode(), f[b"reply_id"].decode())
    hist = {f"{json.loads(x)['ts']:.3f}": json.loads(x)["reply"] for x in r.lrange("dialog:history", 0, -1)}
    rows, used = [], set()
    for t, m in stt:
        spk, score, pause, text = STT_LINE.match(m).groups()
        a = next((x for i, x in enumerate(amb) if i not in used and x["text"] == text.strip() and abs(float(x["ts"]) - t) < 30), None)
        if a is None:
            continue
        used.add(amb.index(a))
        clock = time.strftime("%H:%M:%S", time.localtime(t))
        blk = block or next(b for b, s, e in BLOCKS_1027 if s <= clock < e)
        seg = next((d for ts, d in reversed(speech) if t - 5 <= ts <= t + 0.5), None)
        rid = replies.get(a["ts"])
        rows.append({"clock": clock, "t": float(a["ts"]), "block": blk, "seg_s": seg, "text": text.strip(), "speaker": spk,
                     "spk_score": score, "pause_s": float(pause), "was": a["decision"], "was_why": a["why"],
                     "was_state": a["state"], "reply": hist.get(a["ts"]), "reply_end": quiet.get(rid) if rid else None})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    print(f"{len(rows)} turns -> {out}; by block:", {b: sum(x["block"] == b for x in rows) for b in {x["block"] for x in rows}})


def expected(row):
    """What the evening's script wanted: requests -> respond (the skull's own echo is not a request), the rest -> no content."""
    if row["block"] == "requests" and row["speaker"] != "other":
        return {"respond"}
    return {"silent", "ack"}


def answer(host, messages, text, route, temperature):
    """The reply brain would give (T-042): router guards, policy prefix and max_tokens as in services/brain."""
    from core.router import POLICY, decide, policy_prefix
    label, _ = decide(route, text) if route else ("chat", None)
    prefix = policy_prefix(label, text)
    body = {"messages": messages + [{"role": "user", "content": (prefix + "\n" + text) if prefix else text}],
            "max_tokens": POLICY.get(label, POLICY["chat"])["max_tokens"], "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}
    import requests
    out = requests.post(f"{host}/v1/chat/completions", json=body, timeout=120).json()["choices"][0]["message"]["content"]
    return label, "full" if prefix and prefix == POLICY.get("world", {}).get("prefix") else "soft" if prefix else "-", out.strip()


def replay(path, blocks, engaged_s, yes_p, host, tag="", gen=False):
    import requests
    import yaml
    from core.ambient import Ambient
    from core.memory import SYSTEM_RULE
    from core.router import duck_bin_llamaserver, route_llamaserver
    persona = (ROOT / "persona/system_prompt.md").read_text(encoding="utf-8").strip()
    bp = yaml.safe_load((ROOT / "services/brain/manifest.yaml").read_text(encoding="utf-8"))["params"]
    if bp.get("memory"):
        persona += "\n\n" + SYSTEM_RULE           # brain's SYSTEM (memory on)
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]
    rows = [x for x in rows if x["block"] in blocks]
    amb, hist, opened, out = Ambient(engaged_s=engaged_s, engaged_yes_p=yes_p), [], None, []
    med_reply = 6.0                                # playback length when the evening has no reply for a turn
    for x in rows:
        now = x["t"]
        engaged = opened is not None and now < opened + engaged_s
        messages = [{"role": "system", "content": persona}] + [m for u, a in hist[-8:] for m in
                                                               ({"role": "user", "content": u}, {"role": "assistant", "content": a})]
        try:
            route = route_llamaserver(host, messages, x["text"])
        except requests.RequestException:
            route = {}
        p_aside = (route.get("probs") or {}).get("aside", 0.0)
        duck_calls = []

        def duck(text, pause):
            dk = duck_bin_llamaserver(host, messages, text, pause)
            duck_calls.append(dk)
            return dk
        decision, why, opens = amb.decide(x["text"], x["speaker"], x["pause_s"], p_aside, now, engaged, duck, seg_s=x.get("seg_s"))
        reply = None
        if decision == "respond" and gen:
            x["gen_label"], x["gen_prefix"], reply = answer(host, messages, x["text"], route, bp.get("temperature", 0.7))
            x["gen_reply"] = reply
        if decision == "respond":
            hist.append((x["text"], reply or x["reply"] or "…"))
            if opens:   # the window counts from the end of playback (the evening's, or a typical length)
                opened = x["reply_end"] if x["reply_end"] else now + med_reply
        elif decision == "ack":
            hist.append((x["text"], "Угу."))
        ok = decision in expected(x)
        out.append({**x, "new": decision, "new_why": why, "engaged": engaged, "p_aside": round(p_aside, 3), "ok": ok})
        if x["block"] not in ("household", "podcast") or not ok:
            print(f"{x['clock']} {x['block']:<9} {'OK ' if ok else 'BAD'} {x['was']:<7}-> {decision:<7} "
                  f"{'E' if engaged else ' '} {why[:58]:<58} | {x['speaker']} {x['spk_score']} | {x['text'][:60]}", flush=True)
            if reply is not None:
                print(f"{'':>20}[{x['gen_label']} prefix {x['gen_prefix']}] {reply[:150]!r}", flush=True)
    summary = {}
    for b in blocks:
        bb = [x for x in out if x["block"] == b]
        summary[b] = {"n": len(bb), "bad": sum(not x["ok"] for x in bb), "was_bad": sum(x["was"] not in expected(x) for x in bb),
                      "respond": sum(x["new"] == "respond" for x in bb), "ack": sum(x["new"] == "ack" for x in bb)}
    res = ROOT / f"research/eval/results/ambient_replay_{time.strftime('%Y-%m-%d')}{'_' + tag if tag else ''}.json"
    res.write_text(json.dumps({"source": str(path), "engaged_s": engaged_s, "engaged_yes_p": yes_p, "summary": summary,
                               "turns": [{k: v for k, v in x.items() if k not in ("reply",)} for x in out
                                         if x["block"] not in ("household", "podcast")]},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"-> {res}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["types", "sequence"], default="types")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--since"), ap.add_argument("--until"), ap.add_argument("--out")
    ap.add_argument("--block", help="--dump: one block name for every turn (otherwise by the clock of 2026-10-27)")
    ap.add_argument("--replay")
    ap.add_argument("--blocks", default="household,requests,mutter,phone,podcast")
    ap.add_argument("--engaged-s", type=float, default=20)
    ap.add_argument("--yes-p", type=float, default=0.3)
    ap.add_argument("--host", default="http://127.0.0.1:11437")
    ap.add_argument("--tag", default="", help="suffix of the result file")
    ap.add_argument("--answer", action="store_true", help="--replay: generate the reply for every respond, as brain would")
    a = ap.parse_args()
    if a.dump:
        return dump(a.since, a.until, a.out, a.block)
    if a.replay:
        return replay(a.replay, a.blocks.split(","), a.engaged_s, a.yes_p, a.host, a.tag, a.answer)
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
