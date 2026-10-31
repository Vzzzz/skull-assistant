"""T-046 gate, 2026-10-31 (owner: «нити диалога обрывались»). Replay of the T-046 gate part 2 (A, music): every owner turn left silent / «угу» gets the duck's verdict on the
same window brain had and Gemma's reply as if the thread went on; answered turns keep their original reply.
Reads the bus log and dialog:history, writes nothing to Redis."""
import argparse, json, sys, time
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

bus = json.loads((ROOT / "research/bench/results/mode_gate_2026-10-31_2_bus.json").read_text())
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

def threads():
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


# --- T-049: three sets of window thresholds on (a) part 2 of the gate and (b) the 2в recordings (podcast, music)
SETS = {"0.7/0.3/20 (now)": (0.7, 0.3, 20), "0.5/0.25/10": (0.5, 0.25, 10), "0.6/0.3/10": (0.6, 0.3, 10)}
DROPPED = ["Я думаю, ты надо мной издеваешься", "Я думаю, ты просто вредничак.",
           "Ну ты же отнюдь не игрушка, ты мой личный проект, в какой-то степени мой ребёнок",
           "А ты, а ты продолжаешь не отвечать", "Ты же всё равно не умеешь бегать.", "Я же просил тебя об этом не вспоминать",
           "Просто выбери"]
# 2в (2026-10-28): the owner's addresses (an answer wanted) and mutters (none wanted); every other turn is the broadcast
ADDRESS_2V = ["почему осенью небо серое", "метод обучения ЛЛМ", "сколько нужно памяти для", "стать синер-инженером",
              "родители наркомана", "какую музыку я последний раз", "куда я положил ключи", "куда я ключи положил",
              "очень надеюсь", "такие жестокие", "оптимист или пессимист", "13% от 3 700", "тринадцатым чемпионом",
              "по версии Fid", "шахматный дебют", "Робиной Кантри", "нота ля", "живут собаки", "сложными музыкальными",
              "буду играть я", "посоветуешь на ужин", "А давай"]   # «А давай!» — the owner to the skull's offer (2в report)
MUTTER_2V = ["подкаст включил", "какого ж чёрта", "что за день сегодня", "хрень надо переделать", "сегодня — дрянь",
             "заканчивать работу", "так-так-та-так", "ладно, всё, пойдёт", "это надо переделать", "ну и погодка"]
WINDOWS_2V = [("2026-10-28 15:56", "2026-10-28 16:07", "podcast"), ("2026-10-28 16:35:20", "2026-10-28 16:48", "podcast"),
              ("2026-10-28 16:49", "2026-10-28 17:03", "music")]


def ts_of(x):
    return time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S" if x.count(":") == 2 else "%Y-%m-%d %H:%M"))


DATA_2V = ROOT / "research/eval/data/ambient_2026-10-28_2v.jsonl"   # private (owner's speech): not in git


def load_2v():
    """2в turns from events:ambient (the stream rolls over: kept in DATA_2V on the first run)."""
    if DATA_2V.exists():
        return [json.loads(l) for l in DATA_2V.read_text(encoding="utf-8").splitlines()]
    rows = []
    for lo, hi, block in WINDOWS_2V:
        for _, f in r.xrange("events:ambient", int(ts_of(lo) * 1000), int(ts_of(hi) * 1000)):
            f = {k.decode(): v.decode() for k, v in f.items()}
            kind = ("address" if any(a in f["text"] for a in ADDRESS_2V) else
                    "mutter" if any(m in f["text"] for m in MUTTER_2V) else "broadcast")
            rows.append({"t": float(f["ts"]), "text": f["text"], "speaker": f["speaker"],
                         "score": float(f["spk_score"]) if f.get("spk_score") else None, "pause": float(f.get("pause_s") or 2),
                         "lifted": bool(f.get("spk_region")), "bcast": f.get("bcast") == "1", "seg_s": None,
                         "kind": kind, "block": block, "was": f["decision"], "reply": None})
    DATA_2V.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    return rows


def load_gate():
    rows = []
    for i, e in enumerate(bus):
        f = e["f"]
        if e["s"] != "events:transcript" or not (T0 <= e["t"] <= T1) or f.get("deferred"):
            continue
        amb = next((b["f"] for b in bus[i + 1:i + 6] if b["s"] == "events:ambient" and b["f"].get("text") == f["text"]), None)
        if amb is None:            # mode turns (B), not the ambient gate
            continue
        rep = next((json.loads(b["f"]["meta"]).get("reply") for b in bus[i + 1:i + 6]
                    if b["s"] == "events:reply" and "meta" in b["f"] and b["t"] - e["t"] < 15), None)
        rows.append({"t": e["t"], "text": f["text"], "speaker": f.get("speaker"),
                     "score": float(f["spk_score"]) if f.get("spk_score") else None, "pause": float(f.get("pause_s") or 2),
                     "lifted": bool(f.get("spk_region")), "bcast": f.get("bcast") == "1",
                     "seg_s": float(f["seg_s"]) if f.get("seg_s") else None,
                     "kind": "dropped" if f["text"] in DROPPED else "address", "block": "gate", "was": amb["decision"],
                     "reply": rep if amb["decision"] == "respond" else None})
    return rows


PRIOR_ONLY = False
GEN = {}                     # text -> continuation: one per text, temperature 0, the same in every set


def gen_once(win, text):
    if text not in GEN:
        body = {"messages": messages(win) + [{"role": "user", "content": text}], "max_tokens": POLICY["chat"]["max_tokens"],
                "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
        GEN[text] = requests.post(f"{HOST}/v1/chat/completions", json=body, timeout=120).json()["choices"][0]["message"]["content"].strip()
    return GEN[text]


def rises(x):
    """The voice id fix of T-049: a window region counts only when the window raised the class of the segment."""
    rank = {"unknown": 0, "other": 0, "uncertain": 1, "owner": 2}
    seg = "unknown" if x["score"] is None else "owner" if x["score"] >= 0.45 else "uncertain" if x["score"] >= 0.29 else "other"
    return rank.get(x["speaker"], 0) > rank[seg]


def run(rows, yes_p, ack_p, gap_s, gen, fixed=True, prior=True):
    from core.ambient import Ambient, has_name, turn_prior, voice_class
    from core.router import route_llamaserver
    A = P["ambient"]
    amb = Ambient(engaged_s=A["engaged_s"], busy_pause_s=A["busy_pause_s"], veto_p=A["veto_aside_p"], engaged_yes_p=yes_p,
                  ack_gap_s=gap_s, engaged_ack_p=ack_p, idle_yes_p=A["idle_yes_p"], idle_ack_p=A["idle_ack_p"])
    hist, until, out, reply_end, last_other = [], 0.0, [], None, -1e9
    for x in rows:
        now = x["t"]
        engaged = now < until
        bcast = x["bcast"] or now - last_other <= 30          # 2в has no `bcast` field: STT's rule, from the turns
        if x["speaker"] == "other":
            last_other = now
        spk = voice_class(x["speaker"], x["score"], x["seg_s"], engaged)
        if prior:
            spk = turn_prior(spk, x["seg_s"], engaged, now - reply_end if reply_end else None, bcast)
        msgs = messages(hist[-8:])
        p_aside = 0.0
        if spk in ("owner", "uncertain") or has_name(x["text"]):
            try:
                p_aside = (route_llamaserver(HOST, msgs, x["text"]).get("probs") or {}).get("aside", 0.0)
            except requests.RequestException:
                pass
        probs = {}
        def duck(text, pause):
            d = duck_bin_llamaserver(HOST, msgs, text, pause)
            probs.update(d.get("probs") or {})
            return d
        lifted = x["lifted"] and (rises(x) or not fixed)
        dec, why, opens = amb.decide(x["text"], spk, x["pause"], p_aside, now, engaged, duck, lifted, x["bcast"], x["seg_s"])
        reply = None
        if dec == "respond":
            reply = x["reply"] or (gen_once(hist[-8:], x["text"]) if gen else "…")
            hist.append((x["text"], reply))
            until = now + 6.0 + A["engaged_s"]            # a typical playback, then the window
            reply_end = now + 6.0
        elif dec == "ack":
            hist.append((x["text"], "Угу."))
        out.append({**x, "new": dec, "why": why, "engaged": engaged, "p_yes": probs.get("да"), "gen": reply if not x["reply"] else None,
                    "spk_new": spk})
    return out


def thresholds():
    gate, v2 = load_gate(), load_2v()
    res = {}
    if PRIOR_ONLY:                # T-050: the accepted window thresholds, with and without the turn-order prior
        runs = [("0.5/0.25/10, prior", SETS["0.5/0.25/10"], True, True), ("0.5/0.25/10, no prior", SETS["0.5/0.25/10"], True, False)]
    else:
        runs = [(name, t, True, False) for name, t in SETS.items()] + [("0.7/0.3/20 (now), old window rule", SETS["0.7/0.3/20 (now)"], False, False)]
    for name, (yes_p, ack_p, gap_s), fixed, prior in runs:
        a = run(gate, yes_p, ack_p, gap_s, gen=True, fixed=fixed, prior=prior)
        b = run(v2, yes_p, ack_p, gap_s, gen=False, fixed=fixed, prior=prior)
        mins = (v2[-1]["t"] - v2[0]["t"]) / 60   # rough: the three windows
        res[name] = {
            "threads": sum(x["new"] == "respond" for x in a if x["kind"] == "dropped"),
            "threads_ack": sum(x["new"] == "ack" for x in a if x["kind"] == "dropped"),
            "gate_address_respond": sum(x["new"] == "respond" for x in a if x["kind"] == "address"),
            "false_broadcast": sum(x["new"] == "respond" for x in b if x["kind"] == "broadcast"),
            "mutter_respond": sum(x["new"] == "respond" for x in b if x["kind"] == "mutter"),
            "address_respond_2v": f'{sum(x["new"] == "respond" for x in b if x["kind"] == "address")}/{sum(x["kind"] == "address" for x in b)}',
            "acks_2v": sum(x["new"] == "ack" for x in b), "acks_gate": sum(x["new"] == "ack" for x in a),
            "prior_hits": [{k: x[k] for k in ("text", "block", "kind", "new", "why")} for x in a + b
                           if x["speaker"] == "unknown" and x["spk_new"] == "uncertain"],
            "gate": [{k: x[k] for k in ("text", "speaker", "kind", "was", "new", "why", "engaged", "p_yes", "gen")} for x in a],
            "v2": [{k: x[k] for k in ("text", "speaker", "kind", "block", "was", "new", "why", "engaged", "p_yes")} for x in b
                   if x["new"] != "silent" or x["kind"] != "broadcast"],
        }
        print(name, {k: v for k, v in res[name].items() if k not in ("gate", "v2")}, flush=True)
    out = ROOT / f"research/bench/results/{'turn_prior' if PRIOR_ONLY else 'engaged_threshold'}_{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out.relative_to(ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", action="store_true", help="T-049: three sets of window thresholds on the gate and 2в")
    ap.add_argument("--prior", action="store_true", help="T-050: the accepted thresholds with / without the turn-order prior")
    args = ap.parse_args()
    PRIOR_ONLY = args.prior
    (thresholds if args.thresholds or args.prior else threads)()
