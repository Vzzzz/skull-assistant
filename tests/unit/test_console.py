"""deploy/stand/skull_console.py (T-058): every record of the T-046 gate dump formats without an exception, the
latency chain is counted per owner turn, the record / replay dump round-trips (an unterminated one too)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("skull_console", ROOT / "deploy/stand/skull_console.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
DUMP = ROOT / "research/bench/results/mode_gate_2026-10-31_2_bus.json"


def test_gate_dump_formats_and_counts_latency():
    for demo in (False, True):
        con = sc.Console(demo=demo)
        n = 0
        for e in json.loads(DUMP.read_text(encoding="utf-8")):
            n += len(con.feed(e["t"], e["s"], e["f"]))
        assert n > 50
        assert len(con.done) >= 5                      # turns with a latency line
        assert con.state["mode"] == "A"                # the gate ended back in A
        con.header(host="stand")


def test_latency_chain_with_tts():
    con = sc.Console(expect_tts=True)
    ts = "1000.000"
    con.feed(999.7, "events:wakeword", {"word": "speech", "score": "1.2", "end_ts": "999.70"})
    con.feed(1000.2, "events:transcript", {"text": "Костян, ты тут?", "speaker": "owner", "spk_score": "0.51", "ts": ts})
    con.feed(1000.25, "events:ambient", {"text": "Костян, ты тут?", "decision": "respond", "why": "name", "ts": ts})
    con.feed(1000.9, "events:reply", {"text": "Тут.", "reply_id": "r1", "seq": "0", "final": "0", "ts": ts})
    con.feed(1001.0, "events:reply", {"text": "", "reply_id": "r1", "final": "1", "ts": ts,
                                     "meta": json.dumps({"turn": 5, "route": {"label": "chat", "probs": {"chat": 0.8}}})})
    con.feed(1001.3, "events:tts", {"reply_id": "r1", "phase": "first_audio", "t": "1001.30", "ts": ts})
    out = con.feed(1002.5, "events:tts", {"reply_id": "r1", "phase": "end", "t": "1002.50", "ts": ts})
    line = out[-1]
    for part in ("→STT 0.50", "→gate 0.05", "→1st 0.65", "→sound 0.40", "speech 1.2 s", "total to sound 1.60 s"):
        assert part in line
    assert len(con.done) == 1


def test_record_and_replay_roundtrip(tmp_path):
    p = tmp_path / "bus.json"
    rec = sc.Recorder(p)
    rec.write({"t": 1.0, "s": "events:net", "f": {"phase": "begin"}})
    rec.write({"t": 2.0, "s": "events:net", "f": {"phase": "done", "n": "3"}})
    shown = []
    sc.replay(sc.Console(), p, 0, None, shown.extend)      # killed run: no closing bracket yet
    rec.close()
    assert len(json.loads(p.read_text())) == 2 and len(shown) == 2
    shown.clear()
    sc.replay(sc.Console(), p, 0, None, shown.extend)
    assert len(shown) == 2
