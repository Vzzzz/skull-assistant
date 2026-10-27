"""Dump a reply as the TTS service voices it (T-003 tail): Piper per sentence, without and with the sox timbre
(the same FX as services/tts/speaker.py), concatenated as the player plays it. WAVs are not in git.

  .venv/bin/python research/tts/fx_dump.py [--turn 1] [--out research/tts/results/t003_fx]
The reply is taken from dialog:history (meta.reply form, i.e. before the speech filter) and cut like brain does.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import redis
import soundfile as sf
from piper import PiperVoice

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.speech import for_speech  # noqa: E402

FX = ["pitch", "-300", "reverb", "30", "50", "100"]
SENTENCE_END = re.compile(r'[.!?…]+["»)]?\s+|\n+')          # as services/brain/responder.py
LIST_MARK = re.compile(r"(?:\d+|[a-zа-я])[.)]")


def sentences(text):
    out, buf, pos = [], text + "\n", 0
    while (m := SENTENCE_END.search(buf, pos)):
        if LIST_MARK.fullmatch(buf[:m.end()].strip()):
            pos = m.end(); continue
        s, buf, pos = buf[:m.end()].strip(), buf[m.end():], 0
        if s:
            out.append(for_speech(s))
    return [s for s in out if s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "research/tts/results/t003_fx"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hist = [json.loads(x) for x in redis.Redis().lrange("dialog:history", 0, -1)]
    reply = next(h["reply"] for h in hist if h["turn"] == a.turn)
    voice = PiperVoice.load(str(ROOT / "models/tts/ru_RU-dmitri-medium.onnx"))
    rate = voice.config.sample_rate
    dry, wet = [], []
    for s in sentences(reply):
        d = np.concatenate([c.audio_int16_array for c in voice.synthesize(s)])
        w = np.frombuffer(subprocess.run(["sox", "-t", "raw", "-r", str(rate), "-e", "signed", "-b", "16", "-c", "1", "-",
                                          "-t", "raw", "-r", str(rate), "-e", "signed", "-b", "16", "-c", "1", "-", *FX],
                                         input=d.tobytes(), capture_output=True, check=True).stdout, dtype=np.int16)
        dry.append(d); wet.append(w)
    for name, parts in (("answer1_dry.wav", dry), ("answer1_fx.wav", wet), ("answer1_first3_fx.wav", wet[:3])):
        sf.write(out / name, np.concatenate(parts), rate, subtype="PCM_16")
    print(f"{len(dry)} sentences, {sum(len(x) for x in dry) / rate:.1f} s dry / {sum(len(x) for x in wet) / rate:.1f} s fx -> {out}")
    print("first three:", " | ".join(sentences(reply)[:3]))


if __name__ == "__main__":
    main()
