"""Baseline for the Piper TTS path: real-time factor, time to first audio, RSS, sox cost.

Usage (inside the project venv):
    .venv/bin/python research/tts/piper_rt.py [lines.txt] [voice.onnx]
Writes CSV to research/tts/results/piper_rt_<date>.csv and prints a summary.
"""
import csv
import io
import resource
import subprocess
import sys
import time
import wave
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "research/eval/tts_lines.txt"
VOICE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models/tts/ru_RU-dmitri-medium.onnx"
OUT_DIR = ROOT / "research/tts/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"piper_rt_{date.today().isoformat()}.csv"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    from piper import PiperVoice

    rss0 = rss_mb()
    t0 = time.perf_counter()
    voice = PiperVoice.load(str(VOICE))
    load_s = time.perf_counter() - t0
    rss_loaded = rss_mb()
    rate = voice.config.sample_rate

    lines = [l.strip() for l in LINES.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for i, text in enumerate(lines, 1):
        # streaming API: time to first chunk + total
        t0 = time.perf_counter()
        first = None
        pcm = bytearray()
        for chunk in voice.synthesize(text):
            if first is None:
                first = time.perf_counter() - t0
            pcm += chunk.audio_int16_bytes
        synth_s = time.perf_counter() - t0
        audio_s = len(pcm) / 2 / rate
        rt = synth_s / audio_s if audio_s else float("nan")

        # sox chain cost as used in services/tts/speaker.py
        wav_in = OUT_DIR / "_tmp_in.wav"
        wav_out = OUT_DIR / "_tmp_out.wav"
        with wave.open(str(wav_in), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(bytes(pcm))
        t0 = time.perf_counter()
        subprocess.run(["sox", str(wav_in), str(wav_out), "pitch", "-300", "reverb", "30", "50", "100"],
                       check=False, capture_output=True)
        sox_s = time.perf_counter() - t0

        rows.append(dict(i=i, chars=len(text), audio_s=round(audio_s, 2), synth_s=round(synth_s, 3),
                         rt=round(rt, 3), first_audio_s=round(first or 0, 3), sox_s=round(sox_s, 3)))
        print(f"{i:2d} chars={len(text):3d} audio={audio_s:5.2f}s synth={synth_s:5.3f}s RT={rt:5.3f} "
              f"first={first or 0:5.3f}s sox={sox_s:5.3f}s")

    for f in (OUT_DIR / "_tmp_in.wav", OUT_DIR / "_tmp_out.wav"):
        f.unlink(missing_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    n = len(rows)
    print("\n== summary ==")
    print(f"voice={VOICE.name} rate={rate} load={load_s:.2f}s rss_base={rss0:.0f}MB rss_loaded={rss_loaded:.0f}MB rss_peak={rss_mb():.0f}MB")
    print(f"RT mean={sum(r['rt'] for r in rows)/n:.3f} max={max(r['rt'] for r in rows):.3f}; "
          f"first_audio mean={sum(r['first_audio_s'] for r in rows)/n:.3f}s max={max(r['first_audio_s'] for r in rows):.3f}s; "
          f"sox mean={sum(r['sox_s'] for r in rows)/n:.3f}s")
    print(f"csv: {OUT}")


if __name__ == "__main__":
    main()
