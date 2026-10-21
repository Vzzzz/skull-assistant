"""Voice FX presets for the skull's TTS output (TTS_TRACK: "character" post-processing experiment, 2026-10-22).

Applies distortion / radio / channel-vocoder / hollow-cavity presets to WAV files (24 kHz mono from Qwen3-TTS or
Piper). numpy + scipy only. Each preset is a function sample-array -> sample-array at the same rate.

Usage:
    python research/tts/fx.py <in_dir> <out_dir> [--presets radio,vocoder,hollow,distortion,robot] [--files 1,7,10]
Writes <out_dir>/<preset>/<name>.wav (peak-normalised to -1 dBFS).
"""
import argparse
import os
import wave

import numpy as np
from scipy import signal

SR = 24000


def read_wav(p):
    with wave.open(p) as w:
        sr = w.getframerate(); n = w.getnframes(); ch = w.getnchannels()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(p, x, sr=SR):
    x = np.clip(x, -1, 1)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def normalize(x, peak_db=-1.0):
    m = np.max(np.abs(x)) or 1.0
    return x / m * (10 ** (peak_db / 20))


def butter(x, kind, freq, order=4, sr=SR):
    sos = signal.butter(order, freq, btype=kind, fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def soft_clip(x, drive=4.0):
    return np.tanh(x * drive) / np.tanh(drive)


def comb_reverb(x, sr=SR, combs=(29.7, 37.1, 41.1, 43.7), allpasses=(5.0, 1.7), decay=0.55, wet=0.25):
    """Schroeder reverb: parallel feedback combs (ms) + series allpasses (ms)."""
    def comb(sig, ms, g):
        d = int(sr * ms / 1000); y = np.zeros_like(sig)
        for i in range(len(sig)):
            y[i] = sig[i] + (g * y[i - d] if i >= d else 0.0)
        return y

    def allpass(sig, ms, g=0.7):
        d = int(sr * ms / 1000); y = np.zeros_like(sig)
        for i in range(len(sig)):
            xd = sig[i - d] if i >= d else 0.0
            yd = y[i - d] if i >= d else 0.0
            y[i] = -g * sig[i] + xd + g * yd
        return y

    r = sum(comb(x, ms, decay) for ms in combs) / len(combs)
    for ms in allpasses:
        r = allpass(r, ms)
    return (1 - wet) * x + wet * r


def channel_vocoder(x, sr=SR, f0=95.0, bands=24, fmin=100.0, fmax=7000.0, dry=0.15, noise=0.08, nperseg=1024):
    """Classic channel vocoder: band envelopes of the voice (modulator) imposed on a sawtooth+noise carrier."""
    t = np.arange(len(x)) / sr
    # slight vibrato so the robot is not a dead tone
    f_inst = f0 * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    carrier = signal.sawtooth(phase) * 0.6 + noise * np.random.default_rng(0).standard_normal(len(x))
    hop = nperseg // 4
    f, _, M = signal.stft(x, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
    _, _, C = signal.stft(carrier, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
    edges = np.geomspace(fmin, fmax, bands + 1)
    out = np.zeros_like(C)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi)
        if not sel.any():
            continue
        env_m = np.sqrt(np.mean(np.abs(M[sel]) ** 2, axis=0))
        env_c = np.sqrt(np.mean(np.abs(C[sel]) ** 2, axis=0)) + 1e-6
        out[sel] = C[sel] * (env_m / env_c)[None, :]
    _, y = signal.istft(out, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
    y = y[: len(x)]
    return (1 - dry) * y + dry * x


PRESETS = {
    # walkie-talkie / old radio: band-limited, overdriven, a little hiss
    "radio": lambda x: normalize(soft_clip(butter(butter(x, "high", 300), "low", 3400), 5.0)
                                 + 0.004 * np.random.default_rng(1).standard_normal(len(x))),
    # plain distortion: pre-emphasis, soft clip, tame the top
    "distortion": lambda x: normalize(butter(soft_clip(butter(x, "high", 150), 7.0), "low", 6000)),
    # channel vocoder robot (Daft-Punk-ish), 15 % dry voice for intelligibility
    "vocoder": lambda x: normalize(channel_vocoder(x)),
    # vocoder + radio band-limit + drive: the "skull on a shelf" radio robot
    "robot": lambda x: normalize(soft_clip(butter(butter(channel_vocoder(x, f0=110.0, dry=0.25), "high", 250), "low", 4500), 3.0)),
    # hollow plastic cavity: short comb resonance + small-room reverb + gentle low cut
    "hollow": lambda x: normalize(comb_reverb(butter(x, "high", 120), decay=0.5, wet=0.3)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir"); ap.add_argument("out_dir")
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--files", default="", help="comma list of basenames without .wav; default all")
    a = ap.parse_args()
    names = [f"{n}.wav" for n in a.files.split(",")] if a.files else sorted(f for f in os.listdir(a.in_dir) if f.endswith(".wav"))
    for preset in a.presets.split(","):
        fx = PRESETS[preset]
        for name in names:
            x, sr = read_wav(os.path.join(a.in_dir, name))
            y = fx(x)
            write_wav(os.path.join(a.out_dir, preset, name), y, sr)
            print(f"{preset}/{name}: {len(y)/sr:.2f}s")


if __name__ == "__main__":
    main()
