"""Owner's phrases for the STT bake-off (T-040). Audio is not in git.

  record [--seconds 150]   every stretch of speech from audio:mic (cut by silence, 0.3 s pre-roll, >= 1 s) ->
                           research/stt/data/owner_pieces/pNN.wav; read the lines of data/owner_list.txt aloud,
                           in any order, with a pause >= 1.5 s
  align                    long pieces are also cut at their pauses (>= 0.3 s); each phrase of the list gets the
                           piece whose GigaAM v2 transcript is closest to it
                           (CER <= 0.5, a piece used once) -> data/owner/NN.wav; unmatched phrases are reported

Matching by text is only for cutting: the bake-off then scores every model on the same WAV against the list text.
"""
import argparse
import collections
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
PIECES, OWNER = HERE / "data/owner_pieces", HERE / "data/owner"


def record(seconds, start_rms, silence_s):
    import redis
    PIECES.mkdir(parents=True, exist_ok=True)
    for f in PIECES.glob("*.wav"):
        f.unlink()
    r = redis.Redis(socket_timeout=None)
    last, pre, cur, silent, k = "$", collections.deque(maxlen=10), None, 0, 0
    t_end = time.time() + seconds
    print(f"record: {seconds:.0f} s, read the phrases with pauses", flush=True)
    while time.time() < t_end:
        resp = r.xread({"audio:mic": last}, count=20, block=1000)
        if not resp:
            continue
        for msg_id, fields in resp[0][1]:
            last = msg_id
            f = np.frombuffer(fields[b"pcm"], dtype=np.int16)
            loud = np.sqrt(np.mean(f.astype(np.float32) ** 2)) >= start_rms
            if cur is None:
                pre.append(f)
                if loud:
                    cur, silent = list(pre), 0
                continue
            cur.append(f)
            silent = 0 if loud else silent + 1
            if silent * 0.03 >= silence_s:
                audio = np.concatenate(cur[: len(cur) - silent + 10])
                if len(audio) / 16000 >= 1.0:
                    sf.write(PIECES / f"p{k:02d}.wav", audio, 16000, subtype="PCM_16")
                    print(f"  p{k:02d}.wav  {len(audio) / 16000:.1f} s", flush=True)
                    k += 1
                cur, pre = None, collections.deque(maxlen=10)
    print(f"record: {k} pieces", flush=True)


def align():
    import sys
    sys.path.insert(0, str(HERE))
    import sherpa_onnx
    from bench_stt import STT, edits, norm
    d = STT / "sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19"
    rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=4)
    # long pieces may hold several phrases read with short pauses: also offer their parts cut at pauses >= 0.3 s
    for f in sorted(PIECES.glob("p??.wav")):
        a = sf.read(f, dtype="float32")[0]
        if len(a) / 16000 < 5:
            continue
        fr = a[: len(a) // 480 * 480].reshape(-1, 480)
        quiet = np.sqrt((fr ** 2).mean(axis=1)) * 32768 < 250
        cuts, run = [], 0
        for i, q in enumerate(quiet):
            run = run + 1 if q else 0
            if run == 10:
                cuts.append(i - 5)                       # middle of the pause
        bounds = [0] + cuts + [len(quiet)]
        for j in range(len(bounds) - 1):
            part = a[bounds[j] * 480: bounds[j + 1] * 480]
            if len(part) / 16000 >= 1.0:
                sf.write(PIECES / f"{f.stem}_{j}.wav", part, 16000, subtype="PCM_16")
    hyp = {}
    for f in sorted(PIECES.glob("*.wav")):
        st = rec.create_stream(); st.accept_waveform(16000, sf.read(f, dtype="float32")[0]); rec.decode_stream(st)
        hyp[f] = norm(st.result.text)
    OWNER.mkdir(parents=True, exist_ok=True)
    for f in OWNER.glob("*.wav"):
        f.unlink()
    used = set()
    for line in (HERE / "data/owner_list.txt").read_text(encoding="utf-8").splitlines():
        name, text = line.split("\t", 1)
        ref = norm(text).replace(" ", "")
        best = min(((edits(ref, h.replace(" ", "")) / len(ref), f) for f, h in hyp.items() if f not in used), default=None)
        if best and best[0] <= 0.5:
            used.add(best[1]); shutil.copy(best[1], OWNER / name)
            print(f"  {name} <- {best[1].name}  CER {best[0]:.2f}  «{hyp[best[1]]}»")
        else:
            print(f"  {name}: no piece (best CER {best[0]:.2f})" if best else f"  {name}: no piece")
    print("unused pieces:", [f"{f.name} «{h}»" for f, h in hyp.items() if f not in used])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rr = sub.add_parser("record")
    rr.add_argument("--seconds", type=float, default=150)
    rr.add_argument("--start-rms", type=float, default=200)
    rr.add_argument("--silence-s", type=float, default=0.9)
    sub.add_parser("align")
    a = ap.parse_args()
    record(a.seconds, a.start_rms, a.silence_s) if a.cmd == "record" else align()
