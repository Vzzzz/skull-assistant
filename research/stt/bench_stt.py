"""Russian STT bake-off on CPU without sound (T-040): WAV + reference text -> WER / CER / RTF / RSS per backend.

  prep-cv                       first 200 clips of Common Voice ru test.tsv -> research/stt/data/cv/*.wav (16 kHz)
                                and the fixed list research/stt/cv_list.txt (wav name <TAB> sentence)
  one --backend B --threads N --set cv|owner     one backend in this process: JSON line with the metrics
  run [--sets cv owner]         every backend x threads 1/4 in its own process (RSS: before / loaded / after),
                                writes research/stt/results/stt_<date>.csv, stt_hyps_<date>.csv and stt_<date>.md

Normalisation for WER: lower case, ё -> е, punctuation dropped, numbers as they are. Streaming backends (T-one, Vosk)
get 300-ms chunks; "first word" = audio seconds fed when the first non-empty partial appears, "final" = wall
seconds from input_finished() to the final text.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "research/stt"
DATA = HERE / "data"
OUT = HERE / "results"
STT = ROOT / "models/stt"
CV_DIR = Path.home() / "bitladder/data/cv/cv-corpus-27.0-2026-09-11/ru"
LISTS = {"cv": HERE / "cv_list.txt", "owner": DATA / "owner_list.txt"}   # owner list holds the owner's texts: not in git
RATE = 16000
BACKENDS = ["fw-base", "fw-small", "gigaam-v2", "gigaam-v3", "t-one", "vosk-small"]
STREAMING = {"t-one", "vosk-small"}


def mem():
    d = dict(l.split(":", 1) for l in open("/proc/self/status") if l.startswith(("VmRSS", "VmHWM")))
    return {k: int(v.split()[0]) // 1024 for k, v in d.items()}


def norm(text):
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]", " ", t).replace("_", " ")
    return " ".join(t.split())


def edits(ref, hyp):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def load_list(name):
    rows = []
    for line in LISTS[name].read_text(encoding="utf-8").splitlines():
        if line.strip():
            wav, text = line.split("\t", 1)
            rows.append((DATA / name / wav, text))
    return rows


def prep_cv(n=200):
    import av
    out = DATA / "cv"
    out.mkdir(parents=True, exist_ok=True)
    lines = (CV_DIR / "test.tsv").read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    ip, isent = head.index("path"), head.index("sentence")
    keep = []
    for line in lines[1:n + 1]:
        f = line.split("\t")
        mp3, sent = f[ip], f[isent]
        wav = mp3.replace(".mp3", ".wav")
        if not (out / wav).exists():
            with av.open(str(CV_DIR / "clips" / mp3)) as c:
                rs = av.AudioResampler(format="s16", layout="mono", rate=RATE)
                pcm = [fr.to_ndarray().reshape(-1) for frame in c.decode(audio=0) for fr in rs.resample(frame)]
                pcm += [fr.to_ndarray().reshape(-1) for fr in rs.resample(None)]
            sf.write(out / wav, np.concatenate(pcm).astype(np.int16), RATE, subtype="PCM_16")
        keep.append(f"{wav}\t{sent}")
    LISTS["cv"].write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"prep-cv: {len(keep)} clips -> {out}, list {LISTS['cv'].relative_to(ROOT)}")


def make(backend, threads):
    """Returns transcribe(audio float32) -> (text, extra dict)."""
    import sherpa_onnx
    if backend.startswith("fw-"):
        from faster_whisper import WhisperModel
        m = WhisperModel(backend[3:], device="cpu", compute_type="int8", cpu_threads=threads)

        def tr(a):   # production settings (services/stt/transcriber.py)
            segs, _ = m.transcribe(a, language="ru", vad_filter=True, condition_on_previous_text=False)
            return " ".join(s.text.strip() for s in segs), {}
        return tr
    if backend.startswith("gigaam"):
        d = {"gigaam-v2": STT / "sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19",
             "gigaam-v3": STT / "sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16"}[backend]
        rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"),
                                                         num_threads=threads)

        def tr(a):
            st = rec.create_stream()
            st.accept_waveform(RATE, a)
            rec.decode_stream(st)
            return st.result.text, {}
        return tr
    if backend == "t-one":
        d = STT / "sherpa-onnx-streaming-t-one-russian-2025-09-08"
        rec = sherpa_onnx.OnlineRecognizer.from_t_one_ctc(tokens=str(d / "tokens.txt"), model=str(d / "model.onnx"),
                                                          num_threads=threads)
    else:
        d = STT / "sherpa-onnx-streaming-zipformer-small-ru-vosk-2025-08-16"
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(tokens=str(d / "tokens.txt"), encoder=str(d / "encoder.onnx"),
                                                           decoder=str(d / "decoder.onnx"), joiner=str(d / "joiner.onnx"),
                                                           num_threads=threads)
    chunk = int(0.3 * RATE)

    def tr(a):
        st = rec.create_stream()
        first = None
        for i in range(0, len(a), chunk):
            st.accept_waveform(RATE, a[i:i + chunk])
            while rec.is_ready(st):
                rec.decode_stream(st)
            if first is None and rec.get_result(st).strip():
                first = min(i + chunk, len(a)) / RATE
        st.accept_waveform(RATE, np.zeros(int(1.5 * RATE), dtype=np.float32))   # tail: 0.6 s still cut the last word (T-one, Vosk)
        st.input_finished()
        t0 = time.perf_counter()
        while rec.is_ready(st):
            rec.decode_stream(st)
        return rec.get_result(st), {"first_word_s": first, "final_s": time.perf_counter() - t0}
    return tr


def one(backend, threads, name):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    m0 = mem()
    tr = make(backend, threads)
    m1 = mem()
    rows = load_list(name)
    hyps, w_e = [], [0, 0, 0, 0]   # word edits, ref words, char edits, ref chars
    audio_s = proc_s = 0.0
    firsts, finals = [], []
    for wav, ref in rows:
        a, sr = sf.read(wav, dtype="float32")
        assert sr == RATE, wav
        t0 = time.perf_counter()
        hyp, extra = tr(a)
        proc_s += time.perf_counter() - t0
        audio_s += len(a) / RATE
        r, h = norm(ref), norm(hyp)
        w_e[0] += edits(r.split(), h.split()); w_e[1] += len(r.split())
        w_e[2] += edits(r.replace(" ", ""), h.replace(" ", "")); w_e[3] += len(r.replace(" ", ""))
        if extra.get("first_word_s") is not None:
            firsts.append(extra["first_word_s"])
        if "final_s" in extra:
            finals.append(extra["final_s"])
        hyps.append([wav.name, ref, hyp])
    m2 = mem()
    print(json.dumps({"backend": backend, "threads": threads, "set": name, "clips": len(rows), "audio_s": round(audio_s, 1),
                      "wer": round(w_e[0] / w_e[1], 4), "cer": round(w_e[2] / w_e[3], 4), "rtf": round(proc_s / audio_s, 4),
                      "rss_before": m0["VmRSS"], "rss_loaded": m1["VmRSS"], "rss_after": m2["VmRSS"], "hwm": m2["VmHWM"],
                      "first_word_s": round(float(np.mean(firsts)), 2) if firsts else None,
                      "final_s": round(float(np.mean(finals)), 3) if finals else None,
                      "hyps": hyps if threads == 1 else None}, ensure_ascii=False), flush=True)


def run(sets, backends):
    OUT.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    res = []
    for name in sets:
        for b in backends:
            for t in (1, 4):
                p = subprocess.run([sys.executable, __file__, "one", "--backend", b, "--threads", str(t), "--set", name],
                                   capture_output=True, text=True)
                line = [l for l in p.stdout.splitlines() if l.startswith("{")]
                if not line:
                    print(f"{name} {b} t{t}: FAILED\n{p.stderr[-800:]}", flush=True)
                    continue
                r = json.loads(line[-1])
                res.append(r)
                print(f"{name:<5} {b:<11} t{t}  WER {r['wer']:.3f}  CER {r['cer']:.3f}  RTF {r['rtf']:.3f}  "
                      f"RSS {r['rss_before']}/{r['rss_loaded']}/{r['rss_after']} (HWM {r['hwm']})"
                      + (f"  first {r['first_word_s']} s, final {r['final_s']} s" if r["first_word_s"] is not None else ""), flush=True)
    keys = ["set", "backend", "threads", "clips", "audio_s", "wer", "cer", "rtf", "rss_before", "rss_loaded", "rss_after",
            "hwm", "first_word_s", "final_s"]
    with open(OUT / f"stt_{today}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys); w.writerows([[r[k] for k in keys] for r in res])
    with open(OUT / f"stt_hyps_{today}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "backend", "wav", "ref", "hyp"])
        for r in res:
            for h in r["hyps"] or []:
                # owner clips: the owner's own phrases are kept; only research data, no audio
                w.writerow([r["set"], r["backend"], *h])
    print(f"-> {(OUT / f'stt_{today}.csv').relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep-cv")
    o = sub.add_parser("one")
    o.add_argument("--backend", choices=BACKENDS, required=True)
    o.add_argument("--threads", type=int, default=1)
    o.add_argument("--set", choices=list(LISTS), default="cv")
    rr = sub.add_parser("run")
    rr.add_argument("--sets", nargs="+", default=["cv", "owner"])
    rr.add_argument("--backends", nargs="+", default=BACKENDS)
    a = ap.parse_args()
    if a.cmd == "prep-cv":
        prep_cv()
    elif a.cmd == "one":
        one(a.backend, a.threads, a.set)
    else:
        run(a.sets, a.backends)


if __name__ == "__main__":
    main()
