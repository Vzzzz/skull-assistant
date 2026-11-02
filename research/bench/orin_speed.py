"""T-056 step 6: speed of the mode-A models on the device, against the running units (mode A resident, as deployed).
  .venv/bin/python research/bench/orin_speed.py [--reps 3] [--tag orin]
LLM — the prod llama-server of skull-llm (manifest host, prod args): cold prefill / TTFT on a fresh prompt of ~1k / 3k / 6k
tokens (cache_prompt off), decode tok/s on 128 forced tokens, warm TTFT with the 3k prefix reused (slot cache, P-033);
STT — GigaAM v3 CTC (sherpa-onnx, manifest num_threads) on the model's test wav, 1x and 3x concatenated: RTF;
voice id — CAM++ embedding of 1.5 s and 4 s; embedder — /v1/embeddings of one exchange (skull-embed).
Prints a summary and writes research/bench/results/orin_speed_<date>[_tag].json. Temperatures: tegrastats in parallel.
"""
import argparse
import json
import statistics
import sys
import time
import wave
from datetime import date
from pathlib import Path

import numpy as np
import requests
import sherpa_onnx
import yaml

ROOT = Path(__file__).resolve().parents[2]
BRAIN = yaml.safe_load((ROOT / "services/brain/manifest.yaml").read_text(encoding="utf-8"))["params"]
STT = yaml.safe_load((ROOT / "services/stt/manifest.yaml").read_text(encoding="utf-8"))["params"]
LLM = BRAIN["host"]
EMB = BRAIN["memory"]["embed_host"]
CALIB = (ROOT / "research/quant/calib_ru.txt").read_text(encoding="utf-8")


def templated(user, system=""):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    p = requests.post(f"{LLM}/apply-template", json={"messages": msgs}, timeout=30).json()["prompt"]
    return p.replace("<|think|>\n", "")      # thinking off, as in prod (D-025)


def n_tokens(text):
    return len(requests.post(f"{LLM}/tokenize", json={"content": text}, timeout=30).json()["tokens"])


def history(n_tok):
    """A system turn of ~n_tok tokens of in-domain Russian text (stands for persona + dialog history)."""
    chars = int(n_tok * 3.2)
    text = CALIB[:chars]
    while n_tokens(text) < n_tok and len(text) < len(CALIB):
        text = CALIB[:len(text) + 400]
    return text


def complete(prompt, n_predict, cache, ignore_eos=False):
    r = requests.post(f"{LLM}/completion", json={"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                                                 "cache_prompt": cache, "ignore_eos": ignore_eos, "id_slot": 0},
                      timeout=600).json()
    t = r["timings"]
    return {"prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"], "prefill_tok_s": t["prompt_per_second"],
            "predicted_n": t["predicted_n"], "decode_tok_s": t["predicted_per_second"]}


def llm(reps):
    out = {}
    for n in (1000, 3000, 6000):
        sys_txt = history(n)
        rows = []
        for i in range(reps):   # a different question each rep: no reuse even if the cache kept something
            p = templated(f"Вопрос {i + 1}: о чём этот текст? Ответь одной фразой.", sys_txt)
            rows.append(complete(p, 1, cache=False))
        out[f"cold_{n}"] = {"prompt_n": rows[0]["prompt_n"],
                            "ttft_s": [round(r["prompt_ms"] / 1000, 2) for r in rows],
                            "prefill_tok_s": [round(r["prefill_tok_s"]) for r in rows]}
    sys_txt = history(3000)
    rows = [complete(templated("Расскажи коротко, что ты видишь на столе.", sys_txt), 128, cache=False, ignore_eos=True)
            for _ in range(reps)]
    out["decode_128_at_3k"] = {"decode_tok_s": [round(r["decode_tok_s"], 1) for r in rows]}
    complete(templated("Первый вопрос: кто ты?", sys_txt), 1, cache=True)          # fill the slot with the 3k prefix
    rows = [complete(templated(f"Новая реплика {i + 1}: а что за окном?", sys_txt), 1, cache=True) for i in range(reps)]
    out["warm_3k_prefix_reused"] = {"prompt_n": [r["prompt_n"] for r in rows],
                                    "ttft_s": [round(r["prompt_ms"] / 1000, 3) for r in rows]}
    return out


def read_wav(p):
    with wave.open(str(p)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, p
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def stt(reps):
    d = ROOT / STT["stt"]["model"]
    rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"),
                                                      num_threads=STT["stt"]["num_threads"])
    wav = read_wav(d / "test_wavs/example.wav")
    out = {}
    for k in (1, 3):
        a = np.concatenate([wav] * k)
        ts = []
        for _ in range(reps + 1):
            s = rec.create_stream(); s.accept_waveform(16000, a)
            t0 = time.perf_counter(); rec.decode_stream(s); ts.append(time.perf_counter() - t0)
        ts = ts[1:]                                   # the first decode warms the graph
        out[f"x{k}"] = {"audio_s": round(len(a) / 16000, 2), "decode_s": [round(t, 3) for t in ts],
                        "rtf": round(statistics.median(ts) / (len(a) / 16000), 3), "text": s.result.text[:80]}
    out["num_threads"] = STT["stt"]["num_threads"]
    vm = ROOT / STT["voice_id"]["model"]
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(vm), num_threads=1))
    for sec in (1.5, 4.0):
        a = np.resize(wav, int(sec * 16000))
        ts = []
        for _ in range(reps + 1):
            s = ex.create_stream(); s.accept_waveform(16000, a); s.input_finished()
            t0 = time.perf_counter(); ex.compute(s); ts.append(time.perf_counter() - t0)
        out[f"voiceid_{sec}s"] = [round(t, 3) for t in ts[1:]]
    return out


def embed(reps):
    text = "Владелец: " + CALIB[:600] + "\nКостян: " + CALIB[600:900]
    ts = []
    for _ in range(reps + 1):
        t0 = time.perf_counter()
        requests.post(f"{EMB}/v1/embeddings", json={"input": text}, timeout=60).raise_for_status()
        ts.append(time.perf_counter() - t0)
    return {"chars": len(text), "latency_s": [round(t, 3) for t in ts[1:]]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", nargs="+", default=["llm", "stt", "embed"])
    a = ap.parse_args()
    res = {"date": time.strftime("%F %T"), "props": requests.get(f"{LLM}/props", timeout=10).json().get("default_generation_settings", {}).get("n_ctx")}
    for part, fn in (("llm", llm), ("stt", stt), ("embed", embed)):
        if part in a.only:
            t0 = time.time(); res[part] = fn(a.reps); print(part, f"{time.time() - t0:.0f} s:", json.dumps(res[part], ensure_ascii=False), flush=True)
    out = ROOT / f"research/bench/results/orin_speed_{date.today().isoformat()}{'_' + a.tag if a.tag else ''}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out)
