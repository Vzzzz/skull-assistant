# Skull Assistant

[![tests](https://github.com/Vzzzz/skull-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/Vzzzz/skull-assistant/actions/workflows/tests.yml) [![license: MIT](https://img.shields.io/github/license/Vzzzz/skull-assistant)](LICENSE) ![python 3.10](https://img.shields.io/badge/python-3.10-blue) ![status: work in progress](https://img.shields.io/badge/status-work_in_progress-orange) ![platform: Jetson Orin Nano 8 GB](https://img.shields.io/badge/platform-Jetson_Orin_Nano_8_GB-76B900)

> **Work in progress** (October 2026). The bench part is built and measured; nothing has been deployed to the device (Jetson Orin Nano Super 8 GB) yet. What works today is listed below; day-by-day state is in `docs/JOURNAL.md` (Russian).

<p align="center">
  <img src="docs/media/skull-jetson.jpg" width="420" alt="Jetson Orin Nano inside an anatomical skull model">
  <br><sub>Jetson Orin Nano inside an anatomical skull: the board with its fan takes about a third of the cranial cavity; the rest goes to the microphone, speaker, camera and power.</sub>
</p>

An offline voice assistant with a character — **Kostyan**, an anatomical skull that sees, hears, remembers the person it talks to and answers in its own tone — and, at the same time, a platform for running several compact ML services side by side on one edge device with 8 GB of shared memory. Nothing leaves the device.

The project is run as research: every architectural decision is backed by a measurement; the decision log (`docs/DECISION_LOG.md`) and the peak-memory budget table (`docs/MEMORY_BUDGET.md`) are part of the repository, not an appendix to an article. All numbers on one page: **[docs/RESULTS.md](docs/RESULTS.md)**. Documentation is in Russian (the target audience of the write-ups is Habr); code, identifiers and comments are in English.

## What works today (bench: RTX 3080 Laptop 8 GB, WSL2)

The full voice loop runs as a set of systemd units on a Redis Streams bus:

| Stage | What | Numbers (bench) |
|---|---|---|
| Listening | **ambient mode**: speech-onset trigger (Silero VAD), no wake word; openWakeWord kept as fallback | CPU, 41–87 MB |
| STT | **GigaAM v3 CTC** (sherpa-onnx, CPU) with punctuation and digits; Whisper `base` dropped after a bake-off (WER 25.8 % → 1.6 % on Common Voice ru) | 0.1–0.2 s per phrase, 396 MB |
| Whose voice | owner verification (CAM++, CPU, same process) with sliding windows inside a segment — answers the owner even over a podcast | EER 0.018, +57 MB, 23–66 ms |
| Brain | **Gemma 4 E4B**, own IQ4_XS + imatrix quantization, `llama-server`, 8k context, vision projector attached | 3.8 GB VRAM, 0.3–1.8 s per turn |
| Out-of-window memory | EmbeddingGemma on CPU + keyword match; exchanges that fell out of the context window come back as a note in the request | facts beyond the window: 0.07 → 0.8–1.0 |
| Turn router | turn type (chat / about the speaker / arithmetic / explain / outside world / command) in one model call, policy per type | accuracy 0.88–0.93 |
| "Should I speak now" | rubber-duck mode: addresses, muttering, phone calls, broadcast, music — conversation state + a binary model question | live: 0 replies over 3.5 h of household background, 1 false of 16 mutterings, 6/6 addresses over a podcast |
| TTS | Piper (CPU), sentence streaming, sox timbre; Qwen3-TTS 0.6B measured as a GPU candidate | first audio 0.5–1.3 s; Piper RT 0.06; Qwen3-TTS RT 0.13, 1.9 GB VRAM |
| Vision | camera in the loop on the bench: gaze at the skull as a trigger (YuNet + head pose), scene context next | working posture still confused with gaze — being tuned |

The evaluation set **CTX-1** (60 turns with anchors for memory, arithmetic and "don't make things up") was run on 14 models from 1.7B to 9B; it picked the brain and measured what memory and the router add (`research/eval/`). The memory budget of the whole mode A — voice stack, VLM with projector, embedder, YOLO detector — was measured all at once: **6.3–6.5 GB of the 7.6 GB available on Orin** (bench-based estimate, `docs/MEMORY_BUDGET.md` §2).

Not there yet: running on the Jetson (next milestone), model swapping between modes A/B, an own wake word on an MCU (sibling project), interrupting the skull while it speaks (barge-in), fact checking via the network. Queue: `docs/TASKS.md`; milestone plan: `docs/ROADMAP.md` §7a.

## How it is made

Two commit authors. The **human** — hardware, the bench, running every measurement, listening to voices, architectural decisions and acceptance of results. The **AI assistant** (Claude Code in two roles, see `docs/agents/`) — measurement scripts, service code against task cards, paper and repo reviews, documentation. Yes, it is vibe coding — with a journal that shows what was tried, where it broke and why this option was chosen; every number is reproducible with the scripts in `research/`.

## Layout

```
AGENTS.md     context and role protocol for agents (CLAUDE.md is the Claude Code wrapper)
services/     production: one service = folder with code + manifest.yaml (budget, residency, lifecycle)
core/         shared code: out-of-window memory, router, text filters; the mode dispatcher arrives in M8
deploy/       systemd/ — skull.target + units, install.sh; stand/ — bench control from Git Bash
research/     reproducible measurements: bench/ (memory, speed), eval/ (CTX-1), memory/ (prototypes), tts/, vision/
docs/         session journal, task queue, open questions, roadmap, decision log, memory budgets, paper reviews
persona/      the character's system prompt (dataset is private, see persona/README)
models/       weights (not in git) + README with sources
```

The "production / research" boundary: `services/` holds only what a unit runs on the device. Everything that measures, compares or trains lives in `research/`. Research reaches production through a `docs/DECISION_LOG.md` entry and a manifest change.

## Architecture in brief

- Bus: Redis Streams (`audio:mic` → `events:wakeword` → `events:transcript` → `events:reply`); dialogue history in Redis, one exchange = one memory record.
- Each service is a separate systemd unit; process isolation is the model-swap mechanism (GPU memory comes back only when the process dies).
- Voice stack and embedder are CPU-only; only the VLM and the detector get the GPU. Every CUDA context costs hundreds of MB out of the same 8 GB.
- Three modes: A — 4B VLM + full sensor stack; B — a large MoE (26B-A4B) streamed from NVMe; C — "trophy". Invariant: the wake word / KWS is never evicted (it moves to an MCU in the sibling project `bitladder`).

Details: `docs/ROADMAP.md`; the original plan: `docs/roadmap-v1.md`.

## Running it

The repository currently describes the author's bench (WSL2 on a laptop, a maintenance flag on the Windows side — see the Russian README). A portable install guide for your own hardware is **TBD** and will come with the Jetson bring-up.

## License

MIT (`LICENSE`). Use it, keep the notice, a link back is appreciated. Model and voice weights are under their own licenses (`models/README.md`); the persona dataset is private and not part of the repository.
