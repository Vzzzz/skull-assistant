# Jetson Orin Nano Super: bring-up и первый запуск (M5, день 1)

Цель дня: устройство в известном состоянии, пустой след системы замерен, голосовое ядро запущено как на стенде, первый замер памяти VLM снят через tegrastats. Все числа — в `docs/MEMORY_BUDGET.md`, решения — в `docs/DECISION_LOG.md`.

## 0. Перед всем: версия и сон (для D-013)

```bash
cat /etc/nv_tegra_release; dpkg -l | grep -E "nvidia-jetpack|nvidia-l4t-core" | head -3
# Jetson Linux ≥ 36.4.3 — иначе выход из SC7 сломан (bitladder/docs/hardware.md)
sudo nvpmodel -q; sudo nvpmodel -m 0 && sudo jetson_clocks      # перед ЛЮБЫМ замером
sudo tegrastats --interval 1000 | head -5                         # пустой след: RAM, GPU, температуры
```

Тест сна — отдельно, когда рядом: `sudo systemctl suspend`, пробуждение кнопкой J14; замерить время до `events:wakeword`-готовности после resume. Ваттметр в розетку: простой / SC7 / под VLM.

## 1. Что переносится с стенда (файлы и что в них менять)

| Что | Как | Изменения под Orin |
|---|---|---|
| Репозиторий | `git clone` с зеркала (GitHub позже) или `scp` из WSL | — |
| `requirements.txt` | `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` | aarch64-колёса есть у ctranslate2, onnxruntime, piper-tts, openwakeword; **`SoundCard` тянет PulseAudio** — на headless Jetson заменить захват на `sounddevice`/ALSA (TASKS T-010) |
| `deploy/systemd/*.service`, `skull.target`, `install.sh` | `deploy/systemd/install.sh` | путь флага сервисного режима `/mnt/c/...` на Orin не существует → условие всегда истинно, сервисы стартуют. Для сервисного режима на устройстве — флаг `/var/lib/skull/maintenance` (параметризовать в install.sh, T-010) |
| ollama | официальный скрипт установки (arm64 + CUDA для JetPack 6); override как на стенде: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_KEEP_ALIVE=-1` | **проверить, что сборка включает flash-attn ядра для Ampere (sm_87)** — без них Qwen3.5 раздувается вдвое (D-015) |
| redis | `apt install redis-server` | — |
| Модели | `deploy/fetch_models.sh stand` (ollama), `models/README.md` для wake word / Piper | NVMe 512 ГБ: модели на NVMe, `OLLAMA_MODELS` туда же |
| `research/bench/bench_vlm_mem.sh` | — | **nvidia-smi на Jetson нет**: вариант с `tegrastats` (RAM used, GR3D) и `/proc/<pid>/status` RSS — TASKS T-009 |
| `research/bench/measure_voice_rss.sh`, `research/tts/piper_rt.py` | как есть | — |
| Микрофон/динамик | ReSpeaker Lite (USB, аппаратный AEC) или любой USB | в SC7 USB обесточен → капча должна переживать re-enumerate (T-005) |

## 2. Порядок дня

1. §0: версия, nvpmodel, jetson_clocks, пустой след (tegrastats 60 с) → MEMORY_BUDGET §1 «Пустой след системы».
2. **Пустой CUDA-контекст** (замер D): три варианта — python с `torch.cuda.init()` (если torch есть), `ollama run` любой модели и `tegrastats` до/после, и голый `cudaFree(0)` на C — RAM-дельта → MEMORY_BUDGET строка «Пустой CUDA-контекст».
3. redis, venv, модели wake word/Piper, `install.sh`, `systemctl --user start skull.target` без brain → голосовой CPU-стек на Orin: `measure_voice_rss.sh` → RSS и CPU% (ожидание: RSS ≈ стенд, CPU ×2–3).
4. `piper_rt.py` → RT на Orin (гейт ≤ 0.3).
5. ollama + `gemma3:4b` → первый диалог через шину на устройстве. Это **порог портфолио №1 в черновом виде** — снять видео.
6. `bench_vlm_mem` (tegrastats-вариант) по кандидатам A: gemma3:4b, gemma4:e2b(-qat), e4b(-qat), qwen3.5:4b, ministral-3:3b, phi4-mini — пик RAM по tegrastats, RSS раннера, tok/s, TTFT тёплый. **Отдельно для Gemma 4 E-моделей: RSS + page cache** (PLE в mmap) под длинным диалогом.
7. Гейт со-резидентности: кандидат + wake word + STT + Piper одновременно, затем + YOLO (когда появится TensorRT-движок) → headroom.

Между прогонами — kill раннера (`ollama stop <model>` или `keep_alive: 0`), тепло: tegrastats температуры и троттлинг-флаги в лог, в закрытом корпусе перемерить финалиста.

## 3. Что записать в конце дня

- MEMORY_BUDGET §1: пустой след, CUDA-контекст (3 варианта), пять CPU-сервисов, VLM-кандидаты; §2: первая строка режима A.
- DECISION_LOG: версия JetPack и состояние SC7 (D-013), результат flash-attn на Ampere, любой отсев кандидатов.
- ISSUES: всё, что сломалось при переносе (аудио, права, USB).
