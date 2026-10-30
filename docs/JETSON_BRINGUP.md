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
| **Микрофон + камера: Logitech C270 — основное устройство захвата звука и видео** (решение владельца 2026-10-30, до покупки ReSpeaker Lite и камеры под глазницы) | один USB-кабель в череп: видео UVC (`/dev/video0`, MJPG 640×480) + встроенный моно-микрофон (`snd_usb_audio`, карта `C270` в `arecord -l`; 16 кГц поддерживается штатно, наш `RATE` 16000) | (1) `capture.py` берёт `sc.default_microphone()` — на Orin после подключения выставить C270 источником по умолчанию (`pactl set-default-source alsa_input.usb-046d_...C270...-02.mono-fallback`, имя — из `pactl list sources short`) или задать имя явно через переменную окружения в юните (добавить `SKULL_MIC` в `capture.py`, одна строка); без PulseAudio на Orin — `SoundCard` не работает, тогда `arecord -D plughw:C270 -r 16000 -f S16_LE -c 1` через pipe (T-010); (2) **AEC нет** — barge-in и уточка при речи черепа остаются по флагу `tts:speaking`, как на стенде; (3) микрофон в камере = микрофон там, где глаза черепа: на столе, ~1 м от владельца — проверить уровень (`arecord` 10 с → RMS) и при тихом сигнале поднять адаптивный порог `trigger: speech` (median×2 уже адаптивен); (4) в SC7 USB обесточен → и камера, и микрофон re-enumerate после пробуждения — T-005 касается обоих (`/dev/video0` может стать `video1`: открывать по `/dev/v4l/by-id/`); (5) на стенде та же C270 шла через usbipd (I-16), на Orin — напрямую, usbipd-ограничения не переносятся |
| Динамик | любой USB-динамик или звук через HDMI/DP монитора — **у Orin Nano devkit нет аудиоразъёма 3.5 мм** | плеер `soundcard` → устройство по умолчанию (`pactl set-default-sink`); без Pulse — `aplay -D plughw:...` через pipe (T-010); до ReSpeaker Lite динамик и микрофон — разные устройства, часы не синхронизированы → AEC программный невозможен, только `tts:speaking` |
| Камера (в глазницах) и место черепа | `skull-vision`, `services/vision/gaze.py`, калибровка `gaze_calib.py --natural` по месту | **требование к месту (T-031 1б, 2026-10-30):** направление «владелец → череп» должно отстоять **≥ 35°** от края рабочей зоны (монитор, окно мессенджера) — естественный взгляд на череп это поворот головы, и рабочая поза, повёрнутая в ту же сторону, от него неотличима ни по радужкам, ни по позе; на стенде череп в 10 см от монитора — триггер не проходит гейт |

## 2. Порядок дня

1. §0: версия, nvpmodel, jetson_clocks, пустой след (tegrastats 60 с) → MEMORY_BUDGET §1 «Пустой след системы».
2. **Пустой CUDA-контекст** (замер D): три варианта — python с `torch.cuda.init()` (если torch есть), `ollama run` любой модели и `tegrastats` до/после, и голый `cudaFree(0)` на C — RAM-дельта → MEMORY_BUDGET строка «Пустой CUDA-контекст».
3. **C270**: подключить, `v4l2-ctl --list-devices`, `arecord -l`; кадр из `.venv` (`gaze.py --show` или `cv2.VideoCapture('/dev/v4l/by-id/...')`), 10 с записи с микрофона камеры → RMS и прослушать; C270 — источник по умолчанию (см. §1). Затем redis, venv, модели wake word/Piper, `install.sh`, `systemctl --user start skull.target` без brain → голосовой CPU-стек на Orin: `measure_voice_rss.sh` → RSS и CPU% (ожидание: RSS ≈ стенд, CPU ×2–3).
4. `piper_rt.py` → RT на Orin (гейт ≤ 0.3).
5. ollama + `gemma3:4b` → первый диалог через шину на устройстве. Это **порог портфолио №1 в черновом виде** — снять видео.
6. `bench_vlm_mem` (tegrastats-вариант) по кандидатам A: gemma3:4b, gemma4:e2b(-qat), e4b(-qat), qwen3.5:4b, ministral-3:3b, phi4-mini — пик RAM по tegrastats, RSS раннера, tok/s, TTFT тёплый. **Отдельно для Gemma 4 E-моделей: RSS + page cache** (PLE в mmap) под длинным диалогом.
7. Гейт со-резидентности: кандидат + wake word + STT + Piper одновременно, затем + YOLO (когда появится TensorRT-движок) → headroom.
8. **Smoke-test режима B (Q-045, день 2):** `OLLAMA_MODELS` на NVMe, `ollama run gemma4:26b-a4b-it-mtp-q4_K_M` — tok/s и TTFT на 5 запросах, `tegrastats` RAM + `/proc/meminfo` Cached (page cache = стриминг экспертов), температура NVMe (`nvme smart-log`) за 10 мин генерации. Это решает, двигать ли M8b раньше. На стенде (2026-10-18) та же модель дала 13.9 tok/s и персону 4.38 — но из host RAM, не с NVMe.

Между прогонами — kill раннера (`ollama stop <model>` или `keep_alive: 0`), тепло: tegrastats температуры и троттлинг-флаги в лог, в закрытом корпусе перемерить финалиста.

## 3. Что записать в конце дня

- MEMORY_BUDGET §1: пустой след, CUDA-контекст (3 варианта), пять CPU-сервисов, VLM-кандидаты; §2: первая строка режима A.
- DECISION_LOG: версия JetPack и состояние SC7 (D-013), результат flash-attn на Ampere, любой отсев кандидатов.
- ISSUES: всё, что сломалось при переносе (аудио, права, USB).
