# Jetson Orin Nano Super: bring-up и первый запуск (M5, день 1)

Цель дня: устройство в известном состоянии, пустой след системы замерен, голосовое ядро запущено как на стенде, первый замер памяти VLM снят через tegrastats. Все числа — в `docs/MEMORY_BUDGET.md`, решения — в `docs/DECISION_LOG.md`.

## 0a. Подключение к стенду (владелец, 2026-10-30): Jetson по RJ-45 к ноутбуку

Монитор и клавиатура Jetson нужны только на первую загрузку (пароль, SSH включён по умолчанию в JetPack). Дальше — кабель Ethernet в ноутбук (прямой, кроссовер не нужен: auto-MDI-X) и работа агентом/терминалом по SSH.
- **Адреса.** Два варианта: (а) **общий доступ к интернету Windows (ICS)** на Wi-Fi-адаптере → Ethernet-адаптер ноутбука получает 192.168.137.1, Jetson — адрес по DHCP от Windows и **интернет через ноутбук** (нужен для `apt`, `ollama pull`, `hf download`) — предпочтительно; (б) без ICS — link-local 169.254.x.x с обеих сторон автоматически, SSH работает, интернета у Jetson нет. Имя `<hostname>.local` через mDNS (avahi есть в JetPack; на Windows — Bonjour или по IP из `arp -a`).
- **Из WSL2** (Windows 10, режим NAT): исходящие соединения WSL → Jetson работают как есть (`ssh user@192.168.137.x`, `scp`, `rsync`); обратно Jetson → WSL (Redis, git-remote) без `netsh interface portproxy` недоступно — и не нужно: обмен кодом только через GitHub (AGENTS), шина на устройстве своя.
- **Запасной канал — USB-C device mode** devkit'а: кабель USB-C в ноутбук даёт сеть 192.168.55.1 (Jetson) / .100 (хост) и **последовательную консоль** (`/dev/ttyACM0`, 115200) — единственный способ видеть загрузку и отладить SC7 (D-013), когда сеть ещё не поднялась.
- **Исполнитель на устройстве:** агент остаётся на стенде/Windows, команды — через `ssh` (рабочая копия репо на Jetson: `git clone` с GitHub, push оттуда же); ставить агента на сам Jetson не стоит — память.

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

## 1a. Что везти на Jetson: тяжёлые файлы (владелец, 2026-10-31: «скачивать или копировать со стенда?»)

**Копировать со стенда, не скачивать.** Все веса прода — GGUF и ONNX, форматы не зависят от архитектуры CPU; на стенде лежат ровно те файлы, что проверены замерами (своя квантизация с imatrix на стенде и делалась — повторять её на Orin негде: BF16 15 ГБ + CUDA). Канал — RJ-45 (§0a): гигабит ≈ 100 МБ/с, режим A целиком ≈ 1.5 мин, модель режима B ≈ 3 мин; с HF те же файлы качаются часами (EXECUTOR.md: Xet падает до сотен КБ/с). Из WSL одной командой (пути — как в манифестах, папка `models/` на Jetson создаётся той же раскладки):
```bash
# из WSL стенда; jetson = alias в ~/.ssh/config; --relative сохраняет пути models/...
cd ~/skull-assistant && rsync -avR --progress \
  models/quant/kostyan-e4b-iq4xs-imx.gguf \
  models/hf/gemma-4-E4B-it-GGUF/mmproj-gemma-4-E4B-it-Q8_0.gguf \
  models/hf/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf \
  models/stt/sherpa-onnx-nemo-ctc-punct-giga-am-v3-russian-2025-12-16 \
  models/voice/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx models/voice/owner.npy \
  models/vad models/vision models/tts models/wakeword models/README.md \
  jetson:~/skull-assistant/
sha256sum models/quant/kostyan-e4b-iq4xs-imx.gguf   # сверить с тем же на Jetson после копии
```

| Что | Размер на стенде | Везти | Как |
|---|---|---|---|
| `models/quant/kostyan-e4b-iq4xs-imx.gguf` — VLM режима A (D-023) | 5.1 ГБ | **да** | rsync; на Jetson — на NVMe, `models/` симлинком или mount |
| `models/hf/gemma-4-E4B-it-GGUF/mmproj-…-Q8_0.gguf` — проектор зрения | 0.9 ГБ (папка 1.3) | **да** | rsync только mmproj |
| `models/hf/embeddinggemma-300M-GGUF/…Q8_0.gguf` — эмбеддер памяти | 0.3 ГБ | **да** | rsync |
| `models/stt/…punct-giga-am-v3…` — STT прод (D-004) | 215 МБ | **да** | rsync; v2, v3 без punct, T-one, Vosk — не везти (кандидаты T-040) |
| `models/voice/…campplus….onnx` + `owner.npy` — voice id | ~30 МБ | **да** | rsync; `owner.npy` приватный, не в git — только копией; `voice/data/` (записи) — не нужны; wespeaker — не везти |
| `models/vad/silero_vad.onnx`, `models/vision/*.onnx` (YuNet, Face Mesh 478), `models/tts/*` (Piper dmitri/ruslan), `models/wakeword/*` | < 150 МБ вместе | **да** | rsync папками |
| **режим B**: `gemma4:26b-a4b-it-q4_K_M` (+ `-mtp-` вариант) — ollama-блобы на стенде | 17 ГБ (+0.5 драфт; проектор 1.2 не нужен) | **да, на NVMe** | скопировать `~/.ollama/models` целиком (`rsync -a ~/.ollama/models/ jetson:/nvme/ollama/`, `OLLAMA_MODELS=/nvme/ollama`) — ollama на Orin увидит их без pull; llama-server (T-044) читает тот же блоб по пути из `ollama show --modelfile` |
| `models/quant/gemma-4-E4B-it-BF16.gguf`, `imatrix-e4b-ru.gguf`, `.cache` | 15 ГБ | нет | квантизация сделана; при перевыпуске (T-021 после LoRA) — делать на стенде и везти результат |
| `models/hf/Qwen3-TTS-*`, `MiniCPM-o`, `Spark-X2.6`, `chatterbox`, `emotion2vec*`, `openaudio` | ~26 ГБ | нет | кандидаты и снятые варианты; Qwen3-TTS — только если TTS_TRACK решит ставить его на GPU (отдельно) |
| ollama-модели стенда (`gemma4:e4b-it-qat`, `e2b-it-qat`, кандидаты CTX-1) | десятки ГБ | только `e2b-it-qat` (запасной мозг по памяти, D-021) | в той же копии `~/.ollama/models` — или `ollama pull` на Orin через ICS, если место на NVMe жалко |
| `models/quant/slots/` | — | нет | слоты llama-server создаются заново |
| `.venv` | — | **нет, не копируется** | колёса x86_64; на Orin `pip install -r requirements.txt` заново: aarch64-колёса есть у onnxruntime, sherpa-onnx, opencv-python-headless, piper-tts, redis, numpy; torch в прод-venv не ставить (как на стенде) |
| `~/llama.cpp` (две сборки: CUDA для VLM, CPU для эмбеддера) | — | **нет, собирать** | `git clone` на Orin, тот же commit, что на стенде (`git -C ~/llama.cpp rev-parse HEAD` записать в JOURNAL); CUDA-сборка под sm_87 (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`), ~30–60 мин на Orin; CPU-сборка отдельно (D-024: CUDA-сборка создаёт контекст 176 МБ даже при `-ngl 0`); проверить IQ4_XS-ядра на sm_87 (Q-004) первым запуском |
| ollama, redis, JetPack-компоненты (CUDA, cuDNN) | — | нет | скрипт ollama (arm64 + CUDA JetPack 6), `apt install redis-server`; CUDA в JetPack |
| `persona/system_prompt.md`, манифесты, юниты, код | — | git | `git clone` с GitHub; `persona/dataset/` приватный — везти только если T-020/T-021 пойдут на устройстве (не планируется) |

Итого на NVMe: режим A ≈ 7 ГБ, режим B +17–18 ГБ, запас под слоты и page cache. Порядок §2 п. 3 дополняется: до запуска юнитов — `sha256sum` на четырёх крупных файлах против стенда.

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
