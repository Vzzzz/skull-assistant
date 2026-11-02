# Jetson Orin Nano Super: bring-up и первый запуск (M5, день 1)

Цель дня: устройство в известном состоянии, пустой след системы замерен, голосовое ядро запущено как на стенде, первый замер памяти VLM снят через tegrastats. Все числа — в `docs/MEMORY_BUDGET.md`, решения — в `docs/DECISION_LOG.md`.

## 0a. Подключение к стенду (владелец, 2026-10-30): Jetson по RJ-45 к ноутбуку

Монитор и клавиатура Jetson нужны только на первую загрузку (пароль, SSH включён по умолчанию в JetPack). Дальше — кабель Ethernet в ноутбук (прямой, кроссовер не нужен: auto-MDI-X) и работа агентом/терминалом по SSH.
- **Адреса.** Два варианта: (а) **общий доступ к интернету Windows (ICS)** на Wi-Fi-адаптере → Ethernet-адаптер ноутбука получает 192.168.137.1, Jetson — адрес по DHCP от Windows и **интернет через ноутбук** (нужен для `apt`, `ollama pull`, `hf download`) — предпочтительно; (б) без ICS — link-local 169.254.x.x с обеих сторон автоматически, SSH работает, интернета у Jetson нет. Имя `<hostname>.local` через mDNS (avahi есть в JetPack; на Windows — Bonjour или по IP из `arp -a`).
- **Из WSL2** (Windows 10, режим NAT): исходящие соединения WSL → Jetson работают как есть (`ssh user@192.168.137.x`, `scp`, `rsync`); обратно Jetson → WSL (Redis, git-remote) без `netsh interface portproxy` недоступно — и не нужно: обмен кодом только через GitHub (AGENTS), шина на устройстве своя.
- **Запасной канал — USB-C device mode** devkit'а: кабель USB-C в ноутбук даёт сеть 192.168.55.1 (Jetson) / .100 (хост) и **последовательную консоль** (`/dev/ttyACM0`, 115200) — единственный способ видеть загрузку и отладить SC7 (D-013), когда сеть ещё не поднялась.
- **Исполнитель на устройстве:** агент остаётся на стенде/Windows, команды — через `ssh` (рабочая копия репо на Jetson: `git clone` с GitHub, push оттуда же); ставить агента на сам Jetson не стоит — память.

## 0b. Прошивка и SD-образ (владелец, 2026-11-02: «jetson-orin-nano-devkit-super-SD-image_JP6.2.1.zip — это то, что надо закидывать на SD?»)

Да. Это официальный образ SD-карты для **Jetson Orin Nano Super Developer Kit**, JetPack 6.2.1 (Jetson Linux 36.4.x, Ubuntu 22.04, Python 3.10 — совпадает с `.venv` стенда, §1b). Порядок:
1. **Карта:** microSD ≥ 64 ГБ (UHS-1/A2). Распаковать zip → `.img` → записать **balenaEtcher** или Raspberry Pi Imager на Windows (из Git Bash ничего не нужно). Корень системы на SD — нормально для первого дня; модели режима B и `OLLAMA_MODELS` — на NVMe (§1a); перенос корня на NVMe — потом, отдельным шагом, не сейчас.
2. **Прошивка платы (QSPI).** Образ JetPack 6 грузится только при прошивке UEFI серии 36.x. У devkit, продающегося как **Super** (с декабря 2024), она уже стоит — карта грузится сразу. Если плата старее (куплена с JetPack 5): первая загрузка с JP6-карты не пройдёт; тогда один раз: образ JetPack **5.1.3** на карту → загрузиться → `sudo apt update && sudo apt install nvidia-l4t-jetson-orin-nano-qspi-updater` → перезагрузка (QSPI → 35.5) → снова карта JP6.2.1, при первой загрузке UEFI сама обновится до 36.x. Проверка после загрузки: `cat /etc/nv_tegra_release` (R36), `sudo nvbootctrl dump-slots-info` при желании.
3. **Первая загрузка:** монитор (DP) + клавиатура, мастер OEM-настройки: язык, пользователь, hostname, SSH включён по умолчанию. **Сделано 2026-11-02 без монитора** (у владельца только HDMI, а на плате DP; переходник HDMI→USB-C не поможет — USB-C платы видео не выводит): консоль по USB-C — на Windows COM4 (устройство NVIDIA VID 0955), портативный PuTTY `Downloads\putty\putty.exe`, Serial 115200; мастер текстом: лицензия, язык, часовой пояс, пользователь `vzzzz`, раздел на всю карту (57913 МБ). Сразу: `sudo nvpmodel -m 2` (MAXN SUPER) и `sudo jetson_clocks` — §0. Факты платы после первой загрузки: R36.4.4 (JetPack 6.2.1), ядро 5.15.148-tegra, CUDA 12.6, Python 3.10.12, 6 ядер, RAM 7619 МБ (пустой след по `tegrastats` **701 МБ**, свободно 5.4 ГБ), zram-своп 6×635 МБ = 3.8 ГБ (гейт «swap 0» в T-056 — именно про него), корень на SD 56 ГБ (занято 20), NVMe KC3000 477 ГБ без разделов, hostname `ubuntu`, часы не выставлены (без интернета нет NTP; `sudo timedatectl set-timezone Asia/Novosibirsk` + время с ноутбука, иначе даты в логах и результатах врут).
4. Дальше по порядку: §0 (версия, пустой след) → §0a (RJ-45 к ноутбуку, SSH из WSL) → §1b (сборка) → §1a (копия моделей) → §2 (порядок дня).
5. **NVMe (Kingston KC3000, владелец 2026-11-02: «нужен ли радиатор?»).** Сразу — нет: на Orin Nano слот PCIe Gen3, диск греется меньше, чем на Gen4-хосте, а заводская графеновая наклейка — уже теплораспределитель (не снимать). Решает замер: `sudo nvme smart-log /dev/nvme0 | grep -i temp` в простое, во время копирования 17 ГБ (§1a) и на 10 мин стриминга экспертов режима B (§2 п. 8, Q-045). Выше **70 °C** под нагрузкой — ставить низкопрофильный радиатор (термопрокладка + алюминий 2–3 мм; слот снизу платы, проверить зазор в корпусе черепа), порог троттлинга KC3000 ≈ 75–80 °C. В режиме A диск почти не работает — вопрос только для B.

**Ориентир скорости для режима A на устройстве** (NVIDIA, P-049, Nemotron Bolt E4B = Gemma 4 E4B-класс, llama.cpp CUDA, Q4_K_M, вход 3k / выход 50): **VRAM 4.2 ГБ, TTFT 5.8 с, 15.5 ток/с** на Orin Nano 8 ГБ (AGX Orin: 3.7 с / 24 ток/с). Против стенда (префилл ~2000 ток/с, 66–70 ток/с) это ×4 медленнее по префиллу и ×4.5 по декоду; для диалога это означает: полный пере-префилл истории 3k — 6 с, поэтому переиспользование префикса слота (P-033) на устройстве обязательно, а декод 15 ток/с для речи достаточен. Числа заменить своими в §2 п. 6.

**USB-линк и VPN на ноутбуке (владелец, 2026-11-02, стоило 20 минут).** По USB-C плата поднимает мост `l4tbr0` 192.168.55.1 и раздаёт DHCP, но адаптер Windows «Ethernet 2» (RNDIS) адрес не получил (остался 169.254.x) — прямого маршрута в 192.168.55.0/24 нет, и весь IPv4-TCP к плате уходил в VPN-туннель `xray_tun` (TUN перехватывает TCP: соединение «устанавливается» даже на закрытый порт, баннер SSH не приходит, `ssh-copy-id` падает с `kex_exchange_identification`; пинг при этом идёт). Диагностика: `Test-NetConnection 192.168.55.1 -Port 2222` → True = перехват; `Find-NetRoute -RemoteIPAddress 192.168.55.1` → `xray_tun`. Лечение без правки VPN (PowerShell от администратора, один раз, сохраняется): `New-NetIPAddress -InterfaceAlias "Ethernet 2" -IPAddress 192.168.55.100 -PrefixLength 24` — прямой маршрут точнее маршрута по умолчанию и побеждает туннель. Обход для проверки здоровья sshd — IPv6 link-local: `ssh vzzzz@fe80::1%<ifIndex>` из Windows. В WSL — `~/.ssh/config`: `Host jetson` → `192.168.55.1`, `User vzzzz`, `IdentityFile ~/.ssh/id_ed25519_new` (ключ с нестандартным именем ssh сам не предлагает — поэтому `ssh-copy-id` «не сработал»); проверка: `ssh jetson hostname`. **На плате ничего от агента не живёт** (владелец, 2026-11-02: «не будет ли MCP через VS Code лишней памятью на устройстве?»): сессия исполнителя остаётся в WSL, плата видит только `sshd` на сессию (единицы МБ); VS Code Remote-SSH на плату **не ставить** — его сервер (сотни МБ RAM + фон CPU) испортит пустой след и пики. Для исполнителя по SSH `sudo` спрашивает пароль — на лабораторной плате проще один раз в консоли: `echo "vzzzz ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/vzzzz` (решение владельца).

## 0. Перед всем: версия и сон (для D-013)

```bash
cat /etc/nv_tegra_release; dpkg -l | grep -E "nvidia-jetpack|nvidia-l4t-core" | head -3
# Jetson Linux ≥ 36.4.3 — иначе выход из SC7 сломан (bitladder/docs/hardware.md)
sudo nvpmodel -q; sudo nvpmodel -m 2 && sudo jetson_clocks      # перед ЛЮБЫМ замером; на комплекте Super: 0 = 15W, 1 = 25W, **2 = MAXN_SUPER**, 3 = 7W (2026-11-02, /etc/nvpmodel.conf) — `-m 0` здесь даёт 15 Вт!
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
| Камера (в глазницах) и место черепа | `skull-vision`, `services/vision/gaze.py`, калибровка `gaze_calib.py --natural` по месту | **требование к месту (T-031 1б, 2026-10-30):** направление «владелец → череп» должно отстоять **≥ 35°** от края рабочей зоны (монитор, окно мессенджера) — естественный взгляд на череп это поворот головы, и рабочая поза, повёрнутая в ту же сторону, от него неотличима ни по радужкам, ни по позе; на стенде череп в 10 см от монитора — триггер не проходит гейт. **По факту (T-031 1г, гейт пройден 2026-11-01): череп и камера — справа и выше рабочей зоны, как на стенде 2026-11-01:** в рабочей позе голова повёрнута от камеры на ~+51° (`yaw`, медиана 10 мин работы, p10–p90 +45…+59°), взгляд на череп — −4.5° (p10–p90 −12…+6°), т. е. направление на череп ≈ 50–55° от рабочего; по вертикали (`pitch` −4.9° против +0.6°) не разделяется — высоту не подбирать, разделяет поворот. Полоса — один `yaw` [−15, +18]°, выдержка 1.0 с: 10 из 12 взглядов, 0 ложных за 20 мин работы с мессенджером справа (`research/vision/results/gaze_1g_2026-11-01.md`). На Orin перекалибровать `gaze_calib.py --natural 600` по месту — полоса зависит от угла |

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

## 1b. Что собирается на устройстве заново: пошагово (2026-10-31)

Порядок: сначала это, потом копия моделей (§1a), потом §2. Всё — по SSH с стенда (§0a), в сервисном режиме не нужно: юниты ещё не установлены. JetPack 6.x = Ubuntu 22.04, Python 3.10 — **та же версия, что в `.venv` стенда**, поэтому `requirements.txt` используется как есть.

**0. Система и место.**
```bash
sudo apt update && sudo apt install -y git cmake build-essential python3-venv python3-dev redis-server \
     libasound2-dev portaudio19-dev pulseaudio v4l-utils sox aria2
lsblk; df -h /mnt/nvme 2>/dev/null || true       # NVMe смонтирован? модели режима B и ollama — только туда
mkdir -p ~/nvme && sudo mount /dev/nvme0n1p1 ~/nvme || true   # если не примонтирован — fstab по UUID
```

**1. Репозиторий.**
```bash
git clone git@github.com:Vzzzz/skull-assistant.git ~/skull-assistant   # или https + токен; ключ SSH — заранее
cd ~/skull-assistant && ln -s ~/nvme/skull-models models                 # models/ → NVMe; раскладка как на стенде (§1a)
```

**2. Python-окружение** (колёса x86_64 со стенда не годятся; все наши пакеты имеют aarch64-колёса для cp310: onnxruntime, sherpa-onnx, opencv-python-headless, piper-tts, ctranslate2, tflite-runtime, SoundCard).
```bash
python3 -m venv .venv && .venv/bin/pip install -U pip wheel
.venv/bin/pip install -r requirements.txt 2>&1 | tee ~/pip-orin.log       # если какое-то колесо не найдётся — ISSUES, не компилировать из исходников вслепую
.venv/bin/python -c "import sherpa_onnx, onnxruntime, cv2, piper, soundcard, redis; print('ok')"
```
Что не ставить: torch (как на стенде — только отдельный venv для research), mediapipe (не нужен: Face Mesh в ONNX), faster-whisper можно оставить как fallback (в requirements есть).

**3. llama.cpp — две сборки, тот же commit, что на стенде** (`git -C ~/llama.cpp rev-parse HEAD` в WSL → записать в JOURNAL дня). Пути бинарников в манифестах уже такие: `~/llama.cpp/build-cuda/bin/llama-server` (VLM, GPU) и `~/llama.cpp/build/bin/llama-server` (эмбеддер, CPU-сборка — D-024: CUDA-сборка создаёт контекст 176 МБ даже при `-ngl 0`).
```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp && git checkout <commit со стенда>
# CPU-сборка (эмбеддер) — быстрая
cmake -S . -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF && cmake --build build -j4 --target llama-server
# CUDA-сборка под Orin (Ampere sm_87); nvcc тяжёлый — -j2, иначе OOM на 8 ГБ; 30–60 мин
cmake -S . -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=OFF
cmake --build build-cuda -j2 --target llama-server llama-cli llama-bench
```
Первая проверка после копии моделей (§1a): `build-cuda/bin/llama-bench -m models/quant/kostyan-e4b-iq4xs-imx.gguf -ngl 99 -fa 1` — tok/s и то, что IQ4_XS-ядра на sm_87 живые (Q-004); если IQ4_XS медленнее Q4_K_M на Orin ощутимо — это и есть ответ на Q-004, формат меняется по замеру, не заранее.

**4. ollama** (только стенд-инструменты и режим B до T-044 на устройстве): официальный скрипт (`curl -fsSL https://ollama.com/install.sh | sh`, arm64 + CUDA JetPack 6), override как на стенде (§1: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_KEEP_ALIVE=-1`, **`OLLAMA_MODELS=~/nvme/ollama`**), drop-in сервисного режима `deploy/systemd/install-ollama-dropin.sh`. Модели — копией `~/.ollama/models` со стенда (§1a), не pull.

**5. Юниты и флаг сервисного режима.** `deploy/systemd/install.sh` — подставит путь репо; флаг на Orin — `/var/lib/skull/maintenance` (`maint_flag.sh` выбирает его, когда нет `/mnt/c`); `loginctl enable-linger $USER`, чтобы user-юниты жили без сессии. Юниты сначала **не** стартовать: §2 п. 1–2 (пустой след и CUDA-контекст) снимаются до них.

**6. Звук и камера** — §1 строки «C270» и «Динамик»: `arecord -l`, `pactl list sources short`, источник по умолчанию C270; `v4l2-ctl --list-devices`.

После §1b и §1a — чеклист: `python -m core.manifests` (с T-044 — таблица бюджетов режима A по манифестам с `measured_on: stand`; после замеров дня — `jetson`), `sha256sum` четырёх крупных файлов, `redis-cli ping`.

## 2. Порядок дня

**Карточка дня 1 — T-056** (2026-11-02): цель — режим A резидентен на устройстве и Σ пиков сверена с MEMORY_BUDGET §2; скорость — если успеем. Пункты ниже — исходный план, порядок и гейт берутся из карточки.

1. §0: версия, nvpmodel, jetson_clocks, пустой след (tegrastats 60 с) → MEMORY_BUDGET §1 «Пустой след системы».
2. **Пустой CUDA-контекст** (замер D): три варианта — python с `torch.cuda.init()` (если torch есть), `ollama run` любой модели и `tegrastats` до/после, и голый `cudaFree(0)` на C — RAM-дельта → MEMORY_BUDGET строка «Пустой CUDA-контекст».
3. **C270**: подключить, `v4l2-ctl --list-devices`, `arecord -l`; кадр из `.venv` (`gaze.py --show` или `cv2.VideoCapture('/dev/v4l/by-id/...')`), 10 с записи с микрофона камеры → RMS и прослушать; C270 — источник по умолчанию (см. §1). Затем redis, venv, модели wake word/Piper, `install.sh`, `systemctl --user start skull.target` без brain → голосовой CPU-стек на Orin: `measure_voice_rss.sh` → RSS и CPU% (ожидание: RSS ≈ стенд, CPU ×2–3).
4. `piper_rt.py` → RT на Orin (гейт ≤ 0.3).
5. ollama + `gemma3:4b` → первый диалог через шину на устройстве. Это **порог портфолио №1 в черновом виде** — снять видео.
6. `bench_vlm_mem` (tegrastats-вариант) по кандидатам A: наша `kostyan-e4b-iq4xs-imx` (прод) и, по итогу T-055, Nemotron Bolt E4B; ориентир NVIDIA — §0b (TTFT 5.8 с @3k, 15.5 ток/с, 4.2 ГБ); запасные gemma4:e2b-qat, qwen3.5:4b — пик RAM по tegrastats, RSS раннера, tok/s, TTFT тёплый. **Отдельно для Gemma 4 E-моделей: RSS + page cache** (PLE в mmap) под длинным диалогом.
7. Гейт со-резидентности: кандидат + wake word + STT + Piper одновременно, затем + YOLO (когда появится TensorRT-движок) → headroom.
8. **Smoke-test режима B (Q-045, день 2):** `OLLAMA_MODELS` на NVMe, `ollama run gemma4:26b-a4b-it-mtp-q4_K_M` — tok/s и TTFT на 5 запросах, `tegrastats` RAM + `/proc/meminfo` Cached (page cache = стриминг экспертов), температура NVMe (`nvme smart-log`) за 10 мин генерации. Это решает, двигать ли M8b раньше. На стенде (2026-10-18) та же модель дала 13.9 tok/s и персону 4.38 — но из host RAM, не с NVMe.

Между прогонами — kill раннера (`ollama stop <model>` или `keep_alive: 0`), тепло: tegrastats температуры и троттлинг-флаги в лог, в закрытом корпусе перемерить финалиста.

## 3. Что записать в конце дня

- MEMORY_BUDGET §1: пустой след, CUDA-контекст (3 варианта), пять CPU-сервисов, VLM-кандидаты; §2: первая строка режима A.
- DECISION_LOG: версия JetPack и состояние SC7 (D-013), результат flash-attn на Ampere, любой отсев кандидатов.
- ISSUES: всё, что сломалось при переносе (аудио, права, USB).
