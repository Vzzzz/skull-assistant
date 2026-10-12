# Очередь задач

Канал между ядром проекта (сессия планирования) и исполнителем (VS Code + Claude Code в WSL). Правила:
- Исполнитель берёт **верхнюю открытую** задачу, если не сказано иное. Начало сессии — чтение `CLAUDE.md` и этого файла.
- Задача закрыта, когда выполнен критерий Done **и** сделана запись: коммит + строка в `DECISION_LOG.md` / `MEMORY_BUDGET.md` / `ISSUES.md`, если что-то выбрано или измерено.
- Владение файлами: исполнитель — `services/`, `core/`, `deploy/`, скрипты в `research/`; ядро — `docs/` и сводки `research/**/results/*.md`. Чужие файлы не править без явной задачи.
- Ветки: `feat/T-NNN-кратко`, мерж в `main` после Done. Ядро читает `git log`/`diff` и обновляет docs.
- Статусы: `open` · `wip` · `done` · `blocked` (с причиной).

| ID | Статус | Задача | Done | Связь |
|---|---|---|---|---|
| T-001 | open | **STT: убрать двойной `transcribe`** в `services/stt/transcriber.py` (второй вызов отбрасывает первый с `condition_on_previous_text=False`) | один вызов; латентность STT на фразе 3 с замерена до/после и записана в ISSUES I-01 | ISSUES I-01 |
| T-002 | open | **Brain: история диалога в Redis** вместо списка в памяти процесса (ключ `dialog:history`, обрезка до N реплик) | рестарт `skull-brain` не теряет контекст; проверено вручную | ISSUES I-05 |
| T-003 | open | **Стрим ответа по предложениям**: brain `stream: True`, разрезка по границам предложений → `events:reply` по одному предложению с полем `seq`/`final`; TTS — очередь синтеза (N+1 синтезируется во время проигрывания N), проигрывание потоком через `sounddevice`, а не `paplay` файла | латентность «конец фразы → первое аудио» замерена до/после; ответ из 3+ предложений звучит без пауз между ними | TTS_TRACK §4 п.1–2, ISSUES I-02/I-08 |
| T-004 | open | **Сервис KWS быстрых команд** `services/kws`: openWakeWord multi-head («стоп», «тише»), резидентен всегда, слушает и во время TTS; публикует `tts:stop` → TTS сбрасывает очередь и останавливает поток ≤100 мс | перебивание работает при включённом динамике (полудуплекс: во время TTS слушает только KWS); манифест с `never_evict: true` | TTS_TRACK §4 п.3, P-002, ISSUES I-10 |
| T-005 | open | **Захват звука устойчив к переподключению устройства**: `services/audio_capture` переживает исчезновение/появление USB-микрофона (SC7 resume на Jetson), переоткрывает поток, не падает в restart-loop | отключение/подключение микрофона на стенде → поток в `audio:mic` возобновляется без ручного рестарта | D-013, ISSUES I-07 |
| T-006 | open | **bench_vlm_mem.sh**: TTFT разделить на `load_duration` и `prompt_eval` (тёплый TTFT — второй запрос); RSS раннера (найти процесс по порту/`OLLAMA_HOST`, не по имени); прогревочный запрос перед замером | CSV с колонками `load_s, ttft_warm_s, rss_runner_mb` заполняется ненулевыми значениями на gemma3:4b | results/vlm_mem_stand_2026-10-12.md §5–6 |
| T-007 | open | **Манифесты**: заполнить `budget` в `services/*/manifest.yaml` из `MEMORY_BUDGET.md` (`measured_on: stand`), `healthcheck` для brain (HTTP ollama) | все пять манифестов без `null` в budget | ROADMAP §2 |
| T-008 | blocked | **Оценочный набор** `research/eval/`: 20 диалоговых промптов, 15 tool-calling кейсов, 1 длинный диалог; `run_eval.py` гонит через ollama и пишет JSON | скрипт прогоняет набор на gemma3:4b и сохраняет ответы; содержимое набора — совместно с владельцем (персона) | ROADMAP «долг Фазы 0»; blocked: нужен текст реплик от владельца |
| T-009 | open | **bench для Jetson**: вариант `bench_vlm_mem.sh` без nvidia-smi — пик RAM по `tegrastats` (опрос 100 мс), RSS и page-cache раннера, температуры/троттлинг в лог; тёплый TTFT | CSV с `ram_peak_mb, rss_runner_mb, temp_max, throttled` на gemma3:4b на устройстве | JETSON_BRINGUP §1, T-006 |
| T-010 | open | **Перенос на Orin**: `audio_capture` на `sounddevice`/ALSA вместо `SoundCard` (PulseAudio); `install.sh` — путь флага сервисного режима параметром (`SKULL_MAINT_FLAG`, на Orin `/var/lib/skull/maintenance`); `OLLAMA_MODELS` на NVMe | `skull.target` поднимается на Orin, `audio:mic` наполняется, флаг работает | JETSON_BRINGUP |
| T-011 | open | **Фоновый режим, замеры 1–2** (AMBIENT_MODE.md): скрипт непрерывного STT на 10-мин записи → CPU%/латентность; emotion2vec+ base/large на русских фрагментах (Dusha, 100 шт.) → F1 и RT на CPU | две таблицы в `research/eval/results/ambient_*.md`; запись в MEMORY_BUDGET | AMBIENT_MODE.md |
| T-012 | open | **CTX-1 probe** (`research/eval/CTX_PROBE.md`): `ctx_probe.py` + формат диалога jsonl + авто-метрики + задания судье; прогон на gemma3:4b при 2k/4k/8k/16k | таблица модель × ctx с recall@45, math, persona, ctx-плато; черновик диалога готов 2026-10-13 (research/eval/ctx_probe_dialog.jsonl, 60 реплик), якоря правит владелец | CTX_PROBE.md, P-007 |
| T-013 | open | **TTS/omni-кандидаты на стенде** (TTS_TRACK §7): Qwen3-TTS 0.6B (GPU и через qwentts.cpp на CPU), Chatterbox-multilingual (ONNX, CPU), MiniCPM-o 4.5 GGUF (только память и RT, полный дуплекс не нужен) — на `research/eval/tts_lines.txt`: RT, первое аудио, RSS/VRAM, русский на слух | строки в TTS_TRACK §2; слепое сравнение 15 реплик с Piper | TTS_TRACK §7, `deploy/fetch_models.sh tts` |

## Закрытые

| ID | Дата | Что | Запись |
|---|---|---|---|
