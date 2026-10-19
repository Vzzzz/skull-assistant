# Очередь задач

Канал между исследователем (ядром) и исполнителем. Исполнитель читает `AGENTS.md` → `docs/agents/EXECUTOR.md` → этот файл и берёт **верхнюю задачу `open`** в индексе, если не названа другая. Порядок в индексе = приоритет.

Правила:
- Задача закрыта, когда выполнен критерий **Done** и сделаны записи из поля **Записать**. Без записи задача не закрыта.
- Ветка `feat/T-NNN-slug`, мерж в `main`, `deploy/sync-win.sh`. Отчёт — строка в «Закрытые» (формат в EXECUTOR.md §6).
- Статусы: `open` · `wip` · `done` · `blocked` (с причиной в карточке).
- Исполнитель правит в `docs/` только статус/отчёт здесь и строки, названные в «Записать». Карточки пишет исследователь.

Шаблон карточки (исследователь заполняет все поля; исполнитель не должен нуждаться в чате):

```
### T-NNN · <название> · <статус>
Зачем: 1–2 строки — что сломано или что надо узнать.
Файлы: что править / создать.
Шаги: нумерованный список действий.
Проверка: команда(ы) и ожидаемый результат.
Done: измеримый критерий.
Записать: строка в «Закрытые» + куда положить числа (ISSUES I-NN, MEMORY_BUDGET §1, results/*.md).
Связь: D-/P-/I-/Q-/документы — справочно.
```

## Индекс

| ID | Статус | Задача | Связь |
|---|---|---|---|
| T-001 | open | STT: убрать двойной `transcribe` | I-01 |
| T-002 | open | Brain: история диалога в Redis | I-05 |
| T-016 | open | Brain: модель и `num_ctx` из манифеста, не хардкод | D-016 |
| T-007 | open | Манифесты: бюджеты из MEMORY_BUDGET, healthcheck brain | ROADMAP §2 |
| T-003 | open | Стрим ответа по предложениям + очередь синтеза в TTS | I-02, I-08 |
| T-004 | open | Сервис KWS быстрых команд, `tts:stop` | I-10, P-002 |
| T-005 | open | Захват звука переживает переподключение микрофона | D-013, I-07 |
| T-006 | done | bench_vlm_mem.sh v2.1: загрузка, тёплый TTFT, префилл (мимо кэша), RSS раннера, host RAM | `vlm_mem_prefill_2026-10-19.md` |
| T-014 | done | Персона: few-shot старт, правило «не знаю» — замеры v1/v2/v3 (D-020, D-021); чтение промпта из файла → в T-016 | D-021 |
| T-015 | open | Арифметика через инструмент `calc` | D-016 п.3 |
| T-018 | done | CTX-1 по кандидатам режима A: 12 моделей + 2 MoE режима B (D-018, D-019, D-021); Nanbeige → Q-035 | MEMORY_BUDGET §3/§4 |
| T-013 | open | TTS/omni-кандидаты на стенде | TTS_TRACK §7 |
| T-011 | open | Фоновый режим: замеры 1–2 | AMBIENT_MODE |
| T-010 | open | Перенос на Orin: ALSA-захват, путь флага, пути моделей | JETSON_BRINGUP, I-12 |
| T-009 | open | bench для Jetson (tegrastats) | JETSON_BRINGUP §1 |
| T-017 | open | Обучить головы openWakeWord: «стоп», «тише», «Костян» | I-04, D-013 п.3 |
| T-019 | open | Очистка диска от моделей, не прошедших отбор (после T-018, T-013, bench) | D-018, Q-009 |
| T-020 | open | Датасет персоны: генератор `gen_dataset.py` (учитель 26B-A4B → судья → jsonl), проверка утечки eval | persona/dataset/README, D-021 |
| T-021 | open | QLoRA smoke-test: Soup на gemma4 e2b-it, 50 примеров → мердж → GGUF → ollama → CTX-1 | P-008, Q-043, Q-044 |
| T-022 | done | Dense 9B при Q3/IQ3 как мозг режима A — остановлена на шаге 1: персона 9b Q4 2.8 (D-022) | Q-046 |
| T-023 | done | Своя квантизация e4b-it с imatrix ≈ QAT (−0.1…0.2); без imatrix −0.65; IQ4_XS+imx — кандидат для Orin (D-023) | Q-047, Q-043 |
| T-024 | done | ollama vs llama-server: паритет по памяти/скорости; slots 62/32 мс, thaw ≈ 3.5 с; logprobs есть → llama-server (D-024) | Q-048, A-03, Q-034, M8 |
| T-008 | blocked | Оценочный набор `research/eval/` | Q-029 |

## Карточки

### T-001 · STT: убрать двойной `transcribe` · open
Зачем: в `transcriber.py` `model.transcribe` вызывается дважды подряд; результат первого вызова (с `condition_on_previous_text=False` — профилактика галлюцинаций) выбрасывается. Латентность STT ×2, профилактика не работает.
Файлы: `services/stt/transcriber.py`.
Шаги:
1. Удалить второй вызов `segments, info = model.transcribe(audio, language=None, vad_filter=True)`. Оставить первый (с `condition_on_previous_text=False`).
2. Обернуть оставшийся вызов и сборку `text` в замер: `t0 = time.perf_counter()` до, после — `print(f"stt: transcribe {time.perf_counter()-t0:.2f}s for {len(audio)/RATE:.1f}s audio")`. Заметь: `segments` — генератор, время считается после `" ".join(...)`.
3. До правки (шаг 1 ещё не сделан, но шаг 2 добавлен) — произнести 3 фразы по ~3 с, записать время из лога. После правки — те же 3 фразы.
Проверка: `systemctl --user restart skull-stt && journalctl --user -u skull-stt -f`; wake word + фраза; в логе одна строка `stt: transcribe …` на фразу, текст распознан.
Done: один вызов; таблица «фраза · до · после» (3 строки) в ISSUES I-01.
Записать: ISSUES I-01 — числа до/после в колонку «Действие», класс → `закрыт`; строка в «Закрытые».
Связь: I-01.

### T-002 · Brain: история диалога в Redis · open
Зачем: история диалога — список в памяти процесса; `Restart=on-failure` и любой рестарт юнита стирают контекст (I-05).
Файлы: `services/brain/responder.py`.
Шаги:
1. Вместо `history = []` — Redis list `dialog:history`: каждый элемент — JSON `{"role": ..., "content": ...}`. При старте: `history = [json.loads(x) for x in r.lrange('dialog:history', 0, -1)]`.
2. После каждой реплики пользователя и ответа — `r.rpush('dialog:history', json.dumps(msg, ensure_ascii=False))`, затем `r.ltrim('dialog:history', -2*MAX_TURNS, -1)` (MAX_TURNS = 40 обменов = 80 сообщений).
3. В запрос к ollama передавать список из Redis (не локальную копию), чтобы источник был один.
4. Команда очистки для отладки: если в `events:transcript` пришёл текст ровно `/reset` — `r.delete('dialog:history')`, ответ не генерировать.
Проверка: два обмена через микрофон (или `redis-cli xadd events:transcript '*' text "меня зовут Валерий"`), затем `systemctl --user restart skull-brain`, затем `redis-cli xadd events:transcript '*' text "как меня зовут?"` — ответ в `events:reply` содержит имя. `redis-cli llen dialog:history` растёт и не превышает 80.
Done: рестарт `skull-brain` не теряет контекст; `llen` ≤ 80.
Записать: ISSUES I-05 → `закрыт`, строка в «Закрытые».
Связь: I-05; дальше история переедет в SQLite (M6) и/или объект состояния (P-007, Q-012).

### T-016 · Brain: модель и `num_ctx` из манифеста, не хардкод · open
**Дополнение 2026-10-19 (D-024):** рантайм brain — llama-server, не ollama. В этой же задаче: `params.backend: llama-server`, `params.gguf: models/quant/kostyan-e4b-iq4xs-imx.gguf`, `params.mmproj` (скачать `mmproj-gemma-4-E4B-it-Q8_0.gguf` из ggml-org, снять пик VRAM с проектором — хвост T-024), `thaw_budget_s: 4`, `freeze: slot-save-kill`; `responder.py` ходит в `/v1/chat/completions` на :11437 (сервер — отдельный юнит `skull-llm.service`: `llama-server -m … --mmproj … -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 8192 --reasoning-budget 0 --slot-save-path …`, добавить в `install.sh`); `think` → `--reasoning-budget 0` на сервере. Ollama для стенда не трогать.
Зачем: `MODEL = "gemma3:4b"` и `num_ctx: 8192` зашиты в `responder.py`; при смене кандидата или окна (D-016) придётся править код, а манифест — источник правды по контракту платформы.
Файлы: `services/brain/responder.py`, `services/brain/manifest.yaml`, `requirements.txt`.
Шаги:
1. В `manifest.yaml` добавить блок `params:` с полями `model: gemma3:4b`, `num_ctx: 8192`, `temperature: 0.3`, `max_turns: 40`, `think: false`, **`persona: persona/system_prompt_rules.md`** (путь от корня репо; по D-021 промпт — параметр модели: для gemma3 и e2b-it-qat — `_rules.md` без примеров, для e4b-it-qat — `system_prompt.md`). `responder.py` читает файл из `params.persona` при старте вместо константы `PERSONA` (закрывает I-11); в лог — путь и длина промпта. Последнее обязательно: у думающих моделей (qwen3.5, gemma4) ollama по умолчанию включает режим размышления, и brain получит пустой `content` (так сломался прогон CTX-1 на qwen3.5:4b, T-018). Передавать `"think": params["think"]` в теле запроса `/api/chat`; если ollama отвечает 400 на `think` для модели без этой возможности — повторить запрос без поля.
2. В `responder.py` читать манифест при старте: путь — `Path(__file__).with_name('manifest.yaml')`; `yaml.safe_load`. Если `pyyaml` нет в `requirements.txt` — добавить `pyyaml>=6`.
3. `MODEL`, `num_ctx`, `temperature`, `MAX_TURNS` — из манифеста; `options` запроса: `{"num_ctx": ..., "temperature": ...}`.
4. В лог при старте: `brain: model=<…> num_ctx=<…>`.
Проверка: `systemctl --user restart skull-brain && journalctl --user -u skull-brain -n 5` — строка со значениями из манифеста; один обмен через `redis-cli xadd events:transcript '*' text "привет"` отвечает.
Done: в `responder.py` нет строковых литералов с именем модели и числом контекста.
Записать: строка в «Закрытые».
Связь: D-016 п.1, ROADMAP §2.

### T-007 · Манифесты: бюджеты из MEMORY_BUDGET, healthcheck brain · open
Зачем: все пять манифестов имеют `null` в `budget`; диспетчер (M8) не допустит сервис без `measured_on`. Стендовые значения уже есть в `docs/MEMORY_BUDGET.md` §1 (замер 2026-10-12).
Файлы: `services/{audio_capture,wakeword,stt,tts,brain}/manifest.yaml`.
Шаги:
1. Заполнить `budget` пиками из MEMORY_BUDGET §1, `measured_on: stand`: audio_capture `ram_mb: 44`; wakeword `ram_mb: 176`; stt `ram_mb: 321` (пик при загрузке); tts `ram_mb: 486` (пик синтеза); brain `ram_mb: 30`, `vram_mb: 4056` пока модель gemma3:4b; **после переключения на gemma4:e4b-it-qat (T-016) — `vram_mb: 4258`** (num_ctx 8192, flash + q8_0, замер 2026-10-19). `vram_mb: 0` у CPU-сервисов.
2. brain: `healthcheck: http://127.0.0.1:11434/api/tags`; `thaw_budget_s: 10` (ожидание, пометить комментарием `# expected, not measured`).
3. Остальные `null` (healthcheck CPU-сервисов) оставить, но заменить на `healthcheck: null  # TODO: systemd notify` — чтобы было видно, что осознанно.
Проверка: `grep -n null services/*/manifest.yaml` — только строки с `# TODO`; `python3 -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('services/*/manifest.yaml')]"` без ошибок.
Done: `budget` без `null` во всех пяти манифестах.
Записать: строка в «Закрытые».
Связь: ROADMAP §2, MEMORY_BUDGET §1, D-015, D-016.

### T-003 · Стрим ответа по предложениям + очередь синтеза в TTS · open
Зачем: brain ждёт весь ответ (`stream: False`), TTS синтезирует весь текст в файл, потом sox, потом `paplay`. Латентность «конец фразы → первое аудио» = полная генерация + полный синтез; прервать нечем (I-02, I-08). Piper не стримит внутри предложения — значит, резать на предложения надо до него.
Файлы: `services/brain/responder.py`, `services/tts/speaker.py`, `requirements.txt` (`sounddevice`, если нет).
Шаги:
1. brain: `stream: True`; читать NDJSON-строки ответа ollama, накапливать текст; при обнаружении границы предложения (регулярка `[.!?…]+["»)]?\s` или конец ответа) публиковать `events:reply` с полями `text`, `reply_id` (uuid на весь ответ), `seq` (0,1,2…), `final` (`1` на последнем). В историю (T-002) писать полный ответ один раз по окончании.
2. tts: основной цикл читает `events:reply` и кладёт `(reply_id, seq, text)` в `queue.Queue`. Поток-синтезатор берёт из очереди, синтезирует **в память** (API piper-tts установленной версии: raw int16 PCM без файла), кладёт PCM во вторую очередь. Поток-плеер пишет PCM чанками в `sounddevice.OutputStream` (sample rate — из `voice.config.sample_rate`).
3. Эффекты sox: на время задачи убрать (pitch/reverb — TTS_TRACK §3 говорит, что pitch-shift даёт артефакт и с дообученным голосом не нужен). Если без эффектов голос неприемлем — применять sox к raw через pipe (`sox -t raw -r <sr> -e signed -b 16 -c 1 - -t raw … pitch -300`), не через файлы. Отметить в отчёте, какой вариант остался.
4. `tts:speaking` ставить при первом чанке в плеере, снимать когда очереди пусты и поток дописал.
5. Замер латентности: stt добавляет в `events:transcript` поле `ts` (`time.time()`); tts логирует `tts: first audio +X.XXs` = время первого чанка в плеере минус `ts` (прокинуть `ts` через brain в `events:reply`). Снять на 3 ответах до правки (там же: время до `paplay`) и после.
Проверка: вопрос, на который ответ длиннее 3 предложений («расскажи, как устроен замок»); на слух — нет пауз между предложениями, речь начинается до того, как brain дописал ответ (видно по логам обоих юнитов).
Done: таблица «ответ · до · после» (3 строки) для латентности первого аудио; ответ из 3+ предложений без пауз.
Записать: ISSUES I-02, I-08 — числа и статус; MEMORY_BUDGET §1 строка TTS — новый пик RSS (`VmHWM` из `/proc/<pid>/status` после 10 ответов), если изменился; строка в «Закрытые».
Связь: TTS_TRACK §4 п.1–2, I-02, I-08, D-016.

### T-004 · Сервис KWS быстрых команд, `tts:stop` · open
Зачем: пока череп говорит, его нельзя перебить: wake word глушится флагом `tts:speaking` (I-10). Нужен отдельный сервис, который слушает всегда, в том числе во время TTS, и умеет остановить речь. Пока своих голов для «стоп»/«тише» нет (T-017) — использовать стоковое `hey_jarvis` как временное слово «стоп».
Предусловие: T-003 (поток в TTS, который можно остановить).
Файлы: создать `services/kws/listener.py`, `services/kws/manifest.yaml`, `deploy/systemd/skull-kws.service`; править `deploy/systemd/install.sh` (добавить юнит в список), `services/tts/speaker.py`.
Шаги:
1. `listener.py` — копия структуры `services/wakeword/detector.py`: читает `audio:mic`, openWakeWord с теми же feature-моделями; словарь `COMMANDS = {"hey_jarvis": "stop"}` (имя головы → команда). **Не** проверять `tts:speaking`. При срабатывании: `r.publish('kws:cmd', 'stop')` и `r.xadd('events:kws', {'cmd': 'stop', 'score': …})`; cooldown как в detector.py.
2. `speaker.py`: отдельный поток подписан на `kws:cmd` (pubsub); по `stop` — очистить обе очереди из T-003, остановить `OutputStream` (`abort()`), снять `tts:speaking`; в лог `tts: stop +X ms` (время от сообщения до остановки потока).
3. `manifest.yaml`: `never_evict: true`, `residency: A/B/C: resident`, `budget` — заполнить после замера RSS (`VmHWM`), `measured_on: stand`.
4. Юнит `skull-kws.service` по образцу `skull-wakeword.service`; `install.sh` — добавить в `enable`.
Проверка: длинный ответ, во время речи произнести слово; речь обрывается; в логе tts `stop +N ms`, N ≤ 100. Повторить 5 раз, записать N.
Done: 5 из 5 прерываний ≤ 100 мс при включённом динамике; `services/kws/manifest.yaml` с `never_evict: true` и RSS.
Записать: ISSUES I-10 — числа, статус; MEMORY_BUDGET §1 строка «KWS быстрых команд» — RSS пик, CPU%; строка в «Закрытые».
Связь: TTS_TRACK §4 п.3, P-002, I-10, T-017.

### T-005 · Захват звука переживает переподключение микрофона · open
Зачем: на Jetson после сна SC7 USB обесточен, микрофон переинициализируется (D-013). Сейчас `capture.py` открывает устройство один раз; исчезновение → исключение → `Restart=on-failure` → цикл рестартов, если устройство ещё не вернулось.
Файлы: `services/audio_capture/capture.py`.
Шаги:
1. Обернуть открытие микрофона и цикл записи в `while True: try: … except Exception as e: log; time.sleep(1); continue` — при ошибке заново `sc.default_microphone()`.
2. Сторожок тишины: если 5 с подряд все кадры нулевые (устройство «висит», но не падает) — закрыть и переоткрыть.
3. В лог при каждом (пере)открытии: имя устройства; при ошибке — текст исключения, без трейса на каждой итерации.
Проверка: на стенде отключить и подключить USB-микрофон (Windows-сторона) при работающем юните; `redis-cli xinfo stream audio:mic | grep last-generated-id` — id растёт после подключения; `systemctl --user status skull-capture` — без рестартов (`journalctl` без «Scheduled restart»). Если в WSLg устройство не исчезает при отключении — отметить в отчёте и проверить сторожок тишины (отключить звук в Windows).
Done: после отключения/подключения поток возобновляется без рестарта юнита.
Записать: ISSUES I-07 — результат; строка в «Закрытые».
Связь: D-013, I-07, JETSON_BRINGUP §1; на Orin перепроверить после T-010.

### T-006 · bench_vlm_mem.sh: тёплый TTFT, RSS раннера, прогрев · open
Зачем: в прогоне 2026-10-12 TTFT включает `load_duration` (модель грузится каждый раз), RSS раннера не снялся (`pgrep` не нашёл процесс), первый запрос после старта сервера даёт выбросы tok/s.
Файлы: `research/bench/bench_vlm_mem.sh`.
Шаги:
1. Перед замером — прогревочный запрос (тот же промпт, `num_predict: 8`), не записывать.
2. Второй запрос — замер: из ответа ollama взять `load_duration`, `prompt_eval_duration`, `eval_count`, `eval_duration`; колонки `load_s`, `ttft_warm_s` (= `prompt_eval_duration` второго запроса), `tok_s` (= `eval_count/eval_duration`).
3. RSS раннера: PID искать по порту — `ss -ltnp | grep 11435` даёт PID сервера; раннер — его дочерний процесс (`pgrep -P <pid>`), либо процесс с `ollama` в cmdline и самым большим RSS; `VmHWM` из `/proc/<pid>/status` → `rss_runner_mb`.
4. Шапка CSV обновлена; старый CSV не перезаписывать (новое имя с датой).
Проверка: `bash research/bench/bench_vlm_mem.sh research/bench/results/vlm_mem_stand_$(date +%F).csv` на одной модели `gemma3:4b` при ctx 2048 — колонки `load_s, ttft_warm_s, rss_runner_mb` ненулевые.
Done: CSV с ненулевыми `load_s, ttft_warm_s, rss_runner_mb` на gemma3:4b.
Записать: CSV в `research/bench/results/` (коммитится); строка в «Закрытые» с числами для gemma3:4b.
Связь: `research/bench/results/vlm_mem_stand_2026-10-12.md` §5–6, Q-003, T-009.

### T-014 · Персона: few-shot старт, правило «не знаю» · done (2026-10-18, D-021)
Закрыта владельцем и исследователем: три версии промпта × три модели × два seed, выводы в D-020/D-021. Шаг «brain читает промпт из файла» (I-11) перенесён в T-016 вместе с `params.persona`. Ниже — исходная карточка для истории.
Зачем: CTX-1 (D-016) показал: персона первые ~10 реплик на 3 из 5 (назидательный тон), к середине — 5; на вымышленный прибор модель уверенно выдумала. Промпт персоны при этом продублирован в `responder.py` и не читается из `persona/system_prompt.md` (I-11) — две копии разойдутся.
Состояние (2026-10-17): **прогон 1 сделан** владельцем на промпте v2 — результат D-020: примеры пересекались с диалогом (утечка), e2b-it-qat просела; промпт переписан на **v3** (примеры не из диалога). **Прогон 2**: те же три модели на v3 + второй seed для recall; e2b-it-qat дополнительно с усечённым блоком примеров (2–3) — для этого временно скопировать `persona/system_prompt.md`, убрать последние два примера, прогнать, вернуть файл (или добавить в `ctx_probe.py` параметр `--persona <file>` — 3 строки, предпочтительно). Базовые линии: v1 — `results/ctx_probe_candidates_2026-10-17.md`, v2 — `results/ctx_probe_t014_2026-10-17.md`. Перед прогоном — `git pull --ff-only origin main` (промпт v3 и `expect.alt` в диалоге). Команда:
```bash
for m in gemma3:4b gemma4:e2b-it-qat gemma4:e4b-it-qat; do for s in 42 7; do .venv/bin/python research/eval/ctx_probe.py --model "$m" --ctx 8192 --seed $s --judge qwen3.5:9b 2>&1 | tee -a research/eval/results/ctx_probe_t014b.log; done; ollama stop "$m"; done
```
(Внимание: `ctx_probe.py` пишет в `ctx_probe_<model>_<ctx>.json` без seed в имени — второй seed перезапишет первый; добавить seed в имя файла при `--seed != 42` — тоже 2 строки, сделать в этой задаче.)
Файлы: `services/brain/responder.py`, `research/eval/ctx_probe.py` (только если он тоже дублирует промпт — проверить).
Шаги:
1. `responder.py`: удалить константу `PERSONA`; читать `persona/system_prompt.md` при старте (путь от корня репо: `Path(__file__).resolve().parents[2] / 'persona' / 'system_prompt.md'`).
2. Убедиться, что `ctx_probe.py` берёт промпт из того же файла.
3. Прогон: `.venv/bin/python research/eval/ctx_probe.py --model gemma3:4b --ctx 8192 --judge qwen3.5:9b` (стенд погашен: `systemctl --user stop skull.target`; системный ollama нужен). Сравнить с `research/eval/results/ctx_probe_gemma3_4b_2026-10-13.md`: персона на t1/t5/t8/t10, L2 (выдумал / оговорился).
Проверка: вывод скрипта — таблица; `journalctl --user -u skull-brain -n 3` после рестарта — промпт загружен (лог длины промпта).
Done: персона на t1/t5/t8/t10 ≥ 4 (судья); L2 — оговорка или отказ. Если порог не достигнут — задача всё равно закрывается с числами, вывод делает исследователь.
Записать: сводка `research/eval/results/ctx_probe_gemma3_4b_<дата>.md` (скрипт пишет сам) + JSON логи в git; ISSUES I-11 → `закрыт`; строка в «Закрытые» с числами до/после.
Связь: D-016 п.4–5, I-11, Q-010, Q-030.

### T-015 · Арифметика через инструмент `calc` · open
Зачем: gemma3:4b считает неверно без выкладки (0/3 в CTX-1), а требование краткости в персоне подавляет выкладку. Решение D-016 п.3: считать вне модели.
Файлы: `services/brain/responder.py` (или новый `services/brain/tools.py`).
Шаги:
1. Определить инструмент `calc(expression: str) -> str` — безопасный вычислитель: разбор через `ast`, разрешены числа, `+ - * / // % **`, скобки, функции `round`, `abs`; всё остальное — ошибка. Никакого `eval`.
2. Передавать инструмент в ollama `/api/chat` через поле `tools` (формат ollama tool-calling). Если модель вернула `tool_calls` — выполнить `calc`, добавить сообщение `role: tool` с результатом, запросить финальный ответ (второй вызов; при стриме из T-003 — первый вызов без стрима, второй со стримом).
3. В системный промпт (через `persona/system_prompt.md`, строка от исследователя — есть в Q-030) правило: вычисления — только через инструмент, итог озвучить в образе.
4. Если модель не вызывает инструмент на задачах CTX-1 (`ctx_probe_dialog.jsonl`, блок `math`) — fallback: регулярка на «сколько будет / умножить / процентов / во сколько приеду» → принудительный вызов; отметить в отчёте, сколько из 3 задач сработало по tool-call, сколько по fallback.
Проверка: `.venv/bin/python research/eval/ctx_probe.py --model gemma3:4b --ctx 8192` — блок math 3/3 верно (`math верно` = 1.0). Если `ctx_probe.py` ходит в ollama напрямую, минуя brain, — написать в карточку отчёта, и проверить через шину: `redis-cli xadd events:transcript '*' text "сколько будет семнадцать умножить на двадцать три"` → ответ содержит 391.
Done: три задачи CTX-1 верно через шину; инструмент без `eval`.
Записать: строка в «Закрытые» с «tool-call / fallback» по каждой задаче; MEMORY_BUDGET не меняется.
Связь: D-016 п.3, M5-гейт tool-calling, T-014.

### T-018 · CTX-1 по кандидатам режима A: персона и память × окно · done (2026-10-19)
Закрыта: 12 моделей режима A (gemma3, gemma4 ×4, qwen3.5:4b, ministral ×2, phi4-mini, smollm2, MiniCPM5, minicpm-v4.6) + 2 MoE режима B (gemma4 26B-A4B, qwen3.6 35B-A3B) — все с оценками судьи в JSON; сводки `ctx_probe_candidates_2026-10-17.md`, `ctx_probe_t014b_2026-10-18.md`, `ctx_probe_moeB_2026-10-19.md`, `ctx_probe_minicpm_2026-10-19.md`. Решения D-018, D-019, D-021. Nanbeige4.2-3B не скачан — остаётся вопросом Q-035, не задачей. Прогоны делал владелец, оценки — исследователь. Ниже — исходная карточка.
Зачем: CTX-1 (D-016) прогнан только на gemma3:4b. Для выбора кандидата M5 нужна та же таблица по всем: кто держит факты из контекста и персону при 8k, кто считает, кто выдумывает. Это заполняет колонки «Персона 40+ реплик» и recall в матрице MEMORY_BUDGET §3.
Состояние (2026-10-17): **сделано** 11 моделей — gemma3:4b, gemma4 e2b / e2b-it-qat / e4b / e4b-it-qat, qwen3.5:4b (D-018), ministral-3:3b/8b, phi4-mini, smollm2, MiniCPM5-2B (batch2, D-019: все текстовые сняты). Сводка `results/ctx_probe_candidates_2026-10-17.md`. **Осталось**: minicpm-v4.6:q4 и Nanbeige4.2-3B (P-016) — скачать и прогнать; оба на **новом** промпте (persona с few-shot, с 2026-10-17) — так и отметить в сводке. Команда (стенд погашен, системный ollama запущен, `git pull` сделан):
```bash
ollama pull minicpm-v4.6:q4; ollama pull hf.co/owao/Nanbeige4.2-3B-GGUF:Q4_K_M
for m in minicpm-v4.6:q4 hf.co/owao/Nanbeige4.2-3B-GGUF:Q4_K_M; do .venv/bin/python research/eval/ctx_probe.py --model "$m" --ctx 4096 8192 --judge qwen3.5:9b 2>&1 | tee -a research/eval/results/ctx_probe_batch3.log; ollama stop "$m"; done
```
Если Nanbeige не стартует в ollama (looped-архитектура может не поддерживаться stock llama.cpp) — строка «не прошла: причина» и дальше. После прогона сообщить исследователю список новых JSON — он оценит персону (D-017) и дополнит сводку. Результаты в WSL ещё не в git — `git add research/eval/results` при закрытии.
Предусловие: модели скачаны (`deploy/fetch_models.sh stand`, сервисный режим); T-014 закрыта (промпт персоны с примерами — чтобы все модели шли с одним промптом; gemma3:4b перемеряется там же). Если T-014 ещё не закрыта — гнать с текущим промптом и отметить это в отчёте.
Файлы: `research/eval/ctx_probe.py` (без правок, если не ломается на какой-то модели), результаты в `research/eval/results/`.
Шаги:
1. Стенд погашен (`systemctl --user stop skull.target`), системный ollama запущен, `ollama ps` пуст.
2. Перегнать qwen3.5:4b (2k/4k/8k) исправленным скриптом; затем остальные: `minicpm-v4.6:q4`, `ministral-3:3b`, `phi4-mini:3.8b`, `smollm2:1.7b`, gemma4:e4b-it-qat@8k/16k — `.venv/bin/python research/eval/ctx_probe.py --model <m> --ctx 4096 8192 --judge qwen3.5:9b`. Те же seed (42) и температура, что в прогоне 2026-10-13. Между моделями — `ollama stop <m>`. Оценка 9b пишется как fallback; **итоговую оценку персоны ставит исследователь по JSON** (D-017) — после прогона сообщить ему, какие файлы новые. Уже прогнанные gemma3/gemma4 (18 пар) оценены 2026-10-15, поле `judge` в их JSON заполнено, 9b — в `judge_local`; не перезаписывать.
2a. В сводке любой прогон с `пустых ответов > 0` помечать; если > 5 — модель считать не прошедшей на этом окне.
3. Если модель не тянет диалог (ошибка шаблона чата, нет русского, падение) — строка «не прошла: причина», дальше не разбираться.
4. Судья `qwen3.5:9b` — для `qwen3.5:4b` это та же семья; отметить в сводке как возможное смещение, не менять судью.
5. Собрать одну таблицу `research/eval/results/ctx_probe_candidates_<дата>.md`: модель × ctx → recall (все), recall (≥30), math верно, персона (судья), L2 выдумал, TTFT посл. 10, tok/s, токенов к концу. Строку gemma3:4b взять из последнего прогона (T-014 или 2026-10-13) с пометкой, откуда.
Проверка: таблица с 7 моделями (или «не прошла») × 2 окна; JSON-логи в `results/` для каждой пары.
Done: сводная таблица; для каждой модели — JSON логи обоих окон в git.
Записать: сводка `.md` и JSON в git; MEMORY_BUDGET §3 — колонка «Персона 40+ реплик» (оценка судьи @8k) и recall@8k в колонку «Вердикт» или новую; строка в «Закрытые». Выводы (кого снимать) — исследователь.
Связь: CTX_PROBE.md, D-016, Q-010, Q-009, T-014, MEMORY_BUDGET §3.

### T-013 · TTS/omni-кандидаты на стенде · open
Зачем: TTS_TRACK §7 — проверить новый класс речевых моделей на русском до bake-off; Piper — базовая линия (RT 0.055–0.09, `research/tts/results/piper_rt_2026-10-12.csv`).
Предусловие: модели скачаны `deploy/fetch_models.sh tts` (сервисный режим стенда). Если каких-то нет — замерять те, что есть, остальные в отчёт.
Файлы: создать `research/tts/bench_tts.py` (по образцу `research/tts/piper_rt.py`: те же колонки + `vram_peak_mb`), результаты в `research/tts/results/`.
Шаги:
1. Кандидаты: Qwen3-TTS-12Hz-0.6B-CustomVoice (а) transformers на GPU, (б) GGUF через qwentts.cpp на CPU, если собирается; Chatterbox-multilingual ONNX на CPU; MiniCPM-o 4.5 GGUF — только загрузка и один синтез: пик VRAM/RAM и RT, без дуплекса.
2. На каждом — 15 строк `research/eval/tts_lines.txt`: RT, время до первого аудио, RSS (`VmHWM`), пик VRAM (`nvidia-smi --query-gpu=memory.used --loop-ms=100` в фоне), wav сохранить в `research/tts/results/<model>/<n>.wav` (не в git).
3. Стенд погашен на время GPU-замеров.
Проверка: `research/tts/results/tts_bench_<дата>.csv` со строками для каждого кандидата × 15 реплик; wav-файлы есть.
Done: CSV + короткая сводка `research/tts/results/tts_bench_<дата>.md` (таблица «кандидат · RT · первое аудио · RSS · VRAM · что не завелось»). Слепое сравнение на слух делает владелец — не часть задачи.
Записать: сводка `.md` и CSV в git; MEMORY_BUDGET §1 — строки кандидатов TTS (пики); строка в «Закрытые».
Связь: TTS_TRACK §2, §6, §7; Q-018.

### T-011 · Фоновый режим: замеры 1–2 · open
Зачем: AMBIENT_MODE требует непрерывного STT и оценки эмоции по аудио; до проектирования нужна их стоимость на CPU и качество на русском.
Файлы: создать `research/eval/ambient_stt_cost.py`, `research/eval/ambient_emotion.py`; результаты в `research/eval/results/ambient_*.md` + CSV.
Шаги:
1. STT: 10-минутная запись бытового звука (подкаст/радио; файл в `research/eval/data/`, не в git) → faster-whisper base int8 и small int8 на VAD-сегментах (Silero VAD через `faster_whisper.vad`): CPU% процесса (усреднённо по `/proc/<pid>/stat`), латентность на сегмент, RSS пик. Отдельно — холостой ход на тишине.
2. Эмоции: emotion2vec+ base и large (funasr) на 100 фрагментах Dusha (открытый корпус; скачать подмножество, путь в `research/eval/data/`): weighted F1 на 4 классах (нейтраль / радость / грусть / злость), RT на CPU, RSS.
3. Зависимости (`funasr`, `modelscope`) — в отдельный `research/eval/requirements-ambient.txt`, не в основной.
Проверка: две таблицы в `research/eval/results/ambient_stt_<дата>.md` и `ambient_emotion_<дата>.md`.
Done: обе таблицы; гейты из AMBIENT_MODE (STT ≤ 1 ядра в среднем; эмоции F1 ≥ 0.5 при RT ≤ 0.2) — отмечены «прошёл / не прошёл», вывод делает исследователь.
Записать: MEMORY_BUDGET §1 — строки «STT непрерывный» и «эмоции (emotion2vec+)»; строка в «Закрытые».
Связь: AMBIENT_MODE, Q-022.

### T-010 · Перенос на Orin: ALSA-захват, путь флага, пути моделей · open
Зачем: на headless Jetson нет PulseAudio (`SoundCard` его требует); путь флага сервисного режима `/mnt/c/...` не существует; в `detector.py` и `speaker.py` абсолютные пути `/home/valeriy/...` (I-12).
Файлы: `services/audio_capture/capture.py`, `services/wakeword/detector.py`, `services/tts/speaker.py`, `deploy/systemd/install.sh`, `deploy/systemd/*.service`, `deploy/systemd/ollama-maintenance.conf`, `requirements.txt`.
Шаги:
1. `capture.py`: захват через `sounddevice.InputStream` (ALSA/PortAudio), устройство — из env `SKULL_MIC` (имя или индекс), по умолчанию `default`; сохранить логику T-005.
2. Пути моделей: `MODELS = Path(__file__).resolve().parents[2] / 'models'` в `detector.py` и `speaker.py`.
3. `install.sh`: флаг из `SKULL_MAINT_FLAG` (по умолчанию `/mnt/c/Users/valer/.skull-maintenance`; на Orin — `/var/lib/skull/maintenance`); подставлять в юниты как `@MAINT@` вместе с `@REPO@`; то же в drop-in ollama.
4. `OLLAMA_MODELS` — переменная в drop-in ollama (на Orin — путь на NVMe), документировать в комментарии.
Проверка на стенде: `deploy/systemd/install.sh` → юниты содержат реальный путь флага; `skull.target` поднимается, `audio:mic` наполняется (`redis-cli xlen audio:mic` растёт); `grep -rn /home/valeriy services/` пусто.
Done: стенд работает после правок; на Orin (когда есть) `SKULL_MAINT_FLAG=/var/lib/skull/maintenance deploy/systemd/install.sh` поднимает target.
Записать: ISSUES I-07 (захват), I-12 → статус; строка в «Закрытые».
Связь: JETSON_BRINGUP §1, I-07, I-12, D-014.

### T-009 · bench для Jetson (tegrastats) · open
Зачем: на Jetson нет `nvidia-smi`; память unified — пик надо снимать по `tegrastats`, плюс RSS и page cache раннера (Q-003), температуры и троттлинг.
Предусловие: T-006. Писать на стенде, проверять на Orin — до появления устройства Done частичный (скрипт написан, не запущен).
Файлы: создать `research/bench/bench_vlm_mem_jetson.sh` (ответвление от `bench_vlm_mem.sh`).
Шаги:
1. Вместо nvidia-smi — `tegrastats --interval 100` в фоне в файл; парсить `RAM X/YMB`, `GR3D_FREQ`, температуры `CPU@`, `GPU@`, флаг троттлинга (`nvpmodel`/`jetson_clocks --show` до/после).
2. Колонки: `ram_peak_mb` (из tegrastats, минус idle до старта), `rss_runner_mb` (VmHWM), `pagecache_delta_mb` (из `/proc/meminfo` Cached до/после загрузки — для Gemma 4 E-моделей), `temp_max`, `throttled`, `load_s`, `ttft_warm_s`, `tok_s`.
3. Перед прогоном — `sudo nvpmodel -m 0 && sudo jetson_clocks` (с подтверждением, что выполнено, в CSV-шапке).
Проверка: на Orin — `bash research/bench/bench_vlm_mem_jetson.sh out.csv` на gemma3:4b, все колонки ненулевые.
Done: CSV с `ram_peak_mb, rss_runner_mb, pagecache_delta_mb, temp_max, throttled` на gemma3:4b на устройстве.
Записать: CSV в git; MEMORY_BUDGET §1 строка VLM (`jetson`), §3 колонка «Память INT4» для замеренных; строка в «Закрытые».
Связь: JETSON_BRINGUP §1–2, Q-003, Q-004, Q-005, T-006.

### T-017 · Обучить головы openWakeWord: «стоп», «тише», «Костян» · open
Зачем: KWS (T-004) работает на стоковом `hey_jarvis`; нужны свои слова. openWakeWord обучает голову на синтетике (TTS многими голосами + аугментации + негативы). Датасет общий с DS-CNN на MCU (D-013 п.3) — генерировать один раз и сохранить.
Предусловие: T-004.
Файлы: создать `research/kws/make_dataset.py`, `research/kws/train_heads.md` (как запускалось), модели в `models/wakeword/<word>.onnx` (не в git; источники в `models/README.md`).
Шаги:
1. Синтетика: Piper-голоса ru (все доступные из `rhasspy/piper-voices`), по 300–500 произнесений на слово с вариациями скорости/шума; негативы — те же голоса на случайных словах + 30 мин бытового звука. Сохранить в `research/kws/data/` (не в git) + скрипт воспроизводим.
2. Обучение по документации openWakeWord (`openwakeword.train` / официальный ноутбук) на стенде, экспорт onnx.
3. Замер в комнате: 20 произнесений каждого слова (владелец), 30 мин фона без слов: FR, FA/час; порог подобрать, записать.
4. Подключить в `services/kws/listener.py` (`COMMANDS`) и `services/wakeword/detector.py` («Костян» вместо `hey_jarvis`).
Проверка: таблица «слово · FR · FA/час · порог» в `research/kws/results/kws_heads_<дата>.md`.
Done: три головы, FR ≤ 10 %, FA ≤ 1/час на стендовом микрофоне; wake word черепа — «Костян».
Записать: сводка `.md` в git; `models/README.md` — строки моделей; ISSUES I-04, I-13 → статус; строка в «Закрытые».
Связь: I-04, I-13, D-013 п.3, P-002, T-004.

### T-019 · Очистка диска от моделей, не прошедших отбор · open
Зачем: за 2026-10-13…16 скачано ~200 ГБ кандидатов (ollama + `models/hf`). После отбора большинство не нужно; стенд и Orin (NVMe 512 ГБ) не должны носить мёртвый груз. Владелец (2026-10-16): «после всех тестов добавить очистку».
Предусловие: закрыты T-018 (CTX-1 по всем), T-013 (TTS-кандидаты) и bench памяти по короткому списку — чтобы не удалить то, что ещё не измерено. Список «оставить» даёт исследователь в этой карточке перед стартом (ниже — ожидание по D-018).
Файлы: `deploy/fetch_models.sh` (добавить режим `clean` рядом с `prune`), `models/README.md`.
Шаги:
1. `ollama list` и `du -sh models/hf/*` → таблица «что есть, размер» в отчёт (до).
2. Список от исследователя (2026-10-17, после D-018/D-019 — окончательный для режима A):
   - **Оставить (ollama)**: `gemma4:e4b-it-qat`, `gemma4:e2b-it-qat` (короткий список A); `gemma3:4b` (прод, пока T-016 не переключил); `qwen3.5:9b` (судья-fallback); `qwen3.5:4b` (эталон скорости в bench — до bench QAT-вариантов, потом удалить); `ministral-3:8b` (единственный возможный B-lite, до M8b); режим B и coder — `gemma4:26b-a4b-it-*`, `qwen3.6:35b-a3b-*`, `qwen3-coder:30b`; minicpm-v4.6 и Nanbeige — по итогам их прогона (T-018).
   - **Удалить (ollama)**: `gemma4:e2b`, `gemma4:e4b` (base — QAT лучше по персоне), `ministral-3:3b`, `phi4-mini:3.8b`, `smollm2:1.7b`, `hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M`.
   - **HF**: оставить только то, что перечислено в секции `tts` после `hf_inc` + emotion2vec (после `prune` так и есть); `openaudio-s1-mini` — пустая папка (gated), удалить.
   Если к моменту старта в карточке обновлённый список — брать его.
3. Режим `clean` в `fetch_models.sh`: `ollama rm` по списку «удалить», `rm -rf` лишних папок `models/hf`; список «оставить» — массив в начале скрипта с комментарием-датой, чтобы следующий раз правился в одном месте.
4. `models/README.md`: таблица актуальных моделей с датой отбора и ссылкой на D-NNN.
Проверка: `ollama list` и `du -sh models/hf` после; `skull.target` поднимается, brain отвечает на `redis-cli xadd events:transcript '*' text "привет"`.
Done: таблица «до → после» в ГБ в отчёте; прод работает.
Записать: строка в «Закрытые» с цифрами; `models/README.md`.
Связь: D-018, Q-009, T-018, T-013, `fetch_models.sh prune`.

### T-020 · Датасет персоны: генератор через учителя 26B-A4B · open
Зачем: `persona/dataset/` пуст; D-021 показал, что промпт упёрся в потолок (4.08) и что 26B-A4B даёт 4.38 — это готовый учитель для дистилляции. Формат и правила — `persona/dataset/README.md`.
Предусловие: `user_lines.jsonl` хотя бы на 100 реплик (владелец; можно начать с 30 и синтезом ситуаций). Стенд погашен, ollama с `gemma4:26b-a4b-it-mtp-q4_K_M` и `qwen3.5:9b`.
Файлы: создать `research/persona/gen_dataset.py`; читает `persona/dataset/user_lines.jsonl`, пишет `persona/dataset/dialogs.jsonl`, `negatives.jsonl`, `holdout_lines.jsonl`.
Шаги:
1. Для каждой реплики (с контекстом, если есть): запрос к учителю через `/api/chat`, system = `persona/system_prompt.md` (v3), `think: false`, temperature 0.7, 2 сэмпла на реплику.
2. Судья `qwen3.5:9b` по рубрике из `ctx_probe.JUDGE_RUBRIC` (импортировать, не копировать) — предварительный фильтр: ≥4 → `dialogs.jsonl` с полем `judge`, остальное → `negatives.jsonl` с `why`. Финальную оценку выборки делает исследователь (D-017) — скрипт пишет `review_<дата>.md` на 30 случайных записей по образцу `ctx_review.py`.
3. Проверка утечки: нормализованные строки `user_lines` и ответов сравнить с `research/eval/ctx_probe_dialog.jsonl` (реплики + якоря из README п.1) — совпадения отбрасывать и печатать.
4. 10 % `user_lines` (по хэшу id) — в `holdout_lines.jsonl`, в генерацию не идут.
5. Статистика в конце: сколько сгенерировано / принято / отброшено, распределение тегов, средняя длина ответа.
Проверка: на 30 реплик скрипт даёт ≥20 принятых, `review_*.md` читается, утечек 0.
Done: `dialogs.jsonl` ≥ 300 записей (после пополнения `user_lines`), статистика в отчёте.
Записать: строка в «Закрытые» с числами; `persona/dataset/README.md` — раздел «Состояние» с датой и объёмом.
Связь: D-021, P-008, persona/dataset/README, Q-043, Q-044.

### T-021 · QLoRA smoke-test: Soup на gemma4 e2b-it, конец-в-конец · open
Зачем: проверить стек до того, как датасет вырастет: Soup (P-008) на Gemma 4, память на 3080 Laptop 8 ГБ, мердж, экспорт в GGUF с vision-башней, загрузка в ollama, прогон CTX-1. Качество не цель.
Предусловие: T-020 дал ≥50 записей (или временно 50 реплик учителя без судьи — пометить). Сервисный режим стенда.
Файлы: создать `research/persona/train_smoke.md` (как запускалось, версии, память, время), `research/persona/soup.yaml`.
Шаги:
1. HF-чекпойнт bf16 `google/gemma-4-e2b-it` (имя уточнить на HF; не QAT-GGUF — QLoRA квантует сама, Q-043), `hf_inc` только safetensors/конфиги.
2. Soup: QLoRA r=16, target — attention + MLP языковой части, vision-башня заморожена; 1 эпоха, 50 примеров; записать пик VRAM (`nvidia-smi --loop-ms=500`), время эпохи, работает ли layer streaming на Gemma 4 (PLE!).
3. Мердж LoRA в bf16 → `convert_hf_to_gguf.py` (llama.cpp) → Q4_K_M; проверить, что mmproj (vision) собирается или что текстовая часть работает без него; `ollama create kostyan-smoke -f Modelfile`.
3b. ~~(Q-047) без LoRA~~ — сделано в T-023 (D-023): QAT терять можно, **imatrix обязателен**. Для мерджа: `research/quant/quant_e4b.sh` как образец (imatrix на `research/quant/calib_ru.txt` + `persona/dataset/dialogs.jsonl`), формат — IQ4_XS и Q4_K_M, оба с imatrix.
4. `ctx_probe.py --model kostyan-smoke --ctx 8192 --persona persona/system_prompt_rules.md --judge qwen3.5:9b` — сравнить с e2b-it-qat v1 (4.0) только как проверку, что модель не сломана.
Проверка: модель отвечает по-русски через ollama; JSON прогона есть.
Done: `train_smoke.md` с числами (VRAM пик, время, размер GGUF, персона); если стек не завёлся — где именно и что пробовали.
Записать: MEMORY_BUDGET §1 — строка «QLoRA на стенде: VRAM пик»; DECISION_LOG — предложение D-NNN по стеку (исследователь); строка в «Закрытые».
Связь: P-008, Q-043 (QAT vs bf16), Q-044 (стек под Gemma 4), T-020.

### T-022 · Dense 9B при Q3/IQ3 как мозг режима A · done (2026-10-19, на шаге 1)
Шаг 1 показал персону 2.77 / 2.92 — ниже порога 4.3 и ниже 4B-QAT; шаги 2–5 не выполнялись (D-022). Скачанные Q3-варианты, если докачались, — удалить в T-019. Ниже — исходная карточка.
Зачем: Q4 9B не влезает в режим A (D-015: 6.3 ГБ + голос), а 4B-класс упёрся в персону 4.08. Глубокая квантизация 9B (4.0–4.7 ГБ весов) может дать «9B в бюджете 4B» — если персона 9B того стоит и просадка от Q3 мала. На стенде, стенд погашен.
Файлы: ничего не править; результаты в `research/eval/results/` и `research/bench/results/`.
Шаги:
1. **Есть ли что терять**: `for s in 42 7; do .venv/bin/python research/eval/ctx_probe.py --model qwen3.5:9b --ctx 8192 --seed $s 2>&1 | tee -a research/eval/results/ctx_probe_t022.log; done` — без `--judge` (судья — та же модель; оценка — исследователь по JSON). Если персона < 4.3 на обоих seed — остановиться, записать, Q-046 закрыть отрицательно.
2. `ollama pull hf.co/unsloth/Qwen3.5-9B-GGUF:Q3_K_M` и `ollama pull hf.co/unsloth/Qwen3.5-9B-GGUF:UD-IQ3_XXS` (4.7 и 4.0 ГБ).
3. Bench: `MODELS="hf.co/unsloth/Qwen3.5-9B-GGUF:Q3_K_M hf.co/unsloth/Qwen3.5-9B-GGUF:UD-IQ3_XXS qwen3.5:9b" CTXS="2048 8192 16384" KVS="q8_0" FLASHES="1" bash research/bench/bench_vlm_mem.sh research/bench/results/vlm_mem_9b_q3_$(date +%F).csv 2>&1 | tee research/bench/results/bench_9b_q3.log`.
4. CTX-1 на обоих Q3 (2 seed, @8192, без `--judge`), те же команды, что в п.1, с `--model hf.co/…`.
5. Сообщить исследователю список новых JSON — оценка персоны и сравнение с Q4.
Проверка: CSV с 9 строками, 6 JSON CTX-1 (или 2, если остановились на п.1).
Done: таблица «квантизация × (пик VRAM @8k, tok/s, префилл, персона 2 seed, recall, L2)»; гейт Q-046: пик ≤ 5.0 ГБ @8k при просадке персоны ≤ 0.2 к Q4.
Записать: MEMORY_BUDGET §3 — строка «Qwen3.5 9B Q3/IQ3»; сводка `research/bench/results/vlm_mem_9b_q3_<дата>.md` (исследователь); D-NNN по результату (исследователь); строка в «Закрытые».
Связь: Q-046, Q-039, D-015, P-012 (тернарная Bonsai — следующая ступень той же идеи), Q-004 (IQ-ядра на Orin медленнее — перепроверить там).

### T-023 · Своя квантизация gemma-4-E4B-it с imatrix на русском против официальной QAT · done (2026-10-19, D-023)
Сделано владельцем за день (сборка llama.cpp CPU, BF16 через aria2c после отказа Xet-докачки, imatrix, три квантизации, bench, CTX-1 ×6); два бага скриптов ядра поправлены по ходу (`--chunk` → `-c`, f-string). Остаток: подключить mmproj к `kostyan-e4b-iq4xs-imx` и снять пик VRAM с проектором (строка в MEMORY_BUDGET) — в T-024 заодно. Ниже — исходная карточка.
Зачем: Q-047/Q-043 — что именно даёт официальный `-it-qat` и можно ли его терять при LoRA (T-021 квантует сама). Три своих варианта: Q4_K_M с imatrix, IQ4_XS с imatrix (компактнее), Q4_K_M без imatrix — разделяют эффект imatrix и эффект QAT. Конвертация не нужна: `ggml-org/gemma-4-E4B-it-GGUF` содержит BF16 GGUF.
Предусловие: стенд погашен; системный ollama с `gemma4:26b-a4b-it-mtp-q4_K_M` (для калибровочного текста); `sudo apt install -y cmake build-essential`; ~20 ГБ места; желательно `.wslconfig memory` ≥ 24 ГБ (BF16 16 ГБ через mmap при 19 ГБ RAM — впритык).
Файлы: `research/quant/make_calib.py`, `research/quant/quant_e4b.sh` (есть); результаты — `models/quant/` (не в git), `research/quant/calib_ru.txt` (в git — он же seed для T-020), сводка — исследователь.
Шаги:
1. Калибровочный текст: `.venv/bin/python research/quant/make_calib.py --n 60` → `research/quant/calib_ru.txt` (~10 мин; 26B пишет сценки на темы вне eval-диалога, утечки отбрасываются). Посмотреть глазами 2–3 сценки.
2. `bash research/quant/quant_e4b.sh` — сборка llama.cpp (CPU), загрузка BF16, imatrix (10–30 мин на CPU), три квантизации, `ollama create`. Записать из вывода: время imatrix и maxrss, время и размер каждой квантизации.
3. Bench: `MODELS="kostyan-e4b-q4km-imx kostyan-e4b-iq4xs-imx kostyan-e4b-q4km gemma4:e4b-it-qat" CTXS="8192" KVS="q8_0" FLASHES="1" bash research/bench/bench_vlm_mem.sh research/bench/results/vlm_mem_ownquant_$(date +%F).csv 2>&1 | tee research/bench/results/bench_ownquant.log`.
4. CTX-1, два seed: `for m in kostyan-e4b-q4km-imx kostyan-e4b-iq4xs-imx kostyan-e4b-q4km; do for s in 42 7; do .venv/bin/python research/eval/ctx_probe.py --model "$m" --ctx 8192 --seed $s --judge qwen3.5:9b 2>&1 | tee -a research/eval/results/ctx_probe_t023.log; done; ollama stop "$m"; done` (эталон — `ctx_probe_gemma4_e4b-it-qat_8192.json` / `_s7_`, v3).
5. Сообщить исследователю список новых JSON.
Проверка: `ollama list | grep kostyan-e4b` — три модели; CSV 4 строки; 6 JSON.
Done: таблица «вариант × (размер GGUF, пик VRAM @8k, tok/s, персона 2 seed, recall, L2)» — исследователь; гейт Q-047: |Δ персона| ≤ 0.1 к QAT.
Записать: MEMORY_BUDGET §3 — строка «E4B своя квантизация»; D-NNN (исследователь); `models/README.md` — раздел `models/quant/`; строка в «Закрытые» с временами и размерами.
Связь: Q-047, Q-043, T-021 (шаг 3b), T-020 (calib_ru как seed user_lines), Q-004 (IQ-ядра на Orin — отдельно).

### T-024 · ollama против llama-server на одном GGUF · done (2026-10-19, D-024)
Сделано владельцем (CUDA toolkit, сборка, четыре прогона); три бага скрипта ядра по ходу: подоболочка теряла pid сервера, f-string, голый промпт без шаблона + thinking по умолчанию. Остаток (не блокирует): перезапуск `probs` с исправленным парсером (`top_logprobs`) и шаг 1b (mmproj → пик VRAM с проектором) — в T-016 при переезде brain. Ниже — исходная карточка.
Зачем: решение A-03 (рантайм) принято без замера; ollama для Gemma 4 использует собственный раннер — надбавка ~1 ГБ (D-015) может быть другой у llama-server; у llama-server есть logprobs (Q-034) и slot save/restore (freeze/thaw для M8 — `thaw_budget_s` в манифестах пока «ожидание»). Стенд.
Предусловие: T-023 (llama.cpp уже склонирован в `~/llama.cpp`); CUDA toolkit для WSL (`cuda-toolkit-12-x` из репозитория NVIDIA, без драйвера) → пересобрать с `-DGGML_CUDA=ON --target llama-server`; одинаковый GGUF — взять `kostyan-e4b-q4km-imx.gguf` из T-023 (он и в ollama, и файл на диске) или вытащить blob `gemma4:e4b-it-qat` из `/usr/share/ollama/.ollama/models/blobs` по `ollama show --modelfile`.
Файлы: `research/bench/bench_llamaserver.sh` (есть, 2026-10-19: режимы `bench` / `slots` / `probs`; GGUF по умолчанию — `models/quant/kostyan-e4b-iq4xs-imx.gguf`, он же в ollama как `kostyan-e4b-iq4xs-imx`), результаты в `research/bench/results/`.
Команды (стенд погашен, `ollama ps` пуст):
```bash
# CUDA toolkit для WSL (один раз, ~3 ГБ; драйвер уже на Windows-стороне) и CUDA-сборка в отдельную папку
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb && sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update && sudo apt install -y cuda-toolkit-12-8
export PATH=/usr/local/cuda/bin:$PATH
cmake -S ~/llama.cpp -B ~/llama.cpp/build-cuda -DGGML_CUDA=ON -DLLAMA_CURL=OFF && cmake --build ~/llama.cpp/build-cuda -j$(nproc) --target llama-server
# 1. память/скорость: llama-server, затем ollama на том же GGUF (одна таблица)
bash research/bench/bench_llamaserver.sh bench research/bench/results/runtime_$(date +%F).csv
MODELS="kostyan-e4b-iq4xs-imx" CTXS="2048 8192 16384" KVS="q8_0" FLASHES="1" bash research/bench/bench_vlm_mem.sh research/bench/results/runtime_ollama_$(date +%F).csv
# 1b. с проектором зрения (ответ на «+1 ГБ»): mmproj из того же репо ggml-org
aria2c -c -x 4 -d models/quant -o mmproj-gemma-4-E4B-it-Q8_0.gguf https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/mmproj-gemma-4-E4B-it-Q8_0.gguf
MMPROJ=models/quant/mmproj-gemma-4-E4B-it-Q8_0.gguf CTXS="8192" bash research/bench/bench_llamaserver.sh bench research/bench/results/runtime_$(date +%F).csv
# 2. slot save/restore (thaw_budget) и 3. n_probs
bash research/bench/bench_llamaserver.sh slots 2>&1 | tee research/bench/results/runtime_slots.log
bash research/bench/bench_llamaserver.sh probs 2>&1 | tee research/bench/results/runtime_probs.log
```
Если `cuda-toolkit-12-8` нет в репозитории — `apt-cache search cuda-toolkit | tail -3` и взять старший 12.x. Если `-fa on` не принимается старой сборкой — `FA=1`.
Шаги:
1. Память/скорость @2k/8k/16k на обоих рантаймах, один GGUF, flash + q8_0: пик VRAM, RSS, загрузка, TTFT тёплый, префилл, tok/s — одна таблица.
2. Slot save/restore: `llama-server --slot-save-path models/quant/slots`; прогнать 8k контекста (диалог CTX-1 через `/completion` или 40 реплик `/v1/chat/completions`), затем `POST /slots/0?action=save` → размер файла и время; перезапустить сервер, `action=restore` → время до первого токена. Это `freeze_s`/`thaw_budget_s` для манифеста brain.
3. Логпробы: `/completion` с `n_probs: 5` на вопросе «да/нет» — убедиться, что распределение по первому токену доступно (для Q-034), записать латентность одного прохода на 225-токенном промпте.
4. (опц.) `llama-server` с двумя моделями (`--model` ×2 / router) — работает ли multi-model в текущей сборке (TTS_TRACK §6, Q-019).
Проверка: CSV с ≥6 строк (2 рантайма × 3 окна); файл слота и два времени; JSON с `n_probs`.
Done: таблица рантайм × окно; `thaw_budget_s` и размер KV-слота для 8k; факт наличия logprobs и multi-model.
Записать: MEMORY_BUDGET §1 — строка VLM с обоими рантаймами; сводка `research/bench/results/runtime_<дата>.md`; D-NNN по A-03 (исследователь); `services/brain/manifest.yaml` — `thaw_budget_s` (T-016); строка в «Закрытые».
Связь: Q-048, A-03, Q-034, Q-019, M8, D-015.

### T-008 · Оценочный набор `research/eval/` · blocked
Blocked (2026-10-12): нужен текст реплик от владельца (Q-029). Черновик может подготовить исследователь — тогда владелец правит.
Зачем: долг Фазы 0 — набор для сравнения кандидатов: 20 диалоговых промптов, 15 tool-calling кейсов, 1 длинный диалог (CTX-1 уже есть и закрывает «длинный диалог»); `run_eval.py` гонит через ollama и пишет JSON.
Файлы: `research/eval/dialog_prompts.jsonl`, `research/eval/tool_calls.jsonl`, `research/eval/run_eval.py`.
Шаги: после снятия блокировки — формат из `research/eval/README.md`; `run_eval.py` по образцу `ctx_probe.py` (те же seed, температура, запись в `results/`).
Проверка: `.venv/bin/python research/eval/run_eval.py --model gemma3:4b` → `results/eval_gemma3_4b_<дата>.json`.
Done: набор прогоняется на gemma3:4b, ответы сохранены.
Записать: строка в «Закрытые».
Связь: ROADMAP «долг Фазы 0», Q-029, CTX_PROBE.

## Закрытые

| ID | Дата | Результат | Коммит | Записано | Отклонения / вопросы ядру |
|---|---|---|---|---|---|
| T-012 | 2026-10-13 | CTX-1 probe: скрипт, прогон gemma3:4b ×4 окна, судья qwen3.5:9b; num_ctx 8k достаточно, recall 0.8, math 0/3, персона 3.6 | 94cd205, eb2b54e | D-016, `research/eval/results/ctx_probe_summary_2026-10-13.md` | сделано исследователем, не исполнителем |
| T-014 | 2026-10-18 | Промпт v1/v2/v3 × 3 модели × 2 seed: e4b-it-qat 4.08 с оговоркой и выкладкой, gemma3 и e2b от примеров хуже → промпт — параметр модели | c45c793, 8ae1986, 04caf7b | D-020, D-021, `ctx_probe_t014b_2026-10-18.md` | прогоны — владелец, оценки — исследователь |
| T-018 | 2026-10-19 | CTX-1 по 12 моделям A + 2 MoE B: лидер e4b-it-qat 4.08 (26B-A4B — 4.38 потолок); qwen3.5:4b, текстовые, minicpm — сняты | fea56f9…cb8c815 | D-018, D-019, MEMORY_BUDGET §3/§4, сводки в results/ | Nanbeige не скачан → Q-035 |
| T-006 | 2026-10-19 | bench v2.1: load_s, TTFT тёплый, префилл мимо кэша, RSS через pgrep -P, host RAM; префилл 26B MoE 100 ток/с vs dense 1200–2600, генерация 18 vs 42–88 | e885439, 8529a50, e0bf8f3 + следующий | `vlm_mem_prefill_2026-10-19.md`, MEMORY_BUDGET §3/§4 | сделано ядром; дефект v2 (кэш префикса) найден и исправлен в тот же день |
| T-022 | 2026-10-19 | qwen3.5:9b Q4: персона 2.77 / 2.92 (два seed), recall 0.73–0.80, math 1/6 — глубокая квантизация 9B для режима A не нужна | b773ab9 + следующий | D-022, `ctx_probe_t022_2026-10-19.md`, MEMORY_BUDGET §3 | остановлена на шаге 1; прогон — владелец, оценка — исследователь |
| T-023 | 2026-10-19 | Q4_K_M+imx 3.92/3.85, IQ4_XS+imx 3.92/4.0, Q4_K_M без imx 3.38/3.31 против QAT 4.08/4.08; recall 0.87–0.93 с imx; IQ4_XS −250 МБ VRAM при той же скорости | f148fce…805245e + следующий | D-023, `ctx_probe_t023_2026-10-19.md`, MEMORY_BUDGET §3, `research/quant/calib_ru.txt` | VRAM без mmproj несравним с официальной; времена imatrix/quantize — из терминала владельца, не записаны |
| T-024 | 2026-10-19 | llama-server vs ollama на IQ4_XS+imx: VRAM 2984/3040/3148 vs 2974/3030/3138 МБ, 65–68 vs 66–69 tok/s; slot save 62 мс (71 МБ/4.4k), restore 32 мс, thaw ≈ 3.5 с; n_probs 0.3 с | 9033ff9…367b56c + следующий | D-024, `runtime_2026-10-19.md`, MEMORY_BUDGET §1, Q-048/Q-034 | probs-парсер перезапустить; mmproj — в T-016 |
