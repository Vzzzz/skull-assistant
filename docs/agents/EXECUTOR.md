# Онбординг: исполнитель

Ты выполняешь задачи из `docs/TASKS.md`: код в `services/`, `core/`, `deploy/`, скрипты и прогоны в `research/`. Ты работаешь в WSL Ubuntu-22.04, репозиторий `~/skull-assistant`, Python — `.venv/bin/python`. Решения о том, *что* делать, принимает исследователь; твоя часть — сделать, проверить, записать результат. Если для выполнения нужно решение, которого нет в карточке, — не придумывай его: ставь статус `blocked` и пиши вопрос (см. §6).

## 1. Начало сессии

1. Прочитай `AGENTS.md` (структура, правила) и этот файл.
2. Открой `docs/TASKS.md`. Возьми **верхнюю задачу со статусом `open`** в индексе, если владелец не назвал другую. Прочитай её карточку целиком.
3. Прочитай файлы, названные в карточке в поле «Файлы». Другие документы — только те, что названы в «Связь», и только если карточки не хватает.
4. Поставь статус `wip` в индексе TASKS.md, создай ветку (источник правды — GitHub `origin`, D-от 2026-10-16):
   ```bash
   git checkout main && git pull --ff-only origin main && git checkout -b feat/T-NNN-short-slug
   ```
5. Проверь состояние стенда (§2).

## 2. Стенд: что проверить перед запуском

```bash
ls /mnt/c/Users/<user>/.skull-maintenance 2>/dev/null && echo "СЕРВИСНЫЙ РЕЖИМ: юниты и ollama не стартуют сами" || echo "обычный режим"
systemctl --user list-units 'skull*' --no-pager      # какие юниты живы
systemctl is-active ollama; ollama ps                # что держит GPU
redis-cli ping
```

Правила:
- **Замеры памяти GPU** — только при погашенном стенде: `systemctl --user stop skull.target`, `ollama ps` пуст. После замера вернуть: `systemctl --user start skull.target`.
- Не держать две модели в ollama одновременно, если задача не требует; llama-server и ollama с моделью одновременно на GPU не запускать (8 ГБ).
- Загрузка моделей (`deploy/fetch_models.sh`, `ollama pull`) — в сервисном режиме; флаг ставится из Git Bash на Windows (`deploy/stand/skull-stand.sh service`), тебе достаточно проверить, что он есть.
- Микрофон и динамик — это WSLg/PulseAudio; при проверках, где нужно говорить и слушать, так и сказано в карточке.
- Сервисы читают код из репозитория напрямую (юниты запускают `@REPO@/services/...`): после правки — `systemctl --user restart skull-<name>`, лог — `journalctl --user -u skull-<name> -f`.

## 3. Что можно и что нельзя править

| Можно | Нельзя без явного указания в карточке |
|---|---|
| `services/`, `core/`, `deploy/`, `research/**/*.py`, `research/**/*.sh`, `requirements.txt` | `docs/ROADMAP.md`, `docs/DECISION_LOG.md`, `docs/PUBLICATIONS.md`, `docs/JOURNAL.md`, `docs/OPEN_QUESTIONS.md`, `docs/agents/` |
| `docs/TASKS.md` — только статус в индексе и строка в «Закрытые» | другие карточки задач |
| `docs/ISSUES.md`, `docs/MEMORY_BUDGET.md` §1 — только строки, названные в поле «Записать» | остальные строки и разделы |
| `research/**/results/` — CSV/JSON прогонов и короткая сводка `.md`, если карточка просит | выводы и решения (это делает исследователь) |
| `services/*/manifest.yaml` — поля, названные в карточке | `persona/system_prompt.md`, `persona/dataset/` |

Не трогать: `AGENTS.md`, `CLAUDE.md`, `.gitignore` (если задача не про них), чужие ветки. Секреты и токены не коммитить.

## 4. Как выполнять

- Иди по шагам карточки. Если шаг невыполним как написано (API не такой, файла нет) — сделай ближайшее разумное и **опиши отклонение в отчёте**. Если отклонение меняет смысл задачи — остановись, `blocked`, вопрос.
- Правки минимальные и в стиле соседнего кода: комментарии в коде на английском, без переименований «заодно», без новых зависимостей, если карточка их не называет. Новая зависимость — строка в `requirements.txt` с версией.
- Запусти проверку из поля «Проверка» и сохрани её вывод (числа, строки лога) — они пойдут в отчёт.
- Замер = числа + условия (что было запущено, какая модель, сколько повторов). Один прогон — не замер; минимум то, что просит карточка.
- Не чини то, чего нет в карточке. Заметил проблему — строка в отчёте «замечено: …», исследователь решит.

## 5. Конец задачи

1. Коммит в ветке, сообщение: `T-NNN: <что сделано одной строкой>` + при необходимости абзац с числами. Мерж в `main` и push (исследователь параллельно коммитит docs в GitHub — поэтому сначала `pull --rebase`):
   ```bash
   git checkout main && git merge --no-ff feat/T-NNN-short-slug && git pull --rebase origin main && git push origin main
   ```
   Если `pull --rebase` ругается на незакоммиченные файлы — сначала закоммитить (результаты прогонов тоже), потом rebase.
2. В `docs/TASKS.md`: статус `done` в индексе; строка в «Закрытые» по формату §6.
3. Записи, названные в поле «Записать» карточки (ISSUES, MEMORY_BUDGET §1, сводка в results). Без них задача **не закрыта**, даже если код работает.
4. Если стенд был погашен для замера — поднять обратно.

## 6. Формат отчёта (строка в «Закрытые» TASKS.md)

```
| T-NNN | 2026-MM-DD | <результат одной строкой, с числами: было X → стало Y> | <коммит hash> | <куда записано: I-01 / MEMORY_BUDGET §1 / results/…> | <отклонения от шагов; «замечено: …»; «вопрос ядру: …» или «—»> |
```

Для `blocked`: статус в индексе + строка в «Закрытые» не нужна; вместо неё — в карточке задачи под заголовком добавить строку `Blocked (дата): <что именно нужно решить и кто>`.

## 7. Полезное

```bash
# стенд
systemctl --user start|stop|restart skull.target
systemctl --user restart skull-brain && journalctl --user -u skull-brain -f
redis-cli xlen audio:mic; redis-cli xrevrange events:transcript + - COUNT 3
# замеры
research/bench/measure_voice_rss.sh            # RSS/CPU пяти юнитов (гасит стенд в конце)
bash research/bench/bench_vlm_mem.sh out.csv   # память VLM; поднимает свой ollama на :11435
.venv/bin/python research/tts/piper_rt.py      # RT Piper
.venv/bin/python research/eval/ctx_probe.py --model gemma4:e4b-it-qat --ctx 8192   # CTX-1 через ollama
# llama-server (прод-рантайм brain по D-024; сборка ~/llama.cpp/build-cuda, CUDA toolkit в /usr/local/cuda — PATH!)
~/llama.cpp/build-cuda/bin/llama-server -m models/quant/kostyan-e4b-iq4xs-imx.gguf --mmproj models/hf/gemma-4-E4B-it-GGUF/mmproj-gemma-4-E4B-it-Q8_0.gguf -ngl 99 -c 8192 -fa on -ctk q8_0 -ctv q8_0 --port 11437 --slot-save-path models/quant/slots
curl -s localhost:11437/health                 # {"status":"ok"}
.venv/bin/python research/eval/ctx_probe.py --backend llamaserver --host http://127.0.0.1:11437 --model kostyan --ctx 8192   # CTX-1 против llama-server
bash research/bench/bench_llamaserver.sh bench|slots|probs   # память/скорость, slot save/restore, logprobs
# синхронизация: источник правды — GitHub
git pull --ff-only origin main                 # перед началом работы (правки docs исследователя)
git push origin main                           # после мержа; при отказе — git pull --rebase origin main и снова push
```

Судья персоны: `--judge qwen3.5:9b` в `ctx_probe.py` **не указывать** (D-028, 2026-10-23) — оценку делает исследователь по JSON; парные сравнения — `research/eval/judge_pairs.py` на 26B (запускает исследователь).

Прототипы для T-016 (переносятся из research в core): `research/memory/assoc_memory.py` (память за окном + фильтр `clean_reply` + справка v6), `research/memory/router.py` (роутер типов, `POLICY`, `route_llamaserver`), эталон поведения — `research/eval/ctx_probe.py run_dialog` (окно по токенам, порядок: окно → выборка → сборка сообщения → фильтр → история/память).

Загрузка моделей с HF (2026-10-23, владелец): `hf download` через Xet стартует, но скорость падает до сотен КБ/с и закачка рвётся; с `HF_HUB_DISABLE_XET=1` не стартует вовсе. **Штатный путь для файлов > 1 ГБ — aria2c**: `aria2c -x 8 -s 8 -c -d <dir> -o <file> https://huggingface.co/<repo>/resolve/main/<file>` (докачка по `.aria2`); `hf download` — только для мелких файлов (конфиги, проекторы до ~0.5 ГБ).
