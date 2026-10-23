# Онбординг: исполнитель

Ты выполняешь задачи из `docs/TASKS.md`: код в `services/`, `core/`, `deploy/`, скрипты и прогоны в `research/`. Ты работаешь в WSL Ubuntu-22.04, репозиторий `~/skull-assistant`, Python — `.venv/bin/python`. Решения о том, *что* делать, принимает исследователь; твоя часть — сделать, проверить, записать результат. Если для выполнения нужно решение, которого нет в карточке, — не придумывай его: ставь статус `blocked` и пиши вопрос (см. §6).

## 1. Начало сессии

1. Прочитай `AGENTS.md` (структура, правила) и этот файл.
2. Открой `docs/TASKS.md`. Возьми **верхнюю задачу со статусом `open`** в индексе, если владелец не назвал другую. Прочитай её карточку целиком.
3. Прочитай файлы, названные в карточке в поле «Файлы». Другие документы — только те, что названы в «Связь», и только если карточки не хватает.
4. Поставь статус `wip` в индексе TASKS.md, создай ветку:
   ```bash
   git checkout main && git pull --ff-only win main 2>/dev/null; git checkout -b feat/T-NNN-short-slug
   ```
5. Проверь состояние стенда (§2).

## 2. Стенд: что проверить перед запуском

```bash
ls /mnt/c/Users/valer/.skull-maintenance 2>/dev/null && echo "СЕРВИСНЫЙ РЕЖИМ: юниты и ollama не стартуют сами" || echo "обычный режим"
systemctl --user list-units 'skull*' --no-pager      # какие юниты живы
systemctl is-active ollama; ollama ps                # что держит GPU
redis-cli ping
```

Правила:
- **Замеры памяти GPU** — только при погашенном стенде: `systemctl --user stop skull.target`, `ollama ps` пуст. После замера вернуть: `systemctl --user start skull.target`.
- Не держать две модели в ollama одновременно, если задача не требует.
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

1. Коммит в ветке, сообщение: `T-NNN: <что сделано одной строкой>` + при необходимости абзац с числами. Мерж в `main`:
   ```bash
   git checkout main && git merge --no-ff feat/T-NNN-short-slug && deploy/sync-win.sh
   ```
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
.venv/bin/python research/eval/ctx_probe.py --model gemma3:4b --ctx 8192 --judge qwen3.5:9b
# синхронизация
deploy/sync-win.sh                             # после коммита в WSL: подтянуть Windows-зеркало
git pull win main                              # забрать правки docs, сделанные исследователем в зеркале
```

Если remote `win` не настроен: `git remote add win /mnt/c/Users/valer/git/skull-assistant`.

Загрузка моделей с HF (2026-10-23, владелец): `hf download` через Xet стартует, но скорость падает до сотен КБ/с и закачка рвётся; с `HF_HUB_DISABLE_XET=1` не стартует вовсе. **Штатный путь для файлов > 1 ГБ — aria2c**: `aria2c -x 8 -s 8 -c -d <dir> -o <file> https://huggingface.co/<repo>/resolve/main/<file>` (докачка по `.aria2`); `hf download` — только для мелких файлов (конфиги, проекторы до ~0.5 ГБ).
