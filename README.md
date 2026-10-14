# Skull Assistant

Голосовой ассистент с характером в корпусе анатомического черепа — и платформа для запуска компактных ML-сервисов на одном edge-устройстве (Jetson Orin Nano Super 8 GB).

Проект ведётся как исследовательский: каждое архитектурное решение подкреплено замером, decision-log и таблица бюджетов памяти — часть репозитория, а не приложение к статье.

## Структура

Для агентов (любой инструмент): `AGENTS.md` → `docs/agents/RESEARCHER.md` или `EXECUTOR.md` по роли; состояние — `docs/JOURNAL.md`, очередь — `docs/TASKS.md`, нерешённое — `docs/OPEN_QUESTIONS.md`.

```
AGENTS.md     контекст и протокол ролей для агентов (CLAUDE.md — обёртка для Claude Code)
services/     прод: один сервис = папка с кодом + manifest.yaml (бюджет, резидентность, lifecycle)
core/         общий код: шина, конфиг, диспетчер режимов (появится в M8)
deploy/       systemd/ — skull.target + юниты, install.sh, drop-in для ollama; stand/ — запуск стенда из Git Bash (обычный и сервисный режим); sync-win.sh — зеркало
research/     воспроизводимые замеры и эксперименты: bench/, eval/, tts/
docs/         журнал сессий, очередь задач, открытые вопросы, roadmap, decision-log, бюджеты памяти, журнал публикаций; agents/ — онбординг ролей
persona/      system-prompt и материалы персоны (датасет — см. persona/README)
models/       веса (не в git) + README с источниками
```

Граница «прод / ресёрч»: в `services/` живёт только то, что запускается юнитом на устройстве. Всё, что измеряет, сравнивает или обучает — в `research/`. Результат ресёрча попадает в прод через запись в `docs/DECISION_LOG.md` и правку манифеста.

## Архитектура (кратко)

- Шина: Redis Streams (`audio:mic` → `events:wakeword` → `events:transcript` → `events:reply`), состояние — SQLite.
- Каждый сервис — отдельный systemd-юнит; изоляция процессов = механизм свопа моделей.
- Голосовой тракт CPU-only (openWakeWord, faster-whisper, Piper); VLM 4B INT4 через llama.cpp/Ollama; зрение — YOLO как триггер, VLM по событию.
- Три режима: A — VLM 4B + полный сенсорный стек; B — большая MoE со стримингом экспертов с NVMe; C — «трофей». Инвариант: wake word/KWS не выгружается никогда.

Подробно: `docs/ROADMAP.md`, исходный план — `docs/roadmap-v1.md`.

## Стенд (WSL2 / Linux)

Установка один раз:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# веса: см. models/README.md
deploy/systemd/install.sh                 # skull.target + юниты в ~/.config/systemd/user, enable
deploy/systemd/install-ollama-dropin.sh   # условие сервисного режима для ollama (sudo)
```

Требуются `redis-server` и `ollama` с моделью из `services/brain/manifest.yaml`. Весь стенд — один target:

```bash
systemctl --user start skull.target       # поднять все сервисы
systemctl --user stop  skull.target       # погасить все
systemctl --user list-units 'skull*'      # статус
```

### Два режима запуска: обычный и сервисный

При старте WSL `skull.target` поднимается автоматически — микрофон, wake word, STT, VLM, TTS. **Сервисный режим** нужен, когда WSL требуется без всего этого: скачать модели, обновить пакеты, прогнать замер, не трогая микрофон и не держа ollama с моделью в памяти.

Механизм — флаг-файл на Windows-стороне, `$HOME/.skull-maintenance` в Git Bash (в WSL он виден как `/mnt/c/Users/valer/.skull-maintenance`). Все `skull-*` юниты, `skull.target` и системный `ollama.service` имеют `ConditionPathExists=!<флаг>`: есть флаг — systemd их не стартует. Флаг ставится и снимается при выключенном WSL, гонки со стартом нет.

Из Git Bash на Windows, из папки репозитория:

```bash
deploy/stand/skull-stand.sh service   # поставить флаг, открыть WSL без сервисов
deploy/stand/skull-stand.sh normal    # снять флаг, открыть WSL, сервисы поднимаются
deploy/stand/skull-stand.sh status    # флаг, состояние WSL, юнитов и ollama
deploy/stand/skull-stand.sh stop      # остановить WSL целиком
```

В сервисном режиме ollama при необходимости включается вручную (`sudo systemctl start ollama`), например для `ollama pull`. Bench-скрипты поднимают собственный ollama на отдельном порту и системный не требуют.

## Две копии репозитория

| Где | Роль | Путь |
|---|---|---|
| WSL Ubuntu-22.04 | рабочая: отсюда запускаются юниты, здесь .venv и модели | `~/skull-assistant` |
| Windows | зеркало и точка публикации (GitHub позже) | `C:/Users/valer/git/skull-assistant` |

Синхронизация только через git, не копированием: коммит в WSL → `deploy/sync-win.sh` (из WSL) подтягивает зеркало. Правки, сделанные в Windows-копии, забираются обратно `git pull win main` в WSL. Windows-git не открывает WSL-репозиторий по UNC без `safe.directory`, поэтому все sync-команды выполняются из WSL.

## Статус

См. `docs/ROADMAP.md` §0. Голосовое ядро (wake word → STT → VLM → TTS) работает на стенде; следующий этап — замеры на устройстве (M5).

## Лицензия

TBD — выбрать до публикации (MIT для кода; веса моделей и голосов — по их собственным лицензиям, см. `models/README.md`).
