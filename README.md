# Skull Assistant

Голосовой ассистент с характером в корпусе анатомического черепа — и платформа для запуска компактных ML-сервисов на одном edge-устройстве (Jetson Orin Nano Super 8 GB).

Проект ведётся как исследовательский: каждое архитектурное решение подкреплено замером, decision-log и таблица бюджетов памяти — часть репозитория, а не приложение к статье.

## Структура

```
services/     прод: один сервис = папка с кодом + manifest.yaml (бюджет, резидентность, lifecycle)
core/         общий код: шина, конфиг, диспетчер режимов (появится в M8)
deploy/       systemd-юниты, установка на стенд и на Jetson
research/     воспроизводимые замеры и эксперименты: bench/, eval/, tts/
docs/         roadmap, decision-log, бюджеты памяти, журнал публикаций
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

## Запуск на стенде (WSL2 / Linux)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# веса: см. models/README.md
deploy/systemd/install.sh      # symlink юнитов в ~/.config/systemd/user + enable
systemctl --user status 'skull-*'
```

Требуется запущенный `redis-server` и `ollama` с моделью из `services/brain/manifest.yaml`.

## Две копии репозитория

| Где | Роль | Путь |
|---|---|---|
| WSL Ubuntu-22.04 | рабочая: отсюда запускаются юниты, здесь .venv и модели | `~/skull-assistant` |
| Windows | зеркало и точка публикации (GitHub позже) | `C:/Users/valer/git/skull-assistant` |

Синхронизация только через git, не копированием: коммит в WSL → `deploy/sync-win.sh` (из WSL) подтягивает зеркало. Правки, сделанные в Windows-копии, забираются обратно `git pull win main` в WSL. Windows-git не открывает WSL-репозиторий по UNC без `safe.directory`, поэтому все sync-команды выполняются из WSL.


## Сервисный режим стенда

Иногда WSL нужен без сервисов: скачать модели, обновить пакеты, не включая ollama и тем более захват микрофона. Механизм: флаг-файл на Windows-стороне `%USERPROFILE%\.skull-maintenance` (в Git Bash — `$HOME/.skull-maintenance`, в WSL — `/mnt/c/Users/valer/.skull-maintenance`). Все `skull-*` юниты и `skull.target` имеют `ConditionPathExists=!<флаг>`, системный `ollama.service` — такой же drop-in (`deploy/systemd/install-ollama-dropin.sh`, нужен sudo). Флаг ставится и снимается при выключенном WSL, гонки со стартом нет.

Из Git Bash на Windows (`deploy/stand/skull-stand.sh`):

```bash
deploy/stand/skull-stand.sh service   # флаг + WSL без сервисов
deploy/stand/skull-stand.sh normal    # снять флаг + WSL с сервисами
deploy/stand/skull-stand.sh status
deploy/stand/skull-stand.sh stop      # остановить WSL целиком
```

Внутри WSL весь стенд управляется одним target: `systemctl --user start|stop skull.target`, статус `systemctl --user list-units 'skull*'`.

## Статус

См. `docs/ROADMAP.md` §0. Голосовое ядро (wake word → STT → VLM → TTS) работает на стенде; следующий этап — замеры на устройстве (M5).

## Лицензия

TBD — выбрать до публикации (MIT для кода; веса моделей и голосов — по их собственным лицензиям, см. `models/README.md`).
