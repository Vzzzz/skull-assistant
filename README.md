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

## Статус

См. `docs/ROADMAP.md` §0. Голосовое ядро (wake word → STT → VLM → TTS) работает на стенде; следующий этап — замеры на устройстве (M5).

## Лицензия

TBD — выбрать до публикации (MIT для кода; веса моделей и голосов — по их собственным лицензиям, см. `models/README.md`).
