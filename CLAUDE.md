# skull-assistant: контекст для агентов

Читается Claude Code автоматически. Достаточно, чтобы начать без истории чатов; подробности в `docs/`.

## Что это

Главный проект портфолио Валерия: офлайн голосовой ассистент с характером (**Костян**, анатомический череп) на Jetson Orin Nano Super 8 GB — и платформа для запуска компактных ML-сервисов на одном устройстве. Три результата: R1 устройство, R2 платформа (диспетчер режимов A/B/C, манифесты сервисов), R3 публикации на Хабре. Смежный проект: `bitladder` (`C:/Users/valer/git/bitladder`) — квантование до 1 бита и wake-word триггер на MCU, который будит спящий Jetson (D-013).

## Состояние (2026-10-08)

Голосовое ядро работает на стенде (WSL2, RTX 3080 Laptop): wake word → STT → VLM (ollama gemma3:4b) → TTS (Piper). M4 (диалоговый режим) сделан частично: нет стриминга TTS и barge-in. Jetson и NVMe 512 ГБ на руках, следующий этап — **M5: замеры на устройстве**. План и статусы — `docs/ROADMAP.md` §0 и §7.

## Структура

```
services/<name>/   прод: код + manifest.yaml (бюджет по пикам, резидентность A/B/C, freeze/thaw, топики)
core/              общий код; диспетчер появится в M8
deploy/systemd/    skull.target + юниты (@REPO@), install.sh, drop-in ollama; deploy/stand/ — запуск стенда из Git Bash; sync-win.sh — зеркало
research/          bench/ (замеры), eval/ (оценочный набор), tts/ — ничего отсюда не запускается юнитом
docs/              ROADMAP, DECISION_LOG, MEMORY_BUDGET, PUBLICATIONS, ISSUES, TASKS, UX_BACKLOG, TTS_TRACK, AMBIENT_MODE, JETSON_BRINGUP, roadmap-v1
persona/           system_prompt.md; dataset/ приватный
```

Шина: Redis Streams `audio:mic` → `events:wakeword` → `events:transcript` → `events:reply`; флаг `tts:speaking`. Каждый сервис — отдельный systemd user-юнит; изоляция процессов = механизм свопа моделей (память GPU возвращается только со смертью процесса).

## Протокол работы агентов

Две роли: **ядро** (сессия планирования: архитектура, публикации, docs, обзор результатов) и **исполнитель** (VS Code + Claude Code в WSL: код, отладка, замеры). Единственный канал между ними — git.

- Исполнитель: начало сессии — прочитать этот файл и `docs/TASKS.md`, взять верхнюю открытую задачу (или названную). Конец — коммит в ветке `feat/T-NNN-…`, мерж в `main`, статус в TASKS.md, запись в DECISION_LOG / MEMORY_BUDGET / ISSUES, если что-то выбрано или измерено. Без записи задача не закрыта.
- Владение файлами: исполнитель — `services/`, `core/`, `deploy/`, скрипты в `research/`; ядро — `docs/`, сводки `research/**/results/*.md`. Чужое не править без задачи.
- Стенд перед работой: `deploy/stand/skull-stand.sh status`; замеры GPU — при погашенном `skull.target`.

## Правила

- **Docs и decision-log на русском**; код, идентификаторы и комментарии в коде — на английском.
- Любое архитектурное решение — запись в `docs/DECISION_LOG.md` («пробовали → упёрлись → выбрали», с основанием: замер / публикация / субъективно). Старые записи не правятся — меняют статус.
- Бюджеты памяти — **пики**, не установившиеся значения; три точки на сервис (без модели / загружена / после freeze). Запись в `docs/MEMORY_BUDGET.md` обязательна при каждом замере.
- GPU получают только сервисы с тяжёлыми вычислениями (VLM, детектор). Голосовой стек и эмбеддер — CPU-only: каждый CUDA-контекст на процесс стоит сотни МБ из тех же 8 ГБ.
- Инвариант: wake word/KWS не выгружается никогда (с D-013 переезжает на MCU, на Jetson остаётся fallback).
- Публикации разбираются по протоколу `docs/PUBLICATIONS.md`; архитектура меняется только после замера.
- Основная консоль пользователя на Windows — **Git Bash**, не PowerShell. Рабочее железо (4090) и рабочие данные в проекте не участвуют.

## Две копии репозитория

Рабочая — WSL Ubuntu-22.04 `~/skull-assistant` (здесь .venv, модели, юниты). Зеркало — `C:/Users/valer/git/skull-assistant`. Синхронизация только через git: коммит в WSL → `deploy/sync-win.sh`. Юниты установлены в `~/.config/systemd/user/`; статус: `systemctl --user status "skull-*"`.

## Стенд: обычный и сервисный режим

```bash
.venv/bin/pip install -r requirements.txt
deploy/systemd/install.sh && deploy/systemd/install-ollama-dropin.sh
systemctl --user start skull.target        # stop — погасить всё
```

При старте WSL `skull.target` поднимается сам. Чтобы WSL стартовал **без** сервисов и ollama (скачать модели, замеры), ставится флаг `$HOME/.skull-maintenance` на Windows-стороне — юниты имеют `ConditionPathExists=!` на него. Из Git Bash: `deploy/stand/skull-stand.sh service|normal|status|stop`. Перед любым действием, которое не должно слышать микрофон или занимать GPU, проверять `skull-stand.sh status`.

Нужны запущенные `redis-server` и `ollama` (override: flash attention, KV q8_0, keep_alive -1).

## Ближайшие шаги

Очередь исполнителя — `docs/TASKS.md` (T-001…). План — `docs/ROADMAP.md` §7: после задач T-001…T-005 — старт M5 на Jetson, включая проверку SC7 для D-013. Замер памяти VLM на стенде сделан 2026-10-12 (`research/bench/results/`, D-015).
