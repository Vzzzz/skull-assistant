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
deploy/systemd/    юниты с плейсхолдером @REPO@, install.sh; sync-win.sh — зеркало на Windows
research/          bench/ (замеры), eval/ (оценочный набор), tts/ — ничего отсюда не запускается юнитом
docs/              ROADMAP, DECISION_LOG, MEMORY_BUDGET, PUBLICATIONS, ISSUES, TTS_TRACK, roadmap-v1
persona/           system_prompt.md; dataset/ приватный
```

Шина: Redis Streams `audio:mic` → `events:wakeword` → `events:transcript` → `events:reply`; флаг `tts:speaking`. Каждый сервис — отдельный systemd user-юнит; изоляция процессов = механизм свопа моделей (память GPU возвращается только со смертью процесса).

## Правила

- **Docs и decision-log на русском**; код, идентификаторы и комментарии в коде — на английском.
- Любое архитектурное решение — запись в `docs/DECISION_LOG.md` («пробовали → упёрлись → выбрали», с основанием: замер / публикация / субъективно). Старые записи не правятся — меняют статус.
- Бюджеты памяти — **пики**, не установившиеся значения; три точки на сервис (без модели / загружена / после freeze). Запись в `docs/MEMORY_BUDGET.md` обязательна при каждом замере.
- GPU получают только сервисы с тяжёлыми вычислениями (VLM, детектор). Голосовой стек и эмбеддер — CPU-only: каждый CUDA-контекст на процесс стоит сотни МБ из тех же 8 ГБ.
- Инвариант: wake word/KWS не выгружается никогда (с D-013 переезжает на MCU, на Jetson остаётся fallback).
- Публикации разбираются по протоколу `docs/PUBLICATIONS.md`; архитектура меняется только после замера.
- Рабочее железо (4090) и рабочие данные в проекте не участвуют.

## Две копии репозитория

Рабочая — WSL Ubuntu-22.04 `~/skull-assistant` (здесь .venv, модели, юниты). Зеркало — `C:/Users/valer/git/skull-assistant`. Синхронизация только через git: коммит в WSL → `deploy/sync-win.sh`. Юниты установлены в `~/.config/systemd/user/`; статус: `systemctl --user status "skull-*"`.

## Как запускать на стенде

```bash
.venv/bin/pip install -r requirements.txt
deploy/systemd/install.sh
systemctl --user start skull-capture skull-wakeword skull-stt skull-brain skull-tts
```

Нужны запущенные `redis-server` и `ollama` (override: flash attention, KV q8_0, keep_alive -1).

## Ближайшие шаги

См. `docs/ROADMAP.md` §7. Коротко: перезамер памяти VLM (`research/bench/bench_vlm_mem.sh`), быстрые правки из `docs/ISSUES.md` (I-01 двойной transcribe, I-05 история в Redis), старт M5 на Jetson, включая проверку SC7 для D-013.
