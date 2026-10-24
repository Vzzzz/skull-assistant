# Результаты: что измерено и что из этого решено

Одна страница на все замеры проекта. Каждая строка — что измеряли, главное число, где сводка с таблицами и условиями, какое решение из этого вышло. Все числа — **стенд** (RTX 3080 Laptop 8 ГБ, WSL2), если не сказано иное; на Jetson Orin Nano перемеряются (M5). Сырые CSV/JSON лежат рядом со сводками в `research/*/results/`; скрипты — в `research/bench/`, `research/eval/`, `research/tts/`. Бюджеты памяти по сервисам сведены в [MEMORY_BUDGET.md](MEMORY_BUDGET.md), решения — в [DECISION_LOG.md](DECISION_LOG.md).

## 1. Мозг: выбор модели по персоне и памяти (CTX-1)

CTX-1 — один диалог на 60 реплик с якорями: факты о собеседнике, которые надо вспомнить через 10–50 ходов, три арифметические задачи, вопрос о том, чего модель видеть не могла (L2: выдумает или оговорится). Персона оценивается по 13 репликам (шкала 1–5). Протокол — [`research/eval/CTX_PROBE.md`](../research/eval/CTX_PROBE.md).

| Что измеряли | Главное | Сводка | Решение |
|---|---|---|---|
| 14 моделей 1.7–9B как мозг режима A | Gemma 4 E4B-it-qat: персона 4.08, recall 0.93 @8k, оговаривается на L2; E2B 3.3; Qwen3.5-4B 1.9 (русский/персона не держит); gemma3:4b выдумывает на L2 | [`ctx_probe_candidates_2026-10-17.md`](../research/eval/results/ctx_probe_candidates_2026-10-17.md), [`…10-16.md`](../research/eval/results/ctx_probe_candidates_2026-10-16.md), [`ctx_probe_summary_2026-10-13.md`](../research/eval/results/ctx_probe_summary_2026-10-13.md) | D-016…D-022: E4B основной, E2B запасной |
| Своя квантизация E4B: IQ4_XS + imatrix на русском калибровочном наборе | recall 0.87, персона 4.15, 66 tok/s при **3040 МБ VRAM @8k** против 4258 у QAT Q4_K_M | [`ctx_probe_kostyan-e4b-iq4xs-imx_2026-10-19.md`](../research/eval/results/ctx_probe_kostyan-e4b-iq4xs-imx_2026-10-19.md), скрипт [`research/quant/quant_e4b.sh`](../research/quant/quant_e4b.sh) | D-023: в прод идёт своя квантизация (−1.2 ГБ) |
| Режим thinking (E-1): помогает ли 4B думать | Gemma 4 думает адаптивно сама; принудительное thinking не улучшает память и счёт, а ошибки счёта — это чтение чисел словами | [`ctx_probe_t025_2026-10-19.md`](../research/eval/results/ctx_probe_t025_2026-10-19.md) | D-025: thinking выключен; рычаг — нормализация чисел |
| Память за окном: выпавшие из контекста обмены возвращаются справкой; роутер типов реплик; фильтр ответа | @2k без памяти recall **0.07** → с памятью **0.80–0.93**; шесть форматов справки — формат определяет, обсуждает ли модель память вслух; роутер 0.88–0.93 | [`ctx_probe_t030_2026-10-23.md`](../research/eval/results/ctx_probe_t030_2026-10-23.md), [`ctx_probe_t026_2026-10-20.md`](../research/eval/results/ctx_probe_t026_2026-10-20.md), трек [MEMORY_TRACK.md](MEMORY_TRACK.md) | D-026: память + роутер + фильтр в проде |
| То же через шину на боевом сервисе (`ctx_bus.py`) | recall 0.73–0.87 @2k, 1.0 @8k; math 3/3; ход 0.3–1.8 с | TASKS T-016, T-037 («Закрытые») | подтверждено в проде |
| Судья персоны: можно ли доверить оценку локальной модели | qwen3.5:9b парно — перевороты при смене порядка 0.66 (монета); 26B-A4B — 0.35, согласие с ядром 7/9 | [`judge_pairs_2026-10-23_*.md`](../research/eval/results/) | D-028: оценивает ядро по JSON, 26B — запасной |
| Spark-X2.6-4B как «мозг инструментов» | телеграф и эхо собеседника, math 0/3 | [`ctx_probe_spark-x2.6-4b_2026-10-24.md`](../research/eval/results/ctx_probe_spark-x2.6-4b_2026-10-24.md) | снята (T-034) |

## 2. Память и скорость VLM

| Что измеряли | Главное | Сводка | Решение |
|---|---|---|---|
| Пик VRAM 5 моделей × 4 контекста × 3 конфигурации KV | `ollama ps` занижает пик на 0.7–1.2 ГБ (CUDA-контекст и буферы); flash-attention обязателен | [`vlm_mem_stand_2026-10-12.md`](../research/bench/results/vlm_mem_stand_2026-10-12.md) | D-015: бюджеты — по пикам, flash + KV q8_0 |
| QAT-варианты Gemma 4 | E4B-it-qat 4258 МБ @8k (+164 МБ до 16k), E2B 2978 | [`vlm_mem_qat_2026-10-19.md`](../research/bench/results/vlm_mem_qat_2026-10-19.md) | QAT вместо базового Q4_K_M |
| Префилл и генерация по трём классам (dense 4B / 8–9B / MoE 26B-A4B) | E4B: префилл ~2000 ток/с, 70 tok/s; 26B MoE: префилл **100 ток/с**, TTFT 2.25 с, 18 tok/s | [`vlm_mem_prefill_2026-10-19.md`](../research/bench/results/vlm_mem_prefill_2026-10-19.md), [`vlm_mem_moe_2026-10-19.md`](../research/bench/results/vlm_mem_moe_2026-10-19.md) | 26B — только режим B (Q-045) |
| ollama против llama-server на одном GGUF | память и скорость неразличимы; llama-server грузится быстрее (4.0 против 5.4 с), есть slot save/restore и logprobs | [`runtime_2026-10-19.md`](../research/bench/results/runtime_2026-10-19.md) | D-024: прод на llama-server |
| **Весь режим A одновременно**: голос + VLM с проектором + эмбеддер + YOLO | голос 559 МБ; VLM 3038 VRAM + проектор 720 + кадр 46; эмбеддер CPU 421 МБ; YOLO 11n CPU 160 МБ / 50 мс; кадр 640×480 = 136 токенов, префилл 0.5 с → **Orin: ≈ 6.1–6.3 ГБ из 7.6** | [`stack_2026-10-21.md`](../research/bench/results/stack_2026-10-21.md), [MEMORY_BUDGET.md](MEMORY_BUDGET.md) §2 | D-027: состав порога; Qwen3-TTS на GPU — впритык |
| Прод-сервис brain (llama-server + mmproj) | 3812 МиБ VRAM; RSS 3.5 ГБ после `--cache-ram 0` (с кэшем промптов по умолчанию рос до 10.5 ГБ) | TASKS T-016, T-037 | в манифесте brain (T-007) |

## 3. Голосовой тракт

| Что измеряли | Главное | Сводка | Решение |
|---|---|---|---|
| STT faster-whisper `base` int8 CPU | 1.5 с на фразу → 1.05 (убран двойной вызов) → **0.57** (`language="ru"`) | ISSUES I-01, TASKS T-001/T-037 | русский фиксирован |
| Piper (прод-TTS) | RT 0.055–0.09 на CPU | [`piper_rt_2026-10-12.csv`](../research/tts/results/piper_rt_2026-10-12.csv) | голос режима A до M5 |
| Qwen3-TTS 0.6B через qwentts.cpp | GPU Q4_K_M: TTFA 0.02 с, RT 0.13, 1.9 ГБ VRAM; CPU RT 1.1 — не реальное время | [`qwentts_2026-10-21.md`](../research/tts/results/qwentts_2026-10-21.md), трек [TTS_TRACK.md](TTS_TRACK.md) | кандидат только на GPU; решение после Jetson |
| Wake word openWakeWord (стоковое слово) | отдельно 9/10, слитно с фразой 3/4; срабатывание через 0.27–0.57 с после слова — начало фразы теряется | [`kws_monitor_2026-10-24.md`](../research/bench/results/kws_monitor_2026-10-24.md) | аргумент за триггер по началу речи (AMBIENT_MODE) |

## 4. Чего здесь пока нет

Jetson Orin Nano: ни одного числа с устройства (M5 — следующий этап). Визуальный триггер, свап режимов A/B, barge-in — не измерялись, потому что не реализованы (очередь — [TASKS.md](TASKS.md), план — [ROADMAP.md](ROADMAP.md) §7a).
