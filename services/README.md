# services/ — прод

Один сервис = одна папка: код + `manifest.yaml`. Манифест — контракт с диспетчером (см. `docs/ROADMAP.md` §2): бюджет памяти **по пикам**, резидентность в режимах A/B/C, семантика freeze/thaw, топики шины, healthcheck. Сервис без `budget.measured_on` в admission control не допускается.

| Сервис | Юнит | Вход → выход | Backend |
|---|---|---|---|
| audio_capture | skull-capture | микрофон → `audio:mic` (кольцо 60 с, кадры 30 мс, int16 16 kHz) | cpu |
| wakeword | skull-wakeword | `audio:mic` → `events:wakeword` | cpu (onnx) |
| stt | skull-stt | `events:wakeword` + `audio:mic` → `events:transcript` | cpu (ctranslate2 int8) |
| brain | skull-brain | `events:transcript` → `events:reply` | ollama (CUDA) |
| tts | skull-tts | `events:reply` → динамик; флаг `tts:speaking` | cpu (onnx) + sox |

Общие ключи Redis: `tts:speaking` (полудуплекс). Будущие: `tts:stop` (barge-in), `mode:current` (диспетчер).
