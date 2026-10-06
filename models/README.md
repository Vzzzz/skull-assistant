# models/

Веса не хранятся в git. Ожидаемая раскладка и источники:

| Путь | Что | Источник | Лицензия |
|---|---|---|---|
| `wakeword/hey_jarvis_v0.1.onnx` | openWakeWord, стоковое слово | github.com/dscripka/openWakeWord (models) | Apache-2.0 |
| `wakeword/melspectrogram.onnx`, `wakeword/embedding_model.onnx` | openWakeWord feature-модели | там же | Apache-2.0 |
| `tts/ru_RU-dmitri-medium.onnx(.json)` | Piper, русский голос | huggingface.co/rhasspy/piper-voices | см. MODEL_CARD голоса |
| `tts/ru_RU-ruslan-medium.onnx(.json)` | Piper, русский голос | там же | см. MODEL_CARD голоса |
| faster-whisper `base` int8 | STT | скачивается библиотекой в кэш HF при первом запуске | MIT |
| VLM (ollama) | `gemma3:4b` и кандидаты M5 | `ollama pull <model>` | Gemma Terms of Use |

Скрипт загрузки появится вместе с M5 (`deploy/fetch_models.sh`).
