# models/

Веса не хранятся в git. Ожидаемая раскладка и источники:

| Путь | Что | Источник | Лицензия |
|---|---|---|---|
| `wakeword/hey_jarvis_v0.1.onnx` | openWakeWord, стоковое слово | github.com/dscripka/openWakeWord (models) | Apache-2.0 |
| `wakeword/melspectrogram.onnx`, `wakeword/embedding_model.onnx` | openWakeWord feature-модели | там же | Apache-2.0 |
| `tts/ru_RU-dmitri-medium.onnx(.json)` | Piper, русский голос | huggingface.co/rhasspy/piper-voices | см. MODEL_CARD голоса |
| `tts/ru_RU-ruslan-medium.onnx(.json)` | Piper, русский голос | там же | см. MODEL_CARD голоса |
| faster-whisper `base` int8 | STT | скачивается библиотекой в кэш HF при первом запуске | MIT |
| `quant/kostyan-e4b-iq4xs-imx.gguf` | VLM прод (режим A): Gemma 4 E4B, своя квантизация IQ4_XS + imatrix | `research/quant/quant_e4b.sh` из `hf/gemma-4-E4B-it` (D-023) | Gemma Terms of Use |
| `hf/gemma-4-E4B-it-GGUF/mmproj-gemma-4-E4B-it-Q8_0.gguf` | проектор зрения для VLM | huggingface.co/ggml-org (см. `deploy/fetch_models.sh`) | Gemma Terms of Use |
| `hf/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf` | эмбеддер памяти (CPU) | huggingface.co/ggml-org/embeddinggemma-300M-GGUF | Gemma Terms of Use |
| ollama: `gemma4:e4b-it-qat`, `gemma4:e2b-it-qat`, `gemma4:26b-a4b-it-mtp`, кандидаты CTX-1 | замеры и запасные варианты (не прод) | `ollama pull <model>` | по модели |
| Qwen3-TTS 0.6B (qwentts.cpp) | кандидат TTS на GPU (T-013) | см. `research/tts/bench_qwentts.sh` | Apache-2.0 |

Загрузка кандидатов — `deploy/fetch_models.sh` (файлы больше 1 ГБ — через aria2c, см. `docs/agents/EXECUTOR.md`).
