# models/

Веса не хранятся в git. Ожидаемая раскладка и источники:

| Путь | Что | Источник | Лицензия |
|---|---|---|---|
| `wakeword/hey_jarvis_v0.1.onnx` | openWakeWord, стоковое слово | github.com/dscripka/openWakeWord (models) | Apache-2.0 |
| `wakeword/melspectrogram.onnx`, `wakeword/embedding_model.onnx` | openWakeWord feature-модели | там же | Apache-2.0 |
| `tts/ru_RU-dmitri-medium.onnx(.json)` | Piper, русский голос | huggingface.co/rhasspy/piper-voices | см. MODEL_CARD голоса |
| `tts/ru_RU-ruslan-medium.onnx(.json)` | Piper, русский голос | там же | см. MODEL_CARD голоса |
| faster-whisper `base` int8 | STT | скачивается библиотекой в кэш HF при первом запуске | MIT |
| faster-whisper `small` int8 | STT, кандидат T-040 | кэш HF, как `base` | MIT |
| `stt/sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/` | STT, кандидат T-040: GigaAM v2 CTC int8 | github.com/k2-fsa/sherpa-onnx/releases, тег `asr-models` | MIT (GigaAM; LICENSE в архиве пустой) |
| `stt/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/` | STT, кандидат T-040: GigaAM v3 CTC int8 (официальная сборка sherpa-onnx) | там же | MIT (GigaAM) |
| `stt/sherpa-onnx-streaming-t-one-russian-2025-09-08/` | STT потоковый, кандидат T-040: T-one CTC | там же (из github.com/voicekit-team/T-one) | Apache-2.0 |
| `stt/sherpa-onnx-streaming-zipformer-small-ru-vosk-2025-08-16/` | STT потоковый, только нижняя граница T-040: Vosk small ru zipformer | там же (из huggingface.co/alphacep/vosk-model-small-streaming-ru) | не объявлена — не в прод |
| `voice/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | voice id (T-039, прод): 3D-Speaker CAM++, эмбеддинг говорящего 192 | github.com/k2-fsa/sherpa-onnx/releases, тег `speaker-recongition-models` (опечатка в теге — так у них) | Apache-2.0 (3D-Speaker) |
| `voice/wespeaker_en_voxceleb_resnet34.onnx` | voice id, кандидат (не выбран: EER 0.18) | там же | см. репозиторий WeSpeaker |
| `voice/owner.npy`, `voice/data/` | вектор владельца и записи для замера — **приватно**, голос владельца | `research/voice/enroll.py` | — |
| `quant/kostyan-e4b-iq4xs-imx.gguf` | VLM прод (режим A): Gemma 4 E4B, своя квантизация IQ4_XS + imatrix | `research/quant/quant_e4b.sh` из `hf/gemma-4-E4B-it` (D-023) | Gemma Terms of Use |
| `hf/gemma-4-E4B-it-GGUF/mmproj-gemma-4-E4B-it-Q8_0.gguf` | проектор зрения для VLM | huggingface.co/ggml-org (см. `deploy/fetch_models.sh`) | Gemma Terms of Use |
| `hf/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf` | эмбеддер памяти (CPU) | huggingface.co/ggml-org/embeddinggemma-300M-GGUF | Gemma Terms of Use |
| ollama: `gemma4:e4b-it-qat`, `gemma4:e2b-it-qat`, `gemma4:26b-a4b-it-mtp`, кандидаты CTX-1 | замеры и запасные варианты (не прод) | `ollama pull <model>` | по модели |
| Qwen3-TTS 0.6B (qwentts.cpp) | кандидат TTS на GPU (T-013) | см. `research/tts/bench_qwentts.sh` | Apache-2.0 |

Загрузка кандидатов — `deploy/fetch_models.sh` (файлы больше 1 ГБ — через aria2c, см. `docs/agents/EXECUTOR.md`).
