"""T-055 step 5: one look-line call with a frame (the T-053 path: core.gaze_lines.frame_message, the frame turn of
services/brain) against a llama-server with the model + mmproj; 3 frames x 3 tries -> the lines, judged by hand
(about the frame / general / invented).  .venv/bin/python research/vision/frame_probe.py <port> <frame.jpg> ..."""
import json, sys
from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.gaze_lines import frame_message

PERSONA = (ROOT / "persona/system_prompt.md").read_text(encoding="utf-8").strip()
PROMPT = ("Служебно, не отвечай на это как на реплику: собеседник молча смотрит на тебя уже две секунды. Скажи одно короткое, "
          "в своём тоне. Не говори, что на тебя смотрят; не используй слова с корнями «смотр», «гляд», «наблюда». "
          "На кадре с камеры — твой владелец смотрит на тебя. Одна фраза в образе о том, что видно (одежда, предмет в руках, "
          "поза, что за спиной). Если ничего примечательного — так и скажи, не выдумывай.")
port, frames = sys.argv[1], sys.argv[2:]
for fr in frames:
    for k in range(3):
        body = {"messages": [{"role": "system", "content": PERSONA}, frame_message(PROMPT, Path(fr).read_bytes())],
                "max_tokens": 60, "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}}
        j = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=120).json()
        print(json.dumps({"frame": Path(fr).name, "try": k + 1, "line": j["choices"][0]["message"]["content"].strip(),
                          "prompt_tokens": j["usage"]["prompt_tokens"]}, ensure_ascii=False), flush=True)
