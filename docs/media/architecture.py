# -*- coding: utf-8 -*-
"""Draws docs/media/architecture.jpg — the project diagram: services (systemd units), the Redis bus, the platform layer.
Run from the repository root: python docs/media/architecture.py. Needs Pillow and a font with Cyrillic (Arial on
Windows, DejaVu Sans elsewhere). Numbers are stand peaks (docs/MEMORY_BUDGET.md); update them here when they change."""
import math
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 2300, 1500
BG, INK, GREY = (255, 255, 255), (28, 28, 30), (110, 110, 115)
CPU, GPU, EXT, BUS, PLAT = (225, 236, 250), (222, 243, 226), (250, 240, 222), (240, 240, 244), (246, 246, 248)
CPU_B, GPU_B, EXT_B, BUS_B = (70, 110, 170), (60, 140, 80), (190, 130, 50), (120, 120, 130)
L, R = 60, 2240                      # left / right edge of the content

HERE = os.path.dirname(os.path.abspath(__file__))


def font(size, bold=False):
    cands = (["C:/Windows/Fonts/arialbd.ttf"] if bold else ["C:/Windows/Fonts/arial.ttf"]) + \
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


F_TITLE, F_SUB, F_BOX, F_BODY, F_SMALL, F_BUS = font(46, True), font(24), font(27, True), font(21), font(19), font(23, True)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def text(x, y, s, f=F_BODY, fill=INK, anchor="la"):
    d.text((x, y), s, font=f, fill=fill, anchor=anchor)


def box(x0, y0, x1, y1, title, lines, fill, border, tag=None, f=F_BODY, lh=29):
    d.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=fill, outline=border, width=3)
    text(x0 + 18, y0 + 14, title, F_BOX)
    if tag:
        text(x1 - 18, y0 + 18, tag, F_SMALL, border, anchor="ra")
    y = y0 + 56
    for ln in lines:
        text(x0 + 18, y, ln, f)
        y += lh


def arrow(p0, p1, label=None, color=INK, dashed=False, width=4, label_dx=0):
    (x0, y0), (x1, y1) = p0, p1
    if dashed:
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 18))
        for i in range(0, n, 2):
            a, b = i / n, min(1, (i + 1) / n)
            d.line((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), fill=color, width=width)
    else:
        d.line((x0, y0, x1, y1), fill=color, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    for s in (+0.45, -0.45):
        d.line((x1, y1, x1 - 22 * math.cos(ang - s), y1 - 22 * math.sin(ang - s)), fill=color, width=width)
    if label:
        text((x0 + x1) / 2 + label_dx, (y0 + y1) / 2, label, F_SMALL, GREY, anchor="lm" if label_dx else "mm")


# ---- title
text(L, 36, "Skull Assistant — архитектура: сервисы, шина, платформа", F_TITLE)
text(L, 96, "Стенд RTX 3080 Laptop 8 ГБ / Jetson Orin Nano Super 8 ГБ (общая память). Числа — пики на стенде (docs/MEMORY_BUDGET.md). "
     "Каждый сервис — systemd user-юнит с manifest.yaml.", F_SUB, GREY)

# ---- sensors
text(L, 150, "Датчики и вывод", F_BUS, GREY)
box(L, 185, 400, 305, "Микрофон", ["C270 (USB) или массив ноутбука", "16 кГц, кадры по 30 мс"], EXT, EXT_B)
box(1240, 185, 1520, 305, "Камера", ["C270, 640×480 MJPG", "5 кадров/с"], EXT, EXT_B)
box(1990, 185, R, 305, "Динамик", ["22 кГц моно", "Piper + sox"], EXT, EXT_B)

# ---- voice pipeline (five boxes, 390 wide, 55 gaps)
Y0, Y1 = 350, 580
cols = [(L + i * 445, L + i * 445 + 390) for i in range(5)]     # 60..450, 505..895, 950..1340, 1395..1785, 1840..2230


def pbox(i, title, lines, fill=CPU, border=CPU_B, tag="CPU"):
    box(cols[i][0], Y0, cols[i][1], Y1, title, lines, fill, border, tag, F_SMALL, 27)


pbox(0, "skull-capture", ["захват микрофона → audio:mic", "кольцевой буфер 60 с", "", "CPU · 53 МБ"])
pbox(1, "skull-wakeword", ["Silero VAD: триггер по началу речи", "сегмент ≤ 12 с → events:wakeword", "openWakeWord — запасной; KWS → MCU", "CPU · 76–87 МБ · не выгружается никогда"])
pbox(2, "skull-stt", ["GigaAM v3 CTC + пунктуация (sherpa-onnx)", "voice id CAM++: owner / uncertain / other", "окна в сегменте → events:transcript", "CPU · 396 + 57 МБ · 0.1–0.2 с на фразу"])
pbox(3, "skull-brain", ["роутер типов реплик · «пора ли говорить»", "память за окном · режимы A/B · calc · сеть", "история диалога в Redis → events:reply", "CPU · 48 МБ · llm и embed по HTTP"])
pbox(4, "skull-tts", ["Piper: стрим по предложениям, тембр sox", "звуки: переход режима, модем", "tts:speaking — пока играет", "CPU · 270–290 МБ · первый звук 0.5–1.3 с"])

arrow((230, 305), (230, Y0))
arrow((2115, Y1), (2115, 305))
text(2135, 330, "звук", F_SMALL, GREY)
for i in range(4):
    arrow((cols[i][1], 465), (cols[i + 1][0], 465))

# ---- bus band
BY0, BY1 = 660, 780
d.rounded_rectangle((L, BY0, R, BY1), radius=18, fill=BUS, outline=BUS_B, width=3)
text(L + 20, BY0 + 14, "Redis — шина и состояние", F_BUS)
text(L + 20, BY0 + 52, "Streams: audio:mic → events:wakeword → events:transcript → events:reply · events:vision · cmd:mode → events:mode · events:net", F_BODY)
text(L + 20, BY0 + 84, "Ключи: dialog:history (история = память) · tts:speaking · mode:current / mode:slot · vision:frame (кадр, 10 с) · tts:end", F_BODY)
for (a, b) in cols:
    d.line(((a + b) / 2, Y1, (a + b) / 2, BY0), fill=BUS_B, width=3)
arrow((1380, 305), (1380, BY0))                              # camera -> bus, in the gap between stt and brain
text(1366, 620, "кадры", F_SMALL, GREY, anchor="ra")

# ---- lower row
Y2, Y3 = 860, 1110
box(L, Y2, 600, Y3, "skull-llm · llama-server", ["режим A: Gemma 4 E4B, своя IQ4_XS + imatrix,", "  проектор зрения, контекст 8k · GPU 3.8 ГБ", "режим B: Gemma 4 26B-A4B, эксперты на CPU", "  (-ncmoe 24) · GPU 6.6 ГБ · 18.6 ток/с", "слоты KV save/restore через переход · logprobs"], GPU, GPU_B, "GPU")
box(640, Y2, 1060, Y3, "skull-embed", ["EmbeddingGemma, llama-server CPU", "память за окном: выпавшие обмены", "возвращаются в запрос справкой", "CPU · 421 МБ · resident в A и B"], CPU, CPU_B, "CPU")
box(1100, Y2, 1560, Y3, "skull-vision", ["YuNet + Face Mesh 478 (onnxruntime)", "взгляд на череп = полосы признаков,", "калибровка по месту → events:vision", "кадр по событию → vision:frame · CPU 141 МБ"], CPU, CPU_B, "CPU")
box(1600, Y2, R, Y3, "skull-dispatcher", ["режимы A / B по манифестам (D-030):", "cmd:mode → freeze (slot save, stop) → thaw", "→ events:mode по фазам; откат при ошибке", "бюджеты режимов — deploy/modes.yaml"], CPU, CPU_B, "CPU")

arrow((330, Y2), (330, BY1)); text(350, Y2 - 30, "HTTP /v1 от brain (SSE-стрим)", F_SMALL, GREY)
arrow((850, Y2), (850, BY1)); text(870, Y2 - 30, "HTTP от brain", F_SMALL, GREY)
arrow((1330, Y2), (1330, BY1))
arrow((1920, Y2), (1920, BY1)); text(1940, Y2 - 30, "systemctl --user к юнитам", F_SMALL, GREY)

# ---- network (external, dashed) — under vision + dispatcher, fed from brain through the gap vision | dispatcher
NY0, NY1 = 1160, 1282
box(1100, NY0, R, NY1, "Сеть — по умолчанию выключена (D-031)", ["brain → Wikipedia API (справочник); SearXNG по URL; включается конфигом или фразой владельца",
                                                                   "слышно всегда: «Сейчас гляну.» + звук модема; новости и погода → честное «в сети не нашёл»"], EXT, EXT_B)
arrow((1580, Y1), (1580, NY0), None, EXT_B, dashed=True)
text(1566, 1135, "HTTPS", F_SMALL, EXT_B, anchor="ra")

# ---- platform band + legend
PY0, PY1 = 1312, 1465
d.rounded_rectangle((L, PY0, 1560, PY1), radius=18, fill=PLAT, outline=BUS_B, width=3)
text(L + 20, PY0 + 14, "Платформа", F_BUS)
text(L + 20, PY0 + 52, "• один сервис = папка + manifest.yaml: бюджет по пикам, резидентность в режимах A/B/C, freeze/thaw, топики шины", F_BODY)
text(L + 20, PY0 + 82, "• изоляция процессов = своп моделей: память GPU возвращается только со смертью процесса", F_BODY)
text(L + 20, PY0 + 112, "• GPU — только тяжёлым (VLM); голос и эмбеддер — CPU-only; wake word не выгружается никогда", F_BODY)

lx = 1600
for i, (c, b, lab) in enumerate(((CPU, CPU_B, "CPU-сервис"), (GPU, GPU_B, "GPU-сервис"), (EXT, EXT_B, "устройство / внешнее"))):
    d.rounded_rectangle((lx, PY0 + 20 + i * 42, lx + 40, PY0 + 50 + i * 42), radius=6, fill=c, outline=b, width=3)
    text(lx + 56, PY0 + 24 + i * 42, lab, F_BODY)
text(R, H - 24, "github.com/Vzzzz/skull-assistant · 2026-11-01", F_SMALL, GREY, anchor="ra")

out = os.path.join(HERE, "architecture.jpg")
img.save(out, "JPEG", quality=90)
print(out, os.path.getsize(out) // 1024, "KB")
