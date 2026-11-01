# -*- coding: utf-8 -*-
"""Draws docs/media/architecture.jpg — the project diagram: services (systemd units), the Redis bus, the platform layer.
Run from the repository root: python docs/media/architecture.py. Needs Pillow and a font with Cyrillic (Arial on
Windows, DejaVu Sans elsewhere). Numbers are stand peaks (docs/MEMORY_BUDGET.md); update them here when they change."""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 2200, 1460
BG, INK, GREY = (255, 255, 255), (28, 28, 30), (110, 110, 115)
CPU, GPU, EXT, BUS, PLAT = (225, 236, 250), (222, 243, 226), (250, 240, 222), (240, 240, 244), (246, 246, 248)
CPU_B, GPU_B, EXT_B, BUS_B = (70, 110, 170), (60, 140, 80), (190, 130, 50), (120, 120, 130)

HERE = os.path.dirname(os.path.abspath(__file__))


def font(size, bold=False):
    for cand in (["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"] if bold else ["C:/Windows/Fonts/arial.ttf"]) + \
                ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


F_TITLE, F_SUB, F_BOX, F_BODY, F_SMALL, F_BUS = font(46, True), font(24), font(27, True), font(21), font(19), font(23, True)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def text(x, y, s, f=F_BODY, fill=INK, anchor="la"):
    d.text((x, y), s, font=f, fill=fill, anchor=anchor)


def box(x0, y0, x1, y1, title, lines, fill, border, tag=None):
    d.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=fill, outline=border, width=3)
    text(x0 + 18, y0 + 14, title, F_BOX)
    if tag:
        text(x1 - 18, y0 + 18, tag, F_SMALL, border, anchor="ra")
    y = y0 + 56
    for ln in lines:
        text(x0 + 18, y, ln, F_BODY)
        y += 29


def arrow(p0, p1, label=None, color=INK, dashed=False, width=4):
    (x0, y0), (x1, y1) = p0, p1
    if dashed:
        import math
        L = math.hypot(x1 - x0, y1 - y0); n = max(1, int(L / 18))
        for i in range(0, n, 2):
            a, b = i / n, min(1, (i + 1) / n)
            d.line((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), fill=color, width=width)
    else:
        d.line((x0, y0, x1, y1), fill=color, width=width)
    import math
    ang = math.atan2(y1 - y0, x1 - x0)
    for s in (+0.45, -0.45):
        d.line((x1, y1, x1 - 22 * math.cos(ang - s), y1 - 22 * math.sin(ang - s)), fill=color, width=width)
    if label:
        text((x0 + x1) / 2, (y0 + y1) / 2 - 16, label, F_SMALL, GREY, anchor="mm")


# ---- title
text(60, 36, "Skull Assistant — архитектура: сервисы, шина, платформа", F_TITLE)
text(60, 96, "Стенд RTX 3080 Laptop 8 ГБ / Jetson Orin Nano Super 8 ГБ (общая память). Числа — пики на стенде, docs/MEMORY_BUDGET.md. "
     "Каждый сервис — systemd user-юнит с manifest.yaml.", F_SUB, GREY)

# ---- sensors row
text(60, 150, "Датчики и вывод", F_BUS, GREY)
box(60, 185, 330, 275, "Микрофон", ["USB-камера C270 / массив", "16 кГц, кадры 30 мс"], EXT, EXT_B)
box(1180, 185, 1450, 275, "Камера", ["C270, 640×480 MJPG", "5 кадров/с"], EXT, EXT_B)
box(1930, 185, 2140, 275, "Динамик", ["22 кГц моно", "Piper + sox"], EXT, EXT_B)

# ---- voice pipeline row (y 330..560)
Y0, Y1 = 330, 560
cols = [(60, 440), (480, 860), (900, 1280), (1320, 1700), (1760, 2140)]


def pbox(i, *a):
    box(cols[i][0], Y0, cols[i][1], Y1, *a)


pbox(0, "skull-capture", ["захват → audio:mic", "кольцевой буфер 60 с", "", "CPU · 53 МБ"], CPU, CPU_B, "CPU")
pbox(1, "skull-wakeword", ["Silero VAD: триггер по началу речи,", "сегмент ≤ 12 с → events:wakeword", "(openWakeWord — запасной; KWS → MCU)", "CPU · 76–87 МБ · не выгружается никогда"], CPU, CPU_B, "CPU")
pbox(2, "skull-stt", ["GigaAM v3 CTC + пунктуация (sherpa-onnx)", "voice id CAM++: owner / uncertain / other", "окна внутри сегмента, эфир → events:transcript", "CPU · 396 + 57 МБ · 0.1–0.2 с"], CPU, CPU_B, "CPU")
pbox(3, "skull-brain", ["роутер типов реплик · политика «пора ли говорить»", "память за окном · режимы A/B · calc · сеть", "история диалога в Redis → events:reply", "CPU · 48 МБ · вызовы llm/embed по HTTP"], CPU, CPU_B, "CPU")
pbox(4, "skull-tts", ["Piper, стрим по предложениям, тембр sox", "звуки: переход режима, модем", "tts:speaking — пока играет", "CPU · 270–290 МБ · первый звук 0.5–1.3 с"], CPU, CPU_B, "CPU")

# arrows sensors -> services and along the pipeline
arrow((195, 275), (195, 330))
arrow((1315, 275), (1315, 640), "кадры")
arrow((2035, 560), (2035, 275), "звук")
for i, lab in enumerate(["audio:mic", "events:wakeword", "events:transcript", "events:reply"]):
    arrow((cols[i][1], 445), (cols[i + 1][0], 445), lab)

# ---- bus band
BY0, BY1 = 640, 760
d.rounded_rectangle((60, BY0, 2140, BY1), radius=18, fill=BUS, outline=BUS_B, width=3)
text(80, BY0 + 14, "Redis — шина и состояние", F_BUS)
text(80, BY0 + 52, "Streams: audio:mic → events:wakeword → events:transcript → events:reply · events:vision · cmd:mode → events:mode · events:net",
     F_BODY)
text(80, BY0 + 84, "Ключи: dialog:history (история = память) · tts:speaking · mode:current / mode:slot · vision:frame (кадр, 10 с) · tts:end", F_BODY)
for x in (250, 670, 1090, 1510, 1950):
    d.line((x, Y1, x, BY0), fill=BUS_B, width=3)

# ---- lower row: GPU / helpers (y 840..1090)
Y2, Y3 = 840, 1090
box(60, 520 + 320, 560, Y3, "skull-llm · llama-server", ["режим A: Gemma 4 E4B, своя IQ4_XS + imatrix,", "  проектор зрения, контекст 8k · GPU 3.8 ГБ", "режим B: Gemma 4 26B-A4B, эксперты на CPU", "  (-ncmoe 24) · GPU 6.6 ГБ · 18.6 ток/с", "слоты KV save/restore через переход · logprobs"], GPU, GPU_B, "GPU")
box(600, Y2, 1000, Y3, "skull-embed", ["EmbeddingGemma, llama-server CPU", "память за окном: выпавшие обмены", "возвращаются справкой", "CPU · 421 МБ · resident в A и B"], CPU, CPU_B, "CPU")
box(1040, Y2, 1500, Y3, "skull-vision", ["YuNet + Face Mesh 478 (onnxruntime)", "взгляд на череп = полосы признаков,", "калибровка по месту → events:vision", "кадр по событию → vision:frame · CPU 141 МБ"], CPU, CPU_B, "CPU")
box(1540, Y2, 2140, Y3, "skull-dispatcher", ["режимы A / B по манифестам (D-030):", "cmd:mode → freeze (slot save, stop) → thaw", "→ events:mode по фазам; откат при ошибке", "бюджеты режимов — deploy/modes.yaml"], CPU, CPU_B, "CPU")

arrow((310, Y2), (310, BY1))
arrow((800, Y2), (800, BY1))
arrow((1270, Y2), (1270, BY1))
arrow((1840, Y2), (1840, BY1))
text(330, Y2 - 28, "HTTP /v1 от brain (SSE-стрим)", F_SMALL, GREY)
text(820, Y2 - 28, "HTTP от brain", F_SMALL, GREY)
text(1860, Y2 - 28, "systemctl --user к юнитам", F_SMALL, GREY)

# ---- external (net) — dashed
box(1540, 1130, 2140, 1230, "Сеть — выключена по умолчанию (D-031)", ["brain → Wikipedia API (справочник), SearXNG по URL;", "включается конфигом / фразой владельца; слышно: «Сейчас гляну.» + модем"], EXT, EXT_B, "внешнее")
arrow((1700, 560), (1700, 1130), None, EXT_B, dashed=True)

# ---- platform band
d.rounded_rectangle((60, 1270, 1500, 1420), radius=18, fill=PLAT, outline=BUS_B, width=3)
text(80, 1284, "Платформа", F_BUS)
text(80, 1322, "• один сервис = папка + manifest.yaml: бюджет по пикам, резидентность в режимах A/B/C, freeze/thaw, топики шины", F_BODY)
text(80, 1352, "• изоляция процессов = своп моделей: память GPU возвращается только со смертью процесса", F_BODY)
text(80, 1382, "• GPU — только тяжёлым (VLM); голос и эмбеддер — CPU-only; wake word не выгружается никогда", F_BODY)

# legend
lx = 1540
for i, (c, b, lab) in enumerate(((CPU, CPU_B, "CPU-сервис"), (GPU, GPU_B, "GPU-сервис"), (EXT, EXT_B, "устройство / внешнее"))):
    d.rounded_rectangle((lx, 1290 + i * 42, lx + 40, 1320 + i * 42), radius=6, fill=c, outline=b, width=3)
    text(lx + 56, 1294 + i * 42, lab, F_BODY)
text(lx, 1420, "github.com/Vzzzz/skull-assistant · 2026-11-01", F_SMALL, GREY)

out = os.path.join(HERE, "architecture.jpg")
img.save(out, "JPEG", quality=90)
print(out, os.path.getsize(out) // 1024, "KB")
