# -*- coding: utf-8 -*-
"""Draws docs/media/architecture.jpg — the project diagram: services (systemd units) with their models, the Redis bus,
the platform layer. Numbers and descriptions live in the tables (README, docs/MEMORY_BUDGET.md), not here.
Run from the repository root: python docs/media/architecture.py. Needs Pillow and a font with Cyrillic (Arial on
Windows, DejaVu Sans elsewhere)."""
import math
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 2400, 1280
BG, INK, GREY = (255, 255, 255), (28, 28, 30), (110, 110, 115)
CPU, GPU, EXT, BUS, PLAT = (225, 236, 250), (222, 243, 226), (250, 240, 222), (240, 240, 244), (246, 246, 248)
CPU_B, GPU_B, EXT_B, BUS_B = (70, 110, 170), (60, 140, 80), (190, 130, 50), (120, 120, 130)
L, R = 60, 2240                      # left / right edge of the content; the net arrow runs outside R

HERE = os.path.dirname(os.path.abspath(__file__))


def font(size, bold=False):
    cands = (["C:/Windows/Fonts/arialbd.ttf"] if bold else ["C:/Windows/Fonts/arial.ttf"]) + \
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


F_TITLE, F_SUB, F_BOX, F_BODY, F_SMALL, F_BUS = font(46, True), font(24), font(28, True), font(22), font(19), font(23, True)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def text(x, y, s, f=F_BODY, fill=INK, anchor="la"):
    d.text((x, y), s, font=f, fill=fill, anchor=anchor)


def box(x0, y0, x1, y1, title, lines, fill, border, tag=None):
    d.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=fill, outline=border, width=3)
    text(x0 + 18, y0 + 14, title, F_BOX)
    if tag:
        text(x1 - 18, y0 + 18, tag, F_SMALL, border, anchor="ra")
    y = y0 + 58
    for ln in lines:
        text(x0 + 18, y, ln, F_BODY)
        y += 30


def seg(p0, p1, color, dashed, width):
    (x0, y0), (x1, y1) = p0, p1
    if not dashed:
        d.line((x0, y0, x1, y1), fill=color, width=width); return
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 18))
    for i in range(0, n, 2):
        a, b = i / n, min(1, (i + 1) / n)
        d.line((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), fill=color, width=width)


def arrow(*pts, color=INK, dashed=False, width=4):
    """Polyline with an arrowhead at the last point."""
    for a, b in zip(pts, pts[1:]):
        seg(a, b, color, dashed, width)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    ang = math.atan2(y1 - y0, x1 - x0)
    for s in (+0.45, -0.45):
        d.line((x1, y1, x1 - 22 * math.cos(ang - s), y1 - 22 * math.sin(ang - s)), fill=color, width=width)


# ---- title
text(L, 36, "Skull Assistant — архитектура: сервисы, шина, платформа", F_TITLE)
text(L, 96, "Jetson Orin Nano Super 8 ГБ (стенд — RTX 3080 Laptop 8 ГБ, WSL2). Каждый сервис — systemd user-юнит с manifest.yaml; "
     "бюджеты и числа — в docs/MEMORY_BUDGET.md и README.", F_SUB, GREY)

# ---- sensors (top) : microphone over capture, speaker over tts
text(L, 150, "Датчики и вывод", F_BUS, GREY)
box(L, 185, 450, 280, "Микрофон", ["C270 (USB) / массив ноутбука"], EXT, EXT_B)
box(1840, 185, R, 280, "Динамик", ["22 кГц моно"], EXT, EXT_B)

# ---- voice pipeline: five boxes, 390 wide, 55 gaps
Y0, Y1 = 330, 450
cols = [(L + i * 445, L + i * 445 + 390) for i in range(5)]     # 60..450 … 1840..2230


def pbox(i, title, lines, fill=CPU, border=CPU_B, tag="CPU"):
    box(cols[i][0], Y0, cols[i][1], Y1, title, lines, fill, border, tag)


pbox(0, "skull-capture", ["PulseAudio / ALSA → audio:mic"])
pbox(1, "skull-wakeword", ["Silero VAD (openWakeWord — запасной)"])
pbox(2, "skull-stt", ["GigaAM v3 CTC · CAM++ (voice id)"])
pbox(3, "skull-brain", ["роутер · политика · память · режимы"])
pbox(4, "skull-tts", ["Piper (+ sox)"])

arrow((255, 280), (255, Y0))                      # mic -> capture
arrow((2035, Y0), (2035, 280))                    # tts -> speaker (from the top edge of the box)
for i in range(4):
    arrow((cols[i][1], 390), (cols[i + 1][0], 390))   # stream names are listed on the bus band

# ---- bus band
BY0, BY1 = 530, 650
d.rounded_rectangle((L, BY0, R, BY1), radius=18, fill=BUS, outline=BUS_B, width=3)
text(L + 20, BY0 + 14, "Redis — шина и состояние", F_BUS)
text(L + 20, BY0 + 52, "Streams: audio:mic → events:wakeword → events:transcript → events:reply · events:vision · cmd:mode → events:mode · events:net", F_BODY)
text(L + 20, BY0 + 84, "Ключи: dialog:history (история = память) · tts:speaking · mode:current / mode:slot · vision:frame (кадр, 10 с) · tts:end", F_BODY)
for (a, b) in cols:
    d.line(((a + b) / 2, Y1, (a + b) / 2, BY0), fill=BUS_B, width=3)

# ---- lower row: models and helpers
Y2, Y3 = 730, 850
box(L, Y2, 600, Y3, "skull-llm · llama-server", ["A: Gemma 4 E4B (IQ4_XS + imatrix, mmproj)", "B: Gemma 4 26B-A4B (Q4_K_M, эксперты на CPU)"], GPU, GPU_B, "GPU")
box(640, Y2, 1060, Y3, "skull-embed", ["EmbeddingGemma"], CPU, CPU_B, "CPU")
box(1100, Y2, 1560, Y3, "skull-vision", ["YuNet · Face Mesh 478"], CPU, CPU_B, "CPU")
box(1600, Y2, R, Y3, "skull-dispatcher", ["режимы A / B по манифестам"], CPU, CPU_B, "CPU")

arrow((330, Y2), (330, BY1)); text(350, Y2 - 30, "HTTP /v1 от brain", F_SMALL, GREY)
arrow((850, Y2), (850, BY1)); text(870, Y2 - 30, "HTTP от brain", F_SMALL, GREY)
arrow((1330, Y2), (1330, BY1)); text(1350, Y2 - 30, "events:vision · vision:frame", F_SMALL, GREY)
arrow((1920, Y2), (1920, BY1)); text(1940, Y2 - 30, "cmd:mode · events:mode", F_SMALL, GREY)

# ---- camera under vision, network under dispatcher
box(1100, 910, 1560, 1000, "Камера", ["C270, 640×480 MJPG"], EXT, EXT_B)
arrow((1330, 910), (1330, Y3))                    # camera -> vision
box(1600, 910, R, 1020, "Сеть — по умолчанию выключена (D-031)", ["Wikipedia API (справочник) · SearXNG по URL"], EXT, EXT_B)
arrow((cols[3][1], 420), (2310, 420), (2310, 965), (R, 965), color=EXT_B, dashed=True)     # brain -> net, around the bus
text(2322, 690, "HTTPS", F_SMALL, EXT_B, anchor="lm")

# ---- platform band + legend
PY0, PY1 = 1080, 1230
d.rounded_rectangle((L, PY0, 1560, PY1), radius=18, fill=PLAT, outline=BUS_B, width=3)
text(L + 20, PY0 + 14, "Платформа", F_BUS)
text(L + 20, PY0 + 52, "• один сервис = папка + manifest.yaml: бюджет по пикам, резидентность в режимах A/B/C, freeze/thaw, топики шины", F_BODY)
text(L + 20, PY0 + 82, "• изоляция процессов = своп моделей: память GPU возвращается только со смертью процесса", F_BODY)
text(L + 20, PY0 + 112, "• GPU — только тяжёлым (VLM); голос и эмбеддер — CPU-only; wake word не выгружается никогда", F_BODY)

lx = 1600
for i, (c, b, lab) in enumerate(((CPU, CPU_B, "CPU-сервис"), (GPU, GPU_B, "GPU-сервис"), (EXT, EXT_B, "устройство / внешнее"))):
    d.rounded_rectangle((lx, PY0 + 20 + i * 42, lx + 40, PY0 + 50 + i * 42), radius=6, fill=c, outline=b, width=3)
    text(lx + 56, PY0 + 24 + i * 42, lab, F_BODY)
text(R, H - 26, "github.com/Vzzzz/skull-assistant · 2026-11-01", F_SMALL, GREY, anchor="ra")

out = os.path.join(HERE, "architecture.jpg")
img.save(out, "JPEG", quality=90)
print(out, os.path.getsize(out) // 1024, "KB")
