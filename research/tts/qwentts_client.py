"""Streaming client for qwentts.cpp tts-server: measures time to first audio (first PCM body bytes), total time,
bytes; writes the PCM as a 24 kHz mono WAV. Used by bench_qwentts.sh.

Usage: qwentts_client.py <port> <voice> <language> <out.wav> <text>
Prints: ttfa_s total_s bytes
"""
import http.client
import json
import sys
import time
import wave

port, voice, lang, out, text = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
body = json.dumps({"input": text, "voice": voice, "language": lang, "response_format": "pcm", "seed": 42}, ensure_ascii=False).encode("utf-8")
conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
t0 = time.perf_counter()
conn.request("POST", "/v1/audio/speech", body=body, headers={"Content-Type": "application/json"})
resp = conn.getresponse()
chunks = []
ttfa = None
while True:
    chunk = resp.read(3840)  # one 12.5 Hz frame = 1920 samples * 2 bytes
    if not chunk:
        break
    if ttfa is None:
        ttfa = time.perf_counter() - t0
    chunks.append(chunk)
total = time.perf_counter() - t0
data = b"".join(chunks)
if resp.status != 200:
    print(f"ERR {resp.status} {data[:120]!r}", file=sys.stderr)
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(data)
print(f"{ttfa if ttfa is not None else -1:.3f} {total:.3f} {len(data)}")
