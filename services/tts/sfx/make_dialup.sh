#!/usr/bin/env bash
# The «checking in the net» sound (T-051, U-002/U-003, D-031): a dial-up modem handshake — dial tone, the answer tone,
# the warble and a burst of line noise — a 3-s loop at -6 dBFS peak, 22050 Hz mono (Piper's rate, as thinking.wav).
# An honest signal that the skull has left the offline. skull-tts plays it from events:net begin to done.
#   bash services/tts/sfx/make_dialup.sh     # -> services/tts/sfx/dialup.wav
set -euo pipefail
cd "$(dirname "$0")"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
R=22050
sox -n -r $R -c 1 -b 16 "$W/a.wav" synth 0.45 sine 350 sine 440 remix - gain -3           # dial tone
sox -n -r $R -c 1 -b 16 "$W/b.wav" synth 0.55 sine 2100 gain -6                           # answer tone
sox -n -r $R -c 1 -b 16 "$W/c1.wav" synth 0.12 sine 1200 gain -6
sox -n -r $R -c 1 -b 16 "$W/c2.wav" synth 0.12 sine 2400 gain -6
sox "$W/c1.wav" "$W/c2.wav" "$W/c1.wav" "$W/c2.wav" "$W/c1.wav" "$W/c2.wav" "$W/c.wav"   # the warble, 0.72 s
sox -n -r $R -c 1 -b 16 "$W/d.wav" synth 1.28 whitenoise band 1800 1200 tremolo 18 70 gain -8   # line noise
sox "$W/a.wav" "$W/b.wav" "$W/c.wav" "$W/d.wav" "$W/all.wav"
sox "$W/all.wav" dialup.wav fade t 0.01 0 0.03 gain -n -6
sox dialup.wav -n stat 2>&1 | grep -E "Length|Maximum amplitude"
