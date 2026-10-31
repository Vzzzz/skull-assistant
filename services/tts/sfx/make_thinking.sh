#!/usr/bin/env bash
# Stub for the transition sound (T-046, U-003): «the skull is thinking» — hard-disk head clicks over a faint hiss,
# a 3-s loop at about -20 dB, 22050 Hz mono (Piper's rate, so skull-tts plays it as is). The real sample is the
# owner's choice later: replace thinking.wav, the path is in services/tts/manifest.yaml params.sfx.transition.
#   bash services/tts/sfx/make_thinking.sh     # -> services/tts/sfx/thinking.wav
set -euo pipefail
cd "$(dirname "$0")"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
R=22050
sox -n -r $R -c 1 -b 16 "$W/hiss.wav" synth 3 brownnoise gain -42
# a click: 4 ms of noise with a fast decay; seek/settle pairs at irregular times
sox -n -r $R -c 1 -b 16 "$W/click.wav" synth 0.004 whitenoise fade q 0 0.004 0.0035 gain -20
args=()
for t in 0.21 0.24 0.83 1.37 1.40 1.46 2.18 2.62 2.65; do
  sox "$W/click.wav" "$W/c$t.wav" pad "$t" "$(echo "3 - $t - 0.004" | bc -l)"
  args+=("$W/c$t.wav")
done
sox -m "$W/hiss.wav" "${args[@]}" "$W/mix.wav"     # -m scales every input down by their count
sox "$W/mix.wav" thinking.wav gain -n -20          # peak back to -20 dBFS
sox thinking.wav -n stat 2>&1 | grep -E "Length|Maximum amplitude|RMS     amplitude"
