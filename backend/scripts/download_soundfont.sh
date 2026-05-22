#!/usr/bin/env bash
# FluidSynth 用 SoundFont を backend 内に配置する（初回のみ実行）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/api/models/fluidsynth_assets/soundfonts/FluidR3_GM.sf2"
URL="https://sourceforge.net/projects/pianobooster/files/pianobooster/1.0.0/FluidR3_GM.sf2/download"

mkdir -p "$(dirname "$OUT")"

if [[ -f "$OUT" ]]; then
  echo "Already exists: $OUT ($(du -h "$OUT" | cut -f1))"
  exit 0
fi

echo "Downloading FluidR3_GM.sf2 ..."
curl -fL -o "$OUT" "$URL"
echo "Saved: $OUT ($(du -h "$OUT" | cut -f1))"
