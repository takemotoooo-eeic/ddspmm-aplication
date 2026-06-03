#!/usr/bin/env bash
# MelodyFlow 重みを配置（初回のみ。合計約 4GB）
# サーバが遅い場合は手元で実行 → scp/rsync で転送（README の MelodyFlow 節参照）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/api/models/melodyflow_assets/melodyflow-t24-30secs"
REPO="facebook/melodyflow-t24-30secs"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$OUT"

FILES=(
  "compression_state_dict.bin"
  "state_dict.bin"
)

is_complete() {
  local name="$1"
  local path="$OUT/$name"
  [[ -f "$path" ]] || return 1
  local size
  size="$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path")"
  if [[ "$name" == "compression_state_dict.bin" && "$size" -lt 200000000 ]]; then
    return 1
  fi
  if [[ "$name" == "state_dict.bin" && "$size" -lt 3500000000 ]]; then
    return 1
  fi
  return 0
}

all_complete() {
  local f
  for f in "${FILES[@]}"; do
    is_complete "$f" || return 1
  done
  return 0
}

if all_complete; then
  echo "Already exists: $OUT"
  ls -lh "$OUT"/*.bin
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl が必要です。" >&2
  exit 1
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN を使用してダウンロードします。"
  auth_header=(-H "Authorization: Bearer ${HF_TOKEN}")
else
  echo "未認証です。遅い場合は export HF_TOKEN=hf_xxx"
  auth_header=()
fi

download_one() {
  local name="$1"
  local dest="$OUT/$name"
  local url="${BASE_URL}/${name}"

  if is_complete "$name"; then
    echo "skip (complete): $dest ($(du -h "$dest" | cut -f1))"
    return 0
  fi

  if [[ -f "$dest" ]]; then
    local size
    size="$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")"
    echo "remove incomplete: $dest ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B"))"
    rm -f "$dest"
  fi

  echo "download -> $dest"
  echo "  URL: $url"

  if command -v aria2c >/dev/null 2>&1; then
    local aria_args=(-x 8 -s 8 -k 1M -o "$name" -d "$OUT" "$url")
    if [[ -n "${HF_TOKEN:-}" ]]; then
      aria_args+=(--header="Authorization: Bearer ${HF_TOKEN}")
    fi
    aria2c "${aria_args[@]}"
  else
    curl -fL "${auth_header[@]}" --retry 5 --retry-delay 10 \
      --progress-bar -o "$dest" "$url"
    echo ""
  fi

  if ! is_complete "$name"; then
    echo "警告: $dest が未完成です。再実行してください。" >&2
    return 1
  fi
  echo "done: $dest ($(du -h "$dest" | cut -f1))"
}

echo "Downloading MelodyFlow weights (direct path, no .cache)"
echo "OUT=$OUT"
echo ""

echo "[1/2] compression_state_dict.bin (~230MB)"
download_one "compression_state_dict.bin"

echo ""
echo "[2/2] state_dict.bin (~3.8GB) — 回線次第で長時間かかります"
download_one "state_dict.bin"

echo ""
echo "Saved:"
ls -lh "$OUT"/*.bin
