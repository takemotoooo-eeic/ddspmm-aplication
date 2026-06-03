#!/usr/bin/env bash
# MelodyFlow 用 T5 (t5-base) をホストの backend/.cache/huggingface に取得する。
# compose.yaml がこのディレクトリを api コンテナの /root/.cache/huggingface にマウントする。
#
# 手元の Mac などで実行 → rsync/scp でサーバへ渡す場合:
#   LOCAL=backend/.cache/huggingface
#   rsync -avP "${LOCAL}/" user@host:/path/to/ddspmm-aplication/backend/.cache/huggingface/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${ROOT}/.cache/huggingface"
HUB="${CACHE}/hub"
MODEL_DIR="${HUB}/models--t5-base"
VENV_PY="${ROOT}/.venv/bin/python"

export HF_HOME="${CACHE}"
export HUGGINGFACE_HUB_CACHE="${HUB}"

mkdir -p "${HUB}"

is_t5_cache_ready() {
  [[ -d "${MODEL_DIR}/snapshots" ]] || return 1
  if find "${MODEL_DIR}/blobs" -name '*.incomplete' 2>/dev/null | grep -q .; then
    return 1
  fi
  local snap
  snap="$(find "${MODEL_DIR}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)"
  [[ -n "${snap}" && -f "${snap}/config.json" ]]
}

if is_t5_cache_ready; then
  echo "Already exists: ${MODEL_DIR}"
  du -sh "${CACHE}"
  exit 0
fi

echo "Removing incomplete T5 downloads (if any)..."
find "${MODEL_DIR}" -name '*.incomplete' -delete 2>/dev/null || true

echo "Downloading t5-base (~900MB) into:"
echo "  ${CACHE}"
echo "This directory is mounted into the api container as /root/.cache/huggingface"
echo ""

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "Using HF_TOKEN for authenticated download."
fi

download_with_hf_cli() {
  local cli=""
  if command -v hf >/dev/null 2>&1; then
    cli="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    cli="huggingface-cli"
  else
    return 1
  fi
  echo "Using ${cli} download t5-base..."
  "${cli}" download t5-base
}

download_with_python() {
  local py=""
  if [[ -x "${VENV_PY}" ]] && "${VENV_PY}" -c "import transformers" 2>/dev/null; then
    py="${VENV_PY}"
    echo "Using ${py} (backend .venv)..."
  elif command -v python3 >/dev/null 2>&1 && python3 -c "import transformers" 2>/dev/null; then
    py="python3"
    echo "Using ${py}..."
  else
    return 1
  fi
  # backend .venv は tensorflow も入るため、CLI 取得を優先。Python 経路は TF 初期化を抑える。
  TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 "${py}" <<'PY'
from transformers import T5EncoderModel, T5Tokenizer

print("Fetching T5 tokenizer...")
T5Tokenizer.from_pretrained("t5-base")
print("Fetching T5 encoder weights...")
T5EncoderModel.from_pretrained("t5-base")
print("t5-base ready.")
PY
}

if download_with_hf_cli; then
  :
elif download_with_python; then
  :
else
  echo "ERROR: hf (huggingface-cli) または transformers が必要です。" >&2
  echo "  pip install -U huggingface_hub   # hf download が使えるように" >&2
  echo "  ※ uv run は使いません（.venv が root 所有だと Permission denied になります）" >&2
  exit 1
fi

if ! is_t5_cache_ready; then
  echo "ERROR: T5 cache looks incomplete under ${MODEL_DIR}" >&2
  exit 1
fi

echo ""
echo "Done. Cache size:"
du -sh "${CACHE}"
echo ""
echo "Next: docker compose up -d api   # or: dcu api -d"
