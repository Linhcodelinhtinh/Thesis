#!/usr/bin/env bash
set -euo pipefail

# Run from a writable copy of this repository in /kaggle/working.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Python executable to use: defaults to python3 from Kaggle container,
# or an explicit virtualenv if TRAIN_ENV exists.
PYTHON_BIN="python3"
if [ -n "${TRAIN_ENV:-}" ] && [ -x "${TRAIN_ENV}/bin/python" ]; then
    PYTHON_BIN="${TRAIN_ENV}/bin/python"
fi

ASSETS_DIR="${ASSETS_DIR:-/kaggle/temp/sft-assets}"
DATA_DIR="${DATA_DIR:-/kaggle/working/libero10-memory}"
RAW_DIR="${RAW_DIR:-/kaggle/temp/libero-raw}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/memory-sft-run}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export HF_HUB_DISABLE_XET=1

case "${1:-help}" in
  install)
    echo "Using Python: $($PYTHON_BIN --version) at $(command -v "$PYTHON_BIN" || echo "$PYTHON_BIN")"
    echo "Installing required SFT training dependencies..."
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install -r envs/kaggle_memory_sft.txt
    ;;
  assets)
    "$PYTHON_BIN" scripts/memory_sft_assets.py fetch "$ASSETS_DIR"
    ;;
  prepare)
    shift
    "$PYTHON_BIN" scripts/prepare_memory_sft.py \
      --suite libero_10 --source-root "$RAW_DIR" --output-dir "$DATA_DIR" \
      --tokenizer-path "$ASSETS_DIR/backbone" --download "$@"
    ;;
  train)
    shift
    "$PYTHON_BIN" -m accelerate.commands.launch --multi_gpu --num_processes 2 \
      --num_machines 1 --mixed_precision fp16 --dynamo_backend no \
      scripts/train_memory_sft.py --assets-dir "$ASSETS_DIR" --data-dir "$DATA_DIR" \
      --output-dir "$OUTPUT_DIR" --expected-gpus 2 "$@"
    ;;
  *)
    echo 'Usage: bash scripts/kaggle_memory_sft.sh install|assets|prepare|train [extra arguments]'
    exit 2
    ;;
esac
