#!/usr/bin/env bash
set -euo pipefail

# Run from a writable copy of this repository in /kaggle/working.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
TRAIN_ENV="${TRAIN_ENV:-/kaggle/temp/memory-sft-venv}"
ASSETS_DIR="${ASSETS_DIR:-/kaggle/temp/sft-assets}"
DATA_DIR="${DATA_DIR:-/kaggle/working/libero10-memory}"
RAW_DIR="${RAW_DIR:-/kaggle/temp/libero-raw}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/memory-sft-run}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export HF_HUB_DISABLE_XET=1

case "${1:-help}" in
  install)
    python -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13), "Use Python 3.10-3.12 for the pinned training stack"'
    python -m venv "$TRAIN_ENV"
    "$TRAIN_ENV/bin/python" -m pip install --upgrade 'pip==25.2'
    "$TRAIN_ENV/bin/python" -m pip install 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu118
    "$TRAIN_ENV/bin/python" -m pip install -r envs/kaggle_memory_sft.txt
    "$TRAIN_ENV/bin/python" -m pip check
    ;;
  assets)
    "$TRAIN_ENV/bin/python" scripts/memory_sft_assets.py fetch "$ASSETS_DIR"
    ;;
  prepare)
    shift
    "$TRAIN_ENV/bin/python" scripts/prepare_memory_sft.py \
      --suite libero_10 --source-root "$RAW_DIR" --output-dir "$DATA_DIR" \
      --tokenizer-path "$ASSETS_DIR/backbone" --download "$@"
    ;;
  train)
    shift
    "$TRAIN_ENV/bin/accelerate" launch --multi_gpu --num_processes 2 \
      --num_machines 1 --mixed_precision fp16 --dynamo_backend no \
      scripts/train_memory_sft.py --assets-dir "$ASSETS_DIR" --data-dir "$DATA_DIR" \
      --output-dir "$OUTPUT_DIR" --expected-gpus 2 "$@"
    ;;
  *)
    echo 'Usage: bash scripts/kaggle_memory_sft.sh install|assets|prepare|train [extra arguments]'
    exit 2
    ;;
esac
