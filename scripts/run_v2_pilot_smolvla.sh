#!/usr/bin/env bash
set -e

# Chuyển về thư mục gốc của project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Kích hoạt virtual environment
if [ -f ".venv_smolvla/bin/activate" ]; then
    source .venv_smolvla/bin/activate
elif [ -f ".venv_smolvla/Scripts/activate" ]; then
    source .venv_smolvla/Scripts/activate
fi

# Cấu hình biến môi trường đồ họa / MuJoCo cho GPU acceleration
unset PYOPENGL_PLATFORM LIBGL_ALWAYS_SOFTWARE
export MUJOCO_GL=glx
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

# Chạy benchmark (cho phép truyền thêm hoặc ghi đè tham số qua "$@")
python3 scripts/run_benchmark.py \
  --preset libero_10 \
  --model-name smolvla_libero \
  --checkpoint resources/checkpoints/smolvla_libero \
  --execution-horizon 10 \
  --memory-condition text_only \
  --memory-writer observation \
  --output-dir experiments/results/v2_memory/prompt_shoren_obs_text \
  --seed 42 \
  "$@"
