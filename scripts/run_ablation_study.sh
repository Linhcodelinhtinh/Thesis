#!/usr/bin/env bash
# run_ablation_study.sh
# 4-way Ablation Study on LIBERO-10 Tasks 0 & 1 (Init 0..4, 10 shared episodes, seed 42)
# Conditions:
#   1. original_prompt: Baseline no-memory control (pure instruction)
#   2. shadow: Observation memory tracking in background, unmodified prompt to policy
#   3. static_suffix: Static initial detection suffix appended at all steps
#   4. validated_dynamic: Validated dynamic observation memory with grounded contract

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv_smolvla/bin/activate" ]; then
    source .venv_smolvla/bin/activate
fi

unset PYOPENGL_PLATFORM LIBGL_ALWAYS_SOFTWARE
export MUJOCO_GL=glx
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

CONDITION="${1:-all}"
CHECKPOINT="resources/checkpoints/smolvla_libero"

run_condition() {
    local cond_name="$1"
    local mem_cond="$2"
    local mem_writer="$3"
    local out_dir="experiments/results/v2_memory/ablation_${cond_name}"

    echo "================================================================="
    echo "Starting Ablation Condition: ${cond_name}"
    echo "Memory Condition: ${mem_cond}, Writer: ${mem_writer}"
    echo "Output Directory: ${out_dir}"
    echo "================================================================="

    mkdir -p "${out_dir}"

    local writer_arg=""
    if [ -n "${mem_writer}" ]; then
        writer_arg="--memory-writer ${mem_writer}"
    fi

    python3 scripts/run_benchmark.py \
        --preset custom \
        --task-suite libero_10 \
        --task-ids 0 1 \
        --initial-state-ids 0 1 2 3 4 \
        --model-name smolvla_libero \
        --checkpoint "${CHECKPOINT}" \
        -s 10 \
        --seed 42 \
        --memory-condition "${mem_cond}" \
        ${writer_arg} \
        --output-dir "${out_dir}"

    echo "Completed Ablation Condition: ${cond_name}"
}

case "$CONDITION" in
    original_prompt)
        run_condition "original_prompt" "off" ""
        ;;
    shadow)
        run_condition "shadow" "text_shadow" "observation"
        ;;
    static_suffix)
        run_condition "static_suffix" "text_static_suffix" "observation"
        ;;
    validated_dynamic)
        run_condition "validated_dynamic" "text_only" "observation"
        ;;
    all)
        echo "Running full 4-way ablation sequence..."
        run_condition "original_prompt" "off" ""
        run_condition "shadow" "text_shadow" "observation"
        run_condition "static_suffix" "text_static_suffix" "observation"
        run_condition "validated_dynamic" "text_only" "observation"
        echo "All 4 ablation conditions completed successfully!"
        ;;
    *)
        echo "Unknown condition: $CONDITION"
        echo "Usage: $0 [original_prompt|shadow|static_suffix|validated_dynamic|all]"
        exit 1
        ;;
esac
