#!/usr/bin/env bash
# ==============================================================================
# Step 5: Final Held-Out Test Evaluation (40 Episodes)
# Protocol: CUSTOM_V2_LIBERO_MEM / NON-COMPARABLE_OFFICIAL_PAPER
# Specification: docs/RESULTS_MEM_2_TEST_PLAN.md Section P3
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STATES="${1:-all}"
POLICY_SEED="${2:-42}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
TARGET_DIR="experiments/evaluation/results_mem_2/step5_test_${TIMESTAMP}"

SIM_PYTHON="/mnt/d/Thesis_26/.venv_libero_reference/bin/python"
DETECTOR_PYTHON="D:\\Thesis_26\\.venv_detector\\Scripts\\python.exe"
ADAPTER_DIR="experiments/training/results_mem_2/mem-adapter-20261105-083543/pretrained_model"
BASE_CHECKPOINT="resources/checkpoints/smolvla_libero"
SPLIT_LOCK="experiments/training/results_mem_2/mem-adapter-v1/dataset-split.lock.json"

mkdir -p "$TARGET_DIR"

echo "================================================================="
echo "  Step 5: LIBERO-Mem Final Held-Out Test (40 Episodes Target)   "
echo "  States: $STATES | Policy Seed: $POLICY_SEED                    "
echo "================================================================="

B0_DIR="${TARGET_DIR}/test_B0"
F1_DIR="${TARGET_DIR}/test_F1"

echo ""
echo ">>> [1/2] Starting B0 Test Evaluation (Base Policy) -> $B0_DIR"
python scripts/evaluate_mem_adapter.py \
    --split test \
    --tasks T1 T3 \
    --conditions B0 \
    --states "$STATES" \
    --policy-seed "$POLICY_SEED" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --split-lock "$SPLIT_LOCK" \
    --sim-python "$SIM_PYTHON" \
    --use-wsl \
    --output-dir "$B0_DIR"

echo ""
echo ">>> [2/2] Starting F1 Test Evaluation (LoRA Step 3000 + Memory) -> $F1_DIR"
python scripts/evaluate_mem_adapter.py \
    --split test \
    --tasks T1 T3 \
    --conditions F1 \
    --states "$STATES" \
    --policy-seed "$POLICY_SEED" \
    --adapter-dir "$ADAPTER_DIR" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --split-lock "$SPLIT_LOCK" \
    --sim-python "$SIM_PYTHON" \
    --use-wsl \
    --detector-python "$DETECTOR_PYTHON" \
    --output-dir "$F1_DIR"

echo ""
echo ">>> Generating Consolidated Final Test Report..."
python scripts/summarize_mem_evaluation.py \
    "$B0_DIR" "$F1_DIR" \
    --title "Step 5: LIBERO-Mem Final Held-Out Test Report" \
    --output-report "${TARGET_DIR}/test_consolidated_report.md" \
    --output-csv "${TARGET_DIR}/test_consolidated_summary.csv"

echo "Final Test complete! Results saved to $TARGET_DIR"
