#!/usr/bin/env bash
# Custom V2 Mem evaluation; activate the policy venv before calling this script.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# Native Git avoids expensive cross-filesystem stat scans on WSL-mounted NTFS.
# Both choices perform the same revision/diff checks; the choice is recorded.
if [[ "${REPO_ROOT}" == /mnt/* ]] && command -v git.exe >/dev/null 2>&1; then
  export MEM_EVAL_GIT="${MEM_EVAL_GIT:-git.exe}"
fi
STAGE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "${STAGE}" in preflight|smoke|validation|test|all) ;; *) echo "Usage: $0 {preflight|smoke|validation|test|all} [CLI options]" >&2; exit 2;; esac

export POLICY_PYTHON="${POLICY_PYTHON:-$(command -v python)}"
export SIM_PYTHON="${SIM_PYTHON:-${REPO_ROOT}/.venv_libero_reference/bin/python}"
export DETECTOR_PYTHON="${DETECTOR_PYTHON:-${REPO_ROOT}/.venv_memory_wsl/bin/python}"
for role in POLICY_PYTHON SIM_PYTHON DETECTOR_PYTHON; do
  if [[ ! -x "${!role}" ]]; then
    echo "Missing ${role}: ${!role}. Select the corresponding isolated Linux environment." >&2
    exit 1
  fi
done
if [[ "${STAGE}" == "all" ]]; then
  # Test is an explicit later action after selecting a candidate on validation.
  bash "${BASH_SOURCE[0]}" smoke "$@"
  bash "${BASH_SOURCE[0]}" validation "$@"
  echo "Smoke and validation complete. Select the candidate before requesting stage test."
  exit 0
fi
split=val
states=0
extra=()
case "${STAGE}" in
  preflight) extra+=(--preflight-only);;
  validation) states=all;;
  test) split=test; states=all;;
esac
output="experiments/evaluation/results_mem_2/${STAGE}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p experiments/evaluation/results_mem_2/launcher_logs
console_log="experiments/evaluation/results_mem_2/launcher_logs/$(basename "${output}").log"
echo "Console log: ${console_log}"
echo "Policy: ${POLICY_PYTHON}"
echo "Simulator: ${SIM_PYTHON}"
echo "Detector: ${DETECTOR_PYTHON}"
if "${POLICY_PYTHON}" -u scripts/evaluate_mem_adapter.py \
  --adapter-dir "${ADAPTER_DIR:-experiments/training/results_mem_2/mem-adapter-20261105-083543/pretrained_model}" \
  --base-checkpoint "${BASE_CHECKPOINT:-resources/checkpoints/smolvla_libero}" \
  --split-lock "${SPLIT_LOCK:-experiments/training/results_mem_2/mem-adapter-v1/dataset-split.lock.json}" \
  --split "${split}" --states "${states}" --tasks T1 T3 --conditions B0 F1 \
  --policy-seed 42 --output-dir "${output}" \
  --sim-python "${SIM_PYTHON}" --detector-python "${DETECTOR_PYTHON}" "${extra[@]}" "$@" 2>&1 | tee "${console_log}"; then
  run_status=0
else
  run_status=$?
fi
echo "Launcher exit code: ${run_status}" | tee -a "${console_log}"
if [[ "${run_status}" -ne 0 ]]; then exit "${run_status}"; fi
echo "${STAGE} completed; inspect the report in the selected output directory."
