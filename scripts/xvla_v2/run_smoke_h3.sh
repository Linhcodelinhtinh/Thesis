#!/usr/bin/env bash
set -e
cd /mnt/d/Thesis_26

COMMON="--adapter-dir experiments/training/xvla_v2_f1_step5000 --arm F1 --memory on \
 --writer-source-root experiments/runtime/xvla_e2_memory_source_20261121 \
 --split unseen --policy-seed 42 \
 --sim-python .venv_libero_reference_3813/bin/python \
 --detector-python /mnt/d/Thesis_26/.venv_memory/Scripts/python.exe \
 --inference-python .venv_x_vla/bin/python \
 --smolvla-reference \
experiments/evaluation/e2_unseen_author_v2/batch-0-1/run_manifest.json \
experiments/evaluation/e2_unseen_author_v2/batch-2-3/run_manifest.json \
experiments/evaluation/e2_unseen_author_v2/batch-4-5/run_manifest.json \
experiments/evaluation/e2_unseen_author_v2/batch-6-7/run_manifest.json \
experiments/evaluation/e2_unseen_author_v2/batch-8-9/run_manifest.json \
experiments/evaluation/e2_unseen_author_v2/batch-10-11/run_manifest.json \
experiments/evaluation/e2_unseen_author_v2/batch-12-13/run_manifest.json \
experiments/evaluation/e2_unseen_author_v2/batch-14-15/run_manifest.json \
experiments/evaluation/e2_unseen_author_v2/batch-16-17/run_manifest.json \
experiments/evaluation/e2_unseen_author_v2/batch-18-19/run_manifest.json \
 --halt hold"

PY=.venv_xvla_eval/bin/python

$PY scripts/xvla_v2/evaluate_memory.py $COMMON \
  --halt-source fused --halt-no-support-check --freeze-detector-after-halt \
  --tasks T1 --states 1 --port 9400 \
  --output-dir experiments/evaluation/xvla_e2_f1_halt_h3_smoke_20261123
