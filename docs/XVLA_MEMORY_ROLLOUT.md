# XVLA memory rollout (E2 derivative)

The entry point is `scripts/xvla_v2/evaluate_memory.py`. It accepts F0 or F1
weights with `--memory on|off`. F0 with memory on is an inference intervention
on instruction-only training; the manifest records that explicitly.

Online prompts come from the training shard pipeline: `rgb_relations_v1`,
`relation_text_v1`, `relevant`, pinned SmolVLA tokenizer for the 48-token
projection budget. The exact resulting string goes into XVLA's
`language_instruction`; its own tokenizer must also fit 50 tokens including
special tokens. Overflow raises rather than silently truncating facts.
No hand-written template or evaluator count is supplied to the policy.

The shared E2 runner updates the writer at observations 0…N (including between
queries), with only RGB/gripper evidence and the previously executed action.
Queries occur at 0,10,…; the evaluator determines terminal success at 1,000
actions or irreversible overshoot. Locked raw state vectors, simulator seed 0,
20 delta warmup actions, 20 Hz, and the existing 1,020-step simulator envelope
are preserved. Each policy episode restarts the inference server with seed 42.

XVLA uses its native 30×20 prediction (active Panda block 30×10), author image
flip, gripper threshold and rotation decoder. The simulator changes to absolute
OSC only after warmup and resets to delta before the next warmup. The first
10 actions execute before another query. Proprio starts from controller pose;
later queries use the last predicted pose in the previous full chunk, as in
the author client. This prefix schedule is an explicit E2 derivative of the
author's full-30-action execution. Model preprocessing/normalization is unchanged.

## Historical memory source

The current writer source differs from the XVLA training manifests in seven
files. The SmolVLA E2 `batch-0-1/code_snapshot.zip` contains all 13 Python files
with exactly the training hashes. The following creates a separate source
directory; it does not overwrite the working tree:

```bash
python scripts/xvla_v2/memory_source.py \
  --archive experiments/evaluation/e2_unseen_author_v2/batch-0-1/code_snapshot.zip \
  --destination experiments/runtime/xvla_e2_memory_source
```

The source directory is checked against the shard manifest before imports.
Only attested memory/schema/perception modules are selected by the orchestrator;
the detector subprocess executes the archived worker with its archived source.
Loaded source is copied into the run artifacts. A verified local copy is already
available at `experiments/runtime/xvla_e2_memory_source_20261121`.

## Preflight and execution

Run from the repository root in Linux/WSL with a Python >=3.10 orchestration
environment, plus separate simulator, XVLA inference and detector environments.
Pass the actual environment executables; the variables below are placeholders.

```bash
python scripts/xvla_v2/evaluate_memory.py \
  --adapter-dir experiments/training/xvla_v2_f1_step5000 --arm F1 --memory on \
  --writer-source-root experiments/runtime/xvla_e2_memory_source_20261121 \
  --sim-python "$SIM_PYTHON" --inference-python "$XVLA_PYTHON" \
  --detector-python "$DETECTOR_PYTHON" \
  --split unseen --states all --tasks T1 T3 --policy-seed 42 \
  --smolvla-reference experiments/evaluation/e2_unseen_author_v2/batch-0-1/run_manifest.json \
  --require-matched-reference \
  --output-dir experiments/evaluation/xvla_e2_f1_preflight --preflight-only
```

Each historical batch manifest contains only part of the 40-episode roster.
For a strict check against that batch use exactly its states, e.g. `--states 0,1`
for batch-0-1. For the full roster pass all ten batch manifests to
`--smolvla-reference experiments/evaluation/e2_unseen_author_v2/batch-*/run_manifest.json`
(Linux shell glob). The runner verifies that source, runtime, treatment and seed
agree across batches and rejects duplicate task/state rows. The gate rejects a full run
against a two-state reference. It also compares shared evaluator/writer source,
projection, seed, simulator and detector runtimes. A missing reference remains
`UNVERIFIED_NO_REFERENCE`; mismatches are `DESCRIPTIVE_ONLY`.

After preflight passes, omit `--preflight-only` and use a new output directory.
For the matched F0 control use `--arm F0`, its F0 adapter, and `--memory off`.
For the requested F0 memory-on intervention use `--memory on` and preserve its
separate weight-arm label when interpreting results.

Artifacts include manifests and source hashes, state hashes, per-observation
memory trace, exact query strings, native raw chunks, decoded chunks, actions,
initial/final observations, videos, per-episode summaries, and task reports.

## Comparison scope and verification

`MATCHED_TASK_PROTOCOL_CROSS_BACKBONE` means the audited task conditions match.
XVLA absolute actions, native chunk 30 and official input processing differ from
SmolVLA delta actions and chunk 50. Results describe two policies on a common
task protocol; they do not isolate a causal memory effect and are always
`NON-COMPARABLE_OFFICIAL_PAPER`. Wall time includes the decoder IPC and is not a
pure model-inference latency comparison.

Unit checks cover author payload parity, native proprio update/reset, token
overflow, comparison gates, existing causal writer rollout and default Mem CLI.
Real CUDA/simulator rollout remains a separate runtime gate; unit checks do not
certify a benchmark score.
