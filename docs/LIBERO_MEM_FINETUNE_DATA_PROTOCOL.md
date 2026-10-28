# Mixed SmolVLA fine-tuning: data and evaluator contract

Implementation: 2026-10-26. Scope: **LIBERO-10 retention evaluation + LIBERO-Mem T1/T3**.
The prepared pilot uses **7/10 LIBERO-10 tasks for replay** and all 198 T1/T3 demonstrations.
Three author source files could not be completed because of CDN transfer failures; no substitutes were used.
The selected and excluded tasks are explicit in `resources/manifests/libero_10_sft_pilot7_sources.json`.
This is a custom V2 adaptation experiment, not an official LIBERO or LIBERO-Mem paper score.
It does not establish retention across all LIBERO suites.

## Experiment and notebook

Use `notebooks/kaggle_libero_mem_finetune.ipynb`. It is based on the successful
`notebooks/working_notebooks/memory-sft-nb.ipynb`, which is preserved unchanged.
The new notebook keeps Kaggle's PyTorch/CUDA, installs the LeRobot 0.6.1 /
Transformers 5.5.4 versions observed in the reference output, and runs directly
in the notebook environment. No simulator, detector, venv, manual source patch,
or hard-coded GPU model is needed in training.

| Run | Starting checkpoint | Training data | Memory |
|---|---|---|---|
| B0 | Pinned original SmolVLA-LIBERO | None | Off |
| F0 | Same original checkpoint | Locked seven-task base subset + T1/T3 | Off |
| F1 | Same original checkpoint | Same selected frames and action targets | Predicted causal RGB writer on Mem; original instruction on base |
| F1-off | F1 checkpoint | No additional training | Off at evaluation; a diagnostic ablation |

F1 does **not** start from F0 for the primary matched comparison. Starting from
F0 is a separate staged adaptation experiment with a different total budget.
F0 can use either the no-memory shard artifact or the complete F1 artifact with
`condition=no_memory`; the dataset substitutes the original instruction.

Pilot settings: action-expert LoRA rank/alpha 16, frozen VLM, LR 1e-4, global
batch 32, seed 42, memory dropout 0, 3,000 successful optimizer updates, save
and validation every 500 updates. These are starting settings, not guarantees
of retention or Mem success. Both runs use the same settings and same sample
schedule. Changing GPU count is supported; compare runs with the same effective
batch and document numerical/hardware differences.

Sampler: 50% base / 50% Mem **per global optimizer batch**, shuffled task
round-robin within each domain, uniform episode then uniform frame. Long
trajectories therefore do not dominate solely through their length. The run
records the actual ratio, epoch-0 schedule hash, split hash and sample-selection
hash. Tail action chunks remain within their episode and carry padding masks.
All three known padding-key spellings are supplied; the installed policy must
consume a recognized key. No loss implementation is replaced.

## Immutable initial-state split

`scripts/lock_memory_sft_split.py` hashes the exact full initial-state vector as
little-endian float64, groups it globally across tasks/domains, and assigns
approximately 80/10/10 train/validation/test by deterministic seed. Connected
task families share assignment. A canonical payload hash seals the lock.

T1/T3 demos **1, 5, 31** were already inspected while developing writer v3. Those
states, including every episode sharing them, are forced into train before
held-out selection. Do not use test to calibrate the writer, select checkpoints,
or choose the replay ratio. Val is for development; test is a final paired report.
This split is disjoint for this adaptation only: original checkpoint pretraining
overlap cannot be certified. It is not the author's claimed 100/20 split.

The final lock contains source file hashes, source revisions, evaluator hash,
episode membership, state hashes and exact Mem reset vectors. Protocol/source-text
hashes normalize LF/CRLF so Windows and Linux checkouts agree. Preparation,
prompt-index generation and Mem evaluation consume the same file. Locks are
never overwritten; changes require a new version and new prepared artifact.

## Source, observation and target contract

Sources are pinned in `resources/manifests/libero_10_sft_sources.json` and
`resources/manifests/libero_mem_planning_sources.json`. All downloaded bytes are
verified against author-release Git LFS SHA256 values before use. No synthetic
demonstrations or fallback resource files are accepted.

| Source | Canonical input | First target at input frame i |
|---|---|---|
| Official LIBERO base HDF5 | Raw OpenGL RGB as stored | `actions[i+1]`; last source observation excluded |
| Author refined Mem HDF5 | Undo stored vertical flip once | `actions[i]` |

The base recorder stores its observation after executing the same-index action;
the Mem refiner records the current observation before the same-index action.
These mappings are intentionally different. Base's executed action at input i
is i; Mem's is i-1, and Mem reset frame has no executed action. `samples.jsonl`
records observation index, first target index and last executed action index.

Both cameras are retained. The existing adapter resize and official
`LiberoProcessorStep` then run unchanged, including its image transform and
9D source robot state to 8D policy state conversion. Quaternion xyzw, EEF and
gripper ordering are validated against source observations. Action7 is the
native OSC_POSE target; original checkpoint normalization is preserved rather
than refitted. Depth, segmentation, future annotations and simulator facts are
not policy inputs. Source rewards/dones are not trusted as sequence labels.

Prepared artifacts contain real RGB/state/action HDF5 shards, not just a prompt
index. Preparation validates finite values, dimensions, camera dtype, quaternion
norm, action range, episode lengths, prompt flags, exact tokenizer budget 48,
and every shard hash. Training validates the completed manifest and file hashes.
Sample selection, source provenance, counts and shard membership must agree with
the lock; missing episodes are errors, not silently skipped samples.

F1 requires a **complete predicted RGB v3 index for every locked Mem frame**.
The indices must share the same split, source hashes, detector/config/writer
version, tokenizer and image/action convention. They are created sequentially
from observation prefixes and the last executed action. Correct count/object
labels from annotation are not substituted for predictions. Memory coverage
is not event accuracy; an empty or uncertain memory can be correct. Writer
accuracy and policy benefit still require separate evaluation.

## Mem evaluator protocol

Canonical file: `configs/benchmarks/libero_mem_t1_t3_protocol.json`.
Upstream checkout: `libero-mem/libero-mem@0eee0defa4024e51511b6e15a79f30e9620209de`.
The loader verifies the pinned BDDL, goal evaluator, predicate, wrapper and
refiner source hashes. It uses the real upstream simulator and refuses a
fixture/fallback physics path. Only T1 and T3 are enabled in this protocol.

T1 has one completed-return milestone; T3 has three. Counts come from the
upstream `_satisfied_subgoals`, with the physical upstream `On` predicate.
No writer geometry is used to award success. Critical upstream behavior:

- `_check_success(inc=False)` can still commit a milestone/reset counters.
- A `True` return can mean an intermediate milestone, not the whole task.
- Live/nonlive counters are cumulative between milestones, not consecutive dwell.
- `reset_subgoal_progress()` omits `_overshot`; the adapter explicitly clears it.

Reset restores and verifies the exact locked initial state, warms up 20 native
open-gripper no-op actions, clears progress and overshoot, and observes reset
frame 0. Each policy-visible observation gets exactly one explicit
`_check_success(inc=True)` before the next action. The upstream reward/step
`inc=False` calls are preserved. Logging only reads the resulting state.

The protocol fixes 256x256 two-camera input, Panda/default gripper, the recorded
OSC_POSE controller configuration, 20 Hz simulation, policy chunk 50 and execution
horizon 10. Finite policy action7 goes unchanged to the official controller;
its native input saturation is preserved, without extra adapter clipping.

**Custom terminal rule:** run 1,000 policy actions; do not early-stop on a
transient full/intermediate success. Stop early only on irreversible upstream
overshoot. Full success at the horizon requires every milestone, current final
predicate true and no overshoot. SCR reports milestone fraction separately and
can equal 1 despite overshoot. The public author evaluation runner has no
`libero_mem` horizon branch, so this fixed horizon/terminal rule is an explicit
local choice, not claimed to reproduce the paper's rollout protocol.

Runtime pins are simulator-only: robosuite 1.4.0, numpy 1.22.4, bddl 1.0.1,
robomimic 0.2.0, MuJoCo 3.2.3, Python 3.8.x. The observed Python 3.8.20 / MuJoCo
stack is not a claim of equivalence to the author's runtime or the V1 official
Python 3.8.13 reference environment. This isolated simulator is **not installed
on Kaggle**. Writer/model inference pauses simulation; no real-time 20 Hz claim.

Use the same locked validation/test states, task instructions, horizon, chunk
execution, seed schedule and protocol for B0/F0/F1. Report paired per-state
results, per-task SR/SCR and uncertainty. A 10-state task test has wide uncertainty;
repeated seeds on the same state are correlated, not additional independent states.
Evaluate base retention separately with the existing pinned LIBERO reference
evaluator and the same fixed state list for all checkpoints. Adaptation data
holdout alone does not certify that the official base evaluation states were
unseen in checkpoint training.

## Preparation commands (outside Kaggle)

Commands below use paths relative to the repository root; choose the isolated
preparation Python with h5py/numpy/ijson/tokenizers/huggingface-hub installed.
`TOKENIZER` is the local pinned backbone directory with verified tokenizer.json.

```text
python scripts/download_memory_training_sources.py
python scripts/lock_memory_sft_split.py --base-root resources/demonstrations --base-inventory resources/manifests/libero_10_sft_pilot7_sources.json --output resources/manifests/libero_mem_t1_t3_base7_split_v1.json
python scripts/prepare_mixed_memory_sft.py --split-lock resources/manifests/libero_mem_t1_t3_base7_split_v1.json --tokenizer-path TOKENIZER --output-dir experiments/training/libero10base7_mem_t1t3_f0_v1
python scripts/validate_mixed_memory_sft.py --data-dir experiments/training/libero10base7_mem_t1t3_f0_v1
python scripts/package_memory_sft.py
```

To expand replay to all ten base tasks, finish the pinned downloads, omit `--base-inventory`,
and create a **new** split/artifact version. Never overwrite the pilot lock.

For F1, run `prepare_libero_mem_sft.py` separately for both exact task names from
the protocol, with the same `--split-lock`, pinned detector and tokenizer.
Omit `--max-episodes` and `--split` to include all locked episodes. This expensive
RGB perception pass belongs outside the training notebook. Then run the same
mixed preparation command into a new directory, adding
`--memory-index T1_INDEX_DIRECTORY T3_INDEX_DIRECTORY`. Validate the two artifacts
with `validate_mixed_memory_sft.py --data-dir F0_DIRECTORY --compare F1_DIRECTORY`.
Do not relabel the earlier six-demo development index as a complete F1 dataset.

Upload the source zip and the selected prepared directory as Kaggle inputs;
adjust only `CODE_SOURCE`, `DATA_DIR`, and `CONDITION` in the new notebook.
The notebook runs a separate two-update smoke before the full run and creates
fresh output directories. Main training starts from the original assets, not
the smoke checkpoint. Save a Kaggle Version to preserve outputs/checkpoints.
If interrupted, trainer `--resume PATH` restores the saved epoch/batch, RNG,
optimizer and scheduler with the original schedule/configuration. Do not change
the planned total updates while resuming.

Validation flow loss is a diagnostic, not rollout SR. A final export is the
last checkpoint, not automatically the best checkpoint. Select policy settings
on validation rollout with a base-retention constraint, then report test once.
No training launch or policy-performance claim is made by the data preparation.

## Verification evidence

`scripts/audit_libero_mem.py` now replays real recorded actions through MuJoCo.
It checks exact-state reset repeatability, evaluator call cadence, sequence
progress, pose deviation and sampled camera agreement. It does not execute a
mock policy and does not call demonstration replay a policy score.

Conformance replay of demos 1, 5 and 31: T1 completed 1/1 and T3 completed 3/3
in all six episodes,
final predicate true, no overshoot, repeated-reset error 0. Explicit increment
calls matched the number of advanced observations; upstream non-incrementing
calls were exactly twice that number. Maximum EEF deviations were approximately
4.03 mm (T1) and 10.00 mm (T3); therefore action replay is not asserted to be
bit-exact to the recorded source trajectory. Detailed reports are under
`experiments/audits/mem_training_v1/real_replay_demo1/` and `real_replay_demo5_31/`.

Unit fixtures verify state grouping and development exclusions, corruption
rejection, canonical image transform, episode-bounded masks and matched balanced
sampling. Synthetic fixtures are never benchmark data. The new notebook's code
cells are syntax checked; it still requires its own smoke on the target Kaggle
runtime even though the reference notebook has already passed there.


Additional verified evidence: the real-source QA conversion contains 15 episodes / 3,717 samples.
All 3,717 state/action rows match their source mapping; RGB checks at first/middle/last
frame passed for both cameras (45 frames each). The real installed LeRobot processor
and evaluation adapter produced identical inputs for 30 first/tail samples.
The first GPU smoke ran two updates on one RTX 4060 (peak allocated 2.39 GiB),
exported and strictly reloaded a finite (50,7) action chunk with original normalizers.
This used local LeRobot 0.4.4, not Kaggle's reference 0.6.1.
After the portable protocol-hash fix, QA artifact v2 was regenerated and source
audit passed with the same sample-selection hash. A CPU probe verified iterator
creation preserves the policy RNG. The optional post-change WSL GPU/resume repeat
was not run: automatic approval timed out, and the user chose to retain existing
verification results. Do not report exact resumed-weight equivalence or a new
Kaggle 2-T4 smoke as measured.


## Locked pilot selection

Lock: `resources/manifests/libero_mem_t1_t3_base7_split_v1.json`
(`b2216d9921ffe62591db526334e2d0ad12d83ebdeabe7bd4c9a1aec8971326b9`).
There are 548 episodes / 450 distinct initial states. The state partitions contain
360 train, 45 validation and 45 test states, with zero cross-partition overlap.
Base contributes 280/35/35 episodes; Mem contributes 158/20/20. T1/T3 share the
same held-out states. No state used for writer development appears in validation
or test. These counts are episode/state counts, not training-frame counts.

Artifact directory: `experiments/training/libero10base7_mem_t1t3_f0_v1/`.
It is F0 data: zero memory-bearing frames are intentional, not a writer failure.
A full F1 predicted-memory artifact has not been precomputed. Its builder rejects
incomplete indices, wrong split hashes and inconsistent writer/detector versions.

The completed F0 artifact contains **143,671 samples** in nine HDF5 shards
(13,901,974,556 bytes for the shards, approximately 13.9 GB):

| Domain | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Seven base tasks | 80,350 | 9,949 | 10,515 | 100,814 |
| Mem T1/T3 | 34,305 | 4,299 | 4,253 | 42,857 |
| Combined | 114,655 | 14,248 | 14,768 | 143,671 |

Sample-selection SHA256:
`b5008402ebbbd175a73a82f9ebd7ca43e4b6f0a2c8e930a27e8334bfc017dd0e`.
Upload the entire prepared directory, including `manifest.json`, the copied
split/protocol files, `samples.jsonl` and all nine shards; the source code zip
is a separate, small Kaggle input. Source demonstrations are not required inside
the Kaggle training dataset.

The final independent source audit passed for all 548 episodes: every one of
143,671 state/action rows matched the pinned source mapping; first/middle/last
RGB checks matched for both cameras (1,644 frames per camera). All source and
prepared-shard hashes passed. Report:
`experiments/audits/mem_training_v1/final_pilot_data_report.json`.
Policy-adapter parity was not repeated on this full artifact; the measured
30-sample parity check and GPU smoke above remain the available policy evidence.
