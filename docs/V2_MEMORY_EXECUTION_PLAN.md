# V2 External Memory — Execution Plan

**Status:** Ready to execute, V2.1 scope  
**Baseline:** Frozen SmolVLA-LIBERO internal derived baseline  
**First implementation:** Textual episodic/state memory only

This plan operationalizes the V2 roadmap in `docs/IMPLEMENTATION_PLAN.md` and
the system design in `docs/ARCHITECTURE.md`. It does not redefine the V1
benchmark or claim official LIBERO comparability.

## 1. Objective and boundaries

Test whether a lightweight, external memory layer improves a frozen VLA on
tasks where relevant event or state information is no longer recoverable from
the current observation alone.

The V2 comparison keeps the checkpoint, simulator, robot, controller, cameras,
task success predicate, action decoding, action chunk size, and execution
horizon fixed. The treatment is the memory condition and its resulting input
context. The V1 `OFF` condition must remain callable without a memory component.

### First-slice assumptions

- Implement V2.1 Text Memory before spatial memory or a combined system.
- Keep memory scoped to one episode and clear it on every environment reset.
  Cross-episode retrieval is a separate experiment and is off by default.
- Store structured events and state as the source of truth. Render text only at
  the VLA input boundary.
- Use a deterministic, simulator-grounded event writer for Stage A algorithm
  validation. Mark its evidence as oracle-derived in every record and report.
  Do not present Stage A as a realistic perception result.
- Treat the existing 40-task run as 40 tasks × 4 initial states, a paired pilot
  set. It is not a 50-state-per-task evaluation.

## 2. Invariants

1. Freeze the model checkpoint and hash, processor files, action contract,
   controller, environment settings, and evaluation code revision for each run.
2. V2 uses the baseline's audited 8D state contract and direct native Robosuite
   gripper polarity. Any future contract change requires a new baseline version.
3. With memory `OFF`, pass the original instruction through byte-for-byte and
   preserve the current policy call and action path.
4. A record may only contain evidence available at or before its event step.
   No future state, success outcome, or hidden simulator state may leak into a
   policy prompt.
5. Unknown, stale, contradictory, or low-confidence information is omitted or
   explicitly marked uncertain. Never synthesize an object location.
6. Record evidence source, episode, step, timestamp, confidence, and validity
   for each memory fact. Keep oracle and observation-derived facts separate.
7. Log the original instruction, retrieved record IDs, rendered context,
   context length, memory condition, and final policy input for every replan.
8. Keep all V2 scores labeled `LIBERO-DERIVED`; do not report them as official
   LIBERO benchmark scores.

## 3. Work packages

| Package | Scope | Main outputs | Exit condition |
|---|---|---|---|
| P0 — Baseline contract lock | Read current V1 release/run commits and freeze artifacts; confirm state/action semantics and runtime settings | `v2_baseline_lock.yaml`, hashes, run protocol | V2 run config resolves to one exact baseline and says derived/non-comparable |
| P1 — Memory data model | Define event, object state, provenance, confidence, validity, episode lifecycle | `src/memory/models.py`, schema version | Schema rejects invalid timestamps, confidence, IDs, and unsupported states |
| P2 — Store and updater | Episode-scoped event log and current-state projection; deterministic CREATE/UPDATE/CONFIRM/INVALIDATE | `src/memory/store.py`, `updater.py` | Same evidence sequence always yields the same world state; reset clears all episode data |
| P3 — Retrieval and text interface | Task-entity-conditioned retrieval, staleness filtering, stable ranking, bounded text renderer | `src/memory/retriever.py`, `src/interfaces/text_memory.py` | No relevant record produces the original instruction unchanged; retrieved text is traceable to IDs |
| P4 — Shadow integration | Collect/update/retrieve and log memory while the policy still receives the unmodified instruction | opt-in rollout hooks and shadow logs | OFF and shadow mode have identical policy inputs and actions for the same episode |
| P5 — Text-only treatment | Add rendered memory to task text before inference; leave postprocessor/actions untouched | opt-in `text_memory` condition, run config | Only policy text differs from OFF; empty memory matches OFF byte-for-byte |
| P6 — Paired pilot | Compare OFF and text-only using identical task/state IDs and frozen settings | episode artifacts, paired report | Reconciled per-episode outcomes, provenance, and memory diagnostics |
| P7 — V2.1 decision | Analyze usefulness, regressions, and failure cases before expanding modalities | V2.1 decision record | Continue, revise, or stop based on predeclared criteria; no automatic promotion |
| P8 — Spatial memory | Add 3D anchors and projection as a separate treatment | spatial writer/projector and `spatial_memory` condition | Independently validated without changing text-only behavior |
| P9 — Combined and ablations | Combine text + spatial; compare OFF, text, spatial, combined, and oracle reference | ablation runner and report | Same paired protocol and exact treatment isolation across all conditions |
| P10 — Robustness | Evaluate controlled staleness, confidence degradation, spatial noise, and corruption | robustness configs and curves | Every perturbation is explicit, seeded, and labeled derived |

## 4. Detailed V2.1 implementation sequence

### P0 — Lock the comparison point

1. Preserve the current user edits in the V1 freeze bundle; do not regenerate or
   overwrite them from the old freeze-audit script.
2. Record the V1 evaluation-run commit separately from the release commit.
3. Resolve all V2 runtime values from the frozen baseline configuration rather
   than duplicating them in Python defaults.
4. Record checkpoint, processor, LIBERO checkout, environment, task config,
   action contract, controller, camera size, `H`, `s`, and initial-state IDs.
5. Label the experiment as an internal derived comparison. Stop if the runtime
   differs from the recorded baseline or if the action/state contract is unclear.

### P1–P3 — Implement an isolated memory core

Create these modules without changing `src/models/smolvla/adapter.py`:

```text
src/memory/
  models.py       # versioned Event, ObjectMemory, Evidence, WorldMemory
  store.py        # episode-scoped store and deterministic serialization
  updater.py      # evidence-driven state transitions
  retriever.py    # task-conditioned retrieval and staleness filtering
src/interfaces/
  text_memory.py  # deterministic prompt/context renderer
```

An event must include a stable event ID, episode ID, simulation step/time,
entity IDs, event type, prior/resulting state where known, evidence source,
confidence, and validity. Keep an append-only event history and a derived latest
state; do not overwrite the only copy of history.

For the first implementation, use explicit task/entity IDs supplied by the
benchmark config. Do not infer entities from arbitrary natural-language strings
with an unvalidated parser. Retrieval order must be deterministic and bounded:
filter by episode and task entities, discard invalid/stale/low-confidence
records, then rank by current-state relevance, confidence, and recency with a
stable ID tie-break.

The renderer must produce short factual statements only. It must not turn
uncertain or missing evidence into assertions. Enforce a fixed context/token
budget, and log any truncation. If retrieval returns no usable records, return
the original instruction unchanged.

### P4 — Shadow mode before prompt injection

Add opt-in hooks around the existing rollout lifecycle:

```text
episode_start → observe/update → retrieve/render → policy replan
             → environment step → record available evidence → episode_end
```

In shadow mode, compute and save memory but pass the original task text to the
policy. This isolates event construction and retrieval from policy response.
Do not use post-episode labels to create an event at an earlier step. Any
simulator-ground-truth writer must identify itself as oracle evidence.

### P5 — Text-only condition

Insert the rendered context into the existing task string immediately before
the normal SmolVLA preprocessing call. Do not edit the model checkpoint,
normalizer, action unnormalizer, controller, simulator observation, or success
predicate. The memory condition is explicit (`off`, `text_shadow`, or
`text_only`) and is stored in each episode artifact.

The prompt template, ordering, separators, maximum context length, retrieval
thresholds, and stale-record policy become versioned run configuration. Do not
tune them on evaluation outcomes; changes create a new config/version.

## 5. Evaluation protocol

### Pilot

1. Start with the 10 LIBERO-10 tasks, initial-state IDs `[0, 1, 2, 3]`, as the
   primary long-horizon pilot. Run each condition on the same task/state pair.
2. Add a visible-target control subset from LIBERO-Object to check that memory
   does not materially harm ordinary pick-and-place behavior.
3. First run a small integration smoke set; then run the complete paired pilot.
   Keep retry policy and failure handling identical across conditions.
4. Use fixed checkpoint, code/config hashes, environment, seeds, `H=50`,
   `s=50`, 20 Hz controller, camera settings, and success semantics.
5. Treat any custom occlusion or task composition as a separate derived suite
   built from official assets and explicitly versioned task files. Never mix
   those results with the unchanged LIBERO task scores.

### Primary and diagnostic metrics

- Primary: paired task success difference (text-only minus OFF), with paired
  confidence interval and per-task/state results.
- Secondary: completed subtask count/order, time-to-success, policy replans,
  inference latency, and failure phase.
- Memory quality: event/state accuracy against simulator ground truth,
  retrieval precision/recall, stale/invalid fact rate, context length, and
  oracle-vs-non-oracle breakdown.
- Operational: missing artifacts, policy errors, simulator errors, and exact
  model input hashes for OFF/shadow/text-only.

Do not predeclare success based only on aggregate SR. Report negative results,
uncertainty, and task-level regressions. Oracle memory is an upper-bound or
diagnostic condition, not evidence that a realistic memory writer works.

## 6. Later stages

- **V2.2 Spatial only:** after text-only data and interfaces are stable, add
  world-frame anchors, confidence/staleness, camera projection, and a fixed
  marker renderer. Validate projection independently before policy evaluation.
- **V2.3 Combined:** enable both interfaces with the same writer/retriever
  provenance and paired evaluation protocol.
- **V2.4 Ablation/robustness:** OFF, text, spatial, combined, and oracle;
  separately vary stale age, localization noise, and semantic corruption.
  Each perturbation must be configured and seeded.
- **V3/V4:** only after simulation V2 decisions are complete; UR3 simulation
  and real-robot experiments remain separate from LIBERO-derived scores.

## 7. Stop conditions and acceptance

Stop a run before inference if baseline hashes/settings do not match, a memory
fact has no provenance, timestamps imply future leakage, or the task/entity
mapping is ambiguous. Stop promotion if OFF and shadow inputs differ, the
memory path changes actions outside the task-text treatment, raw artifacts do
not reconcile, or oracle evidence is mixed into a non-oracle condition.

V2.1 is ready for a research decision when the paired pilot is reproducible,
all records and prompts are auditable, treatment isolation is demonstrated, and
the report includes both improvements and regressions with uncertainty. A
positive score is not a prerequisite for completing the experiment; it is a
prerequisite only for claiming benefit.

## 8. First implementation deliverable

The first code slice is P1–P3: typed memory schemas, an episode-scoped store,
deterministic update/retrieval, and a text renderer with a strict no-record
pass-through. It will not alter rollout behavior until shadow mode is designed
and its event evidence sources are explicit.
