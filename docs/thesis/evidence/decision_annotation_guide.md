# Decision-window annotation protocol (development)

Version: `decision-window-annotation-v1`  
Parent protocol: `libero_mem_decision_dev_v1`  
Scope: T1/T3 seen split only; this guide does not label evaluation outcomes for the policy.

## Annotation unit

One row describes one policy query or one candidate chunk at a predeclared decision window. Preserve the initial-state group across T1/T3 and policy seeds. Use the protocol's 12 development groups for annotation-rule development and its 8 calibration groups only for threshold calibration. Do not use the unseen roster to tune labels, thresholds, or writer behavior.

Every row records:

- `task`, `initial_state_sha256`, `policy_seed`, `decision_arm`, `episode_id`;
- `query_id`, `chunk_sha256`, `observation_step`, `annotation_source_sha256`;
- `intent`: `start_cycle`, `finish`, `approach`, `grasp`, `release`, `other`, or `unknown`;
- `target`: visible object/region identity or `null`;
- `commit_step`: earliest observed action at which the proposal commits to the target/effect, or `null`;
- `label_status`: `clear`, `ambiguous`, or `unavailable`;
- `executed`: whether the labeled proposal actually ran;
- `observed_effect` and an effect-source hash when execution is visible.

## Evidence and independence

Annotators use the authorized RGB views, the task instruction, and the observation prefix available at that query. They must not use evaluator predicates, hidden object IDs, subgoal counters, future frames, reward, or later actions to label the decision that was available at the query. Future video may be opened separately to label the observed effect of a proposal that actually executed; keep that label separate from the intent label and preserve the source hash.

Writer predictions and supervisor decisions stay hidden while an annotator assigns the intent, target, commit point, and ambiguity. A second annotator resolves disagreements without seeing the predicted label. If the target or effect cannot be established from the permitted view, use `unknown` or `unavailable`; do not infer it from the evaluator.

## Decision and trigger labels

For a trigger window, record the first observation at which new evidence could change the correct continue/stop decision or entity/container choice. Mark changes caused by progress, identity, relation, or execution separately. A writer event that occurs after that point is late even if the final trajectory later succeeds. A query with no relevant change is a negative trigger window; silence during an unobservable interval is not proof that no physical event occurred.

For T1/T3 counting, annotate `start_cycle` and `finish` only when the instruction and visible trajectory support them. Record the requested count from the instruction as the target label. Do not treat an observed return count as exact when the history is incomplete. The rule checker may return `supported`, `violated`, or `unknown`; annotators do not convert an unknown state into a binary label.

## Metrics and reporting

Report trigger precision/recall and lead time on clear windows, with ambiguous and unavailable windows reported separately. For proposal assessment, report false accept, false reject, unknown coverage, target accuracy, and the fraction of labels that correspond to executed chunks. For episode summaries, keep all decision windows from the same initial-state hash in one cluster across tasks and seeds. Never present repeated windows as independent episodes.

This guide defines a reproducible annotation schema. It does not claim that annotation has been completed or that thresholds have been calibrated.
