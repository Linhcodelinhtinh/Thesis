"""Closed-loop wiring of the sparse writer. Every rollout here is a unit fixture, not a benchmark result."""
import numpy as np
import pytest

from src.evaluation.libero_mem_metrics import SubgoalEvaluationResult
from src.evaluation.libero_mem_rollout import LiberoMemRolloutRunner
from src.evaluation.memory_supervisor import ExecutionTreatment
from src.memory.relation_memory import RelationMemoryWriter
from src.memory.relation_sparse_writer import SparseRelationWriterV1, detector_calls_in_trace
from tests.unit.test_bowl_cycle_writer import PROFILE
from tests.unit.test_relation_memory import renderer
from tests.unit.test_relation_sparse_writer import PICK_AND_PLACE, ReplayDetector, frame_obs

INSTRUCTION = "pick up the bowl and place it back on the plate"
LAST = len(PICK_AND_PLACE) - 1


class SparseUnitEnv:
    task_name = "UNIT_FIXTURE_ONLY"

    def _frame(self, step):
        boxes, aperture = PICK_AND_PLACE[min(step, LAST)]
        obs = frame_obs(step, boxes, aperture)
        obs.pop("visual_evidence")  # as in a real run: only the detector produces evidence
        return dict(obs, robot0_eef_pos=np.zeros(3), robot0_eef_quat=np.zeros(4))

    def reset(self, initial_state_id=0):
        self.step_id = 0
        return self._frame(0)

    def step(self, action):
        self.step_id += 1
        done = self.step_id == LAST
        result = SubgoalEvaluationResult(self.task_name, 1, int(done), float(done), done, False, int(done), 1,
                                         termination_reason="fixture_terminal" if done else "in_progress")
        return self._frame(self.step_id), 0., done, {"eval_result": result}


class QuietPolicy:
    chunk_size, action_dim = 50, 7

    def reset(self):
        pass

    def predict_action_chunk(self, observation, prompt):
        return np.zeros((50, 7), dtype=np.float32)


def treatment(mode, **overrides):
    values = dict(mode=mode, exposure="snapshot", max_queries=200, periodic_prefix=8, minimum_prefix=2,
                  persistence=2)
    return ExecutionTreatment(**{**values, **overrides})


def run(writer_type, mode="text"):
    records = []
    detector = ReplayDetector({step: frame_obs(step, *PICK_AND_PLACE[min(step, LAST)])["visual_evidence"]
                               for step in range(LAST + 1)})
    runner = LiberoMemRolloutRunner(
        SparseUnitEnv(), QuietPolicy(), execution_horizon=8, max_steps=LAST,
        writer_factory=lambda episode_id, **_: writer_type(episode_id, PROFILE, detector, "libero_mem_refined"),
        renderer=renderer(), observer=lambda event, **row: records.append((event, row)),
        execution_treatment=treatment(mode))
    result = runner.run_episode("unit", INSTRUCTION)
    return result, records, detector


def test_trace_gives_the_exact_detector_calls_of_a_sparse_run():
    dense, dense_records, dense_detector = run(RelationMemoryWriter)
    sparse, sparse_records, sparse_detector = run(SparseRelationWriterV1)
    memory = [row for kind, row in sparse_records if kind == "memory"]
    calls = detector_calls_in_trace(memory)
    assert calls == len(sparse_detector.calls) and 0 < calls < .6 * (sparse.total_steps + 1)
    assert len(dense_detector.calls) == dense.total_steps + 1
    # The shared rollout counts one call per frame; it is deliberately left untouched, so it over-counts here.
    assert sparse.decision_metrics["detector_calls"] == sparse.total_steps + 1
    with pytest.raises(ValueError, match="not written by a sparse writer"):
        detector_calls_in_trace([row for kind, row in dense_records if kind == "memory"])


def test_sparse_prompts_carry_the_same_facts_as_the_dense_prompts():
    dense, _, _ = run(RelationMemoryWriter)
    sparse, _, _ = run(SparseRelationWriterV1)
    assert len(dense.prompt_history) == len(sparse.prompt_history)
    strip = lambda prompt: prompt.replace(" History incomplete.", "")
    assert [strip(p) for p in dense.prompt_history] == [strip(p) for p in sparse.prompt_history]
    assert sparse.prompt_history[-1].endswith("Memory: bowl: 1 observed returns.")


@pytest.mark.parametrize("mode", ["shadow", "event"])
def test_supervisor_accepts_the_sparse_snapshot_stream(mode):
    """Skipped frames advance as_of_observation, so the supervisor still sees consecutive observations."""
    baseline, _, _ = run(SparseRelationWriterV1, "text")
    result, records, _ = run(SparseRelationWriterV1, mode)
    assert [kind for kind, _ in records if kind == "decision"]
    if mode == "shadow":
        assert result.prompt_history == baseline.prompt_history
    observed = [row["snapshot"]["as_of_observation"] for kind, row in records if kind == "memory"]
    assert observed == list(range(len(observed)))
