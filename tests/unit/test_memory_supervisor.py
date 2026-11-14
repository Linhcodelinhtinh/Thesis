"""Decision execution regression tests. All rollouts in this file are unit fixtures."""
from dataclasses import replace
import json

import numpy as np
import pytest

from scripts.evaluate_mem_adapter import parse_args
from scripts.aggregate_memory_decisions import compare
from src.evaluation.libero_mem_rollout import LiberoMemRolloutRunner
from src.evaluation.libero_mem_metrics import SubgoalEvaluationResult
from src.evaluation.mem_eval_io import EpisodeRecorder
from src.evaluation.memory_supervisor import (
    MemorySupervisor, execution_treatment, load_decision_protocol,
)
from src.evaluation.proposal_assessment import Verdict, combine_hypotheses, count_precondition
from src.memory.decision_context import DecisionFact, snapshot_facts
from src.memory.libero_mem_profiles import MemoryProfile
from src.memory.relation_memory import RelationMemoryWriter
from test_relation_memory import observation, renderer


INSTRUCTION = "pick up the bowl and place it back on the plate"
PROFILE = MemoryProfile("bowl", 1, "plate", 3, "on", True)


def new_treatment(mode, exposure="snapshot", **overrides):
    return replace(execution_treatment(load_decision_protocol(), mode, exposure), **overrides)


class RelationUnitEnv:
    task_name = "UNIT_FIXTURE_ONLY"

    def _frame(self, step):
        lift = 12 <= step < 17
        obs = observation(step, [("bowl", (.25, .28, .35, .4) if lift else (.25, .45, .35, .57)),
                                 ("plate", (.2, .50, .4, .62))], released=not lift)
        # The writer consumes cached typed evidence whereas the recorder consumes raw state.
        return dict(obs, robot0_eef_pos=np.zeros(3), robot0_eef_quat=np.zeros(4))

    def reset(self, initial_state_id=0):
        self.step_id = 0
        return self._frame(0)

    def step(self, action):
        self.step_id += 1
        done = self.step_id == 28
        result = SubgoalEvaluationResult(self.task_name, 1, int(done), float(done), done,
                                         False, int(done), 1, termination_reason="fixture_terminal"
                                         if done else "in_progress")
        return self._frame(self.step_id), 0., done, {"eval_result": result}


class RandomUnitPolicy:
    chunk_size, action_dim = 50, 7

    def reset(self):
        self.generator = np.random.default_rng(123)

    def predict_action_chunk(self, observation, prompt):
        return self.generator.normal(size=(50, 7)).astype(np.float32)


def run(mode, exposure="snapshot", **overrides):
    records = []
    runner = LiberoMemRolloutRunner(RelationUnitEnv(), RandomUnitPolicy(),
        writer_factory=lambda episode_id, **_: RelationMemoryWriter(episode_id, PROFILE, None, "libero_mem_refined"),
        renderer=renderer(), observer=lambda event, **row: records.append((event, row)),
        execution_treatment=new_treatment(mode, exposure, **overrides), max_steps=30)
    return runner.run_episode("unit", INSTRUCTION), records


def rows(records, event):
    return [row for kind, row in records if kind == event]


def test_shadow_preserves_queries_actions_and_rng():
    baseline, base_trace = run("text")
    shadow, shadow_trace = run("shadow", "projected")
    assert shadow.success == baseline.success
    assert shadow.prompt_history == baseline.prompt_history
    assert rows(shadow_trace, "decision")
    for before, after in zip(rows(base_trace, "query"), rows(shadow_trace, "query"), strict=True):
        assert before["step"] == after["step"]
        np.testing.assert_array_equal(before["chunk"], after["chunk"])
    for before, after in zip(rows(base_trace, "action"), rows(shadow_trace, "action"), strict=True):
        np.testing.assert_array_equal(before["action"], after["action"])
    for query in rows(shadow_trace, "query"):
        assert {fact["value"][0] for fact in query["decision_context"]["facts"]} == set(query["render"].facts_included)


def test_requery_discards_only_unused_prefix_and_preserves_causal_alignment():
    result, trace = run("event")
    queries = rows(trace, "query")
    assert result.decision_metrics["early_requeries"] > 0
    for row in rows(trace, "action"):
        query = queries[row["query_id"]]
        assert row["step"] == query["step"] + row["chunk_index"]
        np.testing.assert_array_equal(row["action"], query["chunk"][row["chunk_index"]])
    for row in rows(trace, "decision"):
        if row["decision"] == "requery":
            assert queries[row["query_id"] + 1]["step"] == row["step"]
            assert all(item["first_observation"] < row["step"] for item in row["changes"])


def test_budget_abstains_without_executing_a_substitute_action():
    result, trace = run("event", max_queries=1)
    assert not result.success
    assert result.total_steps == len(rows(trace, "action"))
    assert len(rows(trace, "query")) == 1
    assert rows(trace, "decision")[-1]["decision"] == "abstain"
    assert result.termination_reason == "decision_abstention"


def test_snapshot_withholds_future_count_and_keeps_uncertainty():
    _, trace = run("event")
    snapshots = [row["snapshot"] for row in rows(trace, "memory")]
    snap = next(snapshot for snapshot in snapshots if snapshot["relations"]["bowl_1"]["returns"] == 1)
    assert snapshot_facts(snap)[0].value[0] == 1
    corrupt = json.loads(json.dumps(snap))
    corrupt["relations"]["bowl_1"]["last_return"]["available_at"] += 1
    fact = snapshot_facts(corrupt)[0]
    assert fact.value[0] is None and fact.validity == "unknown"


def test_semantic_noop_does_not_trigger_and_checks_reject_duplicate_observations():
    _, trace = run("event")
    flat = [row["snapshot"] for row in rows(trace, "memory")]
    supervisor = MemorySupervisor(new_treatment("event"))
    supervisor.bind(0, INSTRUCTION, None, flat[7])
    assert not supervisor.evaluate(INSTRUCTION, None, flat[8])
    assert not supervisor.evaluate(INSTRUCTION, None, flat[9])
    with pytest.raises(ValueError, match="consecutive"):
        supervisor.evaluate(INSTRUCTION, None, flat[9])


def test_intermittent_new_fact_requires_fresh_consecutive_observations(monkeypatch):
    supervisor = MemorySupervisor(new_treatment("event", "projected"))
    fact = DecisionFact("bowl_1", ("Lift observed.",), "projected", ("event",), ("evidence",))
    observations = {0: (), 1: (fact,), 2: (), 3: (fact,), 4: (fact,)}
    monkeypatch.setattr(supervisor, "_facts",
                        lambda _instruction, _snapshot, raw, _renderer, _rendered: observations[raw["as_of_observation"]])
    supervisor.bind(0, INSTRUCTION, None, {"as_of_observation": 0})
    for step in (1, 2, 3):
        assert not supervisor.evaluate(INSTRUCTION, None, {"as_of_observation": step})
    assert supervisor.evaluate(INSTRUCTION, None, {"as_of_observation": 4})[0]["first_observation"] == 3


def test_episode_recorder_tracks_abstention(tmp_path):
    result, trace = run("event", max_queries=1)
    directory = tmp_path / "episode"
    recorder = EpisodeRecorder(directory, {}, save_video=False)
    for kind, row in trace:
        if kind == "observation":
            row = dict(row, observation={k: v for k, v in row["observation"].items() if k != "visual_evidence"})
        recorder(kind, **row)
    recorder.finish(result)
    status = json.loads((directory / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["termination_reason"] == "decision_abstention"
    assert "decisions.jsonl" in status["files_sha256"]


def test_decision_cli_rejects_unseen():
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "out", "--sim-python", "sim", "--detector-python", "det",
                    "--adapter-dir", "adapter", "--conditions", "F1", "--decision-mode", "event",
                    "--split", "unseen"])


def test_offline_count_precondition_keeps_unknown_and_obeys_fact_changes():
    _, trace = run("event")
    snapshots = [row["snapshot"] for row in rows(trace, "memory")]
    before, after = (snapshot_facts(snapshots[index])[0] for index in (7, 27))
    assert count_precondition(before, 1, "start_cycle") == Verdict.SUPPORTED
    assert count_precondition(after, 1, "start_cycle") == Verdict.VIOLATED
    incomplete = replace(before, validity="incomplete")
    assert count_precondition(incomplete, 1, "start_cycle") == Verdict.UNKNOWN
    assert combine_hypotheses([Verdict.SUPPORTED, Verdict.VIOLATED]) == Verdict.UNKNOWN
    assert combine_hypotheses([Verdict.SUPPORTED], residual_unknown=True) == Verdict.UNKNOWN


def test_paired_aggregation_counts_abstention_and_bounds_missing_outcomes():
    observed, missing = ("T1", "state1", 42), ("T1", "state2", 42)
    failed = dict(success=0, queries=125, policy_total_ms=125., abstained=True)
    succeeded = dict(success=1, queries=100, policy_total_ms=100., abstained=False)
    report = compare({observed: failed}, {observed: succeeded, missing: succeeded},
                     [observed, missing], {"seed": 42, "resamples": 10})
    assert report["pairs"] == 1
    assert report["observed_task_macro_success_delta"] == -1
    assert report["missing_outcome_bounds"] == [-1, -.5]
    assert report["complete_roster_state_bootstrap_95"] is None
