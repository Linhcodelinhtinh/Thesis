"""Exercise paired accounting with synthetic bookkeeping records only."""
import json

import pytest

from scripts.aggregate_mem_pilot import aggregate, DEFAULT_PROTOCOL, source_digest, sha256_file


def batch(root, roster, statuses, condition):
    root.mkdir()
    adapter = {"adapter_files": {"adapter_model.safetensors": condition},
        "memory_provenance": [{"writer": "rgb_relations_v1"}],
        "run": {"arguments": {"condition": "memory" if condition == "F1" else "no_memory"},
                "data_manifest_sha256": "dataset", "assets_lock_sha256": "assets",
                "effective_batch_size": 32, "trainable_modules": {}, "mixed_replay": {}}}
    adapter_path = root / f"adapter_manifest_{condition}.json"
    adapter_path.write_text(json.dumps(adapter))
    for row, outcome in zip(roster, statuses):
        episode_id = f"{row['task']}_{condition}_val_state{row['state_idx']}"
        folder = root / "episodes" / episode_id
        folder.mkdir(parents=True)
        common = dict(row, condition=condition, policy_seed=42, episode_id=episode_id)
        status = dict(common, status="failed" if outcome is None else "complete")
        if outcome is not None:
            summary = folder / "summary.json"
            summary.write_text(json.dumps(dict(common, result=dict(condition=condition, success=outcome,
                                                                    avg_latency_ms=24, total_wall_time_s=10))))
            status["files_sha256"] = {"summary.json": sha256_file(summary)}
        (folder / "status.json").write_text(json.dumps(status))
    (root / "run_manifest.json").write_text(json.dumps(dict(
        classification="CUSTOM_V2_LIBERO_MEM", comparability="NON-COMPARABLE_OFFICIAL_PAPER",
        status="failed" if None in statuses else "complete",
        arguments=dict(policy_seed=42, split="val", runtime_check=False, preflight_only=False,
                       conditions=[condition], save_video=False),
        protocol_sha256=source_digest(DEFAULT_PROTOCOL), split_lock_sha256="lock",
        treatments={condition: dict(weight_identity=condition, adapter_manifest_sha256=sha256_file(adapter_path))},
        base={}, source_sha256={}, local_assets_lock_sha256="assets",
        environments={"detector": {"python": "fixture"}} if condition == "F1" else {},
        cuda={}, loaded_policies={condition: {"source_sha256": "fixture"}},
        roster=roster, expected_episodes=len(roster),
        completed_episodes=sum(value is not None for value in statuses))))
    return root


def test_missing_outcome_is_bounded_and_retry_completes_the_same_roster(tmp_path):
    rows = [dict(task=task, initial_state_sha256="shared", state_idx=0) for task in ("T1", "T3")]
    f0 = batch(tmp_path / "f0", rows, [False, False], "F0")
    f1 = batch(tmp_path / "f1", rows, [True, None], "F1")
    lock = {"lock_sha256": "lock"}
    report = aggregate([f0, f1], lock, rows, [42], "val")
    assert report["macro_delta"] is None
    assert report["missing_outcome_bounds"] == [.5, 1]
    assert report["per_task"]["T3"]["timing"]["F1"]["median_rollout_wall_time_s"] is None
    assert report["state_groups"] == 1
    retry = batch(tmp_path / "retry", rows[1:], [False], "F1")
    report = aggregate([f0, f1, retry], lock, rows, [42], "val")
    assert report["complete_paired_roster"]
    assert report["macro_delta"] == .5
    assert report["per_task"]["T3"]["timing"]["F1"]["median_rollout_wall_time_s"] == 10
    with pytest.raises(ValueError, match="Duplicate completed episode"):
        aggregate([f0, f1, retry, retry], lock, rows, [42], "val")
