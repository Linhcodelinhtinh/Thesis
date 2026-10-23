#!/usr/bin/env python
"""Conformance and conformance gate audit for LIBERO-Mem.

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 12:
- Audits benchmark configuration and task specifications.
- Validates evaluator conformance (SCR, sequence progress, overshoot).
- Validates repeated predicate intervals for multi-cycle tasks (T3).
- Validates isolation and clean reset lifecycle.
- Outputs an auditable JSON conformance report.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.libero_mem_metrics import LiberoMemEvaluator
from src.memory.event_writer import LiberoMemEventWriter
from src.simulator.libero_mem_env import LiberoMemEnv


def audit_config() -> Dict[str, Any]:
    config_path = ROOT / "configs/benchmarks/libero_mem_pilot.yaml"
    if not config_path.is_file():
        return {"status": "FAIL", "reason": "Config file missing"}

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    if len(tasks) < 2:
        return {"status": "FAIL", "reason": "Expected at least 2 pilot tasks"}

    t1 = next((t for t in tasks if t.get("alias") == "T1"), None)
    t3 = next((t for t in tasks if t.get("alias") == "T3"), None)

    if not t1 or t1.get("target_cycles") != 1:
        return {"status": "FAIL", "reason": "T1 target_cycles != 1"}
    if not t3 or t3.get("target_cycles") != 3:
        return {"status": "FAIL", "reason": "T3 target_cycles != 3"}

    return {
        "status": "PASS",
        "benchmark_name": data.get("benchmark_name"),
        "provenance_tag": data.get("provenance_tag"),
        "num_tasks": len(tasks),
        "t1_task": t1.get("name"),
        "t3_task": t3.get("name"),
    }


def audit_evaluator_conformance() -> Dict[str, Any]:
    evaluator = LiberoMemEvaluator(
        task_name="KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times",
        target_cycles=3,
        target_object="bowl",
        support_object="plate",
        dwell_steps=2,
        live_time_threshold=0,
        nonlive_time_threshold=0,
        overshoot_threshold=0,
    )

    # Initially at rest
    res0 = evaluator.evaluate_step(step=0, is_on_support=True, is_lifted=False)
    assert res0.scr == 0.0
    assert not res0.full_success
    assert res0.confirmed_cycles == 0

    # Cycle 1: Lift (2 dwell steps)
    evaluator.evaluate_step(step=10, is_on_support=False, is_lifted=True)
    res1 = evaluator.evaluate_step(step=11, is_on_support=False, is_lifted=True)
    assert res1.completed_subgoals == 0  # Lifted is departure phase
    assert res1.scr == 0.0

    # Cycle 1: Return (2 dwell steps)
    evaluator.evaluate_step(step=20, is_on_support=True, is_lifted=False)
    res2 = evaluator.evaluate_step(step=21, is_on_support=True, is_lifted=False)
    assert res2.completed_subgoals == 1
    assert res2.confirmed_cycles == 1
    assert abs(res2.scr - (1.0 / 3.0)) < 1e-4

    # Cycle 2: Lift & Return (2 dwell steps each)
    evaluator.evaluate_step(step=30, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(step=31, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(step=40, is_on_support=True, is_lifted=False)
    res3 = evaluator.evaluate_step(step=41, is_on_support=True, is_lifted=False)
    assert res3.confirmed_cycles == 2
    assert res3.completed_subgoals == 2
    assert abs(res3.scr - (2.0 / 3.0)) < 1e-4

    # Cycle 3: Lift & Return -> Full success!
    evaluator.evaluate_step(step=50, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(step=51, is_on_support=False, is_lifted=True)
    evaluator.evaluate_step(step=60, is_on_support=True, is_lifted=False)
    res4 = evaluator.evaluate_step(step=61, is_on_support=True, is_lifted=False)
    assert res4.confirmed_cycles == 3
    assert res4.completed_subgoals == 3
    assert res4.scr == 1.0
    assert res4.full_success
    assert not res4.overshot

    # Overshoot test: 4th lift with dwell!
    evaluator.evaluate_step(step=70, is_on_support=False, is_lifted=True)
    res5 = evaluator.evaluate_step(step=71, is_on_support=False, is_lifted=True)
    assert res5.overshot
    assert not res5.full_success  # Overshot voids full success

    # Clean reset test
    evaluator.reset()
    res_reset = evaluator.evaluate_step(step=0, is_on_support=True, is_lifted=False)
    assert res_reset.scr == 0.0
    assert not res_reset.overshot
    assert res_reset.confirmed_cycles == 0

    return {
        "status": "PASS",
        "scr_calculation": "verified",
        "ordered_sequence": "verified",
        "repeated_predicate_intervals": "verified",
        "dwell_validation": "verified",
        "overshoot_detection": "verified",
        "reset_isolation": "verified",
    }


def audit_writer_fsm() -> Dict[str, Any]:
    writer = LiberoMemEventWriter(
        episode_id="audit_ep_1",
        target_object_id="bowl_A",
        dwell_steps=2,
    )

    # Initial resting state: no unobserved relation
    snap0 = writer.get_snapshot()
    assert snap0.cycles["bowl_A_plate_A"].confirmed == 0
    assert not snap0.cycles["bowl_A_plate_A"].in_progress

    # Step 1-2: Displaced observation with visual detection evidence
    obs_displaced = {
        "robot0_eef_pos": np.array([0.0, 0.0, 1.2], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.01, 0.01], dtype=np.float32),
        "detected_objects": {"bowl_A": np.array([0.0, 0.0, 1.2], dtype=np.float32)},
    }
    writer.step(1, obs_displaced, None)
    writer.step(2, obs_displaced, None)

    # Departure confirmed after 2 dwell steps
    snap1 = writer.get_snapshot()
    assert snap1.cycles["bowl_A_plate_A"].in_progress

    # Step 3-4: Return steps with visual detection evidence
    obs_return = {
        "robot0_eef_pos": np.array([-0.05, 0.20, 0.85], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.05, 0.05], dtype=np.float32),
        "detected_objects": {"bowl_A": np.array([-0.05, 0.20, 0.85], dtype=np.float32)},
    }
    writer.step(3, obs_return, None)
    writer.step(4, obs_return, None)

    # Return confirmed after 2 dwell steps!
    snap2 = writer.get_snapshot()
    assert snap2.cycles["bowl_A_plate_A"].confirmed == 1
    assert not snap2.cycles["bowl_A_plate_A"].in_progress

    return {
        "status": "PASS",
        "fsm_departure_hysteresis": "verified",
        "fsm_return_hysteresis": "verified",
        "cycle_counter_accuracy": "verified",
    }


def audit_upstream_environment() -> Dict[str, Any]:
    """Check whether upstream LIBERO-Mem repository and assets are linked and validated."""
    upstream_dir = Path(__file__).resolve().parent.parent / "external" / "libero-mem"
    if not upstream_dir.exists():
        return {
            "status": "FAIL",
            "reason": "MISSING_UPSTREAM_CHECKOUT",
            "details": "external/libero-mem not found. Upstream BDDL/MuJoCo integration pending.",
        }

    # Verify canonical dataset assets
    data_dir = Path(__file__).resolve().parent.parent / "data" / "libero_mem"
    meta_path = data_dir / "metainfo.json"
    t1_path = data_dir / "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate_demo.hdf5"
    t3_path = data_dir / "KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times_demo.hdf5"
    if not (meta_path.exists() and t1_path.exists() and t3_path.exists()):
        return {
            "status": "FAIL",
            "reason": "MISSING_PILOT_DATASETS",
            "details": f"Missing required dataset files in {data_dir}.",
        }

    # Verify contract probes report
    probe_results_path = Path(__file__).resolve().parent.parent / "experiments" / "audits" / "libero_mem_completeness_20261023" / "contract_results.json"
    probes_passed = False
    failed_count = 0
    if probe_results_path.exists():
        try:
            results = json.loads(probe_results_path.read_text(encoding="utf-8"))
            failed_count = results.get("failed", 0)
            probes_passed = (failed_count == 0)
        except Exception:
            pass

    return {
        "status": "PASS" if probes_passed else "FAIL",
        "path": str(upstream_dir),
        "dataset_files": "verified",
        "contract_probes_passed": probes_passed,
        "failed_contract_probes": failed_count,
    }


def main():
    report = {
        "benchmark_config": audit_config(),
        "evaluator_conformance": audit_evaluator_conformance(),
        "writer_fsm_conformance": audit_writer_fsm(),
        "upstream_environment_conformance": audit_upstream_environment(),
    }

    unit_pass = (
        report["benchmark_config"].get("status") == "PASS"
        and report["evaluator_conformance"].get("status") == "PASS"
        and report["writer_fsm_conformance"].get("status") == "PASS"
    )
    report["unit_contract_conformance"] = "PASS" if unit_pass else "FAIL"

    upstream_ready = report["upstream_environment_conformance"].get("status") == "PASS"
    report["overall_runtime_readiness"] = "READY" if (unit_pass and upstream_ready) else "NOT_READY"

    print(json.dumps(report, indent=2))
    if not unit_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
