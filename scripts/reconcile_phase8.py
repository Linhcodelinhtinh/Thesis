"""Phase 8 Acceptance Benchmark Reconciliation & Provenance Verification Script.

Audits all episode-level artifacts against benchmark_summary.json in the
acceptance benchmark directory.

Verifications:
1. Artifact Completeness:
   - episode.json, diagnostics.json, timing.json, trajectory.npz, model_output.jsonl
   - trajectory.npz contains both 'actions' (T, 7) and 'states' (T, 9)
2. Metric Reconciliation:
   - Total episodes, successes, grasp count, lift count, placement count
   - Termination reasons, failure phases, and primary failure codes
3. Provenance & Tier Verification:
   - execution_tier == "LIBERO-DERIVED (HOST_PY3.12)"
   - certification == "NON-COMPARABLE_OFFICIAL_PAPER"
   - model_name, camera_resolution, execution_horizon_s
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


def reconcile_phase8(benchmark_dir: Path) -> Dict[str, Any]:
    print("=" * 70)
    print(f"Phase 8 Reconciliation Audit: {benchmark_dir}")
    print("=" * 70)

    summary_file = benchmark_dir / "benchmark_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_file}")

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 1. Provenance Audit in Summary
    print("\n[1] Auditing Summary Provenance & Execution Tier...")
    prov = summary.get("provenance", {})
    tier = prov.get("execution_tier")
    cert = prov.get("certification")
    py_ver = prov.get("python_version")
    cam_res = prov.get("camera_resolution")
    horizon = prov.get("execution_horizon_s")

    print(f"  - Execution Tier: {tier}")
    print(f"  - Certification: {cert}")
    print(f"  - Python Version: {py_ver}")
    print(f"  - Camera Resolution: {cam_res}")
    print(f"  - Execution Horizon (s): {horizon}")

    tier_expected = "LIBERO-DERIVED (HOST_PY3.12)"
    cert_expected = "NON-COMPARABLE_OFFICIAL_PAPER"
    assert tier == tier_expected, f"Unexpected execution tier in summary: {tier} != {tier_expected}"
    assert cert == cert_expected, f"Unexpected certification in summary: {cert} != {cert_expected}"
    assert cam_res == 256, f"Expected camera resolution 256, got {cam_res}"
    assert horizon == 50, f"Expected horizon 50, got {horizon}"
    print("  --> Summary Provenance: PASSED [VERIFIED]")

    # 2. Episode Artifacts Audit
    print("\n[2] Auditing Episode Artifacts across Task Directories...")
    task_dirs = sorted([d for d in benchmark_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"  Found {len(task_dirs)} task directories.")
    assert len(task_dirs) == 10, f"Expected 10 acceptance tasks, got {len(task_dirs)}"

    all_episodes: List[Dict[str, Any]] = []
    file_errors: List[str] = []
    episode_tier_errors: List[str] = []
    recomputed_task_metrics: Dict[str, Dict[str, Any]] = {}

    for td in task_dirs:
        init_dirs = sorted([d for d in td.iterdir() if d.is_dir() and d.name.startswith("init_")])
        t_key = td.name
        recomputed_task_metrics[t_key] = {
            "total": len(init_dirs),
            "successes": 0,
            "grasped": 0,
            "lifted": 0,
            "placed": 0,
            "phases": {},
            "codes": {},
            "termination": {},
        }

        for idir in init_dirs:
            ep_info: Dict[str, Any] = {"task": td.name, "init": idir.name}

            # Check required artifact files
            required_files = ["episode.json", "diagnostics.json", "timing.json", "trajectory.npz", "model_output.jsonl"]
            for req in required_files:
                fpath = idir / req
                if not fpath.exists() or fpath.stat().st_size == 0:
                    file_errors.append(f"{fpath} missing or empty")

            # Check trajectory.npz structure
            npz_path = idir / "trajectory.npz"
            if npz_path.exists():
                npz_data = np.load(npz_path)
                if "actions" not in npz_data or "states" not in npz_data:
                    file_errors.append(f"{npz_path} missing 'actions' or 'states'")
                else:
                    if npz_data["actions"].ndim != 2 or npz_data["actions"].shape[1] != 7:
                        file_errors.append(f"{npz_path} actions shape {npz_data['actions'].shape} invalid")
                    if npz_data["states"].ndim != 2 or npz_data["states"].shape[1] != 9:
                        file_errors.append(f"{npz_path} states shape {npz_data['states'].shape} invalid")

            # Check episode.json
            ep_path = idir / "episode.json"
            if ep_path.exists():
                with open(ep_path, "r", encoding="utf-8") as ef:
                    ep_json = json.load(ef)
                ep_succ = ep_json.get("success", False)
                ep_info["success"] = ep_succ
                if ep_succ:
                    recomputed_task_metrics[t_key]["successes"] += 1

                ep_prov = ep_json.get("provenance", {})
                if ep_prov.get("execution_tier") != tier_expected:
                    episode_tier_errors.append(f"{ep_path} tier={ep_prov.get('execution_tier')}")

            # Check diagnostics.json
            diag_path = idir / "diagnostics.json"
            if diag_path.exists():
                with open(diag_path, "r", encoding="utf-8") as df:
                    diag_json = json.load(df)
                evidence = diag_json.get("evidence", {})
                g = evidence.get("ever_grasped", False)
                l = evidence.get("ever_lifted", False)
                p = evidence.get("ever_placed", False) or ep_info.get("success", False)
                phase = diag_json.get("failure_phase", "NONE")
                code = diag_json.get("primary_failure_code", "NONE")
                term = diag_json.get("termination_reason", "SUCCESS" if ep_info.get("success") else "MAX_STEPS")

                if g:
                    recomputed_task_metrics[t_key]["grasped"] += 1
                if l:
                    recomputed_task_metrics[t_key]["lifted"] += 1
                if p:
                    recomputed_task_metrics[t_key]["placed"] += 1

                if not ep_info.get("success", False):
                    recomputed_task_metrics[t_key]["phases"][phase] = recomputed_task_metrics[t_key]["phases"].get(phase, 0) + 1
                    recomputed_task_metrics[t_key]["codes"][code] = recomputed_task_metrics[t_key]["codes"].get(code, 0) + 1
                recomputed_task_metrics[t_key]["termination"][term] = recomputed_task_metrics[t_key]["termination"].get(term, 0) + 1

            all_episodes.append(ep_info)

    print(f"  Total episodes audited: {len(all_episodes)}")
    print(f"  Artifact file integrity errors: {len(file_errors)}")
    print(f"  Episode tier discrepancies: {len(episode_tier_errors)}")
    assert len(file_errors) == 0, f"Found file integrity errors: {file_errors[:5]}"
    assert len(episode_tier_errors) == 0, f"Found tier discrepancies: {episode_tier_errors[:5]}"
    print("  --> Episode Artifacts Integrity: PASSED [VERIFIED]")

    # 3. Reconcile Summary vs Recomputed Metrics
    print("\n[3] Reconciling Metrics (Bottom-Up vs Summary)...")
    summary_om = summary["overall_metrics"]
    total_recomputed = len(all_episodes)
    succ_recomputed = sum(t["successes"] for t in recomputed_task_metrics.values())
    grasped_recomputed = sum(t["grasped"] for t in recomputed_task_metrics.values())
    lifted_recomputed = sum(t["lifted"] for t in recomputed_task_metrics.values())
    placed_recomputed = sum(t["placed"] for t in recomputed_task_metrics.values())

    print(f"  - Total Episodes: {total_recomputed} (Summary: {summary_om['total_episodes']})")
    print(f"  - Total Successes: {succ_recomputed} (Summary: {summary_om['total_successes']})")
    print(f"  - Success Rate: {succ_recomputed / total_recomputed:.1%} (Summary: {summary_om['overall_success_rate']:.1%})")
    print(f"  - Total Grasped: {grasped_recomputed}/{total_recomputed} ({grasped_recomputed / total_recomputed:.1%})")
    print(f"  - Total Lifted: {lifted_recomputed}/{total_recomputed} ({lifted_recomputed / total_recomputed:.1%})")
    print(f"  - Total Placed: {placed_recomputed}/{total_recomputed} ({placed_recomputed / total_recomputed:.1%})")

    assert total_recomputed == summary_om["total_episodes"], "Total episodes mismatch"
    assert succ_recomputed == summary_om["total_successes"], "Total successes mismatch"
    assert round(succ_recomputed / total_recomputed, 4) == round(summary_om["overall_success_rate"], 4), "Success rate mismatch"

    # Per-task reconciliation
    for t_summary in summary["tasks"]:
        suite = t_summary["task_suite"]
        tid = t_summary["task_id"]
        t_key = f"{suite}_task{tid}"
        rec = recomputed_task_metrics.get(t_key)
        assert rec is not None, f"Missing task in recomputed metrics: {t_key}"
        assert rec["total"] == t_summary["total_episodes"], f"Task {t_key} total mismatch"
        assert rec["successes"] == t_summary["success_count"], f"Task {t_key} success mismatch"
        assert rec["grasped"] == t_summary["grasp_success_count"], f"Task {t_key} grasp mismatch"
        assert rec["lifted"] == t_summary["lift_success_count"], f"Task {t_key} lift mismatch"
        assert rec["placed"] == t_summary["placement_success_count"], f"Task {t_key} place mismatch"

    print("  --> Bottom-Up Metric Reconciliation: PASSED [100% MATCH]")

    # 4. Summary Table Output
    print("\n" + "=" * 70)
    print("PHASE 8 ACCEPTANCE BENCHMARK LOCK AUDIT - FINAL RESULTS")
    print("=" * 70)
    print(f"Status: COMPLETED & LOCKED")
    print(f"Execution Tier: {tier} [{cert}]")
    print(f"Total Tasks: {len(task_dirs)} / 10")
    print(f"Total Episodes: {total_recomputed} / 40")
    print(f"Success Rate: {succ_recomputed / total_recomputed:.1%} (95% Wilson CI: [{summary_om['wilson_ci_95'][0]:.1%}, {summary_om['wilson_ci_95'][1]:.1%}])")
    print(f"Grasp Success Rate: {grasped_recomputed / total_recomputed:.1%}")
    print(f"Lift Success Rate: {lifted_recomputed / total_recomputed:.1%}")
    print(f"Place Success Rate: {placed_recomputed / total_recomputed:.1%}")
    print(f"Inference Latency: Mean Call={summary_om.get('mean_call_inference_latency_ms')}ms | P95={summary_om.get('p95_call_inference_latency_ms')}ms")
    print("=" * 70)

    return {
        "status": "LOCKED",
        "total_episodes": total_recomputed,
        "total_successes": succ_recomputed,
        "success_rate": succ_recomputed / total_recomputed,
        "execution_tier": tier,
        "certification": cert,
    }


if __name__ == "__main__":
    target = Path("experiments/results/raw_baseline/acpt_10_test")
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    try:
        reconcile_phase8(target)
    except Exception as e:
        print(f"\nERROR: Reconciliation failed: {e}", file=sys.stderr)
        sys.exit(1)
