"""Benchmark Telemetry Aggregator and Reporting Engine (Phase 8).

Aggregates per-episode closed-loop evaluation outcomes and diagnostics across tasks
and initial states into standardized machine-readable and human-readable reports:
- benchmark_summary.json: Full structured statistical breakdown.
- benchmark_summary.csv: Per-task tabulated metrics.
- failure_distribution.csv: Breakdown of failure phases and primary taxonomy codes.
- BENCHMARK_REPORT.md: GitHub-flavored markdown report with ADR-0010 provenance tagging.
"""

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate the two-sided Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of successful trials.
        total: Total number of trials.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        (lower_bound, upper_bound) bounded in [0.0, 1.0].
    """
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    # z for 95% is approx 1.95996
    z = 1.95996 if math.isclose(confidence, 0.95, rel_tol=1e-2) else 1.96
    denominator = 1.0 + (z**2) / total
    center_adjusted = (p + (z**2) / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * (total**2)))
    lower = max(0.0, center_adjusted - margin)
    upper = min(1.0, center_adjusted + margin)
    return round(lower, 4), round(upper, 4)


@dataclass
class TaskSummary:
    """Aggregated statistical outcome for a single evaluated benchmark task."""
    task_suite: str
    task_id: int
    task_name: str
    instruction: str
    total_episodes: int
    success_count: int
    success_rate: float
    wilson_ci_95: Tuple[float, float]
    grasp_success_count: int
    grasp_success_rate: float
    lift_success_count: int
    lift_success_rate: float
    placement_success_count: int
    placement_success_rate: float
    mean_steps_to_success: Optional[float]
    mean_episode_steps: float
    mean_inference_latency_ms: float
    p95_inference_latency_ms: float
    mean_simulation_latency_ms: float
    mean_call_inference_latency_ms: Optional[float] = None
    p95_call_inference_latency_ms: Optional[float] = None
    total_inference_calls: int = 0
    failure_phase_counts: Dict[str, int] = field(default_factory=dict)
    primary_failure_code_counts: Dict[str, int] = field(default_factory=dict)
    termination_reason_counts: Dict[str, int] = field(default_factory=dict)
    subtask_milestones: Dict[str, Any] = field(default_factory=dict)
    mean_sequential_survival_steps: Optional[float] = None
    sequence_transition_failures: int = 0
    atomic_manipulation_failures: int = 0


class BenchmarkAggregator:
    """Aggregator collecting episode results and generating evaluation reports."""

    def __init__(self, benchmark_name: str = "Acceptance Benchmark Baseline", provenance: Optional[Dict[str, Any]] = None) -> None:
        self.benchmark_name = benchmark_name
        self.provenance = provenance or {}
        self.episodes: List[Dict[str, Any]] = []

    def add_episode_result(
        self,
        task_suite: str,
        task_id: int,
        task_name: str,
        instruction: str,
        initial_state_id: int,
        episode_result: Any,  # EpisodeResult instance or dict
    ) -> None:
        """Add an evaluated episode result to the aggregator collection."""
        res_dict = episode_result.to_dict() if hasattr(episode_result, "to_dict") else dict(episode_result)
        res_dict["task_suite"] = task_suite
        res_dict["task_id"] = task_id
        res_dict["task_name"] = task_name
        res_dict["instruction"] = instruction
        res_dict["initial_state_id"] = initial_state_id
        self.episodes.append(res_dict)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute aggregated task-level and suite-level summaries."""
        # Group episodes by (task_suite, task_id)
        grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for ep in self.episodes:
            key = (ep["task_suite"], ep["task_id"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(ep)

        task_summaries: List[Dict[str, Any]] = []
        all_successes = 0
        all_total = 0
        all_infer_latencies: List[float] = []
        all_call_infer_latencies: List[float] = []
        all_sim_latencies: List[float] = []
        global_failure_phases: Dict[str, int] = {}
        global_primary_codes: Dict[str, int] = {}
        global_termination_reasons: Dict[str, int] = {}

        for (suite, tid), eps in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            n_total = len(eps)
            n_success = sum(1 for e in eps if e.get("success", False))
            success_steps = [e["num_steps"] for e in eps if e.get("success", False)]
            all_steps = [e["num_steps"] for e in eps]

            # Track grasp, lift, and placement separately
            n_grasped = 0
            n_lifted = 0
            n_placed = 0
            phase_counts: Dict[str, int] = {}
            code_counts: Dict[str, int] = {}
            term_counts: Dict[str, int] = {}
            infer_lats: List[float] = []
            task_call_lats: List[float] = []
            sim_lats: List[float] = []

            # Multi-stage subtask milestone telemetry (Phase 11)
            milestone_completions: Dict[str, int] = {}
            milestone_total_definitions: Dict[str, str] = {}
            survival_step_records: List[int] = []
            transition_failures = 0
            atomic_failures = 0

            for e in eps:
                # Diagnostics inspection
                diag = e.get("diagnostics") or {}
                evidence = diag.get("evidence") or {}
                if evidence.get("ever_grasped", False):
                    n_grasped += 1
                if evidence.get("ever_lifted", False):
                    n_lifted += 1
                if evidence.get("ever_placed", False) or e.get("success", False):
                    n_placed += 1

                # Subtask milestone telemetry
                subtasks = diag.get("subtask_milestones") or []
                s_metrics = diag.get("summary_metrics") or {}
                if "sequential_survival_steps" in s_metrics and s_metrics["sequential_survival_steps"] is not None:
                    survival_step_records.append(int(s_metrics["sequential_survival_steps"]))

                if subtasks:
                    comp_count = sum(1 for m in subtasks if m.get("achieved", False))
                    if comp_count == len(subtasks) or e.get("success", False):
                        pass
                    elif comp_count > 0:
                        transition_failures += 1
                    else:
                        atomic_failures += 1

                    for m in subtasks:
                        m_name = m.get("name", "unknown")
                        milestone_total_definitions[m_name] = m.get("description", m_name)
                        if m.get("achieved", False):
                            milestone_completions[m_name] = milestone_completions.get(m_name, 0) + 1

                # Failure taxonomy counts
                term_reason = diag.get("termination_reason") or ("SUCCESS" if e.get("success") else "MAX_STEPS")
                term_counts[term_reason] = term_counts.get(term_reason, 0) + 1
                global_termination_reasons[term_reason] = global_termination_reasons.get(term_reason, 0) + 1

                if not e.get("success", False):
                    phase = diag.get("failure_phase", "UNKNOWN")
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1
                    global_failure_phases[phase] = global_failure_phases.get(phase, 0) + 1

                    code = diag.get("primary_failure_code", "UNATTRIBUTED")
                    code_counts[code] = code_counts.get(code, 0) + 1
                    global_primary_codes[code] = global_primary_codes.get(code, 0) + 1

                # Latencies
                ep_calls = e.get("inference_latencies_ms") or []
                if ep_calls:
                    task_call_lats.extend(ep_calls)
                    all_call_infer_latencies.extend(ep_calls)

                if "mean_inference_ms" in e and e["mean_inference_ms"] > 0:
                    infer_lats.append(e["mean_inference_ms"])
                    all_infer_latencies.append(e["mean_inference_ms"])
                if "mean_simulation_ms" in e and e["mean_simulation_ms"] > 0:
                    sim_lats.append(e["mean_simulation_ms"])
                    all_sim_latencies.append(e["mean_simulation_ms"])

            sr = n_success / n_total if n_total > 0 else 0.0
            ci_low, ci_high = wilson_score_interval(n_success, n_total)
            grasp_rate = n_grasped / n_total if n_total > 0 else 0.0
            lift_rate = n_lifted / n_total if n_total > 0 else 0.0
            place_rate = n_placed / n_total if n_total > 0 else 0.0

            mean_infer = float(np.mean(infer_lats)) if infer_lats else 0.0
            p95_infer = float(np.percentile(infer_lats, 95)) if infer_lats else 0.0
            mean_call_infer = float(np.mean(task_call_lats)) if task_call_lats else mean_infer
            p95_call_infer = float(np.percentile(task_call_lats, 95)) if task_call_lats else p95_infer
            mean_sim = float(np.mean(sim_lats)) if sim_lats else 0.0

            t_summary = {
                "task_suite": suite,
                "task_id": tid,
                "task_name": eps[0].get("task_name", f"{suite}_{tid}"),
                "instruction": eps[0].get("instruction", ""),
                "total_episodes": n_total,
                "success_count": n_success,
                "success_rate": round(sr, 4),
                "wilson_ci_95": [ci_low, ci_high],
                "grasp_success_count": n_grasped,
                "grasp_success_rate": round(grasp_rate, 4),
                "lift_success_count": n_lifted,
                "lift_success_rate": round(lift_rate, 4),
                "placement_success_count": n_placed,
                "placement_success_rate": round(place_rate, 4),
                "mean_steps_to_success": round(float(np.mean(success_steps)), 2) if success_steps else None,
                "mean_episode_steps": round(float(np.mean(all_steps)), 2) if all_steps else 0.0,
                "mean_inference_latency_ms": round(mean_infer, 2),
                "p95_inference_latency_ms": round(p95_infer, 2),
                "mean_call_inference_latency_ms": round(mean_call_infer, 2),
                "p95_call_inference_latency_ms": round(p95_call_infer, 2),
                "total_inference_calls": len(task_call_lats),
                "mean_simulation_latency_ms": round(mean_sim, 2),
                "failure_phase_counts": phase_counts,
                "primary_failure_code_counts": code_counts,
                "termination_reason_counts": term_counts,
                "subtask_milestones": {
                    m_name: {
                        "description": m_desc,
                        "achieved_count": milestone_completions.get(m_name, 0),
                        "achieved_rate": round(milestone_completions.get(m_name, 0) / n_total, 4) if n_total > 0 else 0.0,
                    }
                    for m_name, m_desc in milestone_total_definitions.items()
                },
                "mean_sequential_survival_steps": round(float(np.mean(survival_step_records)), 2) if survival_step_records else None,
                "sequence_transition_failures": transition_failures,
                "atomic_manipulation_failures": atomic_failures,
            }
            task_summaries.append(t_summary)
            all_successes += n_success
            all_total += n_total

        global_sr = all_successes / all_total if all_total > 0 else 0.0
        global_ci_low, global_ci_high = wilson_score_interval(all_successes, all_total)

        mean_call_lat = float(np.mean(all_call_infer_latencies)) if all_call_infer_latencies else (float(np.mean(all_infer_latencies)) if all_infer_latencies else 0.0)
        p95_call_lat = float(np.percentile(all_call_infer_latencies, 95)) if all_call_infer_latencies else (float(np.percentile(all_infer_latencies, 95)) if all_infer_latencies else 0.0)

        return {
            "benchmark_name": self.benchmark_name,
            "provenance": self.provenance,
            "overall_metrics": {
                "total_tasks": len(task_summaries),
                "total_episodes": all_total,
                "total_successes": all_successes,
                "overall_success_rate": round(global_sr, 4),
                "wilson_ci_95": [global_ci_low, global_ci_high],
                "mean_inference_latency_ms": round(float(np.mean(all_infer_latencies)), 2) if all_infer_latencies else 0.0,
                "mean_episode_inference_latency_ms": round(float(np.mean(all_infer_latencies)), 2) if all_infer_latencies else 0.0,
                "mean_call_inference_latency_ms": round(mean_call_lat, 2),
                "p95_call_inference_latency_ms": round(p95_call_lat, 2),
                "total_inference_calls": len(all_call_infer_latencies),
                "mean_simulation_latency_ms": round(float(np.mean(all_sim_latencies)), 2) if all_sim_latencies else 0.0,
                "global_failure_phases": global_failure_phases,
                "global_primary_failure_codes": global_primary_codes,
                "global_termination_reasons": global_termination_reasons,
            },
            "tasks": task_summaries,
        }

    def save_reports(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Generate and save benchmark_summary.json, summary.csv, failure_distribution.csv, and BENCHMARK_REPORT.md."""
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        summary = self.compute_summary()

        # 1. Save benchmark_summary.json
        json_path = target_dir / "benchmark_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 2. Save benchmark_summary.csv
        csv_path = target_dir / "benchmark_summary.csv"
        csv_headers = [
            "task_suite", "task_id", "task_name", "total_episodes", "success_count",
            "success_rate", "ci_95_low", "ci_95_high", "grasp_success_rate",
            "lift_success_rate", "placement_success_rate", "mean_steps_to_success",
            "mean_episode_steps", "mean_inference_ms", "p95_call_inference_ms", "mean_simulation_ms"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            for t in summary["tasks"]:
                writer.writerow([
                    t["task_suite"],
                    t["task_id"],
                    t["task_name"],
                    t["total_episodes"],
                    t["success_count"],
                    t["success_rate"],
                    t["wilson_ci_95"][0],
                    t["wilson_ci_95"][1],
                    t["grasp_success_rate"],
                    t["lift_success_rate"],
                    t["placement_success_rate"],
                    t["mean_steps_to_success"] if t["mean_steps_to_success"] is not None else "N/A",
                    t["mean_episode_steps"],
                    t["mean_inference_latency_ms"],
                    t.get("p95_call_inference_latency_ms", "N/A"),
                    t["mean_simulation_latency_ms"],
                ])

        # 3. Save failure_distribution.csv
        fail_csv_path = target_dir / "failure_distribution.csv"
        fail_headers = ["task_suite", "task_id", "category_type", "tag_or_code", "count"]
        with open(fail_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fail_headers)
            for t in summary["tasks"]:
                for ph, count in t.get("failure_phase_counts", {}).items():
                    writer.writerow([t["task_suite"], t["task_id"], "failure_phase", ph, count])
                for cd, count in t.get("primary_failure_code_counts", {}).items():
                    writer.writerow([t["task_suite"], t["task_id"], "primary_code", cd, count])
                for tr, count in t.get("termination_reason_counts", {}).items():
                    writer.writerow([t["task_suite"], t["task_id"], "termination_reason", tr, count])

        # 4. Save BENCHMARK_REPORT.md
        md_path = target_dir / "BENCHMARK_REPORT.md"
        md_content = self._generate_markdown_report(summary)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return {
            "summary_json": json_path,
            "summary_csv": csv_path,
            "failure_csv": fail_csv_path,
            "report_md": md_path,
        }

    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Format a rich markdown evaluation report with GitHub styling and ADR-0010 provenance."""
        overall = summary["overall_metrics"]
        prov = summary.get("provenance", {})

        md = []
        md.append(f"# {summary['benchmark_name']}")
        md.append("")
        md.append("> **Certification & Provenance (ADR-0010)**:  ")
        md.append(f"> - Execution Tier: `{prov.get('execution_tier', 'LIBERO-DERIVED')}`  ")
        md.append(f"> - Certification: `{prov.get('certification', 'NON-COMPARABLE_OFFICIAL_PAPER')}`  ")
        md.append(f"> - Note: {prov.get('note', 'Acceptance pilot baseline evaluated across locked initial states 0..N-1. This is a local verification baseline and does NOT constitute official full-suite LIBERO scores.')}  ")
        md.append("")
        md.append("## 1. Overall Performance Summary")
        md.append("")
        md.append(f"- **Total Tasks Evaluated**: {overall['total_tasks']}")
        md.append(f"- **Total Episodes**: {overall['total_episodes']}")
        md.append(f"- **Overall Success Rate**: **{overall['overall_success_rate']*100:.1f}%** ({overall['total_successes']}/{overall['total_episodes']}) [95% CI: {overall['wilson_ci_95'][0]*100:.1f}% – {overall['wilson_ci_95'][1]*100:.1f}%]")
        
        call_count = overall.get('total_inference_calls', 0)
        p95_call = overall.get('p95_call_inference_latency_ms', 0.0)
        mean_call = overall.get('mean_call_inference_latency_ms', overall.get('mean_inference_latency_ms', 0.0))
        mean_ep = overall.get('mean_episode_inference_latency_ms', overall.get('mean_inference_latency_ms', 0.0))
        
        if call_count > 0:
            md.append(f"- **Mean Call Inference Latency**: {mean_call:.2f} ms / call (p95: {p95_call:.2f} ms across {call_count} calls)")
            md.append(f"- **Mean Episode Inference Latency**: {mean_ep:.2f} ms / episode")
        else:
            md.append(f"- **Mean Episode Inference Latency**: {mean_ep:.2f} ms / episode")
        md.append(f"- **Mean Simulation Latency**: {overall['mean_simulation_latency_ms']:.2f} ms / step")
        md.append("")
        md.append("## 2. Per-Task Performance Breakdown")
        md.append("")
        md.append("| Suite | Task ID | Task Name | Runs | Success Rate | 95% Wilson CI | Grasp Rate | Lift Rate | Place Rate | Mean Steps (Succ) | Infer Call (ms) |")
        md.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for t in summary["tasks"]:
            succ_steps_str = f"{t['mean_steps_to_success']:.1f}" if t["mean_steps_to_success"] is not None else "N/A"
            ci_str = f"[{t['wilson_ci_95'][0]*100:.1f}%, {t['wilson_ci_95'][1]*100:.1f}%]"
            infer_disp = t.get("mean_call_inference_latency_ms") or t.get("mean_inference_latency_ms", 0.0)
            md.append(
                f"| `{t['task_suite']}` | {t['task_id']} | {t['task_name']} | {t['total_episodes']} | "
                f"**{t['success_rate']*100:.1f}%** | {ci_str} | {t['grasp_success_rate']*100:.1f}% | "
                f"{t['lift_success_rate']*100:.1f}% | {t['placement_success_rate']*100:.1f}% | {succ_steps_str} | {infer_disp:.1f} |"
            )
        md.append("")
        md.append("## 3. Failure Mode & Behavioral Phase Distribution")
        md.append("")
        md.append("### Primary Failure Phase (Where the robot failed physically):")
        md.append("")
        md.append("| Failure Phase | Count | Percentage of Failures | Description |")
        md.append("| :--- | :---: | :---: | :--- |")
        total_failures = overall["total_episodes"] - overall["total_successes"]
        for phase, count in sorted(overall.get("global_failure_phases", {}).items(), key=lambda x: -x[1]):
            pct = (count / total_failures * 100) if total_failures > 0 else 0.0
            md.append(f"| `{phase}` | {count} | {pct:.1f}% | Behavioral phase at episode termination |")

        md.append("")
        md.append("### Primary Causal Attribution (SRS Section 22 Taxonomy):")
        md.append("")
        md.append("| Attribution Code | Count | Evidence Criteria |")
        md.append("| :--- | :---: | :--- |")
        for code, count in sorted(overall.get("global_primary_failure_codes", {}).items(), key=lambda x: -x[1]):
            criteria = "Concrete fault logged" if code != "UNATTRIBUTED" else "No conclusive instrumentation evidence to isolate visual perception vs control"
            md.append(f"| `{code}` | {count} | {criteria} |")
        md.append("")

        # 4. Multi-Stage Subtask Progression & Sequential Degradation (Phase 11)
        tasks_with_milestones = [t for t in summary["tasks"] if t.get("subtask_milestones")]
        if tasks_with_milestones:
            md.append("## 4. Multi-Stage Subtask Progression & Sequential Degradation (Phase 11)")
            md.append("")
            md.append("| Task ID | Task Name | Subtask Milestone | Completion Rate | Transition Failures | Atomic Failures | Mean Survival Steps |")
            md.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: |")
            for t in tasks_with_milestones:
                m_items = list(t["subtask_milestones"].items())
                surv_str = f"{t['mean_sequential_survival_steps']:.1f}" if t.get("mean_sequential_survival_steps") is not None else "N/A"
                if m_items:
                    for idx, (m_name, m_info) in enumerate(m_items):
                        rate_pct = f"{m_info['achieved_rate']*100:.1f}%"
                        if idx == 0:
                            md.append(
                                f"| {t['task_id']} | `{t['task_name']}` | `{m_name}`: {m_info['description']} | **{rate_pct}** | "
                                f"{t.get('sequence_transition_failures', 0)} | {t.get('atomic_manipulation_failures', 0)} | {surv_str} |"
                            )
                        else:
                            md.append(
                                f"| | | `{m_name}`: {m_info['description']} | **{rate_pct}** | | | |"
                            )
            md.append("")

        return "\n".join(md)
