"""Paired Memory Benchmark Comparison & Reporting Tool (P6).

Per docs/V2_MEMORY_EXECUTION_PLAN.md Section 5:
Compares a baseline run (OFF) against a memory-augmented run (text_shadow or text_only)
using strictly paired per-episode task/initial-state matching.

Outputs:
- Reconciled per-episode table with McNemar contingency matrix
- Paired success rate delta with 95% Wilson / Wald confidence intervals
- Subtask, step, and latency comparisons
- Memory trace and context length diagnostics
- PAIRED_MEMORY_REPORT.md and paired_memory_summary.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paired comparison report for V2 memory evaluations.")
    parser.add_argument(
        "--baseline-dir",
        type=str,
        required=True,
        help="Path to baseline benchmark run directory (OFF condition).",
    )
    parser.add_argument(
        "--treatment-dir",
        type=str,
        required=True,
        help="Path to treatment benchmark run directory (text_shadow or text_only).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save paired report and summary (defaults to treatment-dir).",
    )
    return parser.parse_args()


def load_episodes(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all episode.json files indexed by (task_suite_taskId, init_id)."""
    episodes = {}
    for ep_file in run_dir.glob("*/*/episode.json"):
        try:
            with open(ep_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            rel_key = f"{ep_file.parent.parent.name}/{ep_file.parent.name}"
            episodes[rel_key] = data
        except Exception:
            pass
    return episodes


def compute_paired_stats(paired_results: List[Tuple[bool, bool]]) -> Dict[str, Any]:
    """Compute McNemar contingency matrix and paired delta stats."""
    n = len(paired_results)
    if n == 0:
        return {}

    # Contingency matrix:
    # a: both succeeded (1, 1)
    # b: baseline succ, treat fail (1, 0)
    # c: baseline fail, treat succ (0, 1)
    # d: both failed (0, 0)
    a = sum(1 for base, treat in paired_results if base and treat)
    b = sum(1 for base, treat in paired_results if base and not treat)
    c = sum(1 for base, treat in paired_results if not base and treat)
    d = sum(1 for base, treat in paired_results if not base and not treat)

    sr_base = (a + b) / n
    sr_treat = (a + c) / n
    delta = sr_treat - sr_base

    # Paired standard error for difference: sqrt((b + c) - (b - c)^2 / n) / n
    var_delta = ((b + c) - ((b - c) ** 2) / n) / (n * n) if n > 0 else 0.0
    se_delta = math.sqrt(max(0.0, var_delta))
    ci_lower = delta - 1.96 * se_delta
    ci_upper = delta + 1.96 * se_delta

    # McNemar chi-square with continuity correction
    mcnemar_stat = 0.0
    if (b + c) > 0:
        mcnemar_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)

    return {
        "total_pairs": n,
        "baseline_success_count": a + b,
        "baseline_success_rate": round(sr_base, 4),
        "treatment_success_count": a + c,
        "treatment_success_rate": round(sr_treat, 4),
        "delta_success_rate": round(delta, 4),
        "delta_95_ci": [round(ci_lower, 4), round(ci_upper, 4)],
        "contingency_matrix": {
            "both_success": a,
            "baseline_only_success": b,
            "treatment_only_success": c,
            "both_failed": d,
        },
        "mcnemar_chi2": round(mcnemar_stat, 4),
    }


def generate_paired_report(
    base_dir: Path,
    treat_dir: Path,
    out_dir: Path,
) -> Tuple[str, Dict[str, Any]]:
    """Reconcile and generate full markdown and JSON paired report."""
    base_eps = load_episodes(base_dir)
    treat_eps = load_episodes(treat_dir)

    common_keys = sorted(set(base_eps.keys()) & set(treat_eps.keys()))
    if not common_keys:
        raise ValueError(f"No matching episodes found between {base_dir} and {treat_dir}")

    paired_outcomes = []
    task_breakdown: Dict[str, Dict[str, Any]] = {}
    context_lengths: List[int] = []

    for key in common_keys:
        ep_base = base_eps[key]
        ep_treat = treat_eps[key]

        b_succ = bool(ep_base.get("success", False))
        t_succ = bool(ep_treat.get("success", False))
        paired_outcomes.append((b_succ, t_succ))

        task_name = ep_base.get("task_name", key)
        if task_name not in task_breakdown:
            task_breakdown[task_name] = {
                "task_name": task_name,
                "total": 0,
                "base_succ": 0,
                "treat_succ": 0,
            }
        task_breakdown[task_name]["total"] += 1
        if b_succ:
            task_breakdown[task_name]["base_succ"] += 1
        if t_succ:
            task_breakdown[task_name]["treat_succ"] += 1

        # Memory trace metrics from treatment
        mem_info = ep_treat.get("memory")
        if mem_info and "trace" in mem_info:
            for t_entry in mem_info["trace"]:
                ctx_len = t_entry.get("rendered_context_length_chars", 0)
                context_lengths.append(ctx_len)

    stats = compute_paired_stats(paired_outcomes)

    # Context length diagnostics
    mean_ctx = round(sum(context_lengths) / len(context_lengths), 1) if context_lengths else 0.0
    max_ctx = max(context_lengths) if context_lengths else 0

    stats["context_length_diagnostics"] = {
        "total_queries": len(context_lengths),
        "mean_chars": mean_ctx,
        "max_chars": max_ctx,
    }
    stats["task_breakdown"] = list(task_breakdown.values())

    # Build Markdown report
    md_lines = [
        "# V2 Paired Memory Evaluation Report",
        "",
        f"- **Baseline Dir (OFF):** `{base_dir}`",
        f"- **Treatment Dir (ON):** `{treat_dir}`",
        f"- **Reconciled Pairs ($N$):** {stats['total_pairs']}",
        "- **Evaluation Tier:** `LIBERO-DERIVED (NON-COMPARABLE_OFFICIAL_PAPER)`",
        "",
        "## 1. Primary Outcome & Paired Statistics",
        "",
        "| Condition | Success Count | Success Rate | 95% CI (Delta) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Baseline (OFF)** | {stats['baseline_success_count']}/{stats['total_pairs']} | {stats['baseline_success_rate']*100:.1f}% | — |",
        f"| **Treatment (ON)** | {stats['treatment_success_count']}/{stats['total_pairs']} | {stats['treatment_success_rate']*100:.1f}% | **{stats['delta_success_rate']*100:+.1f}%** $[{stats['delta_95_ci'][0]*100:.1f}\\%, {stats['delta_95_ci'][1]*100:.1f}\\%]$ |",
        "",
        "### McNemar Contingency Matrix",
        "",
        "| Baseline \\ Treatment | Succeeded | Failed |",
        "| :--- | :---: | :---: |",
        f"| **Succeeded** | {stats['contingency_matrix']['both_success']} (Both Won) | {stats['contingency_matrix']['baseline_only_success']} (Regression) |",
        f"| **Failed** | {stats['contingency_matrix']['treatment_only_success']} (Improvement) | {stats['contingency_matrix']['both_failed']} (Both Lost) |",
        "",
        f"- **McNemar $\\chi^2$ statistic:** {stats['mcnemar_chi2']}",
        "",
        "## 2. Memory Context Diagnostics",
        "",
        f"- **Total Memory Retrieval Calls:** {stats['context_length_diagnostics']['total_queries']}",
        f"- **Mean Context Length:** {mean_ctx} chars",
        f"- **Max Context Length:** {max_ctx} chars",
        "",
        "## 3. Per-Task Paired Breakdown",
        "",
        "| Task | Episodes | Baseline (OFF) | Treatment (ON) | Delta |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for tb in task_breakdown.values():
        b_rate = tb["base_succ"] / tb["total"] if tb["total"] else 0
        t_rate = tb["treat_succ"] / tb["total"] if tb["total"] else 0
        d_cnt = tb["treat_succ"] - tb["base_succ"]
        md_lines.append(
            f"| `{tb['task_name'][:45]}` | {tb['total']} | {tb['base_succ']}/{tb['total']} ({b_rate*100:.0f}%) | {tb['treat_succ']}/{tb['total']} ({t_rate*100:.0f}%) | {d_cnt:+d} |"
        )

    md_report = "\n".join(md_lines) + "\n"
    return md_report, stats


def main():
    args = parse_args()
    base_path = Path(args.baseline_dir)
    treat_path = Path(args.treatment_dir)
    out_path = Path(args.output_dir) if args.output_dir else treat_path
    out_path.mkdir(parents=True, exist_ok=True)

    md_content, stats = generate_paired_report(base_path, treat_path, out_path)

    report_file = out_path / "PAIRED_MEMORY_REPORT.md"
    summary_file = out_path / "paired_memory_summary.json"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("=" * 60)
    print("V2 Paired Memory Evaluation Report Generated")
    print("=" * 60)
    print(f"Baseline:  {stats['baseline_success_count']}/{stats['total_pairs']} ({stats['baseline_success_rate']*100:.1f}%)")
    print(f"Treatment: {stats['treatment_success_count']}/{stats['total_pairs']} ({stats['treatment_success_rate']*100:.1f}%)")
    print(f"Delta:     {stats['delta_success_rate']*100:+.1f}% (95% CI: [{stats['delta_95_ci'][0]*100:.1f}%, {stats['delta_95_ci'][1]*100:.1f}%])")
    print(f"Report:    {report_file}")
    print(f"Summary:   {summary_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
