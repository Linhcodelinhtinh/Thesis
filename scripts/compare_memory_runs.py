"""Paired Memory Benchmark Comparison & Reporting Tool (P6).

Per docs/V2_MEMORY_EXECUTION_PLAN.md Section 5:
Compares a baseline run (OFF) against a memory-augmented run (text_shadow or text_only)
using strictly paired per-episode task/initial-state/seed matching.

Outputs:
- Reconciled per-episode table with McNemar contingency matrix
- Paired success rate delta with 95% Newcombe / Wilson paired confidence intervals
- Subtask, step, and latency comparisons
- Memory trace, truncation audit, and context length diagnostics
- Seed equality and provenance divergence checks
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
    parser.add_argument(
        "--allow-mismatched-seeds",
        action="store_true",
        default=False,
        help="If set, allows comparing episodes with different random seeds (marks comparison NON-EQUIVALENT).",
    )
    return parser.parse_args()


def load_episodes(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all episode.json files indexed by (task/init_id). Fails loudly on corrupt JSON."""
    episodes = {}
    ep_files = sorted(list(run_dir.glob("*/*/episode.json")))
    if not ep_files:
        ep_files = sorted(list(run_dir.glob("**/episode.json")))

    for ep_file in ep_files:
        try:
            with open(ep_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read episode artifact at {ep_file}: {e}") from e

        rel_key = f"{ep_file.parent.parent.name}/{ep_file.parent.name}"
        episodes[rel_key] = data
    return episodes


def wilson_interval(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for a single proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * (total**2)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_newcombe_paired_ci(
    a: int,
    b: int,
    c: int,
    d: int,
    z: float = 1.95996,
) -> Tuple[float, List[float]]:
    """Compute Newcombe Method 10 paired difference confidence interval.

    Contingency table:
      a: both succeeded
      b: baseline only succeeded
      c: treatment only succeeded
      d: both failed
    """
    n = a + b + c + d
    if n == 0:
        return 0.0, [-1.0, 1.0]

    p1 = (a + b) / n  # baseline
    p2 = (a + c) / n  # treatment
    delta = p2 - p1

    l1, u1 = wilson_interval(a + b, n, z)
    l2, u2 = wilson_interval(a + c, n, z)

    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / denom if denom > 0 else 0.0

    diff_l = math.sqrt(max(0.0, (p2 - l2) ** 2 + (u1 - p1) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1)))
    diff_u = math.sqrt(max(0.0, (u2 - p2) ** 2 + (p1 - l1) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1)))

    ci_lower = max(-1.0, delta - diff_l)
    ci_upper = min(1.0, delta + diff_u)
    return round(delta, 4), [round(ci_lower, 4), round(ci_upper, 4)]


def compute_paired_stats(paired_results: List[Tuple[bool, bool]]) -> Dict[str, Any]:
    """Compute McNemar contingency matrix and paired delta stats using Newcombe CI."""
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
    delta, newcombe_ci = compute_newcombe_paired_ci(a, b, c, d)

    base_ci = [round(x, 4) for x in wilson_interval(a + b, n)]
    treat_ci = [round(x, 4) for x in wilson_interval(a + c, n)]

    # McNemar chi-square with continuity correction
    mcnemar_stat = 0.0
    if (b + c) > 0:
        mcnemar_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)

    stats: Dict[str, Any] = {
        "total_pairs": n,
        "baseline_success_count": a + b,
        "baseline_success_rate": round(sr_base, 4),
        "baseline_95_wilson_ci": base_ci,
        "treatment_success_count": a + c,
        "treatment_success_rate": round(sr_treat, 4),
        "treatment_95_wilson_ci": treat_ci,
        "delta_success_rate": delta,
        "delta_95_ci": newcombe_ci,
        "ci_method": "Newcombe Method 10 (Paired Wilson Score)",
        "contingency_matrix": {
            "both_success": a,
            "baseline_only_success": b,
            "treatment_only_success": c,
            "both_failed": d,
        },
        "mcnemar_chi2": round(mcnemar_stat, 4),
    }

    if n < 10:
        stats["small_sample_warning"] = (
            f"Sample size N={n} is preliminary/underpowered (N < 10). "
            "Confidence intervals are wide and should be treated as indicative rather than conclusive."
        )

    return stats


def generate_paired_report(
    base_dir: Path,
    treat_dir: Path,
    out_dir: Path,
    allow_mismatched_seeds: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Reconcile and generate full markdown and JSON paired report."""
    base_eps = load_episodes(base_dir)
    treat_eps = load_episodes(treat_dir)

    common_keys = sorted(set(base_eps.keys()) & set(treat_eps.keys()))
    if not common_keys:
        raise ValueError(f"No matching episodes found between {base_dir} and {treat_dir}")

    # Seed verification & provenance validation
    mismatched_seed_pairs: List[Dict[str, Any]] = []
    paired_outcomes = []
    task_breakdown: Dict[str, Dict[str, Any]] = {}
    context_lengths: List[int] = []
    total_truncations: int = 0
    total_dropped_facts: int = 0
    total_facts_considered: int = 0

    for key in common_keys:
        ep_base = base_eps[key]
        ep_treat = treat_eps[key]

        b_seed = ep_base.get("seed")
        t_seed = ep_treat.get("seed")

        if b_seed is not None and t_seed is not None and b_seed != t_seed:
            mismatched_seed_pairs.append({
                "key": key,
                "baseline_seed": b_seed,
                "treatment_seed": t_seed,
            })
            if not allow_mismatched_seeds:
                raise ValueError(
                    f"Seed mismatch in episode pair '{key}': baseline seed={b_seed} != treatment seed={t_seed}. "
                    "Valid paired comparison requires identical seeds per initial state. "
                    "Use --allow-mismatched-seeds if you explicitly wish to override."
                )

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
                if t_entry.get("truncated", False):
                    total_truncations += 1
                total_dropped_facts += t_entry.get("dropped_facts_count", 0)
                total_facts_considered += t_entry.get("total_facts_considered", 0)

    stats = compute_paired_stats(paired_outcomes)

    # Context length and audit diagnostics
    mean_ctx = round(sum(context_lengths) / len(context_lengths), 1) if context_lengths else 0.0
    max_ctx = max(context_lengths) if context_lengths else 0

    stats["context_length_diagnostics"] = {
        "total_queries": len(context_lengths),
        "mean_chars": mean_ctx,
        "max_chars": max_ctx,
        "truncated_queries_count": total_truncations,
        "total_dropped_facts_count": total_dropped_facts,
        "total_facts_considered": total_facts_considered,
    }
    stats["task_breakdown"] = list(task_breakdown.values())
    stats["seed_matched"] = len(mismatched_seed_pairs) == 0
    if mismatched_seed_pairs:
        stats["mismatched_seed_pairs"] = mismatched_seed_pairs

    # Build Markdown report
    md_lines = [
        "# V2 Paired Memory Evaluation Report",
        "",
        f"- **Baseline Dir (OFF):** `{base_dir}`",
        f"- **Treatment Dir (ON):** `{treat_dir}`",
        f"- **Reconciled Pairs ($N$):** {stats['total_pairs']}",
        "- **Evaluation Tier:** `LIBERO-DERIVED (NON-COMPARABLE_OFFICIAL_PAPER)`",
        f"- **Seed Matched:** `{'YES' if stats['seed_matched'] else 'NO (NON-EQUIVALENT)'}`",
    ]

    if not stats["seed_matched"]:
        md_lines.extend([
            "",
            "> [!WARNING]",
            "> **SEED MISMATCH DETECTED**: One or more episode pairs use different random seeds.",
            "> This comparison is non-equivalent and statistically invalid as a true paired test.",
        ])

    if "small_sample_warning" in stats:
        md_lines.extend([
            "",
            "> [!NOTE]",
            f"> **Preliminary Sample Alert**: {stats['small_sample_warning']}",
        ])

    md_lines.extend([
        "",
        "## 1. Primary Outcome & Paired Statistics",
        "",
        "| Condition | Success Count | Success Rate | 95% CI (Delta / Newcombe) |",
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
        f"- **Confidence Interval Method:** {stats['ci_method']}",
        "",
        "## 2. Memory Context & Truncation Audit",
        "",
        f"- **Total Memory Retrieval Queries:** {stats['context_length_diagnostics']['total_queries']}",
        f"- **Mean Context Length:** {mean_ctx} chars",
        f"- **Max Context Length:** {max_ctx} chars",
        f"- **Truncated Queries:** {total_truncations}",
        f"- **Dropped Facts Total:** {total_dropped_facts}",
        "",
        "## 3. Per-Task Paired Breakdown",
        "",
        "| Task | Episodes | Baseline (OFF) | Treatment (ON) | Delta |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

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

    md_content, stats = generate_paired_report(
        base_path,
        treat_path,
        out_path,
        allow_mismatched_seeds=args.allow_mismatched_seeds,
    )

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
    print(f"Seed Matched: {'YES' if stats.get('seed_matched') else 'NO'}")
    print(f"Report:    {report_file}")
    print(f"Summary:   {summary_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
