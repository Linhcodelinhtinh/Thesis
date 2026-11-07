#!/usr/bin/env python
"""Consolidate and pair LIBERO-Mem evaluation results across run directories.

Adheres strictly to docs/RESULTS_MEM_2_TEST_PLAN.md:
- Custom V2 LIBERO-Mem evaluation (CUSTOM_V2_LIBERO_MEM / NON-COMPARABLE_OFFICIAL_PAPER).
- Matches B0 and F1 episodes by (task, initial_state_sha256).
- Computes macro and per-task Success Rate (SR), Subgoal Completion Rate (SCR),
  overshoot rate, policy inference latency, and paired delta SR.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple


def load_episodes(directory: Path) -> List[Dict[str, Any]]:
    """Load episode summaries from a run directory."""
    episodes = []
    summary_csv = directory / "summary.csv"
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append({
                    "episode_id": row["episode_id"],
                    "task": row["task"],
                    "condition": row["condition"],
                    "split": row["split"],
                    "state_idx": int(row["state_idx"]),
                    "source_demo": row.get("source_demo", ""),
                    "initial_state_sha256": row["initial_state_sha256"],
                    "success": row["success"].strip().lower() in ("true", "1"),
                    "scr": float(row["scr"]),
                    "overshot": row["overshot"].strip().lower() in ("true", "1"),
                    "steps": int(row["steps"]),
                    "policy_ms": float(row["policy_ms"]) if row.get("policy_ms") else 0.0,
                    "wall_time_s": float(row["wall_time_s"]) if row.get("wall_time_s") else 0.0,
                    "source_dir": str(directory),
                })
        return episodes

    # Fallback to episode directories if summary.csv not yet written
    episodes_dir = directory / "episodes"
    if episodes_dir.exists():
        for ep_dir in sorted(episodes_dir.iterdir()):
            status_file = ep_dir / "status.json"
            summary_file = ep_dir / "summary.json"
            if summary_file.exists():
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                episodes.append({
                    "episode_id": data.get("episode_id", ep_dir.name),
                    "task": data.get("task", ""),
                    "condition": data.get("condition", ""),
                    "split": data.get("split", ""),
                    "state_idx": int(data.get("state_idx", 0)),
                    "source_demo": data.get("source_demo", ""),
                    "initial_state_sha256": data.get("initial_state_sha256", ""),
                    "success": bool(data.get("success", False)),
                    "scr": float(data.get("scr", 0.0)),
                    "overshot": bool(data.get("overshot", False)),
                    "steps": int(data.get("steps", data.get("total_steps", 0))),
                    "policy_ms": float(data.get("policy_ms", data.get("avg_latency_ms", 0.0))),
                    "wall_time_s": float(data.get("wall_time_s", data.get("total_wall_time_s", 0.0))),
                    "source_dir": str(directory),
                })
    return episodes


def generate_report(episodes: List[Dict[str, Any]], title: str = "LIBERO-Mem Evaluation Summary") -> str:
    lines = [
        f"# {title}",
        "",
        "> [!IMPORTANT]",
        "> **Classification**: `CUSTOM_V2_LIBERO_MEM`",
        "> **Comparability**: `NON-COMPARABLE_OFFICIAL_PAPER`",
        "> This evaluation is conducted under the custom LIBERO-Mem V2 protocol and must not be compared to official LIBERO paper scores.",
        "",
        f"**Total Loaded Episodes**: {len(episodes)}",
        "",
        "### Performance Metrics by Task and Condition",
        "",
        "| Task | Condition | N | Success Rate (SR) | Mean SCR | Overshot Rate | Avg Policy Latency | Avg Wall Time |",
        "|:---|:---:|---:|---:|---:|---:|---:|---:|",
    ]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for ep in episodes:
        key = (ep["task"], ep["condition"])
        grouped.setdefault(key, []).append(ep)

    for (task, condition), group in sorted(grouped.items()):
        n = len(group)
        sr = sum(1 for e in group if e["success"]) / n
        mean_scr = sum(e["scr"] for e in group) / n
        overshot_rate = sum(1 for e in group if e["overshot"]) / n
        avg_lat = sum(e["policy_ms"] for e in group) / n
        avg_wall = sum(e["wall_time_s"] for e in group) / n
        lines.append(
            f"| **{task}** | `{condition}` | {n} | {sr*100:.1f}% ({sum(1 for e in group if e['success'])}/{n}) | "
            f"{mean_scr:.3f} | {overshot_rate*100:.1f}% | {avg_lat:.1f} ms | {avg_wall:.1f} s |"
        )

    # Macro summary across all tasks per condition
    lines.append("")
    lines.append("### Macro Performance across Conditions")
    lines.append("")
    lines.append("| Condition | N | Macro SR | Macro Mean SCR | Macro Overshot | Avg Latency |")
    lines.append("|:---:|---:|---:|---:|---:|---:|")
    cond_groups: Dict[str, List[Dict[str, Any]]] = {}
    for ep in episodes:
        cond_groups.setdefault(ep["condition"], []).append(ep)

    for cond, group in sorted(cond_groups.items()):
        n = len(group)
        sr = sum(1 for e in group if e["success"]) / n
        mean_scr = sum(e["scr"] for e in group) / n
        overshot = sum(1 for e in group if e["overshot"]) / n
        avg_lat = sum(e["policy_ms"] for e in group) / n
        lines.append(f"| `{cond}` | {n} | **{sr*100:.1f}%** | {mean_scr:.3f} | {overshot*100:.1f}% | {avg_lat:.1f} ms |")

    # Paired B0 vs F1 analysis
    pairs: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for ep in episodes:
        key = (ep["task"], ep["initial_state_sha256"])
        pairs.setdefault(key, {})[ep["condition"]] = ep

    complete_pairs = [p for p in pairs.values() if "B0" in p and "F1" in p]
    lines.append("")
    lines.append("### Paired Delta Analysis (F1 vs B0)")
    lines.append("")
    lines.append(f"- **Complete Matched Pairs**: {len(complete_pairs)} pairs (matched on task and exact `initial_state_sha256`).")
    if complete_pairs:
        delta_sr = sum(int(p["F1"]["success"]) - int(p["B0"]["success"]) for p in complete_pairs) / len(complete_pairs)
        delta_scr = sum(p["F1"]["scr"] - p["B0"]["scr"] for p in complete_pairs) / len(complete_pairs)
        lines.append(f"- **Paired $\\Delta$SR ($F1 - B0$)**: {delta_sr*100:+.1f}%")
        lines.append(f"- **Paired $\\Delta$SCR ($F1 - B0$)**: {delta_scr:+.3f}")
        lines.append("- *Note*: In small sample sizes (e.g. N=1 or N=10), paired delta is purely descriptive; confidence intervals are wide.")
    else:
        lines.append("- *No complete B0/F1 pairs found.*")

    # Episode-level table
    lines.append("")
    lines.append("### Individual Episode Results")
    lines.append("")
    lines.append("| Episode ID | Task | Cond | State Idx | Steps | Success | SCR | Overshot | Policy Latency | Wall Time |")
    lines.append("|:---|:---:|:---:|---:|---:|:---:|---:|:---:|---:|---:|")
    for ep in sorted(episodes, key=lambda x: (x["task"], x["state_idx"], x["condition"])):
        succ_mark = "✅ PASS" if ep["success"] else "❌ FAIL"
        over_mark = "⚠️ YES" if ep["overshot"] else "NO"
        lines.append(
            f"| `{ep['episode_id']}` | {ep['task']} | {ep['condition']} | {ep['state_idx']} | "
            f"{ep['steps']} | {succ_mark} | {ep['scr']:.3f} | {over_mark} | {ep['policy_ms']:.1f} ms | {ep['wall_time_s']:.1f} s |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path, help="Run directories containing summary.csv or episode folders")
    parser.add_argument("--output-report", type=Path, help="Path to write consolidated report markdown")
    parser.add_argument("--output-csv", type=Path, help="Path to write consolidated summary CSV")
    parser.add_argument("--title", default="LIBERO-Mem Evaluation Summary", help="Report title")
    args = parser.parse_args()

    all_episodes = []
    for d in args.dirs:
        if not d.exists():
            print(f"Warning: Directory not found: {d}", file=sys.stderr)
            continue
        eps = load_episodes(d)
        all_episodes.extend(eps)

    if not all_episodes:
        print("Error: No episodes found in provided directories.", file=sys.stderr)
        sys.exit(1)

    report_md = generate_report(all_episodes, title=args.title)

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(report_md + "\n", encoding="utf-8")
        print(f"Report written to: {args.output_report}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["episode_id", "task", "condition", "split", "state_idx", "source_demo",
                  "initial_state_sha256", "success", "scr", "overshot", "steps", "policy_ms", "wall_time_s", "source_dir"]
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_episodes)
        print(f"CSV summary written to: {args.output_csv}")

    print(report_md)


if __name__ == "__main__":
    main()
