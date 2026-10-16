"""Summarize 4-Way Ablation Results for SmolVLA on LIBERO-10 Tasks 0 & 1."""

import json
from pathlib import Path

ROOT = Path("experiments/results/v2_memory")
CONDITIONS = [
    ("Original Prompt (OFF)", ROOT / "ablation_original_prompt"),
    ("Shadow (OFF Prompt + BG Memory)", ROOT / "ablation_shadow"),
    ("Static Suffix (Prompt Contamination)", ROOT / "ablation_static_suffix"),
    ("Validated Dynamic (Grounded Memory)", ROOT / "ablation_validated_dynamic"),
    ("Legacy Buggy Compact (prompt_shoren)", ROOT / "prompt_shoren_obs_text"),
]

print("=" * 80)
print(f"{'Condition':<36} | {'Task 0':<10} | {'Task 1':<10} | {'Total Success':<14} | {'Rate':<8}")
print("-" * 80)

data_by_cond = {}

for name, path in CONDITIONS:
    if not path.exists():
        continue
    summary_file = path / "benchmark_summary.json"
    if not summary_file.exists():
        continue
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Count task 0 and task 1
    t0_succ, t0_total = 0, 0
    t1_succ, t1_total = 0, 0

    ep_files = sorted(list(path.glob("*/*/episode.json")))
    for ep in ep_files:
        with open(ep, "r", encoding="utf-8") as f:
            ep_data = json.load(f)
        task_str = ep.parent.parent.name
        init_id = ep_data.get("initial_state_id", 0)
        if init_id > 4:
            continue
        succ = 1 if ep_data.get("success") else 0
        if "task0" in task_str:
            t0_succ += succ
            t0_total += 1
        elif "task1" in task_str:
            t1_succ += succ
            t1_total += 1

    tot_succ = t0_succ + t1_succ
    tot_ep = t0_total + t1_total
    rate_str = f"{(tot_succ / tot_ep * 100):.1f}%" if tot_ep > 0 else "N/A"
    print(f"{name:<36} | {t0_succ}/{t0_total:<8} | {t1_succ}/{t1_total:<8} | {tot_succ}/{tot_ep:<12} | {rate_str:<8}")
    data_by_cond[name] = {
        "t0": f"{t0_succ}/{t0_total}",
        "t1": f"{t1_succ}/{t1_total}",
        "total": f"{tot_succ}/{tot_ep}",
        "rate": rate_str,
    }

print("=" * 80)
