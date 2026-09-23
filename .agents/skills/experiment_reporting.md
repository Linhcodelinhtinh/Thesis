# Experiment Reporting Skill

## Trigger When:
- Compiling evaluation results after benchmark runs.
- Generating automated reports, CSVs, or markdown summaries.
- Preparing comparison tables between candidate VLA models.
- Archiving experiment runs for thesis documentation.

## Guidelines:
1. All reported numbers must originate from logged `episode.json` files.
2. Every report must link to:
   - Git commit hash of the evaluation repository.
   - Pinned benchmark commit hash.
   - Model checkpoint version and hash.
   - Hardware platform and environment manifest.
3. If an experiment used any non-standard modification, it must be explicitly labeled as **NON-COMPARABLE**.

## Required Report Deliverables:
- **`report.md`**: Executive summary with methodology, success metrics, and key observations.
- **`summary.csv`**: Task-level breakdown containing:
  - `task_name`
  - `num_episodes`
  - `success_rate`
  - `grasp_rate`
  - `mean_steps`
  - `mean_latency_ms`
- **`failure_distribution.csv`**: Counts and percentages of each failure type across tasks.
- **Visual Artifacts**:
  - Exactly 3 representative successful rollout videos (`success_ep_{id}.mp4`).
  - Exactly 3 representative failure rollout videos (`failure_ep_{id}.mp4`).
- **Plots (where applicable)**:
  - Per-task success rate bar charts.
  - Failure taxonomy breakdown pie/stacked-bar chart.

## Directory Structure:
```text
experiments/<suite_name>/<model_name>/
├── config.yaml
├── report.md
├── summary.csv
├── failure_distribution.csv
├── videos/
└── raw_episodes/
```
