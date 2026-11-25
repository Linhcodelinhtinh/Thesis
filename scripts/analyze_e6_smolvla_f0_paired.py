#!/usr/bin/env python3
"""Paired B0 / F0 / F1-off comparison for SmolVLA E6 retention on LIBERO-10 (50 local states).

Same bootstrap as analyze_e6_retention_paired_bootstrap.py (10,000 resamples, stratified
within each task, seed 42, percentile interval) plus an exact two-sided McNemar test.
"""
from __future__ import annotations

import hashlib, json, math, random, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/thesis/evidence/e6_smolvla_f0_paired_20261125.json"
ARMS = {
    "B0": ("experiments/results/e6_retention_locked_50", "B0", "resources/checkpoints/smolvla_libero"),
    "F0": ("experiments/results/e6_f0_windows_20261125", "F0", "results_f0_author_split_v2/output_f0/pretrained_model"),
    "F1-off": ("experiments/results/e6_retention_locked_50", "F1_actual", "results_f1_author_split_v2/output_f1/pretrained_model"),
}
N_TASKS, PER_TASK, BOOTSTRAPS, SEED = 10, 5, 10_000, 42


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def percentile(v, q):
    pos = q * (len(v) - 1); lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] if lo == hi else v[lo] * (hi - pos) + v[hi] * (pos - lo)


def load(arm):
    root, prefix, ckpt = ARMS[arm]
    out = {}
    for t in range(N_TASKS):
        files = sorted((ROOT / root / f"{prefix}_task_{t}" / f"libero_10_task{t}").glob("init_*/episode.json"))
        if len(files) != PER_TASK:
            raise ValueError(f"{arm} task {t}: {len(files)} episodes")
        for p in files:
            d = json.loads(p.read_text(encoding="utf-8"))
            pv = d.get("provenance") or {}
            if pv.get("memory_condition") != "off" or d.get("execution_horizon_s") != 10 \
                    or pv.get("policy_chunk_size") != 50 or pv.get("model_name") != "smolvla_libero":
                raise ValueError(f"contract mismatch {p}")
            if not str(d.get("checkpoint", "")).replace("\\", "/").endswith(ckpt):
                raise ValueError(f"checkpoint mismatch {p}")
            key = (t, d["initial_state_id"], d["seed"])
            out[key] = dict(success=bool(d["success"]), task=d["task_name"],
                            eef0=d["settling_snapshot"]["pre_settling"]["eef_pos"],
                            steps=d["num_steps"], term=d["diagnostics"]["termination_reason"],
                            path=p.relative_to(ROOT).as_posix(), sha256=sha256(p))
    return out


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(A, B, a, b):
    keys = sorted(A)
    rows = [(k, A[k]["success"], B[k]["success"]) for k in keys]
    both = sum(x and y for _, x, y in rows); a_only = sum(x and not y for _, x, y in rows)
    b_only = sum(y and not x for _, x, y in rows); none = sum(not x and not y for _, x, y in rows)
    by_task = [[int(y) - int(x) for (k, x, y) in rows if k[0] == t] for t in range(N_TASKS)]
    rng = random.Random(SEED); samples = []
    for _ in range(BOOTSTRAPS):
        s = []
        for r in by_task:
            s.extend(rng.choices(r, k=len(r)))
        samples.append(sum(s) / len(s))
    samples.sort()
    return {"comparison": f"{b} minus {a}", "pairs": len(rows),
            f"{a}_successes": both + a_only, f"{b}_successes": both + b_only,
            "both_success": both, f"{a}_only": a_only, f"{b}_only": b_only, "both_failure": none,
            "paired_difference": (b_only - a_only) / len(rows),
            "bootstrap_95": [percentile(samples, .025), percentile(samples, .975)],
            "mcnemar_exact_two_sided_p": mcnemar_exact(a_only, b_only)}


def main():
    arms = {a: load(a) for a in ARMS}
    keys = set(arms["B0"])
    for a in arms:
        if set(arms[a]) != keys or len(keys) != 50:
            raise ValueError(f"{a} roster differs from B0")
    checks = {a: {"same_task_state_seed_keys": True,
                  "same_task_name": all(arms[a][k]["task"] == arms["B0"][k]["task"] for k in keys),
                  "same_pre_settling_eef": sum(arms[a][k]["eef0"] == arms["B0"][k]["eef0"] for k in keys)}
              for a in arms}
    per_task = [{"task_id": t, **{a: sum(v["success"] for k, v in arms[a].items() if k[0] == t) for a in arms}}
                for t in range(N_TASKS)]
    res = {
        "analysis_id": "e6-smolvla-b0-f0-f1off-paired-2026-11-25",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "CUSTOM_LIBERO_LOCAL", "comparability": "NON-COMPARABLE_OFFICIAL_PAPER",
        "method": "exact two-sided McNemar on discordant pairs; 10,000 task-stratified paired bootstrap, seed 42, percentile 95%",
        "totals": {a: sum(v["success"] for v in arms[a].values()) for a in arms},
        "per_task": per_task, "roster_checks": checks,
        "comparisons": [compare(arms["B0"], arms["F0"], "B0", "F0"),
                        compare(arms["B0"], arms["F1-off"], "B0", "F1-off"),
                        compare(arms["F0"], arms["F1-off"], "F0", "F1-off")],
        "episodes": [{"task_id": k[0], "initial_state_id": k[1], "seed": k[2],
                      **{a: {x: arms[a][k][x] for x in ("success", "steps", "term", "path", "sha256")} for a in arms}}
                     for k in sorted(keys)],
        "analysis_code_sha256": sha256(Path(__file__)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("totals", "per_task", "roster_checks", "comparisons")}, indent=1))


if __name__ == "__main__":
    main()
