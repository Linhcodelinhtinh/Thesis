"""Placebo halt plan: does the benefit of the memory halt come from stopping at the right time, or only from stopping?

Adds control conditions to a `plan.json` from plan_halt_forks.py. Only the 16 episodes where the policy completes the
task can ever succeed, so placebo halts are drawn there; every other episode never halts (its recorded outcome stands).
Each placebo uses no memory evidence:

  PLC_any_s<k>     one random query step per episode, uniform over that episode's query steps (steps >= 10)
  PLC_open_s<k>    one random query step among the queries where the gripper is open (what the halt gate alone allows)
  PLC_fixed_<n>    the same fixed query step n for every episode that completed

`--seeds` random seeds per family. Categories follow classify_halt: early before the completion milestone, late at or
after overshoot, otherwise timely. Output: <output>/plan_placebo.json, same schema, run it with fork_halt_replay.py.
"""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.replay_xvla_e2_writers import load_episode
from src.evaluation.memory_halt import gripper_aperture

FIXED = (100, 150, 200)


def category(step, completion, overshoot):
    if step is None:
        return "no_halt"
    if completion is None or step < completion:
        return "premature"
    if overshoot is not None and step >= overshoot:
        return "late"
    return "timely"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = json.loads((args.run / "run_manifest.json").read_text(encoding="utf-8"))
    cadence = manifest["execution"]["execution_horizon"]
    names = [f"PLC_{fam}_s{k}" for fam in ("any", "open") for k in range(args.seeds)] + [f"PLC_fixed_{n}" for n in FIXED]
    for entry in plan["episodes"]:
        ref = entry["reference"]
        _, _, proprio, _, _, _ = load_episode(args.run / "episodes" / ref["episode"], cadence)
        completion, overshoot = ref["completion"], ref["overshoot"]
        queries = [s for s in range(cadence, len(proprio), cadence)]
        open_queries = [s for s in queries if gripper_aperture({"robot0_gripper_qpos": proprio[s][7:]}) >= 0.055]
        for name in names:
            step = None
            if completion is not None:
                if name.startswith("PLC_fixed_"):
                    n = int(name.rsplit("_", 1)[1])
                    step = n if n in queries else None
                else:
                    fam, seed = name.split("_")[1], int(name.rsplit("_s", 1)[1])
                    rng = random.Random(f"{fam}-{seed}-{ref['episode']}")
                    pool = queries if fam == "any" else open_queries
                    step = rng.choice(pool) if pool else None
            entry["conditions"][name] = dict(halt_step=step, category=category(step, completion, overshoot))
    plan["placebo"] = dict(conditions=names, seeds=args.seeds, note="placebo halts use no memory evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, default=str) + "\n", encoding="utf-8")
    for name in names:
        cats = [e["conditions"][name]["category"] for e in plan["episodes"]]
        print(name, {c: cats.count(c) for c in sorted(set(cats))})


if __name__ == "__main__":
    main()
