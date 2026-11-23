"""Halt-step plan for the closed-loop halt ablation, computed causally from one recorded X-VLA F1-on run.

A rollout with a halt equals the rollout without it up to the halt step (the decision reads only earlier frames), so
each condition is fully described by the query step where its halt latches. This script computes that step for every
episode and condition from the recorded detector evidence and proprioception (no GPU, no detector, no simulator) and
writes `plan.json`; `fork_halt_replay.py` then replays the recorded actions to that step and simulates the hold.

Conditions (hold mode, confirm_queries 1 unless named otherwise, dev-set cue thresholds, target = 1 for T1, 3 for T3):
  H1_oracle   simulator confirmed cycles >= target, gripper open (privileged upper bound)
  H2_v4       V4 writer count and plate support confirmed, gripper open
  H3_v4_cue   per-cycle fusion of V4 returns and the touchdown cue, gripper open, no support guard (main condition)
  H3g_v4_cue  H3 plus the V4 plate-support guard
  H3c2        H3 with two consecutive confirming queries
  CUE_only    touchdown cue alone (no detector), gripper open
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compare_memory_writers import RecordedDetector, SYNTHETIC, synthetic_observation, writer_specs
from scripts.extract_writer_replay_inputs import sha256_file
from scripts.replay_xvla_e2_writers import TARGETS, classify_halt, load_episode
from src.evaluation.memory_halt import HaltConfig, evaluate_halt, gripper_aperture
from src.memory.cycle_fusion import CycleFusionCounter
from src.memory.proprio_cycle_cue import ProprioCueConfig

CONDITIONS = {  # name -> (count source, support guard, confirm queries)
    "H1_oracle": ("oracle", False, 1),
    "H2_v4": ("v4", True, 1),
    "H3_v4_cue": ("fused", False, 1),
    "H3g_v4_cue": ("fused", True, 1),
    "H3c2": ("fused", False, 2),
    "CUE_only": ("cue", False, 1),
}


def latched(decisions, confirm):
    """Mark only the decision at which `confirm` consecutive confirming queries first hold (the latch)."""
    out, streak, done = [], 0, False
    for row in decisions:
        streak = streak + 1 if row["halt_condition"] else 0
        hit = not done and streak >= confirm
        done = done or hit
        out.append(dict(row, halt_condition=hit))
    return out


def plan_episode(directory, cadence, specs, cue_config):
    reference, evidence, proprio, actions, _, eval_frames = load_episode(directory, cadence)
    episode, target = reference["episode"], reference["target"]
    fed = {step: replace(ev, image_hashes=SYNTHETIC[step % 256]) for step, ev in enumerate(evidence)}
    v4 = specs["v4"][-1](episode, RecordedDetector(fed))
    fused, cue_only = CycleFusionCounter(cue_config), CycleFusionCounter(cue_config)
    queries, milestones = set(reference["query_steps"]), reference["milestones"]
    raw = {name: [] for name in CONDITIONS}
    for step, measured in enumerate(proprio):
        observation = synthetic_observation(step, measured[7:])
        aperture = gripper_aperture(observation)
        snapshot = v4.step(step, observation, actions[step - 1] if step else None)
        returns = [event.to_dict() for event in v4.last_events if event.type == "return"]
        fused.step(step, aperture, measured[:3], returns)
        cue_only.step(step, aperture, measured[:3], ())
        if step not in queries:
            continue
        state = snapshot.relations["bowl_1"]
        for name, (source, guard, _) in CONDITIONS.items():
            count = {"oracle": sum(m <= step for m in milestones), "fused": fused.count_at(step),
                     "cue": cue_only.count_at(step), "v4": state.returns}[source]
            view = SimpleNamespace(as_of_observation=step, profile=SimpleNamespace(count_cycles=True),
                                   relations={"bowl_1": SimpleNamespace(returns=count, status=state.status,
                                                                        reference=state.reference, armed=state.armed)})
            raw[name].append(evaluate_halt(view, observation, HaltConfig(target, require_on_support=guard)))
    conditions = {}
    for name, (_, _, confirm) in CONDITIONS.items():
        halt = classify_halt(latched(raw[name], confirm), reference, eval_frames)
        halt.pop("decision")
        conditions[name] = dict(halt_step=halt["step"], category=halt["category"],
                                delay_from_completion=halt["delay_from_completion"],
                                margin_before_overshoot=halt["margin_before_overshoot"])
    return dict(reference={k: reference[k] for k in ("episode", "task", "state_idx", "target", "milestones",
                                                     "completion", "overshoot", "frames")}, conditions=conditions)


def summarize(plan):
    return {name: dict(Counter(ep["conditions"][name]["category"] for ep in plan)) for name in CONDITIONS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="complete X-VLA F1 memory-on run directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "run_manifest.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "complete" or manifest["arguments"]["arm"] != "F1"
            or manifest["arguments"]["memory"] != "on"):
        raise ValueError("Expected a complete X-VLA F1 memory-on run")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cue_config, specs = ProprioCueConfig.load(), writer_specs()
    cadence = manifest["execution"]["execution_horizon"]
    plan = [plan_episode(d, cadence, specs, cue_config)
            for d in sorted(p for p in (args.run / "episodes").iterdir() if p.is_dir())]
    if Counter((e["reference"]["task"], e["reference"]["state_idx"]) for e in plan) != Counter(
            (task, state) for task in TARGETS for state in range(20)):
        raise ValueError("Expected the full 20 unseen states for each of T1 and T3")
    sources = ("scripts/xvla_v2/plan_halt_forks.py", "scripts/replay_xvla_e2_writers.py", "src/memory/cycle_fusion.py",
               "src/memory/proprio_cycle_cue.py", "src/evaluation/memory_halt.py",
               "src/memory/relation_cycle_writer_v4.py", "configs/memory/proprio_cycle_cue_v1.json")
    report = dict(classification="XVLA_E2_FIXED_TRACE_NON_COMPARABLE", run=str(args.run),
                  manifest_sha256=sha256_file(args.run / "run_manifest.json"),
                  source_sha256={path: sha256_file(ROOT / path) for path in sources},
                  conditions={name: dict(source=s, support_guard=g, confirm_queries=c)
                              for name, (s, g, c) in CONDITIONS.items()},
                  cue=cue_config.to_dict(), summary=summarize(plan), episodes=plan)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "plan.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    for name, counts in report["summary"].items():
        print(f"{name:12s} {counts}")


if __name__ == "__main__":
    main()
