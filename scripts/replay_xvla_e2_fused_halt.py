"""Replay per-cycle fusion (vision writer + proprioceptive touchdown) and its memory halt on the X-VLA F1-on E2 run.

Same fixed-trace protocol as replay_xvla_e2_writers.py (recorded detector evidence, recorded proprioception, the
simulator's confirmed-cycle increments as reference, halt decisions at the recorded query boundaries), so a halt is a
counterfactual timing check, not a closed-loop result. It reads only and writes one new directory.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_libero_mem_writer import match_events
from scripts.compare_memory_writers import RecordedDetector, SYNTHETIC, synthetic_observation, writer_specs
from scripts.extract_writer_replay_inputs import sha256_file
from scripts.replay_xvla_e2_writers import TARGETS, classify_halt, load_episode
from src.evaluation.memory_halt import HaltConfig, evaluate_halt, gripper_aperture
from src.memory.cycle_fusion import CycleFusionCounter
from src.memory.proprio_cycle_cue import ProprioCueConfig

VISION = ("v1", "v4")
TOLERANCES = (20, 30)
# Only the dev-set thresholds: relaxing lift/touchdown height on this run bought at most +5 matches for +8 false
# positives (offline sweep), and any relaxed setting would be tuned on the evaluation run itself.
CUES = dict(dev=ProprioCueConfig.load())
GUARDS = ("none", "vision_support")


def replay(directory, cadence, specs):
    reference, evidence, proprio, actions, _, eval_frames = load_episode(directory, cadence)
    episode, target = reference["episode"], reference["target"]
    fed = {step: replace(ev, image_hashes=SYNTHETIC[step % 256]) for step, ev in enumerate(evidence)}
    writers = {name: specs[name][-1](episode, RecordedDetector(fed)) for name in VISION}
    counters = {(v, c): CycleFusionCounter(CUES[c]) for v in VISION for c in CUES}
    decisions = {(v, c, g): [] for v in VISION for c in CUES for g in GUARDS}
    queries = set(reference["query_steps"])
    for step, measured in enumerate(proprio):
        observation = synthetic_observation(step, measured[7:])
        aperture = gripper_aperture(observation)
        snapshots = {}
        for name, writer in writers.items():
            snapshots[name] = writer.step(step, observation, actions[step - 1] if step else None)
            returns = [event.to_dict() for event in writer.last_events if event.type == "return"]
            for cue in CUES:
                counters[(name, cue)].step(step, aperture, measured[:3], returns)
        if step not in queries:
            continue
        for (name, cue), counter in counters.items():
            state = snapshots[name].relations["bowl_1"]
            fused = SimpleNamespace(as_of_observation=step, profile=SimpleNamespace(count_cycles=True),
                                    relations={"bowl_1": SimpleNamespace(returns=counter.count_at(step),
                                                                         status=state.status,
                                                                         reference=state.reference,
                                                                         armed=state.armed)})
            for guard in GUARDS:
                config = HaltConfig(target, require_on_support=guard == "vision_support")
                decisions[(name, cue, guard)].append(evaluate_halt(fused, observation, config))
    methods = {}
    for (name, cue), counter in counters.items():
        credits = sorted(credit.step for credit in counter.credits.values())
        for guard in GUARDS:
            halt = classify_halt(decisions[(name, cue, guard)], reference, eval_frames)
            halt.pop("decision")
            methods[f"{name}+{cue}:{guard}"] = dict(
                credits=[credit.__dict__ for credit in counter.credits.values()],
                duplicates=len(counter.duplicates), halt=halt,
                matching={tol: {k: v for k, v in match_events(credits, reference["milestones"], tol).items()
                                if k != "matches"} for tol in TOLERANCES})
    return dict(reference={k: reference[k] for k in ("episode", "task", "state_idx", "target", "milestones",
                                                     "completion", "overshoot", "frames")}, methods=methods)


def summarize(episodes):
    out = {}
    for method in episodes[0]["methods"]:
        rows = [ep["methods"][method] for ep in episodes]
        halts = Counter(row["halt"]["category"] for row in rows)
        timely = [row["halt"] for row in rows if row["halt"]["category"] == "timely"]
        out[method] = dict(
            matching={tol: {key: sum(row["matching"][tol][key] for row in rows)
                            for key in ("true_positive", "false_positive", "false_negative")} for tol in TOLERANCES},
            halt=dict(halts), timely_final_predicate=sum(bool(row["final_predicate_at_halt"]) for row in timely),
            timely_margins=[row["margin_before_overshoot"] for row in timely],
            premature_episodes=[ep["reference"]["episode"] for ep, row in zip(episodes, rows)
                                if row["halt"]["category"] == "premature"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "run_manifest.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "complete" or manifest["arguments"]["arm"] != "F1"
            or manifest["arguments"]["memory"] != "on"):
        raise ValueError("Expected a complete X-VLA F1 memory-on run")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    specs = writer_specs()
    cadence = manifest["execution"]["execution_horizon"]
    episodes = [replay(d, cadence, specs) for d in sorted(p for p in (args.run / "episodes").iterdir() if p.is_dir())]
    if Counter((e["reference"]["task"], e["reference"]["state_idx"]) for e in episodes) != Counter(
            (task, state) for task in TARGETS for state in range(20)):
        raise ValueError("Expected the full 20 unseen states for each of T1 and T3")
    sources = ("scripts/replay_xvla_e2_fused_halt.py", "scripts/replay_xvla_e2_writers.py",
               "src/memory/cycle_fusion.py", "src/memory/proprio_cycle_cue.py", "src/evaluation/memory_halt.py",
               "src/memory/relation_cycle_writer_v4.py", "configs/memory/proprio_cycle_cue_v1.json")
    report = dict(classification="XVLA_E2_FIXED_TRACE_NON_COMPARABLE",
                  note="Counterfactual halt timing on a fixed trace; cue thresholds are the development-set ones",
                  run=str(args.run), manifest_sha256=sha256_file(args.run / "run_manifest.json"),
                  source_sha256={path: sha256_file(ROOT / path) for path in sources},
                  cues={name: config.to_dict() for name, config in CUES.items()},
                  completions=sum(e["reference"]["completion"] is not None for e in episodes),
                  milestones=sum(len(e["reference"]["milestones"]) for e in episodes),
                  overall=summarize(episodes),
                  by_task={task: summarize([e for e in episodes if e["reference"]["task"] == task]) for task in TARGETS})
    args.output_dir.mkdir(parents=True)
    for name, value in (("summary.json", report), ("episodes.json", episodes)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    for method, row in report["overall"].items():
        print(f"{method:28s} m20={row['matching'][20]} halt={row['halt']} premature={row['premature_episodes']}")


if __name__ == "__main__":
    main()
