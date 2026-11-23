"""Replay existing writers and halt decisions on the recorded X-VLA F1-on E2 cohort."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.audit_libero_mem_writer import match_events
from scripts.compare_memory_writers import (
    SYNTHETIC, RecordedDetector, parse_evidence, synthetic_observation, writer_specs,
)
from scripts.extract_writer_replay_inputs import rows, sha256_file
from src.evaluation.memory_halt import HaltConfig, evaluate_halt, gripper_aperture
from src.memory.proprio_cycle_cue import ProprioCueConfig, ProprioCycleCue
from src.memory.relation_cycle_writer_v4 import v4_source_hashes

WRITERS = ("v1", "v3", "v4")
TOLERANCES = (10, 20, 30)
TARGETS = {"T1": 1, "T3": 3}
RELATION_KEYS = ("returns", "status", "reference", "armed")
ARTIFACTS = ("memory_trace.jsonl", "observations.jsonl", "actions.jsonl", "queries.jsonl", "summary.json")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, episode, message):
    if not condition:
        raise ValueError(f"{episode}: {message}")


def load_episode(directory, cadence):
    episode = directory.name
    status_path = directory / "status.json"
    status = read(status_path)
    task = status["task"]
    require(status["status"] == "complete" and not status["logging_errors"]
            and status["episode_id"] == episode and status["condition"] == "F1-mem-on"
            and status["split"] == "unseen", episode, "invalid episode receipt")
    hashes = {name: sha256_file(directory / name) for name in ARTIFACTS}
    require(all(value == status["files_sha256"][name] for name, value in hashes.items()),
            episode, "artifact SHA-256 mismatch")
    result = read(directory / "summary.json")["result"]
    end = result["total_steps"]
    query_steps = [row["step"] for row in rows(directory / "queries.jsonl")]
    require(query_steps == list(range(0, end, cadence)), episode, "query boundaries differ from E2 cadence")
    actions = []
    for step, row in enumerate(rows(directory / "actions.jsonl")):
        action = np.asarray(row["action"], dtype=float)
        require(row["step"] == step and action.shape == (7,) and np.isfinite(action).all(),
                episode, f"invalid action at {step}")
        actions.append(action)
    require(len(actions) == end, episode, "action count differs from horizon")

    payloads, proprio, trace, milestones, eval_frames = [], [], [], [], []
    overshoot = None
    apertures_checked = 0
    target = TARGETS[task]
    for step, (frame, obs) in enumerate(zip(rows(directory / "memory_trace.jsonl"),
                                           rows(directory / "observations.jsonl"), strict=True)):
        snap = frame["snapshot"]
        require(frame["step"] == obs["step"] == snap["as_of_observation"] == step
                and snap["episode_id"] == episode and snap["last_consumed_action"] == step - 1,
                episode, f"trace and observations are misaligned at {step}")
        measured = np.asarray(obs["raw_proprioception_pos_quatxyzw_gripper"], dtype=float)
        require(measured.shape == (9,) and np.isfinite(measured).all()
                and abs(np.linalg.norm(measured[3:7]) - 1) < 1e-3, episode, f"invalid proprioception at {step}")
        aperture = gripper_aperture({"robot0_gripper_qpos": measured[7:]})
        grip_record = snap["evidence"].get(f"gripper:{step}")
        if grip_record is not None:
            require(abs(aperture - grip_record["aperture"]) <= 1e-9,
                    episode, f"gripper aperture disagrees with recorded writer at {step}")
            apertures_checked += 1
        proprio.append(measured)
        payloads.append(frame["signals"]["visual_evidence"])
        trace.append((snap["relations"]["bowl_1"], frame["signals"]["new_events"]))

        # These measurements are only used to score the predictions.
        ev = obs["info"]["eval_result"]
        count = ev["confirmed_cycles"]
        require(count == ev["completed_subgoals"] and ev["target_cycles"] == target
                and len(milestones) <= count <= min(target, len(milestones) + int(step > 0)),
                episode, f"invalid evaluator progress at {step}")
        if count > len(milestones):
            milestones.append(step)
        if ev["overshot"] and overshoot is None:
            overshoot = step
        eval_frames.append(obs["info"]["final_predicate"])
    require(len(proprio) == end + 1 and len(milestones) == result["confirmed_cycles"]
            and (overshoot is not None) == result["overshot"], episode, "summary disagrees with frame logs")
    completion = milestones[-1] if len(milestones) == target else None
    reference = dict(episode=episode, task=task, state_idx=status["state_idx"], target=target,
                     milestones=milestones, completion=completion, overshoot=overshoot,
                     frames=len(proprio), query_steps=query_steps, apertures_checked=apertures_checked, sha256=hashes,
                     status_sha256=sha256_file(status_path))
    return reference, parse_evidence(payloads), proprio, actions, trace, eval_frames


def classify_halt(decisions, reference, eval_frames):
    decision = next((row for row in decisions if row["halt_condition"]), None)
    completion, overshoot = reference["completion"], reference["overshoot"]
    step = decision["snapshot_step"] if decision else None
    if step is None:
        category = "no_halt"
    elif completion is None or step < completion:
        category = "premature"
    elif overshoot is not None and step >= overshoot:
        category = "late"
    else:
        category = "timely"
    return dict(category=category, step=step, decision=decision,
                final_predicate_at_halt=eval_frames[step] if step is not None else None,
                delay_from_completion=step - completion if step is not None and completion is not None else None,
                margin_before_overshoot=overshoot - step if step is not None and overshoot is not None else None,
                reasons=dict(Counter(row["reason"] for row in decisions)),
                first_query_with_target_count=next((row["snapshot_step"] for row in decisions
                                                    if (row["observed_returns"] or 0) >= reference["target"]), None))


def replay(directory, cadence, specs, cue_config):
    reference, evidence, proprio, actions, trace, eval_frames = load_episode(directory, cadence)
    episode, target = reference["episode"], reference["target"]
    # The adapter rebinds digests to placeholder pixels. Detector boxes, IDs, scores,
    # metadata, and frame order are the archived values. No detector is run on pixels.
    fed = {step: replace(ev, image_hashes=SYNTHETIC[step % 256]) for step, ev in enumerate(evidence)}
    detectors = {name: RecordedDetector(fed) for name in WRITERS}
    writers = {name: specs[name][-1](episode, detectors[name]) for name in WRITERS}
    cue = ProprioCycleCue(cue_config)
    names = (*WRITERS, "proprio")
    events = {name: [] for name in names}
    decisions = {name: [] for name in names}
    queries = set(reference["query_steps"])
    vision_halt = HaltConfig(target)
    cue_halt = replace(vision_halt, require_on_support=False)
    for step, measured in enumerate(proprio):
        observation = synthetic_observation(step, measured[7:])
        action = actions[step - 1] if step else None
        for name, writer in writers.items():
            snapshot = writer.step(step, observation, action)
            new_events = [event.to_dict() for event in writer.last_events]
            if name == "v1":
                state = snapshot.relations["bowl_1"]
                logged_state, logged_events = trace[step]
                require(all(getattr(state, key) == logged_state[key] for key in RELATION_KEYS)
                        and new_events == logged_events, episode, f"V1 replay diverged at {step}")
            events[name].extend(event for event in new_events if event["type"] == "return")
            if step in queries:
                decisions[name].append(evaluate_halt(snapshot, observation, vision_halt))
        event = cue.step(step, gripper_aperture(observation), measured[:3])
        if event is not None and event["available_at"] < reference["frames"]:
            events["proprio"].append(event)
        if step in queries:
            count = sum(event["available_at"] <= step for event in events["proprio"])
            snapshot = SimpleNamespace(as_of_observation=step, profile=SimpleNamespace(count_cycles=True),
                                       relations={"proprio": SimpleNamespace(
                                           returns=count, status=None, reference=None, armed=False)})
            decisions["proprio"].append(evaluate_halt(snapshot, observation, cue_halt))
    for name, detector in detectors.items():
        require(detector.steps == list(range(reference["frames"])), episode, f"{name} did not consume all frames")
    return dict(reference=reference, camera_empty_frames={
                    camera: sum(not ev.views[camera] for ev in evidence) for camera in evidence[0].views},
                stats_v4=writers["v4"].stats, methods={
                    name: dict(events=events[name], halt=classify_halt(decisions[name], reference, eval_frames),
                               matching={tol: match_events(sorted(e["available_at"] for e in events[name]),
                                                           reference["milestones"], tol) for tol in TOLERANCES})
                    for name in names})


def summarize(episodes):
    references = [ep["reference"] for ep in episodes]
    count = sum(len(ref["milestones"]) for ref in references)
    groups = {}
    for name in (*WRITERS, "proprio"):
        methods = [ep["methods"][name] for ep in episodes]
        metrics = {}
        for tol in TOLERANCES:
            totals = {key: sum(row["matching"][tol][key] for row in methods)
                      for key in ("true_positive", "false_positive", "false_negative")}
            tp, fp = totals["true_positive"], totals["false_positive"]
            metrics[tol] = dict(**totals, recall=tp / count if count else None,
                                precision=tp / (tp + fp) if tp + fp else None)
        halts = Counter(row["halt"]["category"] for row in methods)
        timely = [row["halt"] for row in methods if row["halt"]["category"] == "timely"]
        groups[name] = dict(matching=metrics, halt=dict(halts),
                            completed_without_halt=sum(ref["completion"] is not None
                                                       and row["halt"]["category"] == "no_halt"
                                                       for ref, row in zip(references, methods, strict=True)),
                            timely_final_predicate=sum(row["final_predicate_at_halt"] for row in timely),
                            timely_margins=[row["margin_before_overshoot"] for row in timely],
                            timely_delays=[row["delay_from_completion"] for row in timely])
    return dict(episodes=len(episodes), frames=sum(ref["frames"] for ref in references),
                query_boundaries=sum(len(ref["query_steps"]) for ref in references),
                apertures_cross_checked=sum(ref["apertures_checked"] for ref in references),
                milestones=count, complete=sum(ref["completion"] is not None for ref in references),
                overshot=sum(ref["overshoot"] is not None for ref in references), methods=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.run / "run_manifest.json"
    manifest = read(manifest_path)
    directories = sorted(path for path in (args.run / "episodes").iterdir() if path.is_dir())
    if (manifest["status"] != "complete" or manifest["completed_episodes"] != len(directories)
            or manifest["memory_contract"]["writer"] != "rgb_relations_v1"
            or manifest["arguments"]["arm"] != "F1" or manifest["arguments"]["memory"] != "on"):
        raise ValueError("Expected a complete X-VLA F1 memory-on run recorded with V1")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    specs, cue_config = writer_specs(), ProprioCueConfig.load()
    cadence = manifest["execution"]["execution_horizon"]
    episodes = []
    for directory in directories:
        episodes.append(replay(directory, cadence, specs, cue_config))
        print(directory.name, episodes[-1]["reference"]["milestones"], flush=True)
    roster = Counter((ep["reference"]["task"], ep["reference"]["state_idx"]) for ep in episodes)
    if roster != Counter((task, state) for task in TARGETS for state in range(20)):
        raise ValueError("Expected the full 20 unseen states for each of T1 and T3")
    source_paths = ("scripts/replay_xvla_e2_writers.py", "scripts/compare_memory_writers.py",
                    "scripts/audit_libero_mem_writer.py", "scripts/extract_writer_replay_inputs.py",
                    "src/evaluation/memory_halt.py", "src/memory/proprio_cycle_cue.py",
                    "configs/memory/proprio_cycle_cue_v1.json")
    report = dict(
        classification="XVLA_E2_FIXED_TRACE_NON_COMPARABLE",
        source=dict(run=str(args.run), manifest_sha256=sha256_file(manifest_path),
                    git_revision=manifest["git_revision"], model=manifest["model"],
                    adapter_export=manifest["adapter_export"], environments=manifest["environments"]),
        replay=dict(git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                    source_sha256={**v4_source_hashes(ROOT),
                                   **{name: sha256_file(ROOT / name) for name in source_paths}},
                    v1_parity_frames=sum(ep["reference"]["frames"] for ep in episodes),
                    tolerance_steps=list(TOLERANCES), query_cadence=cadence, cue_config=cue_config.to_dict(),
                    vision_halt={task: HaltConfig(target).to_dict() for task, target in TARGETS.items()},
                    cue_halt={task: HaltConfig(target, require_on_support=False).to_dict()
                              for task, target in TARGETS.items()}),
        overall=summarize(episodes),
        by_task={task: summarize([ep for ep in episodes if ep["reference"]["task"] == task]) for task in TARGETS})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary.json", report), ("episodes.json", episodes)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
