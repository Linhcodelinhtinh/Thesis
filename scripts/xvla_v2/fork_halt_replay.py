"""Fork a recorded X-VLA E2 rollout at its halt step and simulate the hold for real.

Until the halt latches, a halted rollout and the recorded no-halt rollout are the same trajectory: the halt decision
reads only earlier frames, so the policy produced the same actions. For every (episode, condition) in `plan.json`
(from plan_halt_forks.py) this script resets the real simulator to the recorded state, replays the recorded actions
up to the halt step (no policy, no detector), then runs exactly the hold chunks of `MemoryHaltPolicy`
(`xvla_hold_chunk_factory`, ramp 10, gripper open, optional retract) until the simulator reports `done`.
Conditions with equal (episode, halt step, retract) share one simulation. Conditions that never halt reuse the
recorded outcome. Run it in the isolated simulator environment's orchestrator Python, one process per shard.

Fidelity: while replaying, the measured end-effector position is compared with the recorded one. `--verify` replays a
whole recorded episode without any halt and reports the deviation and whether the final evaluator result matches.
"""
import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXECUTION_HORIZON = 10
MAX_STEPS = 1000
DEFAULT_JOBS = ("H1_oracle", "H2_v4", "H3_v4_cue", "H3g_v4_cue", "H3c2", "CUE_only", "H3_v4_cue:0.03")


def parse_jobs(text):
    jobs = []
    for item in text:
        name, _, retract = item.partition(":")
        jobs.append((name, float(retract) if retract else 0.0))
    return jobs


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_recording(directory):
    directory = Path(directory)
    actions = [np.asarray(row["action"], dtype=float) for row in read_rows(directory / "actions.jsonl")]
    eef = np.asarray([row["raw_proprioception_pos_quatxyzw_gripper"][:3]
                      for row in read_rows(directory / "observations.jsonl")], dtype=float)
    result = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["result"]
    return actions, eef, result


def outcome(info, step, deviation):
    ev = info["eval_result"]
    return dict(success=bool(ev.full_success), scr=float(ev.scr), overshot=bool(ev.overshot),
                confirmed_cycles=int(ev.confirmed_cycles), steps=int(step),
                final_predicate=bool(info.get("final_predicate")), max_eef_deviation_m=float(deviation))


def fork_episode(env, state_idx, actions, eef, halt_step, retract_m=0.0, hold_chunk=None, max_steps=MAX_STEPS,
                 horizon=EXECUTION_HORIZON, ramp_steps=10):
    """Replay `actions[:halt_step]`, then hold. `halt_step=None` replays everything (fidelity check)."""
    obs = env.reset(state_idx)
    deviation = at_halt = float(np.abs(np.asarray(obs["robot0_eef_pos"], dtype=float) - eef[0]).max())
    info, done, step = dict(env.last_info), False, 0
    stop = len(actions) if halt_step is None else halt_step
    while step < stop:
        obs, _, done, info = env.step(actions[step])
        step += 1
        at_halt = float(np.abs(np.asarray(obs["robot0_eef_pos"], dtype=float) - eef[step]).max())
        deviation = max(deviation, at_halt)
        if done:
            break
    if halt_step is None or done:
        return dict(outcome(info, step, deviation), forked=False, drift_at_halt_m=at_halt,
                    ended_before_halt=bool(done and halt_step is not None))
    current_pose, make_hold = hold_chunk
    position, rotation6d = current_pose()
    anchor = position.copy()
    anchor[2] += retract_m
    while step < max_steps and not done:
        position, _ = current_pose()
        chunk = np.asarray(make_hold(anchor, rotation6d, position, ramp_steps), dtype=np.float32)
        for action in chunk[:horizon]:
            if step >= max_steps or done:
                break
            _, _, done, info = env.step(action)
            step += 1
    return dict(outcome(info, step, deviation), forked=True, ended_before_halt=False,
                drift_at_halt_m=at_halt)


def work_items(plan, jobs, tasks, shard, shards, skip_premature=False):
    """(episode, halt_step, retract) -> conditions, restricted to this shard; never-halting conditions are skipped."""
    items = {}
    for episode in plan["episodes"]:
        ref = episode["reference"]
        if ref["task"] not in tasks:
            continue
        for name, retract in jobs:
            halt_step = episode["conditions"][name]["halt_step"]
            if skip_premature and episode["conditions"][name].get("category") == "premature":
                continue    # a halt before the completion milestone cannot produce a success
            if halt_step is not None:
                items.setdefault((ref["episode"], halt_step, retract), []).append(name)
    keys = sorted(items)
    return {key: items[key] for index, key in enumerate(keys) if index % shards == shard}


def run_shard(args, plan, jobs):
    from src.training.split_lock import DEFAULT_AUTHOR_MEM_LOCK
    from src.evaluation.memory_halt import xvla_hold_chunk_factory
    from src.simulator.xvla_mem_env import ProcessXVLAMemEnv

    out = args.output_dir / f"results-shard{args.shard}.jsonl"
    done_keys = set()
    if out.exists():
        done_keys = {(r["episode"], r["halt_step"], r["retract_m"]) for r in read_rows(out)}
    by_episode = {e["reference"]["episode"]: e["reference"] for e in plan["episodes"]}
    items = work_items(plan, jobs, args.tasks, args.shard, args.shards, args.assume_premature_fails)
    if args.verify:
        chosen = sorted(e["reference"]["episode"] for e in plan["episodes"]
                        if e["reference"]["task"] in args.tasks and e["reference"]["state_idx"] in args.verify_states)
        items = {(name, None, 0.0): ["verify"] for index, name in enumerate(chosen) if index % args.shards == args.shard}
    envs = {}
    try:
        for (episode, halt_step, retract), names in items.items():
            if (episode, halt_step, retract) in done_keys:
                continue
            ref = by_episode[episode]
            task = ref["task"]
            if task not in envs:
                envs[task] = ProcessXVLAMemEnv(task, split_lock=DEFAULT_AUTHOR_MEM_LOCK, split="unseen",
                                               python_executable=args.sim_python, timeout_seconds=args.timeout)
                if envs[task].target_cycles != ref["target"]:
                    raise ValueError("Simulator target cycles differ from the plan")
            env = envs[task]
            actions, eef, recorded = load_recording(args.run / "episodes" / episode)
            result = fork_episode(env, ref["state_idx"], actions, eef, halt_step, retract,
                                  hold_chunk=xvla_hold_chunk_factory(env, 30))
            row = dict(episode=episode, task=task, state_idx=ref["state_idx"], halt_step=halt_step,
                       retract_m=retract, conditions=names, recorded_success=bool(recorded["success"]),
                       recorded_steps=recorded["total_steps"], recorded_overshot=bool(recorded["overshot"]),
                       recorded_confirmed_cycles=recorded["confirmed_cycles"], **result)
            if halt_step is None:
                row["matches_recorded"] = (row["success"] == row["recorded_success"] and row["steps"] == row["recorded_steps"]
                                           and row["overshot"] == row["recorded_overshot"]
                                           and row["confirmed_cycles"] == row["recorded_confirmed_cycles"])
            with out.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(episode, halt_step, retract, {k: result[k] for k in ("success", "overshot", "steps")}, flush=True)
    finally:
        for env in envs.values():
            env.close()


def summarize(args, plan, jobs):
    limit = args.drift_limit
    rows = [r for path in sorted(args.output_dir.glob("results-shard*.jsonl")) for r in read_rows(path)]
    index = {(r["episode"], r["halt_step"], r["retract_m"]): r for r in rows}
    recorded = {}
    for episode in plan["episodes"]:
        ref = episode["reference"]
        _, _, res = load_recording(args.run / "episodes" / ref["episode"])
        recorded[ref["episode"]] = res
    summary = dict(H0_no_halt=dict(n=len(recorded), success=sum(bool(r["success"]) for r in recorded.values()),
                                   overshot=sum(bool(r["overshot"]) for r in recorded.values())))
    for name, retract in jobs:
        label = name if retract == 0.0 else f"{name}@retract{retract:g}"
        per_task, missing, wins, losses = {}, [], 0, 0
        drifted, faithful_success, faithful_n = [], 0, 0
        counts = Counter()
        for episode in plan["episodes"]:
            ref, cond = episode["reference"], episode["conditions"][name]
            if cond["halt_step"] is None:
                success, overshot = bool(recorded[ref["episode"]]["success"]), bool(recorded[ref["episode"]]["overshot"])
            else:
                row = index.get((ref["episode"], cond["halt_step"], retract))
                if row is None and args.assume_premature_fails and cond["category"] == "premature":
                    row = dict(success=False, overshot=False, drift_at_halt_m=0.0)
                if row is None:
                    missing.append(ref["episode"])
                    continue
                success, overshot = row["success"], row["overshot"]
                if row["drift_at_halt_m"] > limit:
                    drifted.append(ref["episode"])
                else:
                    faithful_n += 1
                    faithful_success += success
            base = bool(recorded[ref["episode"]]["success"])
            wins += success and not base
            losses += base and not success
            counts[(ref["task"], "n")] += 1
            counts[(ref["task"], "success")] += success
            counts[(ref["task"], "overshot")] += overshot
            counts[(ref["task"], cond["category"])] += 1
        summary[label] = dict(by_task={t: {k: counts[(t, k)] for k in ("n", "success", "overshot", "timely", "premature", "no_halt")}
                                       for t in ("T1", "T3")},
                              success=sum(counts[(t, "success")] for t in ("T1", "T3")),
                              pairs_won_vs_H0=wins, pairs_lost_vs_H0=losses,
                              exact_p_one_sided_vs_H0=(sum(comb(wins + losses, i) for i in range(wins, wins + losses + 1))
                                                       / 2 ** (wins + losses) if wins + losses else 1.0),
                              drifted_at_halt=drifted, faithful_simulations=faithful_n,
                              faithful_success=faithful_success, missing=missing)
    fidelity = [r["max_eef_deviation_m"] for r in index.values()]
    summary["fidelity"] = dict(max_eef_deviation_m=max(fidelity) if fidelity else None, simulations=len(index),
                               ended_before_halt=[r["episode"] for r in index.values() if r["ended_before_halt"]])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if k in ("H0_no_halt", "fidelity") else {"success": v["success"], "by_task": v["by_task"]})
                      for k, v in summary.items()}, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="recorded F1 memory-on run directory")
    p.add_argument("--plan", type=Path, required=True, help="plan.json from plan_halt_forks.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sim-python", help="isolated simulator Python")
    p.add_argument("--jobs", nargs="+", default=list(DEFAULT_JOBS), help="condition[:retract_m] ...")
    p.add_argument("--tasks", nargs="+", default=["T1", "T3"])
    p.add_argument("--workers", type=int, default=1, help="parallel simulator shards (CPU only)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--verify", action="store_true", help="replay whole recorded episodes without a halt")
    p.add_argument("--verify-states", type=int, nargs="+", default=[0, 4, 16, 17])
    p.add_argument("--drift-limit", type=float, default=1e-3,
                   help="summary flags forks whose EEF was further than this (m) from the recording at the halt step")
    p.add_argument("--assume-premature-fails", action="store_true",
                   help="do not simulate halts before the completion milestone; count them as failures (they cannot succeed)")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    jobs = parse_jobs(args.jobs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        return summarize(args, plan, jobs)
    if not args.sim_python:
        p.error("--sim-python is required to simulate")
    if args.workers > 1:
        base = [sys.executable, str(Path(__file__).resolve()), "--run", str(args.run), "--plan", str(args.plan),
                "--output-dir", str(args.output_dir), "--sim-python", args.sim_python, "--shards", str(args.workers),
                "--tasks", *args.tasks, "--jobs", *args.jobs, "--timeout", str(args.timeout)] + (["--assume-premature-fails"] if args.assume_premature_fails else [])
        processes = [subprocess.Popen(base + ["--shard", str(i)] + (["--verify"] if args.verify else []))
                     for i in range(args.workers)]
        codes = [proc.wait() for proc in processes]
        if any(codes):
            raise SystemExit(f"Shard failure: {codes}")
        return summarize(args, plan, jobs) if not args.verify else None
    run_shard(args, plan, jobs)


if __name__ == "__main__":
    main()
