"""Replay V4 + proprio on verified V2 training demonstrations using cached RGB evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.compare_memory_writers import RecordedDetector, development_jobs, parse_evidence, writer_specs
from scripts.extract_writer_replay_inputs import sha256_file
from src.memory.cycle_fusion import CycleFusionCounter
from src.perception.visual_evidence import image_digest


def build(root, inputs, indexes, tokenizer, output):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roster = {f"{e['task']}_{e['demo']}": e for e in manifest["episodes"]
              if e["domain"] == "mem" and e["split"] == "train"}
    shard_names = sorted({e["shard"] for e in roster.values()})
    for name in shard_names:
        if sha256_file(root / name) != manifest["shards"][name]:
            raise ValueError(f"Shard hash mismatch: {name}")
    audit = {}
    for index in indexes:
        index_manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
        if index_manifest["split_lock_sha256"] != manifest["split_lock_sha256"]:
            raise ValueError(f"Split lock mismatch: {index}")
        path = index / "milestone-audit.json"
        audit.update({e["episode"]: e["milestones"]
                      for e in json.loads(path.read_text(encoding="utf-8"))["episodes"]})
    tok = Tokenizer.from_file(str(tokenizer))
    build_writer = writer_specs()["v4"][-1]
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.h5")
    if output.exists():
        raise FileExistsError(output)
    seen = set()
    totals = Counter()
    with h5py.File(partial, "x") as dst:
        for job in development_jobs(indexes, inputs, ("v4",)):
            if len(job) == 2:  # terminal provenance record
                dst.attrs["cached_detection_sources"] = json.dumps(job[1])
                continue
            episode, task, payloads, grip, _ = job
            entry = roster[episode]
            if episode in seen:
                raise ValueError(f"Duplicate episode: {episode}")
            seen.add(episode)
            evidence = parse_evidence(payloads)
            writer = build_writer(episode, RecordedDetector(dict(enumerate(evidence))))
            fusion = CycleFusionCounter()
            target = int(task[1:])
            milestones = audit[episode]
            with h5py.File(root / entry["shard"], "r") as source:
                g = source[entry["demo"]]
                n = entry["length"]
                states, actions = g["robot_states"][()], g["actions"][()]
                if len(evidence) != n or not np.array_equal(states[:, :2], np.asarray(grip, dtype=states.dtype)):
                    raise ValueError(f"Proprioception/frame mismatch: {episode}")
                instruction = g.attrs["instruction"]
                if isinstance(instruction, bytes):
                    instruction = instruction.decode()
                av, wr = g["agentview_rgb"][()], g["eye_in_hand_rgb"][()]
                prompts, counts = [], []
                for step, (state, ev) in enumerate(zip(states, evidence, strict=True)):
                    obs = dict(agentview_image=av[step, ::-1], robot0_eye_in_hand_image=wr[step, ::-1],
                               robot0_gripper_qpos=state[:2])
                    for camera, digest in ev.image_hashes.items():
                        if image_digest(obs[camera]) != digest:
                            raise ValueError(f"RGB evidence hash mismatch: {episode}/{step}/{camera}")
                    writer.step(step, obs, actions[step - 1] if step else None)
                    returns = [event.to_dict() for event in writer.last_events if event.type == "return"]
                    count = fusion.step(step, np.abs(state[:2]).sum(), state[2:5], returns)
                    counts.append(count)
                    prompts.append(instruction + f" Memory: bowl: {count} observed returns.")
            if max(len(tok.encode(prompt).ids) for prompt in set(prompts)) > 50:
                raise ValueError(f"X-VLA token limit exceeded: {episode}")
            reference = np.searchsorted(milestones, np.arange(n), side="right")
            eligible = (reference == counts) & (len(milestones) == target)
            group = dst.create_group(f"{entry['shard']}/{entry['demo']}")
            group.create_dataset("prompts", data=prompts, dtype=h5py.string_dtype())
            group.create_dataset("eligible", data=eligible)
            group.create_dataset("writer_returns", data=counts)
            group.create_dataset("reference_returns", data=reference)
            group.attrs["target"] = target
            shown = slice(1, max(1, n - 9))
            totals[task + "_episodes"] += 1
            totals[task + "_complete"] += len(milestones) == target
            totals[task + "_eligible_candidates"] += int(eligible[shown].sum())
            totals[task + "_eligible_at_target_before_static_filter"] += int(
                (eligible[shown] & (reference[shown] == target)).sum())
        if seen != roster.keys():
            raise ValueError("Replay and training episode rosters differ")
        dst.attrs["manifest_sha256"] = sha256_file(manifest_path)
        dst.attrs["tokenizer_sha256"] = sha256_file(tokenizer)
        dst.attrs["milestone_audit_sha256"] = json.dumps({
            index.name: sha256_file(index / "milestone-audit.json") for index in indexes})
        dst.attrs["source_sha256"] = json.dumps({
            str(p.relative_to(ROOT)): sha256_file(p) for p in (
                Path(__file__).resolve(),
                *(ROOT / "src/memory").glob("*.py"),
                *(ROOT / "configs/memory").glob("*.json"),
                ROOT / "src/perception/relation_cycle_tracking.py",
                ROOT / "scripts/compare_memory_writers.py",
            )})
        dst.attrs["totals"] = json.dumps(totals)
    partial.rename(output)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--indexes", type=Path, nargs="+", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.shards_root, args.inputs, args.indexes, args.tokenizer, args.out)
