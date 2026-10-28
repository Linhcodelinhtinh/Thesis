#!/usr/bin/env python
"""Prepare complete, matched base/Mem shards using an immutable initial-state split.

Base uses original instructions. Mem memory is optional, but if supplied must
cover EVERY locked episode and match the pinned sources and writer version.
No perception/model/simulator dependencies are needed to prepare F0 data.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.split_lock import load_split_lock, content_digest, state_digest
from src.training.mixed_sft import ARTIFACT, canonical_raw_rgb, verify_shard_contract, validate_mixed_dataset
from src.training.memory_data import validate_episode
from src.training.libero_mem_hdf5_loader import read_episode_metadata
from src.memory.visual_pipeline import load_prompt_tokenizer, TOKENIZER_REVISION, TOKENIZER_SHA256
from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, source_digest
from src.utils.file_hash import sha256_file


def load_memory_indices(paths, lock):
    samples, provenance = {}, []
    for root in paths:
        m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if m.get("status") != "complete" or m.get("writer") != "rgb_cycle_v3" or m.get("stream") != "predicted_memory":
            raise ValueError("Memory must come from a completed RGB v3 predicted index")
        if m.get("split_lock_sha256") != lock["lock_sha256"]:
            raise ValueError("Generate memory indices using the same --split-lock first")
        if (m.get("tokenizer_revision") != TOKENIZER_REVISION or m.get("max_total_tokens") != 48 or
            m.get("obs_action_offset") != 0 or m.get("image_convention") != "libero_mem_refined"):
            raise ValueError("Memory tokenizer/alignment/image contract differs")
        for name, digest in m["artifacts_sha256"].items():
            if sha256_file(root / name) != digest:
                raise ValueError("Corrupt memory index artifact: " + name)
        source_hashes = {e["path"]: e["sha256"] for e in lock["sources"] if e["domain"] == "mem"}
        if (len(m["sources"]) != 2 or "metainfo.json" not in m["sources"] or
            any(source_hashes.get(name) != digest for name, digest in m["sources"].items())):
            raise ValueError("Memory/source data mismatch")
        for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            key = (record["episode_id"], record["step"])
            if key in samples:
                raise ValueError("Duplicate memory sample across indices")
            samples[key] = record
        provenance.append(dict(manifest_sha256=sha256_file(root / "manifest.json"),
            writer=m["writer"], writer_code_sha256=m["writer_code_sha256"],
            writer_config=m["writer_config"], detector=m["detector"], tokenizer_revision=m["tokenizer_revision"]))
    for field in ("writer_code_sha256", "writer_config", "detector", "tokenizer_revision"):
        if provenance and any(p[field] != provenance[0][field] for p in provenance):
            raise ValueError("Memory indices differ in " + field)
    return samples, provenance


def prepare(args):
    import h5py
    import numpy as np
    lock = load_split_lock(args.split_lock)
    if lock["protocol_sha256"] != source_digest(DEFAULT_PROTOCOL):
        raise ValueError("Split lock was generated for a different evaluator protocol")
    roots = dict(base=args.base_root, mem=args.mem_root)
    for s in lock["sources"]:
        if sha256_file(roots[s["domain"]] / s["path"]) != s["sha256"]:
            raise ValueError("Source changed since split locking: " + s["path"])
    tokenizer = load_prompt_tokenizer(args.tokenizer_path)
    memory, memory_provenance = load_memory_indices(args.memory_index, lock)
    if memory_provenance:
        expected = {(e["task"] + "_" + e["demo"], t) for e in lock["episodes"] if e["domain"] == "mem" for t in range(e["frames"])}
        if set(memory) != expected:
            raise ValueError(f"Memory indices must cover the exact locked Mem frames: missing={len(expected-set(memory))}, extra={len(set(memory)-expected)}")
    out = args.output_dir
    if out.exists():
        raise FileExistsError("Use a fresh output directory; completed data is immutable")
    (out / "shards").mkdir(parents=True)
    shutil.copy2(args.split_lock, out / "split.lock.json")
    shutil.copy2(DEFAULT_PROTOCOL, out / "protocol.json")
    manifest = dict(schema_version=2, artifact=ARTIFACT, status="preparing",
        classification="CUSTOM_V2_SFT_NOT_OFFICIAL_LIBERO_SCORE", max_tokens=48,
        split_lock_sha256=lock["lock_sha256"], sources=lock["sources"],
        memory_source="predicted_rgb_v3" if memory_provenance else "none",
        memory_provenance=memory_provenance, tokenizer_revision=TOKENIZER_REVISION,
        tokenizer_files_sha256={"tokenizer.json": TOKENIZER_SHA256},
        image_contract="canonical raw OpenGL RGB; Mem refined vertical flip undone once before unchanged policy preprocessing",
        alignment=dict(base="obs[i] -> action[i+1]; final source observation excluded", mem="obs[i] -> action[i]"),
        state_contract="gripper2, eef_pos3, quaternion_xyzw4 -> existing official processor state8",
        action_contract="native OSC_POSE action7, unchanged checkpoint normalizers, episode-bounded padding",
        retention_scope=sorted({e["suite"] for e in lock["episodes"] if e["domain"] == "base"}),
        retention_replay_tasks=sorted({e["task"] for e in lock["episodes"] if e["domain"] == "base"}),
        preparation_code_sha256={name: sha256_file(ROOT / name) for name in
            ("scripts/prepare_mixed_memory_sft.py", "src/training/mixed_sft.py", "src/training/split_lock.py")},
        episodes=[], shards={})
    groups = defaultdict(list)
    for e in lock["episodes"]:
        groups[e["domain"], e["source"]].append(e)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as audit:
        for (domain, source), episodes in sorted(groups.items()):
            task = episodes[0]["task"]
            relative = "shards/" + domain + "__" + task + ".hdf5"
            meta = read_episode_metadata(args.mem_root / "metainfo.json", task, [e["demo"] for e in episodes]) if domain == "mem" else None
            prepared = []
            with h5py.File(roots[domain] / source, "r") as original, h5py.File(out / relative, "w") as shard:
                if domain == "base":
                    attrs = original["data"].attrs
                    env = json.loads(attrs["env_args"])["env_kwargs"]
                    if (attrs.get("macros_image_convention") != "opengl" or env.get("control_freq") != 20 or
                        env.get("controller_configs", {}).get("type") != "OSC_POSE" or
                        env.get("controller_configs", {}).get("control_delta") is not True or env.get("robots") != ["Panda"]):
                        raise ValueError("Unknown base simulator/controller/image contract")
                    instruction_base = json.loads(attrs["problem_info"])["language_instruction"]
                for e in episodes:
                    g = original["data"][e["demo"]]
                    actions, robot, _ = validate_episode(g)
                    offset = 1 if domain == "base" else 0
                    n = len(actions) - offset
                    state = g.attrs["init_state"] if domain == "base" else meta[e["demo"]]["initial_state"]
                    if state_digest(state) != e["initial_state_sha256"] or n != e["frames"]:
                        raise ValueError("Episode no longer matches its split lock")
                    instruction = instruction_base if domain == "base" else meta[e["demo"]]["task_description"]
                    prompts, flags = [], []
                    for t in range(n):
                        record = memory.get((task + "_" + e["demo"], t)) if domain == "mem" else None
                        if record:
                            if (record["instruction"] != instruction or record["split"] != e["split"] or
                                record["source_obs_idx"] != t or record["source_action_start_idx"] != t):
                                raise ValueError("Misaligned memory record")
                            prompt = record["rendered_prompt"]
                        else:
                            prompt = instruction
                        prompts.append(prompt)
                        flags.append(prompt != instruction)
                        audit.write(json.dumps(dict(domain=domain, task=task, demo=e["demo"], split=e["split"],
                            prepared_index=t, source_obs_idx=t, source_action_start_idx=t+offset,
                            last_executed_action_idx=t+offset-1, has_memory=flags[-1])) + "\n")
                    target = shard.create_group(e["demo"])
                    for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                        data = g["obs/" + camera]
                        shape = (n,) + data.shape[1:]
                        ds = target.create_dataset(camera, shape=shape, dtype="uint8", chunks=(1,)+shape[1:], compression="lzf")
                        for begin in range(0, n, 32):
                            ds[begin:begin+32] = canonical_raw_rgb(data[begin:min(begin+32,n)], domain)
                    target.create_dataset("robot_states", data=robot[:n])
                    target.create_dataset("actions", data=actions[offset:])
                    target.create_dataset("prompts", data=prompts, dtype=h5py.string_dtype("utf-8"))
                    target.create_dataset("has_memory", data=flags)
                    target.attrs.update(instruction=instruction, split=e["split"], domain=domain,
                        initial_state_sha256=e["initial_state_sha256"], source_action_offset=offset)
                    row = {k: e[k] for k in ("domain", "task", "demo", "split", "initial_state_sha256", "source")}
                    row.update(length=n, shard=relative, memory_frames=sum(flags))
                    prepared.append(row)
            verify_shard_contract(out / relative, prepared, tokenizer)
            manifest["episodes"].extend(prepared)
            manifest["shards"][relative] = sha256_file(out / relative)
            print(json.dumps(dict(prepared=task, domain=domain, episodes=len(prepared), frames=sum(e["length"] for e in prepared))), flush=True)
    projection = [{k:e[k] for k in ("domain", "task", "demo", "split", "length", "initial_state_sha256", "source")} for e in manifest["episodes"]]
    manifest["sample_selection_sha256"] = content_digest(projection)
    manifest["audit_sha256"] = sha256_file(out / "samples.jsonl")
    manifest["counts"] = {d:{s:sum(e["length"] for e in manifest["episodes"] if e["domain"]==d and e["split"]==s) for s in ("train","val","test")} for d in ("base","mem")}
    manifest["status"] = "complete"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    validate_mixed_dataset(out, verify_shards=False)
    print(json.dumps(manifest["counts"], indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-lock", type=Path, required=True)
    p.add_argument("--base-root", type=Path, default=ROOT/"resources/demonstrations")
    p.add_argument("--mem-root", type=Path, default=ROOT/"data/libero_mem")
    p.add_argument("--tokenizer-path", type=Path, required=True)
    p.add_argument("--memory-index", type=Path, nargs="*", default=[])
    p.add_argument("--output-dir", type=Path, required=True)
    prepare(p.parse_args())


if __name__=="__main__":
    main()
