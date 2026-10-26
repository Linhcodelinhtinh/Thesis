#!/usr/bin/env python
"""Download/verify author LIBERO data and produce causal V2 SFT shards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.memory_data import (
    DATA_REPO, DATA_REVISION, LIBERO_REVISION, BACKBONE_REPO, BACKBONE_REVISION,
    annotate_episode, assign_splits, code_hashes, sha256, validate_episode, write_json,
)


def source_inventory(args):
    if args.source_lock:
        import yaml
        lock_path = Path(args.source_lock)
        if not lock_path.is_file() and (ROOT / args.source_lock).is_file():
            lock_path = ROOT / args.source_lock
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        if lock["benchmark_provenance"]["pinned_commit"] != DATA_REVISION:
            raise ValueError("Source lock revision differs from audited author dataset")
        files = {f'{r["task_suite"]}/{r["task_name"]}_demo.hdf5': r["sha256"]
                 for r in lock["demonstrations"]}
    else:
        from huggingface_hub import HfApi
        files = {}
        for entry in HfApi().list_repo_tree(DATA_REPO, path_in_repo=args.suite,
                                            repo_type="dataset", revision=DATA_REVISION, recursive=True):
            if entry.path.endswith("_demo.hdf5"):
                lfs = entry.lfs
                if not lfs:
                    raise ValueError(f"No authoritative LFS SHA256 for {entry.path}")
                files[entry.path] = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
    selected = {p: h for p, h in files.items() if p.startswith(args.suite + "/")}
    if args.task_names:
        wanted = {f"{args.suite}/{name}_demo.hdf5" for name in args.task_names}
        if not wanted.issubset(selected):
            raise ValueError(f"Tasks missing from pinned source inventory: {wanted - selected.keys()}")
        selected = {p: h for p, h in selected.items() if p in wanted}
    if not selected:
        raise ValueError("No verified source files selected")
    return selected


def prepare(args):
    import h5py
    import numpy as np
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download, snapshot_download

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prepared data: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir()
    tokenizer_path = args.tokenizer_path or snapshot_download(
        BACKBONE_REPO, revision=BACKBONE_REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer_files = {p.name: sha256(p) for p in Path(tokenizer_path).iterdir()
                       if p.is_file() and p.suffix in {".json", ".txt", ".model", ".jinja"}}
    inventory = source_inventory(args)
    manifest = {
        "schema_version": 1, "classification": "CUSTOM_V2_SFT_NOT_OFFICIAL_LIBERO_SCORE",
        "source_repository": DATA_REPO, "source_revision": DATA_REVISION,
        "libero_revision": LIBERO_REVISION, "writer": "observation_legacy_v1",
        "memory_code_sha256": code_hashes(ROOT), "tokenizer_repository": BACKBONE_REPO,
        "tokenizer_revision": BACKBONE_REVISION, "tokenizer_files_sha256": tokenizer_files,
        "alignment": "source obs[i] is post-action[i]; target starts at action[i+1]",
        "image_contract": "raw opengl RGB; adapter PIL bilinear ->256, official flip H,W ->policy resize512",
        "state_contract": "raw robot_states=[gripper2,eef_pos3,quat_xyzw4]; official processor ->state8",
        "action_contract": "native OSC_POSE delta7, gripper -1=open/+1=close; original checkpoint normalization",
        "max_tokens": 48, "max_context_chars": args.max_context_chars,
        "split_seed": args.seed, "suite": args.suite, "sources": [], "episodes": [], "shards": {},
        "split_note": "Demo/init-state group split, not official evaluation initial-state IDs; base pretraining may overlap.",
    }
    for relative, expected_hash in sorted(inventory.items()):
        source = Path(args.source_root) / relative
        if not source.is_file():
            if not args.download:
                raise FileNotFoundError(f"{source}; use --download for the pinned official file")
            source = Path(hf_hub_download(DATA_REPO, relative, repo_type="dataset", revision=DATA_REVISION,
                                         local_dir=args.source_root))
        if sha256(source) != expected_hash:
            raise ValueError(f"Corrupt or unverified source: {source}")
        task = source.name.removesuffix("_demo.hdf5")
        shard_rel = f"shards/{task}.hdf5"
        with h5py.File(source, "r") as original, h5py.File(out / shard_rel, "w") as shard:
            data = original["data"]
            info = json.loads(data.attrs["problem_info"])
            instruction = info["language_instruction"]
            env = json.loads(data.attrs["env_args"])["env_kwargs"]
            if (data.attrs.get("macros_image_convention") != "opengl" or
                env.get("control_freq") != 20 or env.get("controller_configs", {}).get("type") != "OSC_POSE" or
                env.get("controller_configs", {}).get("control_delta") is not True or env.get("robots") != ["Panda"]):
                raise ValueError(f"Source contract differs from audited Panda/OSC_POSE/opengl/20Hz: {source}")
            episodes = sorted(k for k in data if k.startswith("demo_"))
            groups = {name: hashlib.sha256(np.asarray(data[name].attrs["init_state"], dtype="<f8").tobytes()).hexdigest()
                      for name in episodes}
            split = assign_splits(groups, args.seed, args.val_fraction, args.test_fraction)
            task_counts = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
            for name in episodes:
                demo = data[name]
                actions, robot, resolutions = validate_episode(demo)
                records = annotate_episode(instruction, robot, actions, tokenizer, f"{task}/{name}",
                                           max_chars=args.max_context_chars)
                n = len(records)
                target = shard.create_group(name)
                # Preserve original observations exactly; transform only when loading for policy.
                for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                    shape = (n,) + demo[f"obs/{camera}"].shape[1:]
                    ds = target.create_dataset(camera, shape=shape, dtype="uint8", chunks=(1,) + shape[1:], compression="lzf")
                    for start in range(0, n, 64):
                        ds[start:start + 64] = demo[f"obs/{camera}"][start:min(start + 64, n)]
                target.create_dataset("robot_states", data=robot[:-1])
                target.create_dataset("actions", data=actions[1:])
                target.create_dataset("prompts", data=[r["prompt"] for r in records], dtype=h5py.string_dtype("utf-8"))
                target.create_dataset("has_memory", data=[r["has_memory"] for r in records])
                target.attrs["instruction"] = instruction
                target.attrs["split"] = split[name]
                with (out / "samples.jsonl").open("a", encoding="utf-8") as audit:
                    for record in records:
                        audit.write(json.dumps({"task": task, "demo": name, "split": split[name], **record}) + "\n")
                count = sum(r["has_memory"] for r in records)
                task_counts[split[name]][0] += n
                task_counts[split[name]][1] += count
                manifest["episodes"].append({"task": task, "demo": name, "shard": shard_rel, "length": n,
                                             "split": split[name], "initial_state_sha256": groups[name],
                                             "memory_frames": count, "source_resolution": resolutions})
            manifest["sources"].append({"path": relative, "sha256": expected_hash, "instruction": instruction,
                                        "frame_memory_counts_by_split": task_counts})
        manifest["shards"][shard_rel] = sha256(out / shard_rel)
        print(json.dumps({"prepared": task, "frames_and_memory_frames": task_counts}), flush=True)
    manifest["audit_sha256"] = sha256(out / "samples.jsonl")
    write_json(out / "manifest.json", manifest)  # completion marker, written only after all tasks succeed
    total = sum(e["length"] for e in manifest["episodes"])
    memory = sum(e["memory_frames"] for e in manifest["episodes"])
    print(f"Prepared {total} samples; {memory} with memory ({memory / total:.1%}).")
    if not memory:
        print("WARNING: current writer rendered NO memory. Memory training will fail; no-memory control remains valid.")


def main():
    print("NOTE: this base-LIBERO converter uses the legacy proprioception writer. "
          "For the new T1/T3 RGB writer use prepare_libero_mem_sft.py; legacy memory is not validated training evidence.")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--suite", default="libero_10", choices=["libero_10", "libero_object"])
    p.add_argument("--task-names", nargs="+")
    p.add_argument("--source-lock", help="Offline author-resource YAML manifest with SHA256 inventory")
    p.add_argument("--download", action="store_true")
    p.add_argument("--tokenizer-path", help="Local snapshot of the pinned official tokenizer")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--max-context-chars", type=int, default=512)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
