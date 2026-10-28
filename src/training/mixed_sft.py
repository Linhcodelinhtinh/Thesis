"""Preparation/validation contract for matched base + LIBERO-Mem training views."""
from __future__ import annotations
import json
from pathlib import Path
from src.evaluation.libero_mem_protocol import source_digest
import h5py
import numpy as np
from src.training.split_lock import load_split_lock, locked_episode, content_digest
from src.utils.file_hash import sha256_file
from src.memory.visual_pipeline import TOKENIZER_REVISION, TOKENIZER_SHA256

ARTIFACT = "mixed_base_mem_sft_shards"


def canonical_raw_rgb(images, domain):
    """Undo author Mem vertical flip; the unchanged policy processor handles raw RGB."""
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("Expected uint8 NHWC RGB")
    if domain not in ("base", "mem"):
        raise ValueError("Unknown image source convention")
    return np.ascontiguousarray(images[:, ::-1] if domain == "mem" else images)


def sample_identity(episode, frame):
    return f'{episode["domain"]}/{episode["task"]}/{episode["demo"]}/{frame}'


def validate_mixed_dataset(root, *, verify_shards=True):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact") != ARTIFACT or manifest.get("status") != "complete":
        raise ValueError("Expected a completed mixed SFT shard dataset, not a prompt index")
    split = load_split_lock(root / "split.lock.json")
    if split["lock_sha256"] != manifest["split_lock_sha256"]:
        raise ValueError("Dataset split lock changed")
    if source_digest(root / "protocol.json") != split["protocol_sha256"]:
        raise ValueError("Dataset evaluator protocol differs from split lock")
    if manifest["sources"] != split["sources"]:
        raise ValueError("Dataset source provenance differs from split lock")
    if manifest.get("schema_version") != 2 or manifest.get("max_tokens") != 48:
        raise ValueError("Unknown mixed dataset/token contract")
    if manifest.get("memory_source") not in ("none", "predicted_rgb_v3"):
        raise ValueError("Unknown memory source")
    if (manifest.get("tokenizer_revision") != TOKENIZER_REVISION or
        manifest.get("tokenizer_files_sha256") != {"tokenizer.json": TOKENIZER_SHA256}):
        raise ValueError("Unexpected policy tokenizer provenance")
    if manifest["memory_source"] == "predicted_rgb_v3":
        provenance = manifest.get("memory_provenance", [])
        if not provenance or any(p.get("writer") != "rgb_cycle_v3" or
            any(not p.get(k) for k in ("writer_code_sha256", "writer_config", "detector", "manifest_sha256"))
            for p in provenance):
            raise ValueError("Missing predicted-memory writer/detector provenance")
    seen, covered, projection = set(), set(), []
    for e in manifest["episodes"]:
        row = locked_episode(split, e["domain"], e["task"], e["demo"])
        for key in ("split", "initial_state_sha256", "source"):
            if row[key] != e[key]:
                raise ValueError("Prepared episode differs from split lock: " + key)
        identity = (e["domain"], e["task"], e["demo"])
        if identity in seen or e["length"] != row["frames"] or e["length"] < 1:
            raise ValueError("Duplicate episode or altered frame selection")
        seen.add(identity)
        covered.add((e["domain"], e["split"]))
        if e["domain"] == "base" and e["memory_frames"]:
            raise ValueError("Base replay must retain original instructions")
        if not 0 <= e["memory_frames"] <= e["length"]:
            raise ValueError("Invalid memory count")
        if manifest["memory_source"] == "none" and e["memory_frames"]:
            raise ValueError("Memory frames without memory provenance")
        if e["shard"] != "shards/" + e["domain"] + "__" + e["task"] + ".hdf5":
            raise ValueError("Unexpected task/shard mapping")
        projection.append({k: e[k] for k in ("domain", "task", "demo", "split", "length", "initial_state_sha256", "source")})
    if seen != {(e["domain"], e["task"], e["demo"]) for e in split["episodes"]}:
        raise ValueError("Prepared dataset omitted episodes from its locked selection")
    if covered != {(d, s) for d in ("base", "mem") for s in ("train", "val", "test")}:
        raise ValueError("Both domains need nonempty train/validation/test splits")
    if content_digest(projection) != manifest["sample_selection_sha256"]:
        raise ValueError("Sample selection/order changed")
    if set(manifest["shards"]) != {e["shard"] for e in manifest["episodes"]}:
        raise ValueError("Shard inventory differs from episodes")
    counts = {d: {s: sum(e["length"] for e in manifest["episodes"] if e["domain"] == d and e["split"] == s)
                  for s in ("train", "val", "test")} for d in ("base", "mem")}
    if counts != manifest["counts"]:
        raise ValueError("Sample counts differ from episodes")
    for name, expected in manifest["shards"].items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Shard path escapes dataset")
        if verify_shards and sha256_file(path) != expected:
            raise ValueError("Corrupt prepared shard: " + name)
    if sha256_file(root / "samples.jsonl") != manifest["audit_sha256"]:
        raise ValueError("Sample lineage audit changed")
    return manifest


def verify_shard_contract(path, episodes, tokenizer, max_tokens=48):
    """Check content once during preparation, outside the Kaggle training loop."""
    with h5py.File(path, "r") as f:
        if set(f) != {e["demo"] for e in episodes}:
            raise ValueError("Shard episode list differs from manifest")
        for e in episodes:
            g = f[e["demo"]]
            n = e["length"]
            if g["actions"].shape != (n, 7) or g["robot_states"].shape != (n, 9):
                raise ValueError("State/action dimension mismatch")
            for name in ("actions", "robot_states"):
                if not np.isfinite(g[name][:]).all():
                    raise ValueError("Non-finite " + name)
            if np.abs(g["actions"][:]).max() > 1.00001:
                raise ValueError("Invalid native action range")
            if not np.allclose(np.linalg.norm(g["robot_states"][:, 5:9], axis=1), 1, atol=1e-4):
                raise ValueError("Invalid source quaternion")
            for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                if g[camera].shape[0] != n or g[camera].shape[-1] != 3 or g[camera].dtype != np.uint8:
                    raise ValueError("Invalid RGB shard")
            if g.attrs["split"] != e["split"]:
                raise ValueError("Shard split mismatch")
            prompts = g["prompts"].asstr()[:]
            flags = g["has_memory"][:]
            if len(prompts) != n or len(flags) != n or int(flags.sum()) != e["memory_frames"]:
                raise ValueError("Memory count/length mismatch")
            for prompt, has_memory in zip(prompts, flags):
                if (prompt != g.attrs["instruction"]) != bool(has_memory):
                    raise ValueError("Prompt/flag mismatch")
                text = prompt if prompt.endswith("\n") else prompt + "\n"
                encoded = tokenizer.encode(text, add_special_tokens=True)
                if len(encoded.ids if hasattr(encoded, "ids") else encoded) > max_tokens:
                    raise ValueError("Prompt exceeds policy token budget")
