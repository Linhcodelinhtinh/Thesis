#!/usr/bin/env python
"""Estimate mixed-shard storage from evenly spaced real RGB frames in a locked roster."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.mixed_sft import canonical_raw_rgb
from src.training.split_lock import load_split_lock


def estimate(lock, roots, sample_limit, compression):
    groups = defaultdict(list)
    for episode in lock["episodes"]:
        groups[episode["domain"], episode["source"]].append(episode)
    options = (dict(compression="gzip", compression_opts=1)
               if compression == "gzip" else dict(compression="lzf"))
    sources = {(row["domain"], row["path"]): row for row in lock["sources"]}
    result = []
    for (domain, name), episodes in sorted(groups.items()):
        total_frames = sum(row["frames"] for row in episodes)
        positions = np.linspace(0, total_frames - 1, min(sample_limit, total_frames), dtype=int)
        boundaries = np.cumsum([0, *(row["frames"] for row in episodes)])
        convention = sources[domain, name].get("image_convention",
            "raw_opengl" if domain == "base" else lock.get("mem_image_convention", "libero_mem_refined"))
        buffer = io.BytesIO()
        with h5py.File(roots[domain] / name, "r") as source, h5py.File(buffer, "w") as sample:
            for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                shape = source[f"data/{episodes[0]['demo']}/obs/{camera}"].shape[1:]
                out = sample.create_dataset(camera, shape=(len(positions), *shape),
                                            dtype="uint8", chunks=(1, *shape), **options)
                for i, position in enumerate(positions):
                    episode = int(np.searchsorted(boundaries, position, side="right") - 1)
                    frame = position - boundaries[episode]
                    path = f"data/{episodes[episode]['demo']}/obs/{camera}"
                    out[i] = canonical_raw_rgb(source[path][frame:frame + 1],
                                               source_convention=convention)[0]
            sample.flush()
            bytes_per_frame = sum(sample[camera].id.get_storage_size() for camera in sample) / len(positions)
        result.append(dict(domain=domain, task=episodes[0]["task"], frames=total_frames,
                           sampled_frames=len(positions), estimated_rgb_bytes=round(bytes_per_frame * total_frames)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, default=ROOT / "resources/demonstrations")
    parser.add_argument("--mem-root", type=Path, required=True)
    parser.add_argument("--volume", type=Path, default=ROOT, help="Existing path on intended output volume")
    parser.add_argument("--samples-per-source", type=int, default=48)
    parser.add_argument("--rgb-compression", choices=("gzip", "lzf"), default="gzip")
    args = parser.parse_args()
    if args.samples_per_source < 1:
        parser.error("Sample count must be positive")
    lock = load_split_lock(args.split_lock)
    rows = estimate(lock, {"base": args.base_root, "mem": args.mem_root},
                    args.samples_per_source, args.rgb_compression)
    print(json.dumps(dict(
        split_lock_sha256=lock["lock_sha256"], rgb_compression=args.rgb_compression,
        estimated_rgb_gib=round(sum(row["estimated_rgb_bytes"] for row in rows) / 1024**3, 2),
        free_gib=round(shutil.disk_usage(args.volume).free / 1024**3, 2),
        scope="Sample estimate only; excludes metadata, traces, archives and training checkpoints",
        sources=rows), indent=2))


if __name__ == "__main__":
    main()
