#!/usr/bin/env python
"""CLI tool to prepare, align, and shard LIBERO-Mem demonstration data.

Adheres strictly to docs/V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md Section 8:
- Applies exact observation-action offset (offset 0 for refined Mem data).
- Generates causal memory streams: predicted_memory, oracle_memory, no_memory.
- Enforces strict 48-token prompt ceiling without instruction truncation.
- Splits trajectories deterministically by initial-state/episode hash.
- Writes manifest.json and samples.jsonl.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.libero_mem_data import LiberoMemDataProcessor, LiberoMemHDF5Parser, MemoryStream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, help="Directory to save shards and manifest")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "libero_mem"), help="Path to raw HDF5 directory")
    p.add_argument("--task-name", default="KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate", help="Task name or pattern")
    p.add_argument("--stream", default="predicted_memory", choices=["predicted_memory", "oracle_memory", "no_memory"])
    p.add_argument("--obs-action-offset", type=int, default=0, help="0 for Mem refined, 1 for Base raw")
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--max-episodes", type=int, default=None, help="Max real episodes to process")
    p.add_argument("--split", choices=["train", "val"], default=None, help="Filter by train or val split")
    p.add_argument("--mock-episodes", type=int, default=0, help="Generate mock episodes for testing")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = LiberoMemDataProcessor(
        chunk_size=args.chunk_size,
        action_dim=args.action_dim,
        obs_action_offset=args.obs_action_offset,
    )

    all_samples = []
    task_name = args.task_name
    episodes_count = 0
    raw_hdf5_file = None

    if args.mock_episodes > 0:
        instruction = "pick up the bowl and place it back on the plate"
        logger.info("Generating %d synthetic mock episodes for verification...", args.mock_episodes)
        for ep_idx in range(args.mock_episodes):
            ep_id = f"mock_{ep_idx:03d}"
            observations = []
            actions = np.zeros((20, args.action_dim), dtype=np.float32)

            for step in range(20):
                z_pos = 0.85 + (0.3 if 5 <= step <= 15 else 0.0)
                obs = {
                    "robot0_eef_pos": np.array([-0.05, 0.20, z_pos], dtype=np.float32),
                    "robot0_gripper_qpos": np.array([0.01, 0.01] if 5 <= step <= 15 else [0.04, 0.04], dtype=np.float32),
                    "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "detected_objects": {"bowl_A": np.array([-0.05, 0.20, z_pos], dtype=np.float32)},
                    "oracle_signals": {
                        "is_on_support": not (5 <= step <= 15),
                        "is_held": (5 <= step <= 15),
                        "container_id": "plate_A" if not (5 <= step <= 15) else None,
                        "is_occluded": False,
                    },
                }
                observations.append(obs)
                actions[step, -1] = 1.0 if 5 <= step <= 15 else -1.0

            ep_samples = processor.process_episode(
                episode_id=ep_id,
                instruction=instruction,
                observations=observations,
                actions=actions,
                stream=args.stream,
            )
            all_samples.extend(ep_samples)
        episodes_count = args.mock_episodes
    else:
        # Load from real HDF5 dataset
        data_dir = Path(args.data_dir)
        logger.info("Loading real HDF5 demonstrations from %s for task '%s'...", data_dir, task_name)
        parser = LiberoMemHDF5Parser(data_dir=data_dir)
        hdf5_path = parser.resolve_hdf5_path(task_name)
        raw_hdf5_file = hdf5_path.name

        episodes = parser.parse_task(
            task_name=task_name,
            max_episodes=args.max_episodes,
            split_filter=args.split,
        )
        episodes_count = len(episodes)
        logger.info("Parsing %d real demonstration episodes into aligned samples...", episodes_count)

        for ep in episodes:
            ep_samples = processor.process_episode(
                episode_id=ep.episode_id,
                instruction=ep.instruction,
                observations=ep.observations,
                actions=ep.actions,
                stream=args.stream,
            )
            all_samples.extend(ep_samples)

    # Save samples.jsonl
    jsonl_path = out_dir / "samples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            row = {
                "sample_id": s.sample_id,
                "episode_id": s.episode_id,
                "step": s.step,
                "instruction": s.instruction,
                "rendered_prompt": s.rendered_prompt,
                "has_memory": s.has_memory,
                "source_obs_idx": s.source_obs_idx,
                "source_action_start_idx": s.source_action_start_idx,
            }
            f.write(json.dumps(row) + "\n")

    # Save manifest.json
    manifest = {
        "suite": "libero_mem_pilot",
        "task_name": task_name,
        "raw_hdf5_file": raw_hdf5_file,
        "memory_stream": args.stream,
        "obs_action_offset": args.obs_action_offset,
        "chunk_size": args.chunk_size,
        "total_episodes": episodes_count,
        "total_samples": len(all_samples),
        "samples_with_memory": sum(1 for s in all_samples if s.has_memory),
        "jsonl_file": str(jsonl_path.name),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote %d aligned samples to %s", len(all_samples), jsonl_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

