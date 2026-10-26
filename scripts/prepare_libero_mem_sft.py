#!/usr/bin/env python
"""Prepare a verified causal PROMPT INDEX for T1/T3; not model-training shards.

Runs the same RGB writer as online evaluation. The output retains HDF5 pointers,
strict token counts, predictions and event traces. Fine-tuning/model image
adaptation is deliberately separate. No mock episodes or heuristic oracle.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.replay_libero_mem_writer import TASKS, sha256, write_json
from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.memory.visual_pipeline import load_prompt_tokenizer, visual_writer_factory, TOKENIZER_REVISION
from src.perception.grounding_dino import GroundingDinoDetector
from src.perception.object_tracker import TrackerConfig
from src.training.libero_mem_hdf5_loader import LiberoMemHDF5Parser


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/libero_mem")
    p.add_argument("--task-name", choices=list(TASKS.values()), required=True)
    p.add_argument("--stream", choices=["predicted_memory", "no_memory"], default="predicted_memory")
    p.add_argument("--tokenizer-path", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=ROOT / "resources/models/grounding-dino-tiny")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--split", choices=["train", "val"])
    args = p.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite prompt index: {out}")
    lock = json.loads((ROOT / "resources/manifests/libero_mem_planning_sources.json").read_text(encoding="utf-8"))
    inventory = {item["path"]: item["upstream_sha256"] for item in lock["annotated_dataset"]["selected_pilot_files"]}
    filenames = [args.task_name + "_demo.hdf5", "metainfo.json"]
    sources = {name: sha256(args.data_dir / name) for name in filenames}
    for name, actual in sources.items():
        if actual != inventory[name]:
            raise ValueError(f"Unverified/corrupt source: {name}")
    renderer = LiberoMemTextRenderer(tokenizer=load_prompt_tokenizer(args.tokenizer_path), strict_tokenizer=True)
    detector = GroundingDinoDetector(args.model_dir, args.device) if args.stream == "predicted_memory" else None
    config = TrackerConfig()
    factory = visual_writer_factory(detector, "libero_mem_refined", config)
    loader = LiberoMemHDF5Parser(args.data_dir)
    out.mkdir(parents=True)
    episodes, total, with_memory = [], 0, 0
    with (out / "samples.jsonl").open("w", encoding="utf-8") as samples, \
            (out / "memory_trace.jsonl").open("w", encoding="utf-8") as trace, \
            (out / "detections.jsonl").open("w", encoding="utf-8") as predictions:
        for episode in loader.iter_task(args.task_name, args.max_episodes, args.split):
            writer = factory(episode.episode_id) if detector else None
            count = 0
            for t, observation in enumerate(episode.observations):
                snapshot = writer.step(t, observation, episode.actions[t-1] if t else None) if writer else None
                result = renderer.render(episode.instruction, snapshot, enabled=writer is not None)
                has_memory = bool(result.facts_included)
                count += has_memory
                total += 1
                samples.write(json.dumps(dict(episode_id=episode.episode_id, split=episode.split, step=t,
                    source_hdf5=filenames[0], source_demo="demo_" + episode.episode_id.rsplit("_demo_", 1)[-1],
                    source_obs_idx=t, source_action_start_idx=t, instruction=episode.instruction,
                    rendered_prompt=result.text, has_memory=has_memory, tokens=result.total_tokens_with_instruction,
                    skipped_reason=result.memory_skipped_reason)) + "\n")
                if writer:
                    trace.write(json.dumps(dict(episode_id=episode.episode_id, step=t, signals=writer.last_signals,
                                               snapshot=snapshot.to_dict())) + "\n")
                    predictions.write(json.dumps(dict(episode_id=episode.episode_id,
                                                       evidence=writer.tracker.last_evidence.to_dict())) + "\n")
            with_memory += count
            episodes.append(dict(id=episode.episode_id, split=episode.split, frames=len(episode.actions),
                                 memory_frames=count, confirmed_returns=writer.confirmed_cycles if writer else None,
                                 history_complete=writer.history_complete if writer else None))
            print(json.dumps(episodes[-1]), flush=True)
    if not total:
        raise ValueError("No episodes selected; no completed manifest was written")
    write_json(out / "manifest.json", dict(artifact="causal_prompt_index_not_training_shards", status="complete",
        writer="rgb_cycle_v3" if detector else "none", writer_config=asdict(config), stream=args.stream,
        sources=sources, source_revision=lock["annotated_dataset"]["revision"],
        detector=detector.provenance if detector else None, tokenizer_revision=TOKENIZER_REVISION,
        obs_action_offset=0, image_convention="libero_mem_refined", max_total_tokens=48,
        split_contract="same canonical initial state grouped across T1/T3; not official benchmark split",
        episodes=episodes, total_samples=total, memory_frames=with_memory,
        artifacts_sha256={name: sha256(out / name) for name in ("samples.jsonl", "memory_trace.jsonl", "detections.jsonl")}))
    print(f"Indexed {total} samples; {with_memory} with memory. Coverage is not event accuracy.")


if __name__ == "__main__":
    main()
