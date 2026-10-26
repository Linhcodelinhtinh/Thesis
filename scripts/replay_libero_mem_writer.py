"""Replay the predicted RGB writer on verified T1/T3 demos; no policy or training.

Produces auditable frame-bound detections, event/prompt traces and diagnostics.
Coverage is NOT accuracy. No annotation or target cycle count enters the writer.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.memory.visual_pipeline import (load_prompt_tokenizer, visual_writer_factory,
                                        TOKENIZER_REPO, TOKENIZER_REVISION, TOKENIZER_SHA256)
from src.perception.grounding_dino import GroundingDinoDetector
from src.perception.cached_detector import CachedVisualDetector
from src.perception.object_tracker import TrackerConfig
from src.training.libero_mem_hdf5_loader import LiberoMemHDF5Parser
from src.utils.file_hash import sha256_file

TASKS = {
    "T1": "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate",
    "T3": "KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times",
}


def sha256(path):
    return sha256_file(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/libero_mem")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "resources/models/grounding-dino-tiny")
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=["T1", "T3"])
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--demo-ids", nargs="+", type=int, help="Exact demonstration IDs; overrides max-episodes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detection-cache", type=Path, help="Completed replay directory containing hash-verified RGB predictions")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {args.output}")
    lock = json.loads((ROOT / "resources/manifests/libero_mem_planning_sources.json").read_text(encoding="utf-8"))
    expected = {f["path"]: f["upstream_sha256"] for f in lock["annotated_dataset"]["selected_pilot_files"]}
    sources = {}
    for name in [f"{TASKS[t]}_demo.hdf5" for t in args.tasks] + ["metainfo.json"]:
        actual = sha256(args.data_dir / name)
        if actual != expected[name]:
            raise ValueError(f"Author file checksum mismatch: {name}")
        sources[name] = actual
        print(f"Verified {name}", flush=True)
    tokenizer = load_prompt_tokenizer(args.tokenizer_path)
    renderer = LiberoMemTextRenderer(tokenizer=tokenizer, strict_tokenizer=True)
    cache_manifest = None
    if args.detection_cache:
        cache_manifest = json.loads((args.detection_cache / "manifest.json").read_text(encoding="utf-8"))
        if cache_manifest["status"] != "complete" or cache_manifest["sources"] != sources:
            raise ValueError("Cache must be complete and reference exactly the same author files")
    detector = None if cache_manifest else GroundingDinoDetector(args.model_dir, device=args.device)
    config = TrackerConfig()
    loader = LiberoMemHDF5Parser(args.data_dir)
    args.output.mkdir(parents=True)
    manifest = dict(status="running", purpose="writer diagnostic, not benchmark success or training dataset",
                    execution_mode="cached_predictions" if cache_manifest else "live_detector",
                    prediction_origin_manifest_sha256=sha256(args.detection_cache / "manifest.json") if cache_manifest else None,
                    prediction_origin_code_hashes=cache_manifest["code_hashes"] if cache_manifest else None,
                    writer="rgb_cycle_v3", config=asdict(config), detector=cache_manifest["detector"] if cache_manifest else detector.provenance,
                    image_convention="libero_mem_refined", obs_action_offset=0, sources=sources,
                    data_revision=lock["annotated_dataset"]["revision"],
                    tokenizer=dict(repo=TOKENIZER_REPO, revision=TOKENIZER_REVISION, sha256=TOKENIZER_SHA256),
                    code_hashes={str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for directory in
                                 ("src/perception", "src/memory", "src/interfaces", "src/training")
                                 for p in sorted((ROOT / directory).glob("*.py"))}, episodes=[])
    write_json(args.output / "manifest.json", manifest)
    for task in args.tasks:
        for episode in loader.iter_task(TASKS[task], max_episodes=None if args.demo_ids else args.max_episodes, demo_ids=args.demo_ids):
            if cache_manifest:
                cached_episode = next(e for e in cache_manifest["episodes"] if e["episode"] == episode.episode_id)
                detector = CachedVisualDetector(args.detection_cache / cached_episode["detections"], cached_episode["detections_sha256"])
            writer = visual_writer_factory(detector, "libero_mem_refined", config)(episode.episode_id)
            prefix = f"{task}_{episode.episode_id.rsplit('_demo_', 1)[-1]}"
            reasons, events, memory_frames, grounded, after_departure = Counter(), Counter(), 0, 0, 0
            latencies = []
            seen_events = 0
            with (args.output / f"{prefix}_trace.jsonl").open("w", encoding="utf-8") as trace, \
                    (args.output / f"{prefix}_detections.jsonl").open("w", encoding="utf-8") as detections:
                for step, obs in enumerate(episode.observations):
                    started = time.perf_counter()
                    snapshot = writer.step(step, obs, episode.actions[step-1] if step else None)
                    latencies.append((time.perf_counter() - started) * 1000)
                    rendered = renderer.render(episode.instruction, snapshot)
                    reasons[writer.last_signals["reason"]] += 1
                    grounded += int(writer.last_signals["has_grounding_evidence"])
                    memory_frames += int(bool(rendered.facts_included))
                    after_departure += int(bool(rendered.facts_included) and any(ev.type == "departed_support" for ev in snapshot.events))
                    for ev in snapshot.events[seen_events:]:
                        events[ev.type] += 1
                        print(f"{prefix} frame {step}: {ev.type}", flush=True)
                    seen_events = len(snapshot.events)
                    detections.write(json.dumps(writer.tracker.last_evidence.to_dict()) + "\n")
                    trace.write(json.dumps(dict(step=step, source_action_start_idx=step, prompt=rendered.text,
                        has_memory=bool(rendered.facts_included), tokens=rendered.total_tokens_with_instruction,
                        skipped_reason=rendered.memory_skipped_reason, signals=writer.last_signals,
                        snapshot=snapshot.to_dict())) + "\n")
                    if step % 50 == 0:
                        print(f"{prefix}: {step}/{len(episode.actions)} frames", flush=True)
            summary = dict(task=task, episode=episode.episode_id, split=episode.split,
                           initial_state_sha256=hashlib.sha256(episode.initial_state.astype('<f8').tobytes()).hexdigest(),
                           frames=len(episode.actions), memory_frames=memory_frames,
                           memory_frames_after_departure=after_departure, grounded_frames=grounded,
                           confirmed_returns=writer.confirmed_cycles, history_complete=writer.history_complete,
                           unobserved_intervals=writer.uncertain_count, reasons=dict(reasons), events=dict(events),
                           mean_writer_ms=sum(latencies)/len(latencies), trace=f"{prefix}_trace.jsonl",
                           detections=f"{prefix}_detections.jsonl",
                           detections_sha256=sha256(args.output / f"{prefix}_detections.jsonl"),
                           trace_sha256=sha256(args.output / f"{prefix}_trace.jsonl"))
            if cache_manifest:
                detector.close()
            manifest["episodes"].append(summary)
            write_json(args.output / "manifest.json", manifest)
            print(json.dumps(summary), flush=True)
    manifest["status"] = "complete"
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
