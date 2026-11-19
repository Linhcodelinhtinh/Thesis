"""Measure cached T1/T3 relation-writer latency on verified RGB demonstrations."""

import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from scripts.audit_bowl_cycle_replay import PROFILE, check_hash, read, rows
from src.interfaces.relation_memory_text import RelationMemoryRenderer
from src.interfaces.relation_memory_text_v3 import RelationMemoryRendererV3
from src.memory.relation_cycle_config import BowlCycleConfig
from src.memory.relation_cycle_writer import BowlCycleWriter
from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.memory.relation_writer_versions import RELATION_WRITERS, versioned_relation_source_hashes
from src.memory import relation_memory
from src.perception import visual_evidence
from src.perception.visual_evidence import VisualEvidence
from src.training.memory_data import validate_episode


def load_episode(index, data_dir, limit):
    manifest_path = index / "manifest.json"
    manifest = read(manifest_path)
    if (manifest["status"] != "complete" or manifest["writer"] != "rgb_relations_v1"
            or manifest["source_kind"] != "author_refined"):
        raise ValueError(f"Expected a complete author-refined relation index: {index}")
    episode = manifest["episodes"][0]["id"]
    task, demo = episode.rsplit("_demo_", 1)
    hdf5 = data_dir / (task + "_demo.hdf5")
    check_hash(hdf5, manifest["sources"][hdf5.name])
    detections_path = index / "detections.jsonl"
    check_hash(detections_path, manifest["artifacts_sha256"][detections_path.name])
    detections = [row for row in rows(detections_path) if row["episode_id"] == episode]
    if not detections or [row["evidence"]["step"] for row in detections] != list(range(len(detections))):
        raise ValueError(f"Missing or non-contiguous detection cache: {episode}")

    with h5py.File(hdf5, "r") as source:
        demo_data = source[f"data/demo_{demo}"]
        actions, robot, _ = validate_episode(demo_data)
        count = min(limit, len(detections), len(actions))
        images = {
            "agentview_image": demo_data["obs/agentview_rgb"][:count],
            "robot0_eye_in_hand_image": demo_data["obs/eye_in_hand_rgb"][:count],
        }
    frames = []
    for step in range(count):
        evidence = VisualEvidence.from_dict(detections[step]["evidence"])
        observation = {key: value[step] for key, value in images.items()}
        observation["robot0_gripper_qpos"] = robot[step, :2]
        observation["visual_evidence"] = evidence
        last_action = None if step == 0 else actions[step - 1]
        frames.append((observation, last_action))
    return episode, frames


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def measure(frames, writer_version, config, repeats):
    writer_type = RELATION_WRITERS[writer_version]
    renderer_type = RelationMemoryRendererV3 if writer_version == BowlCycleWriterV3.VERSION else RelationMemoryRenderer
    step_ms, render_ms, total_ms = [], [], []
    digest_calls = []
    original_digest = visual_evidence.image_digest

    def counted_digest(image):
        digest_calls.append(1)
        return original_digest(image)

    relation_memory.image_digest = counted_digest
    visual_evidence.image_digest = counted_digest
    try:
        for repeat in range(repeats + 1):
            options = {} if config is None else {"config": config}
            writer = writer_type("latency", PROFILE, None, "libero_mem_refined", **options)
            renderer = renderer_type(object(), token_counter=lambda text: len(text.split()))
            for step, (observation, last_action) in enumerate(frames):
                start = perf_counter_ns()
                snapshot = writer.step(step, observation, last_action)
                after_step = perf_counter_ns()
                renderer.render("pick up the bowl and place it back on the plate", snapshot)
                after_render = perf_counter_ns()
                if repeat:
                    step_ms.append((after_step - start) / 1_000_000)
                    render_ms.append((after_render - after_step) / 1_000_000)
                    total_ms.append((after_render - start) / 1_000_000)
            expected_calls = len(frames) * 2
            if len(digest_calls) != expected_calls * (repeat + 1):
                raise ValueError("Expected exactly one RGB digest per camera per frame")
    finally:
        relation_memory.image_digest = original_digest
        visual_evidence.image_digest = original_digest
    return {"writer_step_ms": {"p50": percentile(step_ms, .50), "p95": percentile(step_ms, .95)},
            "renderer_ms": {"p50": percentile(render_ms, .50), "p95": percentile(render_ms, .95)},
            "cached_total_ms": {"p50": percentile(total_ms, .50), "p95": percentile(total_ms, .95)},
            "measured_steps": len(total_ms), "camera_digests_per_step": 2,
            "total_digest_calls": len(digest_calls),
            "expected_digest_calls": len(frames) * 2 * (repeats + 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-t1", type=Path, required=True)
    parser.add_argument("--index-t3", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/libero_mem")
    parser.add_argument("--frame-limit", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cycle-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frame_limit < 1 or args.repeats < 2:
        parser.error("Require at least one frame and two measured repeats")
    if args.output.exists():
        raise FileExistsError(args.output)

    episodes = [load_episode(index, args.data_dir, args.frame_limit)
                for index in (args.index_t1, args.index_t3)]
    candidate_config = (BowlCycleConfig.load(args.cycle_config)
                        if args.cycle_config else BowlCycleConfig.load())
    all_results = {}
    for version, config in ((relation_memory.RelationMemoryWriter.VERSION, None),
                            (BowlCycleWriter.VERSION, BowlCycleConfig()),
                            (BowlCycleWriterV3.VERSION, candidate_config)):
        per_episode = []
        for episode, frames in episodes:
            result = measure(frames, version, config, args.repeats)
            per_episode.append({"episode": episode, "frames": len(frames), **result})
        all_results[version] = {
            "source_sha256": versioned_relation_source_hashes(ROOT, version),
            "episodes": per_episode,
            "aggregate": {
                key: {quantile: percentile([row[key][quantile] for row in per_episode], .50)
                      for quantile in ("p50", "p95")}
                for key in ("writer_step_ms", "renderer_ms", "cached_total_ms")
            },
            "total_digest_calls": sum(row["total_digest_calls"] for row in per_episode),
            "expected_digest_calls": sum(row["expected_digest_calls"] for row in per_episode),
        }

    result = {
        "classification": "CACHED_WRITER_MICROBENCHMARK_NON_COMPARABLE_OFFICIAL_LIBERO",
        "protocol": "Two author-refined T1/T3 demos; 256x256 RGB; cached detector evidence; one warm-up plus repeated writer/render passes.",
        "machine": {"platform": platform.platform(), "python": sys.version.split()[0],
                    "numpy": np.__version__},
        "repeats": args.repeats,
        "frame_limit": args.frame_limit,
        "episodes": [episode for episode, _ in episodes],
        "frames": {episode: len(frames) for episode, frames in episodes},
        "v3_config": candidate_config.to_dict(),
        "results": all_results,
        "historical_v2_reference_ms": {"p50": 2.18, "p95": 4.19,
                                        "note": "Prior measurement; hardware and protocol must match before claiming a direct comparison."},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": all_results}, indent=2))


if __name__ == "__main__":
    main()
