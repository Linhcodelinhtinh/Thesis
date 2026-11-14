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
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.replay_libero_mem_writer import TASKS, sha256, write_json
from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.memory.visual_pipeline import load_prompt_tokenizer, visual_writer_factory, TOKENIZER_REVISION
from src.perception.grounding_dino import GroundingDinoDetector
from src.perception.object_tracker import TrackerConfig
from src.training.libero_mem_hdf5_loader import LiberoMemHDF5Parser
from src.training.split_lock import AUTHOR_MEM_SPLIT, load_split_lock
from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, ROOT, load_suite_source, source_digest
from src.utils.file_hash import sha256_file


def recover_completed_index(out_dir, expected_frames=None):
    """Recover metadata for all fully-indexed episodes and prune incomplete trailing frames."""
    s_file = out_dir / "samples.jsonl"
    t_file = out_dir / "memory_trace.jsonl"
    d_file = out_dir / "detections.jsonl"
    if not (s_file.is_file() and t_file.is_file() and d_file.is_file()):
        return [], 0, 0, set()

    samples_by_ep, ep_order = {}, []
    with open(s_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except Exception:
                break
            ep = rec.get("episode_id")
            if not ep:
                break
            if ep not in samples_by_ep:
                samples_by_ep[ep] = []
                ep_order.append(ep)
            samples_by_ep[ep].append((line, rec))

    trace_by_ep = {}
    with open(t_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except Exception:
                break
            ep = rec.get("episode_id")
            if not ep:
                break
            trace_by_ep.setdefault(ep, []).append((line, rec))

    det_by_ep = {}
    with open(d_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except Exception:
                break
            ep = rec.get("episode_id")
            if not ep:
                break
            det_by_ep.setdefault(ep, []).append((line, rec))

    valid_episodes = []
    reconstructed_meta = []
    total_samples = 0
    with_memory = 0
    valid_samples_raw = []
    valid_trace_raw = []
    valid_det_raw = []

    for ep in ep_order:
        s_list = samples_by_ep.get(ep, [])
        t_list = trace_by_ep.get(ep, [])
        d_list = det_by_ep.get(ep, [])
        n = len(s_list)
        if expected_frames is not None and expected_frames.get(ep) != n:
            print(f"Pruning episode '{ep}': recovered {n} of {expected_frames.get(ep)} locked frames")
            break
        if n == 0 or len(t_list) != n or len(d_list) != n:
            print(f"Pruning incomplete episode '{ep}': samples={n}, trace={len(t_list)}, det={len(d_list)}")
            break
        steps = [rec["step"] for _, rec in s_list]
        if steps != list(range(n)):
            print(f"Pruning episode with broken steps '{ep}': steps={steps[:5]}...")
            break

        last_snap = t_list[-1][1]["snapshot"] if t_list else {}
        relations = last_snap.get("relations")
        relation_writer = isinstance(relations, dict)
        cycle_confirmed = sum(r["returns"] for r in relations.values()) if relation_writer else 0
        if last_snap and last_snap.get("cycles"):
            for c in last_snap["cycles"].values():
                cycle_confirmed = c.get("confirmed", 0)

        mem_frames = sum(1 for _, s in s_list if s.get("has_memory"))
        reconstructed_meta.append(dict(
            id=ep,
            split=s_list[0][1]["split"],
            frames=n,
            memory_frames=mem_frames,
            confirmed_returns=cycle_confirmed,
            history_complete=(all(r["history_complete"] for r in relations.values())
                              if relation_writer else last_snap.get("history_complete"))
        ))
        valid_episodes.append(ep)
        total_samples += n
        with_memory += mem_frames
        valid_samples_raw.extend(raw for raw, _ in s_list)
        valid_trace_raw.extend(raw for raw, _ in t_list)
        valid_det_raw.extend(raw for raw, _ in d_list)

    if valid_episodes:
        s_file.write_text("".join(valid_samples_raw), encoding="utf-8")
        t_file.write_text("".join(valid_trace_raw), encoding="utf-8")
        d_file.write_text("".join(valid_det_raw), encoding="utf-8")

    return reconstructed_meta, total_samples, with_memory, set(valid_episodes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/libero_mem")
    p.add_argument("--task-name", required=True)
    p.add_argument("--writer", choices=("rgb_cycle_v3", "rgb_relations_v1"), default="rgb_cycle_v3")
    p.add_argument("--projection", choices=("relevant", "recent"), default="relevant")
    p.add_argument("--source-receipt", type=Path, help="Official pinned LFS receipt for additional author tasks")
    p.add_argument("--metainfo", type=Path, help="Use the existing pinned metadata file from another directory")
    p.add_argument("--demo-ids", nargs="+", type=int, help="Select exact source demos for a diagnostic")
    p.add_argument("--stream", choices=["predicted_memory", "no_memory"], default="predicted_memory")
    p.add_argument("--tokenizer-path", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=ROOT / "resources/models/grounding-dino-tiny")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--split", choices=["train", "val", "test"])
    p.add_argument("--split-lock", type=Path, help="Required when producing memory for the mixed training dataset")
    p.add_argument("--source-kind", choices=("author_refined", "derived_raw"), default="author_refined",
                   help="Select the pinned source RGB convention and provenance contract")
    p.add_argument("--resume", action="store_true", help="Resume from partially written prompt index")
    args = p.parse_args()
    from src.evaluation.libero_mem_protocol import task_spec
    from src.memory.relation_pipeline import relation_config, relation_pipeline, relation_source_hashes
    from src.perception.relation_detector import RelationDetector

    suite = load_suite_source(ROOT / "external/libero-mem")
    args.task_name = task_spec(suite, args.task_name)["name"]
    relation = args.writer == "rgb_relations_v1"
    if not relation and args.task_name not in TASKS.values():
        p.error("rgb_cycle_v3 supports only T1/T3")
    if not relation and args.projection != "relevant":
        p.error("Choose rgb_relations_v1 to change the projection")
    if args.demo_ids and (args.split_lock or args.resume or args.max_episodes):
        p.error("Explicit demo selection is diagnostic and cannot be resumed or used with a lock")
    out = args.output_dir
    split_lock = load_split_lock(args.split_lock) if args.split_lock else None
    if args.resume and not split_lock and out.exists():
        p.error("Resuming an existing index requires a locked roster")
    if split_lock is not None and split_lock["protocol_sha256"] != source_digest(DEFAULT_PROTOCOL):
        raise ValueError("Split lock was generated for a different evaluator protocol")
    if args.split in ("val", "test") and (split_lock is None or split_lock.get("mem_split_policy") == AUTHOR_MEM_SPLIT):
        p.error("Refined Mem is train-only. Raw evaluation uses its own roster")
    expected_frames = None if split_lock is None else {
        f'{row["task"]}_{row["demo"]}': row["frames"]
        for row in split_lock["episodes"]
        if row["domain"] == "mem" and row["task"] == args.task_name
    }
    episodes, total, with_memory, completed_ids = [], 0, 0, set()
    source_manifest = json.loads((ROOT / "resources/manifests/libero_mem_planning_sources.json").read_text(encoding="utf-8"))
    filenames = [args.task_name + "_demo.hdf5", "metainfo.json"]
    metainfo_path = args.metainfo if args.metainfo is not None else args.data_dir / "metainfo.json"
    sources = {filenames[0]: sha256(args.data_dir / filenames[0]), "metainfo.json": sha256(metainfo_path)}
    if args.source_kind == "author_refined":
        inventory = {item["path"]: item["upstream_sha256"]
                     for item in source_manifest["annotated_dataset"]["selected_pilot_files"]}
        if args.source_receipt:
            receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
            upstream = suite["dataset"]
            if (receipt["repository"] != upstream["repository"].split("/datasets/", 1)[1]
                    or receipt["revision"] != upstream["revision"]):
                raise ValueError("Author receipt does not identify the pinned release")
            hashes = receipt.get("sha256", receipt.get("files"))
            if not isinstance(hashes, dict) or ({"sha256", "files"} <= receipt.keys()
                                               and receipt["sha256"] != receipt["files"]):
                raise ValueError("Receipt must contain an unambiguous source hash inventory")
            for name, digest in hashes.items():
                if name in inventory and inventory[name] != digest:
                    raise ValueError("Receipt conflicts with the pinned pilot data")
                inventory[name] = digest
        for name, actual in sources.items():
            if actual != inventory.get(name):
                raise ValueError(f"Unverified/corrupt author source: {name}")
        image_convention = "libero_mem_refined"
        source_revision = source_manifest["annotated_dataset"]["revision"]
    else:
        if split_lock is None or split_lock.get("classification") != "CUSTOM_V2_LIBERO_MEM_DERIVED_TRAINING_DATA":
            raise ValueError("derived_raw requires the derived-data split lock")
        if split_lock.get("mem_image_convention") != "raw_opengl":
            raise ValueError("Derived split lock has an unknown RGB convention")
        locked_sources = {row["path"]: row["sha256"] for row in split_lock["sources"] if row["domain"] == "mem"}
        for name, actual in sources.items():
            if actual != locked_sources.get(name):
                raise ValueError(f"Derived source differs from its lock: {name}")
        image_convention = "raw_opengl"
        source_revision = "derived:" + split_lock["derived_manifest_sha256"]
    tokenizer = load_prompt_tokenizer(args.tokenizer_path)
    config = relation_config(args.projection) if relation else asdict(TrackerConfig())
    writer_convention = "libero_raw" if image_convention == "raw_opengl" else image_convention
    use_detector = args.stream == "predicted_memory"
    code_hashes = relation_source_hashes(ROOT) if relation else {
        str(path.relative_to(ROOT)).replace(chr(92), "/"): sha256(path)
        for directory in ("src/perception", "src/memory", "src/interfaces")
        for path in sorted((ROOT / directory).glob("*.py"))
        if "relation" not in path.name and path.name != "libero_mem_profiles.py"
    }
    detector_files = (json.loads((args.model_dir / "memory_detector_lock.json").read_text(encoding="utf-8"))
                      if use_detector else None)
    contract = dict(task=args.task_name, writer=args.writer, config=config, stream=args.stream,
                    source_hashes=sources, code_hashes=code_hashes, image_convention=image_convention,
                    split_lock_sha256=split_lock["lock_sha256"] if split_lock else None,
                    demo_ids=args.demo_ids, split=args.split, max_episodes=args.max_episodes,
                    detector_lock=detector_files, detector_device=args.device if use_detector else None,
                    indexer_sha256=sha256(Path(__file__)))
    if out.exists() and args.resume:
        receipt_path = out / "index_contract.json"
        if not receipt_path.is_file() or json.loads(receipt_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Resume requires the same source, writer, selection and projection")
    if out.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite prompt index: {out}")
        manifest_path = out / "manifest.json"
        if manifest_path.is_file():
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_data.get("status") == "complete":
                files = manifest_data["artifacts_sha256"]
                if set(files) != {"samples.jsonl", "memory_trace.jsonl", "detections.jsonl"} or any(
                        sha256(out / name) != digest for name, digest in files.items()):
                    raise ValueError("Completed prompt index checksum mismatch")
                print(f"Prompt index at {out} is already complete ({manifest_data.get('total_samples')} samples).")
                return
        if use_detector:
            episodes, total, with_memory, completed_ids = recover_completed_index(out, expected_frames)
        print(f"Resuming index at {out}: recovered {len(completed_ids)} completed episodes ({total} samples).")
    else:
        out.mkdir(parents=True)
    write_json(out / "index_contract.json", contract)
    detector = (RelationDetector if relation else GroundingDinoDetector)(args.model_dir, args.device) if use_detector else None
    if relation:
        factory, renderer = relation_pipeline(suite, args.task_name, detector, writer_convention,
                                              tokenizer, args.projection)
    else:
        factory = visual_writer_factory(detector, writer_convention, TrackerConfig(**config))
        renderer = LiberoMemTextRenderer(tokenizer=tokenizer, strict_tokenizer=True)
    loader = LiberoMemHDF5Parser(args.data_dir, split_lock=args.split_lock, metainfo_path=metainfo_path)
    mode = "a" if completed_ids else "w"
    with (out / "samples.jsonl").open(mode, encoding="utf-8") as samples, \
            (out / "memory_trace.jsonl").open(mode, encoding="utf-8") as trace, \
            (out / "detections.jsonl").open(mode, encoding="utf-8") as predictions:
        for episode in loader.iter_task(args.task_name, args.max_episodes, args.split, demo_ids=args.demo_ids):
            if episode.episode_id in completed_ids:
                continue
            writer = factory(episode.episode_id) if detector else None
            count = 0
            for t, observation in enumerate(episode.observations):
                started = time.perf_counter()
                snapshot = writer.step(t, observation, episode.actions[t-1] if t else None) if writer else None
                writer_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                result = renderer.render(episode.instruction, snapshot, enabled=writer is not None)
                projection_ms = (time.perf_counter() - started) * 1000
                if result.tokenizer_truncated:
                    raise ValueError(f"Instruction exceeds token budget: {episode.episode_id}")
                has_memory = bool(result.facts_included)
                count += has_memory
                total += 1
                samples.write(json.dumps(dict(episode_id=episode.episode_id, split=episode.split, step=t,
                    source_hdf5=filenames[0], source_demo="demo_" + episode.episode_id.rsplit("_demo_", 1)[-1],
                    source_obs_idx=t, source_action_start_idx=t, instruction=episode.instruction,
                    rendered_prompt=result.text, has_memory=has_memory, tokens=result.total_tokens_with_instruction,
                    skipped_reason=result.memory_skipped_reason, dropped_facts=result.dropped_facts,
                    dependency_failures=getattr(result, "dependency_failures", ()),
                    writer_ms=writer_ms, projection_ms=projection_ms)) + "\n")
                if writer:
                    trace.write(json.dumps(dict(episode_id=episode.episode_id, step=t, signals=writer.last_signals,
                                               snapshot=snapshot.to_dict())) + "\n")
                    evidence = writer.last_evidence if relation else writer.tracker.last_evidence
                    predictions.write(json.dumps(dict(episode_id=episode.episode_id,
                                                       evidence=evidence.to_dict())) + "\n")
            with_memory += count
            states = tuple(snapshot.relations.values()) if relation and writer else ()
            episodes.append(dict(id=episode.episode_id, split=episode.split, frames=len(episode.actions),
                                 memory_frames=count,
                                 confirmed_returns=(sum(state.returns for state in states) if states else
                                                    writer.confirmed_cycles if writer else None),
                                 history_complete=(all(state.history_complete for state in states) if states else
                                                   writer.history_complete if writer else None)))
            print(json.dumps(episodes[-1]), flush=True)
    if not total:
        raise ValueError("No episodes selected; no completed manifest was written")
    write_json(out / "manifest.json", dict(artifact="causal_prompt_index_not_training_shards", status="complete",
        writer=args.writer if detector else "none", writer_config=config, stream=args.stream,
        index_contract_sha256=sha256(out / "index_contract.json"),
        split_lock_sha256=loader.split_lock["lock_sha256"] if loader.split_lock else None,
        writer_code_sha256=code_hashes,
        sources=sources, source_kind=args.source_kind, source_revision=source_revision,
        detector=(detector.relation_provenance if relation else detector.provenance) if detector else None,
        tokenizer_revision=TOKENIZER_REVISION,
        obs_action_offset=0, image_convention=image_convention, max_total_tokens=48,
        split_contract=loader.split_lock.get("mem_split_policy", "legacy_custom_initial_state_split")
        if loader.split_lock else "refined_train_only_unlocked",
        episodes=episodes, total_samples=total, memory_frames=with_memory,
        artifacts_sha256={name: sha256(out / name) for name in ("samples.jsonl", "memory_trace.jsonl", "detections.jsonl")}))
    print(f"Indexed {total} samples; {with_memory} with memory. Coverage is not event accuracy.")


if __name__ == "__main__":
    main()
