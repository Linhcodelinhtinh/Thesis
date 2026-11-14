"""Compare relation-writer returns with released T1–T4 development milestones."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_libero_mem_writer import match_events
from src.utils.file_hash import sha256_file


def main():
    import ijson

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--metainfo", type=Path, default=ROOT / "data/libero_mem/metainfo.json")
    parser.add_argument("--tolerance", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.index / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["status"] != "complete" or manifest["writer"] != "rgb_relations_v1"
            or manifest["source_kind"] != "author_refined" or args.tolerance < 0):
        raise ValueError("Audit requires a complete author-refined relation index and nonnegative tolerance")
    if sha256_file(args.metainfo) != manifest["sources"]["metainfo.json"]:
        raise ValueError("Annotation hash differs from the prediction source")
    trace_path = args.index / "memory_trace.jsonl"
    if sha256_file(trace_path) != manifest["artifacts_sha256"]["memory_trace.jsonl"]:
        raise ValueError("Prediction trace checksum mismatch")
    events, sizes = {}, {}
    with trace_path.open(encoding="utf-8") as trace:
        for line in trace:
            row = json.loads(line)
            key = row["episode_id"]
            if row["step"] != sizes.get(key, 0):
                raise ValueError("Prediction steps are not contiguous")
            sizes[key] = row["step"] + 1
            selected = events.setdefault(key, {})
            for event in row["snapshot"]["recent_events"]:
                if event["type"] == "return":
                    selected[event["id"]] = event["available_at"]
    report = {"classification": "DEVELOPMENT_MILESTONES_NOT_POLICY_SCORE",
              "index_sha256": sha256_file(manifest_path), "tolerance_frames": args.tolerance, "episodes": []}
    for episode in manifest["episodes"]:
        key = episode["id"]
        task, demo = key.rsplit("_demo_", 1)
        matched_task = re.match(r"KITCHEN_SCENE1_([1-4])_", task)
        if not matched_task:
            raise ValueError("Milestone parser currently supports the released T1–T4 labels")
        task_number = int(matched_task[1])
        label = "bowl 1" if task_number in (1, 3) else "bottle 1"
        with args.metainfo.open("rb") as stream:
            labels = next(row["exo_boxes"] for name, row in ijson.kvitems(stream, task, use_float=True)
                          if name == f"demo_{demo}")
        if len(labels) != sizes[key] or len(labels) != episode["frames"]:
            raise ValueError("Prediction and annotation frame counts differ")
        vocabulary = {"0": 0, **{f"{name}: {label} on plate 1": i
                                 for i, name in enumerate(("one", "two", "three"), 1)}}
        counts = [vocabulary[str(frame[label][2])] for frame in labels]
        if any(b < a or b > a + 1 for a, b in zip(counts, counts[1:])):
            raise ValueError("Unexpected progression in author labels")
        milestones = [i for i, (a, b) in enumerate(zip(counts, counts[1:]), 1) if b > a]
        predicted = sorted(events[key].values())
        report["episodes"].append(dict(episode=key, milestones=milestones, predicted=predicted,
                                        **match_events(predicted, milestones, args.tolerance)))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
