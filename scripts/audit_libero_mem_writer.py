"""Independent comparison of RGB writer traces with author metadata.

Annotations are read ONLY here, after prediction. Milestone timing is the
author's cumulative task label, not a physical grasp/contact annotation.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.perception.grounding_dino import box_iou
from src.utils.file_hash import sha256_file


def sha256(path):
    return sha256_file(path)


def match_events(predicted, annotated, tolerance=20):
    """Chronological, one-to-one matching; duplicate predictions are false positives."""
    remaining = set(range(len(annotated)))
    matches = []
    for p in predicted:
        choices = [i for i in remaining if abs(p - annotated[i]) <= tolerance]
        if choices:
            i = min(choices, key=lambda i: (abs(p - annotated[i]), i))
            remaining.remove(i)
            matches.append(dict(predicted=p, annotation=annotated[i], delay=p-annotated[i]))
    return dict(matches=matches, true_positive=len(matches), false_positive=len(predicted)-len(matches),
                false_negative=len(annotated)-len(matches))


def main():
    import ijson
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--metainfo", type=Path, default=ROOT / "data/libero_mem/metainfo.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-tolerance", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.replay / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Only completed replay artifacts can be evaluated")
    if sha256(args.metainfo) != manifest["sources"]["metainfo.json"]:
        raise ValueError("Annotation file differs from the replay's pinned source")
    report = dict(purpose="writer diagnostic, NOT official LIBERO policy performance",
                  replay_manifest_sha256=sha256(args.replay / "manifest.json"),
                  milestone_semantics="author cumulative task milestone labels; not grasp/contact labels",
                  event_tolerance_frames=args.event_tolerance, episodes=[])
    for episode in manifest["episodes"]:
        task, demo = episode["episode"].rsplit("_demo_", 1)
        with args.metainfo.open("rb") as file:
            meta = next(value for key, value in ijson.kvitems(file, task, use_float=True) if key == "demo_" + demo)
        for field in ("trace", "detections"):
            if sha256(args.replay / episode[field]) != episode[field + "_sha256"]:
                raise ValueError(f"Corrupt {field} artifact")
        trace = [json.loads(line) for line in (args.replay / episode["trace"]).read_text(encoding="utf-8").splitlines()]
        predictions = [json.loads(line) for line in (args.replay / episode["detections"]).read_text(encoding="utf-8").splitlines()]
        boxes = meta["exo_boxes"]
        if len(trace) != len(boxes) or len(trace) != len(predictions):
            raise ValueError("Annotation/prediction frame lengths differ")
        if any(row["step"] != index for sequence in (trace, predictions) for index, row in enumerate(sequence)):
            raise ValueError("Trace/prediction observation indices are not contiguous")
        counts = []
        for frame in boxes:
            label = str(frame["bowl 1"][2])
            if label == "0":
                counts.append(0)
            else:
                found = re.fullmatch(r"(one|two|three): bowl 1 on plate 1", label)
                if not found:
                    raise ValueError(f"Unknown author milestone label: {label}")
                counts.append({"one": 1, "two": 2, "three": 3}[found.group(1)])
        if any(b < a for a, b in zip(counts, counts[1:])):
            raise ValueError("Non-monotonic author milestone count")
        milestones = [i for i in range(1, len(counts)) if counts[i] > counts[i-1]]
        events = trace[-1]["snapshot"]["events"]
        returns = [e["available_at"] for e in events if e["type"] == "returned_to_support"]
        bowl_detections, localized, bottom_error, ious = 0, 0, [], []
        raw_proposals, raw_localized, support_bottom_error = 0, 0, []
        for step, pred in enumerate(predictions):
            gt = boxes[step]["bowl 1"][1]
            target = [gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]]
            for detected in pred["views"]["agentview_image"]:
                accepted = detected["evidence_id"] in trace[step]["signals"]["evidence_ids"]
                if detected["label"] == "plate" and accepted:
                    plate_gt = boxes[step]["plate 1"][1]
                    support_bottom_error.append(abs(detected["xyxy"][3] - (plate_gt[1] + plate_gt[3])) * 256)
                if detected["label"] != "bowl":
                    continue
                iou = box_iou(detected["xyxy"], target)
                raw_proposals += 1
                raw_localized += int(iou >= .5)
                if not accepted:
                    continue
                bowl_detections += 1
                ious.append(iou)
                localized += int(iou >= .5)
                bottom_error.append(abs(detected["xyxy"][3]-target[3])*256)
        matched = match_events(returns, milestones, args.event_tolerance)
        report["episodes"].append(dict(episode=episode["episode"], split=episode["split"],
            annotated_returns=counts[-1], predicted_returns=len(returns), annotated_milestones=milestones,
            predicted_confirmation_frames=returns, event_matching=matched,
            history_complete=episode["history_complete"], grounded_frames=episode["grounded_frames"],
            total_frames=len(trace), memory_frames=episode["memory_frames"],
            raw_bowl_proposals=raw_proposals, raw_proposals_at_iou_0_5=raw_localized,
            accepted_bowl_detections=bowl_detections, accepted_localized_at_iou_0_5=localized,
            mean_bowl_iou=float(np.mean(ious)) if ious else None,
            p95_support_bottom_error_pixels_256=float(np.percentile(support_bottom_error,95)) if support_bottom_error else None,
            mean_bottom_error_pixels_256=float(np.mean(bottom_error)) if bottom_error else None,
            p95_bottom_error_pixels_256=float(np.percentile(bottom_error,95)) if bottom_error else None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
