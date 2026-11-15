#!/usr/bin/env python
"""Audit T7 RGB detector boxes against the released author exo-box annotations.

This is a descriptive, single-demonstration offline audit. It does not score
relation tracking, policy behavior, or the LIBERO-Mem benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.perception.visual_evidence import image_digest
from src.utils.file_hash import sha256_file


DEFAULT_INDEX = ROOT / "experiments/audits/relation-e1-20261114-t7-demo1"
DEFAULT_METAINFO = ROOT / "data/libero_mem/metainfo.json"
DEFAULT_SOURCE = ROOT / "data/libero_mem/KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate_demo.hdf5"
DEFAULT_OUTPUT = ROOT / "experiments/audits/t7-detector-box-e1-20261115/report.json"
TASK = "KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate"
DEMO = "demo_1"
IOU_THRESHOLD = 0.5


def _box_xywh_to_xyxy(box):
    if len(box) != 4:
        raise ValueError("Author exo-box annotation must be normalized xywh")
    x, y, w, h = (float(value) for value in box)
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1
            and x + w <= 1.000001 and y + h <= 1.000001):
        raise ValueError(f"Invalid normalized xywh annotation: {box}")
    return (x, y, x + w, y + h)


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _best_matches(gt_boxes, pred_boxes, threshold):
    """Maximum-cardinality one-to-one matching, then maximum total IoU."""
    matrix = [[_iou(g, p) for p in pred_boxes] for g in gt_boxes]
    best_pairs, best_score = (), -1.0

    def visit(gt_idx, used, pairs, score):
        nonlocal best_pairs, best_score
        if gt_idx == len(gt_boxes):
            if (len(pairs), score) > (len(best_pairs), best_score):
                best_pairs, best_score = tuple(pairs), score
            return
        visit(gt_idx + 1, used, pairs, score)
        for pred_idx, overlap in enumerate(matrix[gt_idx]):
            if pred_idx not in used and overlap >= threshold:
                visit(gt_idx + 1, used | {pred_idx}, pairs + [(gt_idx, pred_idx)], score + overlap)

    visit(0, set(), [], 0.0)
    overlaps = [matrix[g][p] for g, p in best_pairs]
    return best_pairs, overlaps


def _stream_demo_labels(metainfo_path):
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError("Install the repository's research-analysis dependency ijson") from exc
    with metainfo_path.open("rb") as stream:
        for demo, row in ijson.kvitems(stream, TASK, use_float=True):
            if demo == DEMO:
                return row
    raise ValueError(f"Missing author annotation for {TASK}/{DEMO}")


def _score_frames(gt_frames, pred_frames):
    totals = {label: {name: 0 for name in ("ground_truth", "predicted", "tp", "fp", "fn")}
              for label in ("bowl", "plate")}
    matched_ious = {label: [] for label in ("bowl", "plate")}
    for gt_frame, pred_frame in zip(gt_frames, pred_frames):
        for label in ("bowl", "plate"):
            gt, pred = gt_frame[label], pred_frame[label]
            pairs, overlaps = _best_matches(gt, pred, IOU_THRESHOLD)
            counts = totals[label]
            counts["ground_truth"] += len(gt)
            counts["predicted"] += len(pred)
            counts["tp"] += len(pairs)
            counts["fp"] += len(pred) - len(pairs)
            counts["fn"] += len(gt) - len(pairs)
            matched_ious[label].extend(overlaps)

    def metrics(counts, overlaps):
        precision = counts["tp"] / counts["predicted"] if counts["predicted"] else None
        recall = counts["tp"] / counts["ground_truth"] if counts["ground_truth"] else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else 0.0)
        return dict(**counts, precision=precision, recall=recall, f1=f1,
                    mean_matched_iou=(sum(overlaps) / len(overlaps) if overlaps else None))

    per_class = {label: metrics(totals[label], matched_ious[label]) for label in totals}
    overall_counts = {name: sum(per_class[label][name] for label in per_class)
                      for name in ("ground_truth", "predicted", "tp", "fp", "fn")}
    overall_iou = [iou for values in matched_ious.values() for iou in values]
    return {"by_class": per_class, "overall": metrics(overall_counts, overall_iou)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metainfo", type=Path, default=DEFAULT_METAINFO)
    parser.add_argument("--source-hdf5", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true",
                        help="Replace only the report at --output after a complete new audit")
    args = parser.parse_args()
    if args.output.exists() and not args.replace:
        parser.error(f"Choose a new output path; refusing to overwrite {args.output}")

    manifest_path = args.index / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filename = args.source_hdf5.name
    if (manifest.get("status") != "complete" or manifest.get("writer") != "rgb_relations_v1"
            or manifest.get("source_kind") != "author_refined" or len(manifest.get("episodes", [])) != 1
            or manifest["episodes"][0].get("frames") != 425
            or manifest["episodes"][0].get("id") != f"{TASK}_{DEMO}"):
        raise ValueError("Expected the complete, single-episode T7 demo_1 RGB writer audit")
    source_digest = sha256_file(args.source_hdf5)
    metainfo_digest = sha256_file(args.metainfo)
    if manifest["sources"].get(filename) != source_digest:
        raise ValueError("Local refined HDF5 differs from the writer index source hash")
    if manifest["sources"].get("metainfo.json") != metainfo_digest:
        raise ValueError("Author exo-box annotation source differs from the writer index source hash")

    input_hashes = {"manifest.json": sha256_file(manifest_path)}
    for name in ("samples.jsonl", "memory_trace.jsonl", "detections.jsonl"):
        path = args.index / name
        actual = sha256_file(path)
        if actual != manifest["artifacts_sha256"].get(name):
            raise ValueError(f"Writer artifact checksum mismatch: {name}")
        input_hashes[name] = actual
    input_hashes["metainfo.json"] = metainfo_digest
    input_hashes[filename] = source_digest

    labels = _stream_demo_labels(args.metainfo)
    gt_frames, pred_frames = [], []
    detector_config = None
    with h5py.File(args.source_hdf5, "r") as source:
        rgb = source[f"data/{DEMO}/obs/agentview_rgb"]
        if len(rgb) != manifest["episodes"][0]["frames"] or len(labels.get("exo_boxes", [])) != len(rgb):
            raise ValueError("Frame count differs among RGB source, annotations, and writer index")
        with (args.index / "detections.jsonl").open(encoding="utf-8") as detection_stream:
            for step, line in enumerate(detection_stream):
                row = json.loads(line)
                evidence = row.get("evidence", {})
                if row.get("episode_id") != f"{TASK}_{DEMO}" or evidence.get("step") != step:
                    raise ValueError("Detection rows are not one contiguous episode")
                current_config = {
                    "model_id": evidence.get("model_id"),
                    "revision": evidence.get("revision"),
                    "queries": evidence.get("metadata", {}).get("queries"),
                    "inference": evidence.get("metadata", {}).get("inference"),
                }
                if detector_config is None:
                    detector_config = current_config
                elif current_config != detector_config:
                    raise ValueError("Detector identity or inference settings changed within the episode")
                frame = rgb[step]
                if evidence.get("image_hashes", {}).get("agentview_image") != image_digest(frame):
                    raise ValueError(f"Source RGB frame hash mismatch at step {step}")
                truth = {"bowl": [], "plate": []}
                for name, annotation in labels["exo_boxes"][step].items():
                    if name.startswith("bowl "):
                        truth["bowl"].append(_box_xywh_to_xyxy(annotation[1]))
                    elif name.startswith("plate "):
                        truth["plate"].append(_box_xywh_to_xyxy(annotation[1]))
                predicted = {"bowl": [], "plate": []}
                view = evidence.get("views", {}).get("agentview_image", [])
                for detection in view:
                    if detection.get("label") not in predicted:
                        raise ValueError(f"Unexpected detector class: {detection.get('label')}")
                    predicted[detection["label"]].append(tuple(float(x) for x in detection["xyxy"]))
                gt_frames.append(truth)
                pred_frames.append(predicted)
    if len(gt_frames) != 425:
        raise ValueError(f"Expected 425 contiguous detections, found {len(gt_frames)}")

    metrics = _score_frames(gt_frames, pred_frames)
    early = _score_frames(gt_frames[:20], pred_frames[:20])
    annotation_tags = {}
    for frame in labels["exo_boxes"]:
        for name in ("bowl 1", "bowl 2"):
            if name in frame:
                tag = str(frame[name][2])
                annotation_tags[tag] = annotation_tags.get(tag, 0) + 1

    report = {
        "classification": "CUSTOM_OFFLINE_DETECTOR_AUDIT_AGAINST_AUTHOR_EXO_BOXES",
        "scope": "One author-refined T7 demonstration; per-frame 2D detection only, not relation accuracy or policy score.",
        "task": TASK,
        "demo": DEMO,
        "split": manifest["episodes"][0].get("split"),
        "frames": len(gt_frames),
        "camera": "agentview_image",
        "detector": detector_config,
        "matching": {"annotation_format": "normalized_xywh", "prediction_format": "normalized_xyxy",
                     "iou_threshold": IOU_THRESHOLD,
                     "assignment": "maximum-cardinality one-to-one, then maximum total IoU"},
        "input_sha256": input_hashes,
        "source_frame_hashes_verified": len(gt_frames),
        "all_frames": metrics,
        "first_20_frames": early,
        "author_exo_box_tag_counts_for_bowls": annotation_tags,
        "limitations": [
            "Frames within one demonstration are temporally dependent; counts are descriptive, not independent samples.",
            "Author exo-box labels are used as released and are not a second blind human annotation.",
            "The third exo-box field is preserved as annotation metadata; this report does not interpret it as relation truth.",
            "Detection quality does not establish identity tracking, relation accuracy, memory utility, or task success.",
        ],
        "audit_code_sha256": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.replace else "x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
