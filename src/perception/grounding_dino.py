"""Pinned Grounding DINO RGB backend; imports model dependencies only on construction."""
from pathlib import Path
import json

import numpy as np

from src.perception.visual_evidence import Detection2D, VisualEvidence, image_digest, unambiguous_bowl
from src.utils.file_hash import sha256_file

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    area = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    return float(area / (np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - area + 1e-12))


class GroundingDinoDetector:
    """Causal RGB crop detection for T1/T3, no pose extrapolation or GT access.

    ``detect`` accepts only RGB images. No instruction, action, goal, depth,
    segmentation or benchmark metadata can reach the detector through this API.
    """
    def __init__(self, model_path, device="cuda", box_threshold=0.15, text_threshold=0.15):
        import torch
        from importlib.metadata import version
        if version("transformers") != "4.51.3":
            raise RuntimeError("Detector requires its isolated envs/memory_perception.txt environment (Transformers 4.51.3)")
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        path = Path(model_path)
        lock_path = path / "memory_detector_lock.json"
        if not lock_path.exists():
            raise FileNotFoundError("Missing detector lock; run scripts/download_memory_detector.py first")
        self.lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if self.lock["model_id"] != MODEL_ID or self.lock["revision"] != REVISION:
            raise ValueError("Detector revision does not match pinned source")
        pinned = json.loads((Path(__file__).resolve().parents[2] /
                             "resources/manifests/memory_perception_source.json").read_text(encoding="utf-8"))
        if self.lock["files"] != pinned["files_sha256"]:
            raise ValueError("Detector lock hashes differ from the reviewed pinned resources")
        for name, expected in self.lock["files"].items():
            actual = sha256_file(path / name)
            if actual != expected:
                raise ValueError(f"Corrupt detector resource: {name}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA detector but CUDA is unavailable; choose device explicitly")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            path, local_files_only=True, use_safetensors=True, disable_custom_kernels=True,
        ).eval().to(device)
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.reset()

    def reset(self):
        self._agent_roi = None

    @property
    def provenance(self):
        return dict(self.lock, inference=dict(box_threshold=self.box_threshold, text_threshold=self.text_threshold,
            queries={"bowl": "a metal bowl.", "plate": "a white plate."},
            agent_roi="from the first unique RGB bowl; x +/-1.25 widths, y -2.4/+1.5 heights",
            border_rejection_pixels=1, nms_iou=.5, seed_ambiguity_score_margin=.12,
            seed_ambiguity_score_ratio=1.5, wrist="bowl-only auxiliary detection"))

    def _infer(self, image, label, region):
        import torch
        from PIL import Image
        x1, y1, x2, y2 = region
        cropped = image[y1:y2, x1:x2]
        query = "a metal bowl." if label == "bowl" else "a white plate."
        inputs = self.processor(images=Image.fromarray(cropped), text=query, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[cropped.shape[:2]],
        )[0]
        candidates = []
        for box, score, phrase in zip(result["boxes"], result["scores"], result["text_labels"]):
            if label not in phrase.lower().split():
                continue
            box = box.cpu().numpy()
            # A cut-off object cannot supply a reliable lower edge for a return.
            if box[0] <= 1 or box[1] <= 1 or box[2] >= cropped.shape[1]-1 or box[3] >= cropped.shape[0]-1:
                continue
            box = (box + np.array([x1, y1, x1, y1])) / np.array([image.shape[1], image.shape[0]] * 2)
            if box[2] > box[0] and box[3] > box[1]:
                candidates.append((label, box, float(score)))
        selected = []
        for candidate in sorted(candidates, key=lambda value: -value[2]):
            if not any(box_iou(candidate[1], old[1]) > .5 for old in selected):
                selected.append(candidate)
        return selected

    def detect(self, step, images):
        if step == 0:
            self.reset()
        views, hashes, regions = {}, {}, {}
        for camera, image in images.items():
            hashes[camera] = image_digest(image)
            height, width = image.shape[:2]
            full = (0, 0, width, height)
            if camera == "agentview_image":
                if self._agent_roi is None:
                    seeds = self._infer(image, "bowl", full)
                    seed = unambiguous_bowl([Detection2D(label, tuple(box), score, f"seed:{i}")
                                            for i, (label, box, score) in enumerate(seeds)])
                    if seed is not None:
                        box = np.asarray(seed.xyxy) * np.array([width, height] * 2)
                        center, size = (box[:2]+box[2:])/2, box[2:]-box[:2]
                        low = np.maximum(0, np.floor(center - size * [1.25, 2.4])).astype(int)
                        high = np.minimum([width, height], np.ceil(center + size * [1.25, 1.5])).astype(int)
                        self._agent_roi = tuple([*low, *high])
                region = self._agent_roi or full
                candidates = self._infer(image, "bowl", region) + self._infer(image, "plate", region)
            else:
                region = full
                candidates = self._infer(image, "bowl", region)
            regions[camera] = list(map(int, region))
            selected = []
            for label, box, score in candidates:
                eid = f"rgb:{step}:{camera}:{hashes[camera][:16]}:{len(selected)}"
                selected.append(Detection2D(label, tuple(box), score, eid))
            views[camera] = tuple(selected)
        return VisualEvidence(step, views, hashes, MODEL_ID, REVISION,
                              metadata={"regions_xyxy_pixels": regions, "inference": self.provenance["inference"]})
