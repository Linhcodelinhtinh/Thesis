"""Full agent-view class queries for relation memory, using the pinned backend."""
import base64
import io
import json
from dataclasses import replace

import numpy as np

from src.perception.grounding_dino import GroundingDinoDetector, MODEL_ID, REVISION, box_iou
from src.perception.process_detector import ProcessVisualDetector
from src.perception.visual_evidence import Detection2D, VisualEvidence, image_digest


class RelationDetector(GroundingDinoDetector):
    """Query each class separately to avoid prompt-dependent label competition."""

    @property
    def relation_provenance(self):
        return dict(self.lock, inference=dict(
            mode="pilot_crop_for_bowl_plate_full_agent_view_for_other_profiles",
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold, nms_iou=.5, border_rejection_pixels=1))

    def detect(self, step, images, queries, *, pilot_crop=False):
        if pilot_crop:
            if queries != {"bowl": "a metal bowl.", "plate": "a white plate."}:
                raise ValueError("Pilot crop supports the original bowl and plate queries")
            evidence = super().detect(step, images)
            return replace(evidence, metadata={**evidence.metadata, "queries": queries, "pilot_crop": True})
        import torch
        from PIL import Image

        camera = "agentview_image"
        image = images[camera]
        hashes = {name: image_digest(frame) for name, frame in images.items()}
        height, width = image.shape[:2]
        detections = []
        for label, query in queries.items():
            inputs = self.processor(images=Image.fromarray(image), text=query, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=[(height, width)])[0]
            kept = []
            for box, score in sorted(zip(result["boxes"], result["scores"]), key=lambda x: -float(x[1])):
                x1, y1, x2, y2 = box.cpu().numpy()
                if not (1 < x1 < x2 < width - 1 and 1 < y1 < y2 < height - 1):
                    continue
                normalized = np.array([x1, y1, x2, y2]) / [width, height, width, height]
                if any(box_iou(normalized, prior) > .5 for prior in kept):
                    continue
                kept.append(normalized)
                eid = f"rgb:{step}:{camera}:{hashes[camera][:16]}:{len(detections)}"
                detections.append(Detection2D(label, tuple(normalized), float(score), eid))
        views = {name: tuple(detections) if name == camera else () for name in images}
        return VisualEvidence(step, views, hashes, MODEL_ID, REVISION,
                              metadata={"queries": queries, "pilot_crop": False,
                                        "inference": self.relation_provenance["inference"]})


class ProcessRelationDetector(ProcessVisualDetector):
    """Keep the detector in its versioned perception environment online."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self._process.stdin.write(json.dumps({"relation_contract": True}) + "\n")
            self._process.stdin.flush()
            self.provenance = self._receive()["relation_provenance"]
        except BaseException:
            self.close()
            raise

    def detect(self, step, images, queries, *, pilot_crop=False):
        with self._lock:
            if self._closed:
                raise RuntimeError("RGB detector worker is closed")
            buffer = io.BytesIO()
            np.savez(buffer, **images)
            request = dict(step=step, queries=queries, pilot_crop=pilot_crop,
                           rgb_npz=base64.b64encode(buffer.getvalue()).decode("ascii"))
            try:
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
                evidence = VisualEvidence.from_dict(self._receive()["evidence"])
                evidence.validate(step, images)
                if evidence.metadata.get("queries") != queries or evidence.metadata.get("pilot_crop") != pilot_crop:
                    raise ValueError("Detector did not return evidence for the requested classes")
                return evidence
            except BaseException:
                self.close()
                raise
