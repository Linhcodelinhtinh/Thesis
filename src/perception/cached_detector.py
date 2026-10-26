"""Replay previously predicted detections, bound to exact RGB frames and hashes."""
import json
from pathlib import Path

from src.perception.grounding_dino import MODEL_ID, REVISION
from src.perception.visual_evidence import VisualEvidence
from src.utils.file_hash import sha256_file


class CachedVisualDetector:
    def __init__(self, path, expected_sha256):
        self.path = Path(path)
        if sha256_file(self.path) != expected_sha256:
            raise ValueError("Corrupt prediction cache")
        self._file = self.path.open(encoding="utf-8")

    def detect(self, step, images):
        line = self._file.readline()
        if not line:
            raise ValueError("Prediction cache ended before the demonstration")
        evidence = VisualEvidence.from_dict(json.loads(line))
        if evidence.model_id != MODEL_ID or evidence.revision != REVISION:
            raise ValueError("Cache is not from the pinned RGB detector")
        evidence.validate(step, images)
        return evidence

    def close(self):
        self._file.close()
