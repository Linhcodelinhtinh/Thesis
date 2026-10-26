"""Private JSON-lines worker for ProcessVisualDetector, run in .venv_memory."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.perception.grounding_dino import GroundingDinoDetector
from src.perception.visual_evidence import rgb_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        detector = GroundingDinoDetector(args.model_dir, args.device)
        print(json.dumps(dict(status="ready", provenance=detector.provenance)), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            with np.load(io.BytesIO(base64.b64decode(request["rgb_npz"], validate=True)), allow_pickle=False) as arrays:
                images = rgb_views({key: arrays[key] for key in arrays.files}, "libero_mem_refined")
            evidence = detector.detect(request["step"], images)
            print(json.dumps(dict(evidence=evidence.to_dict())), flush=True)
    except Exception as error:
        print(json.dumps(dict(error=f"{type(error).__name__}: {error}")), flush=True)
        raise


if __name__ == "__main__":
    main()
