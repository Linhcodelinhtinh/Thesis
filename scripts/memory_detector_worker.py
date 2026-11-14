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
from src.perception.relation_detector import RelationDetector
from src.perception.visual_evidence import rgb_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        detector = RelationDetector(args.model_dir, args.device)
        orig_proc_call = detector.processor.__call__
        def safe_proc_call(*p_args, **p_kwargs):
            p_kwargs.setdefault("padding", True)
            return orig_proc_call(*p_args, **p_kwargs)
        detector.processor.__call__ = safe_proc_call
        print(json.dumps(dict(status="ready", provenance=detector.provenance)), flush=True)
        step_idx = 0
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("relation_contract") is True:
                print(json.dumps(dict(relation_provenance=detector.relation_provenance)), flush=True)
                continue
            with np.load(io.BytesIO(base64.b64decode(request["rgb_npz"], validate=True)), allow_pickle=False) as arrays:
                images = rgb_views({key: arrays[key] for key in arrays.files}, "libero_mem_refined")
            evidence = (detector.detect(request["step"], images, request["queries"],
                                        pilot_crop=request["pilot_crop"]) if "queries" in request
                        else GroundingDinoDetector.detect(detector, request["step"], images))
            print(json.dumps(dict(evidence=evidence.to_dict())), flush=True)
            step_idx += 1
            if step_idx % 20 == 0 or request.get("step") == 0:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except Exception as error:
        print(json.dumps(dict(error=f"{type(error).__name__}: {error}")), flush=True)
        raise


if __name__ == "__main__":
    main()
