"""Download the pinned public detector into a dedicated local directory and hash it."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.perception.grounding_dino import MODEL_ID, REVISION
from src.utils.file_hash import sha256_file


def main():
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="resources/models/grounding-dino-tiny")
    args = parser.parse_args()
    path = Path(snapshot_download(MODEL_ID, revision=REVISION, local_dir=args.output,
                                 allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md"]))
    files = {}
    for file in sorted(path.iterdir()):
        if file.is_file() and file.name != "memory_detector_lock.json":
            files[file.name] = sha256_file(file)
    required = {"config.json", "model.safetensors", "preprocessor_config.json", "tokenizer_config.json"}
    if not required <= files.keys():
        raise FileNotFoundError(f"Incomplete pinned detector: missing {required - files.keys()}")
    lock = dict(model_id=MODEL_ID, revision=REVISION, files=files)
    (path / "memory_detector_lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
