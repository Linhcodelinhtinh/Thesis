"""Shared writer construction and exact prompt tokenizer for offline/online use."""
from functools import partial
from pathlib import Path

from src.memory.event_writer import LiberoMemEventWriter
from src.perception.object_tracker import CausalObjectTracker, TrackerConfig
from src.utils.file_hash import sha256_file

TOKENIZER_SHA256 = "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a"
TOKENIZER_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TOKENIZER_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def load_prompt_tokenizer(path):
    from tokenizers import Tokenizer
    path = Path(path)
    if path.is_dir():
        path = path / "tokenizer.json"
    actual = sha256_file(path)
    if actual != TOKENIZER_SHA256:
        raise ValueError("Prompt tokenizer differs from the pinned SmolVLA tokenizer")
    return Tokenizer.from_file(str(path))


def make_visual_writer(episode_id, detector, image_convention, config=None, **entities):
    if entities.get("target_semantic_name", "bowl") != "bowl" or entities.get("support_semantic_name", "plate") != "plate":
        raise ValueError("RGB writer v3 currently supports only the T1/T3 bowl/original-plate profile")
    cfg = config or TrackerConfig()
    tracker = CausalObjectTracker(detector=detector, image_convention=image_convention, config=cfg, **entities)
    return LiberoMemEventWriter(episode_id, tracker=tracker, dwell_steps=cfg.dwell_steps, **entities)


def visual_writer_factory(detector, image_convention, config=None):
    return partial(make_visual_writer, detector=detector, image_convention=image_convention, config=config)
