"""Configuration Dataclass for MiniVLA Models (Phase 9)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import yaml


@dataclass
class MiniVLAConfig:
    """Configuration for loading and executing MiniVLA models."""

    model_id: str = "Stanford-ILIAD/minivla-libero90-prismatic"
    checkpoint_path: Optional[str] = None
    action_tokenizer_type: str = "extra_action_tokenizer"  # or "libero_vq_extra_action_tokenizer"
    action_dim: int = 7
    chunk_size: int = 1  # 1 for non-VQ, H for VQ
    n_action_bins: int = 256
    image_resolution: tuple = (224, 224)
    device: str = "cpu"
    dtype: str = "bfloat16"  # or float32, float16
    dataset_statistics: Optional[Dict[str, Any]] = None
    gripper_polarity: str = "DIRECT"  # or "INVERTED", determined by stats audit
    extra_tokens_offset: int = 151665

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: Union[str, Path], device: str = "cpu") -> "MiniVLAConfig":
        """Load configuration from a local checkpoint directory."""
        path = Path(checkpoint_dir)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")

        config_json = path / "config.json"
        config_yaml = path / "config.yaml"
        dataset_stats_file = path / "dataset_statistics.json"

        cfg_dict: Dict[str, Any] = {}
        if config_json.exists():
            with open(config_json, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
        elif config_yaml.exists():
            with open(config_yaml, "r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f)

        vla_cfg = cfg_dict.get("vla", {})
        action_tok = vla_cfg.get("action_tokenizer", "extra_action_tokenizer")
        is_vq = "vq" in action_tok.lower()

        # Load dataset statistics if available
        stats = None
        if dataset_stats_file.exists():
            with open(dataset_stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)

        # Determine gripper polarity from statistics
        # In LIBERO-90, open gripper typically corresponds to action[-1] > 0 or 1
        gripper_polarity = "DIRECT"

        return cls(
            model_id=cfg_dict.get("run_id", "Stanford-ILIAD/minivla-libero90-prismatic"),
            checkpoint_path=str(path),
            action_tokenizer_type=action_tok,
            action_dim=7,
            chunk_size=10 if is_vq else 1,
            device=device,
            dataset_statistics=stats,
            gripper_polarity=gripper_polarity,
        )
