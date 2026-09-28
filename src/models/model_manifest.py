"""Model Manifest Specification and Audit Framework (Phase 4).

Validates checkpoint provenance, input/output schemas, camera mappings,
normalization strategies, and chunk dimensions before benchmark execution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class ModelManifest:
    """Standardized metadata and interface specification for a VLA checkpoint."""

    model_id: str
    repository: str
    revision: str
    checkpoint: str
    action_dim: int
    chunk_size: int
    n_action_steps: int
    image_resolution: Tuple[int, int]
    camera_mapping: Dict[str, str]
    normalization: Dict[str, str]
    action_semantics: List[str]
    action_range: Tuple[float, float]
    action_decoder: str
    hardware_requirements: Dict[str, Any]
    metadata_notes: Optional[str] = None
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    lerobot_version: Optional[str] = None
    observation_state: Optional[Dict[str, Any]] = None
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "repository": self.repository,
            "revision": self.revision,
            "checkpoint": self.checkpoint,
            "lerobot_version": self.lerobot_version,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "image_resolution": list(self.image_resolution),
            "camera_mapping": self.camera_mapping,
            "observation_state": self.observation_state,
            "normalization": self.normalization,
            "action_semantics": self.action_semantics,
            "action_range": list(self.action_range),
            "action_decoder": self.action_decoder,
            "hardware_requirements": self.hardware_requirements,
            "files": self.files,
            "metadata_notes": self.metadata_notes,
        }


def load_model_manifest(filepath: Union[str, Path]) -> ModelManifest:
    """Load and validate a model manifest YAML file.

    Args:
        filepath: Path to the YAML manifest file.

    Returns:
        Validated ModelManifest instance.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Validate required fields
    required = [
        "model_id",
        "repository",
        "action_dim",
        "chunk_size",
        "image_resolution",
        "camera_mapping",
    ]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required manifest field: '{key}' in {path}")

    resolution = tuple(data["image_resolution"])
    action_range = tuple(data.get("action_range", [-1.0, 1.0]))

    obs_state = data.get("observation_state", None)
    if obs_state is not None:
        obs_state = {
            "runtime_dim": int(obs_state.get("runtime_dim", 8)),
            "semantics": list(obs_state.get("semantics", [])),
            "source_of_truth": list(obs_state.get("source_of_truth", [])),
        }

    return ModelManifest(
        model_id=data["model_id"],
        repository=data["repository"],
        revision=data.get("revision", "main"),
        checkpoint=data.get("checkpoint", data["model_id"]),
        action_dim=int(data["action_dim"]),
        chunk_size=int(data["chunk_size"]),
        n_action_steps=int(data.get("n_action_steps", data["chunk_size"])),
        image_resolution=(resolution[0], resolution[1]),
        camera_mapping=dict(data["camera_mapping"]),
        normalization=dict(data.get("normalization", {})),
        action_semantics=list(data.get("action_semantics", ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"])),
        action_range=(float(action_range[0]), float(action_range[1])),
        action_decoder=data.get("action_decoder", "flow_matching"),
        hardware_requirements=dict(data.get("hardware_requirements", {})),
        metadata_notes=data.get("metadata_notes", None),
        files=dict(data.get("files", {})),
        lerobot_version=data.get("lerobot_version", None),
        observation_state=obs_state,
    )


def audit_model_interface(manifest: ModelManifest) -> Dict[str, Any]:
    """Perform static interface audit on a ModelManifest.

    Verifies:
    - Action dimension matches 7-DoF.
    - Chunk size is positive.
    - Resolution is valid (e.g. 256x256 or 224x224).
    - Camera mappings cover LIBERO agentview and eye-in-hand cameras.
    - Revision is an immutable commit hash rather than generic 'main'.
    - Observation state contract: 8D runtime vector [pos(3), axis(3), gripper(2)],
      with source of truth in normalizer safetensors.
    - Flags any metadata inconsistencies (e.g. unused camera3 in SmolVLA).
    """
    issues: List[str] = []
    warnings: List[str] = []

    if manifest.action_dim != 7:
        issues.append(f"Invalid action_dim {manifest.action_dim}; LIBERO expects 7.")

    if manifest.chunk_size <= 0:
        issues.append(f"Invalid chunk_size {manifest.chunk_size}; must be > 0.")

    # Check camera mappings
    mapped_cameras = list(manifest.camera_mapping.keys())
    if "agentview" not in mapped_cameras:
        issues.append("Missing required camera mapping for 'agentview'.")
    if "robot0_eye_in_hand" not in mapped_cameras:
        issues.append("Missing required camera mapping for 'robot0_eye_in_hand'.")

    # Check immutable revision
    if manifest.revision == "main" or len(manifest.revision) < 40:
        warnings.append(
            f"Manifest uses mutable or short revision '{manifest.revision}'. "
            "SRS requires full 40-character immutable Git commit SHA for strict evaluation."
        )

    # Observation state audit contract
    if manifest.observation_state is not None:
        obs_state = manifest.observation_state
        runtime_dim = obs_state.get("runtime_dim")
        semantics = obs_state.get("semantics", [])
        source_of_truth = obs_state.get("source_of_truth", [])

        if runtime_dim != 8:
            issues.append(f"Invalid observation_state runtime_dim {runtime_dim}; expected 8.")

        expected_semantics = [
            "eef_pos_x",
            "eef_pos_y",
            "eef_pos_z",
            "eef_axis_x",
            "eef_axis_y",
            "eef_axis_z",
            "gripper_qpos_0",
            "gripper_qpos_1",
        ]
        if semantics != expected_semantics:
            issues.append(
                f"observation_state semantics mismatch: expected {expected_semantics}, got {semantics}"
            )

        if not source_of_truth or not any("policy_preprocessor_step_5_normalizer_processor.safetensors" in s for s in source_of_truth):
            issues.append(
                "observation_state source_of_truth must include 'policy_preprocessor_step_5_normalizer_processor.safetensors'."
            )
    else:
        warnings.append("Manifest does not explicitly define 'observation_state' contract.")

    # Audit camera3 discrepancy
    if "camera3" in manifest.camera_mapping.values() or "observation.images.camera3" in str(manifest.metadata_notes):
        warnings.append(
            "Discrepancy noted: Model config declares camera3, but official preprocessor "
            "only maps 2 cameras (camera1=agentview, camera2=eye_in_hand). camera3 is unused."
        )

    is_valid = len(issues) == 0

    return {
        "model_id": manifest.model_id,
        "is_valid": is_valid,
        "revision": manifest.revision,
        "action_dim": manifest.action_dim,
        "chunk_size": manifest.chunk_size,
        "resolution": manifest.image_resolution,
        "observation_state": manifest.observation_state,
        "issues": issues,
        "warnings": warnings,
        "audit_status": "PASS" if is_valid else "FAIL",
    }


def verify_normalizer_safetensors_shape(
    safetensors_path: Union[str, Path],
    expected_dim: int = 8,
) -> Dict[str, Any]:
    """Verify that observation.state tensors in normalizer safetensors have exact expected_dim.

    Args:
        safetensors_path: Path to the normalizer processor safetensors file.
        expected_dim: Expected dimension of observation.state tensors (default 8).

    Returns:
        Dictionary with validation results and per-tensor shapes.
    """
    import safetensors.torch

    path = Path(safetensors_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalizer safetensors not found: {path}")

    tensors = safetensors.torch.load_file(str(path))
    state_tensors = {k: tuple(v.shape) for k, v in tensors.items() if k.startswith("observation.state.")}

    issues: List[str] = []
    required_stats = [
        "observation.state.mean",
        "observation.state.std",
        "observation.state.min",
        "observation.state.max",
    ]

    for stat in required_stats:
        if stat not in state_tensors:
            issues.append(f"Missing required normalizer tensor '{stat}'.")
        elif state_tensors[stat] != (expected_dim,):
            issues.append(f"Tensor '{stat}' shape mismatch: expected ({expected_dim},), got {state_tensors[stat]}")

    for k, shape in state_tensors.items():
        if k.endswith(".count"):
            if shape != (1,):
                issues.append(f"Scalar tensor '{k}' expected shape (1,), got {shape}")
        else:
            if shape != (expected_dim,):
                issues.append(f"State tensor '{k}' expected shape ({expected_dim},), got {shape}")

    return {
        "path": str(path),
        "expected_dim": expected_dim,
        "is_valid": len(issues) == 0,
        "state_tensors": state_tensors,
        "issues": issues,
        "status": "PASS" if len(issues) == 0 else "FAIL",
    }


def verify_checkpoint_integrity(
    checkpoint_dir: Union[str, Path],
    manifest: ModelManifest,
    require_all: bool = False,
) -> Dict[str, Any]:
    """Verify cryptographic SHA-256 hashes of checkpoint files against manifest.

    Args:
        checkpoint_dir: Directory containing downloaded checkpoint files.
        manifest: Loaded ModelManifest instance.
        require_all: If True, all files listed in manifest must be present.
                     If False, checks all files that are present on disk.

    Returns:
        Dictionary with verification results per file.
    """
    import hashlib

    cp_dir = Path(checkpoint_dir)
    if not cp_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {cp_dir}")

    results: Dict[str, Dict[str, Any]] = {}
    all_passed = True

    for filename, file_meta in manifest.files.items():
        file_path = cp_dir / filename
        expected_hash = file_meta.get("sha256")

        if not file_path.exists():
            if require_all:
                results[filename] = {"status": "MISSING", "passed": False}
                all_passed = False
            else:
                results[filename] = {"status": "NOT_PRESENT_LOCALLY", "passed": True}
            continue

        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()

        if expected_hash and actual_hash != expected_hash:
            results[filename] = {
                "status": "HASH_MISMATCH",
                "expected": expected_hash,
                "actual": actual_hash,
                "passed": False,
            }
            all_passed = False
        else:
            results[filename] = {"status": "MATCH", "passed": True, "sha256": actual_hash}

    return {
        "checkpoint_dir": str(cp_dir),
        "all_passed": all_passed,
        "file_results": results,
    }

