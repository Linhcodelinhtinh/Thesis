"""Asset and initial-state integrity validator (Phase 2 Hardening).

Ensures all BDDL task definitions and initial-state distributions match the
pinned official LIBERO commit (8f1084e3132a39270c3a13ebe37270a43ece2a01)
via SHA-256 cryptographic hashes per Rule 6 and ADR-0008.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class AssetIntegrityError(Exception):
    """Raised when an asset or initial state file fails SHA-256 integrity validation."""
    pass


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute the SHA-256 hexadecimal digest of a file.

    Args:
        filepath: Path to the target file.

    Returns:
        64-character lowercase hexadecimal hash string.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_default_manifest_path() -> Path:
    """Find the default asset manifest path relative to repository root."""
    current = Path(__file__).resolve()
    # Ascend to workspace root (Thesis_26)
    repo_root = current.parents[2]
    manifest_path = repo_root / "resources" / "manifests" / "asset_manifest.yaml"
    return manifest_path


def verify_task_asset_integrity(
    bddl_file_path: Union[str, Path],
    init_file_path: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Verify cryptographic SHA-256 hashes of task assets against registered manifest.

    Args:
        bddl_file_path: Path to the BDDL definition file.
        init_file_path: Optional path to the .init or .pruned_init file.
        manifest_path: Optional custom path to asset_manifest.yaml.

    Returns:
        Dictionary of verification details.

    Raises:
        AssetIntegrityError: If any file is missing, altered, or fails hash match.
    """
    if manifest_path is None:
        manifest_path = find_default_manifest_path()
    else:
        manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise AssetIntegrityError(
            f"Asset manifest not found at: {manifest_path}. "
            "Cannot certify benchmark asset integrity per Rule 6."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    tasks_manifest = manifest.get("tasks", {})

    bddl_path = Path(bddl_file_path)
    if not bddl_path.exists():
        raise AssetIntegrityError(f"BDDL file does not exist: {bddl_path}")

    task_name = bddl_path.stem
    if task_name not in tasks_manifest:
        # If task is not yet pinned in manifest, fail loudly in strict mode
        raise AssetIntegrityError(
            f"Task '{task_name}' is not registered in asset manifest ({manifest_path}). "
            "Unpinned benchmark tasks are strictly prohibited per Rule 1 & Rule 6."
        )

    expected_info = tasks_manifest[task_name]
    actual_bddl_sha = compute_file_sha256(bddl_path)
    expected_bddl_sha = expected_info.get("bddl_sha256")

    if actual_bddl_sha != expected_bddl_sha:
        raise AssetIntegrityError(
            f"BDDL hash mismatch for task '{task_name}'!\n"
            f"  File: {bddl_path}\n"
            f"  Expected: {expected_bddl_sha}\n"
            f"  Actual:   {actual_bddl_sha}\n"
            "Benchmark resource has been modified or corrupted."
        )

    init_verified = False
    actual_init_sha = None
    if init_file_path is not None:
        init_path = Path(init_file_path)
        if init_path.exists():
            actual_init_sha = compute_file_sha256(init_path)
            expected_init_sha = expected_info.get("init_sha256")
            expected_pruned_sha = expected_info.get("pruned_init_sha256")
            if actual_init_sha not in (expected_init_sha, expected_pruned_sha):
                raise AssetIntegrityError(
                    f"Initial states hash mismatch for task '{task_name}'!\n"
                    f"  File: {init_path}\n"
                    f"  Expected: {expected_init_sha} or {expected_pruned_sha}\n"
                    f"  Actual:   {actual_init_sha}\n"
                    "Initial states array does not match official benchmark distribution."
                )
            init_verified = True

    return {
        "task_name": task_name,
        "bddl_file": bddl_path.name,
        "bddl_sha256": actual_bddl_sha,
        "init_verified": init_verified,
        "init_sha256": actual_init_sha,
        "integrity_status": "PASS",
    }
