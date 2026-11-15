"""Audit pinned MiniVLA checkpoint files and known official-runtime contract gaps.

This is an offline provenance/interface audit. It never loads a model, runs an
inference call, starts a simulator, or trains weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "resources/manifests/models/minivla_libero90.yaml"
CHECKPOINT_ROOT = ROOT / "resources/checkpoints/minivla_libero90"
VQ_CHECKPOINT_ROOT = ROOT / "resources/checkpoints/minivla_vq_libero90"
VQ_MANIFEST_PATH = ROOT / "resources/manifests/models/minivla_vq_libero90.yaml"
DEFAULT_OUTPUT = ROOT / "experiments/audits/minivla-contract-audit-20261115/report.json"
LOCAL_SOURCES = [
    "src/models/minivla/adapter.py",
    "src/models/minivla/config.py",
    "src/models/minivla/preprocess.py",
    "src/models/minivla/postprocess.py",
    "src/models/minivla/prismatic_model.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def audit_official_source(
    source: Dict[str, Any], upstream_root: Optional[Path]
) -> Dict[str, Any]:
    files = source["source_files"]
    results: Dict[str, Any] = {}
    checkout_revision = None
    if upstream_root is not None:
        checkout_revision = git_text(upstream_root, "rev-parse", "HEAD")
        if checkout_revision != source["revision"]:
            raise ValueError(
                f"Official source checkout is {checkout_revision}, expected pinned {source['revision']}"
            )

    for relative, expected in files.items():
        row: Dict[str, Any] = {"git_blob_sha1": expected["git_blob_sha1"]}
        if upstream_root is not None:
            actual_blob = git_text(upstream_root, "rev-parse", f"HEAD:{relative}")
            if actual_blob != expected["git_blob_sha1"]:
                raise ValueError(f"Pinned source Git blob mismatch: {relative}")
            source_file = upstream_root / relative
            if source_file.is_file():
                actual_sha256 = sha256_file(source_file)
                expected_sha256 = expected.get("sha256")
                if expected_sha256 and actual_sha256 != expected_sha256:
                    raise ValueError(f"Pinned source SHA-256 mismatch: {relative}")
                row["sha256"] = actual_sha256
                row["file_content_verified"] = True
            else:
                row["sha256"] = expected.get("sha256")
                row["file_content_verified"] = False
                row["note"] = "Blob identity verified from the pinned Git tree; file is absent from sparse checkout."
        else:
            row["sha256"] = expected.get("sha256")
            row["file_content_verified"] = False
        results[relative] = row

    return {
        "repository": source["repository"],
        "revision": source["revision"],
        "checkout_revision": checkout_revision,
        "checkout_revision_verified": checkout_revision == source["revision"] if checkout_revision else False,
        "source_files": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        help="Optional checkout of the exact pinned Stanford-ILIAD/openvla-mini source revision.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    checkpoint_results = []
    for candidate in manifest["checkpoint_selection"]["candidates"]:
        path = CHECKPOINT_ROOT / candidate["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Pinned MiniVLA checkpoint is missing: {path}")
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        expected_hash = candidate["sha256"].lower()
        if actual_hash != expected_hash or actual_size != candidate["size_bytes"]:
            raise ValueError(
                f"Checkpoint identity mismatch for {candidate['path']}: "
                f"sha256={actual_hash}, size={actual_size}"
            )
        checkpoint_results.append(
            {
                "path": candidate["path"],
                "size_bytes": actual_size,
                "sha256": actual_hash,
                "matches_pinned_manifest": True,
            }
        )

    config = json.loads((CHECKPOINT_ROOT / "config.json").read_text(encoding="utf-8"))
    statistics = json.loads((CHECKPOINT_ROOT / "dataset_statistics.json").read_text(encoding="utf-8"))
    vq_manifest = yaml.safe_load(VQ_MANIFEST_PATH.read_text(encoding="utf-8"))
    vq_checkpoint_files = (
        sorted(p.relative_to(ROOT).as_posix() for p in VQ_CHECKPOINT_ROOT.rglob("*.pt"))
        if VQ_CHECKPOINT_ROOT.is_dir()
        else []
    )
    action_stats = statistics["libero_90"]["action"]
    if len(action_stats["q01"]) != 7 or len(action_stats["q99"]) != 7 or len(action_stats["mask"]) != 7:
        raise ValueError("Pinned MiniVLA action statistics do not declare seven aligned dimensions")

    local_source_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in LOCAL_SOURCES
    }
    local_adapter = (ROOT / "src/models/minivla/adapter.py").read_text(encoding="utf-8")
    local_config = (ROOT / "src/models/minivla/config.py").read_text(encoding="utf-8")
    local_preprocess = (ROOT / "src/models/minivla/preprocess.py").read_text(encoding="utf-8")
    local_postprocess = (ROOT / "src/models/minivla/postprocess.py").read_text(encoding="utf-8")
    local_model = (ROOT / "src/models/minivla/prismatic_model.py").read_text(encoding="utf-8")

    local_mismatches = {
        "mtime_checkpoint_autoselection_present": "st_mtime" in local_adapter,
        "ambiguous_checkpoint_rejection_present": "No mtime-based selection is permitted" in local_adapter,
        "partial_state_loading_present": "strict=False" in local_model,
        "vertical_flip_missing": "flipud" not in local_preprocess,
        "lanczos_resize_missing": "LANCZOS" not in local_preprocess,
        "bilinear_resize_present": "Image.BILINEAR" in local_preprocess,
        "custom_prompt_present": "In: What action should the robot take to" in local_preprocess,
        "hardcoded_direct_gripper_default_present": 'gripper_polarity = "DIRECT"' in local_config,
        "only_sign_inversion_in_postprocessor": "final_actions[..., -1] = -final_actions[..., -1]" in local_postprocess
        and "np.sign" not in local_postprocess
        and "2 * (" not in local_postprocess,
    }
    if "MINIVLA_OFFICIAL_RUNTIME_VERIFIED = False" not in local_adapter:
        raise ValueError("MiniVLA adapter no longer has its fail-closed evaluation gate")

    upstream = audit_official_source(manifest["official_code_audit"], args.upstream_root)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "OFFLINE_MODEL_PROVENANCE_AND_INTERFACE_AUDIT",
        "evaluation_status": manifest["evaluation_status"],
        "policy_rollouts": 0,
        "model_inference": False,
        "training_performed": False,
        "repository": {
            "git_head": git_text(ROOT, "rev-parse", "HEAD"),
            "working_tree": "The report and its inputs are uncommitted workspace artifacts unless committed by the user.",
        },
        "manifest": {
            "path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(MANIFEST_PATH),
        },
        "checkpoint": {
            "model_id": manifest["model_id"],
            "revision": manifest["revision"],
            "config_run_id": config.get("run_id"),
            "official_config_resume_pointer": manifest["checkpoint_selection"]["official_config_resume_pointer"],
            "selection_status": manifest["checkpoint_selection"]["status"],
            "selection_rule": manifest["checkpoint_selection"]["selection_rule"],
            "candidates": checkpoint_results,
            "config_action_tokenizer": config.get("vla", {}).get("action_tokenizer"),
            "config_data_mix": config.get("vla", {}).get("data_mix"),
            "statistics": {
                "num_trajectories": statistics["libero_90"].get("num_trajectories"),
                "num_transitions": statistics["libero_90"].get("num_transitions"),
                "action_dim": len(action_stats["q01"]),
                "q01": action_stats["q01"],
                "q99": action_stats["q99"],
                "mask": action_stats["mask"],
            },
        },
        "official_inference_contract": manifest["official_inference_contract"],
        "official_source": upstream,
        "vq_candidate": {
            "model_id": vq_manifest["model_id"],
            "revision": vq_manifest["revision"],
            "evaluation_status": vq_manifest["evaluation_status"],
            "checkpoint_directory_present": VQ_CHECKPOINT_ROOT.is_dir(),
            "local_pt_files": vq_checkpoint_files,
            "codebook": vq_manifest["vq_codebook"],
        },
        "local_adapter": {
            "source_sha256": local_source_hashes,
            "known_mismatches": local_mismatches,
            "fail_closed_gate_present": True,
        },
        "finding": (
            "Checkpoint bytes match both pinned Hub candidates, but no evaluation weight is selected. "
            "The existing local implementation does not reproduce the pinned Prismatic loader, prompt, image, "
            "action-statistics and LIBERO gripper path. MiniVLA is blocked from benchmark execution."
        ),
        "limitations": [
            "Checkpoint selection remains unresolved; the config.json pointer is a resume pointer, not an evaluation selection record.",
            "The pinned source commit and Git tree blobs are verified; a sparse checkout may not contain every official source file for SHA-256 recomputation.",
            "The official runtime has not been installed or executed in envs/minivla.yml.",
            "The MiniVLA-VQ candidate has no local checkpoint/codebook and requires a separate audit.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": args.output.as_posix(), "evaluation_status": report["evaluation_status"],
                      "checkpoints_verified": len(checkpoint_results), "policy_rollouts": 0}, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
