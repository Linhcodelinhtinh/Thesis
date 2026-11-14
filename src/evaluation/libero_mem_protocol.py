"""Pinned T1/T3 protocol; source semantics and custom rollout choices are separate."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs/benchmarks/libero_mem_t1_t3_protocol.json"
SUITE_SOURCE_LOCK = ROOT / "configs/benchmarks/libero_mem_suite_source_v1.json"


def source_digest(path):
    # Portable across Windows CRLF and Linux LF checkouts.
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def load_protocol(path=DEFAULT_PROTOCOL, source_root=None):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "libero_mem_t1_t3_v1" or protocol["official_paper_comparable"]:
        raise ValueError("Unknown or mislabeled LIBERO-Mem protocol")
    if source_root is not None:
        for name, expected in protocol["source_text_sha256"].items():
            if source_digest(Path(source_root) / name) != expected:
                raise ValueError("Pinned upstream source differs: " + name)
    return protocol


def load_suite_source(source_root):
    """Verify the ten-task source inventory without enabling unverified rollouts."""
    suite = json.loads(SUITE_SOURCE_LOCK.read_text(encoding="utf-8"))
    if (suite["stage"] != "source_inventory" or suite["official_paper_comparable"]
            or [task["index"] for task in suite["tasks"]] != list(range(10))):
        raise ValueError("Invalid LIBERO-Mem suite source lock")
    for name, expected in suite["source_text_sha256"].items():
        if source_digest(Path(source_root) / name) != expected:
            raise ValueError("Pinned upstream source differs: " + name)
    return suite


def task_spec(protocol, task):
    matches = [s for s in protocol["tasks"] if task in (s["alias"], s["name"])]
    if len(matches) != 1:
        raise ValueError(f"Task {task!r} is absent from {protocol['protocol_id']}")
    return matches[0]
