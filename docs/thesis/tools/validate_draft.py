"""Static consistency checks for the canonical LaTeX manuscript and evidence ledger."""
from pathlib import Path
import csv
import hashlib
import json
import re

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
TEX = BASE / "latex"
ALLOWED_CLAIM_STATES = {
    "verified_within_scope",
    "artifact_verified",
    "development_only",
    "hypothesis",
    "blocked_by_unavailable_training",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expand_tex(path, stack=(), sources=None):
    """Expand only the input/include subset used by this thesis."""
    if sources is None:
        sources = []
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Circular LaTeX input: {path}")
    raw = path.read_text(encoding="utf-8")
    sources.append((path, raw))

    def include(match):
        child = TEX / (match.group(1) + ".tex")
        if not child.is_file():
            raise FileNotFoundError(f"Missing LaTeX input: {child}")
        return expand_tex(child, (*stack, path), sources)

    expanded = re.sub(r"\\(?:input|include)\{([^}]+)\}", include, raw)
    return re.sub(r"(?<!\\)%[^\n]*", "", expanded)


def validate_claim_ledger(source_texts):
    ledger_path = BASE / "evidence/claim_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    claims = ledger.get("claims")
    sources = ledger.get("sources")
    if not isinstance(claims, list) or not isinstance(sources, list):
        raise ValueError("Claim ledger must contain claims and sources arrays")

    source_by_id = {row["id"]: row for row in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("Duplicate evidence source IDs")
    for item in sources:
        path = (REPO / item["path"]).resolve()
        if REPO.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"Missing or out-of-repository evidence source: {item['path']}")
        if item.get("verify_sha256", True) and digest(path) != item["sha256"].lower():
            raise ValueError(f"Evidence source checksum mismatch: {item['path']}")

    claim_by_id = {row["id"]: row for row in claims}
    if len(claim_by_id) != len(claims):
        raise ValueError("Duplicate claim IDs")
    for claim in claims:
        if claim.get("state") not in ALLOWED_CLAIM_STATES:
            raise ValueError(f"Unknown claim state for {claim.get('id')}: {claim.get('state')}")
        linked = claim.get("evidence_source_ids")
        if not isinstance(linked, list) or any(source_id not in source_by_id for source_id in linked):
            raise ValueError(f"Unresolved evidence link for claim {claim['id']}")
        if not claim.get("scope") or not claim.get("permitted_wording"):
            raise ValueError(f"Claim {claim['id']} needs scope and permitted wording")

    marker_ids = []
    for _, raw in source_texts:
        marker_ids.extend(re.findall(r"(?m)^\s*%\s*claim-id:\s*([a-zA-Z0-9_.-]+)\s*$", raw))
    if len(marker_ids) != len(set(marker_ids)):
        raise ValueError("A claim ID is tagged more than once in the included LaTeX")
    if set(marker_ids) != set(claim_by_id):
        raise ValueError(
            f"LaTeX/ledger claim IDs differ: missing ledger={set(marker_ids)-set(claim_by_id)}, "
            f"missing LaTeX tags={set(claim_by_id)-set(marker_ids)}"
        )
    return {"claims": len(claims), "sources": len(sources),
            "states": {state: sum(row["state"] == state for row in claims)
                       for state in sorted(ALLOWED_CLAIM_STATES)
                       if any(row["state"] == state for row in claims)}}


def validate_episode_register():
    """Validate internal arithmetic without freezing any pilot sample size."""
    episode_path = BASE / "evidence/mem_test_episodes.csv"
    counts_path = BASE / "evidence/mem_test_counts.csv"
    if not episode_path.is_file() or not counts_path.is_file():
        return {"episode_rows": 0, "cells": 0}
    with episode_path.open(encoding="utf-8-sig", newline="") as stream:
        episodes = list(csv.DictReader(stream))
    identity = [(row["condition"], row["task"], row["initial_state_sha256"], row["policy_seed"])
                for row in episodes]
    if len(identity) != len(set(identity)):
        raise ValueError("Duplicate policy episode identity in evidence register")
    with counts_path.open(encoding="utf-8-sig", newline="") as stream:
        counts = list(csv.DictReader(stream))
    count_keys = [(row["condition"], row["task"]) for row in counts]
    if len(count_keys) != len(set(count_keys)):
        raise ValueError("Duplicate aggregate cell in evidence register")
    expected = {(row["condition"], task) for row in episodes for task in (row["task"], "ALL")}
    if set(count_keys) != expected:
        raise ValueError("Aggregate cells do not cover the episode register")
    for row in counts:
        selected = [item for item in episodes if item["condition"] == row["condition"] and
                    (row["task"] == "ALL" or item["task"] == row["task"])]
        successes = sum(item["success"] == "True" for item in selected)
        if (len(selected) != int(row["n_task_episodes"])
                or successes != int(row["successes"])
                or abs(successes / len(selected) - float(row["sr"])) > 1e-12):
            raise ValueError(f"Aggregate does not match its episode rows: {row['condition']}/{row['task']}")
    return {"episode_rows": len(episodes), "cells": len(counts)}


draft = expand_tex(TEX / "main.tex", sources=[])
bib = (TEX / "references.bib").read_text(encoding="utf-8")
bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
used_citations = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", draft)
                  for key in group.split(",")}
if not used_citations <= bib_keys:
    raise ValueError(f"Undefined citations: {used_citations - bib_keys}")
labels = re.findall(r"\\label\{([^}]+)\}", draft)
if len(labels) != len(set(labels)):
    raise ValueError("Duplicate LaTeX labels")
refs = set(re.findall(r"\\(?:ref|eqref|pageref)\{([^}]+)\}", draft + bib))
if not refs <= set(labels):
    raise ValueError(f"Undefined references: {refs - set(labels)}")
if "ZZPLACEHOLDER" in draft:
    raise ValueError("Found a document-generation placeholder")
for value in re.findall(r"\\path\{([^}]+)\}", draft + bib):
    if re.search(r"[/\\]|\.(?:jsonl?|csv|hdf5|py|tex|md)$", value, re.I):
        raise ValueError(f"Local file path must not appear in the rendered thesis: {value}")
for value in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", draft):
    if not (TEX / value).is_file():
        raise FileNotFoundError(f"Missing thesis figure: {value}")

abstract_hash = (BASE / "evidence/abstract.sha256").read_text(encoding="ascii").strip().lower()
if digest(TEX / "content/abstract.tex") != abstract_hash:
    raise ValueError("Author-owned abstract checksum changed")

source_texts = []
expand_tex(TEX / "main.tex", sources=source_texts)
ledger_summary = validate_claim_ledger(source_texts)
episode_summary = validate_episode_register()

snapshot_path = BASE / "evidence/snapshot.json"
if snapshot_path.is_file():
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    for field in ("created_at", "git_head", "scope", "limitations"):
        if field not in snapshot:
            raise ValueError(f"Evidence snapshot missing field: {field}")

print(json.dumps({"status": "STATIC_CHECKS_PASS", "citations": len(used_citations),
                  "labels": len(labels), "episode_register": episode_summary,
                  "claim_ledger": ledger_summary, "abstract_sha256": abstract_hash}, indent=2))
