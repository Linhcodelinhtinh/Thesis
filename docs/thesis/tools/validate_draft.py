"""Static document/evidence consistency checks, not a TeX compilation test."""
from pathlib import Path
import csv
import json
import re

BASE = Path(__file__).resolve().parents[1]
tex = BASE / "latex"
bib = (tex / "references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
def read_tex(path, stack=()):
    assert path not in stack, f"Circular input: {path}"
    value = path.read_text(encoding="utf-8")
    value = re.sub(r"(?<!\\)%[^\n]*", "", value)
    return re.sub(r"\\(?:input|include)\{([^}]+)\}",
                  lambda m: read_tex(tex / (m[1] + ".tex"), (*stack, path)), value)

draft = read_tex(tex / "main.tex")
used = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", draft) for key in group.split(",")}
assert used <= keys, f"Undefined citations: {used - keys}"
labels = re.findall(r"\\label\{([^}]+)\}", draft)
assert len(labels) == len(set(labels)), "Duplicate LaTeX label"
refs = set(re.findall(r"\\(?:ref|eqref|pageref)\{([^}]+)\}", draft + bib))
assert refs <= set(labels), f"Undefined references: {refs - set(labels)}"
assert "ZZPLACEHOLDER" not in draft
for name in re.findall(r"\\path\{([^}]+)\}", draft + bib):
    assert not re.search(r"[/\\]|\.(?:jsonl?|csv|hdf5|py|tex|md)$", name, re.I), name
for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", draft):
    assert (tex / name).is_file(), name
snapshot = json.loads((BASE / "evidence/snapshot.json").read_text(encoding="utf-8"))
assert snapshot["training"]["data_manifest_matches_training"] is True
assert snapshot["training"]["adapter_files"] and all(
    item["matches_manifest"] for item in snapshot["training"]["adapter_files"].values())
writer = snapshot["writer_replay"]
assert 0 <= writer["matches"] <= min(writer["annotated_returns"], writer["predicted_returns"])
from hashlib import file_digest

for rel, expected in writer["evidence_sha256"].items():
    with (BASE.parents[1] / rel).open("rb") as source:
        assert file_digest(source, "sha256").hexdigest() == expected, rel
with (BASE / "evidence/mem_test_episodes.csv").open(encoding="utf-8-sig") as stream:
    episodes = list(csv.DictReader(stream))
assert len(episodes) == snapshot["mem_test"]["primary_summary_hashes_verified"]
keys = [(r["condition"], r["task"], r["initial_state_sha256"], r["policy_seed"]) for r in episodes]
assert len(keys) == len(set(keys)), "Duplicate episode identity"
assert len({r["initial_state_sha256"] for r in episodes}) == snapshot["mem_test"]["unique_state_clusters"]
with (BASE / "evidence/mem_test_counts.csv").open(encoding="utf-8-sig") as stream:
    counts = list(csv.DictReader(stream))
expected_cells = {(r["condition"], task) for r in episodes for task in (r["task"], "ALL")}
assert {(r["condition"], r["task"]) for r in counts} == expected_cells
assert len(counts) == len(expected_cells)
for row in counts:
    group = [r for r in episodes if r["condition"] == row["condition"] and
             row["task"] in ("ALL", r["task"])]
    wins = sum(r["success"] == "True" for r in group)
    assert len(group) == int(row["n_task_episodes"]) and wins == int(row["successes"])
    assert abs(wins / len(group) - float(row["sr"])) < 1e-12
assert "NON-COMPARABLE_OFFICIAL_PAPER" in draft
print(json.dumps({"status": "STATIC_CHECKS_PASS", "citations": len(used),
                  "labels": len(labels), "episode_rows": len(episodes)}, indent=2))
