"""Static document/evidence consistency checks, not a TeX compilation test."""
from pathlib import Path
import csv
import json
import re

BASE = Path(__file__).resolve().parents[1]
draft = (BASE / "THESIS_DRAFT.md").read_text(encoding="utf-8")
bib = (BASE / "latex/references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
used = {key.strip().lstrip("@") for group in re.findall(r"\[@([^\]]+)\]", draft) for key in group.split(";")}
assert used <= keys, f"Undefined citations: {used-keys}"
assert keys <= used, f"Uncited bibliography entries: {keys-used}"
assert len(re.findall(r"(?m)^# Chương \d+\.", draft)) == 5
assert "NON-COMPARABLE_OFFICIAL_PAPER" in draft
assert draft.count("$$") % 2 == 0
for target in re.findall(r"!\[[^\]]+\]\(([^)]+)\)", draft):
    assert (BASE / target).is_file(), target
for p in (BASE / "latex").rglob("*.tex"):
    value = p.read_text(encoding="utf-8")
    assert "ZZPLACEHOLDER" not in value, p
    for target in re.findall(r"\\input\{([^}]+)\}", value):
        assert (BASE / "latex" / (target + ".tex")).is_file(), target
    assert not re.search(r"\[@[^\]]+\]", value), p
snapshot = json.loads((BASE / "evidence/snapshot.json").read_text(encoding="utf-8"))
assert snapshot["mem_test"]["primary_summary_hashes_verified"] == 40
assert snapshot["mem_test"]["unique_state_clusters"] == 10
assert snapshot["training"]["data_manifest_matches_training"] is True
assert all(x["matches_manifest"] for x in snapshot["training"]["adapter_files"].values())
with (BASE / "evidence/mem_test_episodes.csv").open(encoding="utf-8-sig") as stream:
    es = list(csv.DictReader(stream))
f1 = [r for r in es if r["condition"] == "F1"]
assert sum(r["success"] == "True" for r in f1) == 1
assert sum(r["final_writer_history_complete"] == "False" for r in f1) == 19
assert sum(int(r["policy_queries"]) for r in f1) == 1927
assert sum(int(r["queries_with_memory_suffix"]) for r in f1) == 1894
assert sum(int(r["queries_with_incomplete_history_text"]) for r in f1) == 1738
result = {"status": "STATIC_CHECKS_PASS", "citations": len(used), "chapters": 5,
          "markdown_whitespace_tokens": len(draft.split()),
          "note": "Not a linguistic word count; Vietnamese words may span spaces.",
          "tex_compiled": False, "pdf_layout_verified": False,
          "evidence_summary_hashes_verified": 40,
          "limitations": "No new policy benchmark, model inference, training, HDF5 audit or TeX compilation in this writing task."}
(BASE / "evidence/document_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
