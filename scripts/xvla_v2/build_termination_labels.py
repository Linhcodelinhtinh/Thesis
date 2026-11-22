#!/usr/bin/env python
"""Per-frame "task complete" labels for a termination head on the V2 Mem training shards.

Label source: the authors' own cumulative milestone annotations in data/libero_mem/metainfo.json
(exo_boxes[frame]["bowl 1"][2] = "0" | "one: bowl 1 on plate 1" | "two: ..." | "three: ..."),
the same parser as scripts/audit_relation_returns.py. These are training-time labels only; at
evaluation the head sees images + the causal writer prompt, never these annotations.

Per demo the output stores
  done_from          first frame whose author count equals the task target (T1: 1, T3: 3);
                     done[t] = t >= done_from
  milestones         frames where the author count increments (hard negatives for T3 lie
                     after the earlier milestones: bowl resting on the plate, task not done)
  prompt_returns     optional, per-frame "N observed returns" parsed from the shard prompts
                     (the causal writer text the policy was trained with); -1 = no memory fact
  done_observable_from  first frame where done AND prompt_returns >= target, if any

Only Mem-domain train demos listed in samples.jsonl are labelled (val/test Mem is empty in V2).
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TARGETS = {"KITCHEN_SCENE1_1_": 1, "KITCHEN_SCENE1_3_": 3}
COUNT_WORDS = {"0": 0, "one": 1, "two": 2, "three": 3}
PROMPT_COUNT = re.compile(r"\bbowl: (\d+) observed returns\.")
LABEL_VERSION = "termination_labels_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def target_for(task):
    for prefix, target in TARGETS.items():
        if task.startswith(prefix):
            return target
    raise ValueError("Termination labels support T1/T3 only: " + task)


def author_counts(frames, label="bowl 1"):
    counts = []
    for frame in frames:
        raw = str(frame[label][2])
        word = raw.split(":", 1)[0].strip()
        if word not in COUNT_WORDS or (word != "0" and not raw.endswith(f"{label} on plate 1")):
            raise ValueError(f"Unknown author milestone label: {raw!r}")
        counts.append(COUNT_WORDS[word])
    if any(b < a or b > a + 1 for a, b in zip(counts, counts[1:])):
        raise ValueError("Unexpected progression in author labels")
    return counts


def prompt_returns(prompts):
    values = []
    for prompt in prompts:
        match = PROMPT_COUNT.search(prompt)
        values.append(int(match.group(1)) if match else -1)
    return values


def label_demo(task, demo, frames_expected, counts, prompts=None):
    target = target_for(task)
    if len(counts) != frames_expected:
        raise ValueError(f"{task}/{demo}: author labels {len(counts)} frames, shard {frames_expected}")
    milestones = [i for i in range(1, len(counts)) if counts[i] > counts[i - 1]]
    done_from = next((i for i, c in enumerate(counts) if c >= target), None)
    row = dict(task=task, demo=demo, frames=frames_expected, target=target, milestones=milestones,
               done_from=done_from, final_author_count=counts[-1],
               tail_frames=(frames_expected - done_from) if done_from is not None else 0)
    if prompts is not None:
        if len(prompts) != frames_expected:
            raise ValueError(f"{task}/{demo}: prompts {len(prompts)} != frames {frames_expected}")
        returns = prompt_returns(prompts)
        row["prompt_returns"] = returns
        row["done_observable_from"] = next((i for i in range(done_from or frames_expected, frames_expected)
                                            if returns[i] >= target), None)
    return row


def mem_train_demos(samples_path):
    frames = defaultdict(int)
    with Path(samples_path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["domain"] != "mem":
                continue
            if row["split"] != "train":
                raise ValueError("V2 Mem shards are train-only; unexpected split " + row["split"])
            key = (row["task"], row["demo"])
            frames[key] = max(frames[key], row["prepared_index"] + 1)
    return dict(frames)


def iter_author_labels(metainfo, task, demos):
    """Yield (demo, exo_boxes) for the requested demos, streaming when ijson is available."""
    try:
        import ijson
    except ImportError:
        ijson = None
    if ijson is not None:
        with Path(metainfo).open("rb") as stream:
            for name, row in ijson.kvitems(stream, task, use_float=True):
                if name in demos:
                    yield name, row["exo_boxes"]
        return
    data = json.loads(Path(metainfo).read_text(encoding="utf-8"))[task]
    for name in demos:
        yield name, data[name]["exo_boxes"]


def read_shard_prompts(shards_root, task, demos):
    import h5py
    path = Path(shards_root) / f"mem__{task}.hdf5"
    with h5py.File(path, "r") as f:
        return {demo: [p.decode() if isinstance(p, bytes) else str(p) for p in f[demo]["prompts"][()]]
                for demo in demos}, sha256_file(path)


def build(samples, metainfo, shards_root=None):
    demos = mem_train_demos(samples)
    by_task = defaultdict(dict)
    for (task, demo), n in demos.items():
        by_task[task][demo] = n
    rows, shard_hashes = [], {}
    for task in sorted(by_task):
        wanted = by_task[task]
        prompts = None
        if shards_root is not None:
            prompts, shard_hashes[f"mem__{task}.hdf5"] = read_shard_prompts(shards_root, task, wanted)
        seen = set()
        for demo, frames in iter_author_labels(metainfo, task, set(wanted)):
            seen.add(demo)
            rows.append(label_demo(task, demo, wanted[demo], author_counts(frames),
                                   None if prompts is None else prompts[demo]))
        missing = set(wanted) - seen
        if missing:
            raise ValueError(f"{task}: no author labels for {sorted(missing)[:5]}")
    rows.sort(key=lambda r: (r["task"], int(r["demo"].split("_")[1])))
    return rows, shard_hashes


def statistics(rows):
    stats = {}
    for task in sorted({r["task"] for r in rows}):
        group = [r for r in rows if r["task"] == task]
        frames = sum(r["frames"] for r in group)
        positive = sum(r["tail_frames"] for r in group)
        entry = dict(demos=len(group), frames=frames, positive_frames=positive,
                     positive_fraction=positive / frames if frames else 0.0,
                     demos_without_done=sum(r["done_from"] is None for r in group),
                     tail_frames_min=min(r["tail_frames"] for r in group),
                     tail_frames_max=max(r["tail_frames"] for r in group))
        if "prompt_returns" in group[0]:
            entry["demos_prompt_reaches_target"] = sum(r["done_observable_from"] is not None for r in group)
            entry["observable_positive_frames"] = sum(r["frames"] - r["done_observable_from"]
                                                      for r in group if r["done_observable_from"] is not None)
        stats[task] = entry
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    shards = ROOT / "experiments/training/relations-author-split-v2/shards"
    p.add_argument("--samples", type=Path, default=shards / "samples.jsonl")
    p.add_argument("--metainfo", type=Path, default=ROOT / "data/libero_mem/metainfo.json")
    p.add_argument("--shards-root", type=Path, default=shards / "shards",
                   help="Uncompressed V2 shard directory; pass --no-prompts to skip reading prompts")
    p.add_argument("--no-prompts", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    rows, shard_hashes = build(a.samples, a.metainfo, None if a.no_prompts else a.shards_root)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    labels = a.output_dir / "termination_labels.jsonl"
    with labels.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    manifest = dict(version=LABEL_VERSION, classification="TRAINING_LABELS_AUTHOR_ANNOTATIONS",
                    label_semantics="done[t] = author cumulative milestone count at frame t >= task target",
                    sources=dict(samples_sha256=sha256_file(a.samples), metainfo_sha256=sha256_file(a.metainfo),
                                 shards_sha256=shard_hashes),
                    labels_sha256=sha256_file(labels), statistics=statistics(rows))
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["statistics"], indent=2))


if __name__ == "__main__":
    main()
