"""Termination-head labels, frame expansion, sampler and shard reader on synthetic files."""
import json

import numpy as np
import pytest

from scripts.xvla_v2 import build_termination_labels as btl
from src.training.termination_data import (HARD_NEGATIVE, NEGATIVE, POSITIVE, TerminationShardReader,
                                           balanced_indices, frame_examples, split_by_demo)

T1 = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
T3 = "KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times"
WORDS = {0: "0", 1: "one: bowl 1 on plate 1", 2: "two: bowl 1 on plate 1", 3: "three: bowl 1 on plate 1"}


def exo(counts):
    return [{"bowl 1": [20, [0.1, 0.1, 0.1, 0.1], WORDS[c]], "plate 1": [40, [0, 0, 0, 0], 0]} for c in counts]


def write_fixture(tmp_path, with_shards=True):
    t1_counts = [0] * 6 + [1] * 4             # done_from 6, 10 frames
    t3_counts = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3  # milestones 3, 6, 9
    samples = tmp_path / "samples.jsonl"
    with samples.open("w") as s:
        for i in range(5):
            s.write(json.dumps(dict(domain="base", task="X", demo="demo_0", split="train", prepared_index=i)) + "\n")
        for task, demo, n in ((T1, "demo_1", 10), (T3, "demo_2", 12)):
            for i in range(n):
                s.write(json.dumps(dict(domain="mem", task=task, demo=demo, split="train", prepared_index=i)) + "\n")
    meta = tmp_path / "metainfo.json"
    meta.write_text(json.dumps({T1: {"demo_1": {"exo_boxes": exo(t1_counts)}, "demo_9": {"exo_boxes": exo([0])}},
                                T3: {"demo_2": {"exo_boxes": exo(t3_counts)}}}))
    shards = None
    if with_shards:
        import h5py
        shards = tmp_path / "shards"
        shards.mkdir()
        prompts = {T1: ["i"] * 8 + ["i Memory: bowl: 1 observed returns."] * 2,
                   T3: ["i"] * 4 + ["i Memory: bowl: 1 observed returns."] * 8}
        for task, demo in ((T1, "demo_1"), (T3, "demo_2")):
            n = len(prompts[task])
            with h5py.File(shards / f"mem__{task}.hdf5", "w") as f:
                g = f.create_group(demo)
                g.create_dataset("prompts", data=np.array(prompts[task], dtype=object), dtype=h5py.string_dtype())
                img = np.arange(n * 4 * 4 * 3, dtype=np.uint8).reshape(n, 4, 4, 3)
                g.create_dataset("agentview_rgb", data=img)
                g.create_dataset("eye_in_hand_rgb", data=img)
    return samples, meta, shards


def test_author_counts_and_validation():
    assert btl.author_counts(exo([0, 1, 1])) == [0, 1, 1]
    with pytest.raises(ValueError):
        btl.author_counts(exo([0, 2]))
    with pytest.raises(ValueError):
        btl.author_counts([{"bowl 1": [0, [], "four: bowl 1 on plate 1"]}])
    with pytest.raises(ValueError):
        btl.target_for("KITCHEN_SCENE1_7_swap")


def test_prompt_returns_parser():
    assert btl.prompt_returns(["i", "i Memory: bowl: 2 observed returns. History incomplete."]) == [-1, 2]


def test_build_labels_with_prompts(tmp_path):
    samples, meta, shards = write_fixture(tmp_path)
    rows, hashes = btl.build(samples, meta, shards)
    t1, t3 = sorted(rows, key=lambda r: r["target"])
    assert (t1["done_from"], t1["tail_frames"], t1["milestones"]) == (6, 4, [6])
    assert t1["done_observable_from"] == 8
    assert (t3["done_from"], t3["milestones"], t3["done_observable_from"]) == (9, [3, 6, 9], None)
    assert len(hashes) == 2
    stats = btl.statistics(rows)
    assert stats[T1]["positive_frames"] == 4 and stats[T1]["demos_prompt_reaches_target"] == 1
    assert stats[T3]["demos_prompt_reaches_target"] == 0


def test_frame_count_mismatch_fails(tmp_path):
    samples, meta, _ = write_fixture(tmp_path, with_shards=False)
    with samples.open("a") as s:
        s.write(json.dumps(dict(domain="mem", task=T1, demo="demo_1", split="train", prepared_index=10)) + "\n")
    with pytest.raises(ValueError, match="author labels"):
        btl.build(samples, meta)


def test_cli_writes_manifest(tmp_path):
    samples, meta, shards = write_fixture(tmp_path)
    out = tmp_path / "out"
    btl.main(["--samples", str(samples), "--metainfo", str(meta), "--shards-root", str(shards),
              "--output-dir", str(out)])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["version"] == "termination_labels_v1" and len(manifest["sources"]["shards_sha256"]) == 2
    assert len((out / "termination_labels.jsonl").read_text().splitlines()) == 2


def test_frame_examples_kinds_and_observable(tmp_path):
    samples, meta, shards = write_fixture(tmp_path)
    rows, _ = btl.build(samples, meta, shards)
    ex = frame_examples(rows, hard_window=2)
    t3 = [e for e in ex if e.task == T3]
    kinds = [e.kind for e in t3]
    assert kinds[9:] == [POSITIVE] * 3
    assert kinds[3:5] == [HARD_NEGATIVE] * 2 and kinds[6:9] == [HARD_NEGATIVE] * 3 and kinds[0] == NEGATIVE
    obs = frame_examples(rows, observable_only=True)
    t1_pos = [e.frame for e in obs if e.task == T1 and e.label]
    assert t1_pos == [8, 9]  # frames 6-7 are done but not yet in memory: dropped, not negative
    assert not any(e.label for e in obs if e.task == T3)
    with pytest.raises(ValueError):
        frame_examples([dict(rows[0], prompt_returns=None)], observable_only=True)


def test_balanced_sampler_and_demo_split(tmp_path):
    samples, meta, shards = write_fixture(tmp_path)
    rows, _ = btl.build(samples, meta, shards)
    ex = frame_examples(rows, hard_window=2)
    idx = balanced_indices(ex, 1000, np.random.default_rng(0), 0.4, 0.3)
    kinds = [ex[i].kind for i in idx]
    assert kinds.count(POSITIVE) == 400 and kinds.count(HARD_NEGATIVE) == 300
    with pytest.raises(ValueError):
        balanced_indices(ex, 10, np.random.default_rng(0), 0.7, 0.4)
    train, val = split_by_demo(ex, 0.5, np.random.default_rng(1))
    assert {(e.task, e.demo) for e in train}.isdisjoint({(e.task, e.demo) for e in val})


def test_reader_matches_shard_convention(tmp_path):
    samples, meta, shards = write_fixture(tmp_path)
    rows, _ = btl.build(samples, meta, shards)
    ex = [e for e in frame_examples(rows) if e.task == T1 and e.frame == 9][0]
    reader = TerminationShardReader(shards)
    item = reader.read(ex)
    import h5py
    with h5py.File(shards / f"mem__{T1}.hdf5") as f:
        raw = f["demo_1"]["agentview_rgb"][9]
    assert np.array_equal(item["agentview"], raw[::-1, ::-1]) and np.array_equal(item["wrist"], raw)
    assert item["prompt"].endswith("1 observed returns.") and item["label"] == 1
    reader.close()
