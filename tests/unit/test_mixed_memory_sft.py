"""Synthetic data contract fixtures; never used as benchmark demonstrations."""
from collections import Counter
import copy
import json
from types import SimpleNamespace
import h5py
import numpy as np
import pytest

from src.training.split_lock import assign_locked_groups, state_digest, seal_lock, validate_lock
from src.training.mixed_sft import canonical_raw_rgb, validate_mixed_dataset, ARTIFACT
from src.training.replay_sampler import BalancedReplaySampler
from src.training.memory_dataset import MemoryDataset
from src.training.split_lock import content_digest
from src.utils.file_hash import sha256_file
from src.memory.visual_pipeline import TOKENIZER_REVISION, TOKENIZER_SHA256


def test_split_groups_shared_states_and_excludes_writer_development():
    rows = [dict(domain="mem", task=task, demo=str(i), initial_state=[float(i)],
                 initial_state_sha256=state_digest([i]), used_for_writer_development=i in (1, 5, 31))
            for task in ("T1", "T3") for i in range(100)]
    assigned = assign_locked_groups(rows)
    reverse = assign_locked_groups(rows[::-1])
    by_id = lambda entries: {(e["task"], e["demo"]): e["split"] for e in entries}
    assert by_id(assigned) == by_id(reverse)
    assert Counter(e["split"] for e in assigned) == {"train": 160, "val": 20, "test": 20}
    validate_lock(seal_lock({"episodes": assigned}))
    assert all(e["split"] == "train" for e in assigned if e["used_for_writer_development"])
    for i in range(100):
        assert by_id(assigned)["T1", str(i)] == by_id(assigned)["T3", str(i)]


def test_split_rejects_tampering_leakage_and_held_out_development():
    rows = [dict(domain="mem", task=t, demo="1", initial_state_sha256="same", split="train") for t in ("T1", "T3")]
    lock = seal_lock({"episodes": rows})
    lock["episodes"][0]["split"] = "test"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_lock(lock)
    with pytest.raises(ValueError, match="leakage"):
        validate_lock(seal_lock({"episodes": rows}))
    rows[1]["split"] = "test"
    rows[1]["used_for_writer_development"] = True
    with pytest.raises(ValueError, match="development"):
        validate_lock(seal_lock({"episodes": rows}))


class SamplerFixture:
    episodes = [dict(domain=d, task=t, length=n) for d in ("base", "mem")
                for t in ("short", "long") for n in (10, 2000)]

    def __len__(self):
        return sum(e["length"] for e in self.episodes)


def labels(dataset):
    return [e for e in dataset.episodes for _ in range(e["length"])]


def test_sampler_exact_optimizer_mixture_and_paired_schedule():
    data = SamplerFixture()
    f0 = BalancedReplaySampler(data, 32, seed=42)
    f1 = BalancedReplaySampler(copy.deepcopy(data), 32, seed=42)
    assert f0.schedule_sha256() == f1.schedule_sha256()
    schedule, lab = f0.indices(), labels(data)
    for begin in range(0, len(schedule), 32):
        assert Counter(lab[i]["domain"] for i in schedule[begin:begin+32]) == {"base":16, "mem":16}
    tasks = Counter((lab[i]["domain"], lab[i]["task"]) for i in schedule)
    assert len(set(tasks.values())) == 1
    # Long trajectories cannot dominate sampling solely because they have more frames.
    assert .45 < np.mean([lab[i]["length"] == 10 for i in schedule]) < .55
    f0.set_epoch(1)
    assert f0.indices() != schedule
    f1.set_epoch(1)
    assert f0.indices() == f1.indices()


def test_accelerate_sharding_preserves_global_optimizer_batches_and_resume():
    pytest.importorskip("accelerate")
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler
    data = SamplerFixture()
    sampler = BalancedReplaySampler(data, 32)
    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=False)
    rank_batches = [list(BatchSamplerShard(batch_sampler, num_processes=2, process_index=rank,
                                          split_batches=False, even_batches=True)) for rank in (0,1)]
    lab = labels(data)
    for start in range(0,len(rank_batches[0]),8):
        samples = [i for rank in rank_batches for batch in rank[start:start+8] for i in batch]
        assert Counter(lab[i]["domain"] for i in samples) == {"base":16,"mem":16}
    # Resume skips rank-local microbatches from the same epoch schedule.
    again = list(BatchSamplerShard(batch_sampler, num_processes=2, process_index=0,
                                  split_batches=False, even_batches=True))
    assert again[16:] == rank_batches[0][16:]


def test_mem_image_conversion_is_exactly_one_vertical_flip():
    raw = np.arange(2*5*7*3, dtype=np.uint8).reshape(2,5,7,3)
    np.testing.assert_array_equal(canonical_raw_rgb(raw, "base"), raw)
    np.testing.assert_array_equal(canonical_raw_rgb(raw[:,::-1], "mem"), raw)
    with pytest.raises(ValueError):
        canonical_raw_rgb(raw.astype(float), "mem")


def make_dataset(root):
    (root/"shards").mkdir()
    rows, episodes, shards = [], [], {}
    for d in ("base", "mem"):
        name = f"shards/{d}__task.hdf5"
        with h5py.File(root/name,"w") as f:
            for i,s in enumerate(("train","val","test")):
                key=f"demo_{i}"; n=4
                row=dict(domain=d,task="task",demo=key,split=s,source=d+".hdf5",frames=n,
                         initial_state_sha256=state_digest([i, d=="mem"]))
                rows.append(row)
                episodes.append(dict(row,length=n,shard=name,memory_frames=0))
                g=f.create_group(key)
                g.create_dataset("actions",data=np.arange(28,dtype=np.float32).reshape(4,7)/28)
                g.create_dataset("robot_states",data=np.tile([0,0,0,0,0,0,0,0,1],(n,1)))
                for c in ("agentview_rgb","eye_in_hand_rgb"):
                    g.create_dataset(c,data=np.zeros((n,8,8,3),dtype=np.uint8))
                g.create_dataset("prompts",data=["instruction"]*n,dtype=h5py.string_dtype())
                g.create_dataset("has_memory",data=[False]*n)
                g.attrs["instruction"]="instruction"
                g.attrs["split"]=s
        shards[name]=sha256_file(root/name)
    (root/"protocol.json").write_text("{}")
    lock=seal_lock(dict(episodes=rows,sources=[],protocol_sha256=sha256_file(root/"protocol.json")))
    (root/"split.lock.json").write_text(json.dumps(lock))
    (root/"samples.jsonl").write_text("unit fixture")
    projection=[{k:e[k] for k in ("domain","task","demo","split","length","initial_state_sha256","source")} for e in episodes]
    m=dict(schema_version=2,artifact=ARTIFACT,status="complete",max_tokens=48,sources=[],memory_source="none",
           tokenizer_revision=TOKENIZER_REVISION,tokenizer_files_sha256={"tokenizer.json":TOKENIZER_SHA256},
           episodes=episodes,shards=shards,split_lock_sha256=lock["lock_sha256"],
           sample_selection_sha256=content_digest(projection),audit_sha256=sha256_file(root/"samples.jsonl"),
           counts={d:{s:4 for s in ("train","val","test")} for d in ("base","mem")})
    (root/"manifest.json").write_text(json.dumps(m))
    return m


def test_dataset_tail_padding_and_conditions_share_targets(tmp_path):
    make_dataset(tmp_path)
    validate_mixed_dataset(tmp_path)
    f0=MemoryDataset(tmp_path,"train",horizon=5,condition="no_memory")
    f1=MemoryDataset(tmp_path,"train",horizon=5,condition="memory")
    try:
        a,b=f0[3],f1[3]
        np.testing.assert_array_equal(a["action"],b["action"])
        assert a["actions_id_pad"].tolist() == [False,True,True,True,True]
        assert a["action_is_pad"].equal(a["actions_is_pad"])
        assert a["domain"] == "base" and f0[4]["domain"] == "mem"
    finally:
        f0.close(); f1.close()


def test_memory_changes_only_mem_text_and_preserves_f0_targets(tmp_path):
    m=make_dataset(tmp_path)
    path=tmp_path/"shards/mem__task.hdf5"
    with h5py.File(path,"r+") as f:
        f["demo_0/prompts"][0]="instruction; unit fixture history"
        f["demo_0/has_memory"][0]=True
    next(e for e in m["episodes"] if e["domain"]=="mem" and e["split"]=="train")["memory_frames"]=1
    m["shards"]["shards/mem__task.hdf5"]=sha256_file(path)
    m["memory_source"]="predicted_rgb_v3"
    # Contract fixture provenance, never a real model, detector or benchmark artifact.
    m["memory_provenance"]=[dict(writer="rgb_cycle_v3",writer_code_sha256={"unit":"fixture"},
        writer_config={"unit":"fixture"},detector={"unit":"fixture"},manifest_sha256="unit fixture")]
    (tmp_path/"manifest.json").write_text(json.dumps(m))
    validate_mixed_dataset(tmp_path)
    f0=MemoryDataset(tmp_path,"train",condition="no_memory")
    f1=MemoryDataset(tmp_path,"train",condition="memory")
    try:
        assert f0[4]["task"] == "instruction"
        assert f1[4]["task"] == "instruction; unit fixture history"
        for i in range(len(f0)):
            a,b=f0[i],f1[i]
            for key in ("agentview","wrist","robot","action","actions_id_pad"):
                np.testing.assert_array_equal(a[key],b[key])
        assert BalancedReplaySampler(f0,32).indices() == BalancedReplaySampler(f1,32).indices()
    finally:
        f0.close();f1.close()


@pytest.mark.parametrize("mutation",["omitted","wrong_count","wrong_shard","corrupt","wrong_source"])
def test_manifest_rejects_altered_selection_and_content(tmp_path,mutation):
    m=make_dataset(tmp_path)
    if mutation=="omitted":m["episodes"].pop()
    elif mutation=="wrong_count":m["counts"]["mem"]["test"]=99
    elif mutation=="wrong_shard":m["episodes"][0]["shard"]="shards/mem__task.hdf5"
    elif mutation=="corrupt":
        with (tmp_path/"shards/base__task.hdf5").open("ab") as f:f.write(b"corrupt")
    else:m["sources"]=[{"path":"unverified"}]
    (tmp_path/"manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError):validate_mixed_dataset(tmp_path)
