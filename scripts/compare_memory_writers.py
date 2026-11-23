"""Compare how the replayable relation memory writers capture task events.

Every writer runs through its real class on the same archived Grounding DINO evidence and the same
measured gripper aperture, so only the writer logic differs. The detector is a replay: it hands out
the recorded evidence of a frame only when the writer asks for that frame, so a sparse writer
cannot read evidence for a frame it skipped. Evidence is validated against synthetic images whose
digests are patched in, because the original frames are not archived.

Reference events are the author's cumulative-count milestones (development demos, E2 rollouts).
Writers are scored with the same one-to-one matching as the earlier audits. The result is a
diagnostic of writer behaviour on recorded evidence, NOT policy success and NOT comparable with
official LIBERO numbers.
"""
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from itertools import groupby
from math import comb
from multiprocessing import Pool
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.audit_libero_mem_writer import match_events
from scripts.extract_writer_replay_inputs import rows, sha256_file
from src.interfaces.relation_memory_text import RelationMemoryRenderer
from src.interfaces.relation_memory_text_v3 import RelationMemoryRendererV3
from src.memory.libero_mem_profiles import MemoryProfile
from src.memory.relation_cycle_config import BowlCycleConfig
from src.memory.relation_cycle_writer import BowlCycleWriter
from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.memory.relation_cycle_writer_v4 import PlateAnchorWriterV4, v4_source_hashes
from src.memory.relation_memory import RelationMemoryWriter
from src.memory.relation_sparse_config import SparseWriterConfig
from src.memory.relation_sparse_versions import source_hashes_for
from src.memory.relation_sparse_writer import SparseBowlCycleWriterV3, SparseRelationWriterV1
from src.memory.relation_writer_versions import versioned_relation_source_hashes
from src.perception.grounding_dino import MODEL_ID, REVISION
from src.perception.visual_evidence import CAMERAS, VisualEvidence, image_digest

PROFILE = MemoryProfile("bowl", 1, "plate", 3, "on", count_cycles=True)
CONVENTION = "libero_mem_refined"
INSTRUCTIONS = {"T1": "pick up the bowl and place it back on the plate",
                "T3": "lift the bowl and place it back on the plate 3 times"}
QUERY_INTERVAL = 8
TOLERANCES = (10, 20, 30)
HEADLINE_TOLERANCE = 20
BOOTSTRAP_REPS = 2000
# p50 of the detector call measured in the D2 logs on the user's GPU (writer_ms per frame).
MEASURED_DETECTOR_MS = 782.0
SYNTHETIC = {value: {camera: image_digest(np.full((4, 4, 3), value, dtype=np.uint8)) for camera in CAMERAS}
             for value in range(256)}
CLASSIFICATION = "RECORDED_WRITER_REPLAY_NON_COMPARABLE"


# -- writers -------------------------------------------------------------------------------------

def writer_specs():
    """name -> (kind, interval, prompt family, dense counterpart, constructor)."""
    sparse = SparseWriterConfig.load()

    def dense(cls, config=None):
        return lambda episode, detector, **_: (cls(episode, PROFILE, detector, CONVENTION)
                                               if config is None else
                                               cls(episode, PROFILE, detector, CONVENTION, config=config))

    def sparse_writer(cls, interval):
        config = replace(sparse, keyframe_interval=interval, ring_buffer_frames=max(sparse.ring_buffer_frames, interval))
        return lambda episode, detector, **_: cls(episode, PROFILE, detector, CONVENTION, sparse=config)

    def v3(episode, detector, **_):
        return BowlCycleWriterV3(episode, PROFILE, detector, CONVENTION, config=BowlCycleConfig.load())

    def v4(episode, detector, **_):
        return PlateAnchorWriterV4(episode, PROFILE, detector, CONVENTION, config=BowlCycleConfig.load())

    specs = {
        "v1": ("dense", 1, "v1", None, dense(RelationMemoryWriter)),
        "v2": ("dense", 1, "v1", None, dense(BowlCycleWriter, BowlCycleConfig())),
        "v3": ("dense", 1, "v3", None, v3),
        "v4": ("dense", 1, "v3", "v3", v4),
        "sparse_v1_k8": ("sparse", 8, "v1", "v1", sparse_writer(SparseRelationWriterV1, 8)),
    }
    for interval in (4, 8, 16):
        specs[f"sparse_v3_k{interval}"] = ("sparse", interval, "v3", "v3", sparse_writer(SparseBowlCycleWriterV3, interval))
    for interval in (4, 8):
        # The same dense V3 writer run at 1/K of the frame rate, with no probe and no backfill.
        specs[f"subsampled_v3_k{interval}"] = ("subsampled", interval, "v3", "v3", v3)
    return specs


def renderer_for(family):
    cls = RelationMemoryRendererV3 if family == "v3" else RelationMemoryRenderer
    # The word count only stands in for the tokenizer; it is the same for every writer.
    return cls(object(), token_counter=lambda text: len(text.split()))


class RecordedDetector:
    """Hand out recorded evidence on request and remember which frames were requested."""

    def __init__(self, evidence):
        self.evidence, self.steps = evidence, []

    def detect(self, step, images, queries, *, pilot_crop=False, image_digests=None):
        evidence = self.evidence[step]
        if evidence.step != step or dict(evidence.image_hashes) != dict(image_digests):
            raise ValueError(f"Recorded evidence does not belong to frame {step}")
        self.steps.append(step)
        return evidence


def synthetic_observation(index, grip):
    rgb = {camera: np.full((4, 4, 3), index % 256, dtype=np.uint8) for camera in CAMERAS}
    return dict(rgb, robot0_gripper_qpos=list(grip))


def parse_evidence(payloads):
    """Typed evidence, validated like the earlier replay audits."""
    parsed = []
    for step, payload in enumerate(payloads):
        evidence = VisualEvidence.from_dict(payload)
        if (evidence.step != step or (evidence.model_id, evidence.revision) != (MODEL_ID, REVISION)
                or set(evidence.views) != set(CAMERAS) or set(evidence.image_hashes) != set(CAMERAS)):
            raise ValueError(f"Invalid recorded evidence at step {step}")
        parsed.append(evidence)
    return parsed


def run_writer(spec, name, episode, task, evidence, grip, query_interval=QUERY_INTERVAL):
    kind, interval, family, _, build = spec
    stride = interval if kind == "subsampled" else 1
    # The replay detector is indexed by the step the writer sees, which a subsampled run renumbers.
    fed = {step // stride: replace(item, step=step // stride, image_hashes=SYNTHETIC[(step // stride) % 256])
           for step, item in enumerate(evidence) if step % stride == 0}
    detector = RecordedDetector(fed)
    writer = build(episode, detector)
    renderer = renderer_for(family)
    events, queries, millis, schedule = [], [], [], {}
    snapshot = None
    for step in range(len(evidence)):
        if step % stride == 0:
            index = step // stride
            started = time.perf_counter()
            snapshot = writer.step(index, synthetic_observation(index, grip[step]),
                                   None if index == 0 else np.zeros(7))
            millis.append((time.perf_counter() - started) * 1000.0)
            if kind == "sparse":
                sparse = writer.last_signals["sparse"]
                label = f"{sparse['mode']}:{sparse.get('reason')}"
                schedule[label] = schedule.get(label, 0) + 1
            events.extend(dict(type=event.type, occurred_at=event.occurred_at * stride,
                               available_at=event.available_at * stride, published=step)
                          for event in writer.last_events)
        if step % query_interval == 0:
            state = snapshot.relations.get("bowl_1")
            text = renderer.render(INSTRUCTIONS[task], snapshot).text
            queries.append([step, state.returns if state else 0, bool(state.history_complete) if state else True,
                            bool(state.armed) if state else False, hashlib.sha1(text.encode()).hexdigest()[:12]])
    if len(set(detector.steps)) != len(detector.steps):
        raise ValueError(f"{name} asked for the same frame twice in {episode}")
    stats = getattr(writer, "stats", None)
    if stats is not None and "detector_calls" in stats and stats["detector_calls"] != len(detector.steps):
        raise ValueError(f"{name} miscounted its detector calls in {episode}")
    state = snapshot.relations.get("bowl_1")
    return dict(events=events, queries=queries, calls=len(detector.steps), frames=len(evidence),
                final_returns=state.returns if state else 0, millis=millis, schedule=schedule,
                stats=dict(stats) if stats else None)


def run_episode(job):
    episode, task, payloads, grip, names = job
    evidence = parse_evidence(payloads)
    specs = writer_specs()
    return episode, {name: run_writer(specs[name], name, episode, task, evidence, grip) for name in names}


# -- inputs --------------------------------------------------------------------------------------

def development_jobs(indexes, inputs, names, limit=None):
    gripper = json.loads((inputs / "development_gripper.json").read_text(encoding="utf-8"))
    if gripper["classification"] != "REPLAY_INPUTS_NOT_A_RESULT":
        raise ValueError("Unexpected development gripper file")
    report = json.loads((inputs / "extraction_report.json").read_text(encoding="utf-8"))
    receipts = {Path(row["index"].replace("\\", "/")).name: row for row in report["development"]}
    references, sources = {}, {}
    for index in map(Path, indexes):
        manifest_path, detections = index / "manifest.json", index / "detections.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = receipts[index.name]
        if (manifest["status"] != "complete" or sha256_file(manifest_path) != receipt["manifest_sha256"]
                or sha256_file(detections) != manifest["artifacts_sha256"]["detections.jsonl"]):
            raise ValueError(f"Development index or its extraction receipt does not match: {index}")
        audit = {row["episode"]: row for row in json.loads(
            (index / "milestone-audit.json").read_text(encoding="utf-8"))["episodes"]}
        expected = {row["id"]: row["frames"] for row in manifest["episodes"]}
        if audit.keys() != expected.keys():
            raise ValueError(f"Milestone roster differs from the manifest: {index}")
        sources[index.name] = dict(manifest_sha256=receipt["manifest_sha256"],
                                   detections_sha256=manifest["artifacts_sha256"]["detections.jsonl"],
                                   hdf5_sha256_verified=receipt["hdf5_sha256_verified"],
                                   apertures_cross_checked=receipt["apertures_cross_checked"],
                                   worst_aperture_difference=receipt["worst_aperture_difference"])
        jobs = []
        for episode, records in groupby(rows(detections), key=lambda row: row["episode_id"]):
            payloads = [row["evidence"] for row in records]
            stored = gripper["episodes"][episode]
            if len(payloads) != expected[episode] or stored["frames"] != expected[episode]:
                raise ValueError(f"Frame count mismatch: {episode}")
            references[episode] = dict(task=stored["task"], frames=len(payloads),
                                       milestones=audit[episode]["milestones"],
                                       recorded_v1=audit[episode]["predicted"],
                                       receipt=receipt["manifest_sha256"])
            jobs.append((episode, stored["task"], payloads, stored["gripper_qpos"], names))
        if {job[0] for job in jobs} != set(expected):
            raise ValueError(f"Detections do not cover the manifest roster: {index}")
        yield from jobs[:limit]
    yield references, sources


def rollout_jobs(inputs, names, limit=None):
    references = {}
    with gzip.open(inputs / "rollout_replay_inputs.jsonl.gz", "rt", encoding="utf-8") as stream:
        for count, line in enumerate(stream):
            row = json.loads(line)
            references[row["episode"]] = dict(task=row["task"], frames=row["frames"], milestones=row["milestones"],
                                              recorded_v1=row["recorded_return_steps"],
                                              receipt=row["source_receipt_sha256"])
            if limit is None or count < limit:
                yield row["episode"], row["task"], row["evidence"], row["gripper_qpos"], names
    yield references, None


def split_references(stream):
    """The generators above yield jobs, then (references, sources) last."""
    items = list(stream)
    return items[:-1], *items[-1]


# -- metrics -------------------------------------------------------------------------------------

def returns_of(run):
    return [event["available_at"] for event in run["events"] if event["type"] == "return"]


def published_returns_of(run):
    return [event["published"] for event in run["events"] if event["type"] == "return"]


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def exact_mcnemar(wins, losses):
    """Two-sided exact sign test on the milestones only one of two writers captured."""
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(wins, losses) + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return precision, recall, f1


def bootstrap(counts, other=None, reps=BOOTSTRAP_REPS, seed=0):
    """Episode-cluster bootstrap of precision/recall/F1 (and of their difference to *other*)."""
    counts = np.asarray(counts, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(counts), size=(reps, len(counts)))

    def metrics(array):
        tp, fp, fn = (array[draws, column].sum(axis=1) for column in range(3))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.stack([tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)], axis=1)

    values = metrics(counts)
    if other is not None:
        values = values - metrics(np.asarray(other, dtype=float))

    def interval(column):
        valid = values[:, column][np.isfinite(values[:, column])]
        return [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))] if len(valid) else None

    return dict(precision=interval(0), recall=interval(1), f1=interval(2))


def score(runs, references, tolerance):
    """Per-episode matches of one writer's returns against the author milestones."""
    scored = {}
    for episode, reference in references.items():
        run = runs[episode]
        match = match_events(returns_of(run), reference["milestones"], tolerance)
        published = match_events(published_returns_of(run), reference["milestones"], tolerance)
        scored[episode] = dict(
            match, hits=[pair["annotation"] for pair in match["matches"]],
            publish_delays=[pair["delay"] for pair in published["matches"]])
    return scored


def summarize(runs, references, scored, tolerance):
    episodes = list(references)
    totals = {key: sum(scored[e][key] for e in episodes) for key in ("true_positive", "false_positive",
                                                                      "false_negative")}
    precision, recall, f1 = prf(totals["true_positive"], totals["false_positive"], totals["false_negative"])
    delays = [pair["delay"] for e in episodes for pair in scored[e]["matches"]]
    publish = [delay for e in episodes for delay in scored[e]["publish_delays"]]
    frames = sum(references[e]["frames"] for e in episodes)
    calls = sum(runs[e]["calls"] for e in episodes)
    millis = [value for e in episodes for value in runs[e]["millis"]]
    ordinal = {}
    for position in range(3):
        have = [(e, reference) for e in episodes for reference in references[e]["milestones"][position:position + 1]]
        if have:
            hit = sum(reference in scored[e]["hits"] for e, reference in have)
            ordinal[str(position + 1)] = dict(reference=len(have), captured=hit, recall=hit / len(have))
    queries = [(e, step, returns, complete) for e in episodes for step, returns, complete, _, _ in runs[e]["queries"]]
    outside = within = incomplete = exact = counted = 0
    for e, step, returns, complete in queries:
        milestones = references[e]["milestones"]
        incomplete += not complete
        if any(abs(step - m) <= tolerance for m in milestones):
            within += 1
            continue
        outside += 1
        exact += returns == sum(m <= step for m in milestones)
    final = [runs[e]["final_returns"] == len(references[e]["milestones"]) for e in episodes]
    schedule = {}
    for e in episodes:
        for label, count in runs[e]["schedule"].items():
            schedule[label] = schedule.get(label, 0) + count
    sparse_stats = {}
    for e in episodes:
        for key, value in (runs[e]["stats"] or {}).items():
            sparse_stats[key] = (max(sparse_stats.get(key, 0), value) if key.startswith("longest")
                                 else sparse_stats.get(key, 0) + value)
    return dict(
        schedule=dict(sorted(schedule.items())), sparse_stats=sparse_stats,
        tolerance=tolerance, episodes=len(episodes), reference_events=sum(len(references[e]["milestones"]) for e in episodes),
        **totals, precision=precision, recall=recall, f1=f1,
        false_positive_per_episode=totals["false_positive"] / len(episodes),
        false_positive_per_1000_frames=1000 * totals["false_positive"] / frames,
        episodes_with_false_positive=sum(scored[e]["false_positive"] > 0 for e in episodes),
        confirm_delay=dict(median=percentile(delays, 50), p90=percentile(delays, 90), max=max(delays, default=None)),
        publish_delay=dict(median=percentile(publish, 50), p90=percentile(publish, 90), max=max(publish, default=None)),
        recall_by_ordinal=ordinal,
        final_count_exact_episodes=sum(final), final_count_exact_rate=sum(final) / len(final),
        query_frames=len(queries), query_frames_outside_tolerance=outside,
        query_count_correct_rate=exact / outside if outside else None,
        history_incomplete_query_rate=incomplete / len(queries) if queries else None,
        detector_calls=calls, frames=frames, detector_calls_per_frame=calls / frames,
        estimated_detector_ms_per_frame=MEASURED_DETECTOR_MS * calls / frames,
        cached_writer_ms=dict(p50=percentile(millis, 50), p95=percentile(millis, 95)),
        bootstrap_95=bootstrap([[scored[e]["true_positive"], scored[e]["false_positive"],
                                 scored[e]["false_negative"]] for e in episodes]))


def agreement(runs, other):
    """How closely *runs* follows *other* (its dense counterpart) in events and in the prompt."""
    episodes = list(runs)
    same_returns = sum(returns_of(runs[e]) == returns_of(other[e]) for e in episodes)
    key = lambda run: [(event["type"], event["occurred_at"], event["available_at"]) for event in run["events"]]
    same_events = sum(key(runs[e]) == key(other[e]) for e in episodes)
    pairs = [(a, b) for e in episodes for a, b in zip(runs[e]["queries"], other[e]["queries"], strict=True)]
    return dict(episodes=len(episodes), identical_return_steps=same_returns, identical_event_sequences=same_events,
                query_frames=len(pairs), identical_prompt_text=sum(a[4] == b[4] for a, b in pairs),
                identical_count=sum(a[1] == b[1] for a, b in pairs),
                identical_history_flag=sum(a[2] == b[2] for a, b in pairs),
                history_complete_only_in_candidate=sum(a[2] and not b[2] for a, b in pairs),
                history_incomplete_only_in_candidate=sum(b[2] and not a[2] for a, b in pairs),
                identical_lift_flag=sum(a[3] == b[3] for a, b in pairs))


def hit_sets(scored, references):
    return {(e, m) for e in references for m in scored[e]["hits"]}


def compare_hits(first, second, references):
    both = len(first & second)
    only_first, only_second = len(first - second), len(second - first)
    total = sum(len(row["milestones"]) for row in references.values())
    return dict(both=both, only_first=only_first, only_second=only_second,
                neither=total - both - only_first - only_second,
                exact_p=exact_mcnemar(only_first, only_second))


def venn(scored_by_writer, names, references):
    sets = {name: hit_sets(scored_by_writer[name], references) for name in names}
    cells = {}
    for e, row in references.items():
        for m in row["milestones"]:
            cell = "+".join(name for name in names if (e, m) in sets[name]) or "none"
            cells[cell] = cells.get(cell, 0) + 1
    return cells


def fidelity(runs_by_writer, references, checks):
    """Replay must reproduce the earlier audits' event steps; report every difference."""
    report = {}
    for check in checks:
        label, path = check.split("=", 1)
        writer, field = label.split(":")
        audit = {row["episode"]: row for row in json.loads(Path(path).read_text(encoding="utf-8"))["episodes"]}
        mismatched = [e for e in references if e in audit and returns_of(runs_by_writer[writer][e]) != audit[e][field]]
        report[label] = dict(path=path, sha256=sha256_file(path), compared=len(set(references) & set(audit)),
                             missing_from_audit=len(set(references) - set(audit)), mismatched=len(mismatched),
                             mismatched_episodes=mismatched[:10],
                             milestones_equal=all(references[e]["milestones"] == audit[e]["milestones"]
                                                  for e in references if e in audit),
                             receipts_equal=all(references[e]["receipt"] == audit[e].get("source_receipt_sha256")
                                                for e in references if e in audit))
    return report


def evaluate(results, references, names, tolerances=TOLERANCES, checks=()):
    runs_by_writer = {name: {e: results[e][name] for e in references} for name in names}
    specs = writer_specs()
    out = dict(episodes=len(references), frames=sum(row["frames"] for row in references.values()),
               reference_events=sum(len(row["milestones"]) for row in references.values()))
    scored_by_tolerance = {t: {name: score(runs_by_writer[name], references, t) for name in names} for t in tolerances}
    out["summary"] = {str(t): {name: summarize(runs_by_writer[name], references, scored_by_tolerance[t][name], t)
                               for name in names} for t in tolerances}
    by_task = {}
    for task in sorted({row["task"] for row in references.values()}):
        selected = {e: row for e, row in references.items() if row["task"] == task}
        by_task[task] = {name: summarize(runs_by_writer[name], selected, scored_by_tolerance[HEADLINE_TOLERANCE][name],
                                         HEADLINE_TOLERANCE) for name in names}
    out["by_task"] = by_task
    headline = scored_by_tolerance[HEADLINE_TOLERANCE]
    out["against_dense"] = {name: agreement(runs_by_writer[name], runs_by_writer[specs[name][3]])
                            for name in names if specs[name][3]}
    pairs = [("v4", "v3"), ("v4", "v1"), ("v3", "v1"), ("v3", "v2"), ("v2", "v1"), ("sparse_v3_k8", "v3"), ("sparse_v1_k8", "v1"),
             ("subsampled_v3_k8", "v3")]
    out["hit_comparison"] = {}
    for first, second in pairs:
        if first in names and second in names:
            row = compare_hits(hit_sets(headline[first], references), hit_sets(headline[second], references), references)
            counts = lambda w: [[headline[w][e]["true_positive"], headline[w][e]["false_positive"],
                                 headline[w][e]["false_negative"]] for e in references]
            row["bootstrap_difference_95"] = bootstrap(counts(first), counts(second))
            out["hit_comparison"][f"{first}_minus_{second}"] = row
    core = [name for name in ("v1", "v2", "v3", "v4") if name in names]
    out["venn_dense_writers"] = venn(headline, core, references)
    out["fidelity"] = fidelity(runs_by_writer, references, checks)
    out["recorded_v1_events_equal_replay"] = dict(
        compared=len(references),
        mismatched=sum(returns_of(runs_by_writer["v1"][e]) != references[e]["recorded_v1"] for e in references))
    return out, runs_by_writer


def run_dataset(jobs, workers):
    if workers <= 1:
        return dict(map(run_episode, jobs))
    with Pool(workers) as pool:
        return dict(pool.imap(run_episode, jobs, chunksize=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Output of extract_writer_replay_inputs.py")
    parser.add_argument("--development-index", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fidelity", action="append", default=[], metavar="DATASET:WRITER:FIELD=AUDIT_JSON",
                        help="Compare a writer's replayed return steps with an earlier audit's field")
    parser.add_argument("--writers", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, help="Smoke test only: the result is marked partial")
    args = parser.parse_args(argv)
    specs = writer_specs()
    names = args.writers or list(specs)
    if unknown := set(names) - set(specs):
        parser.error(f"Unknown writers: {sorted(unknown)}")
    # Every writer is compared with its dense counterpart, and v1 reproduces the recorded events.
    names = [name for name in specs if name in set(names) | {"v1"} | {specs[n][3] for n in names if specs[n][3]}]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    datasets, episodes_out = {}, {}
    for dataset, make in (("development", lambda: development_jobs(args.development_index, args.inputs, names, args.max_episodes)),
                          ("e2", lambda: rollout_jobs(args.inputs, names, args.max_episodes))):
        jobs, references, sources = split_references(make())
        if args.max_episodes:
            references = {job[0]: references[job[0]] for job in jobs}
        results = run_dataset(jobs, args.workers)
        checks = [item.split(":", 1)[1] for item in args.fidelity if item.split(":", 1)[0] == dataset]
        datasets[dataset], runs = evaluate(results, references, names, checks=checks)
        datasets[dataset]["sources"] = sources
        episodes_out[dataset] = {name: {e: dict(returns=returns_of(run), events=run["events"], calls=run["calls"],
                                                final_returns=run["final_returns"], stats=run["stats"],
                                                schedule=run["schedule"],
                                                milestones=references[e]["milestones"])
                                        for e, run in runs[name].items()} for name in names}
        print(dataset, "done", f"{time.time() - started:.0f}s", flush=True)
    cycle, sparse = BowlCycleConfig.load(), SparseWriterConfig.load()
    protocol = dict(
        classification=CLASSIFICATION, complete=args.max_episodes is None, tolerances=list(TOLERANCES),
        headline_tolerance=HEADLINE_TOLERANCE, query_interval=QUERY_INTERVAL, writers={
            name: dict(kind=specs[name][0], interval=specs[name][1], prompt=specs[name][2], dense_counterpart=specs[name][3])
            for name in names},
        v3_config=cycle.to_dict(), v2_config=BowlCycleConfig().to_dict(), sparse_config=sparse.to_dict(),
        measured_detector_ms=MEASURED_DETECTOR_MS, bootstrap=dict(reps=BOOTSTRAP_REPS, unit="episode", seed=0),
        reference="author cumulative task milestones, not grasp or contact labels",
        source_hashes=dict(v1=versioned_relation_source_hashes(ROOT, RelationMemoryWriter.VERSION),
                           v2=versioned_relation_source_hashes(ROOT, BowlCycleWriter.VERSION),
                           v3=versioned_relation_source_hashes(ROOT, BowlCycleWriterV3.VERSION),
                           sparse_v3=source_hashes_for(ROOT, SparseBowlCycleWriterV3.VERSION),
                           v4=v4_source_hashes(ROOT),
                           harness=sha256_file(Path(__file__))),
        inputs={name: sha256_file(args.inputs / name) for name in ("development_gripper.json",
                                                                     "rollout_replay_inputs.jsonl.gz",
                                                                     "extraction_report.json")})
    with (args.output_dir / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(protocol=protocol, datasets=datasets), stream, indent=2)
        stream.write("\n")
    with gzip.open(args.output_dir / "episodes.json.gz", "xt", encoding="utf-8") as stream:
        json.dump(dict(classification=CLASSIFICATION, episodes=episodes_out), stream)
    print("done", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
