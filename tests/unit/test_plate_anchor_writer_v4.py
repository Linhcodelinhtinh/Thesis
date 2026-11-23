"""Plate-anchor writer V4 on synthetic timelines (unit fixtures, not benchmark data)."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.memory.relation_cycle_writer_v4 import (CONFIG_PATH, V4_WRITERS, PlateAnchorWriterV4, PlatePriorConfig,
                                                 v4_source_hashes)
from src.memory.relation_writer_versions import versioned_relation_source_hashes
from tests.unit.test_bowl_cycle_writer import PLATE, PROFILE
from tests.unit.test_relation_sparse_writer import OPEN, PICK_AND_PLACE, box_at, frame_obs, renderer, text

ROOT = Path(__file__).resolve().parents[2]


def rows_without_plate(first, last):
    """PICK_AND_PLACE with the plate undetected in frames first..last-1."""
    rows = []
    for step, (boxes, aperture) in enumerate(PICK_AND_PLACE):
        if first <= step < last:
            boxes = [box for box in boxes if box[0] != "plate"]
        rows.append((boxes, aperture))
    return rows


def run(rows, cls, **kwargs):
    writer = cls("ep", PROFILE, None, "libero_mem_refined", **kwargs)
    snapshots = [writer.step(step, frame_obs(step, boxes, aperture)) for step, (boxes, aperture) in enumerate(rows)]
    return writer, snapshots


def events(writer):
    return [(event.type, event.occurred_at) for event in writer.recent_events]


def test_config_round_trips_and_rejects_unsafe_values():
    config = PlatePriorConfig.load()
    assert json.loads(CONFIG_PATH.read_text(encoding="utf-8")) == config.to_dict()
    for change in (dict(when_missing=1), dict(max_consecutive_frames=0), dict(max_consecutive_frames=2.5)):
        with pytest.raises(ValueError):
            replace(config, **change)


def test_version_is_distinct_and_the_prompt_contract_is_v3s():
    assert PlateAnchorWriterV4.VERSION == "rgb_relations_v4" != BowlCycleWriterV3.VERSION
    assert PlateAnchorWriterV4.CORE_VERSION == BowlCycleWriterV3.VERSION
    assert V4_WRITERS == {"rgb_relations_v4": PlateAnchorWriterV4}


def test_without_detector_losses_v4_is_v3():
    v3, v3_snapshots = run(PICK_AND_PLACE, BowlCycleWriterV3)
    v4, v4_snapshots = run(PICK_AND_PLACE, PlateAnchorWriterV4)
    assert events(v3) == events(v4) and v4.stats["prior_frames"] == 0
    assert [text(s) for s in v3_snapshots] == [text(s) for s in v4_snapshots]


def test_a_return_with_the_plate_lost_is_captured_by_v4_only():
    # The plate vanishes from the lift until well after the bowl is set down.
    rows = rows_without_plate(62, 120)
    v3, _ = run(rows, BowlCycleWriterV3)
    v4, snapshots = run(rows, PlateAnchorWriterV4)
    assert [kind for kind, _ in events(v3)].count("return") == 0
    assert [kind for kind, _ in events(v4)].count("return") == 1
    assert v4.stats["prior_missing"] > 0 and v4.stats["prior_frames"] == v4.stats["prior_missing"]
    assert text(snapshots[-1]) == "lift the bowl Memory: bowl: 1 observed returns."


def test_prior_is_never_used_before_the_origin_is_verified():
    # The plate is absent from the first frame: no anchor, no origin, so nothing to bridge with.
    rows = rows_without_plate(0, 30)
    writer, _ = run(rows, PlateAnchorWriterV4)
    assert writer.stats["prior_frames"] == 0 or writer.reducers["bowl_1"].origin is not None


def test_run_limit_stops_the_stand_in():
    rows = rows_without_plate(62, 120)
    limited, _ = run(rows, PlateAnchorWriterV4, prior=PlatePriorConfig(max_consecutive_frames=3))
    assert limited.stats["longest_run"] == 3
    assert [kind for kind, _ in events(limited)].count("return") == 0


def test_disabled_switches_reproduce_v3():
    rows = rows_without_plate(62, 120)
    off = PlatePriorConfig(when_missing=False, when_ambiguous=False)
    v3, _ = run(rows, BowlCycleWriterV3)
    v4, _ = run(rows, PlateAnchorWriterV4, prior=off)
    assert events(v3) == events(v4) and v4.stats["prior_frames"] == 0


def test_stand_in_frames_are_flagged_and_the_snapshot_keeps_the_real_detections():
    rows = rows_without_plate(62, 120)
    writer = PlateAnchorWriterV4("ep", PROFILE, None, "libero_mem_refined")
    flagged = []
    for step, (boxes, aperture) in enumerate(rows):
        snapshot = writer.step(step, frame_obs(step, boxes, aperture))
        flagged.append(writer.last_signals["plate_prior"])
        # A duplicate delivery keeps the same flag and does not count twice.
        assert writer.step(step, frame_obs(step, boxes, aperture)) is snapshot
    assert any(flagged) and not any(flagged[:62]) and sum(flagged) == writer.stats["prior_frames"]
    # Identity of the plate is reported as measured: the anchor never becomes a current detection.
    assert snapshot.objects["plate_1"].identity_status.value != "tracked" or any(
        box[0] == "plate" for box in rows[-1][0])


def test_pinned_sources_are_untouched():
    audit = ROOT / "experiments/audits/bowl-cycle-v3-motion-development-20261118.json"
    if not audit.exists():
        pytest.skip("audit receipt not present")
    recorded = json.loads(audit.read_text(encoding="utf-8"))["writer_code_sha256"]
    assert versioned_relation_source_hashes(ROOT, BowlCycleWriterV3.VERSION) == recorded
    assert set(v4_source_hashes(ROOT)) > set(recorded)
