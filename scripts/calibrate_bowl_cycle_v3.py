"""Calibrate the versioned bowl-cycle thresholds on locked T1/T3 development data."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_bowl_cycle_replay import PROFILE, development_episodes, replay, score, rows
from src.memory.relation_cycle_config import BowlCycleConfig, CONFIG_PATH
from src.memory.relation_cycle_writer import BowlCycleWriter
from src.memory.relation_cycle_writer_v3 import BowlCycleWriterV3
from src.memory.relation_writer_versions import versioned_relation_source_hashes
from src.utils.file_hash import sha256_file


def evaluate_candidate(templates, inputs, config, tolerance):
    episodes = []
    for template in templates:
        predicted, length = replay(template["episode"], inputs[template["episode"]],
                                   BowlCycleWriterV3.VERSION, config)
        if length != template["frames"]:
            raise ValueError(f"Replay length changed for {template['episode']}")
        episodes.append({**template, "proposed": predicted})
    return episodes, score(episodes, tolerance)


def candidates(baseline):
    """One-factor development sweeps plus a bounded joint candidate."""
    return {
        "locked_baseline": baseline,
        "lower_lift_pair": replace(baseline, lift_center_rise_ratio=.25,
                                   lift_bottom_rise_ratio=.35),
        "lower_lift_pair_2": replace(baseline, lift_center_rise_ratio=.20,
                                     lift_bottom_rise_ratio=.30),
        "lower_lift_pair_3": replace(baseline, lift_center_rise_ratio=.15,
                                     lift_bottom_rise_ratio=.25),
        "lower_lift_pair_4": replace(baseline, lift_center_rise_ratio=.10,
                                     lift_bottom_rise_ratio=.20),
        "lower_lift_pair_5": replace(baseline, lift_center_rise_ratio=.05,
                                     lift_bottom_rise_ratio=.15),
        "lower_lift_pair_6": replace(baseline, lift_center_rise_ratio=0.,
                                     lift_bottom_rise_ratio=.10),
        "wider_return_tolerance": replace(baseline, return_offset_ratio=.40),
        "wider_return_tolerance_2": replace(baseline, return_offset_ratio=.55),
        "wider_return_tolerance_3": replace(baseline, return_offset_ratio=.75),
        "wider_return_tolerance_4": replace(baseline, return_offset_ratio=1.0),
        "longer_missing_window": replace(baseline, max_missing_steps=30),
        "longer_missing_window_2": replace(baseline, max_missing_steps=45),
        "longer_missing_window_3": replace(baseline, max_missing_steps=60),
        "longer_pending_window": replace(baseline, pending_expiry_steps=24),
        "longer_pending_window_2": replace(baseline, pending_expiry_steps=36),
        "longer_pending_window_3": replace(baseline, pending_expiry_steps=48),
        "lower_release_aperture": replace(baseline, released_aperture_meters=.050),
        "wider_stability_tolerance": replace(baseline, stable_motion_ratio=.25),
        "wider_stability_tolerance_2": replace(baseline, stable_motion_ratio=.35),
        "wider_support_region": replace(baseline, support_horizontal_ratio=.55,
                                         support_vertical_tolerance_ratio=.15),
        "wider_support_region_2": replace(baseline, support_horizontal_ratio=.65,
                                           support_vertical_tolerance_ratio=.20),
        "fewer_confirmation_frames": replace(baseline, confirmation_frames=2),
        "wider_association": replace(baseline, association_max_distance=.15,
                                      association_initial_distance=.09,
                                      association_distance_growth=.03),
        "joint_development_candidate": replace(
            baseline, lift_center_rise_ratio=.25, lift_bottom_rise_ratio=.35,
            return_offset_ratio=.40, max_missing_steps=30,
            released_aperture_meters=.050, stable_motion_ratio=.25,
        ),
        "joint_lower_lift_and_longer_pending": replace(
            baseline, lift_center_rise_ratio=.15, lift_bottom_rise_ratio=.25,
            pending_expiry_steps=24, max_missing_steps=30,
        ),
        "joint_moderate_lift_and_wider_support": replace(
            baseline, lift_center_rise_ratio=.20, lift_bottom_rise_ratio=.30,
            support_horizontal_ratio=.55, support_vertical_tolerance_ratio=.15,
            return_offset_ratio=.55, max_missing_steps=30,
        ),
        "joint_broad_bounded_candidate": replace(
            baseline, lift_center_rise_ratio=.10, lift_bottom_rise_ratio=.20,
            pending_expiry_steps=24, max_missing_steps=30, return_offset_ratio=.55,
            released_aperture_meters=.050, stable_motion_ratio=.25,
            support_horizontal_ratio=.55, support_vertical_tolerance_ratio=.15,
        ),
        "joint_broad_lower_lift": replace(
            baseline, lift_center_rise_ratio=.05, lift_bottom_rise_ratio=.15,
            pending_expiry_steps=36, max_missing_steps=30, return_offset_ratio=.75,
            released_aperture_meters=.050, stable_motion_ratio=.30,
            support_horizontal_ratio=.55, support_vertical_tolerance_ratio=.15,
        ),
        "joint_broad_missing_45": replace(
            baseline, lift_center_rise_ratio=.10, lift_bottom_rise_ratio=.20,
            pending_expiry_steps=36, max_missing_steps=45, return_offset_ratio=.55,
            released_aperture_meters=.050, stable_motion_ratio=.25,
            support_horizontal_ratio=.55, support_vertical_tolerance_ratio=.15,
        ),
        "joint_broad_wider_support": replace(
            baseline, lift_center_rise_ratio=.10, lift_bottom_rise_ratio=.20,
            pending_expiry_steps=36, max_missing_steps=30, return_offset_ratio=.55,
            released_aperture_meters=.050, stable_motion_ratio=.25,
            support_horizontal_ratio=.65, support_vertical_tolerance_ratio=.20,
        ),
        "joint_broad_wider_association": replace(
            baseline, lift_center_rise_ratio=.10, lift_bottom_rise_ratio=.20,
            pending_expiry_steps=36, max_missing_steps=30, return_offset_ratio=.55,
            released_aperture_meters=.050, stable_motion_ratio=.25,
            support_horizontal_ratio=.55, support_vertical_tolerance_ratio=.15,
            association_max_distance=.15, association_initial_distance=.09,
            association_distance_growth=.03,
        ),
        "joint_broad_two_frame_confirmation": replace(
            baseline, lift_center_rise_ratio=.10, lift_bottom_rise_ratio=.20,
            pending_expiry_steps=36, max_missing_steps=30, return_offset_ratio=.55,
            released_aperture_meters=.050, stable_motion_ratio=.25,
            support_horizontal_ratio=.55, support_vertical_tolerance_ratio=.15,
            confirmation_frames=2,
        ),
    }


def summary_counts(summary, arm):
    return {task: {key: summary[task][arm][key]
                   for key in ("true_positive", "false_positive", "false_negative")}
            for task in ("T1", "T3")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-index", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/libero_mem")
    parser.add_argument("--tolerance", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("Tolerance must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    if {path.name for path in args.development_index} != {"index-T1", "index-T3"}:
        parser.error("Calibration requires exactly the locked T1 and T3 development indexes")

    baseline_config = BowlCycleConfig.load()
    print("Verifying development inputs and replaying rgb_relations_v2", flush=True)
    cached = {}
    templates = [row for path in args.development_index for row in
                 development_episodes(path, args.data_dir, BowlCycleWriter.VERSION,
                                      BowlCycleConfig(), frame_cache=cached)]
    if len(templates) != 198 or len({row["episode"] for row in templates}) != 198:
        raise ValueError("Calibration requires all 198 locked T1/T3 episodes")
    v1_archive = json.loads((ROOT / "experiments/audits/"
                            "bowl-cycle-v2-development-alias-guard-20261118.json").read_text(
                                encoding="utf-8"))
    archived_v2 = {row["episode"]: row["proposed"] for row in v1_archive["episodes"]}
    current_v2 = {row["episode"]: row["proposed"] for row in templates}
    if archived_v2 != current_v2:
        raise ValueError("Current rgb_relations_v2 replay differs from its archived predictions")
    if set(cached) != {row["episode"] for row in templates}:
        raise ValueError("Cached detection roster differs from the verified annotation roster")

    results = {}
    winner_name = None
    winner_config = None
    winner_episodes = None
    for name, config in candidates(baseline_config).items():
        print("evaluating", name, flush=True)
        episodes, metrics = evaluate_candidate(templates, cached, config, args.tolerance)
        all_metrics = metrics["all"]["proposed"]
        task_metrics = [metrics[task]["proposed"] for task in ("T1", "T3")]
        admissible = all_metrics["false_positive"] == 0
        ranking = (sum(row["true_positive"] for row in task_metrics),
                   min(row["true_positive"] for row in task_metrics),
                   -sum(row["false_negative"] for row in task_metrics))
        results[name] = dict(config=config.to_dict(), admissible_zero_false_positives=admissible,
                             summary=metrics, ranking=list(ranking))
        print(name, "TP/FP/FN", summary_counts(metrics, "proposed"), flush=True)
        if admissible and (winner_config is None or ranking > winner_ranking):
            winner_name, winner_config, winner_episodes = name, config, episodes
            winner_ranking = ranking

    if winner_config is None:
        raise ValueError("No development candidate met the zero-false-positive constraint")
    result = dict(
        status="complete",
        classification="CUSTOM_WRITER_CALIBRATION_NON_COMPARABLE_OFFICIAL_LIBERO",
        writer=BowlCycleWriterV3.VERSION,
        method="one-factor screen and bounded joint sweep on author demonstration milestones",
        annotation_limit="Milestones are task-progress labels, not object-contact labels.",
        initial_config_sha256=sha256_file(CONFIG_PATH),
        writer_code_sha256=versioned_relation_source_hashes(ROOT, BowlCycleWriterV3.VERSION),
        v2_writer_code_sha256=versioned_relation_source_hashes(ROOT, BowlCycleWriter.VERSION),
        calibration_script_sha256=sha256_file(Path(__file__)),
        tolerance_frames=args.tolerance,
        development_roster=[row["episode"] for row in templates],
        development_receipt_hashes=sorted({row["source_receipt_sha256"] for row in templates}),
        frame_count=sum(row["frames"] for row in templates),
        v1_baseline=summary_counts(score(templates, args.tolerance), "baseline"),
        v2_current=summary_counts(score(templates, args.tolerance), "proposed"),
        development_gate=dict(
            required_true_positives={"T1": 68, "T3": 126},
            selected_passed=(results[winner_name]["summary"]["T1"]["proposed"]["true_positive"] >= 68
                             and results[winner_name]["summary"]["T3"]["proposed"]["true_positive"] >= 126),
            selected_zero_false_positives=results[winner_name]["admissible_zero_false_positives"],
        ),
        candidates=results,
        selected_episodes=[{
            "episode": row["episode"], "task": row["task"],
            "milestones": row["milestones"], "baseline": row["baseline"],
            "proposed": row["proposed"], "proposed_match": row["proposed_match"],
        } for row in winner_episodes],
        selected_candidate=winner_name,
        selected_config=winner_config.to_dict(),
        selected_summary=score(winner_episodes, args.tolerance),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(selected=winner_name, metrics=summary_counts(result["selected_summary"], "proposed"),
                          output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
