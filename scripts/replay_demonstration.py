"""Official LIBERO Demonstration Replay Script (Phase 3).

Replays an official demonstration episode using the exact environment reconstruction
per Phase 3 and SRS.md specifications:
  HDF5
   ├── env_args (or env_info)
   ├── problem_info
   ├── model_file / XML
   └── demo_0
        ├── actions
        ├── states
        └── init_state
            ↓
  construct EXACT demo environment
            ↓
  restore exact simulator state
            ↓
  env.step(recorded_action)
            ↓
  compare replayed_state ↔ recorded_state
            ↓
  check_success()
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.demo_replay import DemonstrationReplayer
from src.evaluation.rollout import save_video


def parse_args():
    parser = argparse.ArgumentParser(description="Replay official LIBERO demonstration.")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="resources/demonstrations/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",
        help="Path to demonstration HDF5 file.",
    )
    parser.add_argument("--demo", type=str, default="demo_0", help="Demonstration episode key.")
    parser.add_argument("--tolerance", type=float, default=0.1, help="Tracking tolerance threshold.")
    parser.add_argument("--strict-tolerance", action="store_true", default=False, help="Require max tracking error <= tolerance.")
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=False,
        help="Capture camera frames and render MP4 video.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/demo_replays",
        help="Output directory to save video or telemetry artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Official Demonstration Replay Engine (Phase 3)")
    print("=" * 60)
    print(f"HDF5 File: {args.hdf5}")
    print(f"Episode: {args.demo}")
    print(f"Tracking Tolerance: {args.tolerance} (strict={args.strict_tolerance})")

    replayer = DemonstrationReplayer(args.hdf5)

    # 1. Verify manifest hash
    manifest_path = Path("resources/manifests/demonstrations_manifest.yaml")
    if manifest_path.exists():
        print("Verifying cryptographic manifest hash...")
        replayer.verify_hdf5_hash(manifest_path)
        print("Cryptographic integrity: [VERIFIED]")

    # 2. Replay episode
    print(f"\nReplaying {args.demo} inside exactly reconstructed environment...")
    metrics = replayer.replay_episode(
        demo_name=args.demo,
        tracking_tolerance=args.tolerance,
        strict_tolerance=args.strict_tolerance,
        render=args.record_video,
    )

    print("\n" + "=" * 60)
    print("Replay Telemetry Summary")
    print("=" * 60)
    print(f"Steps Replayed: {metrics.num_steps}")
    print(f"Environment Success: {metrics.env_success}")
    print(f"Tracking Success: {metrics.tracking_success}")
    print(f"Overall Success: {metrics.success}")
    print(f"Mean Tracking Error: {metrics.mean_tracking_error:.6f}")
    print(f"Max Tracking Error: {metrics.max_tracking_error:.6f}")
    print(f"Divergence Count (> {args.tolerance}): {metrics.divergence_count}")

    # 3. Save video if requested
    if args.record_video and metrics.rendered_frames:
        out_dir = Path(args.output_dir) / f"{Path(args.hdf5).stem}_{args.demo}"
        out_dir.mkdir(parents=True, exist_ok=True)
        video_file = out_dir / "replay_video.mp4"
        success_save = save_video(metrics.rendered_frames, video_file, fps=20)
        if success_save:
            print(f"Replay Video saved to: {video_file}")
        else:
            print("Notice: Could not write video file (missing imageio/opencv).")

    print("=" * 60)
    return 0 if metrics.success else 1


if __name__ == "__main__":
    sys.exit(main())
