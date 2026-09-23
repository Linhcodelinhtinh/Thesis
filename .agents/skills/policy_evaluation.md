# Policy Evaluation Skill

## Trigger When:
- Running closed-loop policy evaluation episodes.
- Measuring success rates across benchmark tasks.
- Evaluating a candidate VLA model checkpoint.
- Performing comparative analysis between models.

## Pre-requisites & Rules:
1. Ensure the simulation environment is verified (Phase 2 smoke test passed).
2. Ensure the model adapter interface has passed interface audit (Phase 4).
3. Do NOT modify the environment horizon (default 1000 steps).
4. Do NOT modify the control frequency (default 20 Hz).
5. Always use official initial states for evaluation episodes.
6. Record complete episode trajectories (`episode.json`, `trajectory.npz`, `video.mp4`).

## Standard Metrics:
- **Task Success Rate (SR)**: Binary success predicate checked via `env.check_success()`.
- **Grasp Success Rate**: Percentage of episodes where the robot achieved a secure grasp on the target object.
- **Placement Success Rate**: Percentage of episodes where the target object reached the target fixture.
- **Episode Length / Timesteps to Success**: Number of environment steps before success condition was triggered.
- **Inference Latency**: Mean and 95th percentile latency per inference call (ms).
- **Peak VRAM**: Peak GPU memory consumption during episode rollout.

## Failure Classification:
Every unsuccessful episode must be categorized into one of:
- `reach_fail`: End-effector failed to reach the target object within distance threshold.
- `grasp_fail`: Gripper closed without grasping the object or dropped it immediately.
- `transport_fail`: Object dropped during transit to destination.
- `placement_fail`: Object released outside destination target area.
- `timeout`: Policy ran out of steps without reaching success state.
- `controller_divergence`: Extreme actions caused physics destabilization or safety limits.

## Execution Checklist:
1. Load benchmark suite configuration.
2. Select target task and initial state index.
3. Reset environment and pass initial observation to policy.
4. Run closed-loop rollout until `done` or `horizon` reached.
5. Log telemetry, frames, and actions.
6. Save episode artifacts to `experiments/<benchmark>/<model>/`.
