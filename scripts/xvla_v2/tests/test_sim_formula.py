import sys, numpy as np
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abs_actions import delta_to_abs6d, rot6d_to_mat, rotation_angle_deg, mat_to_rot6d
import robosuite as suite
from robosuite import load_controller_config
cfg = load_controller_config(default_controller="OSC_POSE")
env = suite.make("Lift", robots="Panda", controller_configs=cfg, has_renderer=False,
                 has_offscreen_renderer=False, use_camera_obs=False, control_freq=20)
rng = np.random.default_rng(0)
obs = env.reset()
rs, acts, gp, go, eeori = [], [], [], [], []
for t in range(300):
    a = np.concatenate([rng.uniform(-1, 1, 3) * 0.6, rng.uniform(-1, 1, 3) * 0.4, [rng.choice([-1., 1.])]])
    if t % 17 == 0: a[3:6] = 0.0          # exact-zero rotation rows
    if t % 23 == 0: a[:3] *= 1.8          # some clipped rows
    rob = env.robots[0]
    rs.append(np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]))
    acts.append(a)
    obs, *_ = env.step(a)
    c = rob.controller
    gp.append(c.goal_pos.copy()); go.append(c.goal_ori.copy()); eeori.append(c.ee_ori_mat.copy())
rs, acts, gp, go = map(np.asarray, (rs, acts, gp, go))
for frame in ("panda_rz_m90", "identity"):
    out, info = delta_to_abs6d(rs.astype(np.float32), acts.astype(np.float32), frame)
    pe = np.abs(out[:, :3] - gp).max()
    re = rotation_angle_deg(rot6d_to_mat(out[:, 3:9].astype(np.float64)), go)
    pe_med=np.median(np.linalg.norm(out[:, :3] - gp,axis=1)); print(frame, "max pos err m", f"{pe:.2e}", "median", f"{pe_med:.2e}", "rot err deg max", f"{re.max():.4f}", "median", f"{np.median(re):.4f}", info["zero_rotation_rows"], info["clipped_rows"])
# layout check: our rot6d of the true goal equals client Mat_to_Rotate6D
cl = np.concatenate([go[:, :3, 0], go[:, :3, 1]], -1)
print("layout max diff", np.abs(mat_to_rot6d(go) - cl).max())
