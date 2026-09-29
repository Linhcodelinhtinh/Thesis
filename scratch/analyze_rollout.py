import json
from pathlib import Path

log_path = Path("experiments/results/raw_baseline/libero_object_task1_s10_init0/model_output.jsonl")
if not log_path.exists():
    print("Log path does not exist!")
    exit(1)

with open(log_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

print(f"Total logged steps: {len(lines)}")

print(f"{'Step':>4} | {'Rep':>3} | {'RawNorm[-1]':>11} | {'Unnorm[-1]':>10} | {'SimAct[-1]':>10} | {'GripperQpos':>17} | {'EEF_Z':>7} | {'Obj_DeltaZ':>10} | {'BothCont':>8} | {'AnyCont':>7}")
print("-" * 105)

for i in range(0, len(lines), 20):
    l = lines[i]
    step = l["step"]
    rep = "YES" if l.get("is_replanned") else "   "
    rn = l.get("raw_normalized_action")
    rn_g = f"{rn[-1]:.3f}" if rn else "None"
    un = l.get("unnormalized_action")
    un_g = f"{un[-1]:.3f}" if un else "None"
    act = l.get("action")
    act_g = f"{act[-1]:.3f}" if act else "None"
    g_qpos = l.get("gripper_qpos_actual")
    gq_str = f"[{g_qpos[0]:.3f}, {g_qpos[1]:.3f}]" if g_qpos else "None"
    eef = l.get("eef_pos_actual")
    eef_z = f"{eef[2]:.3f}" if eef else "None"
    diag = l.get("contact_diagnostics") or {}
    dz_val = diag.get("delta_z")
    dz = f"{dz_val:.3f}" if dz_val is not None else "None"
    cb = str(diag.get("both_fingers_contact", False))
    ca = str(diag.get("any_contact", False))
    print(f"{step:4d} | {rep} | {rn_g:>11} | {un_g:>10} | {act_g:>10} | {gq_str:>17} | {eef_z:>7} | {dz:>10} | {cb:>8} | {ca:>7}")

# Check any steps with contact
contact_steps = [l["step"] for l in lines if (l.get("contact_diagnostics") or {}).get("any_contact")]
both_contact_steps = [l["step"] for l in lines if (l.get("contact_diagnostics") or {}).get("both_fingers_contact")]
lifted_steps = [l["step"] for l in lines if (l.get("contact_diagnostics") or {}).get("is_lifted")]

print("\nDetailed Steps 0 to 25:")
for i in range(25):
    l = lines[i]
    act = l.get('action')[-1]
    un = l.get('unnormalized_action')[-1]
    rn = l.get('raw_normalized_action')[-1]
    q = l.get('gripper_qpos_actual')
    print(f"Step {i:2d}: SimAct={act:+.3f} | Unnorm={un:+.3f} | RawNorm={rn:+.3f} | qpos=[{q[0]:+.4f}, {q[1]:+.4f}]")

print("\nDetailed Steps 110 to 135:")
for i in range(110, 135):
    l = lines[i]
    act = l.get('action')[-1]
    un = l.get('unnormalized_action')[-1]
    rn = l.get('raw_normalized_action')[-1]
    q = l.get('gripper_qpos_actual')
    print(f"Step {i:2d}: SimAct={act:+.3f} | Unnorm={un:+.3f} | RawNorm={rn:+.3f} | qpos=[{q[0]:+.4f}, {q[1]:+.4f}]")

import numpy as np
obj_pos = np.array([0.046, -0.103, 0.0])
print("\nStep | EEF (x, y, z) | Dist to Obj (XY) | Dist to Obj (3D)")
print("-" * 60)
for i in range(0, len(lines), 20):
    l = lines[i]
    eef = np.array(l.get('eef_pos_actual'))
    d_xy = np.linalg.norm(eef[:2] - obj_pos[:2])
    d_3d = np.linalg.norm(eef - obj_pos)
    print(f"{l['step']:4d} | [{eef[0]:+.3f}, {eef[1]:+.3f}, {eef[2]:+.3f}] | {d_xy:.3f} m | {d_3d:.3f} m")

# Find minimum distance to object
min_d = 999.0
min_step = -1
min_eef = None
for l in lines:
    eef = np.array(l.get('eef_pos_actual'))
    d = np.linalg.norm(eef - obj_pos)
    if d < min_d:
        min_d = d
        min_step = l['step']
        min_eef = eef

print("\nFirst 15 Action Vectors (dx, dy, dz, dr, dp, dyaw, grip):")
for i in range(15):
    l = lines[i]
    act = l['action']
    s = l['step']
    print(f"Step {s:2d}: dx={act[0]:+.3f}, dy={act[1]:+.3f}, dz={act[2]:+.3f}, grip={act[6]:+.3f}")
