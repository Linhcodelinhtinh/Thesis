# Offline Parity Audit Report (Phase C1)

- Generated: 2026-10-13 04:21:38 UTC
- Target Protocol: `docs/SMOLVLA_WSL_PARITY_AUDIT_PLAN.md`

## 1. Summary of Comparisons

| Comparison Pair | Snapshot | Noise | Stage 1 (Raw State) L2 | Stage 2 (Prep State) L2 | Stage 3 (Raw Model Chunk) L2 | Stage 4 (Action Chunk) L2 | Max Action Diff |
|---|---|---|---|---|---|---|---|
| `win_cpu` vs `wsl_cpu` | `snap_init_state0` | `noise_0` | 0.000000e+00 | 0.000000e+00 | 1.000200e-01 | 2.631007e-02 | 4.771858e-03 |
| `win_cpu` vs `wsl_cpu` | `snap_init_state0` | `noise_1` | 0.000000e+00 | 0.000000e+00 | 1.068807e-01 | 2.649295e-02 | 5.674094e-03 |
| `win_cpu` vs `wsl_cpu` | `snap_near_grasp` | `noise_0` | 0.000000e+00 | 0.000000e+00 | 7.938911e-02 | 2.377805e-02 | 7.417619e-03 |
| `win_cpu` vs `wsl_cpu` | `snap_near_grasp` | `noise_1` | 0.000000e+00 | 0.000000e+00 | 7.649233e-02 | 2.453882e-02 | 6.699622e-03 |
| `wsl_cpu` vs `wsl_cuda` | `snap_init_state0` | `noise_0` | 0.000000e+00 | 0.000000e+00 | 1.163951e-01 | 2.873550e-02 | 6.575972e-03 |
| `wsl_cpu` vs `wsl_cuda` | `snap_init_state0` | `noise_1` | 0.000000e+00 | 0.000000e+00 | 1.181083e-01 | 2.775567e-02 | 5.766422e-03 |
| `wsl_cpu` vs `wsl_cuda` | `snap_near_grasp` | `noise_0` | 0.000000e+00 | 0.000000e+00 | 6.735368e-02 | 1.943004e-02 | 3.948629e-03 |
| `wsl_cpu` vs `wsl_cuda` | `snap_near_grasp` | `noise_1` | 0.000000e+00 | 0.000000e+00 | 7.302818e-02 | 2.310660e-02 | 4.342467e-03 |
| `win_cpu` vs `wsl_cuda` | `snap_init_state0` | `noise_0` | 0.000000e+00 | 0.000000e+00 | 9.975979e-02 | 2.939357e-02 | 5.700499e-03 |
| `win_cpu` vs `wsl_cuda` | `snap_init_state0` | `noise_1` | 0.000000e+00 | 0.000000e+00 | 9.750282e-02 | 2.867881e-02 | 5.811244e-03 |
| `win_cpu` vs `wsl_cuda` | `snap_near_grasp` | `noise_0` | 0.000000e+00 | 0.000000e+00 | 8.300006e-02 | 2.278739e-02 | 6.361902e-03 |
| `win_cpu` vs `wsl_cuda` | `snap_near_grasp` | `noise_1` | 0.000000e+00 | 0.000000e+00 | 8.609982e-02 | 2.469680e-02 | 4.740208e-03 |

## 2. Detailed Dimension-wise Breakdown (Stage 4 Actions)

Action semantics: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`

### `win_cpu vs wsl_cpu` — `snap_init_state0` — `noise_0`
- **Full Chunk (50x7) L2**: `0.026310`
- **Step 0 Action (1x7) L2**: `0.003100`
- **Max Absolute Diff**: `0.004772`
- **Max Diff per dimension**: `[0.004772, 0.004727, 0.002973, 0.001084, 0.000671, 0.00081, 0.002579]`

### `win_cpu vs wsl_cpu` — `snap_init_state0` — `noise_1`
- **Full Chunk (50x7) L2**: `0.026493`
- **Step 0 Action (1x7) L2**: `0.002930`
- **Max Absolute Diff**: `0.005674`
- **Max Diff per dimension**: `[0.005674, 0.003563, 0.003938, 0.000904, 0.000595, 0.001005, 0.00236]`

### `win_cpu vs wsl_cpu` — `snap_near_grasp` — `noise_0`
- **Full Chunk (50x7) L2**: `0.023778`
- **Step 0 Action (1x7) L2**: `0.002902`
- **Max Absolute Diff**: `0.007418`
- **Max Diff per dimension**: `[0.003953, 0.003462, 0.007418, 0.000622, 0.000914, 0.000665, 0.002751]`

### `win_cpu vs wsl_cpu` — `snap_near_grasp` — `noise_1`
- **Full Chunk (50x7) L2**: `0.024539`
- **Step 0 Action (1x7) L2**: `0.002273`
- **Max Absolute Diff**: `0.006700`
- **Max Diff per dimension**: `[0.004002, 0.002564, 0.0067, 0.0005, 0.000832, 0.000704, 0.003362]`

### `wsl_cpu vs wsl_cuda` — `snap_init_state0` — `noise_0`
- **Full Chunk (50x7) L2**: `0.028735`
- **Step 0 Action (1x7) L2**: `0.003412`
- **Max Absolute Diff**: `0.006576`
- **Max Diff per dimension**: `[0.004375, 0.006576, 0.004836, 0.000688, 0.001428, 0.001061, 0.003522]`

### `wsl_cpu vs wsl_cuda` — `snap_init_state0` — `noise_1`
- **Full Chunk (50x7) L2**: `0.027756`
- **Step 0 Action (1x7) L2**: `0.003010`
- **Max Absolute Diff**: `0.005766`
- **Max Diff per dimension**: `[0.004304, 0.005766, 0.004445, 0.000802, 0.001498, 0.001236, 0.003344]`

### `wsl_cpu vs wsl_cuda` — `snap_near_grasp` — `noise_0`
- **Full Chunk (50x7) L2**: `0.019430`
- **Step 0 Action (1x7) L2**: `0.004710`
- **Max Absolute Diff**: `0.003949`
- **Max Diff per dimension**: `[0.003712, 0.003484, 0.003949, 0.000358, 0.000534, 0.000842, 0.003608]`

### `wsl_cpu vs wsl_cuda` — `snap_near_grasp` — `noise_1`
- **Full Chunk (50x7) L2**: `0.023107`
- **Step 0 Action (1x7) L2**: `0.004933`
- **Max Absolute Diff**: `0.004342`
- **Max Diff per dimension**: `[0.003709, 0.004342, 0.004054, 0.000598, 0.000412, 0.000735, 0.002733]`

### `win_cpu vs wsl_cuda` — `snap_init_state0` — `noise_0`
- **Full Chunk (50x7) L2**: `0.029394`
- **Step 0 Action (1x7) L2**: `0.003760`
- **Max Absolute Diff**: `0.005700`
- **Max Diff per dimension**: `[0.0057, 0.003882, 0.003382, 0.000396, 0.000808, 0.000996, 0.003148]`

### `win_cpu vs wsl_cuda` — `snap_init_state0` — `noise_1`
- **Full Chunk (50x7) L2**: `0.028679`
- **Step 0 Action (1x7) L2**: `0.003257`
- **Max Absolute Diff**: `0.005811`
- **Max Diff per dimension**: `[0.005811, 0.003151, 0.003435, 0.000348, 0.001122, 0.000891, 0.003507]`

### `win_cpu vs wsl_cuda` — `snap_near_grasp` — `noise_0`
- **Full Chunk (50x7) L2**: `0.022787`
- **Step 0 Action (1x7) L2**: `0.002881`
- **Max Absolute Diff**: `0.006362`
- **Max Diff per dimension**: `[0.00445, 0.002907, 0.006362, 0.000358, 0.00099, 0.001321, 0.00344]`

### `win_cpu vs wsl_cuda` — `snap_near_grasp` — `noise_1`
- **Full Chunk (50x7) L2**: `0.024697`
- **Step 0 Action (1x7) L2**: `0.003402`
- **Max Absolute Diff**: `0.004740`
- **Max Diff per dimension**: `[0.00474, 0.003778, 0.003638, 0.000394, 0.000942, 0.001438, 0.003458]`
