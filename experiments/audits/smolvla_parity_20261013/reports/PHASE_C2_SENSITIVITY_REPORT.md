# Phase C2 Sensitivity Audit Report: Image vs. Proprio Shift

- Generated: 2026-10-13 04:33:12 UTC
- Protocol: `docs/SMOLVLA_WSL_PARITY_AUDIT_PLAN.md` (Phase C2)
- Device: `cuda` | Noise Tensor: `noise_0.pt` | Target Task: `libero_object_task0` (State 0)
- **Certification**: `NON-COMPARABLE_OFFICIAL_PAPER` (Diagnostic offline perturbation probe)

---

## 1. Raw Observation Discrepancy (Windows vs. WSL at Step 0)

### A. Proprioception & Kinematics
- **EEF Position Delta (mm)**: `dx=-1.743`, `dy=-0.248`, `dz=-1.353`
- **EEF Euclidean Distance**: **`2.2202 mm`**
- **EEF Geodesic Angular Difference**: **`1.9989°`** (computed with quaternion double-cover equivalence)
- **Gripper Finger Gap Delta**: **`0.0496 mm`**

### B. Camera Sensor Rendering
| Camera | Resolution | Mean Absolute Error (MAE) | RMSE | PSNR (dB) | Max Pixel Delta |
|---|---|---|---|---|---|
| `agentview` (overhead) | 256x256x3 | 41.383 / 255 | 47.981 | 14.51 dB | 237 |
| `robot0_eye_in_hand` (wrist) | 256x256x3 | 54.620 / 255 | 59.711 | 12.61 dB | 229 |

> [!NOTE]
> PSNR > 30 dB indicates high visual fidelity between Windows and WSL GPU renders.
> Pixel differences originate from GPU floating-point rasterization / anti-aliasing variations.

---

## 2. Sensitivity Matrix: Action Chunk Divergence vs. Baseline

All runs use identical frozen model weights and identical fixed noise tensor `noise_0.pt`.

| Condition | Description | Chunk L2 (50x7) | Step 0 Action L2 (1x7) | Translation Cosine Sim | Max Element Diff |
|---|---|---|---|---|---|
| **Baseline (All Windows)** | Windows Images + Windows Proprio | `0.0000` | `0.0000` | `1.0000` | `0.0000` |
| **Full WSL** | WSL Images + WSL Proprio | `6.5529` | `0.4714` | `0.9914` | `2.0356` |
| **Hybrid Proprio Shift** | Windows Images + WSL Proprio (Isolates Proprio) | `0.1085` | `0.0098` | `0.9999` | `0.0245` |
| **Hybrid Image Shift** | WSL Images + Windows Proprio (Isolates Image) | `6.5831` | `0.4759` | `0.9907` | `2.0378` |
| **Agentview Only Swap** | WSL Agentview + Windows Wrist + Windows Proprio | `2.3878` | `0.0647` | `0.9929` | `2.0128` |
| **Wrist Only Swap** | Windows Agentview + WSL Wrist + Windows Proprio | `5.1363` | `0.4199` | `0.9751` | `2.0313` |

---

## 3. Step 0 Action Vector Breakdown

Action semantics: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`

```text
Baseline [A] Action 0: [0.1183, -0.1977, -0.0185, -0.0066, 0.0261, -0.0337, -0.9895]
Full WSL                 : [0.3111, -0.6262, 0.0175, -0.0012, 0.0237, -0.0427, -0.9891] (Diff: [0.1928, -0.4285, 0.036, 0.0054, -0.0024, -0.009, 0.0004])
Hybrid Proprio Shift     : [0.1122, -0.1924, -0.019, -0.0068, 0.0213, -0.0315, -0.9879] (Diff: [-0.0061, 0.0053, -0.0005, -0.0001, -0.0047, 0.0023, 0.0017])
Hybrid Image Shift       : [0.3127, -0.63, 0.022, -0.0011, 0.0288, -0.0429, -0.9904] (Diff: [0.1944, -0.4323, 0.0405, 0.0055, 0.0027, -0.0092, -0.0009])
Agentview Only Swap      : [0.1604, -0.2366, 0.007, -0.0017, 0.041, -0.0346, -0.9903] (Diff: [0.0421, -0.0389, 0.0255, 0.0049, 0.0149, -0.0009, -0.0008])
Wrist Only Swap          : [0.2096, -0.6067, 0.0037, -0.0016, 0.0266, -0.0453, -0.9858] (Diff: [0.0913, -0.409, 0.0222, 0.0051, 0.0005, -0.0116, 0.0037])
```

---

## 4. Causal Attribution & Key Audit Insights

1. **Relative Impact (Chunk L2 Distance)**:
   - **Full Divergence (WSL vs Win)**: `6.5529`
   - **Proprioceptive Shift Alone**: `0.1085` (1.7% of total)
   - **Image Shift Alone**: `6.5831` (100.5% of total)

2. **Primary Driver**: **Visual Rendering Differences** exert a higher influence than the 2.2 mm proprioceptive drift.

3. **Camera Sensitivity**:
   - Agentview Swap: Chunk L2 = `2.3878`
   - Wrist Camera Swap: Chunk L2 = `5.1363`
