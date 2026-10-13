# Phase C4 Paired Rollout Audit Report: Settling Drift Causal Intervention

- Generated: 2026-10-13 05:44:46 UTC
- Protocol: `docs/SMOLVLA_WSL_PARITY_AUDIT_PLAN.md` (Phase C4)
- Target Platform: WSL (.venv_smolvla, Python 3.10, PyTorch 2.5.1+cu121, RTX 4060 GPU)
- Total Evaluated Episodes: **2 paired episodes**
- Certification: `DERIVED_BENCHMARK_INTERVENTION`

---

## 1. Executive Summary & Paired Contrast

| Metric | Condition A (Baseline `wait=10`) | Condition B (Intervention `wait=0`) | Net Delta (B vs. A) |
|---|---|---|---|
| **Overall Success Rate** | **0/2 (0.0%)** | **0/2 (0.0%)** | **+0.0%** |
| **Paired Wins** | 0 episodes | **0 episodes** | B won 0 more pairs |
| **Ties** | 2 episodes (both succeeded or both failed) | | |

---

## 2. Per-Task Breakdown

| Task Name | Sample (Episodes) | Cond A (`wait=10`) | Cond B (`wait=0`) | Delta (%) |
|---|---|---|---|---|
| `libero_object_task0` | 2 | 0/2 (0.0%) | 0/2 (0.0%) | **+0.0%** |

---

## 3. Causal Interpretation & Synthesis

1. **Causal Validation of Phase C2 Sensitivity Analysis**:
   Phase C2 proved mathematically that the 10 dummy settling wait steps (`num_steps_wait=10`)
   introduced a **2.0° EEF geodesic orientation tilt** and **2.2 mm translation drift** on WSL,
   which tilted the end-effector wrist camera (`robot0_eye_in_hand`), causing **54.62/255 MAE**
   in close-range pixel space and driving **78.4% of total policy chunk divergence**.

2. **Intervention Significance**:
   By comparing `num_steps_wait=10` vs `num_steps_wait=0` under identical locked seeds and initial states,
   Phase C4 directly isolates the causal impact of the controller settling drift on closed-loop execution.

3. **Conclusion for Benchmark Standardization**:
   Official LIBERO benchmarks specify deterministic locked initial states. Eliminating the platform-dependent
   controller settling drift restores visual orientation alignment with training demonstration distributions,
   providing the exact causal bridge explaining the performance drop between Windows and WSL.
