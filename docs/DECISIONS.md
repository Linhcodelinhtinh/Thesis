# Architecture Decision Records (ADR)

This document records the architectural and design decisions for **VLA Policy Evaluation Sandbox V1**.

---

## ADR-0001: Official LIBERO Stack as Primary Source of Truth
- **Status**: Accepted
- **Context**: Evaluating VLA manipulation policies requires a reliable, standard benchmark with verifiable ground truth.
- **Decision**: All task definitions, BDDL domain files, object assets, Franka Panda robot configurations, and initial state files are pinned directly to the official `Lifelong-Robot-Learning/LIBERO` repository at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- **Consequences**: No custom CAD meshes, hand-crafted scenes, or synthetic initial states may be introduced in the strict benchmark track.

---

## ADR-0002: Panda Robot & robosuite 1.4.0 Baseline
- **Status**: Accepted
- **Context**: LIBERO was built and validated against Franka Panda with the `OSC_POSE` controller in robosuite. The original paper and official reproduction stack specify `robosuite==1.4.0`.
- **Decision**: The primary benchmark baseline (Mode A: STRICT-LIBERO) must target Franka Panda, the default Panda gripper, 20 Hz control frequency, 1000-step horizon, and `robosuite==1.4.0`.
- **Consequences**: Ensures physical dynamics, collision models, and controller gains match the canonical LIBERO literature.

---

## ADR-0003: Exclusion of Memory Module in V1 Baseline
- **Status**: Accepted
- **Context**: The thesis focuses on external robot memory mechanisms. However, before testing memory additions, a rigorous, verified raw-policy baseline is required.
- **Decision**: V1 is strictly a policy baseline and evaluation sandbox. Memory mechanisms (text, spatial, hybrid) are deferred entirely to V2.
- **Consequences**: Eliminates confounding factors during raw policy audit and controller verification.

---

## ADR-0004: Strict Separation of Architecture Layers
- **Status**: Accepted
- **Context**: Tightly coupling model adapters with simulator internals causes bugs, makes model comparison impossible, and leads to undocumented workarounds.
- **Decision**: Maintain strict layer boundaries:
  - `src/simulator/`: Environment management, rendering, physics, and state extraction.
  - `src/models/`: Model adapters, tokenization, visual preprocessing, and action decoding.
  - `src/controllers/`: Robot kinematics, action mapping, and controller interfaces.
  - `src/evaluation/`: Rollout execution, diagnostics, metrics calculation, and reporting.
- **Consequences**: Components can be tested independently with isolated unit tests.

---

## ADR-0005: Fail-Fast Policy Over Silent Fallbacks
- **Status**: Accepted
- **Context**: Silent fallbacks (e.g. falling back to CPU when GPU OOMs, guessing missing actions, scripting grasps) compromise benchmark honesty.
- **Decision**: Any runtime anomaly, shape mismatch, or missing resource must raise an explicit exception and fail loudly.
- **Consequences**: Benchmarks are clean, repeatable, and truthful.

---

## ADR-0006: Multi-Environment Isolation Strategy (WSL & Envs)
- **Status**: Accepted
- **Context**: Different VLA models (e.g. SmolVLA, MiniVLA) and the legacy LIBERO benchmark stack (`robosuite==1.4.0`, Python 3.8, PyTorch 1.11) have mutually incompatible dependency graphs.
- **Decision**:
  - Store environment specifications under `envs/` (`libero_reference.yml`, `smolvla.yml`, `minivla.yml`, `training.yml`, `research.yml`).
  - Use dedicated virtual environments (`.venv_libero`, etc.) managed via `uv` or Conda inside WSL2/Linux for native MuJoCo/EGL execution and exact legacy dependencies.
  - Record full environment provenance in `resources/manifests/`.
- **Consequences**: Prevents dependency cross-contamination and allows running both strict benchmark reproduction and modern PyTorch VLA models without compromise.

---

## ADR-0007: Deferment of UR3 and Real Robot Transfer
- **Status**: Accepted
- **Context**: UR3 simulation and real robot evaluation are important for generalizability, but Franka Panda is the native embodiment of LIBERO.
- **Decision**: UR3 simulation is scheduled for V3, and real UR3 transfer is scheduled for V4. Neither is part of V1 benchmark acceptance.
- **Consequences**: Focuses immediate engineering efforts on reproducing the Panda baseline.

---

## ADR-0008: Strict-Mode Benchmark Invariants and Environment Labeling Policy
- **Status**: Accepted
- **Context**: LIBERO benchmark comparisons require exact adherence to physical and visual constants (20 Hz, horizon 1000, OSC_POSE controller, Franka Panda robot with PandaGripper, 128x128 resolution for `agentview` and `robot0_eye_in_hand`). Running on a non-reference host (such as Windows with Python 3.12 or WGL renderer) introduces subtle kinematic or dependency differences that invalidate official score reporting.
- **Decision**:
  - The simulation wrapper (`LiberoEnv`) shall enforce strict invariants when `mode="STRICT-LIBERO"`, raising `StrictInvariantViolationError` immediately upon any configuration deviation.
  - An alternative mode `mode="LIBERO-DERIVED"` is permitted for exploratory research, but runs under this mode shall never be reported as official LIBERO scores.
  - Every evaluation run shall inspect runtime provenance: if the execution environment deviates from the locked reference stack (`envs/libero_reference.yml`: Python 3.8.x, robosuite 1.4.0, bddl 1.0.1, Linux EGL), the run shall be tagged as `NON-COMPARABLE_HOST_SMOKE` and disqualified from official benchmark certification.
- **Consequences**: Guarantees zero silent deviation from official LIBERO evaluation protocol and prevents invalid score claims.

