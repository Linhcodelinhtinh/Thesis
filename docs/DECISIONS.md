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

---

## ADR-0009: Action Space and Gripper Polarity Alignment Protocol for VLA Adapters
- **Status**: Accepted
- **Context**: SmolVLA (`lerobot/smolvla_libero`) was trained on `lerobot/libero`, which originated from the RLDS dataset `openvla/modified_libero_rlds`. RLDS formats standardize gripper actions with the convention $+1 = \text{Open}, -1 = \text{Close}$. Conversely, robosuite's native `PandaGripper` controller uses $-1 = \text{Open}, +1 = \text{Close}$. Sending raw policy outputs directly to robosuite causes the gripper to close during approach and open during grasp, resulting in 0% grasp success.
- **Decision**:
  - The model adapter (`SmolVLAAdapter`) must explicitly align the gripper action space via `invert_gripper_action: bool = True` during `postprocess()`:
    $$a_{\text{sim}}[-1] = -1.0 \times a_{\text{vla}}[-1]$$
  - This transformation is placed strictly within the model adapter layer per ADR-0004 (Layer Separation), leaving simulator and controller layers unchanged.
  - The model manifest (`smolvla_libero.yaml`) and audit framework (`src/models/model_manifest.py`) must record and audit `gripper_action_polarity: "INVERTED_RLDS_TO_ROBOSUITE"` to prevent silent or undocumented transformations.
  - Telemetry logs (`model_output.jsonl`) must record both the raw policy action (`unnormalized_action`) and the final executed simulator action (`executed_sim_action`) for transparent auditability.
- **Consequences**: Restores correct physical grasping behavior while preserving strict traceability and compliance with AGENTS.md Rule 3.

---

## ADR-0010: Runtime Environment Coordination and Strict Certification Tagging for VLA Evaluation
- **Status**: Accepted
- **Context**: The official reference LIBERO benchmark stack (`envs/libero_reference.yml`) relies on legacy dependencies (Python 3.8.13, robosuite 1.4.0, bddl 1.0.1, numpy 1.22.4). In contrast, modern VLA policies such as SmolVLA (`lerobot/smolvla_libero`) require modern dependency stacks (Python 3.10+, PyTorch >= 2.2, transformers >= 4.40). Running closed-loop evaluation requires coordinating the simulator and the policy model while adhering to isolated environment policies (ADR-0006) and benchmark honesty rules (AGENTS.md Rules 2, 6, 9).
- **Decision**:
  1. **Direct In-Process Execution**: For Phase 8 acceptance benchmark and baseline development, closed-loop evaluation runs within the verified Python 3.10 evaluation environment (`envs/smolvla.yml`), which incorporates robosuite and MuJoCo alongside PyTorch/LeRobot.
  2. **Mandatory Execution Tier Tagging**: Because the runtime Python version is 3.10 (rather than pinned 3.8.13), all evaluation artifacts (`episode.json`, `summary.json`) must explicitly record `execution_tier: "LIBERO-DERIVED"` and mark the run as `NON-COMPARABLE` against official legacy LIBERO publication scores per AGENTS.md Rule 9.
  3. **Strict Physical Invariant Enforcement**: Despite the Python 3.10 runtime, all physical simulation parameters must be locked to benchmark standard: 20 Hz control frequency, 1000-step horizon, Franka Panda with OSC_POSE controller, official initial state arrays (0..49), and 128x128 simulation camera rendering. The model adapter (`SmolVLAAdapter`) is solely responsible for resizing 128x128 images to 256x256 within its `preprocess()` method to fulfill model input contracts without altering simulator physics or rendering geometry.
  4. **IPC Protocol for Strict Certification**: If certified comparison against the legacy Python 3.8.13 reference stack is required, a client-server IPC architecture (shared-memory or local socket between `envs/libero_reference.yml` simulator host and `envs/smolvla.yml` inference worker) shall be utilized rather than modifying reference dependencies.
- **Consequences**: Preserves architectural purity and reproducibility, ensures zero dependency contamination, and completely eliminates misleading benchmark claims.

---

## ADR-0011: Native Simulation Camera Resolution and Controller Configuration Provenance for VLA Evaluation
- **Status**: Accepted
- **Context**: 
  - ADR-0010 item 3 initially specified a strict 128×128 simulation camera rendering resolution matching legacy LIBERO, with `SmolVLAAdapter` resizing observations to 256×256.
  - However, official training of SmolVLA (`lerobot/smolvla_libero`) and the underlying LeRobot dataset used native 256×256 camera rendering. Empirical evaluation during Phase 8 demonstrated that rendering at 128×128 followed by bilinear upsampling introduced visual blur and artifacting that severely degraded visual feature alignment, causing false grasp misses even on deterministic pick-and-place tasks.
  - Furthermore, undocumented assertions regarding controller gains (e.g. `kp=150`) risked violating AGENTS.md Rule 1 and Rule 6 without traceable provenance from official Robosuite/LIBERO controller definitions.
- **Decision**:
  1. **Dual Resolution Support with Transparent Tier Tagging**:
     - The simulation environment supports both `--camera-resolution 128` (legacy standard) and `--camera-resolution 256` (native SmolVLA/LeRobot training distribution).
     - Whenever native 256×256 rendering is selected, the run MUST be explicitly classified under the `LIBERO-DERIVED (HOST_PY...)` execution tier with certification `NON-COMPARABLE_OFFICIAL_PAPER`.
     - Certified strict comparison against official legacy LIBERO papers remains reserved for 128×128 rendering in the isolated reference environment (`envs/libero_reference.yml`).
  2. **Controller Configuration Provenance**:
     - All controller parameters are strictly derived from official Robosuite definitions for `OSC_POSE` with Franka Panda (`robosuite/controllers/config/osc_pose.json` and `robosuite/models/robots/manipulators/panda_robot.py`), operating at 20 Hz (control step dt = 0.05s).
     - Individual parameter values (impedance gains `kp`, `damping_ratio`) must never be hard-coded or asserted without direct reference to the loaded controller specification.
- **Consequences**:
  - Restores the natural perceptual distribution expected by modern 256×256 VLAs while preserving strict benchmark honesty and eliminating undocumented simulation modifications.

---

## ADR-0012: Selection and Formal Freeze of SmolVLA as Primary V1 Baseline for Memory (V2) Research
- **Status**: Accepted
- **Context**: 
  - Progressing to V2 (Memory System) requires a completely frozen, mathematically reproducible raw-policy baseline without any memory augmentation (ADR-0003, AGENTS.md Primary Objective).
  - Two model candidates were evaluated: SmolVLA (`lerobot/smolvla_libero` @ `31d453f`) and MiniVLA (`openvla/minivla` VQ-LIBERO-90).
  - SmolVLA underwent rigorous audit across all Promotion Gates G0–G4:
    - **G0 (Infrastructure)**: Deterministic seed reproducibility, complete test suite passing (105 passed, 0 failed), zero missing files across 200 evaluated episode runs.
    - **G1 (Basic Manipulation)**: Verified capability across atomic pick, place, push, and articulation tasks (Grasp: 68.1%, Lift: 50.0%, Place: 53.8%).
    - **G2 (Benchmark Significance)**: Achieved 70.0% on acceptance 10-task suite [95% CI: 54.6%–81.9%] and 48.13% across full 40-task benchmark across initial states 0..3 (Goal: 67.5% [27/40], Object: 50.0% [20/40], Spatial: 47.5% [19/40], LIBERO-10: 27.5% [11/40]), materially above random baseline (~0%).
    - **G3 (Failure Attribution)**: Behavioral failure phase is tracked for 100% of failures (REACH: 67.5% [56/83], LIFT: 12.0% [10/83], GRASP: 8.4% [7/83], TIMEOUT: 6.0% [5/83], PLACEMENT: 4.8% [4/83], TRANSPORT: 1.2% [1/83]). Causal root-cause classification remains unisolated for 85.5% (71/83) under diagnostic code F0 due to multi-object scene geometries; accepted as an aggregate behavioral phase baseline for memory research rather than fine-grained causal diagnosis.
    - **G4 (Latency Stability)**: Real-time inference latency stable on GPU (mean 2841 ms / chunk call, amortized ~57 ms/step, 0 OOM / runtime crashes).
  - Multi-stage compositional tasks (`LIBERO-10`) demonstrated a pronounced degradation to 27.5% SR, with 67.5% of failures terminating at the transition phase (REACH on subsequent objects). This provides an authentic empirical foundation and clear quantitative headroom for memory augmentation.
- **Decision**:
  1. **Primary Model Lock**: Select and lock `lerobot/smolvla_libero` (commit `31d453f7edd78c839a8bbc39744a292686daf0de`) as the Primary Frozen Baseline for V1.
  2. **Canonical Baseline Config**: Author and freeze `configs/models/selected_baseline.yaml` specifying Franka Panda, OSC_POSE controller, 20 Hz, receding horizon chunk execution ($s=50$), native 256×256 camera rendering, 8-dimensional proprioceptive state vector, and DIRECT gripper polarity (robosuite native convention `-1 = Open, +1 = Close` per `lerobot/libero`).
  3. **Cryptographic Artifact Freeze**: Generate immutable baseline records in `experiments/baseline_v1/`, including `FROZEN_BASELINE_MANIFEST.yaml` (SHA-256 checksums of all 1,000 raw benchmark files), `V1_BASELINE_REPORT.md`, and `freeze_audit_signoff.json`.
  4. **Transparent Evaluation Tier**: Mark baseline explicitly as `LIBERO-DERIVED (HOST_PY3.12)` with certification `NON-COMPARABLE_OFFICIAL_PAPER`. Its scientific role is an internal, locked control baseline for paired evaluation against V2 memory models under identical host conditions (40 tasks, initial states 0..3), not official LIBERO score reproduction.
  5. **Strict Immutability**: The V1 baseline weights, adapter, and evaluation pipeline are frozen. All V2 research will evaluate memory mechanisms as an additive wrapper without modifying the underlying raw-policy baseline.
- **Consequences**:
  - Provides a definitive, transparent, and reproducible foundation for the entire thesis.
  - Any memory enhancement in V2 can be measured cleanly as a differential ($\Delta \text{SR}$) against this locked baseline.



