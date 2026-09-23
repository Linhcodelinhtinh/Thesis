# AGENTS.md — VLA Policy Evaluation Sandbox V1

This repository implements **VLA Policy Evaluation Sandbox V1**.

## Primary Objective
- Reproduce a controlled LIBERO subset strictly against official sources.
- Evaluate frozen VLA policies (e.g. SmolVLA, MiniVLA).
- Establish a reproducible raw-policy baseline before introducing any memory mechanisms.
- **No memory implementation in V1** (Memory is strictly scheduled for V2).

---

## Non-Negotiable Rules
1. **Never invent, fabricate, approximate, or substitute benchmark resources.**
2. **Never use mock/fallback policy execution in benchmark runs.**
3. **Never silently change model preprocessing or action decoding.**
4. **Never replace official benchmark robot/assets with visually similar assets.**
5. **Never report custom experiments as official LIBERO scores.**
6. **Every external resource must have provenance and version/hash.**
7. **Keep simulation, model adapter, controller, and evaluation layers separated.**
8. **If a checkpoint interface is uncertain, inspect its official config/code before implementing.**
9. **If a benchmark result depends on an undocumented modification, mark it NON-COMPARABLE.**
10. **Prefer failing loudly over silently falling back.**

---

## Source of Truth

### 1. LIBERO Benchmark Stack
- **Repository**: Official repository only (`https://github.com/Lifelong-Robot-Learning/LIBERO`)
- **Pinned Git Revision**: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- **Task definitions**: Official BDDL files directly from `libero` package.
- **Initial states**: Official benchmark initial-state arrays (`(50, state_dim)` per task).
- **Robot & Controller**: Franka Panda with default Panda gripper and `OSC_POSE` operational space controller.
- **Robosuite baseline**: `robosuite==1.4.0` in the official reference stack.
- **Assets**: Official assets downloaded from canonical release cache.

### 2. Models
- Original author repository / Hugging Face organization.
- Exact checkpoint revision / commit hash.
- Official model preprocessing (tokenization, resolution, transforms).
- Official model postprocessing (action unnormalization, VQ decoder, chunk execution).

### 3. Isolated Environments Policy
- Different components and models MUST run in dedicated, isolated environments to prevent dependency pollution:
  - `envs/libero_reference.yml`: Strict official LIBERO baseline (Python 3.8.13, robosuite 1.4.0, bddl 1.0.1, robomimic 0.2.0, numpy 1.22.4).
  - `envs/smolvla.yml`: SmolVLA evaluation stack.
  - `envs/minivla.yml`: MiniVLA / OpenVLA evaluation stack.
  - `envs/training.yml`: Fine-tuning / training utilities.
  - `envs/research.yml`: Exploratory analysis and plotting.

---

## Fail-Fast Rules for Coding Agents
When encountering any of the following, **STOP and report immediately** instead of guessing or patching:
- Missing checkpoint or corrupt weights.
- Unknown action dimension, ordering, or normalization parameters.
- Unknown state representation or camera ordering.
- Missing CAD meshes or corrupt collision geometry.
- Undocumented controller impedance or frequency differences.

---

## Repository Structure
```text
Thesis_26/
├── AGENTS.md
├── docs/
│   ├── SRS.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── DECISIONS.md
│   └── RESOURCE_POLICY.md
├── envs/
│   ├── libero_reference.yml
│   ├── smolvla.yml
│   ├── minivla.yml
│   ├── training.yml
│   └── research.yml
├── .agents/
│   └── skills/
│       ├── libero_reproduction.md
│       ├── simulator_validation.md
│       ├── vla_model_audit.md
│       ├── policy_evaluation.md
│       └── experiment_reporting.md
├── src/
│   ├── simulator/
│   ├── models/
│   ├── evaluation/
│   ├── controllers/
│   └── utils/
├── tests/
│   ├── environment/
│   ├── action_interface/
│   ├── model_adapters/
│   └── evaluation/
├── scripts/
│   ├── validate_project.py
│   ├── system_info.py
│   └── validate_environment.py
├── configs/
├── experiments/
└── resources/
    └── manifests/
        ├── environment_manifest.yaml
        ├── software_manifest.yaml
        └── pip_freeze.txt
```

---

## Standard Verification Commands
- Check project structure and governance:
  ```bash
  pytest -q tests/test_project_structure.py
  python scripts/validate_project.py
  ```
- Check environment integrity & provenance:
  ```bash
  python scripts/system_info.py
  python scripts/validate_environment.py
  ```
- Run LIBERO simulation smoke tests:
  ```bash
  pytest -v tests/environment/test_libero_env.py
  ```