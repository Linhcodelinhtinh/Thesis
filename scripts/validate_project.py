#!/usr/bin/env python3
"""Project Structure and Governance Validation Script (Phase 0)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def validate_project():
    print("=" * 60)
    print("VLA Policy Evaluation Sandbox V1 — Project Validation (Phase 0)")
    print("=" * 60)

    checks = []

    # 1. Directories
    required_dirs = [
        "docs",
        "envs",
        ".agents/skills",
        "src/simulator",
        "src/models",
        "src/evaluation",
        "src/controllers",
        "src/utils",
        "tests/environment",
        "tests/action_interface",
        "tests/model_adapters",
        "tests/evaluation",
        "scripts",
        "configs",
        "experiments",
        "resources/manifests",
    ]
    missing_dirs = [d for d in required_dirs if not (REPO_ROOT / d).is_dir()]
    if missing_dirs:
        checks.append((False, f"Missing directories: {missing_dirs}"))
    else:
        checks.append((True, f"All {len(required_dirs)} required directories exist."))

    # 2. Documents
    required_docs = [
        "AGENTS.md",
        "docs/SRS.md",
        "docs/IMPLEMENTATION_PLAN.md",
        "docs/DECISIONS.md",
        "docs/RESOURCE_POLICY.md",
    ]
    missing_docs = [d for d in required_docs if not (REPO_ROOT / d).is_file()]
    if missing_docs:
        checks.append((False, f"Missing documents: {missing_docs}"))
    else:
        checks.append((True, f"All {len(required_docs)} governance documents exist."))

    # 3. Skills
    required_skills = [
        "libero_reproduction.md",
        "simulator_validation.md",
        "vla_model_audit.md",
        "policy_evaluation.md",
        "experiment_reporting.md",
    ]
    missing_skills = [
        s for s in required_skills if not (REPO_ROOT / ".agents" / "skills" / s).is_file()
    ]
    if missing_skills:
        checks.append((False, f"Missing skill files: {missing_skills}"))
    else:
        checks.append((True, f"All {len(required_skills)} agent skills exist and are populated."))

    # 4. Envs
    required_envs = [
        "libero_reference.yml",
        "smolvla.yml",
        "minivla.yml",
        "training.yml",
        "research.yml",
    ]
    missing_envs = [e for e in required_envs if not (REPO_ROOT / "envs" / e).is_file()]
    if missing_envs:
        checks.append((False, f"Missing environment definitions: {missing_envs}"))
    else:
        checks.append((True, f"All {len(required_envs)} isolated env definitions exist."))

    # 5. Imports
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        import src
        import src.controllers
        import src.evaluation
        import src.models
        import src.simulator
        import src.utils

        checks.append((True, "Python package 'src' and all subpackages import successfully."))
    except Exception as exc:
        checks.append((False, f"Failed to import 'src' packages: {exc}"))

    # Print Results
    all_passed = True
    for passed, msg in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {msg}")
        if not passed:
            all_passed = False

    print("-" * 60)
    if all_passed:
        print("RESULT: ALL PHASE 0 VALIDATION CHECKS PASSED.")
        print("Repository bootstrap is complete and verified.")
        return 0
    else:
        print("RESULT: VALIDATION CHECKS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(validate_project())
