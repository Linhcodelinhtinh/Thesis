"""Unit tests verifying repository bootstrap, governance, and structural integrity."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_required_directories():
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
    for rel_dir in required_dirs:
        dir_path = REPO_ROOT / rel_dir
        assert dir_path.is_dir(), f"Required directory missing: {rel_dir}"


def test_required_documentation():
    required_docs = [
        "AGENTS.md",
        "docs/SRS.md",
        "docs/IMPLEMENTATION_PLAN.md",
        "docs/DECISIONS.md",
        "docs/RESOURCE_POLICY.md",
    ]
    for rel_doc in required_docs:
        doc_path = REPO_ROOT / rel_doc
        assert doc_path.is_file(), f"Required document missing: {rel_doc}"
        assert doc_path.stat().st_size > 100, f"Document is unexpectedly empty: {rel_doc}"


def test_required_skills():
    required_skills = [
        ".agents/skills/libero_reproduction.md",
        ".agents/skills/simulator_validation.md",
        ".agents/skills/vla_model_audit.md",
        ".agents/skills/policy_evaluation.md",
        ".agents/skills/experiment_reporting.md",
    ]
    for rel_skill in required_skills:
        skill_path = REPO_ROOT / rel_skill
        assert skill_path.is_file(), f"Required skill missing: {rel_skill}"
        content = skill_path.read_text(encoding="utf-8").strip()
        assert len(content) > 50, f"Skill content too short or empty: {rel_skill}"


def test_environment_specifications():
    required_envs = [
        "envs/libero_reference.yml",
        "envs/smolvla.yml",
        "envs/minivla.yml",
        "envs/training.yml",
        "envs/research.yml",
    ]
    for rel_env in required_envs:
        env_path = REPO_ROOT / rel_env
        assert env_path.is_file(), f"Required environment file missing: {rel_env}"
        content = env_path.read_text(encoding="utf-8")
        assert "name:" in content, f"Invalid environment format in {rel_env}"


def test_decisions_records():
    decisions_file = REPO_ROOT / "docs" / "DECISIONS.md"
    content = decisions_file.read_text(encoding="utf-8")
    for adr in ["ADR-0001", "ADR-0002", "ADR-0003", "ADR-0004", "ADR-0005", "ADR-0006"]:
        assert adr in content, f"Missing {adr} in DECISIONS.md"


def test_resource_policy_tiers():
    policy_file = REPO_ROOT / "docs" / "RESOURCE_POLICY.md"
    content = policy_file.read_text(encoding="utf-8")
    for tier in ["official", "author-released", "community", "custom", "forbidden"]:
        assert tier in content, f"Missing tier {tier} in RESOURCE_POLICY.md"


def test_src_package_importability():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import src
    import src.simulator
    import src.models
    import src.evaluation
    import src.controllers
    import src.utils

    assert src is not None
    assert src.simulator is not None
