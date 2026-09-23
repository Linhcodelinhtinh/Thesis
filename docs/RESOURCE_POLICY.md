# Resource Policy — VLA Policy Evaluation Sandbox V1

## 1. Overview
To guarantee scientific reproducibility and benchmark integrity, every asset, dataset, task definition, checkpoint, and environment component utilized in this repository must comply with this Resource Policy.

---

## 2. Resource Classification Tiers

| Tier | Name | Definition | Acceptable in Primary Benchmark? |
| :--- | :--- | :--- | :--- |
| **Tier 1** | `official` | Directly released and maintained by the benchmark authors (Lifelong-Robot-Learning / Stanford / UT Austin / robosuite). | **YES (Primary source of truth)** |
| **Tier 2** | `author-released` | Checkpoints, model weights, or adaptations released directly by original model authors (e.g. Hugging Face / OpenVLA / SmolVLA teams). | **YES (Required for VLA evaluation)** |
| **Tier 3** | `community` | Unofficial community ports, third-party quantization, or re-implementations. | **NO (Forbidden in Strict Benchmark; allowed only in marked exploratory tracks)** |
| **Tier 4** | `custom` | In-house created scenes, modified task distributions, memory test fixtures. | **NO in Mode A (STRICT-LIBERO); YES in Mode B (LIBERO-DERIVED, marked as NON-COMPARABLE)** |
| **Tier 5** | `forbidden` | Invented CAD meshes, mocked policy outputs, hand-crafted replacement trajectories, silent fallbacks, or approximate initial states. | **STRICTLY PROHIBITED IN ALL TRACKS** |

---

## 3. Strict Benchmark Requirements (Mode A)
1. **Benchmark Stack**: Must use official LIBERO BDDL, assets, official initial states, Franka Panda robot model, and robosuite controllers.
2. **Robosuite Version**: The primary reference baseline requires `robosuite==1.4.0`.
3. **Model Weights**: Checkpoints must be exact author-released artifacts pinned by commit hash or SHA-256 digest.
4. **No Fallback**: If an action prediction fails or exceeds bounds, the system must fail fast or record a failure. Under no circumstances should scripted grasping or rule-based recovery substitute for policy inference.
5. **Score Reporting**: Scores from Mode B (`LIBERO-DERIVED`) or using non-official components must never be reported as official LIBERO scores.

---

## 4. Provenance Metadata Specification
Every external resource consumed by the evaluation pipeline must have an entry in `resources/manifests/` containing:
- **Canonical Source URL**: HTTPS git clone URL or Hugging Face Hub ID.
- **Pinned Version / Commit Hash**: Full 40-character Git SHA or specific release tag.
- **Cryptographic Digest**: SHA-256 checksum for weights and large data archives.
- **Date & Platform**: Date resolved, OS, Python version, CUDA driver, and rendering backend.
- **Classification Tier**: One of `official`, `author-released`, `community`, or `custom`.

---

## 5. Violations & Audit Policy
Any experiment discovered to rely on undocumented approximations, unverified weights, or substituted assets shall immediately be flagged as **INVALID / NON-COMPARABLE** in the benchmark registry.
