# Software Requirements Specification (SRS)

## VLA Policy Evaluation Sandbox — LIBERO-Derived Simulation Evaluation

### Version 1.0 — Pre-Memory Baseline

---

## 1. Document Control

| Field                    | Specification                                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Document                 | Software Requirements Specification                                                                                      |
| System name              | VLA Policy Evaluation Sandbox                                                                                            |
| Version                  | 1.0                                                                                                                      |
| Primary purpose          | Reproduce a controlled subset of LIBERO and evaluate raw VLA manipulation capability before adding external robot memory |
| Primary simulator        | MuJoCo through the official LIBERO / robosuite stack                                                                     |
| Primary robot            | Franka Panda, matching LIBERO                                                                                            |
| Primary task source      | Official LIBERO task definitions                                                                                         |
| Primary model candidates | Official/author-released LIBERO-adapted SmolVLA and MiniVLA checkpoints                                                  |
| Target hardware          | Consumer laptop, Ryzen 7 + NVIDIA RTX 4060 8 GB                                                                          |
| Real robot               | UR3 + laboratory gripper; explicitly out of V1 quantitative benchmark                                                    |
| Memory module            | **Not included in V1**                                                                                                   |
| Reproducibility level    | Strict, source-pinned                                                                                                    |
| Resource policy          | No mock resources, no synthetic replacement assets, no undocumented community checkpoints, no fallback execution mode    |

---

# 2. Purpose

VLA Policy Evaluation Sandbox V1 is a controlled simulation environment for evaluating pretrained Vision-Language-Action (VLA) policies on manipulation tasks before introducing the thesis memory mechanism.

The system shall answer the following questions:

1. Can an existing small VLA reliably perceive and execute language-conditioned manipulation tasks in simulation?
2. Can the selected VLA correctly generate and execute actions for reaching, grasping, transporting, placing, opening, closing, and multi-step manipulation?
3. Which small VLA checkpoint provides sufficiently reliable baseline control for subsequent memory experiments?
4. What are the dominant failure modes of the raw policy before any memory mechanism is added?
5. Can the complete inference pipeline reproduce the intended LIBERO observation, action, camera, robot, object, initialization, and success semantics?

The system is explicitly **not** intended in V1 to demonstrate the thesis memory mechanism.

V1 is a **policy baseline and experimental infrastructure milestone**.

---

# 3. Design Principles

## 3.1 Source-of-truth principle

For the strict LIBERO reproduction track, the source of truth shall be the official:

* LIBERO repository,
* LIBERO task definitions,
* LIBERO BDDL files,
* LIBERO initial-state files,
* LIBERO assets,
* LIBERO demonstration datasets,
* compatible robosuite release specified by LIBERO,
* official/author-released VLA checkpoint,
* official model preprocessing and postprocessing code.

LIBERO's current repository explicitly supplies BDDL task definitions, fixed initial-state files, demonstrations, asset paths and evaluation infrastructure. Its environment wrapper defaults to `Panda`, `OSC_POSE`, `agentview` and `robot0_eye_in_hand`, with 20 Hz control and 128×128 camera observations.

No "equivalent" CAD model is acceptable for the strict reproduction track.

---

## 3.2 No fabricated resources

The following are prohibited in the primary benchmark:

* invented CAD assets;
* manually recreated LIBERO object meshes;
* hand-written replacements for official BDDL tasks;
* arbitrary Panda MJCF files;
* community-modified checkpoints presented as official;
* synthetic demonstrations presented as LIBERO data;
* mocked policy outputs;
* hard-coded actions pretending to be model inference;
* fallback policies;
* scripted grasping inserted when a VLA action fails;
* replacing an unsupported model component with an unrelated component without explicit declaration.

A resource is either:

1. the official/original resource;
2. an exact conversion explicitly required by the original resource's own pipeline; or
3. excluded.

---

## 3.3 Strict reproduction versus extension

The system shall maintain two clearly separated modes.

### Mode A — STRICT-LIBERO

Purpose: determine whether the VLA and pipeline can function against the benchmark distribution.

Everything follows the official LIBERO stack and assets.

### Mode B — LIBERO-DERIVED

Purpose: later perform controlled experiments while preserving the LIBERO manipulation structure.

Changes may include:

* altered task composition;
* custom UR3 scene;
* memory overlays;
* changed observation frequency;
* controlled occlusion;
* custom object placement.

Results from Mode B shall never be reported as official LIBERO benchmark scores.

---

# 4. Scope

## 4.1 Included in V1

V1 includes:

* LIBERO environment installation;
* exact dependency capture;
* official Panda robot;
* official Panda gripper;
* official LIBERO environment assets;
* official LIBERO objects and fixtures;
* official LIBERO camera configuration;
* official task definitions;
* official initial states;
* official demonstrations for validation where needed;
* VLA checkpoint loading;
* model-specific preprocessing;
* model-specific action decoding;
* closed-loop action execution;
* quantitative policy evaluation;
* qualitative video/trajectory inspection;
* action/control diagnostics;
* failure classification;
* comparison between candidate VLA models;
* model promotion decision for later memory work.

## 4.2 Explicitly excluded

V1 does not include:

* external memory;
* semantic event memory;
* spatial memory;
* object tracker;
* green-dot visual memory;
* long-term persistent memory;
* UR3 quantitative evaluation;
* real-robot deployment;
* VLA fine-tuning;
* domain randomization;
* sim-to-real claims;
* custom CAD objects;
* custom robot morphology;
* invented LIBERO tasks.

---

# 5. External System Context

The high-level data flow is:

```text
              Official LIBERO Environment
                        |
              +---------+---------+
              |                   |
          RGB images        robot state
              |                   |
              +---------+---------+
                        |
                 language task
                        |
                        v
               Model Preprocessor
                        |
                        v
                 Frozen VLA
                        |
                  action / chunk
                        |
                        v
              Model Postprocessor
                        |
                        v
             LIBERO / robosuite
                 Controller
                        |
                        v
                   Panda
                        |
                        v
                  next frame
```

There is no memory component in V1.

---

# 6. Benchmark and Simulation Requirements

## 6.1 LIBERO

The official LIBERO repository describes the benchmark as a lifelong / knowledge-transfer manipulation benchmark containing 130 tasks, including LIBERO-Spatial, LIBERO-Object, LIBERO-Goal and the LIBERO-100 split into LIBERO-90 and LIBERO-10.

The first three suites are controlled-distribution-shift suites, while the LIBERO-100 family addresses entangled knowledge transfer.

V1 shall not attempt to reproduce all 130 tasks initially.

---

## 6.2 V1 benchmark subset

The V1 core benchmark shall contain:

### Suite A — LIBERO-Object

All 10 official tasks.

The official task list contains:

1. pick up the alphabet soup and place it in the basket;
2. pick up the cream cheese and place it in the basket;
3. pick up the milk and place it in the basket;
4. pick up the tomato sauce and place it in the basket;
5. pick up the butter and place it in the basket;
6. pick up the orange juice and place it in the basket;
7. pick up the chocolate pudding and place it in the basket;
8. pick up the BBQ sauce and place it in the basket;
9. pick up the ketchup and place it in the basket;
10. pick up the salad dressing and place it in the basket.

These form the **basic manipulation/control suite**.

---

### Suite B — LIBERO-10

All 10 official tasks.

The official task definitions include:

* turn on the stove and put the moka pot on it;
* put the black bowl in the bottom drawer and close it;
* put the yellow-and-white mug in the microwave and close it;
* put both moka pots on the stove;
* put alphabet soup and cream cheese in the basket;
* put alphabet soup and tomato sauce in the basket;
* put cream cheese and butter in the basket;
* put the white mug on the left plate and yellow-and-white mug on the right plate;
* put the white mug on the plate and the chocolate pudding to the right;
* pick up the book and place it in the back compartment of the caddy.

These form the **multi-object, multi-step and compositional manipulation suite**.

---

### Optional Suite C — LIBERO-Spatial

The complete official LIBERO-Spatial suite may be enabled after the basic control pipeline passes acceptance.

Its task definitions must be loaded directly through:

```python
benchmark.get_benchmark_dict()["libero_spatial"]()
```

and task names must be read from the official benchmark object at runtime rather than manually retyped.

This avoids accidental divergence between the local task manifest and the benchmark source. LIBERO explicitly exposes task names, task BDDL files, demonstrations and initial states programmatically.

---

### Optional Suite D — LIBERO-Goal

Same rule as LIBERO-Spatial.

The suite shall be activated only after the core 20-task benchmark has passed.

---

# 7. Robot Specification

## 7.1 Robot

Strict reproduction shall use:

```text
Robot: Franka Panda
```

because the LIBERO environment wrapper explicitly defaults to:

```text
robots = ["Panda"]
controller = "OSC_POSE"
```

and the task collection pipeline is based on this robot configuration.

---

## 7.2 Robot asset source

For STRICT-LIBERO mode:

**Use the Panda model loaded by the pinned LIBERO-compatible robosuite version.**

Do not replace it with the Franka Panda model from MuJoCo Menagerie.

MuJoCo Menagerie does contain a high-quality Panda model, but that is a different model source. The Menagerie Panda is therefore suitable for a future independent UR3/general-MuJoCo experiment, not for strict LIBERO reproduction.

This distinction is mandatory.

---

## 7.3 Gripper

STRICT-LIBERO mode shall use the default Panda gripper associated with the LIBERO/robosuite environment.

No custom gripper geometry is permitted.

The robosuite controller system supports pose-space and position-space operational-space control as well as joint-space controllers; LIBERO's environment specifically configures `OSC_POSE`.

---

# 8. Simulator and Environment Specification

## 8.1 Simulator

Primary simulator:

```text
MuJoCo through LIBERO + robosuite
```

No independent custom MuJoCo environment is the V1 reference environment.

---

## 8.2 Official LIBERO dependency set

The LIBERO repository currently specifies:

```text
Python 3.8.13
robosuite 1.4.0
bddl 1.0.1
robomimic 0.2.0
numpy 1.22.4
opencv-python 4.6.0.66
hydra-core 1.2.0
gym 0.25.2
```

and its README gives the corresponding PyTorch 1.11.0 + CUDA 11.3 installation command.

The exact resolved environment used for V1 shall be captured with:

```text
environment.yml / requirements manifest
pip freeze
conda env export
GPU driver information
CUDA runtime information
Git commit hashes
```

No silent upgrade shall be allowed during the benchmark.

If the legacy environment conflicts with another VLA's current software stack, the systems shall run in separate isolated environments rather than modifying the benchmark dependencies.

---

## 8.3 Control frequency

The official LIBERO environment wrapper uses:

```text
control_freq = 20 Hz
```

V1 shall preserve this in STRICT-LIBERO mode.

---

## 8.4 Episode horizon

The default LIBERO wrapper uses:

```text
horizon = 1000
```

V1 shall not silently shorten the horizon.

A separate diagnostic timeout may be applied only in a reported secondary experiment.

---

## 8.5 Initialization

Each evaluation episode shall use official LIBERO initial states.

LIBERO exposes per-task initial-state files and explicitly uses fixed initial states for benchmark evaluation.

The evaluation system shall never replace official initial states with hand-generated random states in the strict benchmark.

---

# 9. Camera Specification

## 9.1 Camera configuration

The official LIBERO environment uses:

```text
agentview
robot0_eye_in_hand
```

as its camera names, with 128×128 observations in the reference environment configuration.

V1 shall reproduce:

```text
camera topology
camera position
camera orientation
camera projection
image size
camera naming
camera ordering
```

from the official environment.

---

## 9.2 Model-specific resizing

The simulator output shall not be manually resized to whatever resolution the model author happens to use.

Instead:

```text
LIBERO native observation
        |
        v
official model preprocessing
        |
        v
model-native image representation
```

This is important because different checkpoints have different model-side preprocessing.

For example, the current LeRobot `smolvla_libero` configuration maps LIBERO agent-view and eye-in-hand observations to two image inputs and records 7-D actions; the model-side preprocessing is part of the checkpoint configuration and must therefore be used as-is.

---

## 9.3 Image preprocessing contract

Every model run shall log:

```text
source resolution
resize method
padding/cropping method
channel order
dtype
value range
normalization
camera order
frame stacking/history
```

No manual preprocessing shall be inserted unless the official model preprocessing pipeline requires it.

---

# 10. Observation Specification

The benchmark observation contract is model-dependent.

The simulator shall expose the complete raw observation:

```text
agentview RGB
eye-in-hand RGB
joint position
joint velocity
EEF pose
gripper state
task language
```

and the model adapter shall select exactly the fields required by its official checkpoint.

LIBERO's own data configuration includes RGB observations from `agentview` and `eye_in_hand`, gripper states and joint states.

This means the simulation layer and model interface must remain separate.

---

# 11. Action Specification

The environment accepts a 7-dimensional manipulation action in the reference LIBERO setup. The official README demonstrates a seven-element action vector, while robosuite's controller layer maps high-level action representations to robot actuation.

The V1 benchmark adapter shall verify for every candidate:

```text
action dimension
action ordering
action normalization
action range
translation semantics
rotation semantics
gripper semantics
chunk semantics
execution horizon
```

No assumption such as:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

shall be hard-coded unless the selected model's official adapter documents that exact contract.

---

# 12. Model Candidate Registry

Only checkpoints with official or author-maintained provenance shall enter the primary candidate registry.

Community fine-tunes are excluded from the primary model comparison.

---

## 12.1 Candidate A — SmolVLA LIBERO

Checkpoint:

```text
lerobot/smolvla_libero
```

The official model card identifies:

```text
base model: lerobot/smolvla_base
dataset: lerobot/libero
license: Apache-2.0
```

and distributes its model weights, preprocessor, postprocessor and training configuration.

The current LeRobot LIBERO dataset contains 1,693 episodes / 273,465 frames / 40 tasks and two 256×256 RGB cameras with 8-D state and 7-D action in that dataset representation.

This shall be the **primary V1 candidate** because its official ecosystem is directly integrated with LeRobot and LIBERO.

---

## 12.2 Candidate B — MiniVLA LIBERO-90

Checkpoint:

```text
Stanford-ILIAD/minivla-libero90-prismatic
```

The official Stanford-ILIAD repository explicitly provides MiniVLA configs using a Qwen2.5 0.5B backbone and dedicated LIBERO-90 training/evaluation support.

This is the **secondary V1 candidate**.

---

## 12.3 Candidate C — MiniVLA VQ LIBERO-90

Checkpoint:

```text
Stanford-ILIAD/minivla-vq-libero90-prismatic
```

The official Stanford-ILIAD repository provides residual-VQ action chunking and a dedicated LIBERO VQ action tokenizer.

This shall be treated as a **separate model configuration**, not merely a minor inference option.

The VQ action path must be tested independently because VQ decoding introduces another potential source of error.

---

## 12.4 Candidate admission rule

A candidate is admissible only if:

1. checkpoint provenance is official or directly maintained by the original authors;
2. checkpoint files are downloaded without modification;
3. official preprocessing/postprocessing code is available;
4. the model has a documented LIBERO adaptation or evaluation pathway;
5. the model can execute inference on the target machine without fabricated fallback behavior;
6. the model's output contract can be mapped exactly to the simulator;
7. the complete checkpoint and code revisions are recorded.

---

# 13. Model Integrity Requirements

For every model:

```text
model repository
checkpoint revision / commit
model file hash
processor revision
library version
configuration
normalization statistics
action tokenizer
VQ checkpoint if applicable
```

shall be stored in:

```text
model_manifest.yaml
```

Example logical record:

```yaml
model_id:
repository:
revision:
checkpoint_hash:
framework:
framework_version:
preprocessor_revision:
postprocessor_revision:
action_space:
state_space:
camera_inputs:
action_chunk_size:
n_action_steps:
normalization:
license:
source_url:
```

No benchmark result is valid without a complete model manifest.

---

# 14. Model Selection Protocol

The system shall not select a model based on reputation, parameter count or subjective visual quality.

The model shall be selected empirically.

The process is:

```text
candidate model
      |
      v
interface validation
      |
      v
smoke control tasks
      |
      v
core 20-task benchmark
      |
      v
failure analysis
      |
      v
baseline competence decision
```

---

# 15. Baseline Competence Criterion

The purpose is not to reproduce a paper's reported headline score exactly.

The purpose is to select a policy that is:

* sufficiently capable to manipulate objects;
* stable enough for controlled experiments;
* sufficiently imperfect to leave room for later memory improvement.

The project shall therefore define two criteria.

### Hard minimum

A model shall not progress to memory experiments if it:

* cannot reliably complete basic reach/grasp/place tasks;
* repeatedly produces invalid actions;
* has systematic action-decoding errors;
* has systematic camera mismatch;
* has systematic gripper inversion;
* fails the environment integrity tests.

### Preferred operating range

For the final memory-study backbone, a policy should demonstrate:

```text
basic manipulation: clearly > chance
core object suite: approximately 50%+ overall
multi-step suite: non-trivial success with visible headroom
```

These are **project-selection criteria**, not official LIBERO scores.

If multiple models pass, the model with:

1. stable behavior,
2. reproducible inference,
3. lower latency,
4. cleaner official preprocessing,
5. better performance on simple manipulation,

shall be preferred.

---

# 16. Environment Integrity Test Suite

## ENV-001 — Official source verification

**Purpose:** ensure the environment comes from the official source.

Procedure:

1. checkout the pinned LIBERO revision;
2. record Git commit;
3. record robosuite version;
4. verify BDDL files exist;
5. verify initial-state files exist;
6. verify assets directory exists.

Expected result:

```text
all required files present
no local replacement assets
```

---

## ENV-002 — Dependency integrity

Verify every dependency against the pinned environment manifest.

Failure examples:

```text
different robosuite version
different bddl version
different numpy version
different OpenCV version
```

shall cause the reproduction run to be marked invalid until explicitly re-baselined.

---

## ENV-003 — Panda model integrity

Create a scene containing only the official Panda and its gripper.

Verify:

* joint count;
* joint names;
* gripper joint names;
* end-effector body name;
* actuator count;
* model geometry loaded;
* collision geometry loaded.

Expected result:

```text
exactly the robot expected by the LIBERO/robosuite stack
```

---

## ENV-004 — Camera integrity

Verify:

```text
agentview
robot0_eye_in_hand
```

camera existence and pose.

Capture reference images at reset and compare them against the expected camera composition.

The acceptance test shall compare image geometry and scene layout, not pixel-perfect RGB equality alone.

---

## ENV-005 — Initial-state determinism

For each official task:

```text
same task
same initial-state ID
same environment version
same seed
```

must reproduce the same initial physical state within a predefined floating-point tolerance.

---

## ENV-006 — Official success predicate

Verify that completion is determined by LIBERO's own task success predicate.

Do not replace it with:

```text
distance < threshold
```

unless the metric is explicitly labelled as an auxiliary diagnostic.

---

# 17. Action Pipeline Test Suite

## ACT-001 — Output shape

For one observation:

```text
policy -> action
```

verify exact output shape.

No reshaping by guesswork.

---

## ACT-002 — Output range

Log raw policy output and postprocessed action.

Verify the transformation:

```text
raw model output
        ↓
official postprocessor
        ↓
environment action
```

No manual clipping unless required by the official processor.

---

## ACT-003 — Translation direction

Execute a controlled non-task diagnostic with a small positive translation command.

Verify that the EEF moves in the expected simulator-frame direction.

This test validates coordinate conventions.

---

## ACT-004 — Rotation direction

Repeat for each rotational axis.

Verify:

```text +Rx
 +Ry
 +Rz
```

against the intended controller convention.

---

## ACT-005 — Gripper semantics

Test:

```text gripper open
gripper close
```

and verify the actual Panda fingers respond correctly.

No model evaluation is allowed until this passes.

---

## ACT-006 — Chunk decoding

For chunk-based models:

```text policy output
      ↓
official action decoder
      ↓
action chunk
```

must be logged.

For MiniVLA VQ, specifically verify that the official Residual-VQ action tokenizer and corresponding LIBERO VQ checkpoint are used where required by the model configuration. The MiniVLA repository explicitly documents this action-tokenizer path.

---

# 18. Policy Smoke-Test Suite

The smoke suite is designed to determine whether the model is capable of basic control.

## POL-001 — Static observation inference

Input:

```text
one valid LIBERO observation
+
official task language
```

Expected:

```text valid action
no NaN
no Inf
correct shape
reasonable latency
```

---

## POL-002 — Reach-only behavior

Use an official simple pick-and-place task.

Measure:

```text
EEF approach error
time-to-object
minimum distance to object
```

This distinguishes visual grounding failures from grasp failures.

---

## POL-003 — Grasp closure

Measure:

```text
gripper close event
object contact
object lift
```

A robot approaching correctly but never closing the gripper shall be classified as a control/gripper failure rather than a semantic failure.

---

## POL-004 — Transport

After successful grasp:

```text
object remains attached
robot reaches target region
```

---

## POL-005 — Release

Verify:

```text
gripper opens
object separates from gripper
object enters target region
```

---

## POL-006 — Full simple manipulation

A task passes only when the official LIBERO success predicate becomes true.

This is the minimum demonstration of usable VLA control.

---

# 19. Core Quantitative Task Suite

## TASK-OBJECT-ALL

Run all 10 official LIBERO-Object tasks.

Primary purpose:

* object recognition;
* single-object manipulation;
* grasp robustness;
* transport;
* placement;
* language-object grounding.

Metrics:

```text
task success rate
grasp success rate
placement success rate
episode length
time to success
failure phase
```

---

## TASK-L10-ALL

Run all 10 official LIBERO-10 tasks.

Primary purpose:

* compositional manipulation;
* multiple objects;
* sequential actions;
* drawer/container interaction;
* appliance interaction;
* multi-stage execution.

Metrics:

```text
full task success
subtask completion count
first failure step
recovery behavior
episode length
```

---

# 20. Quantitative Metrics

## 20.1 Primary metric

### Task Success Rate

$$
SR = \frac{N_{\text{successful episodes}}}
{N_{\text{episodes}}}
$$

This is the headline metric.

---

## 20.2 Phase metrics

Each episode shall additionally record:

```text
reach_success
grasp_success
lift_success
transport_success
placement_success
release_success
final_task_success
```

These diagnostics are simulator-derived and do not become model inputs.

---

## 20.3 Control quality

Measure:

```text
EEF path length
EEF final position error
EEF orientation error
maximum action magnitude
action smoothness
number of direction reversals
control frequency
inference latency
```

---

## 20.4 Interaction quality

Measure:

```text
collision count
object drops
gripper closure count
unintended object contacts
workspace boundary hits
episode timeout
```

---

## 20.5 Temporal metrics

Record:

```text
time to first approach
time to grasp
time to lift
time to placement
time to task completion
policy inference latency
action execution horizon
```

---

# 21. Qualitative Evaluation Requirements

Every promoted model must produce videos for:

1. one easy successful pick-and-place;
2. one difficult grasp;
3. one multi-object task;
4. one drawer/container task;
5. one failure;
6. one recovery attempt;
7. one timeout.

Each video shall show, where practical:

```text
task instruction
current camera frame
EEF trajectory
predicted action direction
gripper command
task state
success/failure status
```

No visual annotation shall alter the actual model observation unless explicitly declared as an experiment variable.

---

# 22. Failure Taxonomy

Every failed episode shall be assigned one primary failure category.

```text
F1 — visual perception
F2 — language grounding
F3 — target localization
F4 — reaching
F5 — grasp acquisition
F6 — grasp retention
F7 — transport
F8 — placement
F9 — gripper semantics
F10 — action decoding
F11 — controller mismatch
F12 — camera/preprocessing mismatch
F13 — timing/chunking
F14 — task sequencing
F15 — simulator instability
F16 — policy inference failure
F17 — environment/resource mismatch
```

Multiple secondary tags may be attached.

---

# 23. Edge Cases

The V1 edge cases shall be taken from actual benchmark characteristics rather than fabricated scenarios.

## EC-001 — Different object identities

Use the 10 official LIBERO-Object tasks.

Tests whether the model generalizes manipulation across object identities while maintaining identical task structure.

---

## EC-002 — Multiple-object instruction

Use the official LIBERO-10 tasks containing two objects.

Example:

```text
put both the alphabet soup and the cream cheese box in the basket
```

This tests whether the policy can:

```text identify multiple targets
sequence actions
retain task intent
```

without memory augmentation.

---

## EC-003 — Container interaction

Use:

```text
put black bowl in bottom drawer and close it
```

and:

```text
put mug in microwave and close it
```

These test manipulation that changes an articulated/container state, rather than simple placement.

---

## EC-004 — Appliance interaction

Use:

```text
turn on the stove and put the moka pot on it
```

to test object manipulation combined with a state-changing interaction.

---

## EC-005 — Sequential placement

Use tasks requiring:

```text
object A → target 1
object B → target 2
```

to distinguish single-action competence from compositional action sequencing.

---

## EC-006 — Long horizon

Measure degradation as the number of required subtasks increases.

No artificial long-horizon task shall be invented for the core benchmark.

---

## EC-007 — Initial-state variation

Run all official initial states rather than only one convenient initial state.

---

# 24. Reference Demonstration Replay

The simulator shall support replay of official demonstration actions where available.

Purpose:

```text demonstrate that:
environment
+
robot model
+
camera
+
controller
+
asset
+
success predicate
```

are functioning correctly before testing the learned policy.

A demonstration replay is an infrastructure test, not a VLA result.

If an official demonstration cannot execute correctly, policy testing must be blocked.

---

# 25. Resource Manifest

V1 shall create:

```text
resources/
    libero/
    robosuite/
    models/
    checkpoints/
    datasets/
    manifests/
```

The manifest must contain:

```yaml
libero:
  repository:
  revision:
  sha:

robosuite:
  version:
  source:

mujoco:
  installed_version:
  source:

robot:
  name: Panda
  source:
  revision:
  asset_hashes:

gripper:
  name:
  source:
  asset_hashes:

tasks:
  suites:
  task_files:
  initial_state_files:

model:
  name:
  repository:
  revision:
  file_hash:
  processor_revision:
```

No benchmark result may be published without this manifest.

---

# 26. CAD / Mesh Integrity

For every `.obj`, `.stl`, texture, `.xml`, `.mjcf`, `.bddl` or other physical asset:

```text
source repository
source path
commit/tag
license
SHA-256
```

must be recorded.

No mesh may be:

```text
scaled
decimated
retopologized
retextured
re-exported
smoothed
renamed and restructured
```

unless that transformation is already part of the official LIBERO/robosuite distribution pipeline.

This restriction is especially important for the robot and grasp-critical collision geometry.

---

# 27. No External CAD in Strict V1

Although standard external CAD datasets such as YCB are valuable for later experiments, V1 shall **not** introduce them.

For example, a later YCB-based mustard bottle experiment would be a separate extension.

The first baseline should use only the objects already defined by LIBERO so that:

```text task
+
mesh
+
physics
+
appearance
+
language
+
initial state
```

remain on the benchmark's original distribution.

The official LIBERO repository itself contains object definitions and scene-generation code; e.g. its environment generation includes objects such as ketchup, alphabet soup, cream cheese, tomato sauce and others used in the official tasks.

---

# 28. Randomization Policy

Strict V1 shall use:

```text NO domain randomization
NO arbitrary lighting randomization
NO arbitrary camera perturbation
NO random texture replacement
NO custom object scaling
```

Reason:

The first milestone is to determine whether the raw policy works under the distribution for which it was adapted.

Robustness and sim-to-real studies will be later phases.

---

# 29. Closed-Loop Execution

The policy must operate in a closed loop:

```text observation
    ↓
VLA inference
    ↓
action / action chunk
    ↓
environment execution
    ↓
new observation
    ↓
VLA inference
    ↓
...
```

Open-loop execution of an entire episode from one prediction is prohibited.

---

# 30. Action Chunking

Action chunking shall be controlled according to the selected model.

A model's original configuration shall be recorded.

For example, the current SmolVLA LIBERO configuration documents:

```text chunk_size = 50
n_action_steps = 50
```

together with 7-D actions and relative control.

A different execution horizon may be tested later, but it becomes a separate experiment.

The benchmark must never silently change:

```text chunk_size
n_action_steps
number of flow steps
replanning frequency
```

between candidate models.

---

# 31. Reproducibility Requirements

Every experiment shall be reproducible from:

```text source commits
model checkpoint
dependency manifest
task suite
task IDs
initial-state IDs
seed
camera configuration
model configuration
action configuration
environment configuration
```

---

# 32. Random Seed Policy

For benchmark evaluation:

```text environment seed
initial-state ID
policy seed, if applicable
```

shall be logged.

Where the official benchmark supplies fixed initial states, those states take precedence over arbitrary random initialization.

---

# 33. Statistical Protocol

The primary benchmark shall evaluate all available official initial states for each selected task.

For deterministic policies: one evaluation per official initial-state instance

is sufficient for the primary benchmark.

For stochastic policies or stochastic sampling: repeated runs

shall be added and the number of repeats explicitly reported.

The same task/initial-state instances shall be reused when comparing models.

---

# 34. Paired Comparison

When comparing model A and model B:

```text same task
same initial state
same camera
same simulator state
same language instruction
same controller
```

shall be used wherever technically possible.

This prevents a model from receiving an easier distribution merely because its episode seeds differ.

---

# 35. Benchmark Reports

Each model shall produce:

### Summary table

| Metric                   | Value |
| ------------------------ | ----: |
| Overall success          |       |
| Object suite success     |       |
| LIBERO-10 success        |       |
| Grasp success            |       |
| Placement success        |       |
| Average steps            |       |
| Collision rate           |       |
| Drop rate                |       |
| Mean inference latency   |       |
| Median inference latency |       |
| Action execution rate    |       |

### Failure table

| Failure         | Count | Percentage |
| --------------- | ----: | ---------: |
| Reach           |       |            |
| Grasp           |       |            |
| Transport       |       |            |
| Placement       |       |            |
| Gripper         |       |            |
| Action decoding |       |            |
| Controller      |       |            |
| Other           |       |            |

### Per-task result

| Suite | Task ID | Task name | Init ID | Success | Failure phase | Steps |
| ----- | ------: | --------- | ------: | ------: | ------------- | ----: |

---

# 36. Visual Diagnostics

The system shall support optional visualization of:

```text
EEF current position
EEF predicted target
predicted translation vector
predicted rotation
gripper command
object poses
success predicate
```

These visualizations are diagnostic only.

They must never be fed back to the policy in V1 unless explicitly defined as part of that model's official observation interface.

---

# 37. Latency Measurements

Measure separately:

```text perception extraction
model preprocessing
VLA inference
postprocessing
controller execution
rendering
```

The main metric shall be:

```text 
policy cycle time
```

rather than model-only GPU latency.

This matters because the thesis later intends to study continuous closed-loop behavior.

---

# 38. 4060 Hardware Requirements

The system shall target:

```text CPU:
Ryzen 7-class

GPU:
NVIDIA RTX 4060
8 GB VRAM

RAM:
>= 16 GB recommended
32 GB preferred

Storage:
>= 30 GB free
```

The system shall record:

```text GPU model
VRAM
CUDA driver
PyTorch CUDA version
peak VRAM
CPU utilization
GPU utilization
inference latency
```

No model is disqualified solely because its paper uses a larger GPU, but it is disqualified from the V1 primary set if actual local inference cannot be completed within the hardware budget without an undocumented approximation or fallback.

---

# 39. Model Memory Constraint

If a candidate exceeds the available VRAM:

Allowed:

```text official supported dtype
official inference optimization
official quantization method
official CPU offload
```

Not allowed:

```text arbitrary layer replacement
mock VLM
random quantized checkpoint
partial checkpoint loading
hidden CPU fallback presented as normal CUDA inference
```

The exact execution configuration must be reported.

---

# 40. Acceptance Criteria

V1 is complete only when all of the following are true.

### Environment acceptance

* Official LIBERO source is pinned.
* Dependencies are frozen.
* Panda is the correct LIBERO/robosuite robot.
* Panda gripper is correct.
* Official assets are used.
* Official BDDL tasks are used.
* Official initial states are loaded.
* Camera topology is correct.
* 20 Hz control is preserved.
* Environment success predicate works.

### Policy acceptance

For at least one candidate:

* checkpoint loads correctly;
* official preprocessing works;
* action output is valid;
* action decoding is valid;
* gripper semantics are correct;
* closed-loop execution works;
* at least several basic tasks achieve successful manipulation;
* complete evaluation logs are generated.

### Benchmark acceptance

* all 10 LIBERO-Object tasks execute;
* all 10 LIBERO-10 tasks execute;
* per-task success is recorded;
* failure categories are recorded;
* videos are generated;
* raw trajectory/action logs are stored;
* resource manifests are stored.

---

# 41. Model Promotion Gate

A model becomes the **V1 baseline policy** if:

```text
G0: all infrastructure tests pass
AND
G1: basic manipulation is demonstrably functional
AND
G2: success rate is materially above random / broken-control behavior
AND
G3: failures are attributable and inspectable
AND
G4: inference is stable on the target machine
```

The project shall not require the model to reproduce a paper's headline score exactly.

The purpose of V1 is to establish a **credible frozen policy baseline**.

---

# 42. Diagnostic Interpretation Rules

If:

```text reach fails
```

investigate:

```text camera
preprocessing
state normalization
coordinate frames
```

If:

```text reach works but grasp fails
```

investigate:

```text gripper semantics
EEF orientation
collision geometry
action chunking
```

If:

```text grasp works but placement fails
```

investigate:

```text action normalization
controller
target interpretation
```

If:

```text simple tasks work but multi-step tasks fail
```

then the policy may be sufficiently competent for later memory experiments.

If:

```text all basic tasks fail
```

memory experiments shall be blocked.

---

# 43. Definition of “Good Enough” for Memory Research

The policy does not need to achieve near-perfect LIBERO performance.

The ideal baseline has:

```text visible/simple task → strong performance
long-horizon/complex task → meaningful failures
```

because this creates measurable headroom for the later memory system.

A model with:

```text 5% baseline → 7% with memory
```

is not an informative thesis backbone.

A model with:

```text 60% baseline → 80% with memory
```

would be much more informative.

This criterion is a research-design principle, not a LIBERO benchmark rule.

---

# 44. V1 Experiment Matrix

The first complete model comparison shall be:

| Model               | Object-10 | LIBERO-10 | Qualitative | Latency | Memory |
| ------------------- | --------: | --------: | ----------- | ------: | ------ |
| SmolVLA-LIBERO      |         ✓ |         ✓ | ✓           |       ✓ | OFF    |
| MiniVLA-LIBERO90    |         ✓ |         ✓ | ✓           |       ✓ | OFF    |
| MiniVLA-VQ-LIBERO90 |         ✓ |         ✓ | ✓           |       ✓ | OFF    |

The winning model will be selected after the raw-policy experiment.

---

# 45. V2 Preparation Requirement

V1 must leave a clean interface:

```text
Observation
    ↓
Memory-independent policy input
    ↓
Frozen VLA
    ↓
Action
```

The future memory layer shall be inserted **before model inference**, without modifying:

```text simulator
camera
robot
action controller
checkpoint
postprocessor
success function
```

except where a future memory experiment explicitly requires such changes.

---

# 46. Future Memory Interface

V2 may introduce:

```text current observation
       +
retrieved textual memory
       +
spatial memory cue
       ↓
frozen VLA
```

The V1 architecture shall therefore expose:

```python
observe()
get_task_language()
get_robot_state()
run_policy(observation, task)
step(action)
check_success()
```

so that a memory wrapper can later sit around the raw policy.

---

# 47. Future UR3 Extension

The real UR3 is not part of V1's benchmark score.

A future UR3 phase shall be separate:

```text UR3 simulation
      ↓
LIBERO-derived task set
      ↓
same language/task semantics
      ↓
same external memory architecture
      ↓
UR3 hardware
```

This prevents the first V1 result from being confounded by embodiment mismatch.

MuJoCo Menagerie currently provides official/community-curated Universal Robots models such as UR5e and UR10e, but not a direct official UR3 model in its listed arm set; therefore an eventual UR3 simulation should use a separately provenance-verified UR3/UR3e model rather than quietly substituting UR5e.

---

# 48. Explicit Non-Goals

The following statements must not be claimed from V1:

> “Our policy is state of the art.”

> “The policy works zero-shot on arbitrary robots.”

> “The simulation is sim-to-real validated.”

> “The model understands memory.”

> “Memory improves performance.”

Those claims require later experiments.

V1 only establishes:

> **A reproducible, controlled and quantitatively evaluated raw VLA manipulation baseline.**

---

# 49. Required Project Artifacts

At V1 completion:

```text
artifacts/
    environment/
        environment_manifest.yaml
        pip_freeze.txt
        conda_environment.yml

    resources/
        asset_manifest.csv
        task_manifest.csv
        model_manifest.yaml
        checkpoint_hashes.txt

    experiments/
        raw_policy/
            model_x/
            model_y/

    logs/
        inference/
        environment/
        failures/

    videos/
        success/
        failure/

    metrics/
        per_task.csv
        per_episode.csv
        summary.csv

    reports/
        qualitative/
        quantitative/
```

---

# 50. Minimum Traceability Matrix

| Requirement                   | Verification                    |
| ----------------------------- | ------------------------------- |
| Official LIBERO source        | Git revision                    |
| Official task                 | BDDL file                       |
| Official initial state        | `.pruned_init`                  |
| Official object asset         | LIBERO asset path + hash        |
| Official robot                | robosuite model + version       |
| Official camera configuration | environment config              |
| Official model checkpoint     | model repo + revision + hash    |
| Model preprocessing           | checkpoint processor/config     |
| Action decoding               | model-specific official code    |
| Environment action            | controller contract             |
| Success                       | LIBERO success predicate        |
| Performance                   | episode-level metrics           |
| Visual behavior               | recorded video                  |
| Failure attribution           | taxonomy                        |
| Reproducibility               | manifest + environment snapshot |

---

# 51. Final V1 Definition of Done

VLA Policy Evaluation Sandbox V1 is considered operational when:

```text
OFFICIAL RESOURCE
      ↓
PINNED ENVIRONMENT
      ↓
EXACT LIBERO TASK
      ↓
EXACT INITIAL STATE
      ↓
RAW RGB + ROBOT STATE + LANGUAGE
      ↓
OFFICIAL MODEL PREPROCESSOR
      ↓
FROZEN VLA
      ↓
OFFICIAL MODEL POSTPROCESSOR / ACTION DECODER
      ↓
LIBERO CONTROLLER
      ↓
PANDA
      ↓
OFFICIAL SUCCESS EVALUATION
```

can run end-to-end without:

```text
mock data
fake action
fallback controller
replacement CAD
undocumented checkpoint
undocumented preprocessing
undocumented normalization
```

and can generate both:

1. quantitative per-task performance;
2. qualitative videos and failure analysis.

Only after this condition is satisfied shall the thesis memory mechanism be introduced.

---

# 52. Recommended Execution Order

## Phase 0 — Environment verification

```text
LIBERO source
→ dependency lock
→ assets
→ Panda
→ camera
→ initial state
→ success predicate
```

## Phase 1 — Deterministic environment tests

```text
official task
→ official initial state
→ demonstration replay
→ controller validation
```

## Phase 2 — Single-model policy smoke test

Start with:

```text
SmolVLA-LIBERO
```

because the current LeRobot model and LIBERO integration provide an end-to-end maintained path.

## Phase 3 — MiniVLA comparison

```text
MiniVLA non-VQ
```

then:

```text
MiniVLA VQ
```

The non-VQ checkpoint should be tested before the VQ version so that action-token compression is not mixed with basic environment/control debugging. The official MiniVLA repository documents both LIBERO training and the separate residual-VQ action-tokenization mechanism.

## Phase 4 — Full 20-task baseline

```text
LIBERO-Object × 10
LIBERO-10 × 10
```

## Phase 5 — Model selection

Choose the most stable, sufficiently competent model.

## Phase 6 — Freeze

Once selected:

```text checkpoint
processor
controller
camera
environment
```

are frozen.

## Phase 7 — Start thesis memory development

Only now add:

```text episodic event memory
spatial object memory
tracking
visual memory cues
memory-conditioned prompt/input
```

---

# 53. Core Philosophy of V1

The fundamental rule is:

> **First prove that the robot policy works. Then prove that memory helps it.**

The first experiment must therefore isolate:

```text WHAT CAN THE RAW POLICY DO?
```

before asking:

```text WHAT DOES MEMORY ADD?
```

The final V1 output is not merely an accuracy number.

It is a reproducible answer to:

```text
Which frozen small VLA can reliably control a standard LIBERO manipulation environment on the target laptop,
under the original task/asset/action conventions,
and what exactly are its remaining failure modes?
```

That frozen baseline becomes the controlled experimental instrument for V2 memory research.
