# Implementation Plan

## VLA Policy Evaluation Sandbox V1

## 0. Mục tiêu triển khai

Mục tiêu của V1 là đạt được pipeline hoàn chỉnh:

```text
Pinned LIBERO
    ↓
Verified environment
    ↓
Official task + initial state
    ↓
Raw observation
    ↓
Official VLA preprocessing
    ↓
Frozen VLA
    ↓
Official action decoding/postprocessing
    ↓
LIBERO / robosuite controller
    ↓
Panda
    ↓
Official success predicate
    ↓
Metrics + trajectory + video
```

Sau V1 mới mở rộng sang:

```text
V2 = memory
V3 = UR3 simulation
V4 = UR3 real
```

---

# 1. Nguyên tắc triển khai

## 1.1. Không làm theo kiểu “build hết rồi test”

Mỗi phase phải có:

```text
implementation
    +
test
    +
artifact
    +
acceptance criterion
```

Nếu một phase chưa đạt acceptance thì **không tiến sang phase kế tiếp**.

---

## 1.2. Ưu tiên vertical slice

Thứ tự ưu tiên:

```text
Environment
  ↓
One official task
  ↓
One official initial state
  ↓
One model
  ↓
One successful episode
  ↓
10 tasks
  ↓
20 tasks
  ↓
multiple models
```

Không làm:

```text
all models
+
all tasks
+
all metrics
```

ngay từ đầu.

---

# 2. Phase 0 — Repository Bootstrap

## Mục tiêu

Tạo skeleton project và cơ chế kiểm soát agent/codebase trước khi viết simulator.

## Công việc

Tạo:

```text
project/
├── AGENTS.md
├── docs/
│   ├── SRS.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── DECISIONS.md
│   └── RESOURCE_POLICY.md
│
├── .agents/
│   └── skills/
│       ├── libero-reproduction/
│       ├── vla-model-audit/
│       ├── simulator-validation/
│       ├── policy-evaluation/
│       └── experiment-reporting/
│
├── src/
│   ├── simulator/
│   ├── models/
│   ├── evaluation/
│   ├── controllers/
│   └── utils/
│
├── tests/
│   ├── environment/
│   ├── action_interface/
│   ├── model_adapters/
│   └── evaluation/
│
├── scripts/
├── configs/
├── experiments/
└── resources/
```

## Implement

### `AGENTS.md`

Chứa:

* project objective;
* hard constraints;
* source-of-truth rules;
* no mock/fallback;
* fail-fast policy;
* test commands;
* project structure.

### `RESOURCE_POLICY.md`

Định nghĩa:

```text
official
author-released
community
custom
forbidden
```

và rule rằng primary benchmark chỉ chấp nhận `official`/`author-released`.

### `DECISIONS.md`

Ghi các architectural decisions đầu tiên:

* LIBERO official stack;
* Panda official LIBERO/robosuite;
* no custom CAD in strict mode;
* no memory in V1;
* no UR3 in primary benchmark;
* separate model adapters;
* fail-fast instead of fallback.

## Acceptance

Phải chạy được:

```bash
pytest -q
python scripts/validate_project.py
```

dù chưa có simulator.

**Artifact:**

```text
repository skeleton
agent instructions
resource policy
initial architecture
```

---

# 3. Phase 1 — Environment Lock & Provenance

## Mục tiêu

Tạo một môi trường Python có thể tái lập chính xác.

## Công việc

### 1. Pin source

Ghi rõ:

```text
LIBERO repo
LIBERO commit
robosuite version/commit
bddl version
robomimic version
MuJoCo version
Python version
PyTorch version
CUDA
```

Không cài theo kiểu:

```bash
pip install libero
pip install latest-robosuite
```

mà phải tạo environment reproducibly.

### 2. Capture

Tạo:

```text
resources/manifests/environment_manifest.yaml
resources/manifests/software_manifest.yaml
```

và:

```text
pip_freeze.txt
conda_environment.yml
```

### 3. Hardware inventory

Script:

```text
scripts/system_info.py
```

ghi:

```text
CPU
GPU
VRAM
driver
CUDA
RAM
OS
```

## Acceptance

Một environment mới có thể được tạo từ manifest và chạy:

```bash
python scripts/validate_environment.py
```

Expected:

```text
PASS: Python
PASS: MuJoCo
PASS: robosuite
PASS: LIBERO
PASS: required assets
```

**Chưa chạy VLA.**

---

# 4. Phase 2 — Official LIBERO Environment Smoke Test

## Mục tiêu

Chứng minh environment nguyên bản chạy đúng trước khi đụng model.

## Chỉ chọn 1 task

Không chọn cả suite.

Chọn một task đơn giản từ:

```text
LIBERO-Object
```

Ví dụ một task pick-and-place chính thức.

Task phải được lấy programmatically từ official benchmark registry.

## Implement

Tạo:

```text
src/simulator/libero_env.py
```

API:

```python
env = LiberoEnv(task_id=...)
obs = env.reset(initial_state_id=...)
obs, reward, done, info = env.step(action)
success = env.check_success()
```

Tạo:

```text
tests/environment/test_libero_env.py
```

## Kiểm tra

* Panda model;
* gripper;
* cameras;
* task scene;
* object assets;
* initial state;
* control frequency;
* reset;
* step;
* success predicate.

## Acceptance

Có thể:

1. load task;
2. reset exact initial state;
3. render đúng cameras;
4. step a manually-defined diagnostic action;
5. success predicate hoạt động.

**Chưa có learned policy.**

---

# 5. Phase 3 — Official Demonstration Replay

## Mục tiêu

Chứng minh environment + controller + action interface hoạt động.

Đây là phase rất quan trọng vì nó tách:

```text simulator bug
```

khỏi:

```text VLA bug
```

## Implement

Tạo:

```text
src/evaluation/demo_replay.py
```

Pipeline:

```text official demonstration
        ↓
action sequence
        ↓
LIBERO environment
        ↓
Panda
        ↓
success
```

## Kiểm tra

* action dimension;
* action order;
* action normalization;
* gripper semantics;
* controller;
* timestep.

## Acceptance

Ít nhất một official demonstration phải replay thành công.

Nếu replay fail:

```text BLOCK ALL VLA WORK
```

Không được chữa bằng cách thêm adapter “tạm”.

---

# 6. Phase 4 — Model Audit Framework

## Mục tiêu

Trước khi code từng model, tạo một framework để **audit interface chính thức**.

## Implement

Tạo:

```text
src/models/base.py
src/models/registry.py
src/models/model_manifest.py
```

Model interface:

```python
class VLAPolicy:
    def load()
    def preprocess(obs)
    def infer(inputs)
    def postprocess(output)
    def validate_interface()
```

## Model manifest

Ví dụ:

```yaml
model_id:
repository:
revision:
checkpoint:
checkpoint_hash:

image_inputs:
state_inputs:

image_resolution:
camera_order:

action_dim:
action_semantics:
action_range:

normalization:
action_decoder:

chunk_size:
n_action_steps:

official_libero_eval:
official_inference_code:

hardware:
```

## Acceptance

Có thể audit một checkpoint mà **chưa cần chạy full benchmark**.

Output phải cho biết rõ:

```text
what goes in
what comes out
how action is decoded
```

---

# 7. Phase 5 — SmolVLA Adapter

## Mục tiêu

Làm model đầu tiên chạy end-to-end.

Mình khuyên **SmolVLA-LIBERO trước**, vì integration path tương đối rõ.

## Implement

```text
src/models/smolvla/
    adapter.py
    preprocess.py
    postprocess.py
    config.py
```

Không tự viết preprocessing nếu official processor đã có.

## Smoke test

Input:

```text
official LIBERO observation
+
official language instruction
```

Output:

```text valid action/chunk
```

Log:

```text raw output
decoded action
action shape
gripper command
latency
VRAM
```

## Acceptance

Một inference cycle chạy được:

```text
obs
→ preprocess
→ model
→ postprocess
→ action
```

Không cần success task ngay.

---

# 8. Phase 6 — Raw VLA Closed-Loop

## Mục tiêu

Kết nối model → simulator.

Pipeline:

```text
reset
  ↓
observe
  ↓
VLA
  ↓
action chunk
  ↓
execute
  ↓
observe
  ↓
VLA
  ↓
...
```

Tạo:

```text
src/evaluation/rollout.py
```

và:

```text
scripts/run_episode.py
```

## Logging

Mỗi episode lưu:

```text
episode.json
trajectory.npz
video.mp4
model_output.jsonl
timing.json
```

## Acceptance

Một task đơn giản phải có thể chạy end-to-end mà không intervention.

---

# 9. Phase 7 — Control Diagnostics

## Mục tiêu

Nếu task fail, biết **fail ở đâu**.

## Implement

Tạo:

```text
src/evaluation/diagnostics.py
```

Tự động phân loại:

```text
reach
grasp
lift
transport
placement
release
timeout
invalid action
```

## Logging thêm

```text
EEF pose
target pose
distance-to-object
distance-to-target
gripper state
collision
object lift state
```

## Visualization

Render:

```text
current EEF
predicted EEF target
object
target
gripper state
```

## Acceptance

Một failed episode phải có đủ thông tin để trả lời:

> “VLA fail ở reach, grasp, action decoding hay controller?”

---

# 10. Phase 8 — First Acceptance Benchmark

## Mục tiêu

Không chạy 20 task.

Chạy khoảng:

```text
3 simple official tasks
```

Ví dụ:

```text
pick object A → basket
pick object B → basket
pick object C → basket
```

## Evaluation

Mỗi task:

```text 10 episodes
```

hoặc sử dụng official initial-state set nếu pipeline đã ổn.

## Metrics

Tối thiểu:

```text
success rate
grasp success
placement success
failure phase
episode length
inference latency
```

## Acceptance

Ta cần biết:

```text “model actually works”
```

trước khi scale.

---

# 11. Phase 9 — Model Comparison

Chỉ khi Phase 8 pass.

## Candidate order

```text
1. SmolVLA-LIBERO
2. MiniVLA-LIBERO90
3. MiniVLA-VQ-LIBERO90
```

Có thể thêm model khác sau.

## Quan trọng

Tất cả model dùng:

```text
same task
same initial state
same camera
same controller
same success predicate
same episode set
```

Model-specific preprocessing/action decoding được giữ nguyên.

## Acceptance

Sinh bảng:

```text
Model
Overall SR
Grasp SR
Placement SR
Latency
VRAM
Failure distribution
```

---

# 12. Phase 10 — Core LIBERO-Object Benchmark

## Mục tiêu

Mở rộng từ smoke test sang:

```text
LIBERO-Object
```

toàn bộ 10 task.

## Implement

Tạo:

```text
configs/benchmarks/libero_object.yaml
```

không hard-code task list nếu có thể lấy từ official benchmark registry.

## Evaluation

Mỗi task:

```text all official initial states
```

và lưu per-episode results.

## Output

```text
experiments/
    libero_object/
        smolvla/
        minivla/
        minivla_vq/
```

## Acceptance

Có:

```text
per-task success
overall success
failure breakdown
videos
```

---

# 13. Phase 11 — LIBERO-10 Benchmark

## Mục tiêu

Đánh giá capability compositional / multi-stage.

Chạy toàn bộ LIBERO-10.

## Trọng tâm phân tích

Không chỉ:

```text success rate
```

mà:

```text
subtask completion
first failure step
object ordering errors
drawer interaction
multi-object sequencing
```

## Acceptance

Có thể phân biệt:

```text
single-object manipulation failure
```

với:

```text long-horizon sequencing failure
```

---

# 14. Phase 12 — Optional LIBERO-Spatial / Goal

Chỉ làm sau khi Object + 10 ổn.

## Spatial

Tập trung:

* spatial relation;
* placement precision;
* directional language.

## Goal

Tập trung:

* state-dependent goals;
* object/container relationships.

## Acceptance

Không bắt buộc cho milestone “memory-ready”.

Chỉ cần nếu muốn benchmark coverage rộng hơn.

---

# 15. Phase 13 — Benchmark Reporting

## Mục tiêu

Tạo report tự động từ experiment outputs.

Tạo:

```text
src/evaluation/report.py
```

Sinh:

```text
report.md
summary.csv
failure_distribution.csv
latency.csv
```

và plots:

```text
success rate by task
failure type
grasp success
latency
```

## Visual report

Mỗi model phải có:

```text
3 successful videos
3 representative failures
```

Không cần lưu video tất cả episode nếu storage lớn.

---

# 16. Phase 14 — Baseline Promotion

Đến đây mới chọn:

```text PRIMARY VLA
```

## Tiêu chí

Không nhất thiết model cao nhất mọi metric.

Ưu tiên:

```text
reliable
stable
sufficient manipulation competence
reasonable latency
easy to reproduce
clean official integration
```

và quan trọng:

```text
still has measurable failure headroom
```

## Output

Tạo:

```text
configs/models/selected_baseline.yaml
```

Ví dụ:

```yaml
model:
checkpoint:
revision:
processor:
action_decoder:
```

Từ thời điểm này:

> **Model baseline được freeze.**

---

# 17. Phase 15 — V1 Freeze

Đóng toàn bộ baseline:

```text
environment
robot
camera
task
checkpoint
processor
controller
evaluation
metrics
```

Tạo:

```text
experiments/baseline_v1/
```

gắn:

```text
git commit
model hash
environment hash
```

## V1 release checklist

```text
[ ] resource manifest
[ ] environment manifest
[ ] model manifest
[ ] reproducible install
[ ] demo replay
[ ] basic policy rollout
[ ] Object suite
[ ] LIBERO-10
[ ] quantitative report
[ ] qualitative report
[ ] failure taxonomy
[ ] selected frozen baseline
```

Chỉ khi tất cả pass mới tag:

```text
v1.0-baseline
```

---

# 18. Sau V1 mới bắt đầu Memory

V2 architecture:

```text
Current Observation
        +
Task
        +
Memory
        ↓
Frozen VLA
        ↓
Action
```

Memory không được phép thay:

```text
checkpoint
robot
controller
camera
task
success function
```

trong baseline comparison.

---

# 19. V2 Implementation Order

Không implement toàn bộ memory cùng lúc.

## V2.1 — Text memory only

```text
event
→ structured record
→ retrieval
→ prompt
```

So sánh:

```text baseline
vs
text-memory
```

---

## V2.2 — Spatial memory only

```text
tracker
→ object world state
→ projection
→ visual marker
```

So sánh:

```text baseline
vs
spatial-memory
```

---

## V2.3 — Combined memory

```text
text memory
+
spatial memory
```

---

## V2.4 — Memory ablation

```text OFF
Text only
Spatial only
Text + Spatial
Oracle
```

---

# 20. V3 — UR3 Simulation

Sau khi memory mechanism hoạt động trên Panda.

## Mục tiêu

Kiểm tra embodiment transfer.

Pipeline:

```text
Frozen VLA
+
same memory
        ↓
UR3 MuJoCo
        ↓
LIBERO-derived task family
```

Không gọi là official LIBERO score.

---

# 21. V4 — Real UR3

Chỉ sau khi UR3 simulation ổn.

Pipeline:

```text
UR3 sim
   ↓
camera calibration
   ↓
real RGB
   ↓
real robot state
   ↓
VLA
   ↓
controller
   ↓
UR3
```

Thực hiện trước:

```text
reach
grasp
move
place
```

sau đó:

```text
memory-critical tasks
```

---

# 22. Dependency Graph

Toàn bộ roadmap:

```text
P0
Repository
  │
  ▼
P1
Environment Lock
  │
  ▼
P2
LIBERO Environment
  │
  ▼
P3
Demo Replay
  │
  ▼
P4
Model Audit
  │
  ▼
P5
SmolVLA Adapter
  │
  ▼
P6
Closed-loop Rollout
  │
  ▼
P7
Control Diagnostics
  │
  ▼
P8
3-task Smoke Benchmark
  │
  ▼
P9
Model Comparison
  │
  ▼
P10
LIBERO-Object
  │
  ▼
P11
LIBERO-10
  │
  ▼
P12
Optional Spatial / Goal
  │
  ▼
P13
Reporting
  │
  ▼
P14
Baseline Selection
  │
  ▼
P15
V1 Freeze
  │
  ▼
====================
      V2 MEMORY
====================
  │
  ├── Text Memory
  ├── Spatial Memory
  ├── Combined
  └── Ablation
  │
  ▼
====================
      V3 UR3 SIM
====================
  │
  ▼
====================
      V4 UR3 REAL
====================
```

---

# 23. Recommended Coding Strategy

## Do not write the whole project from SRS in one prompt.

Use the agent in focused tasks.

### Prompt 1

```text
Implement Phase 0 only.
Do not implement simulator/model code.
Create repository structure, AGENTS.md, skills,
RESOURCE_POLICY.md and DECISIONS.md.
Run structural tests.
```

### Prompt 2

```text
Implement Phase 1 only.
Pin the exact LIBERO environment and dependencies.
Do not modify benchmark assets.
Produce environment_manifest.yaml.
Run validation.
```

### Prompt 3

```text
Implement Phase 2 only.
Load one official LIBERO task.
Validate robot, gripper, camera, initial state and success predicate.
No VLA.
```

### Prompt 4

```text
Implement Phase 3 only.
Replay an official demonstration.
Do not invent or approximate action decoding.
If the interface is unclear, stop and report.
```

### Prompt 5

```text
Audit SmolVLA-LIBERO.
Do not implement until the official preprocessing,
postprocessing, action space and chunking are identified.
Generate model_manifest.yaml.
```

### Prompt 6

```text
Implement the SmolVLA adapter according to the audited
official interface.
Do not modify the checkpoint or preprocessing semantics.
```

### Prompt 7

```text
Connect SmolVLA to one official LIBERO task.
Run a closed-loop episode.
Record raw output, decoded actions, timing and video.
```

Sau đó mới tăng scope.

---

# 24. Fail-Fast Rules for the Coding Agent

Agent phải **dừng thay vì đoán** khi gặp:

```text
missing checkpoint
unknown action dimension
unknown state normalization
unknown camera order
missing CAD
unknown VQ decoder
different dependency version
missing official preprocessing
ambiguous controller semantics
```

Response phải là:

```text
BLOCKED:
<what is unknown>
<which source should resolve it>
<what evidence is missing>
```

Không được:

```text
guess
approximate
mock
fallback
```

---

# 25. Milestone Definitions

## M0 — Infrastructure Ready

```text
repo + agent system + environment
```

## M1 — Simulator Verified

```text
official task + initial state + demonstration
```

## M2 — First VLA Running

```text
one VLA + one task + closed loop
```

## M3 — Raw Policy Verified

```text
basic manipulation works
```

## M4 — Benchmark Ready

```text
Object + LIBERO-10
```

## M5 — Baseline Frozen

```text
model selected + reports complete
```

## M6 — Memory Research Ready

```text
frozen policy + clean interfaces
```

## M7 — Memory Evaluation

```text
OFF / text / spatial / combined / oracle
```

## M8 — UR3 Simulation

```text
same memory mechanism, different embodiment
```

## M9 — UR3 Real

```text
qualitative transfer
```

---

# 26. Thứ tự ưu tiên nếu resource/thời gian bị hạn chế

Nếu cần cắt scope, cắt theo thứ tự:

```text
KEEP:
P0
P1
P2
P3
P4
P5
P6
P7
P8
P9
P10
P11
P14
P15

OPTIONAL:
P12 Spatial / Goal
P13 advanced reporting
```

Không cắt:

```text
environment integrity
demo replay
action validation
failure diagnostics
basic quantitative benchmark
```

Đây là các phần bảo vệ thesis khỏi việc đánh giá trên một pipeline sai.

---

# 27. Recommended Immediate Next Step

Không implement model ngay.

Task đầu tiên nên là:

```text
PHASE 0:
Repository bootstrap
+
AGENTS.md
+
skills
+
RESOURCE_POLICY.md
+
DECISIONS.md
+
SRS.md
+
test skeleton
```

Sau đó:

```text
PHASE 1:
exact LIBERO environment
```

và chỉ khi environment pass:

```text
PHASE 2:
one official task
```

Điểm dừng đầu tiên có giá trị là:

> **“Tôi có thể reset một official LIBERO task, load đúng Panda/asset/camera/initial state, replay một official demonstration và kiểm tra success predicate — chưa cần VLA.”**

Khi milestone đó pass, toàn bộ phần còn lại sẽ giảm đáng kể về độ rủi ro.
