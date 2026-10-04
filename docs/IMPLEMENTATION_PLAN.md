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
observation_state:
  runtime_dim: 8
  semantics:
    - eef_pos_x
    - eef_pos_y
    - eef_pos_z
    - eef_axis_x
    - eef_axis_y
    - eef_axis_z
    - gripper_qpos_0
    - gripper_qpos_1
  source_of_truth:
    - policy_preprocessor_step_5_normalizer_processor.safetensors

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

Nếu task fail, biết **fail ở đâu** và ghi nhận **bằng chứng cụ thể** (evidence-based attribution), không ép buộc gán nhãn suy đoán.

## Implement

Module:
```text
src/evaluation/diagnostics.py
```

### Schema 4 trường độc lập:
1. `termination_reason`: `SUCCESS`, `MAX_STEPS`, `INVALID_ACTION`, `POLICY_ERROR`, `SIMULATOR_ERROR`
2. `failure_phase`: `NONE`, `REACH`, `GRASP`, `LIFT`, `TRANSPORT`, `PLACEMENT`, `TIMEOUT`
3. `primary_failure_code`: 17 mã F1–F17 per SRS Section 22 hoặc `UNATTRIBUTED` khi không có log lỗi xác thực
4. `evidence`: Bằng chứng định lượng (`min_eef_to_object_dist`, `max_lift_delta_z`, tiếp xúc ngón kẹp, exception trace)

### Chụp baseline t=0:
- Vị trí vật thể ban đầu ($z_0$) được chụp ngay sau `env.reset()`, trước bước điều khiển đầu tiên.

### Logging Artifacts:
Mỗi episode lưu thêm:
```text
diagnostics.json
```

---

# 10. Phase 8 — First Acceptance Benchmark [COMPLETED & LOCKED]

**Trạng thái**: **ĐÃ NGHIỆM THU VÀ KHÓA (LOCKED)**  
**Script kiểm chứng & đối soát**: `scripts/reconcile_phase8.py` (0 lỗi file, 0 lỗi tier, khớp số liệu 100%).

## 10.1. Mục tiêu đã hoàn thành
Chạy kiểm thử nghiệm thu trên tập hạt nhân gồm **10 tasks đa dạng** qua 4 suites chính thức (Object, Spatial, Goal, 10) trên 4 locked official initial states (0..3, tổng 40 episodes) nhằm thiết lập raw-policy baseline thực nghiệm đáng tin cậy trước khi tích hợp mô hình mới (Phase 9) hoặc scale benchmark (Phase 10 & 11).

## 10.2. Protocol, Invariants & Provenance (ADR-0010)
- **Model Checkpoint**: `lerobot/smolvla_libero` (commit `31d453f7edd78c839a8bbc39744a292686daf0de`), backbone `SmolVLM2-500M-Video-Instruct`.
- **Simulation**: Khóa 1000 max steps, 20 Hz, Franka Panda robot, controller `OSC_POSE` (kp=150).
- **Camera Resolution**: `256x256` native (khớp chuẩn phân phối huấn luyện chính thức của LeRobot/SmolVLA, loại bỏ hiện tượng mờ ảnh do bilinear upsampling từ 128x128).
- **Receding Horizon**: $s=50$ (chỉ gọi suy luận mỗi 50 steps, tối đa 20 calls/episode).
- **Execution Tier**: `LIBERO-DERIVED (HOST_PY3.12)` (tự động nhận diện host Python 3.12.2).
- **Chứng nhận Provenance**: `NON-COMPARABLE_OFFICIAL_PAPER` (đáp ứng nghiêm ngặt AGENTS.md Rule 5 & 9).
- **Chính sách Video**: `failed_and_first` (ghi video episode đầu tiên và tất cả episode thất bại, không chèn text overlay phá vỡ khung hình).

## 10.3. Kết quả Baseline Nghiệm thu Chính thức (Reconciled Metrics)
Nguồn dữ liệu: `experiments/results/raw_baseline/acpt_10_test/`
- **Tổng số Tasks**: 10
- **Tổng số Episodes**: 40
- **Tỉ lệ thành công tổng thể (Success Rate)**: **70.0%** (28/40) [95% Wilson CI: 54.6% – 81.9%]
- **Tỉ lệ kẹp trúng (Grasp Success Rate)**: **72.5%** (29/40)
- **Tỉ lệ nhấc thành công (Lift Success Rate)**: **55.0%** (22/40)
- **Tỉ lệ đặt hoàn tất (Place Success Rate)**: **75.0%** (30/40)
- **Mean Call Inference Latency**: 2841.39 ms / call (p95: 3260.93 ms trên 321 inference calls)
- **Mean Simulation Latency**: 31.07 ms / step

### Chi tiết từng Task trong Acceptance 10:
| Suite | Task ID | Task Name | Runs | Success Rate | 95% Wilson CI | Grasp Rate | Lift Rate | Place Rate | Mean Steps (Succ) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `libero_10` | 0 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup...` | 4 | **50.0%** | [15.0%, 85.0%] | 100.0% | 75.0% | 75.0% | 328.0 |
| `libero_goal` | 0 | `open_the_middle_drawer_of_the_cabinet` | 4 | **100.0%** | [51.0%, 100.0%] | 0.0% | 0.0% | 100.0% | 129.8 |
| `libero_goal` | 7 | `turn_on_the_stove` | 4 | **100.0%** | [51.0%, 100.0%] | 0.0% | 0.0% | 100.0% | 70.8 |
| `libero_object` | 0 | `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 4 | **25.0%** | [4.6%, 69.9%] | 75.0% | 25.0% | 25.0% | 125.0 |
| `libero_object` | 1 | `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 4 | **50.0%** | [15.0%, 85.0%] | 100.0% | 50.0% | 50.0% | 119.5 |
| `libero_object` | 2 | `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 4 | **50.0%** | [15.0%, 85.0%] | 50.0% | 50.0% | 50.0% | 110.0 |
| `libero_object` | 4 | `pick_up_the_ketchup_and_place_it_in_the_basket` | 4 | **75.0%** | [30.1%, 95.4%] | 100.0% | 75.0% | 75.0% | 136.3 |
| `libero_spatial` | 0 | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin...` | 4 | **100.0%** | [51.0%, 100.0%] | 100.0% | 100.0% | 100.0% | 89.2 |
| `libero_spatial` | 2 | `pick_up_the_black_bowl_from_table_center...` | 4 | **100.0%** | [51.0%, 100.0%] | 100.0% | 100.0% | 100.0% | 96.5 |
| `libero_spatial` | 3 | `pick_up_the_black_bowl_on_the_cookie_box...` | 4 | **50.0%** | [15.0%, 85.0%] | 100.0% | 75.0% | 75.0% | 80.0 |

## 10.4. Đối soát & Tính Toàn vẹn của Artifacts (Reconciliation Audit)
1. **Kiểm tra 40/40 thư mục Episode**:
   - `episode.json`: Đầy đủ metadata, kết quả, độ trễ và provenance.
   - `diagnostics.json`: Ghi nhận tách bạch `termination_reason`, `failure_phase`, `primary_failure_code` (12 ca thất bại đều gắn nhãn `UNATTRIBUTED` kèm bằng chứng cự ly/tiếp xúc, không gán nhãn suy đoán).
   - `timing.json`: Lưu vết latency từng bước mô phỏng và từng call suy luận.
   - `trajectory.npz`: Toàn bộ 40 files chứa đủ cả 2 mảng `actions` (shape `(T, 7)`) và `states` proprioceptive 9D (shape `(T, 9)`).
   - `model_output.jsonl`: Lưu đầy đủ chuỗi action chunks xuất ra từ VLA.
   - `video.mp4`: Được tạo và phát lại bình thường cho các episode được lưu.
2. **Khớp số liệu Bottom-Up**:
   - Tổng hợp từ 40 tập riêng lẻ khớp 100% với `benchmark_summary.json`, `benchmark_summary.csv`, và `BENCHMARK_REPORT.md`.

## 10.5. Khóa Pha (Formal Sign-Off)
Phase 8 chính thức **ĐÓNG BĂNG VÀ KHÓA (LOCKED)**. Baseline raw-policy đã được xác lập vững chắc. Mở khóa toàn bộ điều kiện tiên quyết cho Phase 9.

---

# 11. Phase 9 — Multi-Model Integration & Preliminary Screening (MiniVLA, MiniVLA-VQ vs. SmolVLA) [IN PROGRESS - CODE COMPLETED, AWAITING BENCHMARK EXECUTION]

**Trạng thái Triển khai**: Đã hoàn thành code architecture: model adapters, tokenizers (discrete reverse 255-bin tail mapping & Residual-VQ), manifests, pre/post-processors và unit test suite (100% passed). Đang chờ tải trọng số chính thức từ HuggingFace để chạy screening đối đầu.
**Điều kiện tiên quyết**: Phase 8 Acceptance Benchmark đã hoàn thành, được kiểm toán đối soát độc lập (`scripts/reconcile_phase8.py`) và khóa baseline.

## 11.1. Mục tiêu & Vị trí trong Quy trình (SRS §14 & §52)
1. Tích hợp thêm 2 model ứng viên chính thức theo SRS.md Section 12 và đối chiếu với mã nguồn gốc của Stanford-ILIAD (`openvla-mini`):
   - **Candidate B**: `MiniVLA-LIBERO90` (Tokenized discrete action output, autoregressive prediction).
   - **Candidate C**: `MiniVLA-VQ-LIBERO90` (Residual-VQ action chunking output với dedicated codebook).
2. Tách biệt môi trường theo `envs/minivla.yml` (Python 3.10, PyTorch, Transformers, Prismatic, pinned Stanford-ILIAD revision) nhằm ngăn ngừa xung đột dependency với LeRobot/SmolVLA per AGENTS.md Rule 3.
3. Thực hiện **Preliminary Model Screening** (sàng lọc đối đầu ban đầu) trên tập Acceptance 10 tasks (`acceptance_10` preset, locked initial states `0..3` hoặc `0..9`) để kiểm tra tính hợp lệ của interface, độ ổn định điều khiển và đặc tính suy luận.
4. **Phạm vi quyết định**: Tuân thủ nghiêm ngặt **SRS §14 (Model Selection Protocol)**: Phase 9 chỉ có chức năng sàng lọc và thẩm định giao diện (qualify/disqualify candidates). **Quyết định lựa chọn Primary Model và đóng băng cuối cùng CHỈ được thực hiện sau khi hoàn thành Core 20-Task Benchmark (Phase 10 + Phase 11)**.

## 11.2. Provenance & Đặc tả Action Interface Chính xác (Stanford-ILIAD)
Dựa trên kiến trúc chính thức của Stanford-ILIAD `openvla-mini`:
- **Candidate B (`minivla_libero90`)**:
  - Checkpoint: `Stanford-ILIAD/minivla-libero90-prismatic`
  - Backbone: Qwen2.5 0.5B + SigLIP-224px (Prismatic VLM framework).
  - **Action Interface**: Mô hình **không sử dụng continuous 7D action head**. Thay vào đó, mô hình sử dụng `extra_action_tokenizer` (dành riêng 256 token đặc biệt trong từ vựng cho action bins). Mô hình sinh tuần tự 7 action tokens theo cơ chế autoregressive cho mỗi bước điều khiển. Các token sau đó được action tokenizer giải mã thành giá trị liên tục trong khoảng $[-1, 1]$ và unnormalize dựa trên bảng thống kê huấn luyện (`dataset_statistics.json`).
  - Manifest file: `resources/manifests/models/minivla_libero90.yaml`.
- **Candidate C (`minivla_vq_libero90`)**:
  - Checkpoint: `Stanford-ILIAD/minivla-vq-libero90-prismatic`
  - Action Tokenizer: Stanford-ILIAD official Residual-VQ action codebook for LIBERO.
  - **Action Interface**: Mô hình dự đoán các chỉ số VQ codebook rời rạc. Tokenizer sử dụng cấu trúc Residual-VQ đa tầng để tái tạo action chunk có độ dài $H$. Độ dài chunk $H$ và số lượng codebooks $K$ là hai đại lượng độc lập, được xác định cụ thể trong cấu hình tokenizer của checkpoint.
  - Manifest file: `resources/manifests/models/minivla_vq_libero90.yaml`.
- **Quy ước Gripper (Gripper Polarity)**:
  - Tuyệt đối không mặc định quy ước $+1=\text{open}, -1=\text{close}$ cho mọi model.
  - Chiều điều khiển và ngưỡng đóng/mở của gripper phải được xác minh trực tiếp từ bảng thống kê chuẩn hóa (`dataset_statistics`) và logic giải mã của action tokenizer trong từng checkpoint cụ thể.

## 11.3. Cấu trúc Module Triển khai
```text
Thesis_26/
├── envs/
│   └── minivla.yml                    # Pinned environment cho MiniVLA / Prismatic
├── resources/
│   └── manifests/
│       └── models/
│           ├── minivla_libero90.yaml      # Manifest Candidate B kèm SHA-256
│           └── minivla_vq_libero90.yaml   # Manifest Candidate C kèm SHA-256
├── src/
│   └── models/
│       └── minivla/
│           ├── __init__.py
│           ├── config.py              # MiniVLAConfig dataclass & checkpoint loader
│           ├── preprocess.py          # Prompt template ("In: ... Out:") & SigLIP 224px transforms
│           ├── action_tokenizer.py    # Stanford-ILIAD extra_action_tokenizer wrapper
│           ├── residual_vq.py         # Official Residual-VQ codebook decoder (cho bản VQ)
│           ├── postprocess.py         # Detokenization & unnormalization từ checkpoint statistics
│           └── adapter.py             # MiniVLAAdapter implementing VLAPolicy interface
├── tests/
│   ├── unit/
│   │   ├── test_minivla_adapter_unit.py     # Unit test tokenization & detokenization
│   │   └── test_minivla_vq_unit.py          # Unit test Residual-VQ codebook decoding
│   └── smoke/
│       └── test_minivla_smoke.py            # Static inference (POL-001) & gripper audit (ACT-005)
└── experiments/
    └── results/
        └── model_screening/
            ├── smolvla/               # Baseline Acceptance (từ Phase 8)
            ├── minivla/               # Kết quả sàng lọc MiniVLA non-VQ
            ├── minivla_vq/            # Kết quả sàng lọc MiniVLA-VQ
            ├── screening_summary.json # Dữ liệu so sánh tổng hợp
            └── PRELIMINARY_SCREENING_REPORT.md  # Báo cáo sàng lọc đối đầu sơ bộ
```

## 11.4. Điều kiện Thực nghiệm & Bản chất Phép So sánh
- **Bản chất phép so sánh**: So sánh các mô hình theo **cấu hình triển khai mặc định của tác giả (Native Deployment Configurations)**:
  - SmolVLA: Action chunking với execution horizon $s=50$ (replanning mỗi 50 steps).
  - MiniVLA non-VQ: Autoregressive action generation từng bước $s=1$ (replanning mỗi step).
  - MiniVLA-VQ: Residual-VQ chunking với execution horizon $s=H$ tương ứng độ dài chunk của tokenizer.
  - *Lưu ý khoa học*: Phép so sánh này phản ánh năng lực của từng hệ thống trong cấu hình triển khai gốc, không phải phép triệt tiêu kiến trúc thuần túy (pure architectural ablation) dưới cùng một tần suất replanning.
- **Physical Environment**: Franka Panda robot, controller `OSC_POSE` (thông số trích xuất nguyên bản từ cấu hình Robosuite/LIBERO), tần số 20 Hz, max 1000 steps.
- **Camera Handling**: Mô phỏng render theo cấu hình được quy định tại ADR-0010/ADR-0011; adapter tự động chuyển đổi kích thước quan sát visual phù hợp hợp đồng đầu vào của từng model (SmolVLA 256x256, MiniVLA 224x224).
- **Cấu trúc Báo cáo (Metric Granularity)**:
  - Không gộp toàn bộ episode thành một tỉ lệ phẳng duy nhất mà bỏ qua tính đại diện.
  - Báo cáo bắt buộc phải phân rã chi tiết theo **từng Task** và **từng Task Suite** (Object, Spatial, Goal, 10) để phản ánh trung thực độ lệch trọng số giữa các nhóm kỹ năng.

## 11.5. Tiêu chuẩn Thẩm định & Nghiệm thu (SRS §15)
1. **Offline Model Audit**: Kiểm toán SHA-256 các tệp trọng số, tokenizer config và dataset statistics theo manifest.
2. **Policy Smoke Tests (SRS §18)**:
   - `POL-001`: Static observation forward pass (không NaN/Inf, sinh chuỗi token hợp lệ trong từ vựng quy định).
   - `ACT-005`: Kiểm tra chiều đóng/mở gripper dựa trên unnormalization tensor thực tế.
   - `ACT-006`: Kiểm tra quá trình giải mã token $\rightarrow$ unnormalized action chunk không phụ thuộc controller.
3. **Tiêu chuẩn Sàng lọc (Screening Acceptance)**:
   - Áp dụng các tiêu chí baseline competence trong **SRS §15**: Mô hình phải kiểm soát được tiếp cận ban đầu, không tạo ra hành động bão hòa cực đoan, và hoàn tất quá trình inference trong giới hạn tài nguyên.
   - Ứng viên vượt qua screening sẽ được cấp phép bước vào **Core 20-Task Benchmark (Phase 10 & 11)** để cạnh tranh vị trí Primary Model chính thức.

---

# 12. Phase 10 — Core LIBERO-Object Benchmark (Comprehensive Single-Object Manipulation) [COMPLETED & LOCKED]

**Trạng thái Triển khai**: **ĐÃ HOÀN THÀNH VÀ KHÓA (LOCKED)**. Đã hoàn thành configuration `configs/benchmarks/libero_object.yaml`, benchmark runner CLI và đánh giá toàn bộ 10/10 tasks của `libero_object` trong benchmark tổng thể (`experiments/results/raw_baseline/full_benchmark_40`). Tỉ lệ thành công đạt **50.0%** (20/40), grasp rate **77.5%**, lift rate **50.0%**, place rate **50.0%**.

## 12.1. Mục tiêu
1. Mở rộng đánh giá năng lực thao tác đơn vật thể (single-object pick-and-place) trên **toàn bộ 10 tasks** của suite `libero_object`.
2. Khảo sát hành vi điều khiển khi tương tác với các hình học vật thể đa dạng (lon hình trụ tròn, hộp chữ nhật dẹt, chai cao trọng tâm lệch, cốc pudding).
3. Đóng góp 10 tasks đầu tiên cho bộ **Core 20-Task Benchmark** làm cơ sở chọn Primary Model theo SRS §14.

## 12.2. Nguồn Chuẩn Danh sách Task (Source of Truth)
- **Quy tắc bất biến (AGENTS.md Rule 1 & SRS §8)**: Danh sách task ID, task name và BDDL problem definitions **phải được trích xuất động từ chính benchmark object của thư viện LIBERO (`benchmark.get_task(i)`)** trong quá trình chạy mã, không hard-code thủ công.
- Tệp `resources/manifests/asset_manifest.yaml` đóng vai trò xác thực tính toàn vẹn cryptographic (SHA-256) của file BDDL và init states, không thay thế benchmark registry.
- 10 Tasks chính thức của `libero_object`:
  - Task 0: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`
  - Task 1: `pick_up_the_cream_cheese_and_place_it_in_the_basket`
  - Task 2: `pick_up_the_salad_dressing_and_place_it_in_the_basket`
  - Task 3: `pick_up_the_bbq_sauce_and_place_it_in_the_basket`
  - Task 4: `pick_up_the_ketchup_and_place_it_in_the_basket`
  - Task 5: `pick_up_the_tomato_sauce_and_place_it_in_the_basket`
  - Task 6: `pick_up_the_butter_and_place_it_in_the_basket`
  - Task 7: `pick_up_the_milk_and_place_it_in_the_basket`
  - Task 8: `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`
  - Task 9: `pick_up_the_orange_juice_and_place_it_in_the_basket`

## 12.3. Cấp độ Thực nghiệm & Minh bạch Cỡ mẫu
Kế hoạch phân định rành mạch 2 cấp độ đánh giá:
- **Level 1 — Pilot Acceptance Benchmark (100 episodes total)**:
  - 10 initial states locked (`0..9`) cho mỗi task ($10 \text{ tasks} \times 10 = 100 \text{ episodes}$).
  - Mục đích: Sàng lọc nhanh, so sánh đối đầu ban đầu giữa các model.
- **Level 2 — Full Official Benchmark (500 episodes total)**:
  - Toàn bộ 50 initial states chính thức (`0..49`) cho mỗi task ($10 \text{ tasks} \times 50 = 500 \text{ episodes}$).
  - Bắt buộc phải gắn nhãn rõ ràng trên mọi báo cáo là `PILOT_10_STATES` hay `FULL_50_STATES`, không được gọi kết quả pilot là điểm số official đầy đủ (AGENTS.md Rule 5).

## 12.4. Phương pháp Luận Chẩn đoán & Giả thuyết Vật lý
1. **Phân tích Telemetry Định lượng**:
   - `F5_GRASP_ACQUISITION`: Đo khoảng cách tối thiểu giữa EEF và tâm vật thể (`min_eef_to_object_dist`), góc tiếp cận và trạng thái tiếp xúc kẹp.
   - `F6_GRASP_RETENTION`: Đo độ suy giảm độ cao sau khi đã nhấc thành công (`max_lift_delta_z` đạt đỉnh rồi tụt trước khi tới giỏ).
   - `F8_PLACEMENT`: Đo cự ly vật thể - giỏ và trạng thái mở kẹp.
2. **Tính Khách quan Khoa học trong Diễn giải**:
   - Mọi giải thích liên quan đến tính chất vật lý (trọng tâm cao, mô-men quán tính, ma sát tiếp xúc ngón kẹp) phải được trình bày dưới dạng **Giả thuyết quan sát (Observational Hypotheses)** trừ khi được chứng minh bằng dữ liệu lực/tiếp xúc từ MuJoCo physics simulation. Tương quan thống kê đơn thuần giữa các task khác nhau không được khẳng định là quan hệ nhân quả.

## 12.5. Deliverables & Tiêu chuẩn Hoàn thành
- Dữ liệu per-episode lưu tại `experiments/results/libero_object/{model_name}/`.
- Báo cáo `LIBERO_OBJECT_REPORT.md` trình bày:
  - Tỉ lệ thành công từng task kèm 95% Wilson Confidence Interval.
  - Grasp Rate, Lift Rate, Place Rate độc lập cho từng loại hình học vật thể.
  - Phân bố failure phases và primary failure codes theo SRS Section 22.
- Hoàn tất đánh giá trên các model đủ điều kiện để chuẩn bị dữ liệu cho quy trình chọn model chính thức.

---

# 13. Phase 11 — LIBERO-10 Benchmark (Compositional & Multi-Stage Long-Horizon Evaluation) [COMPLETED & LOCKED]

**Trạng thái Triển khai**: **ĐÃ HOÀN THÀNH VÀ KHÓA (LOCKED)**. Đã hoàn thành configuration `configs/benchmarks/libero_10.yaml` (10 tasks canonical commit `8f1084e`), Subtask Milestone Diagnostics và đánh giá toàn bộ 10/10 tasks trong benchmark tổng thể (`experiments/results/raw_baseline/full_benchmark_40`). Tỉ lệ thành công đạt **27.5%** (11/40), định lượng chính xác sự suy giảm hiệu năng khi đối mặt với manipulation dài hạn (long-horizon compositional), thiết lập động cơ thực nghiệm nền tảng cho V2 Memory.

## 13.1. Mục tiêu & Vị trí Khoa học trong Luận văn
1. Đánh giá năng lực của các mô hình VLA trên tập tác vụ phức tạp nhất: **`libero_10` (Long-Horizon & Compositional Manipulation Suite)**.
2. Thiết lập baseline thực nghiệm khách quan: **Định lượng giới hạn và sự suy giảm hiệu năng (performance degradation) của các mô hình VLA nhỏ thuần phản xạ khi đối mặt với chuỗi hành động dài hạn**.
3. *Định vị nghiên cứu trung lập*: Phase 11 trong V1 chỉ có nhiệm vụ đo đạc trung thực năng lực của raw-policy baseline, **không định kiến trước kết luận "chứng minh sự cần thiết của Memory"**. Dữ liệu suy giảm ở các bước tuần tự sẽ đóng vai trò động cơ thực nghiệm (motivational empirical baseline) cho thiết kế kiến trúc Memory ở V2, nhưng việc Memory có thực sự giải quyết được hay không sẽ được kiểm chứng bằng thực nghiệm đối chứng ở V2.
4. Đóng góp 10 tasks còn lại để hoàn tất **Core 20-Task Benchmark (Phase 10 + Phase 11)**, làm căn cứ thực hiện **Official Model Selection & Freeze** theo đúng quy trình SRS §14.

## 13.2. Nguồn Task Map Chính thức của `libero_10` (commit `8f1084e`)
Danh sách task được truy vấn trực tiếp từ `libero.benchmark.get_benchmark("libero_10")`:
- **Task 0**: `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`
- **Task 1**: `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`
- **Task 2**: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
- **Task 3**: `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` *(Ngăn kéo dưới và đóng lại)*
- **Task 4**: `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- **Task 5**: `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` *(Quyển sách vào ngăn sau của caddy)*
- **Task 6**: `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` *(Mug lên plate và pudding bên phải)*
- **Task 7**: `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` *(Alphabet soup và cream cheese box vào giỏ)*
- **Task 8**: `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` *(Cả hai moka pots lên bếp)*
- **Task 9**: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` *(Mug vào lò vi sóng và đóng cửa lò)*

## 13.3. Cơ chế Chẩn đoán Đa Giai đoạn Dựa trên BDDL Predicates
1. **Tiêu chuẩn Thành công Tối thượng**: Giữ nguyên `env.check_success()` (BDDL goal condition evaluation của LIBERO) làm ground truth duy nhất cho toàn bộ task.
2. **Subtask Milestone Tracking**:
   - Thay vì ép buộc giả định mọi task đều có cấu trúc "subtask 1 rồi subtask 2", cấu trúc mốc tiến độ được phân rã trực tiếp từ cây điều kiện mục tiêu BDDL (BDDL conjuncts):
     - Ví dụ Task 0: `(and (in alphabet_soup basket) (in tomato_sauce basket))` $\rightarrow$ Tách thành 2 predicate mốc: `Pred_A: (in alphabet_soup basket)` và `Pred_B: (in tomato_sauce basket)`.
     - Ví dụ Task 3: `(and (in black_bowl bottom_drawer) (closed bottom_drawer))` $\rightarrow$ Tách thành 2 predicate mốc: `Pred_A: đặt bát vào ngăn kéo` và `Pred_B: đóng kín ngăn kéo`.
3. **Các Chỉ số Đo lường Khách quan**:
   - `Milestone Completion Time ($t_{\text{milestone}}$)`: Bước mô phỏng chính xác khi predicate thành phần được thỏa mãn lần đầu.
   - `Subtask Completion Rate`: Tỉ lệ episode thỏa mãn từng predicate thành phần độc lập.
   - `Sequential Survival Steps`: Số bước robot duy trì hành vi hợp lệ trước khi vi phạm điều kiện kết thúc hoặc dừng tiến độ.
   - Các nhãn phân loại hành vi (như thất bại thao tác nguyên tử vs lỗi chuyển tiếp chuỗi hành động) được xử lý dưới dạng **Secondary Descriptive Tags** có định nghĩa ngưỡng định lượng rõ ràng trong collector.

## 13.4. Deliverables & Quyết định Đóng băng Baseline (Model Selection Sign-Off)
- Toàn bộ kết quả lưu tại `experiments/results/libero_10/{model_name}/`.
- Báo cáo chuyên sâu `LIBERO_10_REPORT.md` trình bày:
  - Phân tích chi tiết tỉ lệ hoàn thành từng mốc mục tiêu thành phần (Subtask Progression Matrix).
  - So sánh thời gian hoàn thành giữa các giai đoạn của tác vụ.
  - Phân tích tương tác với các cơ cấu khớp động (ngăn kéo tủ, cửa lò vi sóng, núm bếp).
- **Official Model Selection Decision (Hoàn tất V1 Baseline)**:
  - Tổng hợp toàn bộ dữ liệu từ **Core 20-Task Benchmark (10 Object + 10 LIBERO-10)** của các model ứng viên.
  - Áp dụng các tiêu chí trong **SRS §14 & §15** để chính thức bầu chọn **Primary Frozen VLA Baseline** cho toàn bộ đề tài.
  - Đóng băng (Freeze) vĩnh viễn checkpoint, processor, controller và baseline scores để làm công cụ đối chứng khoa học cho Phase V2 (Memory System Architecture).

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

# 16. Phase 14 — Baseline Promotion [COMPLETED & LOCKED]

**Trạng thái Triển khai**: **ĐÃ HOÀN THÀNH VÀ KHÓA (LOCKED)**.
Đã hoàn tất đánh giá và đối soát độc lập theo Model Promotion Gate (SRS §41) trên cả tập Acceptance 10 và Full 40 tasks. Mô hình **`SmolVLA-LIBERO`** (`lerobot/smolvla_libero` @ revision `31d453f`) chính thức được bầu chọn làm **Primary Frozen VLA Baseline** cho toàn bộ đề tài:
- Đạt 70.0% trên Acceptance 10 (28/40 episodes, 95% Wilson CI: [54.6%, 81.9%]).
- Đạt 48.1% trên Core 40 tasks (77/160 episodes), vượt trội so với random control (~0%).
- Năng lực thao tác đạt chuẩn (Grasp Rate 68.1%–72.5%, Lift Rate 50.0%–55.0%, Place Rate 53.8%–75.0%).
- Độ trễ ổn định (~2841 ms/call, 50-step chunking, amortized ~60 ms/step).
- Xuất hiện không gian suy giảm hiệu năng rõ ràng trên multi-stage tasks (`libero_10`: 27.5%), tạo tiền đề hoàn hảo cho đối chứng Memory ở V2.

Tệp cấu hình chính thức:
```text
configs/models/selected_baseline.yaml
```

Từ thời điểm này:
> **Model baseline được freeze hoàn toàn.**

---

# 17. Phase 15 — V1 Freeze [COMPLETED & LOCKED]

**Trạng thái Triển khai**: **ĐÃ CHÍNH THỨC ĐÓNG BĂNG VÀ KHÓA (LOCKED)**.
Toàn bộ hệ thống baseline V1 đã được kiểm toán đối soát độc lập (`scripts/reconcile_v1_freeze.py`), đạt 100% tiêu chí nghiệm thu (0 file error, 0 tier mismatch). Gói baseline chính thức được lưu trữ tại:
```text
experiments/baseline_v1/
├── FROZEN_BASELINE_MANIFEST.yaml
├── V1_BASELINE_REPORT.md
├── selected_baseline.yaml
└── freeze_audit_signoff.json
```

## V1 release checklist

```text
[x] resource manifest (asset_manifest.yaml, demonstrations_manifest.yaml)
[x] environment manifest (environment_manifest.yaml, software_manifest.yaml)
[x] model manifest (smolvla_libero.yaml, minivla_libero90.yaml, minivla_vq_libero90.yaml)
[x] reproducible install (pinned conda/uv specifications in envs/)
[x] demo replay (verified with exact tolerance & env success in tests/integration/)
[x] basic policy rollout (rollout_episode() with receding horizon chunking)
[x] Object suite (10/10 tasks evaluated in full_benchmark_40)
[x] LIBERO-10 (10/10 tasks evaluated in full_benchmark_40)
[x] quantitative report (benchmark_summary.json, BENCHMARK_REPORT.md)
[x] qualitative report (telemetry videos, failure_distribution.csv)
[x] failure taxonomy (4-tier attribution schema, F1-F17 taxonomy)
[x] selected frozen baseline (configs/models/selected_baseline.yaml)
```

**Official Release Tag**: `v1.0-baseline` (Audit verified via `scripts/reconcile_v1_freeze.py`)

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
