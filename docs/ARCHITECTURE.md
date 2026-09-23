# ARCHITECTURE — Persistent External Memory for Frozen VLA Manipulation

> **Research-Oriented System Architecture Document**  
> Trọng tâm hệ thống là nghiên cứu **Persistent External Memory cho Frozen Vision-Language-Action (VLA) Policy trong Closed-Loop Robot Manipulation** dưới điều kiện quan sát một phần (Partial Observability).  
> Simulator (LIBERO/Robosuite), Robot (UR3/Panda), và VLA model (SmolVLA/MiniVLA) chỉ đóng vai trò **experimental substrate**, không phải contribution cốt lõi.

---

## 1. Research Objective & Core Loop

Mục tiêu chính của luận văn là chứng minh một **lớp memory bên ngoài (external, lightweight, training-free)** có thể giúp một **frozen VLA** duy trì và khai thác thông tin từ quá khứ khi thông tin đó không thể khôi phục đầy đủ từ quan sát hiện tại (current observation).

```text
Current Observation (t) ──┐
                          ▼
Evidence Sources ──► Perception / Tracker ──► Memory Update ──► Persistent World Memory
                                                                        │
Task Instruction ───────────────────────────────────────────────► Task Retrieval
                                                                        │
                        ┌───────────────────────────────────────────────┴──────────────┐
                        ▼                                                              ▼
               Text Memory Interface                                        Spatial Memory Interface
             (Compact factual context)                                     (3D anchor → 2D marker)
                        │                                                              │
                        └───────────────────────┬──────────────────────────────────────┘
                                                ▼
                                    Memory-Augmented Context
                                                ▼
                                      Frozen VLA Policy
                                                ▼
                                       Action Chunk (H)
                                                ▼
                                 Commit s steps via Controller
                                                ▼
                                       Robot / Simulator
                                                │
                                                ▼
                                    New Observation (t+s) ──► (Loop)
```

---

## 2. Research Questions (RQ)

* **RQ1 (Utility)**: Liệu persistent external memory có cải thiện đáng kể hiệu năng manipulation của một frozen VLA trong các tác vụ mà thông tin quan trọng bị che khuất hoặc biến mất khỏi observation hiện tại?
* **RQ2 (Modality Ablation)**: Vai trò tương đối và mức độ đóng góp độc lập của Spatial Memory (hình học 3D) so với Textual Episodic Memory (sự kiện, trạng thái) là gì?
* **RQ3 (Robustness & Dynamics)**: Hệ thống thích ứng như thế nào trước nhiễu định vị (spatial noise), độ trễ/lạc hậu (staleness), độ bất định (uncertainty) và độ dài horizon thực thi chunk?

---

## 3. Core Definition of Memory

> **Memory is information originating from past observations, actions, or world states that remains usable when that information is not recoverable from the current observation alone.**

* **Perception / Tracking $\neq$ Memory**: Nếu vật thể đang visible và tracker xác định được vị trí tại $t$, đó là *Perception*.
* **Tracking là một Memory Writer**: Memory bắt đầu hoạt động khi vật thể rời khỏi trường nhìn (out-of-view) hoặc bị che khuất (occluded). Khi đó, thông tin quá khứ vẫn được lưu giữ và truy xuất để robot hành động.

---

## 4. Core Design Principles

1. **Frozen Policy First**: Giữ nguyên toàn bộ trọng số của VLA ($W_{\text{VLA}} = \text{const}$). Memory chỉ thay đổi input/context (prompt text hoặc visual marker). Mọi so sánh OFF vs. ON đều đảm bảo tính quy kết nhân quả (causal attribution).
2. **External & Lightweight**: Không yêu cầu vector database nặng nề hay learned memory network. Memory module triển khai bằng cấu trúc dữ liệu Python, JSON/SQLite và retrieval xác định (deterministic).
3. **Structured First, Natural Language Second**: Nguồn chân lý duy nhất (single source of truth) là dữ liệu có cấu trúc (`WorldMemory`, `ObjectMemory`, `Event`). Ngôn ngữ tự nhiên (text summary) chỉ là một *rendered interface/view* phục vụ prompt VLA.
4. **Spatial Memory as 3D World Anchors**: Tọa độ vật thể được lưu theo hệ quy chiếu thế giới 3D ($[x, y, z]$). Khi robot hoặc camera di chuyển, hệ thống chiếu (project) tọa độ 3D lên mặt phẳng camera hiện tại để tạo visual marker 2D ($[u, v]$). Marker 2D chỉ là giao diện hiển thị, không phải memory gốc.
5. **State Retention Beyond Location**: Memory phải lưu trữ cả trạng thái ngữ nghĩa và quan hệ chứa đựng (container hierarchy, open/closed, grasped, inside) thay vì chỉ lưu tọa độ điểm ảnh cũ.
6. **First-Class Confidence & Provenance**: Mọi record đều gắn nhãn nguồn gốc (`tracker`, `action_inference`, `env_obs`), mốc thời gian (`first_seen`, `last_confirmed`), độ tin cậy ($c \in [0, 1]$), và trạng thái hiệu lực (`confirmed`, `uncertain`, `stale`, `invalidated`, `unknown`).

---

## 5. System Boundaries

| Thuộc phạm vi nghiên cứu (Inside Thesis Contribution) | Hạ tầng thực nghiệm (Outside / Infrastructure) |
|---|---|
| • World Memory Store & State Representation<br>• Memory Update Policy & Lifecycle<br>• Task-Conditioned Retrieval Engine<br>• Text Memory Interface (render prompt)<br>• Spatial Memory Interface (3D $\to$ 2D projection marker)<br>• Causal & Relational Graph Modeling<br>• Memory Corruption, Noise & Staleness Models<br>• Evaluation Protocols & Ablation Benchmarks | • Pre-trained VLA Backbone (SmolVLA, MiniVLA)<br>• Physics Simulator (MuJoCo, Robosuite, LIBERO)<br>• Robot Hardware (Franka Panda, UR3)<br>• Low-level Cartesian / IK Controllers<br>• SOTA Visual Detectors / Segmenters (SAM, YOLO, XMem)<br>• Rendering engine (EGL, OpenGL)<br>• Dataset loading pipelines |

---

## 6. System Architecture & Components

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                     WORLD / SIMULATOR                                  │
└───────────────┬────────────────────────────────────────────────────────┬───────────────┘
                │ Raw RGB-D / Proprioception                             │ State Changes
                ▼                                                        ▼
┌─────────────────────────────────┐                    ┌─────────────────────────────────┐
│   Perception / Tracking Layer   │                    │     Action / Execution Feedback │
│  • Stage A: Oracle Tracker      │                    │  • Gripper contact status       │
│  • Stage B: RGB Detector/Tracker│                    │  • Controller completion flags  │
└───────────────┬─────────────────┘                    └────────────────┬────────────────┘
                │ TrackedObject                                         │ Action evidence
                └───────────────────────────────┬───────────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │       Memory Update Policy      │
                               │  CREATE | UPDATE | CONFIRM      │
                               │  INVALIDATE | DELETE            │
                               └────────────────┬────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │     Persistent World Memory     │
                               │  • Object Memory (3D Pose, State)│
                               │  • Spatial Memory (3D Anchors)  │
                               │  • Event & Causal History       │
                               │  • Semantic Relations           │
                               └────────────────┬────────────────┘
                                                │
                          Query Instruction     ▼
               ────────────────────────►┌─────────────────────────────────┐
                                        │ Task-Conditioned Retrieval      │
                                        │  Filter stale, rank relevance   │
                                        └───────┬─────────────────┬───────┘
                                                │                 │
                                   Compact State│                 │3D Coordinates
                                                ▼                 ▼
                               ┌──────────────────┐   ┌───────────────────────────┐
                               │   Text Interface │   │     Spatial Interface     │
                               │  Prompt injection│   │  3D → 2D Projection marker│
                               └────────┬─────────┘   └───────────┬───────────────┘
                                        │                         │
                                        └───────────┬─────────────┘
                                                    ▼
                                    ┌─────────────────────────────┐
                                    │      Frozen VLA Policy      │
                                    │    Generates Chunk H = 50   │
                                    └───────────────┬─────────────┘
                                                    ▼
                                    ┌─────────────────────────────┐
                                    │  Cartesian / IK Controller  │
                                    │    Executes s steps         │
                                    └─────────────────────────────┘
```

### 6.1 Observation Schema

Dữ liệu quan sát chuẩn hóa truyền vào hệ thống:

```python
class Observation:
    rgb: dict[str, np.ndarray]       # e.g. {"agentview": (128,128,3), "eye_in_hand": (128,128,3)}
    depth: dict[str, np.ndarray]     # Optional depth maps
    instruction: str                 # Natural language task instruction
    proprioception: np.ndarray       # EEF pose (x, y, z, qx, qy, qz, qw) + gripper state
    camera_pose: dict[str, np.ndarray] # Extrinsics [R|t] & Intrinsics K per camera
    timestamp: float                 # Simulation / wall time
```

### 6.2 Perception & Tracking Layer

Hệ thống hỗ trợ 2 giai đoạn thực nghiệm để cô lập lỗi nhận thức khỏi lỗi memory:
* **Stage A (Oracle Tracking)**: Lấy ground-truth bounding box và 3D pose trực tiếp từ simulator để kiểm chứng thuật toán memory và retrieval.
* **Stage B (Realistic Tracking)**: Sử dụng pipeline vision mở (Detector $\to$ Segmenter $\to$ Visual Tracker) để kiểm chứng tính thực tế.

```python
class TrackedObject:
    object_id: str
    semantic_label: str
    image_bbox: tuple[int, int, int, int] # (xmin, ymin, xmax, ymax)
    image_center: tuple[int, int]         # (u, v)
    world_pose: np.ndarray                # 4x4 transformation matrix or [x, y, z, qx, qy, qz, qw]
    confidence: float                     # [0.0, 1.0]
    visible: bool                         # True if currently detected in frame
    timestamp: float
```

### 6.3 Persistent World Memory & Schemas

Trung tâm lưu trữ hợp nhất toàn bộ trạng thái không gian và ngữ nghĩa của môi trường:

```python
class ObjectMemory:
    id: str                               # Unique identifier, e.g. "mustard_bottle_01"
    semantic_label: str                   # "mustard bottle"
    world_position: list[float]           # [x, y, z] in world frame
    orientation: list[float]              # [qx, qy, qz, qw]
    visible: bool                         # True if currently visible
    confidence: float                     # Derived certainty score
    first_seen: float                     # First observation timestamp
    last_seen: float                      # Last confirmed observation timestamp
    state: str                            # "on_table", "grasped", "inside", "hidden"
    container_id: str | None              # ID of parent container (e.g. "drawer_left")
    relations: list[str]                  # ["inside(drawer_left)", "near(ketchup)"]
    last_event: str                       # "placed_into(drawer_left)"
    source: list[str]                     # ["tracker", "action_inference"]

class Event:
    timestamp: float
    event_type: str                       # "pick", "place", "open", "close", "disappear"
    actor: str                            # "robot"
    object: str                           # "mustard_bottle_01"
    source_state: str                     # "grasped"
    target: str | None                    # "drawer_left"
    target_state: str                     # "inside"
    confidence: float
    evidence: list[str]
```

### 6.4 Spatial Memory & Projection Pipeline

Chuyển đổi 3D anchor thành visual marker 2D phục vụ input thị giác của VLA:

$$\mathbf{p}_{\text{cam}} = \mathbf{R}_{\text{cam}} (\mathbf{p}_{\text{world}} - \mathbf{t}_{\text{cam}})$$

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim \mathbf{K} \begin{bmatrix} X_{\text{cam}} / Z_{\text{cam}} \\ Y_{\text{cam}} / Z_{\text{cam}} \\ 1 \end{bmatrix}$$

* Nếu $Z_{\text{cam}} > 0$ và $(u, v)$ nằm trong khung hình, vẽ visual marker (ví dụ: vòng tròn màu nổi bật có bán kính cố định $r=3\text{ px}$).
* Marker chỉ là **visual encoding của spatial memory**, màu sắc hay ký hiệu có thể cấu hình được và giữ cố định suốt quá trình đánh giá.

### 6.5 Causal State Transitions & Relations

Hệ thống lưu giữ chuỗi nhân quả của các tương tác thay vì bản ghi log độc lập:

$$\text{Grasp}(O) \xrightarrow{} O.\text{state}=\text{grasped} \xrightarrow{\text{PlaceInto}(O, C)} O.\text{container}=C \xrightarrow{\text{Close}(C)} C.\text{state}=\text{closed} \land O.\text{visible}=\text{False}$$

Các quan hệ nhị phân được chuẩn hóa: `inside(obj, container)`, `on(obj, surface)`, `left_of(a, b)`, `right_of(a, b)`, `near(a, b)`, `grasped_by(robot, obj)`.

### 6.6 Memory Update Policy

Memory được cập nhật qua 5 thao tác cơ bản:
1. **CREATE**: Khởi tạo object/relation mới khi phát hiện entity chưa từng thấy.
2. **UPDATE**: Làm mới tọa độ 3D, bounding box khi entity đang visible.
3. **CONFIRM**: Xác nhận trạng thái sau khi hành động hoàn tất (ví dụ: gripper đóng và nhấc lên $\implies$ `grasped`).
4. **INVALIDATE**: Đánh dấu `stale` hoặc `uncertain` khi quan sát mâu thuẫn với memory (ví dụ: mở ngăn kéo nhưng không thấy vật).
5. **DELETE**: Xóa hoặc chuyển vào archive sau một khoảng thời gian hết hạn (expiration threshold).

### 6.7 Task-Conditioned Retrieval Engine

Thuật toán retrieval 6 bước xác định (deterministic) để trích xuất ngữ cảnh tối thiểu phục vụ VLA:
1. **Entity Identification**: Phân tích task instruction để xác định target object và container liên quan.
2. **State Lookup**: Truy xuất `ObjectMemory` mới nhất có `confidence` cao nhất của các entity đích.
3. **Event Extraction**: Trích xuất các sự kiện gần nhất gắn với target object.
4. **Relation Resolution**: Lấy các quan hệ không gian hiện thời (ví dụ: vật đang nằm trong hộp nào, hộp đang đóng hay mở).
5. **Staleness Filtering**: Lọc bỏ các bản ghi đã bị invalidate hoặc có timestamp quá hạn.
6. **Interface Rendering**: Render thông tin thành text context và danh sách 3D spatial points.

### 6.8 Memory-to-VLA Interfaces

* **Text Memory Interface**: Định dạng chuỗi sự kiện/trạng thái súc tích nối vào instruction prompt:
  ```text
  Instruction: Take the mustard bottle out.
  [Memory Context]
  - mustard_bottle was placed inside drawer_left at t=12.4s.
  - drawer_left is currently closed.
  ```
* **Spatial Memory Interface**: Render marker đồ họa đè trực tiếp lên ảnh RGB quan sát của VLA tại tọa độ $(u, v)$ được chiếu từ 3D anchor.
* **Tính bổ trợ**: Text Memory giải quyết **CÁI GÌ** (What happened, container, state); Spatial Memory giải quyết **Ở ĐÂU** (Where to search/reach, 3D target coordinates).

---

## 7. Closed-Loop Control & Execution Architecture

VLA hoạt động theo cơ chế **Closed-Loop Action Chunking**:

$$O_t \xrightarrow{} \text{Update Memory} \xrightarrow{} \text{Retrieve} \xrightarrow{} \pi_{\text{VLA}}(O_t, M_t) \xrightarrow{} A_{t:t+H}$$

* **Action Chunk Horizon ($H$)**: Chiều dài chuỗi hành động mô hình dự đoán (ví dụ: $H = 50$).
* **Execution Horizon ($s$)**: Số bước thực thi thực tế trước khi quan sát lại ($s \in \{1, 5, 10, 25, 50\}$).
* **Decoupling Controller**: VLA xuất delta Cartesian pose của End-Effector: $(\Delta x, \Delta y, \Delta z, \Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}, \text{gripper}) \implies$ Cartesian Controller $\implies$ Inverse Kinematics (IK) $\implies$ Joint targets cho robot chạy ở 20 Hz. Tách biệt tần số policy (ví dụ: 2-5 Hz) khỏi tần số điều khiển (20-500 Hz).

---

## 8. Memory-Critical Task Taxonomy

Một tác vụ là **Memory-Critical** khi:

$$P(\text{Success} \mid \text{Obs}_t) \ll P(\text{Success} \mid \text{Obs}_t, \text{Memory}_{<t})$$

Hệ thống phân cấp thành 5 bậc tác vụ (Tiers):

| Tier | Phân loại | Mô tả tình huống quan sát | Vai trò thực nghiệm |
|---|---|---|---|
| **Tier 1** | **Memory-Irrelevant Control** | Target luôn visible trong suốt quá trình thao tác. Ví dụ: gắp khối cube trên bàn để vào khay mở ngay trước mặt. | Đối chứng (Control). Đảm bảo memory ON không làm giảm hiệu năng cơ bản ($OFF \approx ON$). |
| **Tier 2** | **Short-Term Occlusion** | Target bị che khuất tạm thời trong vài bước hành động (ví dụ: robot di chuyển qua vật cản). | Đánh giá khả năng bám bắt lại mục tiêu nhanh chóng. |
| **Tier 3** | **Long-Term Spatial Memory** | Robot di chuyển sang vùng khác, target hoàn toàn rời khỏi trường nhìn camera. Cần quay lại vị trí cũ để thao tác tiếp. | Đánh giá Spatial Memory: giảm thời gian tìm kiếm (search time) và quỹ đạo tìm kiếm. |
| **Tier 4** | **Event & State Memory** | Đặt vật vào hộp/ngăn kéo $\to$ đóng ngăn kéo lại $\to$ yêu cầu lấy vật ra. Quan sát hiện tại chỉ thấy ngăn kéo đóng. | Đánh giá State Memory: nhận diện đúng container cần mở thay vì tìm kiếm mù quáng. |
| **Tier 5** | **Identity & Relational Memory** | Nhiều vật thể giống hệt nhau hoặc nhiều ngăn kéo tương đồng; cần phân biệt bằng lịch sử tương tác ("lấy vật đặt vào lúc nãy"). | Đánh giá Relational & Disambiguation Memory. |

---

## 9. Evaluation Protocol, Metrics & Robustness

### 9.1 Evaluation Protocol Invariants

Mọi so sánh giữa các điều kiện phải giữ nguyên tuyệt đối:
* Cùng Model Checkpoint, seed, initial environment state.
* Cùng Action Horizon $H$ và Execution Horizon $s$.
* Cùng Controller gains, camera parameters, và robot kinematics.
* Điểm khác biệt duy nhất là **Memory Condition**.

### 9.2 Core Metrics

1. **Task Success Rate (%)**: Tỷ lệ hoàn thành tác vụ thành công trong số lần thử.
2. **Target Reacquisition Time ($t_{\text{reacq}}$)**: Thời gian từ khi target biến mất đến khi nó xuất hiện trở lại trong camera view.
3. **Search Time ($t_{\text{search}}$)**: Thời gian robot thực hiện các hành vi thăm dò tìm kiếm target.
4. **Search Trajectory**: Chiều dài đường đi của End-Effector/Camera (Path length) và mức độ chuyển động thừa (Exploratory distance).
5. **Exploratory Action Count**: Số lượng hành động không hướng tới mục tiêu chính.
6. **Replan Count**: Số lần gọi policy inference trong một episode.
7. **Time-to-Success**: Tổng thời gian hoàn thành tác vụ từ đầu đến cuối.

### 9.3 Ablation Matrix

Hệ thống đánh giá trên 5 điều kiện chính:

| Điều kiện (Condition) | Text Memory | Spatial Memory | Oracle State | Mục đích khoa học |
|---|:---:|:---:|:---:|---|
| **Stateless (OFF)** | ❌ | ❌ | ❌ | Raw VLA baseline |
| **Text-Only Memory** | ✅ | ❌ | ❌ | Đóng góp riêng của Semantic/Event Memory |
| **Spatial-Only Memory** | ❌ | ✅ | ❌ | Đóng góp riêng của 3D Spatial Tracking |
| **Full Memory (ON)** | ✅ | ✅ | ❌ | Hiệu quả kết hợp hoàn chỉnh |
| **Oracle Reference** | Opt | Opt | ✅ | Thước đo giới hạn trên (Upper bound/Diagnosis) |

### 9.4 Memory Noise & Staleness Benchmarking

Hệ thống hỗ trợ tiêm nhiễu chủ động để đo lường đường cong suy giảm hiệu năng (Memory Quality Curves):
* **Spatial Localization Noise**: Thêm offset Gaussian vào tọa độ 3D anchor: $\epsilon \in \{0, 2, 5, 10, 20\}\text{ cm}$.
* **Temporal Staleness**: Trì hoãn cập nhật memory sau $k$ giây hoặc $N$ bước hành động can thiệp: $\Delta t \in \{0, 1, 5, 15, 30, 60\}\text{ s}$.
* **Semantic Corruption**: Làm sai lệch có chủ đích nội dung text context (sai container, sai trạng thái, đảo lộn thứ tự sự kiện) để kiểm tra xem policy tin tưởng mù quáng hay có khả năng fallback khi mâu thuẫn quan sát.

---

## 10. Hardware, Simulation & Provenance Strategy

* **Simulation Track**: Là môi trường đánh giá định lượng chính (quantitative benchmark) nhờ tính tất định, khả năng thiết lập initial state chính xác, kiểm soát độ che khuất và ground-truth poses.
* **UR3 Real-Robot Track**: Dùng cho kiểm chứng định tính (qualitative validation), kiểm tra nhận thức trong môi trường thực và demo video cuối cùng; không dùng để kết luận định lượng thay cho benchmark chính nếu chưa giải quyết xong bài toán sim-to-real.
* **Provenance Manifest**: Mỗi lần chạy thực nghiệm tự động ghi lại file JSON metadata:
  ```json
  {
    "git_commit": "8f1084e...",
    "vla_checkpoint": "smolvla-base",
    "checkpoint_hash": "a1b2c3...",
    "environment": "LIBERO-Spatial",
    "controller": {"type": "OSC_POSE", "H": 50, "s": 5},
    "memory_condition": "full",
    "noise_config": {"spatial_noise_cm": 0.0, "staleness_s": 0.0},
    "seed": 42
  }
  ```

---

## 11. Codebase Architecture & Core Python Interfaces

### 11.1 Project Structure

```text
src/
├── memory/
│   ├── models.py            # Data schemas: ObjectMemory, Event, WorldMemory
│   ├── store.py             # Memory storage backend (in-memory / SQLite)
│   ├── updater.py           # Evidence integration & update rules
│   ├── retriever.py         # Task-conditioned retrieval algorithms
│   ├── confidence.py        # Uncertainty & decay estimators
│   └── corruption.py        # Noise & staleness injection utilities
├── interfaces/
│   ├── text_memory.py       # Prompt formatting & context renderer
│   └── spatial_memory.py    # 3D-to-2D projection & visual marker painter
├── perception/
│   ├── detector.py          # Vision object detector
│   ├── tracker.py           # Multi-object tracker
│   └── oracle_tracker.py    # Simulator ground-truth tracker
├── policies/
│   ├── base.py              # Abstract VLA policy interface
│   └── adapters/            # SmolVLA, MiniVLA, OpenVLA wrappers
├── controllers/
│   ├── cartesian.py         # Operational space Cartesian controller
│   └── ik.py                # Inverse kinematics solver
├── environments/
│   ├── base.py              # Abstract environment interface
│   └── libero/              # LIBERO suite wrappers
└── evaluation/
    ├── rollout.py           # Execution loop (H, s chunking)
    ├── metrics.py           # Metrics calculators (reacquisition, search time)
    ├── protocols.py         # OFF vs ON benchmark runners
    └── analysis.py          # Plotting & statistical verification
```

### 11.2 Core Class Interfaces

```python
class VLAPolicy(ABC):
    @abstractmethod
    def reset(self) -> None: ...
    
    @abstractmethod
    def predict_action_chunk(self, observation: Observation) -> np.ndarray:
        """Outputs an action chunk of shape (H, action_dim)."""
        ...

class MemorySystem(ABC):
    @abstractmethod
    def update(self, evidence: dict) -> None: ...
    
    @abstractmethod
    def retrieve(self, query: str) -> dict: ...
    
    @abstractmethod
    def get_state(self) -> WorldMemory: ...

class TextMemoryRenderer(ABC):
    @abstractmethod
    def render(self, retrieved_memory: dict, query: str) -> str: ...

class SpatialMemoryRenderer(ABC):
    @abstractmethod
    def project(self, points_3d: list[np.ndarray], camera_params: dict) -> list[tuple[int, int]]: ...
    
    @abstractmethod
    def render(self, rgb_image: np.ndarray, projected_points: list[tuple[int, int]]) -> np.ndarray: ...

class RobotEnv(ABC):
    @abstractmethod
    def reset(self) -> Observation: ...
    
    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, dict]: ...
    
    @abstractmethod
    def check_success(self) -> bool: ...
```

### 11.3 Memory Lifecycle

```text
[OBSERVE] ──► [DETECT] ──► [ASSOCIATE] ──► [UPDATE] ──► [STORE]
                                                          │
[RE-OBSERVE] ◄── [ACT] ◄── [RENDER] ◄── [RETRIEVE] ◄──────┘
```

### 11.4 Failure Handling & Non-Goals

* **Fail-Fast & Zero Hallucination**: Khi thông tin không chắc chắn hoặc không tìm thấy, hệ thống gán trạng thái `unknown` hoặc `uncertain`. Nghiêm cấm tự tạo (fabricate) tọa độ giả định.
* **Explicit Non-Goals**:
  - Không huấn luyện end-to-end một mạng neural memory.
  - Không sử dụng vector database quy mô lớn.
  - Không triển khai SLAM đầy đủ hoặc tái tạo lưới 3D phức tạp.
  - Không train lại VLA backbone.

---

## 12. Thesis Contribution Framing & 7-Phase Execution Priority

### 12.1 Canonical Contribution Statement

> **"A lightweight external persistent memory architecture for frozen VLA manipulation, combining structured episodic/state memory with persistent spatial memory and exposing the retrieved memory through text and visual interfaces without retraining the underlying VLA."**

### 12.2 Strict 7-Phase Implementation Priority

Mọi quyết định kỹ thuật phải tuân thủ nghiêm ngặt thứ tự ưu tiên sau:

```text
Phase 1: Chuẩn hóa một small VLA suy luận tin cậy trên môi trường (ví dụ: SmolVLA)
   │
   ▼
Phase 2: Freeze hoàn toàn VLA checkpoint và chuẩn hóa controller baseline
   │
   ▼
Phase 3: Định nghĩa bộ tác vụ Memory-Critical (Tiers 1 - 5)
   │
   ▼
Phase 4: Thực thi benchmark so sánh OFF vs. ON trong điều kiện đồng nhất
   │
   ▼
Phase 5: Thu thập và phân tích định lượng các chỉ số (Success, Reacquisition, Search Trajectory)
   │
   ▼
Phase 6: Tiến hành Ablation: Text-only vs. Spatial-only vs. Full Memory
   │
   ▼
Phase 7: Đánh giá độ bền vững trước nhiễu không gian và độ trễ (Noise & Staleness)
```

> [!IMPORTANT]
> Chỉ mở rộng sang các mô hình khác, task suite mở rộng, hoặc robot thực (UR3) sau khi toàn bộ 7 phase trên đã được nghiệm thu định lượng trên simulation benchmark.
