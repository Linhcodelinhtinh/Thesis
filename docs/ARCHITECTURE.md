# ARCHITECTURE — Persistent External Memory for VLA Robot Manipulation

> **Research-Oriented System Architecture Document**  
> Trọng tâm hệ thống là nghiên cứu **Persistent External Memory cho Vision-Language-Action (VLA) Policy trong Closed-Loop Robot Manipulation** dưới điều kiện quan sát một phần (Partial Observability) và nhiệm vụ thao tác nhiều chu kỳ (Multi-cycle Manipulation).  
> Simulator (LIBERO/Robosuite), Robot (Franka Panda), và VLA model (SmolVLA) đóng vai trò **experimental substrate**, phục vụ đo đạc hiệu năng của cơ chế memory.

---

## 1. Research Objective & System Dual-Loop Architecture

Mục tiêu chính của luận văn là chứng minh một **lớp memory bên ngoài (external, structured, perception-grounded memory layer)** có thể giúp VLA policy duy trì và khai thác thông tin từ quá khứ khi thông tin đó không thể khôi phục đầy đủ từ quan sát hiện tại (current observation).

Hệ thống hỗ trợ hai đường tích hợp thực nghiệm:
1. **Zero-Shot Prompt Conditioning (V2.1 Frozen VLA)**: Tiêm context memory dạng văn bản vào task prompt của mô hình VLA đã freeze ($W_{\text{VLA}} = \text{const}$).
2. **Mixed Memory-Augmented SFT Adaptation (V2.2 Action-Expert LoRA)**: Thích ứng Action-Expert của VLA thông qua Supervised Fine-Tuning kết hợp **50% Base Replay + 50% Memory Data**, giúp mô hình học cách điều kiện hóa phân phối hành động theo token memory mà không bị lãng quên thảm khốc (catastrophic forgetting) các kỹ năng thao tác cơ bản.

```text
                                 [WORLD / SIMULATOR]
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │ Observation RGB-D (256x256)                 │ Proprioception (8D)
                   ▼                                             ▼
     ┌───────────────────────────┐                 ┌───────────────────────────┐
     │ Perception Worker (IPC)   │                 │ Action & State Feedback   │
     │ Grounding DINO Zero-Shot  │                 │ Gripper aperture /contacts│
     │ Causal Object Tracker     │                 │ Execution chunk status    │
     └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                   │ Bounding Boxes & Anchors                    │ Kinematic events
                   └──────────────────────┬──────────────────────┘
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Visual Event Memory Writer    │
                         │   (FSM: Departure / Return)     │
                         │   Hysteresis & Gap Tracking     │
                         └────────────────┬────────────────┘
                                          ▼
                         ┌─────────────────────────────────┐
                         │    Persistent Episode Memory    │
                         │    • Immutable Event History    │
                         │    • Cycle Count / Status       │
                         │    • Provenance & Confidence    │
                         └────────────────┬────────────────┘
                                          │
                    Task Instruction      ▼
            ────────────────────────►┌─────────────────────────────────┐
                                     │ Task-Conditioned Retrieval      │
                                     │ Filter stale, rank relevance    │
                                     └────────────────┬────────────────┘
                                                      ▼
                                     ┌─────────────────────────────────┐
                                     │ Token-Aware Text Renderer       │
                                     │ Strict Budget <= 48 tokens      │
                                     │ Privilege Gated (No Oracle)     │
                                     └────────────────┬────────────────┘
                                                      │
                                                      ▼
                                            Augmented Prompt Context
                                                      │
                                                      ▼
                                     ┌─────────────────────────────────┐
                                     │         VLA Policy Stack        │
                                     │ • Track 1: Frozen SmolVLA       │
                                     │ • Track 2: Action-Expert LoRA   │
                                     └────────────────┬────────────────┘
                                                      │ Action Chunk H = 50
                                                      ▼
                                     ┌─────────────────────────────────┐
                                     │    OSC_POSE Controller (20 Hz)  │
                                     │    Receding Horizon (s = 50)    │
                                     └────────────────┬────────────────┘
                                                      │ Joint Torques
                                                      ▼
                                             [Franka Panda Robot]
```

---

## 2. Research Questions (RQ)

* **RQ1 (Utility & Task Headroom)**: Liệu external episodic memory có cải thiện đáng kể tỷ lệ hoàn thành tác vụ (Task Success Rate) trong các bài toán thao tác đa chu kỳ (multi-cycle) hoặc yêu cầu nhớ trạng thái trước đó (như `LIBERO-Mem` T1/T3)?
* **RQ2 (Retention & Non-Interference)**: Việc huấn luyện thích ứng với memory tokens (Mixed SFT) có giữ nguyên năng lực thao tác nguyên bản trên các tác vụ không cần memory (LIBERO-10 suite retention) hay gây ra hiện tượng suy giảm kỹ năng?
* **RQ3 (Perception Grounding vs. Oracle Upper Bound)**: Khoảng cách hiệu năng giữa memory sinh từ visual perception thực tế (`Grounding DINO` + tracker FSM) so với memory sinh từ simulator ground-truth (`OracleMemoryWriter`) là bao nhiêu?
* **RQ4 (Modality & Interface Robustness)**: Tác động của việc giới hạn token budget ($\le 48$ tokens), hiện tượng mất dấu quan sát (incomplete history / occlusion gap) và định dạng prompt lên độ ổn định điều khiển của policy.

---

## 3. Core Definition of Memory

> **Memory is information originating from past observations, actions, or world states that remains usable when that information is not recoverable from the current observation alone.**

* **Perception $\neq$ Memory**: Nếu vật thể đang visible và tracker đo được bounding box tại frame $t$, đó là *Perception*.
* **Tracking là một Memory Writer**: Memory bắt đầu hoạt động khi sự kiện diễn ra trong quá khứ (ví dụ: vật thể được nhấc lên, di chuyển ra khỏi vị trí ban đầu rồi đặt trở lại đĩa), và thông tin đó được lưu trữ có cấu trúc để làm căn cứ quyết định cho các chu kỳ tiếp theo khi quan sát tức thời tại $t$ không thể hiện số lần đã lặp lại.

---

## 4. Core Design Principles

1. **Strict Provenance & Zero Hallucination**: Mọi fact và event trong memory đều phải gắn liền với nguồn chứng cứ (`EvidenceSource.PREDICTED_RGB_TRACKER`, `OBSERVATION_TRACKER`, `ORACLE_SIMULATOR`), mốc thời gian xuất hiện, và độ tin cậy. Nghiêm cấm tự sinh fact giả hoặc suy diễn trạng thái không có căn cứ quan sát.
2. **Decoupled Perception Worker**: Module thị giác mở (`GroundingDINO`) chạy trong tiến trình worker độc lập (`ProcessVisualDetector`) giao tiếp qua IPC, bảo đảm cách ly hoàn toàn phụ thuộc thư viện (CUDA, PyTorch, Transformers) khỏi môi trường thực thi policy hoặc simulator.
3. **Structured First, Natural Language as a View**: Single Source of Truth là schema dữ liệu có cấu trúc (`SchemaV2EpisodeMemory`, `V2MemoryEvent`, `ObjectMemory`). Chuỗi văn bản đưa vào prompt chỉ là một phép chiếu (rendered view) tuân thủ ngân sách token của tokenizer gốc.
4. **Deterministic Paired Comparisons**: Mọi thực nghiệm so sánh (B0 vs F0 vs F1 vs F1-off) phải giữ nguyên tuyệt đối initial state, seed sinh ngẫu nhiên, camera resolution, controller gains và execution horizon $H=50, s=50$.
5. **Dual-Domain Replay Preservation**: Trong pha SFT, cân bằng chính xác 50% dữ liệu base manipulation và 50% dữ liệu memory-augmented để ngăn ngừa catastrophic forgetting.
6. **Immutable State Locking**: Tập split train/val/test được nhóm theo hash vector initial-state 64-bit IEEE-754 (`split_lock.py`), bảo đảm không rò rỉ trạng thái khởi tạo giữa tập huấn luyện và tập kiểm định.

---

## 5. System Boundaries & Layer Separation

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. RESEARCH LAYER (Thesis Contributions)                                               │
│    • Visual Memory Perception & Tracking (Grounding DINO + Causal Tracker)             │
│    • Visual Event Writer v3 (FSM: Departure / Return / Incomplete History)             │
│    • Episode Memory Store, Updater, & Immutable Schemas                                │
│    • Tokenizer-Budgeted Prompt Renderer (LiberoMemTextRenderer, <= 48 tokens)          │
│    • Mixed Replay SFT Training Protocol (50% Base / 50% Mem)                           │
│    • Multi-Condition Evaluation Protocol (B0, F0, F1, F1-off) & Milestone Metrics      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. MODEL ADAPTATION LAYER                                                              │
│    • SmolVLA Base Policy Adapter (lerobot/smolvla_libero @ 31d453f)                    │
│    • Action-Expert LoRA Adapter (Rank 16 / Alpha 16, Frozen VLM)                       │
│    • Receding Horizon Chunk Execution Engine (H = 50, s = 50)                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. SIMULATION & BENCHMARK SUBSTRATE                                                    │
│    • LIBERO Official Benchmark (Git commit 8f1084e)                                    │
│    • LIBERO-Mem Multi-Cycle Benchmark (Git commit 6df130c)                             │
│    • Franka Panda Robot + OSC_POSE Operational Space Controller (20 Hz)                │
│    • MuJoCo 3.x / Robosuite 1.4.0 Engine (256x256 Dual RGB Cameras)                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Detailed Component Architecture

### 6.1 Perception & Object Tracking Layer (`src/perception/`)

Lớp nhận thức chịu trách nhiệm trích xuất thông tin thực thể từ quan sát RGB thô mà không sử dụng ground-truth từ simulator:

* **[GroundingDINO Zero-Shot Detector](file:///d:/Thesis_26/src/perception/grounding_dino.py)**:
  - Sử dụng checkpoint `IDEA-Research/grounding-dino-tiny` được pin mật mã SHA-256.
  - Nhận text queries nhắm mục tiêu: `"a metal bowl."`, `"a white plate."`.
  - Trích xuất bounding boxes và scores chuẩn hóa $[0, 1]$.
* **[ProcessVisualDetector](file:///d:/Thesis_26/src/perception/process_detector.py)**:
  - Khởi chạy tiến trình con chạy môi trường `.venv_memory` độc lập.
  - Giao tiếp tuần tự qua stdin/stdout JSON IPC, truyền ảnh frame hash và bounding boxes.
  - Ngăn ngừa xung đột phiên bản CUDA/PyTorch giữa LeRobot/SmolVLA và GroundingDINO.
* **[CausalObjectTracker & VisualObjectTracker](file:///d:/Thesis_26/src/perception/object_tracker.py)**:
  - Khởi tạo anchor cho support container (`white plate`) tại frame đầu tiên khi gripper mở.
  - Theo dõi bounding box của target (`metal bowl`) trong không gian ảnh 2D.
  - Áp dụng cơ chế hysteresis và temporal continuity ($\le 3$ frames) để loại trừ proposal nhiễu.
  - Hỗ trợ cơ chế cache detection (`--detection-cache`) có kiểm tra hash SHA-256 từng frame RGB.

### 6.2 Visual Event Memory Writer v3 (`src/memory/`)

Bộ sinh memory thế hệ 3 loại bỏ hoàn toàn các giả định heuristic về proprioception (như ngưỡng $z \le 0.95$ vốn gây ra 0 memory frames trên task sách và đĩa):

* **[LiberoMemEventWriter](file:///d:/Thesis_26/src/memory/event_writer.py)**:
  - **FSM State Machine**: Quản lý các chuyển trạng thái:
    1. `initial_support_observed`: Xác định quan hệ ban đầu giữa target và support.
    2. `departed_support`: Mép dưới bounding box của vật thể dịch chuyển lên trên so với support anchor theo tỷ lệ kích thước vật ban đầu (xác nhận nhấc vật).
    3. `returned_to_support`: Vật thể quay trở lại footprint ban đầu của support, vị trí ổn định, gripper mở (aperture $\ge 0.055$m) liên tục trong 3 frame liên tiếp.
  - **Hysteresis & Anti-Double-Count**: Bắt buộc phải có một chu kỳ departure mới trước khi cho phép ghi nhận một return tiếp theo.
  - **Gap & Incomplete History Handling**: Khi mất dấu quan sát (không có detection hợp lệ), hệ thống **không** giả định vật đã rời đi hoặc gán vị trí bằng EEF. Thay vào đó, hệ thống đánh dấu `history_complete = False` và tăng `uncertain_count`.
* **[LiberoMemOracleWriter](file:///d:/Thesis_26/src/memory/libero_mem_oracle.py)**:
  - Truy vấn trực tiếp contact physics và simulation state của MuJoCo/Robosuite.
  - Đóng vai trò đối chứng chẩn đoán (Diagnostic Upper Bound / Oracle Reference).

### 6.3 Memory Store & Immutable Schemas (`src/memory/models.py`)

* **`V2MemoryEvent`**:
  - Ghi nhận `event_id`, `event_type` (`initial_support_observed`, `departed_support`, `returned_to_support`), `target_entity`, `support_entity`, `occurred_at`, `available_at`, `evidence_ids`, `source`.
* **`SchemaV2EpisodeMemory`**:
  - Đóng gói toàn bộ trạng thái memory của episode: `episode_id`, `task_name`, `confirmed_cycles`, `cycle_in_progress`, `history_complete`, `uncertain_count`, `events`.
  - Thiết kế **Deep Immutability**: Sử dụng `MappingProxyType` và tuple cho các danh sách sự kiện và metadata, loại trừ khả năng sửa đổi trạng thái sau khi đã snapshot.

### 6.4 Token-Aware Text Memory Renderer (`src/interfaces/`)

* **[LiberoMemTextRenderer](file:///d:/Thesis_26/src/interfaces/libero_mem_text.py)**:
  - **Tokenizer-Strict Budgeting**: Tích hợp trực tiếp tokenizer gốc của SmolVLA (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`), bảo đảm context bổ sung cộng với instruction gốc không bao giờ vượt quá **48 tokens**.
  - **Privilege Gate**: Cấm tuyệt đối sự rò rỉ của các fact mang nhãn `ORACLE_SIMULATOR` khi chạy ở chế độ evaluation thực tế (`allow_oracle = False`).
  - **Compaction Syntax**:
    - Khi lịch sử đầy đủ:
      ```text
      lift the bowl and place it back on the plate 3 times
      Memory: bowl A: 2 returns to original plate. Cycle underway.
      ```
    - Khi có khoảng mất dấu (gap):
      ```text
      lift the bowl and place it back on the plate 3 times
      Memory: bowl A: at least 2 returns to original plate; history incomplete.
      ```
    - Khi `condition = "off"`: Bảo toàn nguyên vẹn 100% chuỗi instruction gốc không thêm bất kỳ khoảng trắng hay ký tự lạ nào.

### 6.5 Mixed Memory SFT Training Pipeline (`src/training/`)

* **[MixedMemoryDataset & ReplaySampler](file:///d:/Thesis_26/src/training/mixed_sft.py)**:
  - **Dual-Domain 50/50 Balance**: Mỗi optimizer batch bao gồm đúng 50% mẫu từ LIBERO-10 Base Replay và 50% mẫu từ LIBERO-Mem Multi-Cycle Demonstrations.
  - **Round-Robin Task Balancing**: Lấy mẫu đồng đều giữa các task trong từng domain để ngăn các trajectory dài lấn át trajectory ngắn.
* **[SplitLock Management](file:///d:/Thesis_26/src/training/split_lock.py)**:
  - Băm toàn bộ vector trạng thái khởi tạo (64-bit float IEEE-754) để nhóm các demo có cùng initial state vào cùng một split (80% Train, 10% Val, 10% Test).
  - Khóa mật mã `split_lock.json` bảo đảm tuyệt đối không có rò rỉ initial state giữa tập huấn luyện và tập kiểm định.
* **Kaggle 2×T4 Distributed Execution (`scripts/train_memory_sft.py`)**:
  - Huấn luyện phân tán qua HuggingFace Accelerate / DDP trên 2 GPU NVIDIA T4.
  - Sử dụng official Flow-Matching Loss của SmolVLA với Action-Expert LoRA (Rank 16, Alpha 16).
  - Lưu checkpoint và chạy validation định kỳ mỗi 500 optimizer updates trong tổng số 3,000 updates.

### 6.6 Closed-Loop Rollout Engine & Protocol (`src/evaluation/`)

* **[LiberoMemRolloutRunner](file:///d:/Thesis_26/src/evaluation/libero_mem_rollout.py)**:
  - Vòng lặp điều khiển kín (Closed-Loop) chạy ở chu kỳ policy receding horizon $H=50, s=50$.
  - Cập nhật visual memory tại mỗi bước quan sát thông qua `ProcessVisualDetector`.
  - Quản lý 4 điều kiện thực nghiệm chuẩn hóa:
    1. **B0**: Checkpoint SmolVLA nguyên bản (Frozen Baseline, Memory OFF).
    2. **F0**: Checkpoint fine-tune trên tập hỗn hợp nhưng Memory OFF (đối chứng tác động của SFT).
    3. **F1**: Checkpoint fine-tune với Memory ON (sinh từ visual detector thực tế).
    4. **F1-off**: Checkpoint F1 nhưng tắt Memory khi inference (ablation kiểm tra mức độ phụ thuộc vào memory).
* **[LiberoMemMetrics & Protocol](file:///d:/Thesis_26/src/evaluation/libero_mem_metrics.py)**:
  - Đánh giá theo các milestone chính thức của LIBERO-Mem: nhấc vật (lifted), đặt lên đĩa (on-support), thời gian dwell ($\ge 5$ steps), số chu kỳ hoàn thành, và phát hiện lỗi overshoot.

---

## 7. Closed-Loop Execution Lifecycle

```text
[Episode Init: Reset Env & Seed]
              │
              ▼
[Observer: Capture 256x256 RGB & 8D State]
              │
              ▼
[Perception: ProcessVisualDetector worker generates bounding boxes]
              │
              ▼
[Writer: LiberoMemEventWriter updates FSM transitions & cycle count]
              │
              ▼
[Store: Snapshot immutable SchemaV2EpisodeMemory]
              │
              ▼
[Interface: LiberoMemTextRenderer encodes prompt (<= 48 tokens)]
              │
              ▼
[Policy: SmolVLA predicts action chunk (50, 7)]
              │
              ▼
[Controller: OSC_POSE executes s=50 steps at 20 Hz in MuJoCo]
              │
              ▼
[Milestone Evaluator: Checks dwell time & cycle success]
              │
              ├─────────── If not done ───────────┐
              ▼                                   │
[Episode Complete: Log telemetry, video, & traces]│
                                                  │
                                                  ▲
                                                  └── (Loop back to Observer)
```

---

## 8. Memory-Critical Task Taxonomy

| Task Family | Tác vụ cụ thể | Đặc tính quan sát | Vai trò thực nghiệm |
|---|---|---|---|
| **Retention Suite (LIBERO-10 Subset)** | 7/10 tasks đã khóa nguồn (như `put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`) | Quan sát thông thường, chuỗi hành động đơn lẻ | Đánh giá Non-Interference: Đo lường xem mô hình có duy trì được năng lực thao tác cơ bản sau khi SFT không. |
| **LIBERO-Mem T1** | `pick_up_the_bowl_and_place_it_back_on_the_plate` (1 chu kỳ) | Thao tác 1 chu kỳ hoàn chỉnh | Đánh giá năng lực xác nhận hoàn thành thao tác và dừng lại đúng lúc. |
| **LIBERO-Mem T3** | `pick_up_the_bowl_and_place_it_back_on_the_plate_3_times` (3 chu kỳ) | Vị trí bát và đĩa tại cuối mỗi chu kỳ giống hệt nhau; quan sát tức thời tại $t$ không chứa số chu kỳ đã hoàn thành | Đánh giá cốt lõi: Chứng minh memory cung cấp thông tin lịch sử cần thiết để hoàn thành đủ 3 chu kỳ và dừng lại (tránh dừng sớm hoặc overshoot). |

---

## 9. Evaluation Protocol & Statistical Rigor

### 9.1 Invariants Across Evaluations
- Cùng Model Architecture & Tokenizer.
- Cùng bộ seed ngẫu nhiên cho từng episode: $\text{seed} = \text{base\_seed} + \text{task\_index} \times 100 + \text{init\_id}$.
- Cùng cấu hình camera (Dual 256×256) và Controller `OSC_POSE` (20 Hz).
- Cùng Execution Horizon $H=50, s=50$.

### 9.2 Statistical Estimation (Newcombe Method 10)
Để loại trừ lỗi suy biến của công thức Wald khi mẫu nhỏ ($N \le 50$) hoặc tỷ lệ thành công ở biên ($0\%$ hoặc $100\%$), mọi so sánh ghép cặp (paired comparison) đều sử dụng:
* **Newcombe Method 10** cho khoảng tin cậy 95% của hiệu số tỷ lệ thành công: $\Delta \text{SR} = \text{SR}_{\text{Treatment}} - \text{SR}_{\text{Control}}$.
* **Exact McNemar / Permutation Test** hai phía để kiểm định mức ý nghĩa thống kê ($p$-value).

---

## 10. Codebase Architecture & Python Interfaces

Cấu trúc mã nguồn thực tế của dự án:

```text
Thesis_26/
├── src/
│   ├── perception/
│   │   ├── grounding_dino.py       # Zero-shot Grounding DINO model wrapper
│   │   ├── process_detector.py     # Subprocess IPC worker for isolated perception
│   │   ├── object_tracker.py       # Visual & causal object trackers with anchor memory
│   │   ├── visual_evidence.py      # Structured evidence data classes
│   │   └── cached_detector.py      # Frame-hash verified detection cache
│   ├── memory/
│   │   ├── models.py               # Immutable data schemas: V2MemoryEvent, SchemaV2EpisodeMemory
│   │   ├── event_writer.py         # LiberoMemEventWriter (FSM: lift, return, hysteresis)
│   │   ├── observation_writer.py   # Legacy proprioceptive FSM writer (diagnostic baseline)
│   │   ├── libero_mem_oracle.py    # MuJoCo simulation contact/predicate ground-truth writer
│   │   ├── store.py                # Episode-scoped memory store
│   │   ├── updater.py              # State transition updater
│   │   ├── retriever.py            # Task-conditioned memory retrieval
│   │   └── visual_pipeline.py      # High-level visual memory factory
│   ├── interfaces/
│   │   ├── libero_mem_text.py      # Tokenizer-budgeted prompt renderer (<= 48 tokens)
│   │   └── text_memory.py          # General episodic memory text interface
│   ├── training/
│   │   ├── mixed_sft.py            # Mixed SFT dataset & collator
│   │   ├── replay_sampler.py       # 50/50 dual-domain balanced sampler
│   │   ├── split_lock.py           # 64-bit IEEE-754 initial-state split locking
│   │   ├── libero_mem_data.py      # LIBERO-Mem HDF5 data processor
│   │   └── libero_mem_hdf5_loader.py # Streaming HDF5 trajectory loader
│   ├── simulator/
│   │   ├── libero_env.py           # Standard LIBERO MuJoCo environment wrapper
│   │   ├── libero_mem_env.py       # LIBERO-Mem environment wrapper
│   │   └── libero_mem_loader.py    # Upstream BDDL loader for T1/T3
│   ├── models/
│   │   ├── base.py                 # Abstract VLA policy interface
│   │   ├── smolvla/                # SmolVLA adapter, action unnormalization, state vector
│   │   └── minivla/                # MiniVLA adapter (reserved)
│   ├── evaluation/
│   │   ├── libero_mem_rollout.py   # Closed-loop evaluation runner (B0, F0, F1, F1-off)
│   │   ├── libero_mem_metrics.py   # Dwell-time & milestone evaluation
│   │   ├── libero_mem_protocol.py  # Evaluation condition definitions
│   │   ├── rollout.py              # Standard rollout runner
│   │   └── diagnostics.py          # Behavioral failure attribution (G3)
│   └── controllers/                # Robosuite OSC_POSE controller definitions
├── scripts/
│   ├── train_memory_sft.py         # Accelerate DDP training script
│   ├── prepare_libero_mem_sft.py   # Pilot prompt index & shard generator
│   ├── lock_memory_sft_split.py    # Initial-state split lock generator
│   ├── replay_libero_mem_writer.py # Visual writer verification on real telemetry
│   ├── audit_libero_mem_writer.py  # Diagnostic audit script for writer fidelity
│   ├── compare_memory_runs.py      # Seed-matched paired evaluation with Newcombe CI
│   ├── run_benchmark.py            # Main evaluation runner with baseline lock
│   └── kaggle_memory_sft.sh        # Shell automation for Kaggle 2xT4 execution
├── notebooks/
│   └── kaggle_libero_mem_finetune.ipynb # Kaggle execution notebook
└── configs/
    ├── v2_baseline_lock.yaml       # Formal lock for runtime evaluation parameters
    └── models/selected_baseline.yaml # Frozen SmolVLA baseline parameters
```

---

## 11. Implementation Roadmap & Execution Status

```text
Phase 0: Research Problem Definition & Scoping ─────────────────► [HOÀN THÀNH]
Phase 1: Standardized Simulation Infrastructure ────────────────► [HOÀN THÀNH]
Phase 2: Frozen SmolVLA Adapter Integration ────────────────────► [HOÀN THÀNH]
Phase 3: V1 Raw Baseline Validation & Freeze (ADR-0012) ────────► [HOÀN THÀNH & ĐÃ KHÓA]
Phase 4: Task Suite Selection (LIBERO-10 7-task + Mem T1/T3) ───► [HOÀN THÀNH]
Phase 5: Memory Architecture & Visual Perception Worker ────────► [HOÀN THÀNH & AUDITED]
Phase 6: Mixed Memory SFT Adaptation Pipeline ──────────────────► [HOÀN THÀNH & READY TO TRAIN]
Phase 7: Closed-Loop Benchmark Evaluation (B0, F0, F1, F1-off) ──► [READY FOR EXECUTION]
Phase 8: Thesis Defense & Final Reporting ──────────────────────► [IN PROGRESS]
```

> [!IMPORTANT]
> Toàn bộ kiến trúc và mã nguồn đã được thiết kế tuân thủ nguyên tắc **Fail-Fast**, **Strict Provenance**, và **Zero Fabrication**. Mọi kết quả thực nghiệm sẽ được công bố với nhãn phân cấp minh bạch, bảo đảm tính liêm chính học thuật tuyệt đối của luận văn.
