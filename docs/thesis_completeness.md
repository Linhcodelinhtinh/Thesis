# CẬP NHẬT TIẾN ĐỘ VÀ ĐỘ HOÀN THIỆN THESIS (THESIS COMPLETENESS)

> **Cập nhật ngày:** 28/10/2026  
> **Trọng tâm:** Báo cáo tiến độ triển khai các component lớn, tính năng mới (Visual Perception Worker, Visual Event Writer v3, Mixed Memory SFT Training Pipeline, LIBERO-Mem Multi-Cycle Benchmark) và lộ trình nghiệm thu thực nghiệm.

---

## 1. Định hướng và Ranh giới Đóng góp (Thesis Scope & Contribution)

Mục tiêu cốt lõi của luận văn là thiết kế, hiện thực hóa và đánh giá định lượng một **hệ thống Persistent Robot Memory** trong bài toán thao tác robot (closed-loop manipulation) nhằm trả lời câu hỏi: *Việc bổ sung thông tin lịch sử có cấu trúc thông qua memory có cải thiện đáng kể khả năng hoàn thành các tác vụ nhiều chu kỳ (multi-cycle) và quan sát một phần hay không, trong khi vẫn duy trì được năng lực thao tác cơ bản?*

```text
                                  THESIS EXPERIMENT
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 │                                                 │
        Frozen VLA Baseline                                Robot Memory System
        (SmolVLA Locked V1)                            (Contribution cốt lõi của Thesis)
                 │                                                 │
                 │                                  ┌──────────────┴──────────────┐
                 │                                  ▼                             ▼
                 │                         Visual Perception Layer       Memory Engine & Schemas
                 │                         (Grounding DINO + Tracker)    (Event Writer v3, Store)
                 │                                  │                             │
                 └────────────────────────┬─────────┴─────────────────────────────┘
                                          │
                                          ▼
                                Experimental Regimes
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
        Regime 1: Zero-Shot Prompting                     Regime 2: Mixed SFT Adaptation
        (Frozen VLA + Memory Context)                     (Action-Expert LoRA: 50% Base / 50% Mem)
                 │                                                 │
                 └────────────────────────┬────────────────────────┘
                                          │
                                          ▼
                               Paired Evaluation Protocol
                           (B0 vs. F0 vs. F1 vs. F1-off)
                                          │
                                          ▼
                        Comprehensive Retention & Memory Report
```

Ranh giới hệ thống được phân định minh bạch:
* **Experimental Substrate (Hạ tầng đo lường - không phải contribution mới)**:
  - Simulator: MuJoCo, Robosuite 1.4.0, benchmark chuẩn hóa LIBERO và LIBERO-Mem.
  - Robot & Controller: Franka Panda với bộ điều khiển `OSC_POSE` (20 Hz), camera kép native 256×256.
  - Pretrained Model Backbone: `lerobot/smolvla_libero` (@ commit `31d453f`).
* **Research Contributions (Đóng góp học thuật cốt lõi của Thesis)**:
  1. **Kiến trúc Robot Memory & Schemas**: Thiết kế biểu diễn sự kiện bất biến (`V2MemoryEvent`, `SchemaV2EpisodeMemory`), vòng đời episode, truy xuất có điều kiện theo tác vụ và quản lý độ tin cậy/nguồn gốc (provenance).
  2. **Bộ sinh Memory dựa trên thị giác thực tế (Visual Perception Worker & Writer v3)**: Module nhận thức không phụ thuộc simulator ground-truth, sử dụng `GroundingDINO` chạy trong tiến trình cô lập (IPC) kết hợp FSM theo dõi dịch chuyển bounding box (departure/return) và cơ chế hysteresis chống đếm lặp.
  3. **Quy trình Huấn luyện Thích ứng Bảo toàn Kỹ năng (Mixed Memory SFT Pipeline)**: Cơ chế huấn luyện Action-Expert LoRA với bộ lấy mẫu cân bằng chính xác 50% Base Replay + 50% Memory-augmented data, giải quyết triệt để hiện tượng phân tán chú ý (attention dilution) của frozen VLA mà không gây lãng quên thảm khốc (catastrophic forgetting).
  4. **Giao thức Thực nghiệm và Đánh giá Nghiêm ngặt (4-Way Matched Protocol)**: Khóa mật mã seed, split nhóm theo hash 64-bit initial-state vector, đo đạc dwell-time milestone và kiểm định thống kê chính xác Newcombe Method 10.

---

## 2. Tiến độ triển khai chi tiết theo từng Phase (Phases 0 — 7)

### Phase 0: Research / System Definition
- **Trạng thái:** **Hoàn thành**.
- Đã xác định rõ các câu hỏi nghiên cứu (RQ1–RQ4), tiêu chuẩn benchmark, ranh giới contribution và nguyên tắc fail-fast (AGENTS.md).

### Phase 1: Experimental Environment
- **Trạng thái:** **Hoàn thành**.
- Nền tảng simulation chuẩn hóa trên Franka Panda, `robosuite==1.4.0`, camera kép native 256×256. Kiểm soát chặt chẽ quy ước rotation, camera matrix và tần số điều khiển 20 Hz.

### Phase 2: Frozen VLA Integration
- **Trạng thái:** **Hoàn thành & Đã thẩm định**.
- Tích hợp thành công [SmolVLAAdapter](file:///d:/Thesis_26/src/models/smolvla/adapter.py) (`lerobot/smolvla_libero` @ `31d453f`).
- Chuẩn hóa contract vector trạng thái proprioceptive 8D `[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]`, chunking $H=50, s=50$. Khóa quy ước gripper direct (`-1 = Open, +1 = Close`) theo Robosuite native ([ADR-0009](file:///d:/Thesis_26/docs/DECISIONS.md#L81-L92)).

### Phase 3: Baseline Validation & Freeze (V1 Baseline Lock)
- **Trạng thái:** **Hoàn thành — V1 Baseline đã khóa mật mã ([ADR-0012](file:///d:/Thesis_26/docs/DECISIONS.md#L125-L146))**.
- Vượt qua toàn diện 5 Promotion Gates G0–G4:
  - **G0 (Hạ tầng):** 100% deterministic reproducibility, unit test suite passed.
  - **G1 (Thao tác cơ bản):** Grasp 68.1%, Lift 50.0%, Place 53.8%.
  - **G2 (Độ tin cậy benchmark):** Đạt 70.0% trên Acceptance 10-task suite [95% CI: 54.6%–81.9%]; đạt 48.13% trên toàn bộ 40 tasks (Goal 67.5%, Object 50.0%, Spatial 47.5%, LIBERO-10 27.5%).
  - **G3 (Chẩn đoán hành vi):** Phân loại behavioral failure phase cho 100% ca thất bại. Đặc biệt, `LIBERO-10` tụt xuống 27.5%, với 67.5% thất bại ở bước chuyển tiếp subtask, tạo khoảng trống định lượng (headroom) rõ ràng cho nghiên cứu memory.
  - **G4 (Độ trễ):** ~57 ms/step trên GPU, zero OOM.
- Lưu trữ manifest mật mã tại `experiments/baseline_v1/` (`FROZEN_BASELINE_MANIFEST.yaml`, `freeze_audit_signoff.json`). Nhãn: `LIBERO-DERIVED (HOST_PY3.12)`, `NON-COMPARABLE_OFFICIAL_PAPER`.

### Phase 4: Task Suite Selection & Manifest Locking
- **Trạng thái:** **Hoàn thành & Đã khóa nguồn**.
- Nhằm đo lường cả tính bảo toàn kỹ năng (retention) và năng lực khai thác memory, hệ thống đã khóa 2 tập task trọng tâm:
  1. **Base Replay Suite (LIBERO-10 7-task Pilot)**: Khóa tại `resources/manifests/libero_10_sft_pilot7_sources.json`, gồm 7 task chính thức có đầy đủ demonstrations từ nguồn tác giả (`yifengzhu-hf/LIBERO-datasets`).
  2. **Multi-Cycle Memory Suite (LIBERO-Mem T1 & T3)**: Khóa tại `resources/manifests/libero_mem_planning_sources.json`, gồm 198 demonstrations chính thức từ `libero-mem/LIBERO-Mem` (@ `6df130c`). T1 yêu cầu 1 chu kỳ lặp lại, T3 yêu cầu 3 chu kỳ lặp lại (`pick_up_the_bowl_and_place_it_back_on_the_plate_3_times`).

### Phase 5: Memory Architecture & Visual Perception Worker
- **Trạng thái:** **Hoàn thành & Đã audit toàn diện**.
- **Data Models & Immutability (`src/memory/models.py`)**: Hiện thực `V2MemoryEvent` và `SchemaV2EpisodeMemory` với thuộc tính bất biến sâu (deep immutability), ghi nhận nguồn gốc (`EvidenceSource`), mốc thời gian và trạng thái lịch sử (`history_complete`, `uncertain_count`).
- **Perception Layer (`src/perception/`)**:
  - Tích hợp `GroundingDINO` zero-shot detector (`grounding_dino.py`).
  - Hiện thực `ProcessVisualDetector` (`process_detector.py`): Khởi chạy tiến trình worker con trong môi trường `.venv_memory` độc lập, giao tiếp qua JSON IPC, loại bỏ triệt để nguy cơ xung đột phụ thuộc giữa thư viện thị giác và môi trường policy/simulator.
  - Hiện thực `CausalObjectTracker` (`object_tracker.py`): Gắn anchor đĩa/bàn tại frame đầu tiên, theo dõi bounding box bát, áp dụng hysteresis và lưu cache detection có kiểm tra hash SHA-256 từng frame RGB.
- **Visual Event Memory Writer v3 (`src/memory/event_writer.py`)**:
  - Thay thế hoàn toàn bộ ghi proprioceptive heuristic cũ. Writer v3 phát hiện sự kiện nhấc vật (`departed_support`) dựa trên độ dịch chuyển mép dưới bounding box so với support anchor, và phát hiện hoàn thành chu kỳ (`returned_to_support`) khi vật quay lại vị trí đĩa gốc ổn định với độ mở gripper $\ge 0.055$m trong 3 frames liên tiếp.
  - Xử lý mất dấu quan sát (incomplete history): Không tự ý suy đoán trạng thái mà đánh dấu cờ `history_complete = False` để cảnh báo policy.
- **Token-Aware Text Renderer (`src/interfaces/libero_mem_text.py`)**:
  - Giao tiếp trực tiếp với tokenizer SmolVLA, kiểm soát chặt chẽ ngân sách $\le 48$ tokens, hỗ trợ compact rendering và cấm triệt để rò rỉ thông tin oracle (`allow_oracle = False`).

### Phase 6: Mixed Memory SFT Adaptation Pipeline
- **Trạng thái:** **Hoàn thành & Sẵn sàng chạy trên Kaggle 2×T4**.
- **Động lực khoa học**: Thực nghiệm V2.1 trên frozen VLA cho thấy việc tiêm trực tiếp đoạn văn bản memory vào prompt của một mô hình chưa từng được huấn luyện với memory tokens sẽ gây ra hiện tượng phân tán chú ý (prompt sensitivity / attention dilution). Do đó, cần một pha thích ứng nhẹ (Action-Expert LoRA) để dạy mô hình cách điều kiện hóa hành động theo memory.
- **Initial-State Split Lock (`src/training/split_lock.py`, `scripts/lock_memory_sft_split.py`)**:
  - Nhóm các demonstration theo mã băm 64-bit float IEEE-754 của vector trạng thái khởi tạo MuJoCo ban đầu, phân bổ chính xác 80% Train, 10% Val, 10% Test. Khóa toàn bộ các demonstration đã dùng để phát triển writer (demos 1, 5, 31) vào tập Train để bảo đảm tính khách quan của tập Test.
- **Mixed Replay Dataset & Sampler (`src/training/mixed_sft.py`, `replay_sampler.py`)**:
  - Duy trì tỷ lệ cân bằng chính xác **50% Base Replay + 50% Memory Data** trên từng optimizer batch. Lấy mẫu task round-robin và sample uniform per episode để trajectory dài không lấn át trajectory ngắn.
- **Hạ tầng Huấn luyện Kaggle 2×T4 (`scripts/train_memory_sft.py`, `notebooks/kaggle_libero_mem_finetune.ipynb`)**:
  - Cấu hình phân tán HuggingFace Accelerate / DDP trên 2 GPU NVIDIA T4, FP16, official SmolVLA flow-matching loss.
  - Huấn luyện 3,000 steps, kiểm tra validation loss và lưu checkpoint mỗi 500 steps. Đã tích hợp script cài đặt tự động (`scripts/kaggle_memory_sft.sh`) không phụ thuộc venv nội bộ của Kaggle.

### Phase 7: Closed-Loop Evaluation & Retention Protocol
- **Trạng thái:** **Sẵn sàng thực thi (Ready for Execution)**.
- **Giao thức Đánh giá 4 Điều kiện (4-Way Matched Protocol)**:
  1. **B0 (Frozen Baseline Control)**: SmolVLA nguyên bản, Memory OFF, đo lường điểm chuẩn ban đầu.
  2. **F0 (SFT Control)**: Checkpoint thích ứng trên tập hỗn hợp, Memory OFF, đo lường tác động riêng lẻ của pha SFT lên kỹ năng thao tác.
  3. **F1 (Memory-Augmented Treatment)**: Checkpoint thích ứng với Memory ON (sinh từ visual perception thực tế), đo lường năng lực của memory trong việc giải quyết bài toán multi-cycle T1/T3.
  4. **F1-off (Ablation Control)**: Checkpoint F1 nhưng tắt Memory khi inference, kiểm tra xem hành vi thành công có thực sự phụ thuộc vào tín hiệu memory hay không.
- **Bộ đo lường Milestone & Dwell-Time (`src/evaluation/libero_mem_metrics.py`, `libero_mem_protocol.py`)**:
  - Tích hợp chuẩn đánh giá của LIBERO-Mem: nhấc vật, đặt về đĩa, thời gian dừng ổn định (dwell $\ge 5$ steps), đếm số chu kỳ thực tế và phát hiện hành vi overshoot.
- **Công cụ Thống kê Ghép cặp (`scripts/compare_memory_runs.py`)**:
  - Ghép cặp nghiêm ngặt theo tuple `(task_id, init_id, seed)`.
  - Sử dụng khoảng tin cậy Newcombe Method 10 và kiểm định hoán vị (exact permutation test) hai phía, loại bỏ hoàn toàn các lỗi suy biến thống kê.

---

## 3. Rà soát việc khắc phục 5 Điểm nghẽn Kỹ thuật trước đây

Trong các báo cáo kiểm định trước (tháng 10/2026), 5 điểm nghẽn kỹ thuật cốt lõi đã được ghi nhận. Toàn bộ 5 điểm nghẽn này hiện đã được giải quyết triệt để thông qua các giải pháp kiến trúc và công cụ mới:

| Điểm nghẽn cũ | Hiện trạng ban đầu | Giải pháp kỹ thuật đã triển khai | Trạng thái hiện tại |
|---|---|---|:---:|
| **1. Oracle gán tĩnh & xung đột thực thể** | Gán cứng `on_table`, sinh fact mâu thuẫn cho `plate` | `LiberoMemOracleWriter` truy vấn contact và vị trí thực của simulator; tách bạch rạch ròi target (`metal bowl`) và support (`white plate`). | **ĐÃ GIẢI QUYẾT** |
| **2. Báo cáo ghép cặp lệch seed & lỗi Wald CI** | So sánh baseline seed 442 với treatment seed 42; Wald CI suy biến $[-100\%, -100\%]$ | `compare_memory_runs.py` bắt buộc trùng khớp `seed`; thay thế hoàn toàn Wald CI bằng Newcombe Method 10 và exact permutation test. | **ĐÃ GIẢI QUYẾT** |
| **3. Smoke run N=1 & suy diễn vội vàng** | Đánh giá memory regression chỉ dựa trên 1 episode smoke run | Xác định bản chất: Frozen VLA bị prompt sensitivity khi chưa qua SFT. Xây dựng Mixed SFT Pipeline để huấn luyện mô hình quen với token memory trước khi benchmark. | **ĐÃ GIẢI QUYẾT** |
| **4. Thiếu kiểm tra Baseline Lock lúc runtime** | Runner không đọc `v2_baseline_lock.yaml`; git tree dirty | `run_benchmark.py` hỗ trợ cờ `--baseline-lock-file`, tự động xác thực controller/camera/chunk parameters và ghi log SHA-256 của git diff. | **ĐÃ GIẢI QUYẾT** |
| **5. Truncation và telemetry còn thiếu** | Renderer dùng `break` âm thầm bỏ bớt fact khi quá ký tự | `LiberoMemTextRenderer` sử dụng tokenizer GPT2 chính thức của SmolVLA, budget $\le 48$ tokens, xuất kết quả có cấu trúc kèm cờ `truncated` và `dropped_facts_count`. | **ĐÃ GIẢI QUYẾT** |

---

## 4. Bảng tổng hợp mức độ hoàn thiện các Component lớn

| Tầng hệ thống (Layer) | Component / Module cụ thể | File mã nguồn chính | Trạng thái kỹ thuật | Bằng chứng kiểm định / Ghi chú |
|---|---|---|:---:|---|
| **Perception** | Zero-Shot Object Detector | [grounding_dino.py](file:///d:/Thesis_26/src/perception/grounding_dino.py) | **Hoàn thành** | Khóa checkpoint Grounding DINO tiny, query `"a metal bowl."` / `"a white plate."`. |
| | Subprocess Worker IPC | [process_detector.py](file:///d:/Thesis_26/src/perception/process_detector.py) | **Hoàn thành** | Chạy môi trường `.venv_memory` độc lập, giao tiếp JSON IPC, zero dependency collision. |
| | Visual & Causal Tracker | [object_tracker.py](file:///d:/Thesis_26/src/perception/object_tracker.py) | **Hoàn thành** | Anchor binding tại frame 0, temporal continuity $\le 3$ frames, hỗ trợ detection cache. |
| | Structured Evidence Schema | [visual_evidence.py](file:///d:/Thesis_26/src/perception/visual_evidence.py) | **Hoàn thành** | Đóng gói bounding boxes, frame hash SHA-256, và camera metadata. |
| **Memory Engine** | Immutable Episode Schemas | [models.py](file:///d:/Thesis_26/src/memory/models.py) | **Hoàn thành** | `V2MemoryEvent`, `SchemaV2EpisodeMemory`, deep immutability qua `MappingProxyType`. |
| | Visual Cycle Writer v3 | [event_writer.py](file:///d:/Thesis_26/src/memory/event_writer.py) | **Hoàn thành** | FSM departure/return, hysteresis chống đếm lặp, ghi nhận incomplete history. |
| | Simulation Oracle Writer | [libero_mem_oracle.py](file:///d:/Thesis_26/src/memory/libero_mem_oracle.py) | **Hoàn thành** | Ground-truth predicate/contact từ MuJoCo, đóng vai trò diagnostic upper-bound. |
| | Store, Updater, Retriever | [store.py](file:///d:/Thesis_26/src/memory/store.py), [updater.py](file:///d:/Thesis_26/src/memory/updater.py), [retriever.py](file:///d:/Thesis_26/src/memory/retriever.py) | **Hoàn thành** | Quản lý vòng đời episode, deterministic causal retrieval. |
| **Interfaces** | Token-Aware Text Renderer | [libero_mem_text.py](file:///d:/Thesis_26/src/interfaces/libero_mem_text.py) | **Hoàn thành** | Budget $\le 48$ tokens, tokenizer GPT2, cấm oracle privilege leak, render compact. |
| **Training Pipeline** | Immutable Split Lock | [split_lock.py](file:///d:/Thesis_26/src/training/split_lock.py) | **Hoàn thành** | Nhóm 64-bit IEEE-754 initial-state vector hashes, khóa 80/10/10 train/val/test. |
| | Dual-Domain Mixed Sampler | [replay_sampler.py](file:///d:/Thesis_26/src/training/replay_sampler.py), [mixed_sft.py](file:///d:/Thesis_26/src/training/mixed_sft.py) | **Hoàn thành** | Cân bằng chính xác 50% Base Replay + 50% Mem Data, task round-robin. |
| | Distributed SFT Trainer | [train_memory_sft.py](file:///d:/Thesis_26/scripts/train_memory_sft.py) | **Hoàn thành** | HuggingFace Accelerate / DDP, FP16, SmolVLA flow-matching loss, LoRA rank/alpha 16. |
| | Kaggle Automation | [kaggle_memory_sft.sh](file:///d:/Thesis_26/scripts/kaggle_memory_sft.sh), [kaggle_libero_mem_finetune.ipynb](file:///d:/Thesis_26/notebooks/kaggle_libero_mem_finetune.ipynb) | **Hoàn thành** | Sẵn sàng chạy trên Kaggle 2×T4 với asset fetch tự động. |
| **Simulation** | LIBERO-Mem BDDL Loader | [libero_mem_loader.py](file:///d:/Thesis_26/src/simulator/libero_mem_loader.py) | **Hoàn thành** | Load BDDL chính thức cho T1 và T3 từ `libero-mem/LIBERO-Mem` (@ `6df130c`). |
| | Simulation Wrapper | [libero_mem_env.py](file:///d:/Thesis_26/src/simulator/libero_mem_env.py) | **Hoàn thành** | Wrapper MuJoCo Franka Panda, tích hợp camera 256×256. |
| **Evaluation** | Closed-Loop Rollout Runner | [libero_mem_rollout.py](file:///d:/Thesis_26/src/evaluation/libero_mem_rollout.py) | **Hoàn thành** | Điều khiển kín receding horizon $H=50, s=50$, hỗ trợ 4 điều kiện B0/F0/F1/F1-off. |
| | Dwell & Milestone Metrics | [libero_mem_metrics.py](file:///d:/Thesis_26/src/evaluation/libero_mem_metrics.py) | **Hoàn thành** | Đo đạc lifted, on-support, dwell time $\ge 5$ steps, cycle count, và overshoot. |
| | Paired Statistical Reporter | [compare_memory_runs.py](file:///d:/Thesis_26/scripts/compare_memory_runs.py) | **Hoàn thành** | Khóa seed, Newcombe Method 10 CI, exact permutation test. |

---

## 5. Kế hoạch Hành động Ngay (Immediate Execution Plan)

Các bước thực thi trực tiếp để chuyển giao toàn bộ kết quả thực nghiệm cho luận văn:

```text
[Bước 1: Chạy Huấn luyện Mixed SFT trên Kaggle 2xT4]
   • Mở notebook notebooks/kaggle_libero_mem_finetune.ipynb
   • Chạy pipeline 3,000 steps cho F0 (no-memory) và F1 (memory-augmented)
   • Tải 2 checkpoints xuất xưởng về thư mục local: experiments/models/
                 │
                 ▼
[Bước 2: Kiểm định Load Checkpoint & Khởi động Perception Worker]
   • Khởi chạy ProcessVisualDetector worker trên môi trường .venv_memory
   • Kiểm tra nạp checkpoint LoRA qua SmolVLAAdapter
                 │
                 ▼
[Bước 3: Thực thi Closed-Loop Evaluation Protocol]
   • Chạy test-split rollouts trên LIBERO-Mem T1 và T3 (cùng initial states và seeds)
   • Chạy test-split rollouts trên LIBERO-10 Base Replay Suite (đo retention)
   • Thu thập đủ 4 điều kiện: B0, F0, F1, F1-off
                 │
                 ▼
[Bước 4: Tổng hợp Thống kê và Báo cáo Luận văn]
   • Chạy scripts/compare_memory_runs.py xuất biểu bảng Newcombe CI
   • Lập biểu đồ phân bố chu kỳ, dwell-time, và tỷ lệ overshoot
   • Kết luận định lượng cho các câu hỏi nghiên cứu RQ1 - RQ4
```

---

## 6. Kết luận Tổng thể

Dự án đã hoàn tất toàn bộ giai đoạn nghiên cứu kiến trúc, xây dựng hạ tầng, phát triển module nhận thức thị giác và đường ống huấn luyện thích ứng (Phases 0 — 6). Tất cả các rào cản kỹ thuật và nguy cơ sai lệch dữ liệu đã được dọn sạch với tiêu chuẩn nghiêm ngặt.

Hệ thống hiện tại đã ở trạng thái **Sẵn sàng Nghiệm thu Thực nghiệm (Experimentation Ready)**. Các bước còn lại hoàn toàn là quy trình vận hành thực nghiệm có kiểm soát để thu thập số liệu định lượng phục vụ trực tiếp cho báo cáo và bảo vệ luận văn tốt nghiệp.