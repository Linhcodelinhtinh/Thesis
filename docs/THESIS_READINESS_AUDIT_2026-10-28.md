# THESIS READINESS & COMPONENT AUDIT — 28/10/2026

> **Đánh giá mức độ sẵn sàng toàn diện của hệ thống Thesis Memory**  
> **Phiên bản:** V2.2 (Mixed SFT Adaptation, Zero-Shot Visual Perception Worker, LIBERO-Mem Benchmark)  
> **Tiêu chuẩn kiểm định:** Tuân thủ tuyệt đối [AGENTS.md](file:///d:/Thesis_26/AGENTS.md), không làm giả dữ liệu, không silent fallback, lưu trữ đầy đủ provenance và kiểm soát chặt chẽ ranh giới khoa học.

---

## 1. Kết luận Tổng thể & Trạng thái Hệ thống

**KẾT LUẬN: HỆ THỐNG ĐÃ SẴN SÀNG VẬN HÀNH THỰC NGHIỆM (EXPERIMENTATION & EVALUATION READY).**

Toàn bộ các component cốt lõi, bao gồm:
1. **Perception Layer** (`GroundingDINO` zero-shot detector + `ProcessVisualDetector` IPC worker).
2. **Visual Memory Engine** (`LiberoMemEventWriter` v3 với FSM departure/return, hysteresis, và incomplete history handling).
3. **Immutable Memory Schemas & Store** (`V2MemoryEvent`, `SchemaV2EpisodeMemory` với deep immutability).
4. **Token-Aware Prompt Interface** (`LiberoMemTextRenderer` với budget $\le 48$ tokens và strict privilege gating).
5. **Mixed Memory SFT Adaptation Pipeline** (50% Base Replay + 50% Memory data, initial-state split lock 64-bit float IEEE-754, HuggingFace Accelerate DDP trên Kaggle 2×T4).
6. **Closed-Loop Evaluation Engine** (`LiberoMemRolloutRunner`, đo đạc milestone dwell-time $\ge 5$ steps, và kiểm định thống kê Newcombe Method 10).

đều đã được xây dựng hoàn chỉnh, tích hợp chặt chẽ và vượt qua các bài kiểm định thành phần (unit/integration tests). Toàn bộ 7 rào cản kỹ thuật (R01–R07) phát hiện trong các kỳ audit trước đã được đóng dứt điểm.

---

## 2. Báo cáo Thẩm định Chi tiết theo từng Component lớn

### Component A: Perception Layer & Decoupled Subprocess Worker (`src/perception/`)
* **Mục tiêu**: Nhận diện vật thể và xác định anchor trong không gian quan sát RGB thực tế mà không sử dụng ground-truth từ simulator.
* **Mã nguồn chính**:
  - [grounding_dino.py](file:///d:/Thesis_26/src/perception/grounding_dino.py): Bọc mô hình `IDEA-Research/grounding-dino-tiny` được pin mật mã SHA-256 trong `resources/manifests/memory_perception_source.json`.
  - [process_detector.py](file:///d:/Thesis_26/src/perception/process_detector.py): Khởi chạy tiến trình worker độc lập chạy môi trường `.venv_memory` qua giao tiếp JSON IPC qua stdin/stdout.
  - [object_tracker.py](file:///d:/Thesis_26/src/perception/object_tracker.py): Gắn anchor đĩa tại frame 0 (khi gripper mở), bám bắt bounding box của bát trong không gian 2D, áp dụng temporal continuity ($\le 3$ frames) và hỗ trợ caching có kiểm tra SHA-256 từng frame RGB.
* **Kết quả thẩm định**:
  - Không có xung đột phụ thuộc giữa thư viện Transformer của detector và môi trường PyTorch/LeRobot của policy.
  - Fail-fast khi worker gặp timeout hoặc lỗi mô hình; không silent fallback sang dữ liệu giả định.

### Component B: Visual Event Memory Writer v3 (`src/memory/`)
* **Mục tiêu**: Thay thế bộ ghi proprioceptive heuristic cũ (vốn gây 0/9,420 memory frames ở bài toán sách và 1.91% ở bài toán bát do các ngưỡng kẹp $z \le 0.95$ và finger gap sai lệch).
* **Mã nguồn chính**:
  - [event_writer.py](file:///d:/Thesis_26/src/memory/event_writer.py): `LiberoMemEventWriter` (v3) vận hành theo máy trạng thái hữu hạn (FSM):
    1. `initial_support_observed`: Ghi nhận quan hệ ban đầu giữa bát và đĩa gốc.
    2. `departed_support`: Mép dưới bounding box của bát nâng lên so với đĩa theo tỷ lệ chiều cao ban đầu của vật (xác nhận nhấc vật).
    3. `returned_to_support`: Bát quay lại vị trí đĩa gốc, tọa độ ổn định, gripper mở (aperture $\ge 0.055$m) trong 3 frame liên tiếp.
  - **Hysteresis Guard**: Bắt buộc phải có một chu kỳ departure mới trước khi cho phép ghi nhận một return tiếp theo, loại trừ hiện tượng đếm trùng chu kỳ khi robot đứng yên.
  - **Incomplete History Handling**: Khi mất dấu quan sát (occlusion / detection gap), hệ thống giữ nguyên số chu kỳ đã xác nhận làm giới hạn dưới, đồng thời đánh dấu `history_complete = False` và tăng `uncertain_count`.
* **Kết quả thẩm định**:
  - Tái hiện chính xác chuỗi sự kiện trên các demonstration tác giả, không bị ảnh hưởng bởi chiều cao mặt bàn hay vị trí cảm biến cổ tay EEF.

### Component C: Memory Models & Store (`src/memory/models.py`, `store.py`)
* **Mục tiêu**: Lưu trữ và quản lý vòng đời dữ liệu sự kiện episode với tính toàn vẹn và bất biến.
* **Mã nguồn chính**:
  - `V2MemoryEvent`: Ghi nhận `event_id`, `event_type`, `occurred_at`, `available_at`, `evidence_ids`, `source`.
  - `SchemaV2EpisodeMemory`: Sử dụng `MappingProxyType` và immutable collections, bảo đảm metadata và event details không thể bị sửa đổi sau khi snapshot.
  - [libero_mem_oracle.py](file:///d:/Thesis_26/src/memory/libero_mem_oracle.py): Truy vấn trực tiếp contact physics từ MuJoCo, đóng vai trò chuẩn chẩn đoán (Oracle Diagnostic Upper Bound).

### Component D: Tokenizer-Budgeted Text Memory Interface (`src/interfaces/`)
* **Mục tiêu**: Định dạng thông tin memory thành context đưa vào prompt VLA mà không vượt quá ngân sách token hoặc làm sai lệch instruction gốc.
* **Mã nguồn chính**:
  - [libero_mem_text.py](file:///d:/Thesis_26/src/interfaces/libero_mem_text.py): `LiberoMemTextRenderer` sử dụng trực tiếp tokenizer GPT2 gốc của SmolVLA (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @ `7b375e1`).
  - **Strict Token Budget**: Tổng độ dài prompt luôn $\le 48$ tokens. Nếu context vượt budget, renderer chủ động lược bỏ fact thay vì cắt xén instruction.
  - **Privilege Gating**: Loại bỏ tuyệt đối các fact mang nhãn `ORACLE_SIMULATOR` khi chạy thực nghiệm `allow_oracle = False`.
  - **Memory OFF Invariance**: Khi `condition = "off"`, giữ nguyên 100% chuỗi instruction gốc.

### Component E: Mixed Memory SFT Adaptation Pipeline (`src/training/`)
* **Mục tiêu**: Giải quyết hiện tượng prompt sensitivity / attention dilution của frozen VLA bằng cách huấn luyện Action-Expert LoRA với tỷ lệ cân bằng 50/50 giữa base skills và memory-augmented data.
* **Mã nguồn chính**:
  - [split_lock.py](file:///d:/Thesis_26/src/training/split_lock.py), [lock_memory_sft_split.py](file:///d:/Thesis_26/scripts/lock_memory_sft_split.py): Băm vector trạng thái khởi tạo (64-bit float IEEE-754) để nhóm các demo có cùng initial state, phân bổ 80% Train, 10% Val, 10% Test. Khóa toàn bộ các demo đã kiểm tra (demos 1, 5, 31) vào tập Train.
  - [replay_sampler.py](file:///d:/Thesis_26/src/training/replay_sampler.py), [mixed_sft.py](file:///d:/Thesis_26/src/training/mixed_sft.py): Bộ lấy mẫu cân bằng chính xác 50% LIBERO-10 Base Replay + 50% LIBERO-Mem Data trong từng optimizer batch. Lấy mẫu task round-robin và uniform per episode.
  - [train_memory_sft.py](file:///d:/Thesis_26/scripts/train_memory_sft.py), [kaggle_memory_sft.sh](file:///d:/Thesis_26/scripts/kaggle_memory_sft.sh), [kaggle_libero_mem_finetune.ipynb](file:///d:/Thesis_26/notebooks/kaggle_libero_mem_finetune.ipynb): Pipeline huấn luyện phân tán Accelerate DDP, FP16, official SmolVLA flow-matching loss trên 2 GPU NVIDIA T4.

### Component F: LIBERO-Mem Closed-Loop Evaluation Protocol (`src/evaluation/`)
* **Mục tiêu**: Đánh giá định lượng hiệu quả của memory trên các tác vụ đa chu kỳ (T1/T3) đồng thời đo lường mức độ duy trì kỹ năng trên tập base (LIBERO-10).
* **Mã nguồn chính**:
  - [libero_mem_loader.py](file:///d:/Thesis_26/src/simulator/libero_mem_loader.py), [libero_mem_env.py](file:///d:/Thesis_26/src/simulator/libero_mem_env.py): Tải BDDL chính thức của T1 (`pick_up_the_bowl_and_place_it_back_on_the_plate`) và T3 (`... 3 times`) từ `libero-mem/LIBERO-Mem` (@ `6df130c`).
  - [libero_mem_rollout.py](file:///d:/Thesis_26/src/evaluation/libero_mem_rollout.py): Vòng lặp điều khiển kín receding horizon $H=50, s=50$, cập nhật visual memory qua detector worker mỗi frame quan sát.
  - [libero_mem_metrics.py](file:///d:/Thesis_26/src/evaluation/libero_mem_metrics.py): Đánh giá milestone (lifted, on-support, dwell time $\ge 5$ steps, cycle count, và phát hiện overshoot).
  - [compare_memory_runs.py](file:///d:/Thesis_26/scripts/compare_memory_runs.py): Ghép cặp chính xác theo tuple `(task_id, init_id, seed)` và tính khoảng tin cậy Newcombe Method 10 kèm kiểm định hoán vị hai phía.

---

## 3. Rà soát việc khắc phục 7 Rào cản Kỹ thuật (R01 — R07 từ kỳ Audit 23/10)

| Mã rào cản | Nội dung vấn đề ban đầu | Biện pháp xử lý kiến trúc | Kết quả xác minh |
|---|---|---|:---:|
| **R01** | Simulator fallback sang fixture khi lỗi backend | Constructor loại bỏ hoàn toàn silent fallback; ném ngoại lệ rõ ràng khi thiếu backend hoặc asset không hợp lệ. | **ĐÃ ĐÓNG** |
| **R02** | Evaluator và reset lifecycle chưa tương đương upstream | Tích hợp đầy đủ tiêu chuẩn dwell-time ($\ge 5$ steps) và milestone predicate upstream; reset sub-goal progress đúng chuẩn. | **ĐÃ ĐÓNG** |
| **R03** | Writer predicted chưa có perception và xác nhận quá mức | Thay thế bằng Grounding DINO detector + visual bounding box vertical displacement FSM; hysteresis chống đếm lặp. | **ĐÃ ĐÓNG** |
| **R04** | Pilot data chưa phản ánh sự kiện & split rò rỉ | Áp dụng split lock theo mã băm 64-bit IEEE-754 initial-state vector; tạo causal prompt index có đầy đủ lineage và frame hashes. | **ĐÃ ĐÓNG** |
| **R05** | Privilege gate và snapshot chưa bất biến sâu | Triển khai `MappingProxyType` cho toàn bộ collections; renderer lọc nghiêm ngặt nguồn `ORACLE_SIMULATOR`. | **ĐÃ ĐÓNG** |
| **R06** | Rollout online chưa đồng nhất với data preparation | Đồng nhất quy ước ảnh (`libero_raw` vs `refined`), dùng chung factory perception worker và renderer tokenizer. | **ĐÃ ĐÓNG** |
| **R07** | Script audit báo READY quá sớm | Tách bạch rõ giữa unit test contract và runtime benchmark readiness; bổ sung 31 unit tests chuyên biệt trong `tests/unit/`. | **ĐÃ ĐÓNG** |

---

## 4. Ma trận Thực nghiệm 4 Điều kiện (4-Way Matched Protocol)

Nhằm cô lập rạch ròi đóng góp khoa học của Memory khỏi tác động của việc fine-tune và chứng minh tính bảo toàn kỹ năng, thực nghiệm được cấu hình theo ma trận 4 điều kiện:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   EVALUATION MATRIX (4-WAY)                                     │
├──────────┬─────────────────────────────┬───────────────────┬────────────────────────────────────┤
│ Điều kiện│ Checkpoint xuất phát        │ Memory lúc chạy   │ Mục đích khoa học đo lường         │
├──────────┼─────────────────────────────┼───────────────────┼────────────────────────────────────┤
│ B0       │ SmolVLA Baseline (Frozen)   │ OFF               │ Điểm chuẩn cơ sở nguyên bản        │
│ F0       │ SFT Adapter (No Memory)     │ OFF               │ Tác động riêng rẽ của pha SFT      │
│ F1       │ SFT Adapter (Memory-Trained)│ ON (Visual RGB v3)│ Đóng góp cốt lõi của Memory        │
│ F1-off   │ SFT Adapter (Memory-Trained)│ OFF               │ Ablation: Chứng minh phụ thuộc     │
│          │                             │                   │ nhân quả vào tín hiệu memory       │
└──────────┴─────────────────────────────┴───────────────────┴────────────────────────────────────┘
```

### Các giả thuyết khoa học cần kiểm định:
1. **Hypothesis 1 (Retention trên Base Suite)**: $\text{SR}(F0, \text{LIBERO-10}) \approx \text{SR}(B0, \text{LIBERO-10})$ và $\text{SR}(F1, \text{LIBERO-10}) \approx \text{SR}(B0, \text{LIBERO-10})$ (Chứng minh 50% Base Replay bảo vệ mô hình khỏi catastrophic forgetting).
2. **Hypothesis 2 (Memory Utility trên Multi-Cycle Suite)**: $\text{SR}(F1, \text{T3}) > \text{SR}(B0, \text{T3})$ và $\text{SR}(F1, \text{T3}) > \text{SR}(F0, \text{T3})$ với mức ý nghĩa $p < 0.05$ (Chứng minh memory giúp hoàn thành chu kỳ lặp lại).
3. **Hypothesis 3 (Causal Reliance)**: $\text{SR}(F1, \text{T3}) > \text{SR}(F1\text{-off}, \text{T3})$ (Chứng minh khi tắt memory, mô hình bị giảm hiệu năng trên bài toán nhiều chu kỳ do thiếu thông tin trạng thái).

---

## 5. Danh mục Kiểm tra Thực thi (Kaggle & Evaluation Execution Checklist)

- [x] **Mã nguồn và kiến trúc**: Đã commit sạch, các file core nằm đúng cấu trúc.
- [x] **Test suite**: 7 project structure tests passed; toàn bộ unit tests cho core, tracker, writers, split lock và dataset passed.
- [x] **Notebook & Scripts**: `kaggle_libero_mem_finetune.ipynb` và `kaggle_memory_sft.sh` đã sẵn sàng, tải asset tự động.
- [ ] **Chạy Huấn luyện trên Kaggle**: Khởi chạy phiên Kaggle 2×T4 với 3,000 steps cho F0 và F1; ghi nhận loss curves.
- [ ] **Tải Checkpoints**: Tải weights LoRA về `experiments/models/sft_f0/` và `experiments/models/sft_f1/`.
- [ ] **Khởi chạy Perception Worker**: Chạy `ProcessVisualDetector` trên `.venv_memory`.
- [ ] **Chạy Rollouts Đánh giá**: Thực thi `libero_mem_rollout.py` trên tập Test split (10% held-out initial states) cho 4 điều kiện.
- [ ] **Xuất Báo cáo So sánh Thống kê**: Chạy `compare_memory_runs.py` và tổng hợp kết quả vào báo cáo tốt nghiệp.
