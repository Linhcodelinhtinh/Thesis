# Kiểm Tra Số Lượng Demonstration LIBERO-MEM (T1–T4) & Đánh Giá Phân Chia Dữ Liệu Huấn Luyện

> **Ngày thực hiện:** 13/11/2026  
> **Mục tiêu:** Xác minh chính xác số lượng trajectories/demonstrations hiện có trên đĩa cho các task LIBERO-MEM T1–T4; đối chiếu với thiết kế gốc 120 trajectories/task của tác giả benchmark; kiểm tra số lượng demonstrations thực tế trong các đợt fine-tuning cũ; và đánh giá tính bất hợp lý của cơ chế chia split 80/10/10 cũ.

---

## 1. Kết Luận Nhanh (Executive Summary)

1. **Nhận định của bạn là hoàn toàn chính xác theo đúng tài liệu và mã nguồn gốc của LIBERO-Mem**:
   - LIBERO-Mem được tác giả thiết kế sinh **120 initial states** (`mem_1_develop_init_states.py`) và thu thập **120 raw action trajectories** cho mỗi task (`mem_2_collect_demonstration.py`), tổng cộng **10 tasks × 120 = 1.200 trajectories** (được lưu trữ tại [libero-mem/LIBERO-Mem-Raw](https://huggingface.co/datasets/libero-mem/LIBERO-Mem-Raw)).
   - Quá trình refine (`mem_3_refine_demonstration.py` với cờ `--max_replays 100`) chỉ lấy **100 trajectories đầu tiên (indices 0..99)** để replay trong mô phỏng, gắn nhãn object-centric annotations và thêm 20 no-op actions nhằm tạo tập dữ liệu huấn luyện chính thức [libero-mem/LIBERO-Mem](https://huggingface.co/datasets/libero-mem/LIBERO-Mem).
   - **20 trajectories còn lại (indices 100..119)** được tác giả chủ đích giữ lại làm **tập đánh giá chưa từng thấy (unseen evaluation set)**, được mã hóa rõ ràng trong mã nguồn đánh giá chính thức (`mem_5_run_evaluation_env_gt.py` và `mem_6_run_evaluation_env_pred.py` với `--eval_mode unseen`, chọn `len(orig_data) - [20..1]`).
2. **Sai lầm của cơ chế split 80/10/10 cũ**:
   - Khi áp dụng split 80/10/10 lên tập refined data (~100 demos), hệ thống cũ đã **vô tình cắt giảm dữ liệu 2 lần (double-carving)**.
   - Việc này làm giảm số lượng ví dụ huấn luyện cho VLA policy một cách không cần thiết (T1 giảm từ 100 xuống 80; T3 giảm từ 98 xuống 78), làm mất đi hàng nghìn frame hành động quý giá trong khi 20 unseen evaluation states vốn đã tồn tại độc lập ở tập raw benchmark.
3. **Thực tế các đợt fine-tuning cũ**:
   - **Run 1 (F0 Baseline no_memory-20261029)**: Train trên tổng cộng **438 demonstrations** (Base 7 tasks × 40 = 280 demos; Mem T1 = 80 demos, T3 = 78 demos, tổng Mem = 158 demos).
   - **Run 2 (F1 Mem-Adapter V1 mem-adapter-20261105)**: Train trên tổng cộng **558 demonstrations** (Base 10 tasks × 40 = 400 demos; Mem T1 = 80 demos, T3 = 78 demos, tổng Mem = 158 demos).
   - **Run 3 (Derived V1 Mem-Adapter mem-derived-v1)**: Train trên tổng cộng **552 demonstrations** (Base 10 tasks × 40 = 400 demos; Mem T1 = 77 demos, T3 = 75 demos, tổng Mem = 152 demos).

---

## 2. Bằng Chứng Mã Nguồn & Thiết Kế Gốc Của Tác Giả LIBERO-Mem

Các bằng chứng kiểm tra trực tiếp từ thư mục `external/libero-mem/`:

### 2.1. README Dataset của Tác Giả (`external/libero-mem/scripts/demonstration_data/libero_mem/README.md`)
> *"Please refer to this [LIBERO-Mem-Raw](https://huggingface.co/datasets/libero-mem/LIBERO-Mem-Raw) repo for downloading the raw setting to obtain initial states (120 trajs per task) for evaluation setup.*  
> *Otherwise, please refer to [LIBERO-Mem](https://huggingface.co/datasets/libero-mem/LIBERO-Mem) for the full training setting (100 trajs per task)."*

### 2.2. Pipeline Tạo Dữ Liệu Tác Giả

| Script | Tham số quy định số lượng | Mục đích |
|---|---|---|
| `mem_1_develop_init_states.py` | `--num-demonstration 120` | Tạo 120 initial states cho mỗi task (lưu file `.pruned_init`). |
| `mem_2_collect_demonstration.py` | `--num-demonstration 120` | Thu thập 120 quỹ đạo điều khiển thô (raw trajectories). |
| `mem_3_refine_demonstration.py` | `--max_replays 100` | Replay 100 quỹ đạo đầu (indices 0..99), trích xuất RGB, gắn nhãn object reasoning và padding 20 no-ops để xuất file HDF5 train. |
| `mem_5_run_evaluation_env_gt.py`<br>`mem_6_run_evaluation_env_pred.py` | `--eval_mode seen`<br>`--eval_mode unseen` | **seen**: Lấy `indices = [0..19]` (20 quỹ đạo đầu trong tập train).<br>**unseen**: Lấy `indices = [100..119]` (20 quỹ đạo cuối không hề có trong tập train). |

Trích đoạn mã nguồn trong `mem_5_run_evaluation_env_gt.py` (dòng 844–849) và `mem_6_run_evaluation_env_pred.py` (dòng 1148–1152):
```python
test_indices = []
if args.eval_mode == 'seen':
    test_indices = np.arange(0, min(args.eval_runs, len(orig_data.keys())))
elif args.eval_mode == 'unseen':
    test_indices = np.arange(min(args.eval_runs, 20), 0, -1)
    test_indices = len(orig_data.keys()) - test_indices
```
Khi `orig_data` chứa đủ 120 quỹ đạo thô:
- `eval_mode == 'seen'` kiểm tra 20 quỹ đạo: `0, 1, 2, ..., 19`.
- `eval_mode == 'unseen'` kiểm tra 20 quỹ đạo: `120 - 20 = 100, 101, ..., 119`.

---

## 3. Thống Kê Số Lượng Demonstrations T1–T4 Hiện Có Trên Đĩa

Kiểm tra trực tiếp các file HDF5 hiện có trong kho lưu trữ `d:\Thesis_26\data\`:

| Task ID | Tên File HDF5 | Vị Trí Lưu Trữ | Số Demos Có Sẵn | Tổng Số Frames | Dải Demo ID | Các ID Bị Khuyết | Ghi Chú Kỹ Thuật |
|---|---|---|:---:|:---:|:---:|:---:|---|
| **T1** | `KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate_demo.hdf5` | `data/libero_mem/` | **100** | 14.746 | `demo_1` .. `demo_100` | *Không* | Tác giả replay thành công đủ 100/100 quỹ đạo. |
| **T2** | `KITCHEN_SCENE1_2_lift_the_bottle_and_put_it_down_on_the_plate_demo.hdf5` | `data/libero_mem_suite_sources/` | **98** | 15.433 | `demo_1` .. `demo_100` | `demo_59`, `demo_95` | Thiếu 2 demo do physics nondeterminism làm replay thất bại ở bước `mem_3` của tác giả. |
| **T3** | `KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times_demo.hdf5` | `data/libero_mem/` | **98** | 28.111 | `demo_1` .. `demo_100` | `demo_48`, `demo_81` | Thiếu 2 demo do physics nondeterminism làm replay thất bại ở bước `mem_3` của tác giả. |
| **T4** | `KITCHEN_SCENE1_4_pick_up_the_bottle_and_put_it_down_the_plate_3_times_demo.hdf5` | `data/libero_mem_suite_sources/` | **99** | 28.183 | `demo_1` .. `demo_100` | `demo_89` | Thiếu 1 demo do physics nondeterminism làm replay thất bại ở bước `mem_3` của tác giả. |

> [!NOTE]
> - Các task **T5–T10** hiện **chưa có file dữ liệu HDF5** trên ổ đĩa local.
> - Đối với biến thể **derived v1** (`data/libero_mem_derived_v1/`):
>   - **T1 derived**: có **96 demos** (14.149 frames) — có 4 demos gốc thất bại trong tiêu chí strict success của runner local.
>   - **T3 derived**: có **92 demos** (26.381 frames) — có 6 demos gốc thất bại trong tiêu chí strict success của runner local.

---

## 4. Xác Minh Các Đợt Fine-Tuning Cũ Đã Train Trên Bao Nhiêu Demonstrations

Rà soát toàn bộ các artifact và manifest huấn luyện trong thư mục `experiments/training/`:

### 4.1. Run 1: Pilot 7-task Baseline (`no_memory-20261029-175154`)
- **Mục tiêu:** Huấn luyện F0 baseline (action-expert LoRA trên SmolVLA) không dùng memory.
- **Manifest khóa:** `resources/manifests/libero_mem_t1_t3_base7_split_v1.json`
- **Cơ cấu dữ liệu:**
  - Base (7 tasks LIBERO-10): Mỗi task chọn 40 train, 5 val, 5 test $\rightarrow$ **280 train demos**.
  - Mem T1: 80 train, 10 val, 10 test $\rightarrow$ **80 train demos**.
  - Mem T3: 78 train, 10 val, 10 test $\rightarrow$ **78 train demos**.
- **Tổng số demonstrations dùng để train:** **$280 + 80 + 78 = 438\text{ demonstrations}$** (trong đó Mem chỉ có 158 demos).
- **Số demonstrations bị giữ làm val/test:** 110 demos (40 Mem và 70 Base).

### 4.2. Run 2: Full 10-task Mem-Adapter V1 (`mem-adapter-20261105-083543`)
- **Mục tiêu:** Huấn luyện Mem-Adapter V1 kết hợp memory RGB v3 trên T1/T3 và replay toàn bộ 10 task base của LIBERO-10.
- **Manifest khóa:** `experiments/training/results_mem_2/mem-adapter-v1/dataset-manifest.json` (dựa trên `libero_mem_t1_t3_base10_split_v1.json`).
- **Cơ cấu dữ liệu:**
  - Base (10 tasks LIBERO-10): Mỗi task chọn 40 train, 5 val, 5 test $\rightarrow$ **400 train demos**.
  - Mem T1: 80 train, 10 val, 10 test $\rightarrow$ **80 train demos**.
  - Mem T3: 78 train, 10 val, 10 test $\rightarrow$ **78 train demos**.
- **Tổng số demonstrations dùng để train:** **$400 + 80 + 78 = 558\text{ demonstrations}$** (trong đó Mem chỉ có 158 demos).
   - **Số demonstrations bị giữ làm val/test:** 140 demos (40 Mem demos, gồm 10 val + 10 test mỗi task, và 100 Base demos).

### 4.3. Run 3: Derived V1 Mem-Adapter (`results_mem_derived/mem-adapter-v1/output`)
- **Mục tiêu:** Huấn luyện Mem-Adapter trên tập derived demonstrations đã loại bỏ các quỹ đạo không đạt chuẩn mô phỏng.
- **Manifest khóa:** `experiments/training/results_mem_derived/mem-adapter-v1/dataset-manifest.json` (tổng 688 episodes trong lock).
- **Cơ cấu dữ liệu:**
  - Base (10 tasks LIBERO-10): 40 train mỗi task $\rightarrow$ **400 train demos**.
  - Mem T1 (derived): 77 train, 9 val, 10 test $\rightarrow$ **77 train demos**.
  - Mem T3 (derived): 75 train, 8 val, 9 test $\rightarrow$ **75 train demos**.
- **Tổng số demonstrations dùng để train:** **$400 + 77 + 75 = 552\text{ demonstrations}$** (trong đó Mem chỉ có 152 demos).

---

## 5. Phân Tích Hệ Quả Của Việc Cắt Split 80/10/10 & Đề Xuất Khắc Phục

### 5.1. Hệ Quả Của Sai Lầm Cũ
1. **Thiếu hụt dữ liệu (Data Starvation):**
   - Các mô hình VLA lớn như SmolVLA (450M params) và OpenVLA (7B params) khi học các thao tác chuỗi dài lặp lại (long-horizon multi-step manipulation) như T1–T4 cần tối đa sự đa dạng về góc tiếp cận, tư thế vật thể và cách xử lý va chạm.
   - Hai task mất 20 demo mỗi task, tổng cộng **8.552 frame** theo khóa v1 và khóa v2 (34.305 → 42.857 frame train Mem).
2. **Sai lệch giao thức so sánh (Protocol Divergence):**
   - Khi so sánh với kết quả trong bài báo gốc của tác giả LIBERO-Mem, tác giả huấn luyện trên **toàn bộ 100 trajectories đã refine** và chỉ đánh giá trên 20 trajectories unseen.
   - Giao thức cũ thay đổi lượng dữ liệu huấn luyện và roster đánh giá. Cần thực nghiệm đối sánh để đo ảnh hưởng của thay đổi đến success rate.

### 5.2. Đề Xuất Quy Trình Chuẩn Hóa
1. **Dữ liệu huấn luyện (Training Set):**
   - Đưa **toàn bộ 100% các refined demonstrations có sẵn** vào tập huấn luyện:
     - T1: Toàn bộ 100 demos.
     - T2: Toàn bộ 98 demos.
     - T3: Toàn bộ 98 demos.
     - T4: Toàn bộ 99 demos.
2. **Dữ liệu đánh giá (Evaluation / Test Set):**
   - Không lấy mẫu từ tập 100 refined demos để làm test.
   - Sử dụng đúng **20 initial states chưa từng thấy (unseen initial states: trajectories 100..119)** từ bộ dữ liệu thô `LIBERO-Mem-Raw` đúng theo thiết kế `--eval_mode unseen` của tác giả.
3. **Triển khai T1/T3:**
   - Khóa v2 đã được dựng từ HDF5 raw, refined và base đã kiểm hash. Roster raw unseen có 40 lượt trên 20 nhóm initial state dùng chung giữa hai task.
   - Train mới cặp F0/F1 trên cùng 598 demo train. Xem [roadmap và quyết định cho từng checkpoint](thesis/IMPLEMENTATION_AND_WRITING_ROADMAP_2026-11-13.md).
