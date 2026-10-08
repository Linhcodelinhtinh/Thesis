# CẬP NHẬT TIẾN ĐỘ TRIỂN KHAI THESIS

## 1. Định hướng

Mục tiêu cuối của thesis là xây dựng và thực nghiệm một **robot memory system** nhằm đánh giá định lượng: liệu việc bổ sung memory vào một VLA policy có cải thiện khả năng thực hiện các manipulation task có yêu cầu thông tin lịch sử hay không.

Theo định hướng này, toàn bộ implementation được tổ chức thành ba tầng chặt chẽ:

```
                    THESIS EXPERIMENT
                           │
             ┌─────────────┴─────────────┐
             │                           │
        Frozen VLA                  Robot Memory
        Baseline                    Mechanism
             │                           │
             └─────────────┬─────────────┘
                           │
                    Experimental
                       Protocol
                           │
                           ▼
                  Memory OFF vs ON
                           │
                           ▼
                     Evaluation
```

Trong đó:

- **VLA** là policy backbone được **freeze hoàn toàn** (không fine-tune, không train lại).
- **LIBERO / MuJoCo / robosuite** là **experimental substrate/environment**, không phải sản phẩm nghiên cứu cuối.
- **Memory** là **contribution chính của thesis**.
- **OFF vs ON** là experimental comparison cốt lõi được đo đạc ghép cặp (paired comparison) trên cùng điều kiện.
- Các metric được thiết kế để đo tác động của memory lên task performance, tính ổn định và khả năng duy trì/truy xuất thông tin lịch sử.

---

## 2. Scope của implementation

Implementation được phân định ranh giới rõ ràng thành hai khối:

### A. Experimental Infrastructure (Nền tảng thực nghiệm)

Bao gồm:
- Simulation environment tương thích LIBERO (MuJoCo, robosuite).
- Cấu hình robot Franka Panda và bộ điều khiển `OSC_POSE` (20 Hz).
- Camera observation pipeline (256×256 cho `agentview` và `robot0_eye_in_hand`).
- Quản lý nạp task và deterministic initial-state arrays.
- Tích hợp model VLA ([SmolVLAAdapter](file:///D:/Thesis_26/src/models/smolvla/adapter.py)), tiền xử lý và giải mã action.
- Closed-loop rollout, action chunking ($H=50, s=50$).
- Logging, telemetry, trajectory recording, video rollout và chẩn đoán thất bại.

Khối này đã hoàn thiện và đóng vai trò **hạ tầng đo lường ổn định, có kiểm soát**. Đây không phải là đóng góp nghiên cứu mới của luận văn.

### B. Memory Research System (Trọng tâm nghiên cứu)

Khối nghiên cứu can thiệp vào chu trình quan sát – hành động:

```
Observation
     │
     ▼
Perception / Representation (Event / Spatial Anchors)
     │
     ▼
Memory
 ┌───────────────┐
 │ Store         │ (Episode-scoped / Persistent)
 │ Update        │ (Event-driven state transitions)
 │ Retrieve      │ (Task-entity conditioned)
 │ Represent     │ (Textual / Spatial)
 └───────┬───────┘
         │
         ▼
   Memory Context
         │
         ▼
       VLA  (Prompt-injected / Visual prompt)
         │
         ▼
      Action
         │
         ▼
      Robot
         │
         └──────► New Observation
                         │
                         └──────► Memory Update
```

Toàn bộ đóng góp học thuật và thực nghiệm của thesis nằm tại khối này.

---

## 3. Tiến độ triển khai chi tiết theo từng Phase & Work Package

### Phase 0: Research / System Definition
- **Trạng thái:** Hoàn thành.
- Đã xác định bài toán nghiên cứu, phạm vi VLA backbone, tiêu chuẩn benchmark và nguyên tắc cô lập treatment.

### Phase 1: Experimental Environment
- **Trạng thái:** Hoàn thành.
- Nền tảng simulation chuẩn hóa trên Franka Panda, `robosuite==1.4.0`, camera kép native 256×256.

### Phase 2: Frozen VLA Integration
- **Trạng thái:** Hoàn thành & Đã thẩm định.
- Tích hợp [SmolVLAAdapter](file:///D:/Thesis_26/src/models/smolvla/adapter.py) (`lerobot/smolvla_libero` @ `31d453f`).
- Chuẩn hóa contract vector trạng thái 8D, cơ chế receding horizon $H=50, s=50$.
- Khóa quy ước gripper direct (`-1 = Open, +1 = Close`) theo Robosuite native ([ADR-0009](file:///D:/Thesis_26/docs/DECISIONS.md#L81-L92)).

### Phase 3: Baseline Validation & Freeze (V1 Baseline Lock)
- **Trạng thái:** Hoàn thành — V1 Baseline đã khóa ([ADR-0012](file:///D:/Thesis_26/docs/DECISIONS.md#L125-L146)).
- Thẩm định toàn diện qua các Promotion Gates G0–G4:
  - **G0 (Hạ tầng):** 100% deterministic reproducibility, test suite passed.
  - **G1 (Thao tác cơ bản):** Grasp 68.1%, Lift 50.0%, Place 53.8%.
  - **G2 (Độ tin cậy benchmark):** Đạt 70.0% trên Acceptance 10-task suite [95% CI: 54.6%–81.9%]; đạt 48.13% trên toàn bộ 40 tasks (init states 0..3: Goal 67.5%, Object 50.0%, Spatial 47.5%, LIBERO-10 27.5%).
  - **G3 (Chẩn đoán hành vi):** Phân loại behavioral failure phase cho 100% ca thất bại (Reach 67.5%, Lift 12.0%, Grasp 8.4%). Task phức tạp chuỗi (`LIBERO-10`) rơi xuống 27.5%, trong đó 67.5% thất bại ở bước chuyển tiếp subtask, tạo khoảng trống định lượng (headroom) rõ ràng cho memory.
  - **G4 (Độ trễ):** Mean ~57 ms/step trên GPU, zero OOM.
- Khóa manifest mật mã trong `experiments/baseline_v1/` (`FROZEN_BASELINE_MANIFEST.yaml`, `freeze_audit_signoff.json`).
- Xếp hạng thực thi: `LIBERO-DERIVED (HOST_WIN32_PY3.12)` với nhãn `NON-COMPARABLE_OFFICIAL_PAPER`. Baseline được freeze hoàn toàn.

### Phase 4: Experimental Task Design
- **Trạng thái:** Đang định hình tập task trọng tâm.
- Chuyển hướng từ việc chạy dàn trải sang tập trung vào các task nhạy cảm với lịch sử (History-Dependent / Multi-stage như `LIBERO-10`, và các task có đối tượng gây nhiễu không gian như `LIBERO-Spatial`).
- Pilot set hiện tại sử dụng 40 tasks × 4 initial states (0..3) làm tập so sánh cặp chuẩn.

---

### Phase 5: Memory Architecture & Implementation (Trọng tâm V2 hiện tại)

Theo kế hoạch tại [V2_MEMORY_EXECUTION_PLAN.md](file:///D:/Thesis_26/docs/V2_MEMORY_EXECUTION_PLAN.md), phạm vi hiện tại tập trung hoàn toàn vào **V2.1 Text Memory**.

> [!IMPORTANT]
> **Hiện trạng bộ sinh Memory (Writer):**
> Codebase hiện tại **chưa có writer lấy memory từ quan sát thực tế (realistic perception)**. Bộ ghi duy nhất đang được tích hợp vào chu trình rollout là [OracleMemoryWriter](file:///D:/Thesis_26/src/memory/oracle_writer.py) (sử dụng thông tin mô phỏng oracle).

Tiến độ chi tiết theo các Work Packages (P0 – P10):

| Gói công việc | Nội dung kế hoạch | Trạng thái thực tế | Đánh giá / Mã nguồn liên quan |
|---|---|---|---|
| **P0 — Baseline contract lock** | Khóa cấu hình, commit, state/action semantics và runtime settings | **Đã tạo config, chưa khóa runtime** | Đã tạo [v2_baseline_lock.yaml](file:///D:/Thesis_26/configs/v2_baseline_lock.yaml), nhưng runner chưa nạp hoặc kiểm tra file này lúc chạy. |
| **P1 — Memory data model** | Định nghĩa cấu trúc sự kiện, thực thể, độ tin cậy và provenance | **Hoàn thành** | [models.py](file:///D:/Thesis_26/src/memory/models.py): `MemoryEvent`, `ObjectMemory`, `Evidence`, enum `EvidenceSource`, `MemoryStatus`. |
| **P2 — Store & updater** | Quản lý vòng đời episode, deterministic state projection | **Hoàn thành** | [store.py](file:///D:/Thesis_26/src/memory/store.py), [updater.py](file:///D:/Thesis_26/src/memory/updater.py): Store reset theo từng episode, cập nhật deterministic. |
| **P3 — Retrieval & text interface** | Truy xuất theo task entity, lọc độ cũ, render prompt | **Hoàn thành** | [retriever.py](file:///D:/Thesis_26/src/memory/retriever.py), [text_memory.py](file:///D:/Thesis_26/src/interfaces/text_memory.py): Xếp hạng theo relevance/recency, render block `[Episode memory]`. |
| **P4 — Shadow integration** | Thu thập, cập nhật, lưu memory trong khi policy nhận instruction gốc | **Hoàn thành & Smoke run** | [rollout.py](file:///D:/Thesis_26/src/evaluation/rollout.py) hỗ trợ `--memory-condition text_shadow`; đã chạy smoke test độc lập. |
| **P5 — Text-only treatment** | Ghép chuỗi text memory vào prompt VLA trước inference | **Hoàn thành & Smoke run** | [rollout.py](file:///D:/Thesis_26/src/evaluation/rollout.py) hỗ trợ `--memory-condition text_only`; đã chạy smoke test. |
| **P6 — Paired pilot** | Đánh giá so sánh cặp OFF vs text-only trên cùng task/states | **Công cụ báo cáo lỗi logic ghép cặp** | Đã viết [compare_memory_runs.py](file:///D:/Thesis_26/scripts/compare_memory_runs.py), nhưng cách ghép cặp sai seed, khoảng tin cậy suy biến. |
| **P7 — V2.1 Decision** | Quyết định xúc tiến, điều chỉnh hay dừng text memory | **Chưa thực hiện** | Phải chờ kết quả pilot sau khi xử lý xong các điểm nghẽn. |
| **P8 — Spatial memory** | Lưu tọa độ 3D và chiếu visual prompt | **Chưa triển khai** | Hoãn lại theo đúng kế hoạch (sau V2.1). |
| **P9 — Combined & ablations** | Kết hợp Text + Spatial, ablation các thành phần | **Chưa triển khai** | Hoãn lại theo đúng kế hoạch. |
| **P10 — Robustness** | Đánh giá nhiễu độ trễ, sai lệch ngữ nghĩa, lỗi quan sát | **Chưa triển khai** | Hoãn lại theo đúng kế hoạch. |

---

## 4. Các điểm nghẽn kỹ thuật cần xử lý dứt điểm trước khi chạy Pilot

Qua rà soát thực tế mã nguồn và các smoke run ban đầu, có **5 vấn đề cốt lõi** bắt buộc phải khắc phục trước khi tiến hành thực nghiệm Pilot chính thức:

### 1. Dữ kiện khởi tạo đang bị gán cứng nhưng mang nhãn oracle (`ORACLE_SIMULATOR`)
- **Vấn đề:** Trong [oracle_writer.py:49-85](file:///D:/Thesis_26/src/memory/oracle_writer.py#L49-L85), phương thức `on_episode_start` gán cứng toàn bộ thực thể mục tiêu thành trạng thái `"on_table"`; trạng thái của goal container/fixture được suy diễn hoàn toàn dựa trên tên chuỗi (ví dụ: chứa `"drawer"` gán là `"closed"`, chứa `"stove"` gán là `"off"`, còn lại mặc định là `"empty"`). Phương thức này có nhận tham số `env` và `obs` nhưng **hoàn toàn không truy vấn simulator hay observation** để kiểm tra tính chân thực của trạng thái vật lý ban đầu.
- **Xung đột thực tế:** Trong [rollout.py:259-269](file:///D:/Thesis_26/src/evaluation/rollout.py#L259-L269), hàm giải nén thực thể đưa cả target lẫn goal vào danh sách `task_entities`, rồi truyền `task_entities` vào `target_entity_names` của `OracleMemoryWriter`. Khi gặp task có goal container là `plate`, hệ thống sinh ra hai sự kiện mâu thuẫn đồng thời cho cùng đối tượng `plate`: vừa là `"on_table"`, vừa là `"empty"`.
- **Hệ quả:** Các fact này đều được đánh nhãn `EvidenceSource.ORACLE_SIMULATOR` với `confidence: 1.0`. Việc tiêm các fact mâu thuẫn hoặc suy diễn thiếu căn cứ vật lý vào prompt VLA làm sai lệch hành vi của mô hình và vi phạm nguyên tắc liêm chính dữ liệu.
- **Yêu cầu xử lý:** `OracleMemoryWriter.on_episode_start` phải đọc vị trí và quan hệ hình học thực từ simulator (MuJoCo/robosuite state) thay vì gán nhãn tĩnh; phân định rạch ròi giữa target và goal container trong `rollout.py`.

### 2. Báo cáo "paired" không ghép cùng seed (Mismatched Seeds & Collapsed Statistics)
- **Vấn đề:** Trong báo cáo thử nghiệm [PAIRED_MEMORY_REPORT.md:5-13](file:///D:/Thesis_26/experiments/results/v2_memory/smoke_test_text_only/PAIRED_MEMORY_REPORT.md#L5-L13), kết quả so sánh text-only so với baseline ghi nhận chênh lệch $-100$ điểm phần trăm (Baseline 1/1, Treatment 0/1).
- **Nguyên nhân:** Kiểm tra chi tiết cho thấy run baseline sử dụng `seed: 442` ([baseline episode.json:214](file:///D:/Thesis_26/experiments/results/raw_baseline/ablation_wait10_seed42/libero_spatial_task0/init_0/episode.json#L214)), trong khi run treatment sử dụng `seed: 42` ([text-only episode.json:1547](file:///D:/Thesis_26/experiments/results/v2_memory/smoke_test_text_only/libero_spatial_task0/init_0/episode.json#L1547)). Việc chỉ trùng khớp thư mục `init_id=0` nhưng khác seed không tạo thành một cặp tương đương hợp lệ trong thực nghiệm có kiểm soát.
- **Lỗi thống kê:** Hàm tính toán trong [compare_memory_runs.py:80-84](file:///D:/Thesis_26/scripts/compare_memory_runs.py#L80-L84) dùng công thức Wald cho phương sai của chênh lệch tỷ lệ paired. Khi số cặp $N=1$, phương sai tính ra bằng $0$, dẫn tới khoảng tin cậy 95% suy biến thành $[-100\%, -100\%]$. Công thức Wald không hợp lệ khi cỡ mẫu nhỏ hoặc tỷ lệ ở biên ($0\%$ hoặc $100\%$).
- **Yêu cầu xử lý:** Sửa công cụ ghép cặp bắt buộc kiểm tra trường `seed` trong `episode.json`; thay thế ước lượng Wald bằng phương pháp phù hợp cho mẫu nhỏ / phân phối nhị thức (ví dụ: Newcombe paired interval hoặc exact permutation test).

### 3. Smoke run mới dừng ở mức tín hiệu ban đầu, chưa kết luận được tác động của Memory
- **Vấn đề:** Trong smoke run với `seed: 42`, điều kiện `text_shadow` đạt 1/1 ([shadow episode.json:6](file:///D:/Thesis_26/experiments/results/v2_memory/smoke_test_text_shadow/libero_spatial_task0/init_0/episode.json#L6)), còn `text_only` đạt 0/1 do timeout sau 1000 bước ([text-only episode.json:6](file:///D:/Thesis_26/experiments/results/v2_memory/smoke_test_text_only/libero_spatial_task0/init_0/episode.json#L6)).
- **Nhận định:** Đây là một hiện tượng quan trọng cần điều tra (liệu việc chèn text memory có làm phân tán attention vào instruction gốc hay do thông tin khởi tạo mâu thuẫn làm sai lệch chuyển động). Tuy nhiên, với kích thước mẫu smoke run ($N=1$), **tuyệt đối chưa đủ cơ sở khoa học để kết luận memory gây ra regression**.

### 4. Baseline Lock chưa khóa được quy trình chạy (Missing Runtime Enforcement & Dirty Git Tree)
- **Vấn đề:** Tệp cấu hình [v2_baseline_lock.yaml](file:///D:/Thesis_26/configs/v2_baseline_lock.yaml) đã được tác giả soạn thảo, nhưng trình thực thi [run_benchmark.py](file:///D:/Thesis_26/scripts/run_benchmark.py) không hề nạp, so khớp hoặc kiểm tra ràng buộc của file này trước khi chạy.
- **Sai lệch phiên bản:** Lock ghi nhận commit `43ddd06e...`, trong khi metadata của các V2 smoke run ghi `a11be...`. Ngoài ra, thư mục làm việc hiện có nhiều file sửa đổi và file mới chưa commit (working tree `dirty`).
- **Yêu cầu xử lý:** Runner phải nạp `v2_baseline_lock.yaml`, xác thực các tham số runtime ($H, s$, controller, resolution) và kiểm tra working tree / git hash trước khi cho phép ghi nhận kết quả thực nghiệm.

### 5. Một số yêu cầu audit và telemetry còn thiếu sót
- **Vấn đề trong Renderer:** Trong [text_memory.py:79-82](file:///D:/Thesis_26/src/interfaces/text_memory.py#L79-L82), khi candidate context vượt ngưỡng `max_context_chars`, renderer dùng lệnh `break` dừng lại nhưng không ghi nhận fact nào bị lược bỏ và không phát cờ `truncated: True` vào metadata của episode.
- **Vấn đề trong Công cụ So sánh:** [compare_memory_runs.py:45-56](file:///D:/Thesis_26/scripts/compare_memory_runs.py#L45-L56) duyệt qua các file `episode.json` với cấu trúc `try ... except Exception: pass` âm thầm bỏ qua các episode đọc lỗi mà không cảnh báo, đồng thời không kiểm tra provenance hay model config hash giữa hai thư mục chạy.

---

## 5. Trạng thái tổng thể

| Hạng mục | Trạng thái hiện tại | Đánh giá / Vai trò |
|---|---|---|
| **Định nghĩa bài toán nghiên cứu** | Hoàn thành | Đã xác định rõ mục tiêu memory-centric |
| **Hạ tầng mô phỏng (MuJoCo/robosuite)** | Hoàn thành | Nền tảng thực nghiệm ổn định |
| **Chuẩn hóa Franka Panda + Controller** | Hoàn thành | Khóa `OSC_POSE` 20 Hz, gripper direct polarity |
| **Tích hợp mô hình VLA (SmolVLA)** | Hoàn thành | Khóa checkpoint, 8D state, chunking $H=50, s=50$ |
| **V1 Raw Baseline Evaluation & Lock** | **Hoàn thành (Đã khóa)** | Đạt các gates G0–G4; lưu trữ manifest mật mã |
| **Đặc tả V2 Baseline Lock (P0)** | Đã author config | Cần bổ sung validator runtime trong runner |
| **Memory Data Model & Store (P1–P2)** | Hoàn thành | Quản lý sự kiện, state transition theo episode |
| **Memory Retrieval & Text Renderer (P3)** | Hoàn thành | Cần bổ sung cờ telemetry khi truncation xảy ra |
| **Shadow Mode Integration (P4)** | Hoàn thành | Đã tích hợp rollout hook & chạy smoke test |
| **Text-Only Mode Integration (P5)** | Hoàn thành | Đã tích hợp rollout hook & chạy smoke test |
| **Perception-based Memory Writer** | **Chưa triển khai** | Hiện chỉ có [OracleMemoryWriter](file:///D:/Thesis_26/src/memory/oracle_writer.py) |
| **Công cụ Paired Reporting (P6)** | Cần sửa đổi logic | Phải sửa ghép đúng seed và tính khoảng tin cậy chuẩn |
| **Thực nghiệm Paired Pilot (V2.1)** | **Đang chuẩn bị** | Bị chặn bởi 5 điểm nghẽn kỹ thuật nêu trên |
| **Quyết định V2.1 (P7)** | Chưa thực hiện | Chờ dữ liệu pilot hợp lệ |
| **Spatial / Combined / Robustness (P8–P10)** | Chưa triển khai | Hoãn lại sau khi hoàn thành V2.1 decision |

---

## 6. Kế hoạch hành động trước mắt (Immediate Action Plan)

Để chuyển từ giai đoạn smoke test sang thực nghiệm Pilot hợp lệ, thứ tự triển khai tiếp theo gồm:

```
[1. Sửa Oracle Writer & Rollout Entities]
   - Đọc simulator state thật trong on_episode_start
   - Tách biệt target object vs goal container
          │
          ▼
[2. Hoàn thiện Telemetry & Audit]
   - Thêm cờ truncation & dropped facts vào TextMemoryRenderer
   - Runner kiểm tra v2_baseline_lock.yaml và git clean status
          │
          ▼
[3. Chuẩn hóa Công cụ So sánh Paired (compare_memory_runs.py)]
   - Ghép cặp chặt chẽ theo (task_id, init_id, seed)
   - Thay thế Wald CI bằng phương pháp thống kê mẫu nhỏ chính xác
          │
          ▼
[4. Thực hiện Paired Pilot Run (V2.1)]
   - Chạy tập 40 tasks × 4 initial states (160 episodes) cùng base seed
   - So sánh OFF vs Text-Only (và Text-Shadow)
          │
          ▼
[5. Đánh giá Cột mốc P7 (V2.1 Decision)]
   - Báo cáo phân tích hồi quy / cải thiện và quyết định mở rộng V2.2
```

---

## 7. Ý nghĩa của LIBERO trong project sau khi đổi scope

LIBERO được định vị chính xác là:

> **controlled experimental substrate** (nền tảng thực nghiệm có kiểm soát)

chứ không phải:

> **research target** (mục tiêu nghiên cứu cần tối ưu điểm benchmark)

Vai trò của LIBERO là cung cấp môi trường thao tác chuẩn hóa, trạng thái khởi tạo xác định và các chuỗi tác vụ manipulation để đo lường. Đóng góp khoa học cốt lõi của luận văn nằm ở tầng phía trên môi trường: **Cơ chế Robot Memory (biểu diễn, cập nhật, truy xuất và giao tiếp với VLA)** cùng với **thực nghiệm so sánh đối chứng nghiêm ngặt OFF vs ON**.