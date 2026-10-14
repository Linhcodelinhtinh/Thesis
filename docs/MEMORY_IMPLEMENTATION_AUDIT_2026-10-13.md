# Kiểm tra mức sẵn sàng của Memory — 2026-10-13

## Kết luận

V2.1 text memory đã có pipeline chạy xuyên suốt và artifact rollout thật. Chưa đủ điều kiện chạy full benchmark nhằm kết luận memory cải thiện hoặc làm giảm hiệu năng. Cần sửa tính đúng của writer, ngân sách token, baseline lock và tính toàn vẹn của paired evaluation trước.

Không cần viết lại toàn bộ memory core. Spatial memory, combined memory và robustness ablation chưa được triển khai trong đường chạy hiện tại.

## Phạm vi và bằng chứng kiểm tra

- HEAD lúc kiểm tra: `12890e5` (working tree sạch trước khi thêm tài liệu này).
- Đọc code `src/memory/`, `src/interfaces/text_memory.py`, rollout, runner, comparator, checkpoint processor config và episode artifacts.
- Chạy trong `.venv_smolvla` trên WSL: **34 passed in 116.56s**, gồm bốn file test memory/comparison và project structure.
- Chạy tokenizer offline trên prompt đã lưu, với tokenizer được chỉ định bởi checkpoint; không tải weights hoặc thay preprocessing.
- Chạy các probe ngữ nghĩa của writer bằng input tổng hợp. Đây là kiểm thử phần mềm, **không phải benchmark episode hoặc policy fallback**.
- Đối soát số thành công trực tiếp từ `episode.json`.
- Không chạy thêm closed-loop policy rollout và không sửa implementation trong lượt kiểm tra này.
- Có tiến trình WSL đang chạy `run_benchmark.py --preset full --episodes-per-task 5 -s 10 --seed 42 --memory-condition off --output-dir experiments/results/full-clean --record-video`. Đó là raw baseline; chưa phải phép so sánh memory. Không can thiệp tiến trình này.

Lệnh kiểm thử đã chạy:

```bash
cd /mnt/d/Thesis_26
.venv_smolvla/bin/python -m pytest -q \
  tests/unit/test_memory_core.py \
  tests/unit/test_memory_writers.py \
  tests/unit/test_rollout_memory.py \
  tests/unit/test_compare_memory_runs.py \
  tests/test_project_structure.py
```

## Những gì đã hoạt động

| Thành phần | Trạng thái và giới hạn |
|---|---|
| MemoryEvent / ObjectMemory / MemoryQuery | Có schema, episode/step, confidence, validity và evidence source; một số trường ngữ nghĩa vẫn là chuỗi tự do. |
| Store / updater | Có lịch sử event và latest object state; store mới theo episode, kiểm tra episode mismatch và duplicate event ID. |
| Retriever | Lọc entity, confidence, tuổi thông tin và validity; thứ tự xác định. Mặc định hết hạn sau 200 bước. |
| Text renderer | Pass-through khi không có fact; tách nguồn oracle; có audit cắt theo **ký tự**, chưa theo token thực tế. |
| Rollout | Có `off`, `text_shadow`, `text_only`; ghi snapshot, retrieved IDs, context và policy prompt. Memory đi vào instruction trước official preprocessing. |
| Writers | Có OracleMemoryWriter và ObservationMemoryWriter; chất lượng và lifecycle chưa đủ cho mọi task. |
| Paired report | Có ghép task/init, kiểm tra seed khi cả hai đều tồn tại, contingency table và CI; chưa khóa đủ provenance và coverage. |
| Spatial / combined / robustness | Chưa có đường thực thi tương ứng; CLI chỉ chấp nhận ba condition text nêu trên. |

Unit test rollout hiện dùng mock policy để kiểm tra wiring. Test pass không chứng nhận model hiểu memory, writer đúng trên cảnh thật, hoặc OFF/shadow đồng nhất trong runtime thật.

## Kết quả thực nghiệm hiện có

| Artifact | Kết quả đếm lại | Cách sử dụng |
|---|---|---|
| `experiments/results/v2_memory/pilot_smoke_off` | OFF: **4/5** | Pilot lịch sử, năm task, một state/task. |
| `experiments/results/v2_memory/pilot_smoke_text_only` | Oracle text: **0/5** | Cùng seed từng cặp với OFF; tín hiệu giảm hiệu năng trong cấu hình cũ, chưa đủ kết luận về memory tổng quát. |
| `experiments/results/v2_memory/smoke_obs_shadow` | Observation shadow: **1/1** | Chứng minh một lần rollout hoàn tất và có trace. |
| `experiments/results/v2_memory/smoke_obs_text` | Observation text: **0/1**, 1000 bước | Smoke rất nhỏ; không phải paired OFF pilot đầy đủ. |
| `experiments/results/memory_text_only` | **46** oracle text episodes (**2** success) và **22** observation shadow episodes (**5** success) | Thư mục trộn treatment/writer, chỉ phủ bảy task Spatial; không được gộp thành một score text memory. |

Các nhóm trên không được cộng dồn thành một benchmark chung. Các run memory này được ghi dưới stack Windows/Python 3.12.2/robosuite 1.5.2, execution horizon `s=50`, và mã nguồn trước HEAD hiện tại.

Baseline đóng băng `raw_baseline/full_benchmark_40` có **160 episode = 40 task × 4 states**, 77 success, `s=50`; đây không phải 40 × 50. Các run mới như `baseline_full_40` và `full-clean` dùng `s=10`; `full-clean` còn dùng WSL/Python 3.10.21/robosuite 1.4.0. Không dùng chúng làm đối chứng trực tiếp cho memory cũ.

`PAIRED_MEMORY_REPORT.md` của pilot cũ vẫn chứa CI kiểu cũ có cận dưới −115.1 điểm phần trăm. Không dùng CI đó. Comparator hiện tại tính lại từ cùng episode được delta −80 điểm phần trăm và CI xấp xỉ [−96.38, −19.26], nhưng mẫu vẫn chỉ năm cặp và mang các hạn chế về implementation/provenance nêu trong tài liệu này. Không suy diễn thành tác dụng của phiên bản hiện tại.

Tất cả kết quả này là **LIBERO-DERIVED / NON-COMPARABLE_OFFICIAL_PAPER**, không phải official LIBERO score.

## Những vấn đề cần sửa trước benchmark lớn

### 1. Prompt memory bị tokenizer cắt mà audit hiện tại không phát hiện

`resources/checkpoints/smolvla_libero/policy_preprocessor.json` đặt `max_length=48`, `truncation=true`; config model cũng ghi 48. Adapter vẫn gọi official preprocessor. Renderer cho phép tối đa 512 ký tự memory và thêm nhiều metadata vào prompt.

Kiểm tra offline bằng tokenizer hiện có:

- `smoke_obs_text`: **20/20** prompt có memory vượt 48 token; prompt đầu có **150 token**. 48 token đầu kết thúc giữa metadata `confidence`, nên phần lớn fact không tới model.
- Pilot oracle: cả **29/29** prompt có context đều vượt 48 token. Prompt đầu tùy task dài **109–189 token**.
- Observation smoke còn có **19/20** lần bị cắt ngay ở renderer theo ký tự. Đây là hai tầng cắt khác nhau.

Cần renderer tính ngân sách bằng tokenizer chính thức, giữ nguyên instruction và chỉ đưa fact ngắn, ưu tiên thông tin hữu ích. Đưa source/ID/confidence chi tiết vào log thay vì lặp trong prompt. Ghi token IDs/mask hoặc hash và phần text thực tế sau tokenizer. Không âm thầm tăng `max_length` vì sẽ đổi preprocessing contract.

### 2. Observation writer chưa xác nhận được ngữ nghĩa sự kiện

Nguồn: `src/memory/observation_writer.py`.

- Khởi tạo `detected` từ danh sách tên task mà không kiểm tra ảnh; `obs={}` vẫn tạo fact confirmed.
- `closing && finger_gap > 0.02` có thể phát `observed_grasp` ngay khi gripper còn mở. Chưa có xác nhận vật thể hoặc theo dõi identity.
- Confidence 0.70–0.80 là hằng số trong code; chưa tìm thấy artifact hiệu chuẩn để hỗ trợ mô tả “calibrated”.
- Sau release không reset `_is_grasped` / `_is_lifted`; `_is_released` chặn release tiếp theo. Probe hai chu kỳ chỉ ghi nhận chu kỳ đầu.
- Sau slip, `_is_lifted` vẫn true; probe tái hiện slip lặp ở hai bước liên tiếp.
- Thiếu observation bắt buộc được thay bằng giá trị zero ở một số nhánh, thay vì reject hoặc đánh dấu unknown.

Cần state machine theo từng object, debounce/hysteresis, quy tắc xác nhận/release/regrasp/slip rõ ràng và trạng thái uncertain cho bằng chứng chưa đủ. Đo event precision/recall trên telemetry thật với oracle làm nhãn chẩn đoán, không đưa nhãn oracle vào condition observation.

### 3. Entity mapping và oracle chưa đủ cho long-horizon

Nguồn: `src/evaluation/rollout.py:265`, `src/evaluation/diagnostics.py:109`, `src/memory/oracle_writer.py`.

- Rollout tìm entity bằng dò chuỗi instruction, chỉ truyền một target; writers thao tác `target_entity_names[0]`.
- Trong artifact task “put both alphabet soup and tomato sauce”, memory chỉ có `alphabet_soup` và `basket`; thiếu tomato sauce.
- Task mở middle drawer được map thành `cabinet`; memory không biểu diễn đúng drawer identity và tiến trình mở/đóng.
- Oracle có nhánh trả `on_table` / `empty` / `closed` khi không resolve được dữ liệu, rồi ghi `confidence=1.0`, `CONFIRMED`. Probe với `env=None` tái hiện việc tạo `soup=on_table` và `basket=empty`. Đây là nhánh code có thật; không khẳng định nhánh này đã chạy trong mọi benchmark cũ.
- Placement oracle dùng điều kiện khoảng cách 0.08 m, chưa tương đương predicate đặt vật của từng task.

Cần mapping rõ từ task/BDDL chính thức, identity đầy đủ cho nhiều vật và fixture, writer riêng cho các loại sự kiện liên quan. Thiếu bằng chứng phải fail hoặc unknown. Không thay robot/assets/task để làm writer dễ hơn.

### 4. Baseline lock và paired comparator còn thiếu kiểm tra

Nguồn: `configs/v2_baseline_lock.yaml`, `scripts/run_benchmark.py:482`, `scripts/compare_memory_runs.py:180`.

- Lock vẫn ghi commit `43ddd06`, Windows CPU, `s=50`; HEAD và runtime đang chạy khác.
- Validator chưa kiểm tra phần lớn thông số đã khai báo: actual checkpoint/processor hash, model-config hash equality, `s`, device, camera, max_steps, đầy đủ simulator/controller contract.
- Validator đọc `getattr(policy, 'state_dim', None)`, trong khi SmolVLA adapter báo state dim qua `validate_interface()`/observation spec, không có attribute đó.
- `hashlib.sha256` được gọi trước `import hashlib` bên trong cùng function; exception bị nuốt. Artifact quan sát được có `git_diff_sha256=null`.
- Seed là `base_seed + task_index*100 + init_id`: đổi preset/thứ tự task có thể đổi seed của cùng task/state.
- Comparator chỉ lấy giao tập episode; không reject coverage thiếu hoặc condition trộn. Seed thiếu vẫn có thể được báo matched. Chưa so đầy đủ runtime/provenance/horizon/checkpoint.

Cần lock phiên bản thí nghiệm mới, tách baseline contract khỏi experiment code revision; bắt buộc seed/identity/config và coverage đầy đủ trước khi thống kê. Run directory phải riêng cho từng protocol/writer/condition, không ghi đè episode cũ.

### 5. Chưa chứng minh OFF/shadow invariance với policy thật

Unit tests xác nhận prompt và action của mock policy giống nhau. Artifact thật cùng seed 442 hiện có:

- `smoke_test_seed442_off` vs `smoke_test_seed442_shadow`: **81 vs 76** bước; max absolute difference ở action đầu **0.1241513**.
- `test_debug_off` vs `test_debug_shadow_fresh`: **78 vs 77** bước; action đầu cũng khác.

Chưa xác định khác biệt bắt nguồn từ RNG/reset/runtime hay memory path. Cần probe mới với cùng runtime, observation/noise và protocol đã khóa, rồi kiểm tra input/action. Các báo cáo parity Windows–WSL không thay thế được gate này. C4 hiện chỉ có hai cặp raw-policy intervention, không phải memory pilot đầy đủ.

### 6. Chất lượng retrieval, tuổi fact và báo cáo cần được đo

TTL mặc định 200 bước có thể làm mất trạng thái đã hoàn thành ở task dài nếu không được xác nhận lại. Trong pilot oracle, bốn task chỉ có context ở 5/20 replans, một task 9/20. Chưa có metric chính thức về fact accuracy, retrieval precision/recall và thông tin còn sống sau tokenizer. Không tự tăng TTL toàn bộ; cần phân biệt fact đang được xác nhận với sự kiện lịch sử bền vững và bằng chứng đã mất hiệu lực.

AGENTS.md/SRS vẫn tập trung V1, execution plan nói “Ready to execute”, còn ADR-0013 mô tả acceptance rộng hơn bằng chứng hiện tại. Nên đồng bộ tài liệu theo trạng thái thực tế: P1–P5 có code, P0 chưa enforce đầy đủ, P6 có smoke/partial artifacts, P7 chưa đạt; P8–P10 chưa triển khai.

## Trình tự kiểm nghiệm đề xuất

1. Sửa token budget, writer lifecycle và entity grounding; thêm regression checks cho các lỗi đã tái hiện.
2. Chạy writer offline trên trajectory/telemetry đã có để đo false grasp/lift/slip, coverage nhiều object và fact visibility sau tokenizer.
3. Chốt một runtime/protocol và lock mới. Giữ baseline V1 lịch sử nguyên vẹn. Nếu chọn WSL/s=10 thì OFF và tất cả treatment phải chạy lại trên cùng cấu hình đó.
4. Chạy smoke OFF/shadow/text trên cùng task/state/seed, thư mục riêng; xác nhận OFF/shadow invariant và text treatment thật sự thay đúng phần input đã định nghĩa.
5. Chạy paired pilot LIBERO-10: 10 task × states [0,1,2,3] = 40 episode/condition, kèm nhóm Object control đã chọn trước. Quy mô này là pilot, không tự bảo đảm statistical power.
6. Chỉ khi dữ liệu, input và provenance đối soát được mới mở rộng 40 task × 50 official states = 2.000 episode/condition, tổng 4.000 cho OFF + một treatment. Chọn trước seed/protocol và cách tính bất định; không tối ưu prompt trên tập đánh giá cuối.

Chưa ưu tiên spatial/combined memory trước các bước trên. Nếu text memory vẫn giảm điểm sau khi các gate kỹ thuật đạt, đó là negative result có giá trị và cần báo trung thực.
