# Khảo sát benchmark `full-clean` — 14/10/2026

**Kết luận:** 92/200 episode thành công (46%) là con số khớp với raw logs. Tuy nhiên, có một lỗi binding/render đã tái hiện được trong WSL, cộng với nhiều lỗi diagnostics/provenance. Nên xử lý các vấn đề này trước khi kết luận giới hạn năng lực SmolVLA hoặc đưa memory vào nghiên cứu. Chưa có pilot sau sửa lỗi, nên chưa thể hứa mức tăng success rate.

**Phạm vi thực hiện:** đọc toàn bộ 200 `episode.json`, 200 `model_output.jsonl`, 200 `trajectory.npz`; kiểm tra 122.460 bước và 12.292 policy calls; kiểm tra code, checkpoint/config/hash, audit Windows–WSL cũ, frame trích từ 6 video; thực hiện probe nhỏ của MuJoCo binding trên hai environment hiện có. Không chạy thêm policy rollout, không sửa dependency, controller, adapter, checkpoint hoặc raw benchmark.

Nguồn số liệu gốc: [full-clean report](D:/Thesis_26/experiments/results/full-clean/BENCHMARK_REPORT.md). Số liệu tính lại: [audit_metrics.json](D:/Thesis_26/experiments/audits/full_clean_20261014/audit_metrics.json), [episode_audit.json](D:/Thesis_26/experiments/audits/full_clean_20261014/episode_audit.json). Báo cáo này là audit của **LIBERO-DERIVED / NON-COMPARABLE_OFFICIAL_PAPER**, không phải công bố official LIBERO score.

## 1. Kết quả thực tế và giới hạn thống kê

| Suite | Thành công | SR | Wilson 95% CI, mô tả theo episode | Nhận xét |
|---|---:|---:|---:|---|
| Spatial | 32/50 | 64% | 50,1–75,9% | Có năng lực bowl manipulation, chưa ổn định |
| Object | 13/50 | 26% | 15,9–39,6% | Nút thắt lớn ở atomic manipulation |
| Goal | 36/50 | 72% | 58,3–82,5% | Mạnh nhất, nhiều task 5/5 |
| LIBERO-10 / Long | 11/50 | 22% | 12,8–35,2% | Kém ở nhiều task nhiều vật thể, nhưng có task 5/5 và 4/5 |
| Tổng | 92/200 | 46% | 39,2–52,9% | 40 task × states 0–4 |

![Success overview](D:/Thesis_26/experiments/audits/full_clean_20261014/success_overview.png)

Các khoảng trên coi episode như các thử nghiệm Bernoulli để mô tả; chưa mô hình hóa cụm task/state. Với 5 episode/task, 0/5 vẫn có cận trên Wilson khoảng 43,4%, còn 5/5 có cận dưới 56,6%. Không thể coi 0/5 là task bất khả thi hoặc 5/5 là độ tin cậy 100% khi triển khai. Đây là full **task coverage**, vẫn là pilot về initial-state coverage: 5/50 state chính thức mỗi task và một policy seed cho mỗi state.

11 task đạt 0/5: Object 0, 3, 5, 6; Goal 6; Long 0, 1, 4, 5, 8, 9. Riêng Object + Long đóng góp 76/108 thất bại (70,4%). Vì Object cũng rất thấp, không có cơ sở gán phần lớn vấn đề cho việc quên tiến độ giữa các subtask.

Toàn bộ 108 failure có `termination_reason=MAX_STEPS` và 1.000 bước. Không có POLICY_ERROR, SIMULATOR_ERROR hay action NaN/Inf trong artifact đã lưu. Nhãn `TIMEOUT=9` trong bảng failure phase không có nghĩa chỉ 9 episode hết thời gian: đó là một category phụ thuộc thứ tự phân loại; thực tế cả 108 đều hết budget.

## 2. P0 — Lỗi render/binding cụ thể đã tái hiện

**Phát hiện mạnh nhất của audit.** Trong snapshot WSL cũ và các video full-clean đã kiểm tra, robot có cánh tay xanh lá, gripper xanh dương, mount/bề mặt một số cảnh có màu vàng xanh và vùng chồng bề mặt lốm đốm. Snapshot Windows dùng robot trắng/xám. Đây không chỉ là sai khác pose nhỏ.

![Raw snapshot comparison](D:/Thesis_26/experiments/audits/full_clean_20261014/prior_snapshot_comparison.png)

Ảnh trên là raw arrays từ C2, cùng cách hiển thị, chưa áp dụng rotation cho policy. Nhãn Windows/WSL theo nguồn snapshot C2; ảnh này không phải input tensor đã lưu trực tiếp từ full-clean. Các frame full-clean xác nhận dấu hiệu màu tương tự: [Object 0](D:/Thesis_26/experiments/audits/full_clean_20261014/libero_object_task0_init_0_frames.png), [Spatial 0](D:/Thesis_26/experiments/audits/full_clean_20261014/libero_spatial_task0_init_3_frames.png), [Long 4](D:/Thesis_26/experiments/audits/full_clean_20261014/libero_10_task4_init_0_frames.png).

Màu này khớp với định nghĩa collision geometry trong Robosuite 1.4.0: robot `[0,0.5,0,1]`, gripper `[0,0,0.5,1]`, mount/environment `[0.5,0.5,0,1]`. Mặc định environment phải tắt collision rendering và bật visual rendering. [Nguồn màu chính thức](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/utils/mjcf_utils.py), [nguồn thiết lập geomgroup](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/environments/base.py).

Probe không tạo simulator, chỉ tạo `mujoco.MjvOption()`, tái hiện đúng hai phép gán của Robosuite:

```python
opt.geomgroup[0] = 0  # ẩn collision geometry
opt.geomgroup[1] = 1  # hiện visual geometry
```

| Quan sát | `.venv_smolvla` hiện tại | `.venv_libero` hiện tại |
|---|---|---|
| MuJoCo / NumPy | 2.3.7 / 2.2.6 | 3.2.3 / 1.22.4 |
| Stride của array uint8 shape (6,) | **(0,)** | (1,) |
| Raw bytes ban đầu | `[1,1,1,0,0,0]` | `[1,1,1,0,0,0]` |
| Sau `geomgroup[0]=0` | `[0,1,1,0,0,0]` | `[0,1,1,0,0,0]` |
| Sau `geomgroup[1]=1` | **`[1,1,1,0,0,0]`** | `[0,1,1,0,0,0]` |
| Tắt collision có hiệu lực? | **Không** | Có |

Stride 0 khiến các index trỏ vào cùng byte đầu. Lệnh bật visual làm collision bật lại. Còn `.tolist()` cũng đọc sai thành các giá trị lặp, vì vậy chỉ log danh sách geomgroup chưa đủ phát hiện lỗi. Probe đọc cả 6 byte liên tiếp để kiểm chứng. [Kết quả WSL policy env](D:/Thesis_26/experiments/audits/full_clean_20261014/mujoco_array_probe.json), [kết quả environment đối chiếu](D:/Thesis_26/experiments/audits/full_clean_20261014/mujoco_array_probe_reference.json).

**Mức chắc chắn:** lỗi array view trong runtime hiện tại đã được tái hiện; code upstream và dấu hiệu ảnh liên kết rất mạnh lỗi này với collision overlay trong full-clean. Chưa đo riêng phần giảm SR do lỗi này. Cặp đối chiếu thay cả MuJoCo/Python/NumPy nên chưa phải thí nghiệm cô lập riêng NumPy. Run lịch sử thiếu NumPy/binary hashes theo episode, vì vậy không thể chứng nhận ngược mọi byte của runtime cũ.

**Cải thiện cụ thể:** tạo environment candidate cô lập; trước tiên giữ MuJoCo 2.3.7, Robosuite 1.4.0, model/runtime policy cố định và kiểm tra một NumPy 1.x tương thích, chẳng hạn 1.26.4 cho Python 3.10, sau khi resolver xác minh dependency. Đây là ứng viên cần test, không phải bản sửa đã xác nhận. NumPy 2 support ở tầng pybind11 là thay đổi có phiên bản riêng; không thể suy rằng import được thì bindings hoạt động đúng. [pybind11 2.12 changelog](https://pybind11.readthedocs.io/en/stable/changelog.html#version-2-12-0-march-27-2024).

**Ràng buộc dependency đã kiểm tra:** environment hiện có `opencv-python==5.0.0.93`; file METADATA của package yêu cầu `numpy>=2` trên Python ≥3.9. Vì vậy chỉ hạ NumPy sẽ tạo environment xung đột. Candidate NumPy 1.x phải đồng thời chọn/pin phiên bản OpenCV chấp nhận NumPy 1.x, kiểm tra toàn bộ dependency bằng resolver và `pip check`, rồi xác minh image preprocessing parity. Nếu không giữ được policy runtime hợp lệ, tách simulator vào environment riêng qua IPC và giữ nguyên preprocessing/action contract. Không coi một lệnh downgrade NumPy là bản sửa hoàn chỉnh.

Điều kiện đạt trước khi rollout:

1. Array view có stride hợp lệ, hai phép gán cho raw flags đúng `[0,1,...]`.
2. Lưu ảnh raw cả hai camera, ảnh sau environment processor và image tensor vào policy. Robot/cảnh phải dùng đúng visual geometry, không có collision overlay; đối chiếu official assets/demo.
3. Ghi loaded controller, XML/hash, camera extrinsics, actual GL vendor/renderer/version. Giữ nguyên physics, gripper, 20 Hz và protocol đã chọn.
4. Pilot cặp cùng task/state/seed/horizon/wait; đo lại SR, không chỉnh gamma/màu ảnh hoặc xóa collision meshes để che biểu hiện lỗi.

Không dùng ctypes ghi vào bộ nhớ hoặc đổi asset như một workaround benchmark. ctypes trong probe này chỉ đọc bộ nhớ của array đã biết kích thước để xác minh binding.

## 3. P0 — Diagnostics đang làm sai diễn giải thất bại

Các lỗi dưới đây không nằm trên đường quyết định action khi `memory_condition=off`, nên **không trực tiếp làm SR giảm**. Chúng làm sai lựa chọn hướng cải thiện.

**3.1 Predicate false bị thay bằng khoảng cách.** Trong [diagnostics.py:651](D:/Thesis_26/src/evaluation/diagnostics.py:651), `_update_subtasks()` gọi predicate thật; nếu kết quả false, code vẫn đi vào nhánh khoảng cách ≤8 cm. Comment nói chỉ fallback khi không có eval environment, nhưng điều kiện thực tế là `if not achieved`. Khoảng cách tâm vật thể–container không tương đương `On`, `In`, `Close` hoặc `Turnon`.

Hậu quả quan sát: **10 episode Spatial thất bại nhưng tất cả milestone được ghi achieved**. Rollout kiểm tra official success trước diagnostics trên cùng bước và dừng ngay khi success. Với task một goal, việc milestone đạt trong khi success luôn false đặc biệt chỉ ra discrepancy ở diagnostics. Không cộng 10 episode này vào SR.

**Đề xuất:** official predicate trả false phải được giữ false; lỗi/thiếu predicate thành `UNKNOWN/diagnostic_error`. Lưu riêng `predicate_current`, `predicate_ever`, `distance_heuristic`, `evidence_source`; không hợp nhất heuristic vào milestone benchmark.

**3.2 Entity resolution sai ở mug.** `extract_target_and_goal()` dò chuỗi `white_mug` / `yellow_and_white_mug`, trong khi official BDDL dùng `porcelain_mug_1` / `white_yellow_mug_1`. **10/10 episode Long task 4 và 9 không có object position trong toàn bộ telemetry**, nhưng bị xếp REACH. Video Long 4 init0 cho thấy robot đưa mug tới vùng plate dù diagnostics vẫn ghi grasp/lift 0. [Resolver](D:/Thesis_26/src/evaluation/diagnostics.py:109), [BDDL Long 4](D:/Thesis_26/external/LIBERO/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl).

Goal 0/7 cũng thiếu target object trong 10 episode, nhưng đây là drawer/stove articulation nên phải dùng joint/fixture metrics hoặc N/A. Không dùng pick-and-place grasp rate cho tất cả task.

**Đề xuất:** resolve object/fixture bằng official parsed BDDL và exact body/geom IDs; theo dõi mỗi object riêng. Log resolved IDs ngay đầu episode, fail diagnostics nếu manipulation target bắt buộc không tồn tại. Không sửa tên instruction đưa vào policy.

**3.3 “Transition failure” chưa chứng minh lỗi chuyển giai đoạn.** Milestone được latch vĩnh viễn khi từng đạt; không theo dõi mất trạng thái. Predicate `close(microwave_1)` hoặc stove có thể được tính ngay bước 0. Nhánh khoảng cách cũng có thể so fixture với chính nó và đánh dấu đạt giả. [Aggregator:168](D:/Thesis_26/src/evaluation/benchmark_aggregator.py:168) gọi mọi failure có ít nhất một milestone là transition failure; không yêu cầu subtask trước thực sự do policy hoàn thành, đúng thứ tự hoặc còn đúng ở cuối.

Long 8/9 đang có 5/5 transition failures mỗi task dù stove/door milestone đã “đạt” ngay bước 0. Không dùng dữ liệu này để tuyên bố memory sẽ giải quyết vấn đề. Cần baseline predicate trước action, event false→true, regression true→false và conjunction tại cùng timestep. BDDL conjunction cũng không tự quy định thứ tự thao tác.

**3.4 Reach/grasp metrics không tạo thành chuỗi nhất quán.** Có **44 episode ever_grasped=true nhưng ever_reached=false**. Spatial ghi reach 24/50 nhưng grasp 44/50. Ngưỡng EEF–body center 5 cm không phải fingertip reach cho mọi hình học. Task thành công còn bị tự động gán `ever_placed=true`, kể cả mở drawer/bật stove. Nên đổi thành metrics theo nhóm task, đo fingertip/contact và ghi rõ heuristic; không tăng ngưỡng để làm đẹp báo cáo.

## 4. P1 — Checkpoint và execution protocol phải được phân biệt

Checkpoint local hiện tại khớp manifest:

- Model revision: `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`.
- `model.safetensors` SHA-256: `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`.
- `config.json` SHA-256: `5c9f3ba9f5f37ea7024c9501b0b20b1941f232989c2167853cb46d9071a70dd7`.
- Source benchmark checkout hiện tại đúng `8f1084e3132a39270c3a13ebe37270a43ece2a01`, không có tracked diff.

Đây là kiểm tra file hiện tại, không thay thế manifest checkpoint/binary được ghi tại thời điểm từng episode chạy.

| Thuộc tính | Checkpoint đang chạy | `HuggingFaceVLA/smolvla_libero` |
|---|---|---|
| Revision đối chiếu | `31d453f...` | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| `n_action_steps` trong config | 50 | 1 |
| `num_vlm_layers` | 16 | 0 |
| `expert_width_multiplier` | 0.75 | 0.5 |
| `pad_language_to` | max_length | longest |
| Image keys | camera1/camera2/camera3 khai báo | image/image2 |
| State shape khai báo | 6, normalizer thực tế 8 | 8 |

Hai repo không phải alias có thể thay tùy ý. Khác config chưa tự chứng minh checkpoint nào mạnh hơn. Không gán paper score cho checkpoint đang chạy chỉ vì cùng tên SmolVLA. [Config hiện dùng](https://huggingface.co/lerobot/smolvla_libero/blob/31d453f7edd78c839a8bbc39744a292686daf0de/config.json), [config checkpoint đối chiếu](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/config.json).

Paper báo 87,3% cho một cấu hình SmolVLA 0.45B. Cùng paper, ablation action execution steps cho trung bình 82,8% ở 10 bước và 51,8% ở 50 bước. Điều này hỗ trợ việc kiểm tra replan cadence, không dự đoán SR của stack hiện tại. [SmolVLA, Tables 2 và 13](https://arxiv.org/html/2506.01844v1).

Run full-clean thực tế **H=50, s=10**, wait=10, 20 Hz theo code: mỗi replan dự đoán 50 action rồi chạy 10, tương đương 0,5 giây simulated time giữa observations. `execution_horizon_s` dùng chữ s như ký hiệu số bước, không phải seconds. `num_steps=10` trong model config lại là số bước flow matching; đây là ba tham số khác nhau.

`selected_baseline.yaml` vẫn ghi execution horizon 50, `LOCKED_AND_FROZEN`, metadata host Python 3.12; run lại dùng Python 3.10, s=10. Đây là **protocol mới so với baseline frozen cũ**, chưa phải bằng chứng s=10 sai. Không đổi về 50 chỉ để khớp YAML. Cần một config/run ID chính xác cho full-clean và baseline sửa lỗi sau này.

**Ablation đề xuất sau P0:** s=10 làm control, s=5 và s=1 làm candidate; giữ H=50, flow steps=10, weights, images và normalization. Chỉ kiểm tra s=50 như đối chứng frozen cũ nếu cần. Với synchronous simulator, s=1 tốn inference khoảng 10 lần trên cùng số control steps; phải báo cả SR và chi phí. Không dùng temporal smoothing hoặc trộn chunks như sửa lỗi mặc định, vì làm thay đổi action decoding.

## 5. P1 — Kiểm tra lại các kết luận Windows–WSL cũ

| Bằng chứng | Điều được hỗ trợ | Điều chưa được hỗ trợ |
|---|---|---|
| C1: 2 snapshots × 2 noise tensors, processor outputs khớp | Trong panel nhỏ này, model output khác ít hơn các thay đổi observation ở C2 | “PROVEN IDENTICAL”; max action difference tới khoảng 0,00742, không bằng 0; chưa bao phủ 40 task |
| C2: full L2 6,5529; image-only 6,5831; proprio-only 0,1085 | Với snapshot/noise này, image bundle ảnh hưởng lớn hơn proprio bundle | Do GPU rasterization, do tilt 2°, hoặc 78,4% phương sai được giải thích |
| C3: 13/15 outcome agreement | Nhiều action trajectories chuyển được giữa hai stack | Simulator tương đương hoàn toàn; hai trajectory success nguồn thất bại replay; global EEF error tới 110,37 mm |
| C4: wait10 0/2, wait0 0/2 | Chưa thấy success gain khi bỏ wait trên 2 cặp | “Exact causal bridge”, bỏ settling phục hồi training alignment |

Nguồn: [C1](D:/Thesis_26/experiments/audits/smolvla_parity_20261013/reports/OFFLINE_PARITY_AUDIT_REPORT.md), [C2](D:/Thesis_26/experiments/audits/smolvla_parity_20261013/reports/PHASE_C2_SENSITIVITY_REPORT.md), [C3](D:/Thesis_26/experiments/audits/smolvla_parity_20261013/reports/PHASE_C3_REPLAY_REPORT.md), [C4 raw](D:/Thesis_26/experiments/audits/smolvla_parity_20261013/outputs/c4_paired_rollouts.json).

C2 ghi PSNR thực tế 14,51/12,61 dB, nên dòng template nói “PSNR >30 dB” không mô tả kết quả đo. Tỷ số các L2 norm không phải causal variance decomposition; không cộng các tỷ lệ đó để phân bổ nguyên nhân. Phát hiện collision-overlay/binding mới là một lời giải thích cụ thể hơn cần kiểm tra trước giả thuyết camera tilt.

Giữ wait=10 cho control hiện tại. LeRobot v0.4.4 thực hiện reset → official init state → 10 dummy actions. Bỏ wait là một experimental protocol khác, chưa có bằng chứng giúp run này. [LeRobot reset code](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/envs/libero.py).

## 6. P1 — Các lỗ hổng tái lập và nhãn benchmark

**Dirty code không có diff.** Cả 200 episode ghi git `dc959c4203fd903c7c1c4f5677c73273e73f92a4`, `git_clean_working_tree=false`, `git_diff_sha256=null`, `baseline_lock=null`. Tên thư mục `full-clean` không chứng minh clean Git/runtime.

Trong [run_benchmark.py:491](D:/Thesis_26/scripts/run_benchmark.py:491), `hashlib` được dùng trước `import hashlib` trong cùng hàm ở dòng 497; import cục bộ làm biến chưa được gán ở đoạn trước, exception bị nuốt. Điều này giải thích diff hash null. Ngay cả khi hash được ghi, hash không kèm diff bytes cũng chưa đủ tái dựng code; untracked files cũng phải được kiểm kê.

Một khác biệt quan trọng: HEAD hiện tại thêm `self._env.reset()` trước `set_init_state`; commit được log trong run chưa có dòng đó. Dirty snapshot bị thiếu khiến không xác định chắc file chạy lúc ấy có bản sửa hay không. Không kết luận run chắc chắn thiếu reset; lưu source snapshot ở run mới và kiểm tra episode independence bằng cùng state/seed trước và sau một episode khác. [Reset hiện tại](D:/Thesis_26/src/simulator/libero_env.py:253).

**Version detection:** BDDL null là kết quả lấy `bddl.__version__`, không chứng minh BDDL không cài. Dùng `importlib.metadata.version` và import paths. Manifest C0 trước đó ghi bddl 1.0.1/LeRobot 0.4.4, trong khi model manifest ghi LeRobot 0.6.1. Cần per-run resolved versions, không dùng manifest chung làm bằng chứng cho run cụ thể.

**Reference environment chưa đúng tuyệt đối:** probe hiện tại thấy `.venv_libero` là Python 3.8.20/MuJoCo 3.2.3/NumPy 1.22.4; AGENTS yêu cầu Python 3.8.13. Không dùng nhãn tên environment để chứng nhận reference. Chưa chạy smoke test hoặc chứng nhận lại environment đó trong lượt audit này.

**Seeds phụ thuộc task order:** hiện tại full-clean có đủ seeds theo `42 + t_idx*100 + init_id`. Ví dụ Object0 dùng 1042–1046. Khi chạy riêng Object0 với CLI `--seed 42`, t_idx trở về 0, seed thành 42–46. Do đó “cùng base seed” không phải paired rerun. Lưu episode schedule tường minh theo suite/task/state/replicate; seed lookup không phụ thuộc preset, resume hoặc batch order. Gán `PYTHONHASHSEED` sau khi interpreter khởi động cũng không reset hash seed của process hiện tại. [Seeding](D:/Thesis_26/scripts/run_benchmark.py:635).

**Protocol comparison:** repository dùng budget 1.000; LeRobot v0.4.4 defaults là 280 Spatial/Object, 300 Goal, 520 Long. Cắt hậu kiểm success times của run này theo các ngưỡng đó cho **85/200 =42,5%**, vì 7 success đến muộn. Đây chỉ là sensitivity calculation trên logs, **không phải** một official-compatible rerun. Không tự thay max_steps của V1 để nâng hoặc làm đẹp score. [Nguồn giới hạn episode](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/envs/libero.py).

**Đầu ra cần thêm cho run mới:** command/argv, resolved config, full git diff + source files, exact weights/processors/tokenizer/backbone revisions/hashes, dependency lock + binary/import paths, actual GL strings, controller loaded values, camera pose/flags, official BDDL/init-state hashes, per-episode seed, max_steps, source file checksums. Runtime flags nên được assert từ object đã khởi tạo, không chỉ ghi giá trị dự định truyền vào.

## 7. Điểm hiện tại chưa phải nghi phạm chính

- **Gripper polarity:** 200/200 episode có logged executed action bằng unnormalized action; không có inversion trong telemetry. PandaGripper chính thức dùng −1 mở, +1 đóng. Không quay lại ADR-0009 đảo dấu cũ. [PandaGripper 1.4.0](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/grippers/panda_gripper.py).
- **State 9D trong trajectory:** đây là log post-action `pos3 + quat4 + gripper2`. Input adapter chuyển thành `pos3 + axis-angle3 + gripper2 =8D`. Không phải chứng cứ policy nhận state9 sai. Normalizer đã kiểm tra có 8 phần tử state và 7 phần tử action; không cắt xuống6 chỉ theo metadata cũ.
- **Camera3 trong config:** official SmolVLA code chỉ dùng image keys có mặt; `empty_cameras=0` không thêm camera thiếu. Hai camera là phù hợp pipeline đang khai báo. Cần strict check hai camera bắt buộc tại adapter, không thêm camera giả. [Official prepare_images](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py).
- **Image orientation:** adapter dùng official `LiberoProcessorStep` flip cả H,W và official quat→axis-angle. Không thêm flip nữa để thử nâng SR. Snapshot/video hiển thị có thể khác orientation với tensor policy. [Official environment processor](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/processor/env_processor.py).
- **Action amplitude:** chỉ 48 giá trị arm vượt |1| trên 122.460 steps ×6 chiều; gripper có 57.544 bước vượt nhẹ |1|, range khoảng −1,092…1,128. Controller có clipping, Panda gripper dùng sign. Đây chưa phải bằng chứng controller divergence; thêm log action sau controller scaling nếu cần. Không nhân gain hoặc clip khác nguồn chính thức để tối ưu điểm.
- **Language truncation:** probe offline với tokenizer cache Windows snapshot `7b375e1b73b11138ff12fe22c8f2822d8fe03467` cho tối đa22 tokens trên 40 instruction, dưới48. Tuy nhiên Transformers host phát cảnh báo pad_token_id ngoài vocabulary; phép thử này chưa xác minh tokenizer thực tế WSL. Lần chạy mới cần log token IDs, tokenizer files/revision và attention mask; không coi kiểm tra host này là chứng nhận tokenization.
- **256px và control rate:** nguồn checkpoint có policy image shape256 nhưng `train_config.env.observation_height/width=360`, `env.fps=30`; các trường này không tự chứng minh training camera render hoặc physics rate. Model còn resize_with_padding lên512 bên trong. Không đổi controller sang30Hz theo trường fps. Xác minh data export/eval recipe, giữ20Hz theo repository. [Train config pinned](https://huggingface.co/lerobot/smolvla_libero/blob/31d453f7edd78c839a8bbc39744a292686daf0de/train_config.json).

Có một nhánh latent cần harden: [adapter.py](D:/Thesis_26/src/models/smolvla/adapter.py:223) tự tạo gripper_qpos `[0.02,-0.02]` nếu thiếu. Không có bằng chứng nhánh này đã chạy trong full-clean, nhưng nên thay bằng lỗi rõ ràng ở run tương lai. Tương tự, thiếu một trong hai camera có thể vẫn được upstream model chấp nhận. Shape contracts phải kiểm tra presence, order, dtype/range và tính hữu hạn.

## 8. P2 — Tốc độ đang tốn ở đâu

| Thành phần đo từ raw logs | Kết quả |
|---|---:|
| Mean inference / call | 674,21 ms |
| P95 inference / call | 910,82 ms |
| Số calls | 12.292 |
| Tổng inference | 2,302 giờ |
| Mean env.step, gộp theo tất cả steps | **155,83 ms** |
| Tổng env.step | 5,301 giờ |
| Tổng hai phần đã đo | 7,603 giờ |
| Simulated control time | 1,701 giờ |
| Bước chạy trong failure episodes | 108.000/122.460 =88,2% |

Trong phần đã đo, simulation chiếm khoảng69,7%, inference30,3%. Tốc độ khoảng4,47 simulated control steps/giây wall-clock; còn diagnostics, render video, encode, I/O và model loading chưa bao gồm đầy đủ. Không lấy chênh lệch timestamp bắt đầu/kết thúc làm compute time vì có pause giữa run.

Hai nhãn hiện tại trong aggregator gây hiểu nhầm: 692,04 ms “per episode” thực chất là mean của các episode-level **mean-per-call**, không phải tổng inference/episode; 170,16 ms “simulation per step” là mean của episode means, không phải trung bình theo122.460 steps. Hai kiểu weighting đều có thể báo, nhưng phải đặt tên đúng. [Aggregator](D:/Thesis_26/src/evaluation/benchmark_aggregator.py:203).

**Tối ưu ít tác động protocol trước:** profile physics, observation render từng camera, extra video render, diagnostics, CPU↔GPU copies, tokenizer và model forward riêng. Khi record_video=true, rollout hiện gọi `env.render()` mỗi bước dù next_obs đã có agentview; cả episode cuối cùng không lưu video vẫn có thể bị capture frame. Có thể dùng lại frame observation và xử lý orientation riêng cho video, sau đó xác nhận policy inputs/actions không đổi. Cache tokenization của instruction cố định và body/geom lookup; hạn chế repeated tensor→CPU sync dùng cho telemetry. Dùng inference_mode chỉ sau kiểm tra numerical parity.

**Tối ưu cần kiểm chứng:** actual GPU rendering phải đọc GL_VENDOR/GL_RENDERER, CUDA available không chứng minh renderer dùng GPU. C0 chỉ có biến môi trường GLX/D3D12, chưa có actual strings. So backend EGL/GLX/OSMesa trong môi trường riêng, cùng model/state và render flags đã đúng, rồi kiểm tra pixel/action parity. Có thể tách simulator và policy qua IPC để giữ reference stack và tận dụng CUDA, nhưng phải bảo toàn reset/action timing.

AMP/BF16, torch.compile, giảm flow steps, quantization, asynchronous execution hoặc nhiều env song song có thể giúp throughput nhưng cũng thay đổi số học/RNG/timing. Đưa từng lựa chọn thành candidate riêng, kiểm tra fixed-observation/fixed-noise và paired rollouts; không coi là tối ưu vô hại. Trong loop synchronous hiện tại, simulator không tự tiến theo wall-clock lúc đợi inference, nên latency lớn tự nó chưa giải thích SR thấp.

## 9. Kế hoạch cải thiện có kiểm chứng

| Thứ tự | Việc cần làm | Điều kiện hoàn thành | Chi phí / mục tiêu |
|---|---|---|---|
| 1 | Đóng băng full-clean và source/dependency hiện tại; gắn ghi chú rendering issue | Raw artifact/hash bất biến; không gọi frozen clean baseline mới | Bảo toàn evidence |
| 2 | Tạo candidate env, sửa tính tương thích binding và kiểm render flags | Array probe pass; ảnh hai camera không collision overlay; official assets/controller giữ đúng | Ưu tiên cao nhất cho SR |
| 3 | Sửa diagnostics/provenance trên nhánh audit | Predicate false không thành true; mug IDs resolve; states at reset tách khỏi progress; diff hash/source snapshot đầy đủ | Tăng độ tin cậy phân tích |
| 4 | Chạy pilot control/candidate renderer-runtime | Cùng checkpoint H50/s10/wait10/max1000, explicit seed schedule | Đo gain do fix trước tune |
| 5 | Ablation s=5,1 so với s=10 trên dev panel | Báo paired wins/losses, ΔSR, latency và failure evidence hợp lệ | Cân bằng reactivity/compute |
| 6 | Nếu cần, audit author checkpoint thứ hai | Manifest/interface riêng, baseline ID riêng | Không thay thế âm thầm weights frozen |
| 7 | Chạy confirmation trên state/seed chưa dùng để lựa chọn | Config khóa trước; đủ50 states/task cho kết quả chính | 40×50=2.000 episode/condition |

Pilot đề nghị cho bước4: 6 task ×5 official states ×3 policy seeds =90 episode/condition, tổng180. Panel bao gồm Object0 (soup), Object5 (tomato sauce), Spatial0 (bowl/placement), Goal8 (bowl-on-plate đối chứng mạnh), Long4 (mug + diagnostics lỗi), Long7 (multi-object đối chứng có năng lực). Nếu quá tốn, chạy bước sàng lọc30 episode/condition trước rồi chỉ mở rộng candidate có evidence phù hợp; đó là smoke/pilot, không phải kết luận powered.

Chốt schedule tường minh trước khi chạy. Thay đúng một nhóm yếu tố; ghi rõ nếu sửa dependency ảnh hưởng nhiều hơn renderer. Với mỗi cặp báo both-success, control-only, candidate-only, both-fail; paired ΔSR và khoảng bất định phù hợp, bootstrap theo task/state cho repeated seeds. McNemar có thể dùng cho cặp nhị phân khi các giả định phù hợp, không coi mọi seed lặp là task độc lập. Không lựa chọn phiên bản trên cả200 episode rồi dùng chính các episode ấy làm bằng chứng generalization.

Về năng lực thao tác, diagnostics Object có coverage tốt hơn Long: reach38/50, grasp24/50, lift21/50, success13/50. Soup có reach/grasp failures; tomato sauce có3/5 episode đã lift nhưng placement fail; butter có2/5 lift nhưng placement fail. Sau khi sửa render, xem lại các nhóm này để chọn can thiệp. Chưa có một nguyên nhân chung “không biết grasp” giải thích hết.

Nếu còn thấp sau protocol/runtime parity, mới khảo sát dataset coverage, scene/object appearance và chất lượng checkpoint. Fine-tuning chỉ là thí nghiệm mới trong training environment, checkpoint/hash riêng, không còn frozen raw-policy baseline. V1 chưa triển khai memory; Long thấp cùng diagnostics hiện tại không đủ biện minh memory implementation.

## 10. Artifact kiểm tra và giới hạn còn lại

Đã đối chiếu số bước, action arrays NPZ–JSONL và boolean success JSONL–episode trên200 episode: không phát hiện mismatch. Kiểm tra giá trị action không NaN/Inf và raw→executed không inversion. Có125 video tồn tại, đúng125 đường dẫn video được claim;75 episode không claim video. Video không đủ thay thế hai camera input tensors/noise được lưu nguyên giá trị.

Sáu video đại diện đã trích frame, không phải xem toàn bộ video frame-by-frame:

| Thành công | Thất bại |
|---|---|
| [Goal8 init0](D:/Thesis_26/experiments/results/full-clean/libero_goal_task8/init_0/video.mp4) | [Object0 init0](D:/Thesis_26/experiments/results/full-clean/libero_object_task0/init_0/video.mp4) |
| [Object9 init0](D:/Thesis_26/experiments/results/full-clean/libero_object_task9/init_0/video.mp4) | [Spatial0 init3](D:/Thesis_26/experiments/results/full-clean/libero_spatial_task0/init_3/video.mp4) |
| [Long2 init0](D:/Thesis_26/experiments/results/full-clean/libero_10_task2/init_0/video.mp4) | [Long4 init0](D:/Thesis_26/experiments/results/full-clean/libero_10_task4/init_0/video.mp4) |

Probe array là kiểm thử thật trên các thư viện được cài, không chạy mock policy hoặc tạo benchmark assets. Render-only probe đầy đủ đã thử nhưng process thoát mã1 mà không trả diagnostics, nên **không có** render A/B mới hoặc GL renderer string đã xác minh. Chưa thực hiện thí nghiệm cùng MuJoCo2.3.7 chỉ đổi NumPy; chưa xác nhận candidate dependency fix; chưa đo SR sau sửa. Đây là các bước cần tiếp theo, không phải các kết quả đã đạt.

Audit source HEAD: `12890e51ff4ddf5f0792a17d6474bb9ab32f4d17`. Hardware của audit C0 cũ: RTX4060 Laptop8GB; manifest environment chung ghi Ryzen7 7840H. Full-clean không có hardware snapshot đủ để chứng nhận toàn bộ runtime tại thời điểm chạy. [Environment manifest](D:/Thesis_26/resources/manifests/environment_manifest.yaml), [WSL C0 manifest](D:/Thesis_26/experiments/audits/smolvla_parity_20261013/manifest/manifest_wsl_cuda.json).

Các scripts và kết quả audit được lưu tại [full_clean_20261014](D:/Thesis_26/experiments/audits/full_clean_20261014). `input_episode_sha256.json` khóa200 episode JSON; inventory bổ sung khóa các raw logs/NPZ/timing/diagnostics và code dùng cho audit. Không ghi đè báo cáo gốc hoặc file audit memory có sẵn của người dùng.
