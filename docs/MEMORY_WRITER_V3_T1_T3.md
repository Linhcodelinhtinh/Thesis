# RGB memory writer v3 — pilot T1/T3

Phạm vi của thay đổi này là writer/perception, causal prompt index và nối cùng writer vào replay/rollout. Không training/fine-tuning, không thay preprocessing/action decoding của policy, không đo success rate policy. Đây là thí nghiệm V2 tùy chỉnh, không phải điểm LIBERO chính thức.

## Thiết kế đã triển khai

`RGB hiện tại → Grounding DINO → tracker bát/đĩa gốc → FSM → snapshot có provenance → prompt ≤48 token`.

- **Nguồn quan sát:** hai camera RGB và proprioception Panda. Detector chỉ nhận RGB. Reward, segmentation/depth simulator, `detected_objects` cũ, annotation subgoal và action target tương lai không được dùng để xác nhận sự kiện.
- **Phạm vi identity:** T1/T3 có một bát và các đĩa riêng biệt. Tracker chọn bát từ detection, xác định đĩa dưới bát khi gripper mở ở đầu episode, rồi giữ anchor đó. Không tự đổi sang đĩa gần bát nhất sau khi di chuyển. Nhiều bát khả dĩ hoặc nhiều đĩa riêng biệt khớp anchor là trạng thái không rõ. Chưa phải giải pháp re-identification cho nhiều vật giống nhau hoặc đĩa xếp chồng.
- **Perception thực thi được:** Grounding DINO tiny, model ID/revision và SHA256 từng file trong `resources/manifests/memory_perception_source.json`. Truy vấn riêng `a metal bowl.` / `a white plate.`; crop agent-view được tính từ detection bát ban đầu, không từ tọa độ annotation. Mép ảnh bị cắt không được dùng làm grounding. Wrist detection là bằng chứng phụ; bản này chưa dựng geometry 3D từ hai camera.
- **Ảnh:** raw camera dùng `image[::-1]` để về quy ước refined của tác giả (lật ngang rồi quay 180°). HDF5 refined không lật thêm. Phép đổi này thuộc perception, không sửa input policy.
- **Sự kiện:** `initial_support_observed`, `departed_support`, `returned_to_support`. Không phát `observed_grasp`, `placed`, `slip` chỉ từ EEF/gripper. T1/T3 cần chuỗi lift/return và bộ đếm; writer không đọc số lần yêu cầu từ instruction.
- **Lift:** mép dưới bát dịch lên so với đĩa gốc, theo tỉ lệ kích thước bát quan sát ban đầu. Không có ngưỡng tuyệt đối `eef_z <= 0.95` hay tọa độ đĩa mặc định.
- **Return:** bát quan sát hiện tại trở lại gần footprint ban đầu, vị trí ổn định, aperture gripper đo được ≥0.055 m, đủ 3 frame xác nhận. Action “open” không thay thế aperture. Đây là **ước lượng thị giác được writer xác nhận**, không phải chứng minh contact vật lý hay predicate `On` chính thức.
- **Hysteresis:** candidate bị xóa khi có quan sát trái ngược hoặc mất grounding. Sau một return phải thấy departure mới thì mới tăng tiếp; nằm yên không cộng nhiều lần. Box đĩa lồng nhau mô tả inner disk được gộp theo profile T1/T3.
- **Mất dấu:** frame thiếu detection không trở thành “vật đã rời đĩa”. Không lấy EEF để điền vị trí bát. Lịch sử bị đánh dấu không đầy đủ; count đã xác nhận được giữ như giới hạn dưới. So sánh ổn định có thể dùng vị trí RGB đo gần nhất trong tối đa 3 frame, nhưng **frame hiện tại vẫn phải có grounding** và gap vẫn được ghi nhận.
- **Causality/audit:** mỗi detection gắn hash RGB, camera, frame, model revision; mỗi event gắn `occurred_at`, `available_at`, evidence IDs và writer version. Frame cũ/tương lai, ảnh khác hash, annotation/oracle giả làm prediction đều bị từ chối. Snapshot giữ provenance khi serialize và bất biến sâu cho metadata/event details.

## Memory đưa vào policy

Ví dụ khi quan sát đầy đủ:

```text
lift the bowl and place it back on the plate 3 times
Memory: bowl A: 2 returns to original plate. Cycle underway.
```

Khi có gap:

```text
Memory: bowl A: at least 2 returns to original plate; history incomplete.
```

Không đủ bằng chứng cho initial support thì không ghi “zero times”. Count phải được đối chiếu với toàn bộ event IDs; một event được phép không thể hợp thức hóa cả counter. Renderer mặc định loại snapshot có nguồn oracle/annotation. Token được đếm bằng tokenizer SmolVLA đã pin, bao gồm newline/special-token convention. Nếu không còn chỗ, bỏ cả fact, không cắt instruction. Memory-off giữ nguyên chuỗi instruction.

`uncertain_count` trong schema là số khoảng lịch sử chưa được kiểm chứng, **không phải** số chu kỳ bị suy đoán là đã bỏ sót.

Association không coi mọi proposal yếu là một vật thứ hai: trong profile một bát, chọn giả thuyết trội nếu chênh score ≥0.12 và tỉ lệ ≥1.5; các giả thuyết sát điểm nhau vẫn là ambiguous. Đây là score detector chưa hiệu chuẩn xác suất, không phải độ tin cậy vật lý 100%.

Khi score chưa tách được, tracker chỉ được dùng continuity của vị trí/kích thước RGB đo gần nhất để ghép identity nếu có một candidate hiện tại gần rõ rệt hơn các candidate khác và khoảng cách thời gian ≤3 frame. Tọa độ dự đoán không được ghi thành detection. Jump lớn với score thấp bị từ chối; khoảng mất dấu dài không được dùng continuity để ép chọn một vật. Evidence của phép ghép giữ cả detection tham chiếu và detection hiện tại.

## Data và rollout

- Loader đọc metadata theo từng episode bằng `ijson`, không `json.load` toàn bộ file 713 MB. Metadata trong input predicted chỉ cấp instruction/initial state để split; không đưa `exo_boxes` hoặc subgoal vào observations, không tạo oracle từ EEF.
- Thiếu camera/quaternion/state sẽ lỗi, không có ảnh zero hoặc quaternion mặc định. Observation/action refined dùng offset 0; base raw có offset +1 riêng.
- Split nhóm theo hash initial state canonical, không theo tên demo/task; cùng state T1/T3 vào cùng split. Đây là split tùy chỉnh, không phải official evaluation split.
- `LiberoMemDataProcessor` và `LiberoMemRolloutRunner` nhận cùng `visual_writer_factory`. Rollout cập nhật mỗi observation và có trace riêng; F1 cần backend + camera convention rõ ràng. Runner từ chối kinematic fixture/fallback của simulator. O1 không được tự giả định initial support.
- `prepare_libero_mem_sft.py` xuất **causal prompt index**, HDF5 pointers, detections và memory trace. Nó chưa phải multimodal training shards. Chuyển ảnh vào đúng contract policy và training nằm ngoài thay đổi này.

## Cách chạy

Môi trường perception riêng, tránh cài đè stack simulator/SmolVLA:

```powershell
python -m venv .venv_memory
.venv_memory/Scripts/python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.venv_memory/Scripts/python.exe -m pip install -r envs/memory_perception.txt
.venv_memory/Scripts/python.exe scripts/download_memory_detector.py
```

`TOKENIZER_DIR` dưới đây là thư mục local của `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Loader kiểm SHA256 tokenizer, không lấy bản latest.

```powershell
.venv_memory/Scripts/python.exe scripts/replay_libero_mem_writer.py --tasks T1 T3 --demo-ids 5 31 --tokenizer-path TOKENIZER_DIR --output experiments/audits/my_writer_replay
.venv_memory/Scripts/python.exe scripts/audit_libero_mem_writer.py --replay experiments/audits/my_writer_replay --output experiments/audits/my_writer_comparison.json
```

Chạy lại FSM/renderer từ detection đã lưu để debug mà không gọi model lần nữa:

```powershell
.venv_memory/Scripts/python.exe scripts/replay_libero_mem_writer.py --tasks T1 T3 --demo-ids 5 31 --tokenizer-path TOKENIZER_DIR --detection-cache experiments/audits/my_writer_replay --output experiments/audits/my_writer_replay_again
```

Cache phải khớp file hash, model revision và hash **từng RGB frame**. Thời gian chạy từ cache không được báo như latency end-to-end.

Tạo prompt index cho pilot (chưa phải dataset nạp trực tiếp vào trainer):

```powershell
.venv_memory/Scripts/python.exe scripts/prepare_libero_mem_sft.py --task-name KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate --max-episodes 1 --stream predicted_memory --tokenizer-path TOKENIZER_DIR --output-dir experiments/audits/t1_predicted_index
.venv_memory/Scripts/python.exe scripts/prepare_libero_mem_sft.py --task-name KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate --max-episodes 1 --stream no_memory --tokenizer-path TOKENIZER_DIR --output-dir experiments/audits/t1_no_memory_index
```

Hai nhánh dùng cùng nguồn/episode/split và action index. Không trỏ notebook base vào index này như thể nó đã là training shards. Lệnh `scripts/prepare_memory_sft.py` cũ vẫn dùng writer legacy cho base.

Online wiring dùng cùng factory, nhưng chọn quy ước ảnh của adapter:

```python
detector = ProcessVisualDetector(perception_python, model_dir, device="cuda")
factory = visual_writer_factory(detector, image_convention="libero_raw")
renderer = LiberoMemTextRenderer(tokenizer=load_prompt_tokenizer(tokenizer_dir), strict_tokenizer=True)
runner = LiberoMemRolloutRunner(env, policy, condition="F1", writer_factory=factory, renderer=renderer)
# close the worker when evaluation finishes (or use its context manager)
detector.close()
```

Đây là wiring API; không chứng nhận simulator/evaluator hiện tại đã conform với tác giả. Các gap simulator/evaluator trong audit 23/10 phải được đóng trước benchmark policy.

`ProcessVisualDetector` (`src/perception/process_detector.py`) truyền RGB qua worker `.venv_memory/Scripts/python.exe`; không cần cài Transformers của detector vào môi trường policy. Worker không nhận proprioception, annotation hoặc simulator handle. Lỗi model/process/timeout phải báo lỗi, không fallback. Một worker phục vụ tuần tự từng episode.

## Kiểm định và tiêu chí sử dụng

- Unit tests kiểm tra count/rearm, nhiễu, missing/ambiguous identity, release command giả, EEF cao, gap, frame/hash/source sai, grouping state, tokenizer, memory-off và offline/online prompt parity.
- Replay thực dùng demo 1 và sau đó demo 5 T1/T3 để phát triển. Demo 5 ban đầu nằm trong split val, nhưng đã dùng để sửa association nên không được báo là holdout của writer. Demo 31 là kiểm tra bổ sung, không dùng để chọn ngưỡng; replay của nó đã được xem trong vòng phát triển nên cũng không gọi là blind holdout. Báo count, event timing, grounding coverage, lịch sử không đầy đủ và localization; không chỉ báo phần trăm frame có prompt.
- Script audit đọc annotation **sau dự đoán**. Milestone annotation là tiến độ task tích lũy; không phải nhãn grasp/contact. Event matching một-một có tolerance ±20 frame được báo rõ, kèm timing gốc để không che latency.
- Không coi việc unit tests qua hoặc hai demo khớp là chứng minh đầy đủ. Trước SFT lớn cần audit writer trên nhiều initial states, kiểm failure trajectories/false returns, chốt precision/recall và latency chấp nhận được. Không tăng coverage bằng fact giả hoặc đưa annotation vào nhánh predicted.

Kết quả đo của lần triển khai này được lưu trong `experiments/audits/memory_writer_v3/`; thư mục audit/model là gitignored và phải được chuyển riêng nếu đổi máy. Kết luận thực nghiệm cuối cùng nằm ở phần kết quả bên dưới.

## Kết quả đã chạy ngày 26/10/2026

**96 tests passed**: 49 tests cho memory/Mem và 47 regression tests của memory base, data preparation và cấu trúc project. Đây là unit/regression tests, không phải 96 benchmark rollouts. Báo cáo: `unit_results.xml` và `base_regression.xml` trong thư mục audit.

Replay **6 demonstration thật, 1.276 frame**, tương ứng **3 initial state khác nhau** ghép cặp T1/T3. Bytes của hai HDF5 và metainfo đã được hash đối chiếu nguồn đã pin. Detector chạy từ RGB thật; annotation chỉ được mở trong script audit sau khi lưu predictions.

| Task / demo | Số return annotation / writer | Frame có grounding / tổng | Frame có memory / tổng | Độ trễ xác nhận so với milestone (frame) | Lịch sử cuối episode |
|---|---:|---:|---:|---|---|
| T1 / 1 | 1 / 1 | 130 / 162 | 159 / 162 | +10 | Không đầy đủ |
| T3 / 1 | 3 / 3 | 211 / 276 | 273 / 276 | +20, +10, +8 | Không đầy đủ |
| T1 / 5 | 1 / 1 | 139 / 142 | 139 / 142 | +8 | Không đầy đủ |
| T3 / 5 | 3 / 3 | 269 / 272 | 269 / 272 | +9, +13, +8 | Không đầy đủ |
| T1 / 31 | 1 / 1 | 147 / 147 | 144 / 147 | +9 | Đầy đủ theo quan sát writer |
| T3 / 31 | 3 / 3 | 276 / 277 | 274 / 277 | +10, +6, +8 | Không đầy đủ |

- **12/12 return khớp** bằng matching một-một với tolerance ±20 frame; không có return dư trong sáu demo này. Median delay 9 frame, range 6–20. Kết quả này chưa đo precision trên failure trajectories hoặc suy ra performance trên initial state chưa thấy.
- **1.169/1.172 detection bát được tracker chấp nhận có IoU ≥0,5** so với box annotation. Ba detection còn lại đều ở T3/demo 31. Mean IoU theo episode 0,910–0,934; P95 sai số mép dưới bát 0,65–0,94 pixel ở ảnh 256×256. Đây là kiểm tra localization, không chứng nhận contact hoặc identity vật lý.
- Grounding coverage **91,85%**. Memory coverage **98,59%** vì fact quá khứ vẫn tồn tại khi frame hiện tại mất dấu; coverage bao gồm initial-zero hoặc history-incomplete text. **Không đọc 98,59% thành event accuracy.** Năm episode không đủ lịch sử được render bằng `at least N` thay vì count chính xác.
- Prompt thực tế tối đa **31/48 token**, tính bằng tokenizer đã pin. Memory-off giữ nguyên instruction. Không tăng token budget của policy.
- End-to-end RGB writer ở các replay live mất trung bình **999–1.058 ms/frame theo episode** trên RTX 4060 Laptop; đây là nút thắt hiện tại, **chưa đạt 20 Hz**. Final FSM được replay từ detection cache sau khi sửa association, mất khoảng 1–2 ms/frame; số đó không bao gồm detector. Manifest giữ hash của cache gốc và code đã tạo prediction.
- Worker chạy trong môi trường perception riêng đã kiểm tra bằng RGB thật: **12 frame đầu/cuối của 6 episode** cho boxes/scores trùng cache với sai số cho phép `1e-5`, cùng image hash, thoát sạch. Kiểm tra này xác minh worker/provenance parity, chưa phải closed-loop policy rollout.
- Chạy thực `prepare_libero_mem_sft.py` nhánh predicted trên T1/demo 1 được **162 sample, 159 có memory**; so từng frame cho prompt/snapshot/signals khớp replay, detection khớp với sai số `1e-5`. Nhánh no-memory cùng episode giữ nguyên instruction, split và action index. Hai index này chưa phải training shards.

Artifacts cuối cùng: `final_association_dev/`, `final_association_check/`, `final_dev_comparison.json`, `final_check_comparison.json`, `worker_parity_12_frames.json`. `replay_dev1/` và `replay_validation/` chứa detector predictions gốc; kết quả FSM cũ trong hai thư mục đó không phải kết quả cuối cùng.

`verification_summary.json` tổng hợp checks và hashes; `verify_final_artifacts.py` kiểm parity giữa replay và index. Manifest replay lưu hash cả thư mục training nên có một thay đổi ngoài writer sau replay ở `src/training/memory_dataset.py`; báo cáo ghi riêng hash cũ/mới, không sửa manifest lịch sử hoặc quy thay đổi này cho writer. File đó không được import bởi đường replay RGB.

## Những gate còn lại trước khi mở rộng

1. Cố định cấu hình hiện tại, kiểm writer trên initial state chưa dùng trong phát triển và trên failure trajectories: nhấc hụt, giữ bát trên đĩa nhưng chưa thả, trả nhầm đĩa, che khuất dài, rung tại ranh giới, bỏ sót và làm dư chu kỳ. Tách false-return/false-departure, count error, delay và unknown rate; không chỉ kiểm count cuối.
2. Giảm latency perception rồi đo lại event accuracy. Giảm tần suất detector có thể bỏ lỡ chu kỳ; không đổi sampling âm thầm hoặc coi frame chưa đo là frame có grounding.
3. Đóng các gap simulator/evaluator đã nêu trong audit 23/10 trước closed-loop benchmark. Xác minh ảnh online, reset và memory trace trên cùng runtime thật; worker parity không thay thế bước này.
4. Sau các gate writer/runtime mới đánh giá policy và retention bằng đối chứng no-memory cùng dữ liệu/protocol. Không dùng pilot sáu demo này để tuyên bố giữ performance LIBERO base hoặc ready cho SFT lớn.

## Giữ LIBERO base

Đường memory-off và policy preprocessing/action decoding của base không đổi. Writer proprioception cũ được giữ để tái lập thí nghiệm cũ và được ghi rõ `observation_legacy_v1` trong converter base; nó **không được nâng thành writer v3 bằng cách đổi ngưỡng z**. V3 hiện chỉ dành cho T1/T3; chưa giải quyết task sách/bát drawer hay toàn bộ LIBERO-Mem.

Giữ performance sau fine-tuning vẫn cần đối chứng matched no-memory, dữ liệu base/replay và đánh giá retention trên base. Thay đổi writer không tự chứng minh retention, success rate hoặc chất lượng policy sau adaptation.
