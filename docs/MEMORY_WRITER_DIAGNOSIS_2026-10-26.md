# Xác minh writer memory trên task book/caddy và bowl/drawer — 26/10/2026

**Kết luận: writer hiện tại có vấn đề về cả coverage và độ trung thực của fact. Nên thiết kế lại phần nhận biết/xác nhận sự kiện, giữ lại phần lưu trữ và hạ tầng kiểm toán còn dùng được. Không nên chỉ tăng ngưỡng EEF để tạo nhiều prompt memory hơn.**

Người dùng xác nhận hai log được tạo với writer hiện tại, chưa sửa trên Kaggle. Working tree audit tại `ac4b950760d5ff8895d437bcb510060b85b2563a`; các thay đổi có sẵn ở notebook, dependency list và source-lock path được giữ nguyên. Không sửa production writer, không chạy training hoặc simulator/policy mới.

## 1. Đường code thực sự được sử dụng

`prepare_memory_sft.py → annotate_episode() → ObservationMemoryWriter → store/updater → retriever → TextMemoryRenderer(compact_mode=True)`.

Đây là **observation writer cũ**, không phải `LiberoMemEventWriter`. [memory_data.py](../src/training/memory_data.py) dòng 89, 96–97 chỉ cấp gripper joint positions và EEF position cho writer, không có ảnh, object pose hoặc detections. Vì vậy việc thêm module LIBERO-Mem trong repository chưa làm pipeline preparation này có perception mới.

`has_memory` được định nghĩa bằng `result.text != instruction` tại dòng 113. Thống kê memory frames đo **prompt cuối cùng có thêm nội dung**, không đo số lần ghi event, recall của grasp, độ chính xác fact hoặc giá trị của lịch sử đối với quyết định.

## 2. Cách xác minh dữ liệu

Workspace không có HDF5/prepared shards của hai task người dùng vừa chạy. Audit đọc **riêng action/robot-state arrays** từ HDF5 tác giả bằng HTTP range, tại `yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`; không tải toàn bộ ảnh/dataset. Từng response được kiểm HTTP 206, byte range và file size theo inventory. Các array extract được lưu cùng SHA256 và URL/revision nguồn. **Không tuyên bố đã tính lại SHA256 toàn file HDF5**, vì chỉ đọc một phần file.

Replay giữ nguyên `annotate_episode`, writer, retrieval và rendering; cast state/action float32 như loader gốc. Dùng native backend của `tokenizer.json` ở snapshot `7b375e1b73b11138ff12fe22c8f2822d8fe03467`: cấu hình GPT2, không có post-processor thêm BOS/EOS; renderer vẫn thêm newline và truyền flag special tokens như source. Không cần load VLA weights. Các prompt holding của cả hai task chỉ 22 tokens, dưới budget 48; nguyên nhân chính không phải prompt thiếu chỗ.

**Tái hiện đúng cả hai log:** book 9.420/0, bowl 12.384/237. Tổng dữ liệu đọc qua HTTP range khoảng 26,88 MB, không phải tải hai file HDF5 khoảng 2,18 GB. Source arrays được kiểm tra đối chiếu `robot_states[:,2:5]` với `obs/ee_pos`, `robot_states[:,:2]` với `obs/gripper_states`, shape và finite trước khi sử dụng.

Artifacts:

- [HTTP range extraction script](../experiments/audits/memory_writer_20261026/read_author_telemetry.py), [nguồn và extract hashes](../experiments/audits/memory_writer_20261026/author_telemetry_sources.json).
- [Replay/ablation script](../experiments/audits/memory_writer_20261026/diagnose_writer.py), [kết quả từng episode, gates và event timeline](../experiments/audits/memory_writer_20261026/writer_replay_results.json).
- [Telemetry rollout độc lập có sẵn](../experiments/audits/memory_writer_20261026/existing_rollout_evidence.json) và [script kiểm tra](../experiments/audits/memory_writer_20261026/inspect_existing_rollouts.py).
- [Unit test results](../experiments/audits/memory_writer_20261026/writer_tests.xml): 17 tests hiện có pass; điều này không chứng minh writer đúng trên task thật.
- [Geometry evidence và source hashes](../experiments/audits/memory_writer_20261026/geometry_evidence.json).

Các artifacts ở `experiments/audits` bị gitignore. Các rollout cũ giữ nguyên nhãn `NON-COMPARABLE_OFFICIAL_PAPER`; chúng chỉ là bằng chứng phụ về gate trên telemetry, không phải score mới hay chính dataset preparation.

## 3. Năm quan sát về task sách

| Quan sát người dùng | Kết quả |
|---|---|
| Mặt bàn ở z=0.85 m | Cần sửa: XML ghi **tâm** collision box z=0.84929. Sau tính rotation và half-size, mặt trên collision của asset là **0.88513 m**; nominal workspace offset là **0.867 m**. Không đồng nhất ba đại lượng này. |
| Sách cao khoảng 13.4 cm | Đúng gần đúng cho kích thước collision theo trục z **local**: 0.1344 m. Không tự động bằng chiều cao world ở mọi pose. |
| Sách đứng thẳng, đỉnh z=0.984 m | Chưa được chứng minh; phép cộng dùng 0.85 làm mặt bàn không đúng. Cần object pose/geometry đã transform để tính world top. |
| EEF khi gắp khoảng 0.98–1.02 m | Telemetry xác nhận vấn đề height: minimum của toàn bộ 50 demo là **0.985072 m**, lớn hơn 0.95. Không dùng 0.98–1.02 làm khoảng của toàn trajectory: max đạt **1.331742 m**. `robot0_eef_pos` là world position của **grip_site**, không phải cảm biến cổ tay. |
| Gate luôn false nên không tạo memory | **Đã tái hiện chính xác:** 9.420 samples, 0 memory; train 7.602/0, val 903/0, test 915/0; cả 50 demo không có event. |

Ở các observation được writer xử lý, có **6.077 frames** thỏa close command + gap range + stalled gap, nhưng **0 frame** thỏa thêm height gate. Vì vậy height gate là blocker trực tiếp trong task này, không chỉ là một giả thuyết dựa trên kích thước asset.

Ablation chẩn đoán thay riêng `0.95 → 1.10` trong một bản AST nằm trong bộ nhớ, không sửa source: memory tăng lên **6.080/9.420 frames (64,54%)**. Đây chỉ chứng minh độ nhạy của trigger; **không chứng minh 6.080 fact đúng**, và không phải đề xuất threshold production.

Nguồn geometry: [scene XML dòng 60](../external/LIBERO/libero/libero/assets/scenes/libero_study_base_style.xml), [study problem](../external/LIBERO/libero/libero/envs/problems/libero_study_tabletop_manipulation.py), [book XML dòng 9](../external/LIBERO/libero/libero/assets/turbosquid_objects/black_book/black_book.xml). External LIBERO đúng pin `8f1084e3132a39270c3a13ebe37270a43ece2a01`, working tree sạch. Ý nghĩa EEF được đối chiếu với [robosuite 1.4.0 SingleArm source](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.0/robosuite/robots/single_arm.py), nơi eef observation đọc `site_xpos` của `grip_site`.

## 4. Vì sao bowl/drawer cũng là bài khó với writer này

Replay 50 demonstration tác giả trả đúng **12.384 samples, 237 memory frames (1,91376%)**, nhưng chỉ **2/50 demo** có event/memory:

| Split | Samples | Memory frames |
|---|---:|---:|
| Train | 9.889 | 237 |
| Validation | 1.134 | 0 |
| Test | 1.361 | 0 |

Chỉ có 5 event trong toàn bộ 50 demo: 2 grasp, 1 lift, 1 slip, 1 release. Các prompt có memory gồm **99 frames “Holding black bowl.”** và **138 frames “Lost contact with black bowl.”**. `demo_23` tạo grasp tại source observation 84, slip tại 90 và giữ câu lost-contact tới observation 227; `demo_30` có 93 frames holding, sau release ở observation 179 trở về instruction gốc. Chưa có nhãn độc lập xác minh sự kiện slip là thật; không gọi 138 frames ấy là 138 positive examples độc lập hoặc sự kiện đã xác nhận đúng.

Toàn bộ 12.147 frame không có memory có reason `no_candidate_facts`; không có token-budget drop trong replay này. Validation/test hiện không chứa memory input, nên validation loss ở đó không đo khả năng policy sử dụng memory.

Height chỉ là một gate. [observation_writer.py](../src/memory/observation_writer.py) dòng 131 còn đòi `0.015 <= finger_gap <= 0.065`, với `finger_gap = abs(qpos[0])+abs(qpos[1])`.

Trong demonstration preparation, có 12.334 observation được writer xử lý, chỉ 698 qua gap range, 102 qua close+gap+stall, và 12 qua cả height. Nếu chỉ đổi height thành 1.10, memory vẫn chỉ **340/12.384 (2,74548%)**. Đây là bằng chứng rằng sửa riêng height không giải quyết task bowl; ablation vẫn không đo độ đúng của fact.

Trong **5 rollout thành công có sẵn** của bowl/drawer, cả **386 frames có bilateral contact theo diagnostic log** đều có gap **0.003635–0.012013 m**, dưới 0.015. Ví dụ `init_0`, step 85: EEF z=0.921494 m hợp lệ, contact=true, gap=0.007997 m lại bị loại. Đây là bằng chứng độc lập về false-negative risk của gap gate; không phải ground-truth event annotation cho toàn bộ demonstration vừa prepare. Các contact log chưa được replay simulator để tái xác minh.

Task này còn đòi hai outcome: bowl **ở trong** bottom drawer và drawer **đóng** ([BDDL](../external/LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl), dòng 87–91). Writer hiện chỉ phát grasp/lift/slip/release. Nó không quan sát quan hệ bowl–drawer hoặc trạng thái drawer, nên không có cơ sở để xác nhận hai outcome đó.

Sau `observed_release`, object state thành `released`. Compact renderer không render state này; prompt quay lại instruction gốc. Do đó, ngay đoạn robot cần nhớ “bowl đã được đặt trong drawer” để tiếp tục đóng drawer, lịch sử hữu ích có thể không được cung cấp. Không được sửa bằng cách đổi mọi `released` thành `placed/in_container`: release command không chứng minh placement thành công.

## 5. Tỷ lệ 1.9% có bình thường không?

Không tồn tại tỷ lệ memory frame chuẩn áp dụng cho mọi task. Event có thể ít trong khi fact đã xác nhận được giữ lâu; hoặc retrieval có thể chỉ đưa fact vào đúng thời điểm cần lịch sử. Vì vậy **1.9% tự nó không chứng minh writer sai, nhưng cũng không phải tiêu chí chấp nhận**.

Với implementation hiện tại, tỷ lệ thấp đi cùng các lỗi cơ chế đã xác minh, nên không nên coi là bình thường rồi tiếp tục train lớn. Cần tách bốn lớp:

1. **Detection:** bao nhiêu grasp/placement/cycle thật được phát hiện, bỏ sót và phát hiện nhầm?
2. **Memory state:** fact nào được giữ, sửa, hạ confidence hoặc chuyển unknown khi mất quan sát?
3. **Retrieval/rendering:** bao nhiêu frame mất memory do không có candidate, identity chưa xác định, stale, sai entity hoặc token budget?
4. **Task relevance:** tại đoạn cần phân biệt hành động theo lịch sử, memory có thông tin đúng không?

Các frame liên tiếp rất tương quan; 237 memory frames không đồng nghĩa 237 sự kiện độc lập. Nếu toàn bộ 9.420 samples đều không có memory, nhánh “memory” này không cung cấp tín hiệu học sử dụng memory. Với 1.9%, đa số sample của hai nhánh vẫn có cùng prompt; chưa đủ để kết luận tác động memory, dù SFT loss có giảm.

## 6. Những vấn đề về độ trung thực cần sửa cùng coverage

- **Identity lấy từ instruction:** nếu chỉ có một target, writer gán ngay target đó và `CONFIRMED`, confidence cố định 0.70 (dòng 146–150). Gripper có thể đang chạm drawer handle hoặc vật khác; instruction không phải bằng chứng đang cầm đúng vật.
- **Grasp từ blocked gap:** một lần stalled sau observation trước đã đủ (`blocked_counter >= 1`), không phải confirmation window nhiều bước. Proprioception không phân biệt mọi vật cản với grasp.
- **Lift từ EEF:** tăng EEF 4 cm + gap phù hợp không chứng minh object cùng chuyển động (dòng 185–186).
- **Release từ command:** chỉ đổi action sang mở đã phát confirmed release, ngay cả nếu finger gap chưa mở (dòng 244–269).
- **Không có containment/count/order:** không có bằng chứng cho mục tiêu memory LIBERO-Mem; những fact này không thể sinh đáng tin chỉ từ EEF và gripper.
- **Writer Mem mới chưa thay thế trực tiếp được:** `CausalObjectTracker` vẫn cần externally supplied detections tọa độ 3D và fixed support mặc định. Chỉ đổi class import không tạo detector hoặc giải quyết provenance.

## 7. Hướng thay đổi đề xuất

**Giữ hạ tầng, thay phần tạo bằng chứng và xác nhận sự kiện.** Store, episode isolation, provenance, causal indexing và token-budget audit có thể giữ sau regression checks. Writer hiện tại nên giữ thành baseline heuristic được ghi nhãn rõ; không dùng như nguồn fact chuẩn cho thesis memory.

Luồng triển khai:

`hai camera + proprio → perception/tracking causal → relations có evidence → event FSM → structured memory → retrieval/rendering`.

- Ground instance identity từ ảnh, giữ ID/origin ổn định. Proprio/action là evidence phụ. Nếu dùng tọa độ 3D, phải có calibration/độ sâu hoặc nguồn hợp lệ; không đổi bbox 2D thành 3D bằng giả định.
- Grasp candidate từ gripper; xác nhận bằng object association/co-motion và bằng chứng quan sát phù hợp. Dùng relative geometry với support đã quan sát, không dùng một ngưỡng world-z cho mọi scene. Dwell và threshold phải được kiểm định trên dữ liệu held-out, không chỉ tăng coverage.
- Release command chỉ mở một candidate. Placement/return cần thấy vật ở quan hệ support/container hợp lệ, ổn định đủ thời gian. Không xác nhận thành công chỉ vì gripper mở hoặc object biến mất.
- Lưu cả latest state và event history: `occurred_at`, `available_at`, evidence frame/camera IDs, source, validity, confidence, correction/supersession. Mất observation phải thể hiện unknown/occluded/history incomplete.
- Bowl/drawer: nhớ **last-confirmed containment** xuyên occlusion; theo dõi drawer open/closed riêng. Không biến last-confirmed thành quan sát hiện tại chắc chắn khi bằng chứng đã yếu đi.
- LIBERO-Mem: giữ instance/original support/container, số chu kỳ return đã xác nhận, in-progress/uncertain count, thứ tự sự kiện và lịch sử không đầy đủ. Không đưa “subgoal tiếp theo” từ annotation vào như memory fact.
- Cùng writer phải chạy causal từng prefix ở offline preparation và online rollout. Ground-truth simulator/annotation dùng làm nhãn kiểm định hoặc nhánh oracle riêng; predicted memory vẫn phải tạo được từ đúng input deployment.

Trình tự trước lần SFT tiếp theo: (1) bộ clip/episode có nhãn kiểm chứng cho book và bowl, (2) audit precision/recall/timing/identity/placement, (3) offline/online parity và prompt coverage ở decision windows, (4) shadow writer và paired memory-off regression trên LIBERO base, (5) pilot T1/T3 cho count memory. Sau đó mới chuẩn bị dataset lớn và đánh giá model memory/no-memory cùng demonstration, split và ngân sách.

Mục tiêu là **memory đúng và hữu ích ở thời điểm cần**, không phải mọi frame đều có text. Giữ performance base cần đo bằng rollout có đối chứng; không thể bảo đảm chỉ bằng giữ instruction format hoặc tăng memory coverage.
