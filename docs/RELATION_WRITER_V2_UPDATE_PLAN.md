# Kế hoạch cập nhật Relation Writer V2 cho T1/T3

## Trạng thái sau cập nhật motion

[Kết quả mới](RELATION_WRITER_V3_MOTION_RESULTS_2026-11-18.md) của `rgb_relations_v3` đạt 195/394 mốc development và 1 FP trên E2 replay. Kết quả bên dưới ghi lần thử candidate trước sửa motion. Writer RGB Cycle lịch sử mang mã `rgb_cycle_v3`. [Bảng đối chiếu checkpoint và writer](thesis/EXPERIMENT_CONFIGURATION_REGISTER.md)

## Trạng thái candidate trước sửa motion — 2026-11-18

Đã triển khai nhánh opt-in `rgb_relations_v3`; `rgb_relations_v2` vẫn có replay dự đoán khớp archive trên 198 demo T1/T3. V3 bổ sung anchor muộn có điều kiện liên tục, phục hồi một khoảng mất dấu lúc nghỉ, giữ departure đã xác nhận qua mất dấu ngắn, hết hạn lift còn chờ và cấu hình ngưỡng theo phiên bản. Renderer V3 không phát số lượt nếu origin chưa xác minh. V3 chưa được nghiệm thu cho rollout.

Sweep dev chọn được cấu hình đạt gate đề xuất: T1 **68/100**, T3 **128/294**, 0 dự đoán ngoài cửa sổ; median delay 8 frame, max 17. Nhưng lần replay E2 sau khi khóa cấu hình cho **30** dự đoán ngoài cửa sổ/40 episode, so với **1** của V2; chỉ ghép được 1/3 mốc. Vì vậy candidate không được promote. File cấu hình V3 đang giữ giá trị đối chứng ban đầu; candidate và hash đã dùng cho E2 được ghi trong các audit receipt bên dưới. E2 có ít mốc dương, nhưng mức báo dư này đủ để chặn rollout.

Đường băm RGB đã được sửa để tính một SHA-256 mỗi camera mỗi bước và tái sử dụng trong detector cùng validate; test xác nhận đúng hai lần băm cho hai camera. Snapshot bất biến vẫn được giữ. Cached benchmark trên hai demo RGB 256×256 cho p50/p95 tổng V2 **1,20/2,33 ms** và candidate V3 **1,17/2,00 ms**; cùng phép đo hiện tại, V3 không chậm hơn V2. Đây là microbenchmark 2 episode × 5 lượt, không cô lập chi phí JSON/SHA signature riêng, và chưa xác nhận cùng máy với phép đo lịch sử 2,18/4,19 ms.

Writer hiện tại là `rgb_relations_v2` với điều kiện loại bbox bát và đĩa gần trùng nhau. Bản cập nhật cần giảm bỏ sót lượt đặt lại, giữ bằng chứng cho từng lượt đã xác nhận và phân biệt trường hợp chưa xác minh được đĩa gốc. Phạm vi là profile một bát quay lại đĩa gốc trên T1/T3. T7–T10 cần nghiệm thu identity và containment theo từng profile, trên dữ liệu có nhãn riêng. [Writer hiện tại](../src/memory/relation_cycle_writer.py) · [Audit T7–T10](thesis/MEMORY_DECISION_HANDOFF_2026-11-14.md)

## Thứ tự triển khai

1. **Thay hạn neo 20 bước bằng điều kiện xác minh đĩa gốc.** Theo dõi từ bước reset xem gripper có liên tục mở và bát có giữ vị trí quan sát được hay không. Cho phép neo muộn nếu chuỗi đầu vào liên tục, bát có identity nhất quán, chỉ một đĩa phù hợp và quan hệ ổn định qua nhiều mẫu. Khi dấu hiệu đóng kẹp, dịch chuyển vật hoặc thiếu bước khiến nguồn gốc khó phân xử, giữ `origin` chưa xác minh. Renderer bỏ fact đếm hoặc ghi rõ đĩa gốc chưa xác minh. Khi đã neo, các quan sát sau chỉ có thể xác minh lại cùng identity. Kiểm cả đường tracker đã có neo bbox nhưng reducer chưa có `origin`.
2. **Xử lý khoảng mất dấu và lượt nhấc đang chờ.** Tách lượt return đã xác nhận, khoảng mất dấu còn mở và khoảng chưa thể giải quyết. Một flicker ngắn lúc bát đứng yên có thể được giải quyết bằng cùng identity, cùng đĩa, kẹp mở được đo liên tục và ổn định trước/sau. Nếu khoảng che khuất có thể chứa thao tác, giữ định tính `History incomplete.` cùng số lượt quan sát được. Giữ `departure` có bằng chứng khi mất detection ngắn và tái nhận diện đáng tin cậy. Giới hạn tuổi cửa sổ xác nhận từ mẫu hợp lệ đầu tiên để hai khoảng mất dấu riêng rẽ không nối các mẫu nhấc đã cũ. Thiếu bước thật hoặc xung đột kéo dài làm hết hạn dữ kiện đang chờ. Mọi return cần một `departure` chưa dùng, cùng identity và bằng chứng có timestamp hợp lệ.
3. **Đưa ngưỡng ra cấu hình có phiên bản.** Khai báo đơn vị cho score detector, tỷ lệ bbox, độ mở kẹp, số mẫu và thời gian theo observation. Dùng cấu hình hiện tại làm đối chứng. Trên development, đo độ nhạy quanh ngưỡng theo sai lệch bbox quan sát được trong dữ liệu, gồm từng ngưỡng trước rồi mới thử phối hợp. Báo riêng ca mất đĩa gốc, bbox bát/đĩa trùng, nhấc chưa xác nhận, đến sai đĩa và đặt chưa ổn định. Kẹp và camera cổ tay chỉ cung cấp tín hiệu phối hợp với bằng chứng hình học. Chốt cấu hình cùng hash trước khi mở tập nghiệm thu độc lập.
4. **Giảm chi phí kiểm toán trên đường chạy bình thường.** Tính SHA-256 RGB một lần mỗi camera trong writer và tái sử dụng các digest tính tại chỗ để xác thực evidence. Giữ so sánh trùng observation theo nội dung ảnh, gripper, preceding action và evidence. Đo riêng chi phí serialize, lọc evidence và freeze trước khi thay cách lưu snapshot. Giữ snapshot bất biến, tập evidence hữu hạn và đầy đủ dependency của event còn được dùng. Kiểm parity snapshot, event và prompt cho phần tối ưu thuần hiệu năng.
5. **Đăng ký revision và xác nhận tương thích.** Chụp snapshot mã và cấu hình hiện tại trước khi sửa. Lưu hash cấu hình, source, detector, dữ liệu và protocol cho từng revision. Dùng đúng snapshot để chạy lại receipt cũ, kể cả registry nằm trong source hash. Đối chiếu parity trên cùng nguồn input ở đường dựng dữ liệu và rollout, gồm `available_at`, evidence ID, text và token budget. Smoke F1 dùng đường chạy thông thường chọn writer theo manifest checkpoint. Nhánh decision execution đang khóa F1 với `rgb_relations_v1` và cần nghiệm thu contract riêng nếu muốn mở cho revision mới. [Kiểm contract khi nạp F1](../src/training/mem_adapter_artifact.py) · [Runner F1](../scripts/evaluate_mem_adapter.py)

## Test hồi quy và nghiệm thu

| Nhóm | Điều kiện nghiệm thu |
|---|---|
| Phục hồi sau nhiễu | Một frame mất detection lúc nghỉ được giải quyết sau khi quan hệ ổn định trở lại và lượt đặt tiếp theo hiện đúng số. Mất dấu giữa lúc nhấc vẫn giữ định tính bất toàn nếu thiếu bằng chứng về khoảng che khuất. |
| Vòng đời của `departure` | Mất 1 hoặc 22 frame với dữ kiện tái nhận diện phù hợp vẫn ghép đúng departure đã xác nhận. Hết hạn, xung đột kéo dài hoặc bỏ bước làm mất hiệu lực lượt đang chờ. Ba mẫu nhấc cách nhau bởi hai khoảng mất dấu dài không được ghép thành một lần nhấc. Lượt return đã ghi chỉ đếm một lần. |
| Neo muộn | Đĩa lần đầu xuất hiện ở bước 35 trước thao tác có thể được neo và xác nhận. Sau đóng kẹp khi chưa xác minh nguồn gốc, trạng thái chưa có origin không được biểu diễn thành số 0. Thử thêm nhiều đĩa, đổi identity và bỏ bước trước khi xác minh neo. |
| Ngưỡng hình học | Dịch chuyển nhỏ chung giữa hai bbox giữ được một chu kỳ hợp lệ. Bbox nhấp nháy, bbox trùng nhãn bát/đĩa và đặt sai đĩa không tự tạo return. Bổ sung ca quanh từng ngưỡng sau khi có cấu hình và phân bố nhiễu từ development. |
| Tính toàn vẹn và chi phí | Hai camera được băm đúng hai lần tổng cộng trên mỗi bước cached. Gửi lại cùng observation trả cùng snapshot. Sửa gripper, action hoặc detection khi gửi trùng phải báo lỗi và giữ snapshot. Giữ các test stale/future evidence, sai ảnh, sai camera và nguồn đặc quyền. |

Các fixture chỉ kiểm tra logic ở cấp unit test; chúng không thay thế kiểm chứng với RGB thật. Test mới hiện là assertion bắt buộc cho các hành vi của V3. Ngày 2026-11-18, bộ hồi quy bên dưới qua **68 test**. [Test mới](../tests/unit/test_bowl_cycle_writer_update.py) · [Test cycle sẵn có](../tests/unit/test_bowl_cycle_writer.py) · [Test evidence dùng chung](../tests/unit/test_relation_memory.py)

```powershell
python -B -m pytest -q -p no:cacheprovider -rx tests/unit/test_bowl_cycle_writer_update.py tests/unit/test_bowl_cycle_writer.py tests/unit/test_relation_memory.py tests/unit/test_rgb_cycle_writer.py tests/unit/test_libero_mem_tracker_writer.py tests/unit/test_libero_mem_text_renderer.py tests/unit/test_libero_mem_rollout.py tests/unit/test_mem_adapter_artifact.py
```

## Replay và điều kiện chuyển sang rollout

1. **Khóa baseline.** Dùng audit đã xác minh SHA-256 trên 198 demo (42.857 frame) và E2 với 40 quỹ đạo F1 từ writer V1 (39.729 frame). So theo mốc tác giả cho demo và `confirmed_cycles` của evaluator cho E2, với cửa sổ ghép một-một ±20 frame. V1 và V2 hiện tại khớp lần lượt **194/394** và **164/394** mốc demo, cùng 0 dự đoán ngoài cửa sổ. T1 tương ứng **68 → 65**, T3 **126 → 99**. Trên E2, cả hai khớp **0/3** mốc, số dự đoán ngoài cửa sổ **5 → 1**. Dùng các quỹ đạo đã xem này cho kiểm tra hồi quy. [Replay demo](../experiments/audits/bowl-cycle-v2-development-alias-guard-20261118.json) · [Replay E2](../experiments/audits/bowl-cycle-v2-e2-alias-guard-20261118.json)
2. **Chạy đối sánh trên cùng input.** Mở rộng script replay để chọn đúng revision, kiểm source receipt và giữ bảng sự kiện từng episode. Thêm replay đầy đủ `writer.step()` và renderer trên một tập nhỏ RGB tĩnh đã hash-check để phát hiện sai khác tích hợp mà replay reducer riêng bỏ qua. Báo theo T1/T3 số mốc khớp, dự đoán ngoài cửa sổ, bỏ sót, độ trễ khớp, tỷ lệ không có origin, tỷ lệ có fact và tỷ lệ fact `History incomplete.`. Ngưỡng nghiệm thu development đề xuất là khôi phục ít nhất **68/100** mốc T1 và **126/294** mốc T3, giữ **0** báo dư. E2 là kiểm tra hồi quy với tối đa **1** dự đoán ngoài cửa sổ và theo dõi riêng ba mốc dương ít ỏi. [Script replay](../scripts/audit_bowl_cycle_replay.py)
3. **Kiểm chứng sau khi khóa cấu hình.** Dùng quỹ đạo có nhãn độc lập theo nhóm initial state, gồm lượt đặt đúng và ca không được đếm cho từng task. So sánh ghép cặp với cả V1 và V2 hiện tại, báo bất định theo nhóm state. Mục tiêu là recall theo từng task ít nhất bằng V1 và số dự đoán ngoài cửa sổ mỗi episode tối đa bằng V2 hiện tại. Báo trạng thái chưa nghiệm thu nếu tập này thiếu nhãn dương hoặc thiếu độ phủ theo task.
4. **Đo latency rồi kiểm runtime.** Với cùng máy, hai camera RGB 256×256 và cùng detection cache, lấy p50/p95 qua nhiều lượt trên hai demo dùng trong audit. So với V2 hiện tại (**2,18 / 4,19 ms** trong phép đo ban đầu), yêu cầu p95 của đường cached không tăng khi đo lại cùng điều kiện. Ghi riêng thời gian detector/IPC, `writer.step()`, renderer và toàn vòng trên lần kiểm live tiếp theo. Sau khi qua gate writer, chạy smoke với F1 có contract phù hợp và giữ khóa preprocessing, action decoding, controller cùng evaluator. Báo kết quả writer offline và mọi rollout custom với nhãn **NON-COMPARABLE** so với điểm LIBERO chính thức. [Runner và phép đo writer](../src/evaluation/libero_mem_rollout.py)

Lệnh tái lập baseline chạy từ thư mục gốc của repo.

```powershell
python -B scripts/audit_bowl_cycle_replay.py --development-index experiments/audits/relation-e1-20261113-232401/index-T1 experiments/audits/relation-e1-20261113-232401/index-T3 --data-dir data/libero_mem --tolerance 20
python -B scripts/audit_bowl_cycle_replay.py --rollout-root experiments/evaluation/e2_unseen_author_v2 --tolerance 20
```

### Receipt đã tạo

- [Sweep calibration V3](../experiments/audits/bowl-cycle-v3-calibration-sweep3-20261118.json): 198 demo, 42,857 frame; ghi từng candidate, cấu hình được chọn và gate dev. Candidate được chọn là `joint_broad_two_frame_confirmation`, config SHA-256 `b4174f85030711c983b000642fd4efc4a751daafebe65b195a30024c0215eb9c`.
- [Replay E2 sau khóa](../experiments/audits/bowl-cycle-v3-e2-locked-20261118.json): 40 episode, 39,729 frame, 3 mốc dương; writer `rgb_relations_v3`, phân loại `RECORDED_WRITER_REPLAY_NON_COMPARABLE`. Kết quả 1 TP, 30 FP, 2 FN; V2 trên cùng E2 có 0 TP, 1 FP, 3 FN.
- [Cached latency benchmark](../experiments/audits/bowl-cycle-v3-cached-latency-20261118.json): hai demo đã hash-check, 2,190 bước đo cho mỗi writer sau warm-up; ghi p50/p95 theo writer, renderer và tổng, cùng số lần digest thực tế/ kỳ vọng (5,256/5,256 mỗi writer).

Do candidate không qua E2, chưa mở rollout F1 và chưa xem số liệu này là điểm LIBERO chính thức. Một vòng sửa tiếp theo cần xử lý nguyên nhân dự đoán lặp trong các episode thất bại, rồi dùng một tập nghiệm thu độc lập mới; E2 hiện tại đã dùng cho gate này nên không còn là holdout cho các vòng tuning sau.
