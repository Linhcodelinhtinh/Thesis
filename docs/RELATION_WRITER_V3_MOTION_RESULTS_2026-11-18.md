# Kết quả cập nhật Relation V3 (`rgb_relations_v3`) cho T1/T3

Writer trong báo cáo này là Relation V3 motion. Kết quả vòng kín của adapter F1 lịch sử dùng RGB Cycle (`rgb_cycle_v3`) nằm trong [bảng đối chiếu checkpoint](thesis/EXPERIMENT_CONFIGURATION_REGISTER.md).

V3 giảm **30 → 1 báo dư trên E2** so với candidate cũ và giữ **195/394 mốc development**, thấp hơn candidate cũ một mốc và cao hơn V2 31 mốc. E2 đã được dùng để chẩn đoán và chọn cấu hình, nên các số E2 dưới đây là kết quả hồi quy. [Chẩn đoán 30 báo dư](RELATION_WRITER_V3_E2_FP_DIAGNOSIS_2026-11-18.md)

## Thay đổi trong writer

- V3 giữ pose bát tương đối với đĩa trong lần quan sát nghỉ gần nhất đã xác nhận. Departure yêu cầu cả mép trên và mép dưới của bbox bát dâng ít nhất 4% chiều cao anchor so với pose đó. Điều kiện này loại trường hợp bbox co từ phía dưới khi kẹp đóng.
- Return cần mép dưới hạ ít nhất 8% chiều cao anchor từ đỉnh đã quan sát lúc kẹp đóng và về gần pose nghỉ. Writer tiếp tục dùng kiểm tra vùng đỡ, mở kẹp, ổn định và identity đĩa gốc. Sự kiện đính kèm evidence của pose nghỉ và đỉnh lift. Các tham chiếu này được giữ trong bộ nhớ với kích thước hữu hạn. [Writer](../src/memory/relation_cycle_writer_v3.py) · [Các ca kiểm thử](../tests/unit/test_bowl_cycle_writer_update.py)
- Cấu hình chọn từ sweep dùng hai frame xác nhận, cửa sổ mất dấu 45 frame và tuổi pending tối đa 48 frame. Khi dùng cửa sổ 30/36 cùng các kiểm tra hình học mới, development đạt 184/394. Kéo dài cửa sổ phục hồi 11 mốc mà số FP E2 giữ ở 1. Thử giới hạn chiều cao bbox đĩa theo frame anchor làm development giảm còn 31/394 và E2 không cải thiện thêm. Kiểm tra đó đã được bỏ. [Cấu hình V3](../configs/memory/relation_cycle_v3.json)

## So sánh trên cùng cache

Development có 198 demo và 394 mốc do tác giả gán. E2 có 40 episode và ba mốc `confirmed_cycles`. Ghép sự kiện một-một trong ±20 frame. V1 lấy từ trace gốc. V2 dùng phiên bản có alias guard. Hai bản V3 trước cập nhật được ghi riêng do dùng cấu hình khác nhau. [Development V1/V2](../experiments/audits/bowl-cycle-v2-development-alias-guard-20261118.json) · [E2 V1/V2](../experiments/audits/bowl-cycle-v2-e2-alias-guard-20261118.json) · [Cấu hình V3 trước cập nhật](../experiments/audits/bowl-cycle-v3-calibration-sweep3-20261118.json) · [E2 V3 trước cập nhật](../experiments/audits/bowl-cycle-v3-e2-fp-ablation-20261118.json)

| Writer | T1 dev TP / 100 | T3 dev TP / 294 | Dev TP / FP / FN | E2 TP / FP / FN |
|---|---:|---:|---:|---:|
| V1 | 68 | 126 | 194 / 0 / 200 | 0 / 5 / 3 |
| V2 | 65 | 99 | 164 / 0 / 230 | 0 / 1 / 3 |
| V3 cũ, mặc định | 54 | 76 | 130 / 0 / 264 | 0 / 1 / 3 |
| V3 cũ, candidate | 68 | 128 | 196 / 0 / 198 | 1 / 30 / 2 |
| **V3 cập nhật** | **66** | **129** | **195 / 0 / 199** | **0 / 1 / 3** |

Độ trễ ghép development của V3 cập nhật có median 8 và tối đa 17 frame, bằng candidate cũ. So theo ID từng mốc, T1 mất bốn mốc và lấy lại hai, còn T3 mất tám và lấy lại chín. V3 cập nhật vẫn thấp hơn mục tiêu 68/100 của T1. Trên E2, báo dư còn lại là **T3 state 7, frame 341**. Cả ba mốc E2 đều không khớp. Match duy nhất của candidate cũ là ở T3 state 4 với departure ghi nhận sau khi predicate `On` đã phục hồi. [Replay development](../experiments/audits/bowl-cycle-v3-motion-development-20261118.json) · [Replay E2](../experiments/audits/bowl-cycle-v3-motion-e2-20261118.json)

Nới offset so với pose nghỉ từ 0,35 lên 0,48 tăng kết quả development lên 197/394, gồm 67 T1 và 130 T3. Biến thể này tạo 3 FP trên E2, nên cấu hình sử dụng 0,35 để giữ mức 1 FP của V2. [Development biến thể 0,48](../experiments/audits/bowl-cycle-v3-motion-final-sweep-development-longer_wider_settle-20261118.json) · [E2 biến thể 0,48](../experiments/audits/bowl-cycle-v3-motion-final-sweep-e2-longer_wider_settle-20261118.json)

## Latency và kiểm thử

| Writer | Cached p50 | Cached p95 |
|---|---:|---:|
| V1 | 1,08 ms | 1,76 ms |
| V2 | 1,14 ms | 1,74 ms |
| V3 cập nhật | 1,11 ms | 1,53 ms |

Phép đo dùng lại ảnh 256×256 và detector cache của hai demo, mỗi writer có 2.190 bước đo. Số tổng hợp là trung vị của các percentile theo demo. Latency gồm writer và renderer với bộ đếm token đơn giản, chưa gồm detector, IPC hoặc policy. Cả ba writer đạt 5.256/5.256 lần gọi digest kỳ vọng tính cả warm-up. **71 test đã qua** với cấu hình trong repo, gồm ba ca mới cho bbox co, mở kẹp khi bát vẫn ở pose nâng và hai lift nhỏ hợp lệ liên tiếp. [Receipt latency](../experiments/audits/bowl-cycle-v3-motion-cached-latency-20261118.json)

Kết quả là replay writer tùy chỉnh và thuộc phân loại **NON-COMPARABLE với điểm LIBERO chính thức**. Kiểm định tổng quát cần thêm quỹ đạo có nhãn ngoài E2. Các checkpoint hoặc index khóa SHA mã nguồn writer phải khớp revision đã tạo artifact, kể cả khi hành vi phát dự đoán của V1/V2 giữ nguyên.
