# So sánh Relation Writer V1, V2 và V3

Ngày 18/11/2026. Snapshot trước sửa motion, trên T1/T3. V3 trong tài liệu này là candidate `rgb_relations_v3` cũ. Số liệu cấu hình sau cập nhật nằm trong [kết quả motion](RELATION_WRITER_V3_MOTION_RESULTS_2026-11-18.md). Đối chiếu với RGB Cycle lịch sử `rgb_cycle_v3` và checkpoint F1 ở [bảng định danh](thesis/EXPERIMENT_CONFIGURATION_REGISTER.md).

## Kết quả định lượng

Các replay offline dùng cùng detection cache và ghép sự kiện với cửa sổ ±20 frame. Development có 198 demo, 394 mốc do tác giả gán. E2 có 40 rollout F1 nhưng chỉ 3 mốc dương, nên tỷ lệ TP ở E2 có bất định lớn; số FP vẫn là tín hiệu hồi quy rõ.

### Development

| Phiên bản | T1 TP / 100 | T3 TP / 294 | Tổng TP / 394 | FP | FN | Median / max delay |
|---|---:|---:|---:|---:|---:|---:|
| V1 (`rgb_relations_v1`) | 68 | 126 | 194 (49.2%) | 0 | 200 | 9 / 19 frame |
| V2 (`rgb_relations_v2`) | 65 | 99 | 164 (41.6%) | 0 | 230 | 9 / 18 frame |
| V3 candidate (`rgb_relations_v3`) | 68 | 128 | 196 (49.7%) | 0 | 198 | 8 / 17 frame |

V2 giảm 30 TP so với V1, phần lớn ở T3. Candidate V3 khôi phục hai TP so với V1 và 32 so với V2, không tăng FP trên development. Đây là milestone matching, không phải nhãn chính xác cho tiếp xúc vật lý.

Có một receipt V2 cũ hơn (`bowl-cycle-v2-development-20261118.json`) cho 170/394 TP, khác 164/394 ở alias-guard receipt. Hai receipt có SHA-256 khác nhau cho `relation_cycle_writer.py`; vì vậy bảng này dùng alias-guard snapshot được calibration xác nhận là replay đúng code V2 hiện tại. E2 cũng có receipt trước alias-guard với 4 FP so với 1 FP của snapshot hiện tại. Không gộp các kết quả từ hai snapshot V2.

### E2 sau khi khóa candidate

| Phiên bản | Mốc dương | TP | FP | FN | FP / episode |
|---|---:|---:|---:|---:|---:|
| V1 baseline | 3 | 0 | 5 | 3 | 0.125 |
| V2 hiện tại | 3 | 0 | 1 | 3 | 0.025 |
| V3 candidate | 3 | 1 | 30 | 2 | 0.750 |

V3 cải thiện một TP nhưng tăng 29 FP so với V2. Candidate không qua gate độc lập (tối đa 1 FP) và không được promote. Ba mốc dương không đủ để kết luận recall tổng quát, nhưng 30 báo dư trên 40 episode là đủ để chặn rollout.

### Latency cached path

Đo trên hai demo T1/T3, ảnh 256×256, cached detections, một warm-up và 5 lượt đo. Mỗi writer có 2,190 bước đo. Renderer dùng bộ đếm `split()` đơn giản; benchmark không gồm detector/IPC.

| Phiên bản chạy trong benchmark | Writer p50 / p95 | Renderer p50 / p95 | Tổng p50 / p95 |
|---|---:|---:|---:|
| V2 | 1.152 / 2.245 ms | 0.047 / 0.091 ms | 1.204 / 2.332 ms |
| V3 candidate | 1.116 / 1.902 ms | 0.049 / 0.093 ms | 1.168 / 1.996 ms |

V3 candidate không tăng latency trong phép đo này; p95 tổng thấp hơn khoảng 14%. Mỗi writer có 5,256 lần gọi digest, đúng bằng số kỳ vọng. Đây là microbenchmark nhỏ, không cô lập chi phí JSON signature, không chứng minh toàn bộ chi phí crypto đã được loại bỏ, và chưa xác minh điều kiện máy/phép đo lịch sử 2.18/4.19 ms có giống nhau.

## Hành vi và hợp đồng

| Hạng mục | V1 | V2 | V3 candidate |
|---|---|---|---|
| Origin sau bước 20 | Không neo muộn; có thể làm mất toàn bộ bộ đếm | Vẫn dùng acquisition cutoff cũ | Cho neo muộn khi chuỗi từ reset liên tục, bát được theo dõi và gripper mở; renderer không phát số 0 khi origin chưa xác minh |
| Flicker lúc nghỉ | `complete` bị hạ và không tự phục hồi | Có thể giữ trạng thái incomplete sau nhiễu ngắn | Khôi phục completeness sau khi cùng origin ổn định trở lại và không có dấu hiệu thao tác trong gap |
| Departure và mất dấu | `gap`/`conflicted` xóa departure | Bảo lưu qua cửa sổ mất dấu ngắn; gap dài làm mất attempt | Bảo lưu departure đã xác nhận qua mất dấu ngắn; attempt chờ có TTL; gap dài hoặc bỏ bước không được nối thành chu kỳ |
| Ngưỡng bbox/gripper | Rải trong code | Có ngưỡng cho cycle writer nhưng chưa có sweep như candidate | `BowlCycleConfig` có version/hash; sweep one-factor và tổ hợp trên development |
| Integrity/crypto | Hash RGB và signature observation; snapshot bất biến | Cùng lớp writer nền | Một SHA-256 mỗi camera mỗi bước, tái sử dụng ở detector/validate; duplicate signature, snapshot bất biến và event evidence vẫn giữ |
| Phạm vi | Relation profiles hiện có | Cycle writer cho một bát T1/T3 | Vẫn chỉ một bát T1/T3; không nghiệm thu T7–T10 |

V1 là baseline tham chiếu, V2 là điểm so sánh đang dùng, còn V3 là revision opt-in riêng để tránh đổi ngầm checkpoint/source contract. V2 replay dự đoán khớp archive trên development. Bộ unit/regression cho các nhánh thay đổi có **68 test passed**.

## Trạng thái cấu hình V3

Candidate development có:

- `confirmation_frames=2`, `max_missing_steps=30`, `pending_expiry_steps=36`.
- `lift_center_rise_ratio=0.10`, `lift_bottom_rise_ratio=0.20`, `return_offset_ratio=0.55`.
- `released_aperture_meters=0.05`, `support_horizontal_ratio=0.55`, `support_vertical_tolerance_ratio=0.15`.
- Config SHA-256: `b4174f85030711c983b000642fd4efc4a751daafebe65b195a30024c0215eb9c`.

Các giá trị đó đạt development gate nhưng gây FP cao trên E2. Vì vậy chúng chỉ nằm trong calibration receipt; [config V3 hiện tại](../configs/memory/relation_cycle_v3.json) được trả về giá trị đối chứng ban đầu. V3 chưa sẵn sàng cho rollout F1 và chưa có checkpoint được huấn luyện với V3.

## Artifact và tái lập

- [V2 development alias-guard replay](../experiments/audits/bowl-cycle-v2-development-alias-guard-20261118.json)
- [V2 E2 alias-guard replay](../experiments/audits/bowl-cycle-v2-e2-alias-guard-20261118.json)
- [V3 calibration sweep](../experiments/audits/bowl-cycle-v3-calibration-sweep3-20261118.json)
- [V3 locked E2 replay](../experiments/audits/bowl-cycle-v3-e2-locked-20261118.json)
- [V2/V3 cached latency](../experiments/audits/bowl-cycle-v3-cached-latency-20261118.json)

Tất cả kết quả trên là replay/thử nghiệm writer tùy chỉnh, **NON-COMPARABLE với điểm LIBERO chính thức**.
