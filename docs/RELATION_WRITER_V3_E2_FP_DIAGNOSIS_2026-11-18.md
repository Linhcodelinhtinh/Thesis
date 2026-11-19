# Chẩn đoán 30 lượt báo dư của Relation Writer V3 trên E2

Ngày 18/11/2026 · T1/T3 · Replay ngoại tuyến · NON-COMPARABLE với điểm LIBERO chính thức

**23 lượt bắt đầu bằng departure sai, 7 lượt còn lại kết thúc bằng return sai.** Trong nhóm đầu, predicate `On` đúng xuyên suốt cặp sự kiện. Trong nhóm sau, `On` đã sai trước departure và vẫn sai lúc writer tăng bộ đếm. Ngưỡng lift thấp khuếch đại thay đổi hình dạng bbox. Hình học return rộng và điều kiện ổn định cục bộ tiếp tục cho phép đếm khi vị trí ngoài đĩa không đổi. [Trace chẩn đoán từng sự kiện](../experiments/audits/bowl-cycle-v3-e2-fp-diagnosis-20261118.json)

## Dữ liệu và cách đối chiếu

- Candidate là `rgb_relations_v3` với cấu hình `joint_broad_two_frame_confirmation`. SHA-256 cấu hình tái tạo từ calibration khớp receipt E2, và các hash mã nguồn liên quan đều khớp. Replay tái hiện đúng mọi frame return trên 40 episode, 39.729 frame, sau khi đối chiếu hash trace và observation với receipt từng episode. [Calibration](../experiments/audits/bowl-cycle-v3-calibration-sweep3-20261118.json) · [E2 đã khóa](../experiments/audits/bowl-cycle-v3-e2-locked-20261118.json)
- Đối chiếu hai tầng nhãn. TP/FP theo phép ghép một-một ±20 frame với mốc `confirmed_cycles`. Chẩn đoán dùng predicate `On` từng frame ghi trong `final_predicate` và cặp `departure`/ `return` mà writer thực sự ghép. E2 có ba mốc dương. Evaluator dùng bộ đếm tích lũy `live > 5` và `nonlive > 15` giữa các mốc. Predicate cho biết trạng thái theo simulator, nhưng trace không có nhãn tiếp xúc ngón kẹp từng frame để phân loại trượt, va chạm hoặc che khuất cụ thể. [Adapter ghi log](../src/simulator/libero_mem_env.py#L49) · [Evaluator ở checkout đã pin](../external/libero-mem/libero/libero/envs/problems/libero_kitchen_tabletop_manipulation.py#L206)
- `MEMORY_WRITER_V3_T1_T3.md` mô tả pilot RGB writer lịch sử với event schema và harness khác. Số liệu 12/12 trên sáu demo trong báo cáo đó thuộc pilot. Prompt counterfactual state 16 dùng F1 đã có và thay đúng một query. Kết quả overshoot ở đó không quy kết được cho candidate relation V3. [Báo cáo pilot](MEMORY_WRITER_V3_T1_T3.md) · [Prompt counterfactual](../experiments/evaluation/prompt_counterfactual_state16_20261118_followup/combined_report.md)

## Phân rã từng lượt báo dư

Các số dưới đây là frame writer **phát return**, tương ứng `available_at`.

| Episode E2 | Departure sai, `On` đúng xuyên suốt | Return sai, `On` vẫn sai |
|---|---|---|
| T1 state 4 | — | 349 |
| T1 state 17 | 498 | 797, 879, 967 |
| T1 state 18 | 103, 233, 398, 518, 655 | — |
| T3 state 2 | 155 | — |
| T3 state 7 | — | 341, 531, 645 |
| T3 state 15 | 563, 637, 701, 831, 877 | — |
| T3 state 18 | 92, 249, 343, 441, 520, 874, 968 | — |
| T3 state 19 | 529, 597, 689, 925 | — |
| **Tổng** | **23** | **7** |

### 23 lượt vượt ngưỡng lift khi bát vẫn trên đĩa

Bowl và plate anchor đều lấy từ frame 0 trong tám episode có FP. Trong 23 lượt này, tại lúc xác nhận departure, chiều cao bbox bát chỉ còn **66–79%** chiều cao anchor, median **77%**. Mép dưới báo dâng **0,216–0,261** chiều cao bbox anchor sau khi trừ chuyển động đĩa, nhưng mép trên bát dâng nhiều nhất **0,045** trong hệ tọa độ ảnh, và nhiều ca còn đi xuống. Hình học phù hợp với bbox co lại hơn là toàn bộ bát tịnh tiến lên. Trong 21/23 lượt, wrist cue xuất hiện ở ít nhất một frame xác nhận departure. Kẹp đóng và wrist cue đủ mở `pending` nhưng chỉ thể hiện tay ở gần bát. Cặp ngưỡng thấp **0,10/0,20** cho tâm/mép dưới biến sai lệch so với anchor thành lift. Sau khi kẹp mở, hộp gần như ở cùng độ cao mà FSM vẫn xác nhận return và có thể đếm lần nữa. [Điều kiện lift và return](../src/memory/relation_cycle_writer.py#L33)

State T3/15 làm rõ cơ chế này. Predicate `On` đúng ở mọi frame, nhưng writer đếm năm lượt. Trên toàn bộ nhóm 23, mức giảm của tín hiệu mép dưới từ đỉnh trong khi kẹp đóng tới frame return không quá **0,040** chiều cao anchor. Có 17/23 lượt mà `On` chưa từng sai kể từ lần settle trước của writer. Sáu lượt còn lại có một số frame `On=false` trước departure được ghép, nên kết luận ở đây giới hạn vào cặp bằng chứng mà writer đã sử dụng. [Trace từng frame xác nhận](../experiments/audits/bowl-cycle-v3-e2-fp-diagnosis-20261118.json)

### 7 lượt chấp nhận vị trí còn ngoài đĩa

Trong cả bảy lượt, `On` đã sai trước departure và sai xuyên suốt cặp sự kiện. Khi báo return, tay đã mở, bbox đủ ổn định hai mẫu và vượt qua phép chiếu overlap. Bốn lượt có offset return **0,390–0,469**, nằm giữa giới hạn V2 **0,35** và candidate **0,55**. Ba lượt khác có offset **0,157–0,285**, nên thu hẹp offset đơn lẻ không xử lý hết. Điều kiện IoU ≥ 0,90 hiện chỉ loại bbox bát và đĩa gần trùng tuyệt đối. Cả bảy lượt có IoU return **0,154–0,686**. Trace không xác định được bát ở đĩa khác, lơ lửng hay ở mặt bàn trong từng lượt. [Kiểm tra footprint và IoU](../src/memory/relation_cycle_writer.py#L43)

T1 state 17 cho thấy cả lỗi rearm. Sau lúc `On` rơi về false ở frame **758**, nó giữ false tới hết episode. Writer vẫn đếm ở **797, 879, 967**, vì mỗi return hình học lại đặt `ready=true`, cho phép lần đóng kẹp sau mở attempt mới. T3 state 7 báo return ở 341 và 531 trong khi `On` chỉ trở lại ở 596, rồi báo thêm ở 645 sau khi bát lại rời. [Điều kiện rearm](../src/memory/relation_memory.py#L116)

### Lượt TP của E2 cũng cần xét liên kết bằng chứng

T3 state 4 có `On=false` từ 436 đến 464 và mốc upstream tại **465**. Writer xác nhận departure ở **468** và return ở **481**, nên được tính TP với độ trễ +16. Predicate đã đúng trước hai mẫu departure 467–468 và đúng xuyên suốt tới 481. Đây là một match theo thời gian, nhưng cặp bằng chứng không ghi nhận lần rời đĩa đã tạo mốc 465. Tính cả trường hợp này có **24 cặp departure/return** mà predicate luôn đúng trong cặp. [Receipt E2](../experiments/audits/bowl-cycle-v3-e2-locked-20261118.json)

## Ablation trên cùng 40 episode

Mỗi dòng thay một nhóm tham số so với candidate, trừ dòng cấu hình V3 mặc định.

| Biến thể | TP | FP | FN |
|---|---:|---:|---:|
| V3 candidate | 1 | 30 | 2 |
| Cặp ngưỡng lift V2 0,30/0,40 | 0 | 5 | 3 |
| Offset return 0,35 | 1 | 26 | 2 |
| 3 frame xác nhận | 1 | 26 | 2 |
| Pending TTL 12 | 1 | 28 | 2 |
| Giới hạn mất dấu 22 frame | 1 | 30 | 2 |
| Aperture thả 0,055 hoặc vùng overlap kiểu V2, thử riêng | 1 | 30 | 2 |
| V3 config mặc định hiện tại | 0 | 1 | 3 |

Khôi phục cặp ngưỡng lift loại 25/30 FP. Thu hẹp offset loại đúng bốn lượt return ngoài đĩa thuộc T1. Chỉ **4/30** FP ghép hai frame xác nhận departure không liên tiếp. Gap có thể tạo FP nhưng tăng dwell hay giảm cửa sổ mất dấu toàn cục không giải quyết cơ chế chính. Config V3 mặc định giữ 1 FP trên E2 nhưng chỉ đạt **54/100 T1, 76/294 T3** trên development, so với **68/100 và 128/294** của candidate. E2 đã được dùng cho chẩn đoán nên dùng như tập hồi quy ở vòng tiếp theo. [Ablation E2](../experiments/audits/bowl-cycle-v3-e2-fp-ablation-20261118.json) · [Kết quả development](../experiments/audits/bowl-cycle-v3-calibration-sweep3-20261118.json)

## Đề xuất giữ recall và chi phí xử lý thấp

1. **Xác nhận lift yếu bằng chuyển động so với vị trí nghỉ gần nhất.** Giữ identity đĩa từ origin. Lưu vài mẫu bbox ổn định khi kẹp mở và chưa có attempt. Đóng băng mốc hình học này khi bắt đầu attempt. Nhánh lift có dịch chuyển mạnh tiếp tục dùng hai frame xác nhận. Với tín hiệu chỉ vừa qua ngưỡng thấp, yêu cầu tâm và hai mép cùng đi lên so với vị trí nghỉ, hoặc có bằng chứng rõ bát đã ra khỏi vùng đỡ. Kiểm độ co bbox trước khi dùng độ dâng mép dưới. Chỉ nối mẫu của cùng attempt khi còn bằng chứng chuyển động mới. Giữ khả năng bảo lưu departure đã xác nhận qua mất dấu ngắn. Mục tiêu là loại kiểu bbox co cố định ở 23 lượt mà vẫn cho phép lift nhỏ có chuyển động thực.
2. **Chỉ commit return khi có bằng chứng quay về vùng nghỉ, rồi mới rearm.** Với lượt có lift rõ, cho phép dùng mức hạ theo quỹ đạo và vị trí kết thúc tương thích với footprint đã xác minh. Với lượt biên hoặc bbox đĩa đổi kích thước rõ, đòi bằng chứng mạnh hơn về identity/footprint và thay đổi từ pose ngoài đĩa về pose nghỉ. Giữ kết quả tentative khi chỉ thấy kẹp mở và hộp đứng yên, đồng thời giữ nguyên số lượt đã xác nhận. Cách này nhắm tới bảy lượt return ngoài đĩa và chặn lặp sau false settle.
3. **Đo cùng điều kiện trước khi chọn revision.** Hai kiểm tra chỉ dùng tọa độ bbox, kẹp và một số mẫu quá khứ hữu hạn, với chi phí O(1) mỗi frame. Chạy lại trên cùng 198 demo để giữ ít nhất **68/100 T1 và 128/294 T3**, 0 FP theo mốc, kiểm các mốc bị mất khi so ghép cặp và độ trễ riêng ở nhánh lift rõ. Kiểm trên tập rollout có nhãn độc lập mới với ca gắp hụt, đặt sai, xê dịch bbox và chu kỳ thật ở cả hai task. So cả match ±20 và liên kết cặp departure/return theo predicate; goal state của simulator chỉ dùng ở audit. Lặp phép đo latency theo task và theo thứ tự xen kẽ. Mốc của candidate là p95 tổng **2,00 ms** trong microbenchmark cached hai demo, tính bằng trung vị p95 hai episode. Riêng T3 candidate có p95 **2,15 ms** so với **2,00 ms** của V2 trong phép đo này. Vì vậy cần theo dõi cả tail latency mỗi task. [Receipt latency](../experiments/audits/bowl-cycle-v3-cached-latency-20261118.json)

Hai kiểm tra hình học là đề xuất cho revision kế tiếp. Ảnh hưởng lên recall và latency cần được đo qua replay trước khi thay cấu hình hoặc huấn luyện F1.
