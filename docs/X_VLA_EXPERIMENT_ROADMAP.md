# Kế hoạch thực nghiệm X-VLA và SmolVLA

Ngày 20/11/2026. Đây là thiết kế cho các đợt chạy mới. X-VLA là backbone chính để khảo sát đánh đổi giữa học tác vụ đếm chu kỳ và duy trì kỹ năng thao tác. SmolVLA là backbone đối chứng. Phần V1 kết thúc ở baseline frozen trên LIBERO. Huấn luyện có dữ liệu Mem, writer và đánh giá T1/T3 thuộc V2.

## Mốc xuất phát

| Phép đo đã có | Kết quả | Phạm vi sử dụng |
|---|---|---|
| X-VLA-Libero full frozen | 10/10 trên LIBERO-10, mỗi task dùng state 0. Task 00 có một lần gắp lại vật trước khi hoàn tất. | Smoke trên đủ task. Dùng video để định nghĩa cách gán nhãn recovery. |
| SmolVLA E6 lịch sử | B0 37/50, F1-off 23/50, giảm 28 điểm phần trăm | Windows, robosuite 1.5.2, BDDL 3.6.0. ID state trùng split nhưng hash vector khác. So sánh ghép cặp hợp lệ trong đợt cũ. |
| SmolVLA E2 V2 | F0 đạt 3/20 T1 và 1/20 T3. F1 đạt 0/20 trên từng task. | Mốc tham chiếu trên 20 nhóm unseen và writer Relation V1. |

[Receipt X-VLA](../experiments/results/x_vla_libero_10_pilot10_20261120_085204/run.json) · [Định danh checkpoint và bằng chứng SmolVLA](thesis/EXPERIMENT_CONFIGURATION_REGISTER.md)

## Thiết kế chính và ngân sách

| Phép đo | Arm cho mỗi backbone | Roster cho mỗi arm | Đầu ra chính |
|---|---|---|---|
| E6, duy trì kỹ năng LIBERO-10 | B0, F0, F1-off. Cả ba chỉ nhận chỉ dẫn. F1-off dùng đúng export F1 của E2. | 10 task × 5 state chính thức × 1 seed = 50 | SR trước/sau, từng task, chênh lệch ghép cặp theo điểm phần trăm |
| E2, LIBERO-Mem | B0, F0, F1-on, F1-off | T1 và T3 × 20 nhóm raw unseen `demo_101`–`demo_120` × seed 42 = 40 | SR cuối episode, SCR, số chu kỳ và overshoot, riêng T1/T3 |

E6 cần 300 rollout cho sáu arm. E2 cần 160 rollout mỗi backbone. Ưu tiên hoàn tất 160 lượt X-VLA. Để có bảng E2 hai backbone đủ ô cần 320 lượt tổng cộng, tính cả kết quả có thể tái sử dụng. Chỉ tái sử dụng 80 lượt SmolVLA F0/F1 đã có sau khi đối chiếu checkpoint, state hash, seed, scorer, warmup, horizon và cadence. Nếu khác điều kiện chính, chạy lại và giữ mã đợt riêng. Pilot, debug, ablation và lần chạy lại do lỗi hạ tầng có sổ riêng.

## 1. Khóa giao thức và đo B0 trước khi train

- Chốt roster E6 trước khi xem kết quả. Kiểm tra năm ID/task của E6 cũ, ánh xạ từng ID vào array chính thức ở LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`. Ghi SHA-256 của BDDL, array và từng vector thực nạp, đồng thời đối chiếu với split training. Nếu không xác nhận được ánh xạ với demo held-out, đặt tên roster mới và ghi rõ phạm vi tuyên bố về unseen. Cả sáu arm đọc cùng một lock.
- Dùng simulator tham chiếu WSL/Linux, robosuite 1.4.0, cùng robot, physics, control 20 Hz, quy tắc reset, 10 bước settling, tiêu chí success và ngân sách đề xuất 900 policy step/episode cho E6. Ghi seed simulator và seed inference từng episode, reset RNG trước từng lượt, để sharding/resume tái lập được. Log renderer thực, mã evaluator, config, checkpoint và hash export đang nạp. Giữ simulator và từng môi trường model tách biệt.
- Giữ contract riêng của policy. X-VLA dùng full checkpoint `129e71460678b7236cee6fc9707f09d9fa0c3590`, processor và absolute OSC, chunk 30 chạy 30. SmolVLA giữ processor, delta OSC và lịch thực thi của cấu hình đã audit. Ghi cả chunk dự đoán lẫn số action thực thi mỗi query. Bảng ngang backbone đo toàn bộ policy ở giao diện bản địa. Chạy E6 B0 của hai backbone ngay khi smoke xác nhận cùng state và scorer.
- Trên Mem, dùng nguồn raw, episode evaluator và split V2 đã khóa. Smoke B0 ở seen với T1 và T3. Kiểm tra riêng chuyển sang absolute sau 20 bước warmup của Mem, reset bộ đếm tiến độ, hướng ảnh cho model và hướng ảnh cho writer, reward, overshoot và terminal predicate. Giữ scorer đặc quyền trong evaluator, writer chỉ đọc tín hiệu quan sát được. Ghi B0 trên roster unseen đã khóa và giữ kết quả ngoài vòng phát triển.

[Runner và contract X-VLA](X_VLA_LIBERO_BENCHMARK_PLAN.md) · [Giao thức Mem](../configs/benchmarks/libero_mem_t1_t3_protocol.json) · [Split V2](../resources/manifests/libero_mem_t1_t3_base10_split_v2.json)

## 2. Chuẩn bị dữ liệu và kiểm chứng đường huấn luyện X-VLA

- Chốt cùng 598 trajectory train như SmolVLA V2, gồm 400 base từ 10 task và 198 Mem. Khóa episode, source hash, thứ tự frame, nhãn hành động, sampler và ngân sách mẫu. Giữ nguyên phân chia refined train và raw unseen. Ghi riêng giới hạn do lịch sử pretraining của checkpoint.
- Từ demo delta, replay trong đúng môi trường nguồn để thu `controller.goal_pos` và `goal_ori`, mã hóa absolute EEF6D và gripper theo mã tác giả. Kiểm replay hai chiều trên base, T1, T3 với cùng state, so chuỗi target, ảnh, vật và mốc scorer. Xác nhận offset ảnh vì upstream LiberoHandler bỏ frame đầu. Kiểm domain 3, hai camera, proprio, tay padding, chunk cuối và quy tắc sampling. Dừng khi frame, action, camera hoặc tiến độ chưa đối chiếu được. Lưu receipt cho từng demo, mọi loại trừ làm thay đổi corpus phải được khai báo cho cả hai arm.
- F1 gắn `rgb_relations_v1` + `relation_text_v1` ở chế độ `relevant` vào đúng query trên demo Mem. Base giữ nguyên chỉ dẫn. Test snapshot chỉ sử dụng bằng chứng sẵn có tới thời điểm query, log prompt sau tokenizer, truncation và tỷ lệ query có sự kiện. Đối chiếu với trace F0 để xác nhận ảnh, state và target cùng frame.
- Smoke training trong env riêng trên GPU dự kiến dùng. Chạy forward/backward, một đoạn train ngắn, export, reload ở server và đối chiếu output trước/sau reload với cùng payload và noise. Ghi peak VRAM, throughput, dung lượng shard và thời gian ước tính cho 5.000 update. Xác minh manifest gắn adapter với đúng full checkpoint từ pilot.

[Chuyển action theo X-VLA](../external/X-VLA/evaluation/libero/preprocess.md) · [LiberoHandler](../external/X-VLA/datasets/domain_handler/simulations.py) · [Trainer PEFT](../external/X-VLA/peft_train.py)

## 3. Train hai arm độc lập từ B0

| Thành phần | F0 | F1 |
|---|---|---|
| Checkpoint khởi đầu | Cùng X-VLA-Libero full frozen tại thời điểm khởi tạo | Cùng bản B0 và cùng khởi tạo adapter |
| Dữ liệu base | Chỉ dẫn gốc | Chỉ dẫn gốc |
| Dữ liệu Mem | Chỉ dẫn gốc | Chỉ dẫn + sự kiện RGB nhân quả |
| Ngân sách chính | 5.000 optimizer update thành công | Cùng 5.000 update, cùng lịch lấy mẫu |

Điểm xuất phát là PEFT tác giả với LoRA rank 8, alpha 16, target `all-linear`, lưu soft prompt hub và action encoder/decoder. Audit tham số thực sự trainable. Chốt LR, batch hiệu dụng, precision và gradient accumulation sau smoke trên train/dev, trước khi chạy cả hai arm. Đề xuất trộn 50% base và 50% Mem theo global batch. Giữ nguyên scheduler, seed, sample schedule, augmentation, số cập nhật và quy tắc chọn checkpoint giữa F0/F1. Lưu snapshot mỗi 500 update, dùng step 5.000 làm endpoint chính đã định trước. Log loss theo miền và cùng noise khi so prompt. Đánh giá loss Mem trên train/dev phải ghi đúng phạm vi split. Export hai adapter độc lập, ghi hash và kiểm server nạp đúng trọng số qua probe xác định trước.

## 4. Đo sau fine-tune và đọc kết quả

- **E6.** Trên từng backbone, tính F0−B0, F1-off−B0 và F1-off−F0. Xuất số cặp cùng đúng, mất success, thêm success, cùng sai. Giữa hai backbone, báo chênh lệch B0 và chênh lệch mức suy giảm theo cùng task/state. SR đi kèm số đếm, khoảng tin cậy 95% ghép cặp phân tầng task và bảng từng task. Với năm state/task, ưu tiên đọc mẫu lỗi qua episode cụ thể.
- **E2.** Giữ tiêu chí cuối 1.000 bước của evaluator, dừng sớm khi overshot không thể phục hồi. Chấm thành công khi đủ đúng số mốc, predicate cuối đúng và không overshot. Dùng cadence chính đề xuất 10 action/query để khớp E2 SmolVLA. Với X-VLA, đây là biến thể 10/30, phải định nghĩa và kiểm proprio theo action thực thi cuối prefix trước khi chạy. Kiểm trên seen đối chiếu với cadence bản địa 30/30 và ghi commit biến thể. Nếu chưa qua kiểm chứng, chạy 30/30 với nhãn riêng và giới hạn so sánh ngang. Duy trì cùng cadence cho cả bốn arm X-VLA trong mỗi bảng.
- **Ảnh hưởng của memory.** F0−B0 đo thay đổi sau SFT chỉ dẫn. F1-on−F0 đo tác dụng của cả recipe memory. F1-on−F1-off đo độ nhạy với memory lúc suy luận ở cùng trọng số. Báo từng task, SCR, phân phối 0/1/2/3 mốc, vượt chu kỳ, predicate cuối và bước đạt mốc đầu. Tính CI theo nhóm state, giữ T1/T3 cùng nhóm khi resample. Một seed cho kết luận có điều kiện theo seed đó.
- **Chi phí.** Báo số policy query, median/P95 latency mỗi chunk, thời gian detector/writer và simulator, wall time/episode, peak VRAM. Ghi cả lần đầu, sau đó báo riêng steady state. Dự toán GPU-giờ từ smoke riêng của E6 và E2 F1 có writer trên đúng máy.
- **Quy tắc công bố.** Điểm trên LIBERO-10 ở roster cục bộ mang nhãn `NON-COMPARABLE_OFFICIAL_PAPER`; T1/T3 theo `CUSTOM_V2_LIBERO_MEM`. Báo số episode dự kiến, hoàn tất, lỗi hạ tầng và thiếu. Giữ bảng paired theo các cặp đủ, chỉ kết luận trên roster đầy đủ sau khi khôi phục các lượt lỗi. Lưu key task/state/seed, trajectory, timing, video, success và lý do kết thúc. Cập nhật sổ checkpoint và claim ledger sau khi đối chiếu log.

## 5. Thực nghiệm bổ sung theo thứ tự ưu tiên

1. **Recovery tự nhiên trên E6.** Khóa rubric từ video pilot rồi áp dụng cho toàn roster B0/F0/F1-off của cả hai model. Tách hụt gắp, rơi vật và đặt lệch. Ghi frame lỗi, lần thử lại trên cùng vật, tái đạt mục tiêu con và success cuối. Báo tần suất episode gặp lỗi, tỷ lệ khắc phục trong số episode gặp lỗi và số bước tới recovery. Có người thứ hai rà các ca mơ hồ. Tần suất lỗi ban đầu có thể khác giữa policy. Nếu cần đo nhân quả, bổ sung cặp perturbation/no-perturbation cố định trên dev thành một thí nghiệm custom riêng.
2. **Độ chính xác của writer trên rollout X-VLA.** Ghép sự kiện writer với mốc evaluator theo cửa sổ chốt trên dev. Báo TP/FP/FN, độ trễ, coverage và số lần prompt lặp sau khi đã đủ mốc. Đọc riêng lỗi trước mốc đầu và lỗi tiếp tục thao tác sau khi đủ chu kỳ. Dùng writer Relation V1 cho kết quả chính. Writer khác có mã arm riêng.
3. **Can thiệp ngữ cảnh tại cùng trạng thái.** Trên seen/dev, cùng trọng số F1, ảnh, proprio và noise, so prompt đúng, bỏ lịch sử, lịch sử sai số đếm và prompt đối chứng khớp ngân sách token. Đo thay đổi action và outcome vòng kín từ cùng snapshot. Ghi số snapshot và seed. Tách độ nhạy action khỏi lợi ích task success.
4. **Độ bền của kết luận.** Nếu E6 có suy giảm, ưu tiên đối chứng cùng budget với và không có base replay. Nếu chênh lệch E2 tập trung ở cadence, chạy 10/30 và 30/30 trên cùng roster dev trước. Khi tài nguyên cho phép, lặp seed huấn luyện và mở rộng E6 lên 50 state/task cho các arm đã chốt.

## Thứ tự bàn giao

1. Lock E6/E2, audit checkpoint SmolVLA B0/F0/F1 V2, smoke hai backbone và bảng E6 trước train.
2. Receipt replay base/T1/T3, dataset X-VLA, kiểm chứng prompt, smoke PEFT/export và dự toán tài nguyên.
3. Hai export X-VLA F0/F1 step 5.000, lịch mẫu và log chi phí khớp nhau.
4. Bảng E6 sáu arm và bảng E2 bốn arm/backbone, delta ghép cặp, phân tích lỗi và recovery có video.
5. Đưa câu hỏi thích ứng–duy trì lên đầu phần thực nghiệm của luận văn. Trình bày X-VLA và SmolVLA cạnh nhau trong từng phép đo, tiếp theo là phân tích writer, recovery và chi phí. Tách riêng các ô còn thiếu và các pilot lịch sử trong sổ bằng chứng.
