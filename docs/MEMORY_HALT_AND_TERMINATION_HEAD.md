# Memory halt và chuẩn bị head dự đoán kết thúc (2026-11-22)

## 1. Memory halt (`memory_halt_v1`), đã hiện thực

**Vấn đề.** Benchmark LIBERO-Mem không tự dừng khi task đã hoàn thành. Nếu robot nhấc bát lên lại, episode bị tính overshoot và thất bại. Ở E2, mọi episode X-VLA đạt SCR = 1 đều bị overshoot.

**Cơ chế.** `src/evaluation/memory_halt.py` bọc policy adapter. Runner `libero_mem_rollout.py` được giữ nguyên byte, nên hash dùng trong audit so sánh chéo backbone không đổi.

Tại mỗi lần query (cứ 10 bước một lần), wrapper đọc snapshot của writer đã bind và độ mở gripper đo được. Halt được kích hoạt khi đủ cả ba điều kiện:

1. `returns >= target` (T1: 1, T3: 3).
2. Bát đang ở trạng thái `confirmed` trên đĩa và không có departure đang mở.
3. Độ mở gripper ≥ 0.055, cùng ngưỡng mà writer dùng để coi là đã nhả.

Khi điều kiện đúng liên tiếp `confirm_queries` lần thì halt được chốt cho đến hết episode.

Cơ chế này chỉ dùng memory dự đoán và proprioception đo được. Nó không đọc `_satisfied_subgoals`, cờ success hay bất kỳ trạng thái đặc quyền nào khác.

**Các mode:**
- `hold` (mặc định): không query policy nữa. Mỗi chunk sau đó giữ pose EEF đo được lúc halt, gripper mở. Chunk được tạo bằng chính decoder `decode_xvla` của tác giả từ raw chunk (30, 20). Có thể dùng `--halt-retract-m` để nâng tay lên vài cm, nội suy trong 10 bước.
- `gripper_lock`: policy vẫn chạy, nhưng lệnh gripper luôn bị ép mở.

**Cách chạy:**
```
python scripts/xvla_v2/evaluate_memory.py ... --arm F1 --memory on --halt hold \
    [--halt-confirm-queries 1] [--halt-retract-m 0.0] \
    [--halt-no-support-check] [--halt-allow-closed-gripper]
```
Kết quả được ghi thành condition riêng `F1-mem-on-halt-hold`, và comparison bị ép về `DESCRIPTIVE_ONLY`. Mỗi episode có thêm:
- `decisions.jsonl`: lý do quyết định ở từng lần query.
- `memory_halt.json`: tóm tắt halt của episode.

Run có thêm `halt_summary.json`. Schema của `summary.csv` không đổi.

**Kiểm tra offline** (`scripts/xvla_v2/replay_memory_halt.py`) trên log `xvla_e2_f1_mem_on_20261122`, gồm 40 episode:

| | T1 | T3 |
|---|---:|---:|
| Episode hoàn thành (upstream) | 12 | 4 |
| Rule halt trước overshoot | 4 | 1 |
| Rule halt sai (trước khi hoàn thành) | 0 | 0 |
| Không halt vì writer không đếm được | 7 | 3 |
| Không halt vì lúc query gripper đã đóng lại (state17) | 1 | 0 |

Khoảng cách từ lúc halt đến lúc overshoot là 22 đến 35 bước. Writer báo return chậm hơn mốc upstream từ 10 đến 35 bước.

Đây chỉ là chẩn đoán thời điểm, không phải ước lượng success, vì sau khi halt quỹ đạo sẽ khác so với log. **Nút thắt là recall của writer**, không phải rule.

## 2. Head dự đoán kết thúc: đã chuẩn bị, chưa train

**Nhãn.** `scripts/xvla_v2/build_termination_labels.py` gán nhãn `done[t] = 1` khi số mốc tích lũy theo annotation của tác giả (`metainfo.json`, parser giống `audit_relation_returns.py`) đã đạt target. Script đồng thời đọc prompt trong shard để biết từng frame mà memory đã hiển thị đủ số lần hay chưa (`done_observable_from`). Chạy trên máy:
```
python scripts/xvla_v2/build_termination_labels.py --output-dir experiments/training/termination_labels_v1
```
Script cần `h5py` và nên có `ijson`, vì `metainfo.json` nặng 681 MB.

**Dữ liệu.** `src/training/termination_data.py` gồm:
- Bung nhãn thành từng frame, chia ba loại: positive, hard negative, negative. Hard negative là lúc bát đã nằm trên đĩa nhưng task chưa xong, ví dụ sau cycle 1 và 2 của T3, hoặc ngay trước `done_from`.
- Sampler cân bằng theo tỉ lệ cố định.
- Chia train/val theo demo.
- Reader đọc ảnh và prompt tại frame t, theo cùng quy ước ảnh với `shard_handler`.

Action handler không bị thay đổi.

**Hook khi đánh giá.** `MemoryHaltPolicy(termination_fn=..., combine=...)` hỗ trợ ba cách kết hợp:
- `shadow`: chỉ ghi lại xác suất, không tác động.
- `or`: head có thể kích hoạt halt khi rule bỏ sót.
- `and`: head có thể phủ quyết halt của rule.

**Lưu ý thiết kế cần chốt trước khi train:**
- Chỉ nhìn ảnh thì không phân biệt được cycle 1 với cycle 3 ở T3, và ở T1 cũng không phân biệt được trạng thái ban đầu với lúc đã xong, vì cả hai đều là bát nằm trên đĩa. Do đó head bắt buộc phải nhận prompt memory làm input.
- Trong shard, writer hiếm khi hiện đủ số lần (theo audit, chỉ 19/98 demo T3 đạt 3 returns). Vì vậy nhãn GT `done` thường mâu thuẫn với prompt. Có hai lựa chọn:
  - `observable_only=True`: bỏ các frame "đã xong nhưng memory chưa thấy", head học một rule mềm.
  - Giữ nguyên nhãn GT: head phải tự suy ra từ hình ảnh, nhưng làm vậy thực chất là học thay cho writer.
- Đề xuất: dùng backbone ảnh nhỏ đóng băng cho hai view, kết hợp encoder text đã dùng cho prompt, cộng một MLP. Chạy ở chế độ `shadow` trên E2 trước khi cho phép `or`.

## 3. Kết quả thực nghiệm Memory Halt vòng kín và kiểm chứng Seen (2026-11-24)

### 3.1. Kết quả E2 vòng kín thật và đối chứng Placebo (40 episodes: T1 20, T3 20)
- **H0 (Không dừng):** 0/40 success (mọi lượt hoàn thành đều overshoot).
- **H3 (V4 + bản thể không chặn, chạy vòng kín thật):** 9/40 success (T1: 8/20, T3: 1/20), $p = 0{,}002$ (one-sided exact sign test). Toàn bộ 9 lượt dừng kịp lúc đều thành công; 2 lượt dừng sớm đều thất bại.
- **Oracle (đếm của simulator):** 10/40 success (trần lý thuyết bị chặn bởi nhịp truy vấn chu kỳ 10 bước).
- **Placebo halt (dừng ngẫu nhiên không bằng chứng):** 28 lượt rẽ nhánh (ranh giới truy vấn ngẫu nhiên, ranh giới có gripper mở, fixed steps 100/150/200) đạt 1--5/40 thành công, trung bình 3,7/40. Bằng chứng bộ nhớ nâng tỷ lệ thành công từ ~3,7/40 lên 9/40, sát trần oracle 10/40.

### 3.2. Kiểm chứng trên 12 trạng thái Seen chốt trước (T1 & T3, states 0--5)
- **Thiết lập:** Cấu hình V4 + bản thể không chặn, chạy vòng kín thật trên 12 trạng thái seen (chốt trước khi chạy), ghép cặp với H0 cùng seed.
- **Kết quả:**
  - H0: 8/12 lượt hoàn thành mốc (T1 6/6, T3 s0 và s2), cả 8 đều overshoot nên 0/12 success.
  - H3: dừng kịp lúc 6/8 lượt hoàn thành (T1 s1--s5, T3 s0) -> 6/12 success (Wilson 95%: 25--75%). Bỏ lỡ 2 lượt: T1 s0 (overshoot bước 115) và T3 s2 (overshoot bước 445).
  - Dừng sớm: 0/12. 4 lượt T3 chưa hoàn thành (s1, s3, s4, s5) không bị dừng, SCR giữ nguyên như H0.
  - Tỉ lệ chuyển hóa hoàn thành thành thành công: 6/8 (75%) so với 9/16 (56%) trên E2.
  - Kiểm định dấu chính xác hai phía 6/6 cặp lệch: $p = 0{,}031$ (6 có lợi, 0 bất lợi, 6 hòa).
  - Tác hại: Số lượt dừng sớm = 0/12. Phép thử chưa đo được tác hại trên lượt tự thành công vì H0 = 0/12.

