# Kiểm chứng grasp failure, replanning và memory — 18/10/2026

**Kết luận sau khi mở rộng kiểm tra:** đã rà soát video và logs của cả 15 episode thất bại trong run no-memory `libero-10-wsl`. Xác nhận hai case rõ về mất vật rồi tiếp tục đi về phía đích và phát lệnh mở gripper; hai case khác có biểu hiện tương tự nhưng chuyển động/điểm mở chưa đủ rõ để xếp chung. Các recording còn lại có nhiều kiểu lỗi khác. Không có bằng chứng để quy mọi grasp failure về cùng hành vi này. Một action chunk cũ cũng không giải thích đầy đủ các case đã xác nhận, vì hành vi tiếp diễn qua nhiều fresh inference. Chưa chạy counterfactual s5/s2; chưa biết giảm horizon có ngăn lỗi tiếp xúc ban đầu không. Không cần triển khai RTC để trả lời câu hỏi hiện tại.

## Phạm vi và nguồn

- Kiểm tra source rollout, simulator, adapter, observation writer và installed official SmolVLA implementation.
- Đối chiếu video, `model_output.jsonl`, `diagnostics.json`, `episode.json` trong các run hiện có.
- Trích frame phục vụ phân tích; không chạy simulation, inference, training hoặc thay source production.
- Các failure video có 1001 frames cho 1000 actions: frame 0 là reset; frame N là observation sau action N−1 / trước action N. Không đồng nhất frame index với action-step index.
- Screening bằng frame lấy mẫu trên toàn chiều dài video, sau đó đọc dày hơn quanh grasp/transfer nghi vấn. Không phải annotation mọi frame/mọi attempt. "Không xác nhận" không đồng nghĩa chứng minh sự kiện không thể xuất hiện trong một đoạn ngắn chưa đọc đủ dày.
- Các kết luận chỉ áp dụng cho recording được kiểm tra; chưa ước lượng tỷ lệ lỗi của toàn LIBERO-10 hoặc mọi run có memory. Chọn failure để phân tích cũng không cho biết tỷ lệ recovery trên tất cả các lần gắp trượt, vì episode success có thể chứa retry.

## Kiểm tra toàn bộ failure trong run no-memory

Inventory: 50 episode, 35 success, 15 failure. Cả 15 failure có video. Có 8 failure từng đạt ít nhất một mục tiêu đặt vật không được thỏa sẵn khi reset; chỉ 7 còn giữ mục tiêu ấy ở cuối. Task6/init3 đạt rồi làm mất trạng thái cốc trên đĩa. Không dùng số predicate đã đạt để tự suy thứ tự gắp hoặc nguyên nhân thất bại.

[Inventory có source hashes](D:/Thesis_26/experiments/audits/failed_grasp_20261018/failure_inventory.json).

Quy ước: **Rõ** = quan sát được mất vật/tay rỗng, sau đó chuyển động về khu vực đích và có open command; không khẳng định gripper vào chính xác vùng success. **Tương tự/chưa chắc** = có mất vật và mở/di chuyển cùng hướng đích, nhưng khó tách khỏi withdrawal cục bộ. Các hàng còn lại nêu pattern quan sát được, không khẳng định absence tuyệt đối.

| Task / init | Kết quả kiểm tra | Evidence quan trọng |
|---|---|---|
| 0 / 0 | Sai vật + local attempts | Đưa carton orange juice vào rổ khoảng 210–430; hai target chính thức chưa đạt. Sau đó gắp hụt quanh vật tại nguồn. Không xác nhận empty goal transport. |
| 0 / 2 | Local attempts / departure | Nhiều close/lift/open gần nguồn rồi lệch ra trái. EEF y < 0 trong mọi log, basket y khoảng +0.251; khoảng cách XY nhỏ nhất tới tâm basket khoảng 0.270 m ngay reset. Không xác nhận đưa tay rỗng tới rổ. |
| 4 / 0 | **Rõ, vật thứ hai** | Cốc đầu đạt 93. Cốc sau tuột 187–189; fresh query 190; open 196, tiếp tục về đĩa. |
| 4 / 1 | **Tương tự/chưa chắc, vật thứ hai** | Cốc đầu đạt 92. Cốc sau tách khỏi tay khoảng frame 205; replan 210, open 214; tay dịch về phía đĩa nhưng mở trước khi rõ ràng tới đích. |
| 4 / 2 | **Tương tự/chưa chắc, vật đầu** | Chưa goal nào đạt. Close 328; cốc tách khỏi tay khoảng frame 347; replan 350, open 352; tay đi trái về phía đĩa. Không xác nhận release ngay trên đĩa. |
| 5 / 1 | Giữ vật, insertion/placement fail | Sách được giữ trong các đoạn quan sát khi cố đưa vào caddy. Mọi command 71–999 đều close. Không phải empty transport. |
| 6 / 1 | Local misses / stall ở vật tiếp theo | Cốc đạt 132; pudding close 194 rồi open 226, retry close 246/open 269. Có fresh queries nhưng không xác nhận empty transport tới đích. |
| 6 / 3 | Release stall + local miss | Cốc từng đạt 112 nhưng không còn đạt cuối episode. Stall quanh cốc rất lâu; pudding miss khoảng 928–957, local open rồi retry 980. |
| 7 / 2 | Local misses | Gắp hụt quanh soup/cream cheese; không target nào đạt. Khoảng cách XY nhỏ nhất tới tâm basket khoảng 0.200 m; không xác nhận empty goal transport. |
| 8 / 0 | Giữ vật rồi placement fail | Pot2 đạt 178. Pot1 được giữ và chuyển thật ở frames 310–350, tới cạnh stove rồi mở 378/387/395/402; đặt không đạt, sau đó pot nghiêng/đổ. |
| 8 / 1 | Placement fail + local miss | Chưa goal nào đạt. Vật đầu mang tới stove nhưng rơi cạnh stove. Vật tiếp theo close 886, local empty lift, open 926/931 rồi quay lại; chưa xác nhận tới stove rỗng. |
| 8 / 2 | Local misses ở vật thứ hai | Pot2 đạt 203. Pot1 nằm ngang; close 672–681, 703–712, 730–745, open 682/713/746 tại vùng nguồn. Fresh query 680/710/740. |
| 8 / 3 | Approach/topple/departure | Pot2 đạt 181. Pot1 bị làm nghiêng trong lúc gripper open, sau đó tay đi khỏi vùng vật. Mọi command 185–909 đều open. |
| 8 / 4 | Approach/contact stall | Pot2 đạt 192. Tiếp cận pot1 rồi hover/contact kéo dài; mọi command 195–999 đều open. Không có sustained second grasp attempt. |
| 9 / 2 | **Rõ, task chỉ có một target mug** | Cốc nhấc rồi rơi về bàn tại 117; frame sau step 120 cho thấy tay rỗng. Fresh queries 120/130/140/150/160; open 164 gần microwave. Chưa xác nhận tay vào đúng heating region. |

Như vậy, trong screening này có **2 case rõ, 2 case tương tự/chưa chắc, 11 case chưa xác nhận pattern ấy**. Đây là kết quả phân loại bằng chứng của recording, không phải tỷ lệ thống kê chính xác của mọi failed grasp. Đặc biệt, local missed attempts trong task8/init2 là ví dụ trực tiếp cho kiểu phản ứng khác: thử lại gần vật thay vì đưa tay rỗng tới đích.

Artifacts theo nhóm: [task4/6](D:/Thesis_26/experiments/audits/failed_grasp_20261018/batch_task4_6/findings.json), [task8](D:/Thesis_26/experiments/audits/failed_grasp_20261018/batch_task8/FINDINGS.md), [các task còn lại](D:/Thesis_26/experiments/audits/failed_grasp_20261018/batch_other/FINDINGS.md).

## Case bổ sung rõ: task9/init2

Task `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` chỉ yêu cầu một target mug, nên failure này không phải vấn đề riêng của vật thứ hai. Logs cho thấy mug z tăng từ khoảng 0.9001 lên 0.9566 tại 115 rồi về 0.9002 tại 117. Video cho thấy mug ở bàn và tay rỗng; nhiều fresh queries sau đó vẫn dẫn đến chuyển động về phía microwave rồi open 164. Không suy internal belief "đã grasp thành công" từ hành vi này.

[Chuỗi ảnh và log liên quan](D:/Thesis_26/experiments/audits/failed_grasp_20261018/batch_other/t9_i2_detail.jpg).

## Case đã xác nhận: no-memory, task4/init0

Nguồn: `experiments/results/libero-10-wsl/libero_10_task4/init_0/`.
Task: `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`.
Điều kiện: memory off, execution horizon10, prediction horizon50,20Hz.

| Mốc | Bằng chứng |
|---|---|
| Step93 | Official predicate `on(porcelain_mug_1,plate_1)` thành true; cuối episode vẫn true |
| Step176 | Bắt đầu close trong lần tiếp cận cốc vàng/trắng |
| Video frames176–185 | Cốc thứ hai nghiêng và nhấc lên thoáng qua |
| Video frames188–190, tức sau steps187–189 | Cốc tách khỏi tay và rơi xuống bàn |
| Step190 | `is_replanned=true`, `chunk_step=0`; inference mới dùng observation sau step189 |
| Step196 | Phát lệnh open `-0.9703` trong lúc tay đi về đĩa bên phải |
| Step200 và202 | Lại fresh inference200; sau đó phát open tại202 |
| Kết thúc | Goal cốc thứ hai chưa từng thành công |

Telemetry corroboration: gap sau step188 khoảng3.15mm, sau190 khoảng2.4mm; chỉ riêng gap không chứng minh grasp, nên kết luận dựa thêm vào ảnh/video. Lệnh open không đồng nghĩa fingers đã mở ngay cùng timestep.

Raw references: `model_output.jsonl:191` (step190), `:197` (step196), `:201` (step200), `:203` (step202); `diagnostics.json:32` (milestones).

Contact sheet có metadata/frame alignment/video SHA256:

![Case xác nhận](D:/Thesis_26/experiments/audits/failed_grasp_20261018/task4_init0_verified.png)

Metadata: `experiments/audits/failed_grasp_20261018/task4_init0_verified.json`.
Ảnh chi tiết12frames: `experiments/results/trajectory_audits/verification_20261018/task4_init0_detail.jpg`.

**Diễn giải giới hạn:** case này là grasp không bền/tuột sau khi đã có tương tác, không phải một lần close hoàn toàn không chạm vật. Nó xác nhận hành vi tiếp tục về đích sau lỗi grasp qua fresh query. Không đủ để kết luận model có một internal belief rằng đã grasp thành công, hoặc chỉ cần thêm memory là recovery được.

## Cases bổ sung và giới hạn diagnostics

- Task6/init1 no-memory: cốc đầu đạt predicate tại132; nhiều lần gắp pudding không thành công, qua replans200/210/220 và các lần sau. Có failure ở vật tiếp theo, nhưng không dùng case này để khẳng định empty transport tới goal.
- Task6/init3: một predicate đạt tại112 nhưng tay còn thao tác/hover quanh cốc đầu. Vì vậy một predicate thành công không tự chứng minh đang gắp vật thứ hai.
- Task0/init0: video cho thấy đưa carton màu cam vào rổ rồi gắp hụt lặp lại quanh lon. Hai goal chính thức chưa hoàn thành; không dùng case này làm bằng chứng đúng vật đầu đã xong.
- `contact_diagnostics` theo một target được resolver chọn, không theo thứ tự vật thực sự đã thao tác. Không dùng `both_fingers_contact=false` của target ấy để suy tất cả vật đều không được giữ.
- `trajectory.npz` chỉ có actions và robot states, không đủ reconstruct per-object history mà không replay simulator. Video lưu agentview, không lưu đầy đủ tensor hai camera mà policy đã nhận; chưa xác minh mức độ dễ thấy lỗi trong wrist view hoặc representations của model.

## Những thông tin đã xác minh lại

1. Rollout đồng bộ: inference tại `rollout.py:469` hoàn tất trước `env.step` tại521. Không có background physics stepping; latency thực tăng wall time, không tự làm simulation chuyển động trong lúc chờ.
2. H50 khác s10. Ở20Hz, s10 cho0.5s simulation giữa các lần query; controllerOSC vẫn có feedback bên dưới. Không gọi toàn robot là open-loop.
3. Writer cập nhật sau mỗi step, nhưng memory chỉ được policy đọc ở lần replan kế tiếp; chưa có event-triggered interruption.
4. Không phát hiện queue cũ vô tình tồn tại qua episode. Runner gọi trực tiếp `predict_action_chunk`; checkpoint `n_obs_steps=1`, `rtc_config=null`.
5. `use_cache=true` là cache nội bộ lần denoise; prefix cache khởi tạo lại trong mỗi `sample_actions`. Đây không phải explicit episode memory.
6. Writer chỉ có slip khi `_is_grasped`/`_is_lifted` đã true (`observation_writer.py:213`). Close hụt hoàn toàn mà chưa xác nhận grasp không tạo failed-attempt event. Multi-target còn bị gán unidentified rồi lọc khỏi text.
7. Những gaps này thuộc khả năng nhận biết outcome và truyền thông tin qua thời gian. Chúng không tự chứng minh thiếu memory là nguyên nhân duy nhất; perception, action precision, distribution drift và thiếu recovery skill vẫn là giả thuyết.

## Giảm horizon, dynamic horizon và RTC

- Giảm s: query thường xuyên hơn trong rollout đồng bộ; có thể giúp tránh/nhận ra contact error sớm, có thể tăng discontinuity và chi phí.
- Dynamic s: chọn thời điểm query tùy state/history/event; có thể chạy hoàn toàn đồng bộ. Không đồng nghĩa RTC.
- RTC: inference bất đồng bộ trong khi chunk trước đang thực thi, freeze committed prefix và inpaint phần còn lại. Nguồn gốc: [Real-Time Execution of Action Chunking Flow Policies, v2](https://arxiv.org/abs/2506.07339v2).
- Trade-off phản ứng/nhất quán khi thay horizon có cơ sở nghiên cứu: [Adaptive Action Chunking, v2](https://arxiv.org/abs/2604.04161v2). Không suy hiệu quả của checkpoint hiện tại từ kết quả bài báo.

## Quyết định nghiên cứu đề xuất

**Ưu tiên memory về kết quả tương tác, giữ dynamic horizon như đối chứng hoặc extension.** Cần phân biệt `close_commanded`, `attempt_in_progress`, `holding_unverified`, `holding_supported`, `failed_attempt`, `contact_lost`. Những trạng thái này phải xuất phát từ evidence, có unknown/confidence; không biến command thành fact thành công.

Kết quả rà soát 15 failure làm rõ giới hạn của hướng này: policy đã có hành vi local retry trong một số case, và nhiều lỗi là approach hoặc placement khi vẫn giữ vật. Không thể mặc định mọi failure do thiếu memory hoặc model luôn chuyển pha bất chấp kết quả grasp. Memory phải được đánh giá trên khả năng phân biệt outcome và cải thiện recovery cụ thể; kết quả kiểm tra hiện tại không chứng minh fine-tuning là tất yếu hoặc chỉ fine-tune action expert là đủ.

Memory có thể giữ mốc attempt, observation/features trước-sau, chuyển động tương đối vật/tay, kết quả và số lần thử. Không bắt buộc gán objectID để nhớ một attempt vừa thất bại; nhưng identity vẫn cần cho per-object progress. Chỉ gripper gap/proprioception không đủ bảo đảm binary holding đúng cho mọi vật.

Một outcome monitor chỉ phát hiện lỗi, không tự tạo corrective action. Nếu frozen policy vẫn dự đoán transport sau fresh query, requery cùng policy có thể tiếp tục thất bại. Cần kiểm tra khả năng dùng outcome context; nếu adaptation, supervision cần successful corrective actions từ failed approaches, không imitate các failed actions.

Các bước thực nghiệm trên task LIBERO không đổi:

1. Diagnostic sweep no-memory s10/s5, tùy kết quả thêm s2; H50, policy, instruction, preprocessing và environment-step budget cố định. `run_benchmark.py` chấp nhận s2; CLI `run_episode.py` hiện không có choice2. Same seeds không tạo cùng noise trajectory khi số query thay đổi.
2. Phân biệt acquisition success, loss-of-grasp, continued-empty-transport và recovery. Không chỉ nhìn episode success.
3. Memory treatment đầu tiên giữ fixed s. So current-observation-only outcome estimator với history-based estimator, cùng observation sources, readout/recovery mechanism và training budget; ablate lịch sử.
4. Khi có evidence outcome memory hữu ích, mới xét history-triggered dynamic horizon. Đối chứng với fixed small s và current-state trigger, có chi phí inference tương đương.
5. Giữ original task/initial states/success predicates; chọn hyperparameters trên development states, báo kết quả trên held-out states. Simulator truth chỉ dùng ở annotation/evaluation hoặc privileged condition riêng.

Chưa xác nhận: s5/s2 có tăng SR hoặc ngăn slip không; tỷ lệ empty transport toàn suite; model có thật sự bỏ qua cue trong ảnh/state không; outcome memory có giúp recovery không; cần fine-tune expert hay cả backbone. Các điểm này cần intervention, không thể kết luận từ video/logs quan sát đơn thuần.
