# Đối chiếu lượt chạy và nguồn hình Chương 4

## T3 lịch sử, trạng thái test 2

Nguồn hình là hai video 256 × 256, 20 fps, mỗi video gồm 1.001 khung của lượt `T3_F1_test_state2`, checkpoint F1 lịch sử 3.000 bước, writer `rgb_cycle_v3`, seed 42. Mã băm trạng thái khởi tạo `demo_29` của F1 và đối chứng B0 là `462a59bc3e3f73a9af523c41a0fa2dedcc8b1beec97af122881683b02e052ef7`. Đường dẫn và SHA-256 của video, log, hình và các bản kê đã kiểm tra nằm ở mục `t3-success-video` của [sổ bằng chứng](claim_ledger.json).

Bộ ghi của runner ghi một khung ở quan sát ban đầu và sau mỗi action. Vì vậy chỉ số khung đếm từ 0 trùng với bước quan sát sau làm nóng. Dùng FFmpeg chọn theo chỉ số khung ở cả hai video, giữ hướng ảnh, rồi cắt theo tọa độ pixel `(trái, trên, phải, dưới)` là `(5, 20, 245, 200)` cho camera ngoài và `(15, 25, 255, 205)` cho camera cổ tay. Ghép theo thứ tự bước 190, 626, 863, 1.000, cách nhau 8 pixel. [Video camera ngoài](../../../experiments/evaluation/results_mem_2/step5_test_20261109-093744/test_F1/episodes/T3_F1_test_state2/agentview_image.mp4) · [Video camera cổ tay](../../../experiments/evaluation/results_mem_2/step5_test_20261109-093744/test_F1/episodes/T3_F1_test_state2/robot0_eye_in_hand_image.mp4) · [Hình trong khóa luận](../figures/libero_mem_t3_success.png)

| Bước bộ chấm | Thời gian video | Mốc tích lũy | Xác nhận của bộ ghi và lời nhắc |
| ---: | ---: | ---: | --- |
| 190 | 9,50 s | 1/3 | Sự kiện có thời điểm ước lượng 196, khả dụng ở 198. Truy vấn 200 lần đầu nhận cận dưới 1. |
| 626 | 31,30 s | 2/3 | Chưa có xác nhận mới. Lời nhắc vẫn ghi cận dưới 1. |
| 863 | 43,15 s | 3/3 | Sự kiện có thời điểm ước lượng 880, khả dụng ở 882. Truy vấn 890 lần đầu nhận cận dưới 2. |
| 1.000 | 50,00 s | 3/3 | Kết thúc thành công. Bộ ghi xác nhận 2 và báo lịch sử chưa đầy đủ. |

Đếm mốc theo `info.eval_result` trong `observations.jsonl`. Lấy thời điểm sự kiện và khả dụng từ trường `events` trong `memory_trace.jsonl`, rồi đối chiếu nội dung thực sự cấp ở `queries.jsonl`. Lượt B0 cùng trạng thái kết thúc tại bước 1.000 với 0/3 mốc. Cả hai bản tóm tắt trùng mã băm trong `status.json` tương ứng.

## E2 và E6

- E2 dùng đúng 80 bản `summary.json` có trạng thái hoàn tất và khớp SHA-256 trong mười thư mục `batch-*`. Ghép theo tác vụ, băm trạng thái và seed. Cột chưa đạt mốc đếm các lượt có `completed_subgoals=0`. Các mốc của cặp T3 `demo_108` và T1 `demo_117` đọc từ `observations.jsonl` của lượt gốc. [Tổng hợp E2](../../../experiments/evaluation/e2_unseen_author_v2/paired_e2_final.json) · [Can thiệp lời nhắc T1](../../../experiments/evaluation/prompt_counterfactual_state16_20261118_followup/combined_report.md)
- E6 dùng bản ghi `episode.json` của `B0_task_*` và `F1_actual_task_*`, ghép theo tác vụ, ID trạng thái và seed. Chẩn đoán tác vụ 5 lấy từ bản kê cùng thư mục. [B0, tác vụ 5](../../../experiments/results/e6_retention_locked_50/B0_task_5/benchmark_summary.json) · [F1-off, tác vụ 5](../../../experiments/results/e6_retention_locked_50/F1_actual_task_5/benchmark_summary.json)

## Kiểm tra kích hoạt truy vấn theo bằng chứng

Mười `run_manifest.json` của E2 đều ghi `arguments.decision_mode=null`, không có `decisions.jsonl`, và dùng `execution_horizon=10`. Vì vậy, dù F1 có nhận prompt memory, các lượt E2 chỉ truy vấn theo lịch cố định; chúng không kích hoạt cơ chế truy vấn sớm theo thay đổi đã được memory xác nhận. Kết quả E2 không được diễn giải là thử nghiệm duyệt chủ động dựa trên bằng chứng. [Audit kèm hash của cả mười manifest](e2_decision_mode_audit.json).

Lượt thử `decision_evidence_text_t1_20261120_wsl` dừng ở quan sát ban đầu do detector RGB timeout sau 300 giây, trước action đầu tiên (`completed_episodes=0`); không tính là kết quả điều khiển.

Replay shadow offline trên 40 trace F1 của E2 dựng lại chính xác **3.969/3.969 prompt** bằng tokenizer đã pin. Trên các quỹ đạo cố định đó, bộ giám sát nhận diện 70 thay đổi có evidence đủ điều kiện xin query sớm ở 38/40 episode (19/20 mỗi task); 38 thay đổi đầu tiên xuất hiện ở bước 3 khi fact `bowl: 0 observed returns.` được chiếu vào bộ nhớ. Năm thay đổi còn lại trong nhóm 70 là tăng số lượt quay về quan sát được: bốn ở T1 state 4, một ở T3 state 7. Các đổi trạng thái này có đủ evidence ID trong snapshot. Phép replay không chạy lại policy, không phát query và không đổi action; sau một query giả định đầu tiên, các thay đổi tiếp theo cũng không còn là dự đoán đáng tin vì hành động sẽ rẽ nhánh. Đây là phân tích coverage hồi cứu ngoài protocol D2 (E2 là split unseen, D2 quy định seen), không phải 70 requery thật hay bằng chứng hiệu quả. [Receipt replay](e2_projected_shadow_replay_20261120.json) · [Script tái lập](../../../scripts/analyze_e2_shadow_replay.py).

Lượt `decision_evidence_event_t1_20261120_wsl` hoàn thành một episode development trên T1, seen `demo_1`, state 0, seed 42. Event supervisor ghi một quyết định `requery` ở bước 3 sau tiền tố ba hành động: fact `bowl: 0 observed returns.` vừa khả dụng ở quan sát 2, gắn với event `T1_F1_seen_state0:0` và evidence RGB/gripper. Episode dùng hết 1.000 hành động nhưng không hoàn thành subgoal (SCR 0, không overshot).

Một lượt periodic cùng task/state/seed cũng hoàn tất 1.000 hành động và đạt 0/1 subgoal. Event có 101 query, gồm một requery sớm, và wall time 1.144,01 s; periodic có 125 query và 1.229,75 s. Tuy nhiên, cadence khác nhau theo protocol (event: 10 action sau requery; periodic: 8 action), nên số query/wall time chỉ mô tả hai run này, không cô lập tác động của trigger. Hai video đều cho thấy robot không nhấc bát khỏi vùng đĩa ban đầu. Đây là một cặp development cùng state/seed, có override nguồn writer được ghi log; không phải bằng chứng lợi ích hoặc điểm LIBERO chính thức. Giữ ngoài phép so hiệu năng chính. [Manifest, trace, summary, video và frame có hash](d2_event_mode_pilot_20261120.json).
## D2 paired shadow/event cohort (completed 21/11/2026)

Two F1 arms ran on T1/T3 seen-development states 0–11, seed 42, 24 episodes per arm, with the same checkpoint, split lock, runtime, writer, and base query cadence. The analyzer checked the exact locked roster, episode identity, paired initial-state hashes, and every declared episode artifact hash. All 24 pairs matched on initial state.

Shadow recorded 1/24 task successes, mean SCR 0.111, 97.17 queries/episode, and no requery. Event also recorded 1/24 successes, mean SCR 0.167, 99.88 queries/episode, and 2 overshoots. Its supervisor emitted 56 requery decisions in 22/24 episodes; all 56 carried event IDs and evidence IDs. Two episodes had no requery. Paired success difference was 0; mean SCR difference was +0.056 with state-group bootstrap 95% interval [-0.125, 0.250]. T1 had one event-only success and one shadow-only success; T3 had no terminal success in either arm. Treat this as a development mechanism check, not evidence that event retrieval improves task success. [Paired analysis JSON](../../../experiments/evaluation/decision_dev12_t1_t3_20261121_paired_analysis.json) · [Analyzer](../../../scripts/analyze_d2_decision_pair.py) · [Shadow manifest](../../../experiments/evaluation/decision_shadow_dev12_t1_t3_20261120_wsl/run_manifest.json) · [Event manifest](../../../experiments/evaluation/decision_event_dev12_t1_t3_20261120_wsl/run_manifest.json).

## T1 event-mode successful rollout video crop

The current-cohort successful example is T1 event-projected state 7, source demo_8, seed 42, initial-state SHA-256 04fd858d13c1cf680de900d219b6208a764e6886153ebda86f89bc580666941d. The evaluator recorded the one required subgoal at step 409 and terminal success at step 1,000, without overshoot. The run issued four evidence-backed semantic-change requeries at steps 3, 142, 180, and 312, and used 102 policy queries. It is one successful episode in the development cohort, not a causal comparison.

The figure uses frame indices 0, 142, 409, and 1,000 from each 256 × 256 video at 20 fps. The agent-view crop is pixel box [5, 20, 245, 200]; the wrist-camera crop is [15, 25, 255, 205]. Video hashes are 651987c8a58d77b08a0da97fd837fe8a8a1b9077f5db8999f4c2569a5bf7168b (agent view) and d8652375d2d00153aa1ae5873464050c3ff1f7f45730ada544f49c6acbdd3525 (wrist view); the contact sheet SHA-256 is bb749b5fcf541d0450edc655c9023355fc6f63f6f4aafb77e586d054dba6a399. [T1 event success receipt](d2_t1_event_success_video_20261121.json) · [Agent-view video](../../../experiments/evaluation/decision_event_dev12_t1_t3_20261120_wsl/episodes/T1_F1_seen_state7/agentview_image.mp4) · [Wrist-camera video](../../../experiments/evaluation/decision_event_dev12_t1_t3_20261120_wsl/episodes/T1_F1_seen_state7/robot0_eye_in_hand_image.mp4) · [Figure](../figures/libero_mem_d2_event_t1_state7_success.png).
