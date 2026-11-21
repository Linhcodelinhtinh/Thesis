# Trạng thái và các tuyên bố được phép viết

[Định danh checkpoint, writer và số liệu từng đợt](EXPERIMENT_CONFIGURATION_REGISTER.md). Relation V3 motion đạt 195/394 mốc development (T1 66/100, T3 129/294) và E2 replay 0 TP, 1 FP, 3 FN. RGB Cycle lịch sử là `rgb_cycle_v3`, dùng trong cấu hình F1 lịch sử đạt 1/20 thành công. Hai phép đo có đơn vị và roster riêng.


Mốc rà soát: **21/11/2026**, theo working tree, [claim ledger](evidence/claim_ledger.json) và các receipt ghi trong sổ. [Snapshot máy đọc được](evidence/snapshot.json) là mốc provenance lịch sử; working tree hiện có thay đổi. Mọi kết luận kỹ thuật ràng buộc vào artifact/hash và phạm vi nêu dưới đây. Không phát sinh huấn luyện model mới trong đợt hoàn thiện gap này.

## 1. Bốn mức bằng chứng

- **S — Source:** đọc được implementation/config; chưa khẳng định runtime.
- **A — Artifact:** có output của một lần chạy, checkpoint hoặc log; cần xác định đúng nguồn và phạm vi.
- **V — Verified in scope:** đã đối chiếu những thuộc tính cụ thể, ví dụ hash export hoặc thống kê episode. Không đồng nghĩa đã kiểm tra lại toàn runtime.
- **H — Hypothesis:** mục tiêu/giả thuyết chưa được kiểm định đủ.

## 2. Ma trận hoàn thiện

### Trạng thái hiện hành sau 21/11

| Gói | Trạng thái | Kết quả đã xác minh | Chưa được xem là hoàn tất |
|---|---|---|---|
| P0/P7 · Bản viết | Đã biên tập Chương 1–5 theo 3 RQ; abstract có SHA-256 lock; validator/ledger pass | Sơ đồ kiến trúc, timeline khái niệm, public provenance; PDF 51 trang trên đĩa (SHA-256: `b4263cf2578bf2459e8f3dec86548a5bde95e778ab8760e5e9da726af13543fc`), render toàn bộ và rà trực quan | Không có layout gate còn mở; các nội dung tác giả tự xác nhận vẫn cần họ đọc trước khi nộp |
| P1 · Nguồn và dữ liệu | Split V2, refined source, index T1/T3, shards và alignment audit; LIBERO source pin cài trong venv Python 3.8.13 | 698 demo; 180.447 hàng state/action; 2.094 RGB frame/camera; 1.396 parity mẫu; strict validator pass; smoke 14/14 | EGL dùng fallback `kms_swrast`; smoke không phải policy score; MiniVLA audit đã chốt blocker, chưa qua runtime gate |
| P3/P4 · Writer và profile | T1/T3 writer cũ có audit riêng; profile T1–T10 tồn tại; raw/refined T7/T8/T9/T10 source receipts được hash-check local | T7/T8/T9/T10 RGB `demo_1` không cấp fact memory; T7 exo-box audit ở IoU 0,5 cho bowl P/R 0,893/0,747 và plate 0,971/0,646. Author-action replay upstream một demo/task T8/T9/T10 ghi T9/T10 terminal success; T8 raw/refined action/state đầu vào trùng nhưng robot-state replay vượt `1e-5` ở bước 219, có 3 milestone và không terminal | Chưa có nhãn blind identity/relation/containment; writer diagnostics không phải accuracy hoặc policy score; nguyên nhân T8 drift chưa rõ |
| P5 · Dữ liệu và adapter | Shards V2 và alignment pass; cặp adapter F0 (no-memory) và F1 (rgb_relations_v1) V2 5.000 bước đã hoàn tất export, strict reload, và đã hoàn thành rollout E2 (40 cặp / 80 episodes trên 20 unseen states T1/T3) cùng E6 retention (50 cặp / 100 episodes trên 10 LIBERO-10 tasks) | Inference env Python 3.12.3/CUDA 12.8; runtime check đạt F0 (4/4) và F1 (2/2); training collator khớp hash artifact; rollout E2 unseen và E6 retention đã có kết quả định lượng đầy đủ | Package source hash processor phía training không được lưu (nhưng recipe, hyperparams và commit LeRobot 0.6.1 đã pin cố định) |
| P6 · Policy và cơ chế quyết định | E0 pilot 4/4; E2 F0/F1 matched 40 cặp unseen T1/T3; E6 retention 50 cặp LIBERO-10; D1 parity 2 cặp; D2 closed loop 24 cặp; B0 raw 40/40; D3/D5 cycle-count alignment mô tả trên 48 trace D2; E4/E5 one-query diagnostic trên hai state thành công | E2: F0 4/40, F1 0/40; E6: B0 37/50, F1-off 23/50; D2: hai arm 1/24; event 56 requery có evidence ID, trong đó 41 thêm ID mới và 15 giữ ID cũ sau khi gắn `History incomplete`; D3/D5: endpoint count khớp event T1 10/12, T3 8/12; shadow T1 6/12, T3 10/12; E3: renderer delivery 2.397 query; E4/E5: state7 q142 và state5 q10 đều có replay control success 1/1, khớp 1.000/1.000 action; hai nhánh bỏ memory đều 0/1 success sau prefix khớp lần lượt 142/142 và 10/10; B0 raw 0/40 chỉ mô tả do runtime/source mismatch | Alignment D3/D5 chưa có nhãn blind identity/relation/intent và không phải accuracy; E3 chưa đo budget policy; E4 còn thiếu nhiều mức delay/noise và risk–coverage; E5 mới hai counterfactual đơn state, chưa ước lượng efficacy; multi-seed/heldout; derived v1 bị chặn ở hash preflight; nguyên nhân chênh lệch hiệu năng chưa xác định |
| Backbone thứ hai | MiniVLA checkpoint bytes hash-match và official openvla-mini source commit đã pin (`0822b36227b5a771be4eb2680e34c559734c8fdc`); runtime contract đã nghiệm thu (`VERIFIED`) với runtime parity receipt trên CUDA | Official loader (`load_vla`), DinoSigLIP transform, LANCZOS 224x224, prompt builder, quantile unnormalization và Robosuite gripper inversion đã kiểm tra runtime parity thành công (`minivla_libero90_runtime_receipt.json`) | Candidate VQ thiếu checkpoint/codebook cục bộ; chưa triển khai rollout so sánh đa backbone trên toàn bộ LIBERO benchmark |

Bảng chi tiết dưới đây lưu góc nhìn của snapshot 12/11 và các ghi chú lịch sử; khi khác với trạng thái hiện hành ở trên, dùng bảng mới này cùng snapshot ngày 15/11.

| Thành phần | Trạng thái đúng | Bằng chứng chính | Viết được | Còn thiếu / tạm hoãn |
|---|---|---|---|---|
| V1 baseline không memory | A; frozen theo báo cáo cũ; raw tham chiếu chưa tìm thấy | `experiments/baseline_v1/{V1_BASELINE_REPORT.md,freeze_audit_signoff.json}` | Mục tiêu, thiết kế và lịch sử freeze | Không trình bày 77/160 như đã kiểm lại; cần phục hồi đúng raw nếu dùng bảng V1 chính |
| Các baseline/sft lịch sử khác | V về số file và đếm outcome; chưa chứng nhận so sánh | `historical_run_inventory.csv`, 1.520 `episode.json` | Danh mục tiến độ, chọn candidate audit tiếp | Không gộp các run, không coi số file tìm thấy là planned sample size hoàn tất |
| SmolVLA adapter | S+A; đã được dùng trong rollout tháng 11 | Model manifest, `src/models/smolvla/adapter.py`, run manifests | Pipeline state 8/action7/chunk 50, camera, normalization | Không coi manifest version chung là resolved runtime của mọi run |
| MiniVLA / OpenVLA | V ở mức runtime parity receipt; candidate B provenance và official Prismatic runtime đã audit/nghiệm thu; candidate VQ thiếu local checkpoint/codebook | `src/models/minivla/`, model manifests, contract audit report, `minivla_libero90_runtime_receipt.json` | Nêu kết quả nghiệm thu runtime parity, thông số trễ và tính tương thích interface | Rollout benchmark quy mô lớn trên toàn bộ 90 task, candidate VQ |
| Simulator–policy–perception bridge | S+A; đã chạy thật cho B0/F1 | Worker/IPC code, environments và episode status tháng 11 | Kiến trúc tách tiến trình, allowlist, source lock | Không nói policy chạy trong venv riêng hoàn hảo: manifest ghi Python host Windows cho policy, WSL venv cho simulator, venv riêng cho detector |
| RGB Cycle lịch sử (`rgb_cycle_v3`) | S+A; trace rollout F1 lịch sử và audit demonstration riêng | `object_tracker.py`, `event_writer.py`, `experiments/audits/memory_writer_v3/verification_summary.json`, F1 memory traces | 12/12 return ghép milestone trong ±20 frame trên sáu demo đã xem | Chỉ ba nhóm state; milestone không phải contact label |
| Relation V3 motion (`rgb_relations_v3`) | A+V ở mức replay ngoại tuyến | [Receipt và cấu hình motion](../RELATION_WRITER_V3_MOTION_RESULTS_2026-11-18.md) | Dev 195/394, 0 FP; E2 0 TP, 1 FP, 3 FN; cached p95 1,53 ms | E2 đã dùng chọn cấu hình; chưa có rollout F1 với writer này |
| Renderer | S+A | `libero_mem_text.py`, `queries.jsonl`/prompt history | Budget toàn prompt 48 token, giữ instruction, privilege gate | Trạng thái causal/nguồn hợp lệ không chứng minh nội dung đúng hoặc model sử dụng hữu ích |
| Split và dataset tác giả | A+V ở mức manifest/training identity | [Định danh split và corpus](EXPERIMENT_CONFIGURATION_REGISTER.md) | Lịch sử: 698 tổng, 558/70/70 train/val/test. V2: 698 tổng; 598 train (400 base + 198 Mem), raw unseen tách riêng | Holdout của fine-tune; độ phủ dữ liệu pretraining chưa xác minh |
| Huấn luyện F1 lịch sử vs V2 | A+V | Lịch sử: `mem-adapter-20261105-083543` (step 3000, `rgb_cycle_v3`); V2 của E2: `results_f1_author_split_v2/output_f1` (step 5000, `rgb_relations_v1`) | LoRA attention q/v, rank 16, 645.120 tham số; 2×T4; loss validation 0,4048 (base replay) | Step 5000 validation loss chỉ đo base replay; loss không phải SR; hiệu quả memory cần rollout vòng kín |
| F0 V2 no-memory adapter | A+V; artifact đã được kiểm manifest, strict reload và runtime check | `results_f0_author_split_v2/output_f0/pretrained_model/mem_adapter_manifest.json`; run manifest E0 | Adapter LoRA no-memory, 5.000 bước, dùng cùng V2 data lock | Không diễn giải F0 so với B0 thành hiệu ứng memory; cần đối sánh cặp với F1 V2 |
| Đánh giá B0/F1 T1/T3 | V ở mức 40 episode summary/status/hash và roster | `step5_test_20261109-093744` | Bảng sơ bộ B0 0/20, F1 1/20; một seed, 10 state groups | Chưa là cải thiện đáng tin cậy, chưa là causal effect của memory |
| F0/F1 matched, F1-off, adapter-base | A+V ở mức artifact, runtime và rollout E2 | Cặp F0/F1 V2 khớp data lock, recipe 5.000 bước; kết quả rollout `paired_e2_final.json` (40 cặp / 80 episodes) | Báo cáo chi tiết kết quả E2 unseen: F0 10% (T1 15%, T3 5%), F1 0% (T1 0%, T3 0%), delta -10%; wall time F1 khoảng 5 lần F0 trong stack đồng bộ | Multi-seed rollout cho E2; mở rộng ngoài T1/T3 |
| Retention | A+V ở mức rollout E6 hoàn tất | `experiments/results/e6_retention_locked_50` (50 cặp B0/F1 trên 10 task LIBERO-10) | Báo cáo kết quả E6: B0 37/50 (74%), F1 23/50 (46%), delta -28%; phân tích theo nhóm dung sai vị trí (T3, T8 bảo toàn; T5 giảm mạnh) | Retention trên bộ LIBERO-90 đầy đủ |
| Derived replay đời cũ | A ở mức metadata | `data/libero_mem_derived/T3_train_summary.json`: 78 replay, 73 được summary gắn success; pilot T1/T3 5 mỗi task | Ghi tiến độ tạo dữ liệu riêng, có giới hạn | Chưa xác minh HDF5/hash/geometry/alignment, không phải policy SR hoặc dataset v1 hoàn tất |
| Derived generation version mới | A+V ở mức dataset/export; closed-loop gate bị chặn | Audit dữ liệu derived, lock `c1691137...`, dataset manifest, log training step 5.000, export model; [preflight receipt](evidence/derived_f1_preflight_blocked_20261121.json) | 198 replay (giữ 188 episode / 40.530 frame); export LoRA ghi `rgb_cycle_v3` (`predicted_rgb_v3`) | Preflight T1/T3 test dừng trước rollout vì `src/perception/grounding_dino.py` khác SHA kỳ vọng trong export; 0 episode được chạy. Cần source snapshot có attestation hoặc export lại; không override hash |
| Multi-seed, held-out confirmatory test | H | Kế hoạch protocol | Kế hoạch phân tích uncertainty | Chưa có nhiều seed và holdout mới chưa dùng phát triển |
| UR3/robot thật, multi-object memory, RL/SGS | Ngoài phạm vi hiện tại | SRS và phạm vi nghiên cứu | Hướng phát triển nếu thực sự liên quan | Tạm bỏ triển khai, bảng kết quả và tuyên bố chuyển giao |

## 3. Những cập nhật quan trọng so với tài liệu cũ

Rà soát D/E ngày 21/11: các lượt mới đã hoàn tất D1 parity, D2 closed loop và B0 raw unseen. Chúng khép các gate thực thi tương ứng, nhưng chưa khép nhãn độc lập, kiểm định nhân quả hoặc xác nhận nhiều seed.

**Cập nhật 20/11/2026:** pilot D2 cùng state/seed T1 event–periodic kích hoạt một requery dựa trên fact/evidence ở bước 3; event dùng 101 query, periodic dùng 125 (cadence 10 so với 8 action), wall time lần lượt 1.144,01/1.229,75 giây. Cả hai đạt 0/1 subgoal. Đây là xác nhận wiring trên một state, không phải so sánh matched-budget hoặc bằng chứng cải thiện hiệu quả. [Pilot D2](evidence/d2_event_mode_pilot_20261120.json).

Replay shadow offline trên trace E2 F1 tái tạo đúng 3.969 prompt; 70 thay đổi đủ ngưỡng early-requery xuất hiện trên 38/40 quỹ đạo cố định, trong đó năm lần tăng return quan sát được. Replay không gọi lại policy hoặc đổi action, và nằm ngoài split seen của D2. [Receipt](evidence/e2_projected_shadow_replay_20261120.json).

**Hoàn tất · 21/11/2026:** D2 ghép cặp shadow/event trên 12 state seen development mỗi task, seed 42, đủ 24/24 episode mỗi arm; hash trạng thái khởi tạo khớp ở mọi cặp. Hai arm đều thành công 1/24; mean SCR là 0,111/0,167. Event phát 56 requery trong 22/24 episode; 56/56 log có evidence ID nhưng chỉ 41 thêm ID mới. 15 lần còn lại chỉ gắn `History incomplete`, giữ 9 ID từ quan sát bước 0–2; tuổi evidence median 73 action (32–758), chưa có ngưỡng tuổi định trước để chấm đúng/sai. Shadow không phát requery. Chênh lệch SR ghép cặp bằng 0; chênh lệch SCR mean +0,056, bootstrap theo nhóm state 95% CI [-0,125; 0,250]. Kết quả xác nhận wiring closed loop, không chứng minh lợi ích hiệu năng. [Phân tích ghép cặp](../../experiments/evaluation/decision_dev12_t1_t3_20261121_paired_analysis.json) · [Phân tích độ mới evidence](../../experiments/evaluation/d2_requery_evidence_freshness_20261121_v2/report.md) · [Shadow run](../../experiments/evaluation/decision_shadow_dev12_t1_t3_20261120_wsl/run_manifest.json) · [Event run](../../experiments/evaluation/decision_event_dev12_t1_t3_20261120_wsl/run_manifest.json) · [Video T1 thành công](evidence/d2_t1_event_success_video_20261121.json).

**E3 · renderer-only, 21/11:** Sweep residual budget trên snapshot event-projected của đúng 24 episode D2. Tái dựng chính xác prompt và metadata ở 48 token cho 2.397 query; ở 1/3 budget, fact delivery là 90,2% với T1 và 35,1% với T3; từ 2/3 đạt 100% ở cả hai. Prompt budget thấp chưa được đưa vào policy, nên không có kết luận về action hay success. [Report](../../experiments/evaluation/e3_token_budget_d2_dev12_20261121/report.md) · [Receipt](../../experiments/evaluation/e3_token_budget_d2_dev12_20261121/receipt.json).

**E4/E5 · one-query prompt intervention trên hai D2 success state, 21/11:** Ở T1 state7 event, exact-prompt control khớp 1.000/1.000 action và thành công; bỏ memory ở q142 đổi chunk (first-action L2 0,188), giữ prefix 142/142 rồi đạt 0/1 subgoal. Ở T1 state5 shadow, control cũng khớp 1.000/1.000 action và thành công; bỏ fact “bowl: 0 observed returns” tại q10 đổi chunk (first-action L2 0,150), giữ prefix 10/10 rồi đạt 0/1 subgoal, không overshoot. Cả hai là single-state counterfactual; không ước lượng efficacy tổng quát. [State7 report](../../experiments/evaluation/d2_success_prompt_intervention_20261121/report.md) · [State7 receipt](../../experiments/evaluation/d2_success_prompt_intervention_20261121/receipt.json) · [State5 report](../../experiments/evaluation/d2_state5_prompt_intervention_20261121_retry/report.md) · [State5 receipt](../../experiments/evaluation/d2_state5_prompt_intervention_20261121_retry/receipt.json).

**D3/D5 · đối chiếu cycle count offline trên rollout D2, 21/11:** Ghép `bowl_1.returns` của memory với `confirmed_cycles` trong evaluator LIBERO trên cùng 48 episode shadow/event. Endpoint count khớp event T1 10/12, event T3 8/12, shadow T1 6/12, shadow T3 10/12. Các lệch thời điểm chuyển count được lưu theo từng cycle; đây là mô tả từ evaluator, không phải nhãn blind về identity/relation/intent, cũng không được gọi là writer accuracy. [Report](../../experiments/evaluation/d2_evaluator_cycle_alignment_20261121/report.md) · [Receipt](../../experiments/evaluation/d2_evaluator_cycle_alignment_20261121/receipt.json) · [Per-episode counts](../../experiments/evaluation/d2_evaluator_cycle_alignment_20261121/episode_alignment.jsonl).

Preflight D2 trên calibration states 12–19 đã thử ngày 21/11 nhưng dừng trước rollout: 7/14 file writer trong export F1 khác checkout hiện tại; SHA dự kiến của `src/perception/grounding_dino.py` không có trong Git local. Run manifest ghi 0 episode. Không preflight arm event riêng vì dùng cùng checkpoint và bộ source. [Receipt calibration preflight](evidence/d2_calibration_preflight_blocked_20261121.json) · [Manifest](../../experiments/evaluation/decision_shadow_cal8_t1_t3_20261121_wsl/run_manifest.json).

Lượt D1 `text` đầu tiên dừng ở action 615 do detector worker chờ WSL `/mnt/d` I/O; receipt này được giữ như một lần chạy dở, không tính vào kết quả. Đợt tiếp theo hoàn tất text và shadow trên T1/T3 seen state0, seed42: cả hai arm dùng 100 query mỗi episode; query rows và toàn bộ action/query-step/action-chunk arrays khớp chính xác ở 2/2 cặp. Shadow ghi một thay đổi ngữ nghĩa có evidence ở bước 3 mỗi task nhưng quyết định là `continue`, không phát requery. Cả bốn lượt đạt 0/1 subgoal. Đây là kiểm tra observer/parity development-only; D2 multi-state đã hoàn tất; đối chứng periodic 8-action trước đó khác cadence và số query. [Receipt D1](evidence/d1_shadow_parity_20261120.json) · [Script audit](../../scripts/analyze_d1_shadow_parity.py) · [Receipt lượt D1 dở](evidence/d1_text_partial_20261120.json).

**B0 raw unseen bổ sung, 21/11:** hoàn tất 40/40 episode trên đúng 40 initial-state hashes E2, seed 42; đạt 0/40 success, mean SCR 0,0167, không overshoot. Đây là mốc mô tả: chỉ 115/134 source hash trùng với E2 và WSL policy runtime khác Windows E2 (Python/Torch/CUDA), nên không quy chênh lệch cho fine-tuning hoặc memory. [Receipt](evidence/e2_b0_unseen_paired_20261121.json) · [Phân tích](../../experiments/evaluation/e2_b0_unseen_paired_20261121_analysis.json) · [Run manifest](../../experiments/evaluation/e2_b0_raw_unseen_20261120_wsl/run_manifest.json).

| Gói | Đã thực hiện | Gap còn mở / lý do chưa chạy tiếp |
|---|---|---|
| D0 | Offline replay lịch truy vấn trên quỹ đạo mẫu và trace E2 | Chưa có nhãn độc lập về đúng/sai của quyết định hoặc điểm commit; replay không đổi hành động |
| D1 | Text/shadow parity chính xác trên 2 cặp T1/T3 seen state 0 | Chỉ xác minh plumbing; không có requery hay success trong các lượt này |
| D2 | 24 cặp closed loop T1/T3 seen; event phát 56 requery, cả 56 log có evidence ID, 41 thêm ID mới và 15 giữ ID cũ sau khi gắn `History incomplete` | Hai arm đều 1/24; chưa có lợi ích efficacy. Độ mới ID chỉ mô tả trace, không phải nhãn đúng/sai. Calibration states 12–19 bị chặn ở preflight hash (0 episode); cần khôi phục provenance trước khi chạy thêm, rồi mới có confirmatory roster/nhiều seed |
| D3–D4 | Rule probe offline, hợp đồng chunk, và đối chiếu `bowl_1.returns`–evaluator trên 48 rollout D2 | Alignment endpoint khớp 10/12 event T1, 8/12 event T3, 6/12 shadow T1, 10/12 shadow T3; chỉ là mô tả cycle count. Thiếu nhãn độc lập intent/target/effect, reader đã hiệu chuẩn, candidate-recall và queue/cache/RNG parity; chưa thể bật reject hoặc chọn nhiều chunk |
| D5–D6 | Writer diagnostics T7–T10 và đối chiếu evaluator-cycle count trên cohort D2 | Chưa có blind identity/relation/containment labels; alignment count không đo độ đúng identity/quan hệ; bất đồng replay T8 ở bước 219 chưa giải thích; thiếu trace nhãn và protocol xác nhận |
| E0–E1 | E0 pilot 4/4 episode; E1 writer/source diagnostics | E0 là pilot nhỏ; E1 chưa có nhãn độc lập trên policy rollout |
| E2 | F0/F1 matched 40 cặp unseen; B0 bổ sung 40 lượt | E2 dùng query định kỳ, không kiểm trigger evidence. B0 không so sánh được trực tiếp do source/runtime mismatch |
| E3 | Fixed-trace renderer sweep trên 24 episode D2; tái dựng chính xác 2.397/2.397 prompt ở 48 token; sweep 0, 1/3, 2/3, full residual budget | Chỉ đo fact delivery: 0 token = 0%; 1/3 = 90,2% T1 và 35,1% T3; 2/3/full = 100%. Chưa đưa prompt budget thấp vào policy, nên chưa đo hành động/SR hoặc efficacy. [Report](../../experiments/evaluation/e3_token_budget_d2_dev12_20261121/report.md) · [Receipt](../../experiments/evaluation/e3_token_budget_d2_dev12_20261121/receipt.json) |
| E4 | D2 state7 q142 và state5 q10: one-query memory suppression với exact-prompt replay control trên hai success states | Cả hai control khớp 1.000/1.000 action và thành công; suppression branches đổi chunk, giữ prefix tương ứng 142/142 và 10/10, rồi đạt 0/1 success. Mới hai single-state delay; chưa có nhiều mức delay/noise/corruption, roster rộng, nhãn độc lập hoặc risk–coverage. [State7 report](../../experiments/evaluation/d2_success_prompt_intervention_20261121/report.md) · [State5 report](../../experiments/evaluation/d2_state5_prompt_intervention_20261121_retry/report.md) |
| E5 | State16: 8 prompt branches; D2 state7/state5: exact-prompt replay và one-query memory suppression | State16 các nhánh đều overshoot. State7 và state5 controls khớp 1.000/1.000 action và success; hai nhánh bỏ memory có prefix khớp trước query, chunk đổi, 0/1 success. Đây là hai counterfactual đơn state, chưa phải efficacy estimate; fact-changing/preserving và C0/L0 controls trên roster rộng chưa chạy. [State7 report](../../experiments/evaluation/d2_success_prompt_intervention_20261121/report.md) · [State5 report](../../experiments/evaluation/d2_state5_prompt_intervention_20261121_retry/report.md) |
| E6 | 50 cặp B0/F1-off trên LIBERO-10 local | Phạm vi local 10 task; chưa tách cơ chế retention hoặc khái quát sang LIBERO-90 |

Preflight closed loop cho derived v1 đã dừng trước rollout do `src/perception/grounding_dino.py` không khớp SHA attested trong export; 0 episode được chạy. Hash expected `84e832…`, current `daace7…`; rà 3 revision Git local không tìm thấy file khớp, nên cần phục hồi snapshot có attestation hoặc export lại checkpoint có provenance khớp. [Receipt preflight](evidence/derived_f1_preflight_blocked_20261121.json).

1. `THESIS_READINESS_AUDIT_2026-10-28.md` và phần chính của `thesis_completeness.md` không còn là trạng thái hiện hành. Câu “toàn bộ rào cản đã đóng” mạnh hơn bằng chứng; ngược lại, “ready to train” đã lạc hậu vì hiện có training artifact thật.
2. `RESULTS_MEM_2_TEST_PLAN.md` ghi chưa có rollout hoàn tất vào 07/11. Run 09/11 hiện có đủ B0/F1. Cần dùng trạng thái mới này khi viết, nhưng không suy rằng F0/F1-off đã chạy.
3. T1 là nhấc và đặt lại cùng bát một lần; T3 là ba lần. Không phải swap ba bát. Protocol local dùng 1.000 policy actions, H=50, s=10, warmup 20; không dùng horizon 150/300 hoặc s50 từ tài liệu cũ cho run mới.
4. Counter evaluator upstream tích lũy giữa các milestones, với điều kiện `>5`, `>15`, `>5` được protocol ghi rõ; không mô tả thành “5 frame liên tiếp”. Dwell3 của visual writer là cơ chế khác.
5. Bộ nhớ hiện reset từng episode. “Persistent” nghĩa là duy trì qua thời gian trong episode, không phải lifelong memory giữa các nhiệm vụ.
6. Summary replay T3 ngày 11/11 là artifact lịch sử. Đến 14/11, split/index/shards V2 từ nguồn refined đã qua alignment audit; không nhầm corpus đó với dữ liệu derived hoặc checkpoint lịch sử.

## 4. Số liệu được đưa vào bản thảo

### Rollout local tháng 11

Nguồn: [episode-level CSV được tái tổng hợp](evidence/mem_test_episodes.csv); từng `summary.json` khớp checksum trong `status.json`, status complete và roster đủ. [Bảng tổng hợp](evidence/mem_test_counts.csv) được tính lại từ các summary, không chép từ bảng trong docs.

| Task | Condition | Thành công / N | SR | Mean SCR | Overshot |
|---|---|---:|---:|---:|---:|
| T1 | B0 | 0/10 | 0% | 0,1000 | 1/10 |
| T1 | F1 | 0/10 | 0% | 0,1000 | 1/10 |
| T3 | B0 | 0/10 | 0% | 0,0000 | 0/10 |
| T3 | F1 | 1/10 | 10% | 0,1667 | 0/10 |
| Macro hai task | B0 | 0/20 | 0% | 0,0500 | 1/20 |
| Macro hai task | F1 | 1/20 | 5% | 0,1333 | 1/20 |

N=20 là task-episodes mỗi condition, cùng 10 initial-state hashes qua T1/T3, policy seed 42. Chênh lệch 5 điểm phần trăm không phải chứng minh tăng hiệu năng có ý nghĩa hoặc do riêng memory. Hai source snapshots khác ba launcher/diagnostic scripts, không được viết “toàn bộ code giống hệt”; tên/hash đã lưu trong snapshot. Protocol, split và checkpoint gốc khớp.

### Pilot development E0 V2 ngày 15/11

Nguồn là [manifest pilot](../../experiments/evaluation/e0-b0-f0-pilot-20261115/run_manifest.json), [report](../../experiments/evaluation/e0-b0-f0-pilot-20261115/report.md) và bốn bộ `summary.json`/`status.json` có SHA-256. Mỗi điều kiện chạy một state seen (state index 0, seed 42) trên T1/T3; từng episode đủ 1.000 policy actions và dừng do `horizon_reached`.

| Task | Condition | Thành công / N | Mean SCR | Overshot | Completed subgoals | Mean policy inference / query | Wall time / episode |
|---|---|---:|---:|---:|---:|---:|---:|
| T1 | B0 | 0/1 | 0,000 | 0/1 | 0/1 | 652,60 ms | 202,25 s |
| T1 | F0 | 0/1 | 0,000 | 0/1 | 0/1 | 707,72 ms | 208,46 s |
| T3 | B0 | 0/1 | 0,000 | 0/1 | 0/3 | 596,03 ms | 202,04 s |
| T3 | F0 | 0/1 | 0,000 | 0/1 | 0/3 | 673,31 ms | 205,23 s |

Mean policy inference/query là thời gian suy luận chính sách được runner đo riêng; wall time bao trùm episode. Runtime check riêng qua 4/4 treatment-task checks (2 action/check). Đây là pilot custom `NON-COMPARABLE_OFFICIAL_PAPER`, với một state/task và không có memory trace hoặc decision metric. Không đo hiệu ứng memory, không kiểm điểm quyết định từ memory, và không gộp với bảng lịch sử B0/F1.

### Thực nghiệm E2: Rollout so sánh cặp F0/F1 trên 20 trạng thái Unseen (T1 & T3)

Nguồn là kết quả nghiệm thu [paired_e2_final.json](../../experiments/evaluation/e2_unseen_author_v2/paired_e2_final.json) và [e2_summary_report.md](../../experiments/evaluation/e2_unseen_author_v2/e2_summary_report.md). Đánh giá gồm 40 cặp đối chứng (80 episodes) trên 20 nhóm trạng thái ban đầu chưa từng gặp trong huấn luyện (`demo_101` đến `demo_120`), cố định seed 42, H=50, chunk=50, tối đa 1.000 actions.

| Task | Condition | Thành công / N | SR | Delta (F1 - F0) | Median Wall Time / Episode |
|---|---|---:|---:|---:|---:|
| T1 | F0 (no-memory) | 3/20 | 15% | — | ~213 s |
| T1 | F1 (rgb_relations) | 0/20 | 0% | -15% (-0,15) | ~1.065 s |
| T3 | F0 (no-memory) | 1/20 | 5% | — | ~214 s |
| T3 | F1 (rgb_relations) | 0/20 | 0% | -5% (-0,05) | ~1.072 s |
| **Macro** | **F0** | **4/40** | **10%** | — | **~213 s** |
| **Macro** | **F1** | **0/40** | **0%** | **-10% (-0,10)** | **~1.068 s** |

Trong stack đồng bộ E2, median wall time là khoảng 213 s/lượt với F0 và 1.066 s/lượt với F1. Các log xác nhận chênh lệch thời gian và kết quả, nhưng không đủ để kết luận nguyên nhân hiệu năng.

### Thực nghiệm E6: Đánh giá duy trì kỹ năng (Retention) trên 10 nhiệm vụ LIBERO-10

Nguồn là các thư mục `B0_task_*` và `F1_actual_task_*` trong [e6_retention_locked_50](../../experiments/results/e6_retention_locked_50). F1-off dùng export F1 V2 của E2 và tắt memory. Đánh giá 50 cặp matched episodes (100 episodes tổng cộng) giữa mô hình gốc B0 (SmolVLA base) và mô hình tinh chỉnh F1-off (LoRA không cấp chuỗi memory) trên 10 nhiệm vụ LIBERO-10 (5 test states mỗi task).

| Chỉ số | B0 (Base) | F1-off (Adapted) | Chênh lệch (F1 - B0) |
|---|---:|---:|---:|
| Tỷ lệ thành công tổng | 37/50 (74%) | 23/50 (46%) | -28% (-14 task-episodes) |
| Bảng liên hợp cặp (1,1) / (0,0) | 18 cặp cùng thành công | 8 cặp cùng thất bại | — |
| Bảng liên hợp phân kỳ (1,0) / (0,1) | 19 (B0 thắng, F1 thua) | 5 (F1 thắng, B0 thua) | — |

- T3 và T8 ghi nhận cùng số thành công ở hai điều kiện (5/5 và 3/5); T0 ghi nhận 0/5 ở B0 và 2/5 ở F1-off.
- Chênh lệch lớn nhất quan sát được là T5 (5/5 B0, 0/5 F1-off); T9 ghi nhận 5/5 và 3/5. Các số này mô tả outcome, không xác định nguyên nhân chuyển động.

### Nghiệm thu runtime parity MiniVLA (Backbone 2)

Nguồn từ biên bản runtime parity [minivla_libero90_runtime_receipt.json](../../resources/manifests/models/minivla_libero90_runtime_receipt.json) và [báo cáo audit](../../experiments/audits/minivla-contract-audit-20261115/report.json). Đối chiếu trực tiếp với upstream Stanford-ILIAD/openvla-mini tại commit `0822b36227b5a771be4eb2680e34c559734c8fdc` và checkpoint `step-060000-epoch-26-loss=0.1209.pt` (SHA-256: `2baadbf7...`).

- **Môi trường & Phần cứng:** PyTorch 2.6.0+cu124, CUDA 12.4, GPU NVIDIA GeForce RTX 3090 (24GB VRAM).
- **Thời gian thực thi:** Nạp mô hình và tokenizer 148,45 s; suy luận forward mạng nơ-ron 2.357,66 ms; chu kỳ step hoàn chỉnh 578,05 ms.
- **Vectơ hành động suy luận:** `[-0.000746, 0.323792, -0.152600, -0.001975, -0.007639, -0.000303, -1.0]` (gripper đóng hoàn toàn sau bước nghịch đảo Robosuite).
- **Kiểm định bất biến:** 5/5 kiểm tra đạt chuẩn tuyệt đối (`shape_matches_1x7`, `dtype_is_float32`, `no_nans_or_infs`, `values_within_minus1_plus1`, `gripper_binarized`). Mọi thành phần loader (`load_vla`), DinoSigLIP transforms, resize LANCZOS 224x224, prompt builder và unnormalization phân vị đã được nghiệm thu runtime chính thức (`VERIFIED`).

### Chất lượng thông tin memory quan sát được

Audit RGB Cycle (`rgb_cycle_v3`) ngày 26/10 là một tập chẩn đoán riêng với rollout policy tháng 11. Trên sáu demonstration và 1.276 frame, có 12/12 return ghép một-một với milestone annotation trong dung sai 20 frame; median delay 9 frame. 1.169/1.172 detection bát được chấp nhận có IoU ≥0,5. Kết quả này chỉ mô tả các demonstration đã xem, trong đó không có failure trajectory. Nó không phải precision/recall trên phân phối thao tác, và milestone không đồng nghĩa nhãn vật lý về grasp hoặc contact. Detector RGB mất khoảng 999–1.058 ms/frame trong audit live nên chưa đáp ứng 20 Hz. Artifact con và source archive được hash-kiểm trong snapshot.

Trong 20 F1 episodes, **19 snapshot cuối có `history_complete=False`**. Trong 1.927 policy queries, 1.894 prompt có suffix memory; 1.738 prompt có chuỗi `history incomplete`. Đây là phép đếm trace/văn bản, không phải event recall, không phải 1.927 mẫu độc lập và không phải đo tác động nhân quả.

Episode thành công duy nhất `T3_F1_test_state2`, từ `demo_29`, có evaluator ghi ba chu kỳ nhưng writer cuối ghi hai và lịch sử chưa đầy đủ. Quan sát này ngăn việc diễn giải success duy nhất thành bằng chứng writer đếm đúng rồi policy dùng count đó. Nó không chứng minh memory vô dụng hoặc mọi trường hợp đều sai.

### Training

Tại step 3000, validation Mem flow loss của cùng F1 là **0,23023575** khi có memory và **0,33553660** khi bỏ memory; base loss **0,42984212** ở cả hai mode. Có 4.299 Mem validation frame-samples và 13.525 base frame-samples. Các frame trong cùng episode phụ thuộc nhau; số frame không phải số trajectory độc lập. Không có F0 riêng trong phép đo này, và `memory_dropout=0` khiến F1-off có thể là input ngoài phân phối huấn luyện.

## 5. Sổ các câu nên và không nên viết

| Viết được | Chưa được viết |
|---|---|
| “Đã xây dựng bộ nhớ sự kiện từ RGB và ghi nhận bằng chứng theo thời điểm khả dụng.” | “Đã giải quyết triệt để partial observability.” |
| “Đã fine-tune 3.000 bước và xác minh các file export của adapter.” | “Huấn luyện thành công nên policy đã tốt hơn.” |
| “F1 có một success trong 20 task-episodes dưới protocol local.” | “Memory cải thiện 5% trên LIBERO-Mem chính thức.” |
| “Loss giảm khi cung cấp memory trong phép đánh giá offline của cùng checkpoint.” | “Ablation chứng minh memory hiệu quả trong closed loop.” |
| “Writer dùng bằng chứng thị giác và chủ động báo lịch sử chưa đầy đủ.” | “Writer có accuracy cao hoặc count luôn là cận dưới đúng của sự thật.” |
| “Replay 50/50 được dùng để khảo sát duy trì kỹ năng.” | “Replay ngăn catastrophic forgetting.” |
| “Dataset derived v1 đã có lock và export LoRA 5.000 bước được kiểm tra lineage/hash.” | “Dataset derived đã tái tạo toàn bộ shard hoặc thay thế cho author split v2 trong benchmark chính.” |
| “Kết quả có provenance trong phạm vi artifact đã kiểm.” | “100% reproducible, 100% fair, tương đương runtime tác giả.” |
| “Các failure gợi ý cần kiểm tra grounding/interface/data mismatch.” | “Failure do quên tiến độ, do attention dilution, hoặc do tác giả có cấu hình bí mật.” |
| “Trên một demo T7, writer relation không cấp fact memory; relation giữ unknown.” | “Writer T7 đạt relation accuracy hoặc có thể theo dõi identity qua hoán đổi.” |
| “Trên một demo T8, writer relation không cấp fact memory trong 613 frame; relation giữ unknown.” | “Writer T8 đạt identity tracking hoặc relation accuracy qua ba bát.” |
| “Trên một demo T9, writer relation không cấp fact memory trong 356 frame; relation giữ unknown.” | “Writer T9 xác định đúng giỏ chứa/giỏ rỗng hoặc xử lý containment qua che khuất.” |
| “T2/T4 bị loại khỏi suite chính do bất định tư thế chai khiến thí nghiệm bị chi phối bởi thích ứng grasp physics; ưu tiên T7 và hướng đến T9/T10.” | “Đánh giá T2/T4 là đủ để kết luận về khả năng thích ứng của memory trên vật thể mới (thực tế bị confound bởi physics tiếp xúc chai đổ).” |

## 6. Điều kiện mở lại các mục đang hoãn

| Mục luận văn đang hoãn | Bằng chứng tối thiểu cần bổ sung / Tình trạng nghiệm thu hiện tại |
|---|---|
| Bảng baseline V1 chính thức của project | Phục hồi đúng raw tham chiếu và khớp frozen manifest; không thay run |
| Dataset derived đã hoàn thiện | **Đã nghiệm thu artifact:** Đã có lock `c1691137...`, dataset manifest (188 episode / 40.530 frame) và export LoRA 5.000 bước (`results_mem_derived/mem-adapter-v1`); cần nghiệm thu kiểm tra policy closed-loop riêng |
| Hiệu quả riêng của memory | **Đã nghiệm thu qua E2:** 40 cặp F0/F1 trên 20 unseen states (`paired_e2_final.json`), SR F0 10% vs F1 0% (delta -10%), đo lường đầy đủ trade-off độ trễ 5x |
| Retention | **Đã nghiệm thu qua E6:** 50 cặp B0/F1 trên 10 nhiệm vụ LIBERO-10 (`experiments/results/e6_retention_locked_50`), SR B0 74% vs F1 46% (delta -28%) |
| Chất lượng writer | Annotation độc lập, event matching rule/time tolerance đặt trước, precision/recall/delay/unknown coverage |
| Runtime/chi phí | **Đã đo lường trong E2 & E6:** E2 median gộp 213,03 s với F0 và 1.065,89 s với F1 Relation V1; cached p95 Relation V3 motion 1,53 ms từ phép đo riêng |
| So sánh nhiều backbone | **Đã nghiệm thu runtime parity MiniVLA (Candidate B):** Pinned commit `0822b36227b5a771be4eb2680e34c559734c8fdc`, PyTorch/CUDA runtime receipt và 5/5 invariant checks đã xác lập; Candidate VQ tiếp tục hoãn do thiếu local weights/codebook |
| Kết luận cuối và tóm tắt | Chốt toàn bộ evidence dùng trong chương 4; loại mọi suy luận không được dữ liệu hỗ trợ |
