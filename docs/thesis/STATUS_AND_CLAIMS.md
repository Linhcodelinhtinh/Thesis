# Trạng thái và các tuyên bố được phép viết

Mốc rà soát: **15/11/2026**, theo [snapshot máy đọc được](evidence/snapshot.json) và [claim ledger](evidence/claim_ledger.json). Working tree có thay đổi; mọi kết luận kỹ thuật ràng buộc vào artifact/hash và phạm vi nêu dưới đây. Huấn luyện model mới nằm ngoài phạm vi người dùng yêu cầu.

## 1. Bốn mức bằng chứng

- **S — Source:** đọc được implementation/config; chưa khẳng định runtime.
- **A — Artifact:** có output của một lần chạy, checkpoint hoặc log; cần xác định đúng nguồn và phạm vi.
- **V — Verified in scope:** đã đối chiếu những thuộc tính cụ thể, ví dụ hash export hoặc thống kê episode. Không đồng nghĩa đã kiểm tra lại toàn runtime.
- **H — Hypothesis:** mục tiêu/giả thuyết chưa được kiểm định đủ.

## 2. Ma trận hoàn thiện

### Trạng thái hiện hành sau 15/11

| Gói | Trạng thái | Kết quả đã xác minh | Chưa được xem là hoàn tất |
|---|---|---|---|
| P0/P7 · Bản viết | Đã biên tập Chương 1–5 theo 3 RQ; abstract có SHA-256 lock; validator/ledger pass | Sơ đồ kiến trúc, timeline khái niệm, public provenance; PDF 57 trang build, render toàn bộ và rà trực quan | Không có layout gate còn mở; các nội dung tác giả tự xác nhận vẫn cần họ đọc trước khi nộp |
| P1 · Nguồn và dữ liệu | Split V2, refined source, index T1/T3, shards và alignment audit; LIBERO source pin cài trong venv Python 3.8.13 | 698 demo; 180.447 hàng state/action; 2.094 RGB frame/camera; 1.396 parity mẫu; strict validator pass; smoke 14/14 | EGL dùng fallback `kms_swrast`; smoke không phải policy score; MiniVLA audit đã chốt blocker, chưa qua runtime gate |
| P3/P4 · Writer và profile | T1/T3 writer cũ có audit riêng; profile T1–T10 tồn tại; raw/refined T7/T8/T9/T10 source receipts được hash-check local | T7/T8/T9/T10 RGB `demo_1` không cấp fact memory; T7 exo-box audit ở IoU 0,5 cho bowl P/R 0,893/0,747 và plate 0,971/0,646. Author-action replay upstream một demo/task T8/T9/T10 ghi T9/T10 terminal success; T8 raw/refined action/state đầu vào trùng nhưng robot-state replay vượt `1e-5` ở bước 219, có 3 milestone và không terminal | Chưa có nhãn blind identity/relation/containment; writer diagnostics không phải accuracy hoặc policy score; nguyên nhân T8 drift chưa rõ |
| P5 · Dữ liệu và adapter | Shards V2 và alignment pass; F0 V2 no-memory artifact 5.000 bước đã tồn tại và strict reload/runtime check đạt; audit image preprocessing không thấy mismatch ở transform đã khai báo | Inference env Python 3.12.3/CUDA 12.8; F0 runtime check 4/4; training collator khớp hash artifact, LeRobot 0.6.1 ở train/eval | F1 V2 chưa có; package source hash processor phía training không được lưu; không có cặp F0/F1 matched; training mới nằm ngoài phạm vi |
| P6 · Policy và cơ chế quyết định | Pilot B0/F1 lịch sử đủ 40 summary; E0 B0/F0 V2 pilot 4/4 episode complete; T7 detector đã đối chiếu nhãn exo-box; T7/T8/T9/T10 writer và author-action diagnostics trên demo train | Pilot V2: state seen 0, một state/task, 1.000 actions/episode, 0/4 success, SCR 0, overshot 0; author-action replay không chạy policy và không phải score official | Memory competence/decision point, matched F0/F1, E3–E6, blind identity/relation labels, retention, multi-seed; nguyên nhân T8 drift cần giải thích |
| Backbone thứ hai | MiniVLA checkpoint bytes hash-match và official `openvla-mini` source commit đã pin; contract và sai khác adapter ghi trong [audit report](../../experiments/audits/minivla-contract-audit-20261115/report.json) | Chưa chọn eval checkpoint; local adapter không khớp official loader/prompt/image/gripper path; Candidate VQ thiếu checkpoint/codebook | Không so sánh backbone hoặc chạy rollout |

Bảng chi tiết dưới đây lưu góc nhìn của snapshot 12/11 và các ghi chú lịch sử; khi khác với trạng thái hiện hành ở trên, dùng bảng mới này cùng snapshot ngày 15/11.

| Thành phần | Trạng thái đúng | Bằng chứng chính | Viết được | Còn thiếu / tạm hoãn |
|---|---|---|---|---|
| V1 baseline không memory | A; frozen theo báo cáo cũ; raw tham chiếu chưa tìm thấy | `experiments/baseline_v1/{V1_BASELINE_REPORT.md,freeze_audit_signoff.json}` | Mục tiêu, thiết kế và lịch sử freeze | Không trình bày 77/160 như đã kiểm lại; cần phục hồi đúng raw nếu dùng bảng V1 chính |
| Các baseline/sft lịch sử khác | V về số file và đếm outcome; chưa chứng nhận so sánh | `historical_run_inventory.csv`, 1.520 `episode.json` | Danh mục tiến độ, chọn candidate audit tiếp | Không gộp các run, không coi số file tìm thấy là planned sample size hoàn tất |
| SmolVLA adapter | S+A; đã được dùng trong rollout tháng 11 | Model manifest, `src/models/smolvla/adapter.py`, run manifests | Pipeline state 8/action7/chunk 50, camera, normalization | Không coi manifest version chung là resolved runtime của mọi run |
| MiniVLA / OpenVLA | Có code/manifest; candidate B provenance đã audit nhưng adapter bị fail-fast; candidate VQ thiếu local checkpoint/codebook | `src/models/minivla/`, model manifests, contract audit report | Nêu candidate bị chặn và lý do | Bảng so sánh, kết quả rollout, claim tốt hơn backbone khác |
| Simulator–policy–perception bridge | S+A; đã chạy thật cho B0/F1 | Worker/IPC code, environments và episode status tháng 11 | Kiến trúc tách tiến trình, allowlist, source lock | Không nói policy chạy trong venv riêng hoàn hảo: manifest ghi Python host Windows cho policy, WSL venv cho simulator, venv riêng cho detector |
| Tracker và writer RGB v3 | S+A; có trace rollout và audit demonstration riêng | `object_tracker.py`, `event_writer.py`, `experiments/audits/memory_writer_v3/verification_summary.json`, F1 memory traces | Anchor gốc, FSM, hysteresis, unknown, causal timestamps; 12/12 return ghép milestone trong ±20 frame trên sáu demo | Chỉ ba state và demo thành công đã xem; milestone không phải contact label; chưa có ước lượng fidelity trên failure trajectory |
| Renderer | S+A | `libero_mem_text.py`, `queries.jsonl`/prompt history | Budget toàn prompt 48 token, giữ instruction, privilege gate | Trạng thái causal/nguồn hợp lệ không chứng minh nội dung đúng hoặc model sử dụng hữu ích |
| Split và dataset tác giả | A+V ở mức manifest/training identity | `mem-adapter-v1/dataset-manifest.json`, split lock | 698 demo tổng; 558/70/70 train/val/test; Mem 158/20/20 | Chưa hash lại toàn bộ HDF5/shards; holdout không bảo đảm checkpoint pretrained chưa từng thấy dữ liệu |
| Huấn luyện F1 hiện tại | A+V | `mem-adapter-20261105-083543`, progress step 3000, metrics, 4 export hashes khớp | LoRA attention q/v, rank 16, 645.120 tham số; 2×T4; loss validation | Step3000 chưa được chứng minh tốt nhất; loss không phải SR; không gọi là train trên derived mới |
| F0 V2 no-memory adapter | A+V; artifact đã có trước cập nhật 15/11, được kiểm manifest, strict reload và runtime | `results_f0_author_split_v2/output_f0/pretrained_model/mem_adapter_manifest.json`; run manifest E0 | Adapter LoRA no-memory, 5.000 bước, dùng cùng V2 data lock | Không có F1 V2 đối sánh; không diễn giải F0 so với B0 thành hiệu ứng memory |
| Đánh giá B0/F1 T1/T3 | V ở mức 40 episode summary/status/hash và roster | `step5_test_20261109-093744` | Bảng sơ bộ B0 0/20, F1 1/20; một seed, 10 state groups | Chưa là cải thiện đáng tin cậy, chưa là causal effect của memory |
| F0/F1 matched, F1-off, adapter-base | H / đường tích hợp chưa đóng | CLI và F0 artifact V2 no-memory đã qua runtime; không có F1 V2 matched | Ý nghĩa từng đối chứng và protocol | Không đánh dấu hoàn tất từ B0/F0 pilot; F0/F1 cần cùng dữ liệu, recipe và checkpoint-selection |
| Retention | H cho adapter hiện tại | Có mixed replay; các SFT run cũ khác identity | Rationale và protocol kiểm tra | Routed-base dùng weights gốc không chứng minh chính adapter bảo toàn kỹ năng |
| Derived replay đời cũ | A ở mức metadata | `data/libero_mem_derived/T3_train_summary.json`: 78 replay, 73 được summary gắn success; pilot T1/T3 5 mỗi task | Ghi tiến độ tạo dữ liệu riêng, có giới hạn | Chưa xác minh HDF5/hash/geometry/alignment, không phải policy SR hoặc dataset v1 hoàn tất |
| Derived generation version mới | S, thay đổi chưa commit | Generator/split/preparation và phần update của audit11/11 | Mô tả hướng khắc phục, yêu cầu nghiệm thu | `data/libero_mem_derived_v1` chưa có tại snapshot; chưa có lock/index/shards/train/eval mới đủ chuỗi |
| Multi-seed, held-out confirmatory test | H | Kế hoạch protocol | Kế hoạch phân tích uncertainty | Chưa có nhiều seed và holdout mới chưa dùng phát triển |
| UR3/robot thật, multi-object memory, RL/SGS | Ngoài phạm vi hiện tại | SRS và phạm vi nghiên cứu | Hướng phát triển nếu thực sự liên quan | Tạm bỏ triển khai, bảng kết quả và tuyên bố chuyển giao |

## 3. Những cập nhật quan trọng so với tài liệu cũ

Cập nhật 14/11 về cơ chế quyết định. D0–D2 có code và replay lịch query trên quỹ đạo mẫu T1/T3. Replay hỗ trợ mô tả độ trễ, độ bao phủ mốc writer và số query dự kiến. Tác động lên lựa chọn hành động, tỷ lệ thành công và hiệu quả dưới cùng compute cần rollout F1 quan hệ với các arm đối sánh. [Bàn giao quyết định](MEMORY_DECISION_HANDOFF_2026-11-14.md).

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

### Chất lượng thông tin memory quan sát được

Audit writer ngày 26/10 là một tập chẩn đoán riêng với rollout policy tháng 11. Trên sáu demonstration và 1.276 frame, có 12/12 return ghép một-một với milestone annotation trong dung sai 20 frame; median delay 9 frame. 1.169/1.172 detection bát được chấp nhận có IoU ≥0,5. Kết quả này chỉ mô tả các demonstration đã xem, trong đó không có failure trajectory. Nó không phải precision/recall trên phân phối thao tác, và milestone không đồng nghĩa nhãn vật lý về grasp hoặc contact. Detector RGB mất khoảng 999–1.058 ms/frame trong audit live nên chưa đáp ứng 20 Hz. Artifact con và source archive được hash-kiểm trong snapshot.

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
| “Source đã bổ sung đường derived và cần kiểm export thực.” | “Dataset derived đã hoàn tất và model đã được train trên nó.” |
| “Kết quả có provenance trong phạm vi artifact đã kiểm.” | “100% reproducible, 100% fair, tương đương runtime tác giả.” |
| “Các failure gợi ý cần kiểm tra grounding/interface/data mismatch.” | “Failure do quên tiến độ, do attention dilution, hoặc do tác giả có cấu hình bí mật.” |
| “Trên một demo T7, writer relation không cấp fact memory; relation giữ unknown.” | “Writer T7 đạt relation accuracy hoặc có thể theo dõi identity qua hoán đổi.” |
| “Trên một demo T8, writer relation không cấp fact memory trong 613 frame; relation giữ unknown.” | “Writer T8 đạt identity tracking hoặc relation accuracy qua ba bát.” |
| “Trên một demo T9, writer relation không cấp fact memory trong 356 frame; relation giữ unknown.” | “Writer T9 xác định đúng giỏ chứa/giỏ rỗng hoặc xử lý containment qua che khuất.” |
| “T2/T4 bị loại khỏi suite chính do bất định tư thế chai khiến thí nghiệm bị chi phối bởi thích ứng grasp physics; ưu tiên T7 và hướng đến T9/T10.” | “Đánh giá T2/T4 là đủ để kết luận về khả năng thích ứng của memory trên vật thể mới (thực tế bị confound bởi physics tiếp xúc chai đổ).” |

## 6. Điều kiện mở lại các mục đang hoãn

| Mục luận văn đang hoãn | Bằng chứng tối thiểu cần bổ sung |
|---|---|
| Bảng baseline V1 chính thức của project | Phục hồi đúng raw tham chiếu và khớp frozen manifest; không thay run |
| Dataset derived đã hoàn thiện | Manifest version mới + hashes parent/source/runtime/code; inventory đủ roster; outcome/terminal/resume/ảnh thật được kiểm; split giữ nguyên quan hệ group |
| Hiệu quả riêng của memory | F0/F1 cùng dữ liệu/budget và đối chứng thông tin, rollout cùng state/seed/protocol, uncertainty phù hợp |
| Retention | Adapter hiện tại chạy base, so B0 matched; margin đặt trước và sample size đủ diễn giải |
| Chất lượng writer | Annotation độc lập, event matching rule/time tolerance đặt trước, precision/recall/delay/unknown coverage |
| Runtime/chi phí | Policy + perception + rendering + IPC + simulation end-to-end; hardware/runtime và cách pause đo rõ |
| So sánh nhiều backbone | Adapter đúng nguồn từng model, inference thật, shared information/protocol, data/budget và tuning budget được kiểm soát |
| Kết luận cuối và tóm tắt | Chốt toàn bộ evidence dùng trong chương 4; loại mọi suy luận không được dữ liệu hỗ trợ |
