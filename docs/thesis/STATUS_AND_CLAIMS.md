# Trạng thái và các tuyên bố được phép viết

Mốc rà soát: **11/11/2026**, theo [snapshot máy đọc được](evidence/snapshot.json). HEAD tại rà soát là `933ad9eeb0ee73caa1e5c6523df327c41f1b05d4`; working tree có thay đổi đang phát triển. Snapshot dùng hashes nội dung, không coi HEAD là toàn bộ code đã chạy. Không chạy thêm policy benchmark hoặc training trong công việc viết này.

## 1. Bốn mức bằng chứng

- **S — Source:** đọc được implementation/config; chưa khẳng định runtime.
- **A — Artifact:** có output của một lần chạy, checkpoint hoặc log; cần xác định đúng nguồn và phạm vi.
- **V — Verified in scope:** đã đối chiếu những thuộc tính cụ thể, ví dụ hash export hoặc thống kê episode. Không đồng nghĩa đã kiểm tra lại toàn runtime.
- **H — Hypothesis:** mục tiêu/giả thuyết chưa được kiểm định đủ.

## 2. Ma trận hoàn thiện

| Thành phần | Trạng thái đúng | Bằng chứng chính | Viết được | Còn thiếu / tạm hoãn |
|---|---|---|---|---|
| V1 baseline không memory | A; frozen theo báo cáo cũ; raw tham chiếu chưa tìm thấy | `experiments/baseline_v1/{V1_BASELINE_REPORT.md,freeze_audit_signoff.json}` | Mục tiêu, thiết kế và lịch sử freeze | Không trình bày 77/160 như đã kiểm lại; cần phục hồi đúng raw nếu dùng bảng V1 chính |
| Các baseline/sft lịch sử khác | V về số file và đếm outcome; chưa chứng nhận so sánh | `historical_run_inventory.csv`, 1.520 `episode.json` | Danh mục tiến độ, chọn candidate audit tiếp | Không gộp các run, không coi số file tìm thấy là planned sample size hoàn tất |
| SmolVLA adapter | S+A; đã được dùng trong rollout tháng 11 | Model manifest, `src/models/smolvla/adapter.py`, run manifests | Pipeline state 8/action7/chunk 50, camera, normalization | Không coi manifest version chung là resolved runtime của mọi run |
| MiniVLA / OpenVLA | Có code/manifest ứng viên; chưa có kết quả được nhận vào luận văn | `src/models/minivla/`, model manifests | Nêu candidate ngoài phạm vi hiện tại | Bảng so sánh nhiều model, tuyên bố tốt hơn backbone khác |
| Simulator–policy–perception bridge | S+A; đã chạy thật cho B0/F1 | Worker/IPC code, environments và episode status tháng 11 | Kiến trúc tách tiến trình, allowlist, source lock | Không nói policy chạy trong venv riêng hoàn hảo: manifest ghi Python host Windows cho policy, WSL venv cho simulator, venv riêng cho detector |
| Tracker và writer RGB v3 | S+A; có trace thật | `object_tracker.py`, `event_writer.py`, F1 memory traces | Anchor gốc, FSM, hysteresis, unknown, causal timestamps | Chưa có kiểm định event precision/recall/delay đầy đủ; không gọi count là ground truth |
| Renderer | S+A | `libero_mem_text.py`, `queries.jsonl`/prompt history | Budget toàn prompt 48 token, giữ instruction, privilege gate | Trạng thái causal/nguồn hợp lệ không chứng minh nội dung đúng hoặc model sử dụng hữu ích |
| Split và dataset tác giả | A+V ở mức manifest/training identity | `mem-adapter-v1/dataset-manifest.json`, split lock | 698 demo tổng; 558/70/70 train/val/test; Mem 158/20/20 | Chưa hash lại toàn bộ HDF5/shards; holdout không bảo đảm checkpoint pretrained chưa từng thấy dữ liệu |
| Huấn luyện F1 hiện tại | A+V | `mem-adapter-20261105-083543`, progress step 3000, metrics, 4 export hashes khớp | LoRA attention q/v, rank 16, 645.120 tham số; 2×T4; loss validation | Step3000 chưa được chứng minh tốt nhất; loss không phải SR; không gọi là train trên derived mới |
| Đánh giá B0/F1 T1/T3 | V ở mức 40 episode summary/status/hash và roster | `step5_test_20261109-093744` | Bảng sơ bộ B0 0/20, F1 1/20; một seed, 10 state groups | Chưa là cải thiện đáng tin cậy, chưa là causal effect của memory |
| F0 matched, F1-off, adapter-base | H / đường tích hợp còn thiếu | CLI hiện chỉ B0/F1; runner có nhãn F0 nhưng adapter contract cần đường riêng | Thiết kế đối chứng và ý nghĩa phép so sánh | Không đánh dấu đã hoàn thành chỉ vì enum/class có tên condition |
| Retention | H cho adapter hiện tại | Có mixed replay; các SFT run cũ khác identity | Rationale và protocol kiểm tra | Routed-base dùng weights gốc không chứng minh chính adapter bảo toàn kỹ năng |
| Derived replay đời cũ | A ở mức metadata | `data/libero_mem_derived/T3_train_summary.json`: 78 replay, 73 được summary gắn success; pilot T1/T3 5 mỗi task | Ghi tiến độ tạo dữ liệu riêng, có giới hạn | Chưa xác minh HDF5/hash/geometry/alignment, không phải policy SR hoặc dataset v1 hoàn tất |
| Derived generation version mới | S, thay đổi chưa commit | Generator/split/preparation và phần update của audit11/11 | Mô tả hướng khắc phục, yêu cầu nghiệm thu | `data/libero_mem_derived_v1` chưa có tại snapshot; chưa có lock/index/shards/train/eval mới đủ chuỗi |
| Multi-seed, held-out confirmatory test | H | Kế hoạch protocol | Kế hoạch phân tích uncertainty | Chưa có nhiều seed và holdout mới chưa dùng phát triển |
| UR3/robot thật, multi-object memory, RL/SGS | Ngoài phạm vi hiện tại | SRS và phạm vi nghiên cứu | Hướng phát triển nếu thực sự liên quan | Tạm bỏ triển khai, bảng kết quả và tuyên bố chuyển giao |

## 3. Những cập nhật quan trọng so với tài liệu cũ

1. `THESIS_READINESS_AUDIT_2026-10-28.md` và phần chính của `thesis_completeness.md` không còn là trạng thái hiện hành. Câu “toàn bộ rào cản đã đóng” mạnh hơn bằng chứng; ngược lại, “ready to train” đã lạc hậu vì hiện có training artifact thật.
2. `RESULTS_MEM_2_TEST_PLAN.md` ghi chưa có rollout hoàn tất vào 07/11. Run 09/11 hiện có đủ B0/F1. Cần dùng trạng thái mới này khi viết, nhưng không suy rằng F0/F1-off đã chạy.
3. T1 là nhấc và đặt lại cùng bát một lần; T3 là ba lần. Không phải swap ba bát. Protocol local dùng 1.000 policy actions, H=50, s=10, warmup 20; không dùng horizon 150/300 hoặc s50 từ tài liệu cũ cho run mới.
4. Counter evaluator upstream tích lũy giữa các milestones, với điều kiện `>5`, `>15`, `>5` được protocol ghi rõ; không mô tả thành “5 frame liên tiếp”. Dwell3 của visual writer là cơ chế khác.
5. Bộ nhớ hiện reset từng episode. “Persistent” nghĩa là duy trì qua thời gian trong episode, không phải lifelong memory giữa các nhiệm vụ.
6. Summary replay T3 được ghi lúc 17:12:31 UTC+7 ngày 11/11, muộn hơn snapshot audit cũ 15:56. Vì vậy câu “thư mục derived trống” trong phần đầu audit là sự kiện lịch sử, không phải trạng thái hiện tại. Output version mới vẫn chưa được chứng nhận.

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

### Chất lượng thông tin memory quan sát được

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
