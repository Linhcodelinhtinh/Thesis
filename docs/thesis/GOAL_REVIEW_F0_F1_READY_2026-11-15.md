# Review lượt goal trước khi cặp F0/F1 author-split V2 đã có

> Các trạng thái gate bên dưới ghi theo ngày của tài liệu. Xem [đối chiếu cập nhật 19/11/2026](EXPERIMENT_CONFIGURATION_REGISTER.md) cho export F1 V2, kết quả E2/E6 và replay Relation V3.


Ngày review: **15/11/2026**. Phạm vi: kiểm tra kết quả và kết luận của lượt goal trước, đối chiếu checkpoint hiện có, và xác định việc nào còn thiếu. Review không huấn luyện model và không chạy chiến dịch tính điểm. Git HEAD khi review: `50a4e7c08c22437f6c83c2993b5fa3951a233917`; working tree chưa sạch. Receipt của review: [f0_f1_readiness_review_2026-11-15.json](evidence/f0_f1_readiness_review_2026-11-15.json).

**Đánh giá: cần sửa trạng thái nghiệm thu và kết luận blocker.** Hai export F0/F1 tương thích với nhau đã hiện diện. Không còn căn cứ để gọi việc thiếu F1 V2 là blocker hiện tại. Lượt goal trước tạo được bằng chứng hữu ích về nguồn, dữ liệu, runtime và các chẩn đoán âm tính, nhưng chưa hoàn tất những mục tiêu ngoài training của roadmap. Checkpoint sẵn có mở đường cho các kiểm tra và pilot T1/T3; bản thân checkpoint chưa phải kết quả so sánh memory trong vòng kín.

## Cặp checkpoint được kiểm

| Thuộc tính | F0 | F1 |
|---|---|---|
| Export | `experiments/training/results_f0_author_split_v2/output_f0/pretrained_model` | `experiments/training/results_f1_author_split_v2/output_f1/pretrained_model` |
| Training condition | `no_memory` | `memory` |
| Bước cuối trong progress/log | 5.000 | 5.000 |
| Writer | Không có provenance writer trong adapter F0 | `rgb_relations_v1`; `relation_text_v1`, strategy `relevant` |
| SHA-256 weights | `414e4e0f3b6bd2bc42d5a044a3ecde9038eb6065bc23bb91b3734a40d6656d2c` | `572bd889d5bd3cf0aba83cb4f806b83794fcb650a7b4853ac505d025a08e4aad` |
| SHA-256 manifest | `04d9f259d21426615d26da0c23b70bfdd7f44df43bcdb3655265e58c4dee236b` | `624136821e330abac7a9c0fbfe4d59fdad0b3c3ae53c62e98c7da49e36b69d9e` |
| `validate_adapter` | PASS | PASS |

`validate_matched_adapters` PASS. Đối chiếu bổ sung cũng xác nhận base weights, versions, implementation hashes, official policy source, normalization, trainable modules và sampler bằng nhau. Data manifest chung có SHA-256 `fd5e7dca8d22598f3c3598af5c9bd35f775f98f0902b63c5787731e178522f87`, khớp manifest của bộ shards local. Khóa đánh giá chung có semantic hash `8275e468382835c06f9cc738361df26b7452965850878e7ca882f392c9ca4165`; hai bản `dataset-split.lock.json` bằng khóa V2 local. Policy của split là `libero_mem_refined_train_raw_eval_v2`.

Corpus có 198 demo Mem train và 400 demo base train; 50 demo base validation và 50 demo base retention. Hai nhánh có cùng seed 42, LoRA rank 16, learning rate `2e-5`, effective batch 32, base fraction 0,5 và 645.120 trainable parameters. Đây là các adapter LoRA chưa merge; khi nạp vẫn phải dùng đúng base checkpoint đã pin.

F1 có đủ provenance của hai index T1/T3, bằng provenance trong dataset; toàn bộ hash writer khớp chính xác bytes của code hiện tại. Index T1 có 14.079/14.746 frame cung cấp memory; T3 có 26.481/28.111. Tổng là 40.560/42.857 frame, với prompt dài nhất lần lượt 27 và 29 token. Đây là độ phủ projection, không xác nhận tính đúng của fact hoặc việc policy sử dụng fact.

Validation flow loss cuối là 0,4074395713 ở F0 và 0,4048465389 ở F1. Cả hai chỉ được đo trên 13.525 mẫu **base validation**, với 0 mẫu Mem validation. Không dùng hai số này để kết luận memory hiệu quả trên T1/T3 hoặc thay thế paired rollout.

## Findings

1. **[P1] Trạng thái “không có F1 V2” đã sai so với artifact hiện có.** `P5_P6_HANDOFF_2026-11-13.md:5`, `STATUS_AND_CLAIMS.md:21`, `evidence/claim_ledger.json:113`, `latex/content/chapter4.tex:43` và `chapter5.tex:9,23` đều còn dựa vào thiếu F1 để đóng nhánh đánh giá. F1 mới dùng đúng writer quan hệ, khác checkpoint pilot lịch sử `rgb_cycle_v3`. Cần sửa thành “cặp export đối sánh đã xác minh; kết quả paired rollout vẫn chưa có”, và cập nhật đồng bộ nguồn LaTeX, ledger, snapshot, handoff và PDF. Không thay số liệu pilot cũ bằng trạng thái checkpoint mới.

2. **[P1] Evidence pipeline chưa kiểm tính hiện hành của claim vắng checkpoint.** `tools/collect_evidence.py:750` chỉ nạp manifest F0 mới; nhánh `f0_v2_artifact` không kiểm inventory F1 và không gọi kiểm cặp. `validate_draft.py` vẫn PASS với 14 claims/55 sources và một claim `blocked_by_unavailable_training` khi cả hai adapter đã qua kiểm cặp. PASS này xác nhận cấu trúc/hash của tập nguồn được khai báo; nó không chứng minh “F1 không tồn tại”. Cần đưa manifest, weight identity, split/data hashes và trạng thái kiểm cặp vào evidence thay vì suy vắng checkpoint từ một preflight của export lịch sử.

3. **[P1] Phạm vi blocker bị mở rộng quá mức.** Thiếu nhãn target/effect cần cho D3 và đo chất lượng trigger; sai khác T8 thuộc nhánh D5/T8; contract MiniVLA thuộc backbone thứ hai. Các điều kiện này không ngăn việc nạp F1, kiểm wiring T1/T3, chuẩn bị E0 đối sánh hoặc kiểm observer D1. Theo dependency ở `MEMORY_DECISION_INTEGRATION_PLAN_2026-11-14.md:109–118`, D2 cần competence/writer phù hợp và D1 đã nghiệm thu trước khi diễn giải tác động. Do đó cần tiếp tục các bước đó, thay vì kết luận toàn bộ phần ngoài training không thể tiến thêm.

4. **[P2] Lệnh handoff chưa tái lập môi trường đã được báo cáo.** `MEMORY_DECISION_HANDOFF_2026-11-14.md:65–67` và `P5_P6_HANDOFF_2026-11-13.md:30–37` dùng Python mặc định của shell và simulator `.venv_libero_reference/bin/python`; kiểm trực tiếp đường dẫn sau trả **Python 3.8.20**. Runtime mới dùng policy `/home/linh/.venvs/smolvla_f0_eval_20261115/bin/python`, simulator `.venv_libero_reference_3813/bin/python` (**3.8.13**) và detector `.venv_memory_wsl/bin/python`. Handoff cần ghi đúng interpreter đã xác minh. Ngoài ra, `validate_memory_sft_export.py` trong policy env hiện dừng tại import `MemoryDataset` với `ModuleNotFoundError: No module named 'h5py'`; `envs/smolvla_inference.txt` chưa khai báo dependency này. Đây là lỗi tái lập lệnh kiểm export, không phải bằng chứng weights hỏng.

## Bằng chứng của lượt trước vẫn giữ giá trị

- Split/index/shards và audit alignment T1/T3; provenance nguồn raw/refined T7–T10.
- Pilot B0/F1 lịch sử và E0 B0/F0 V2 là hai tập kết quả khác nhau. Pilot B0/F0 V2 có 4 episode trên một state/task, 0/4 success; mẫu nhỏ này chưa định lượng competence của F1 mới hoặc tác động memory.
- Replay D0–D2 trên đường quan sát cố định cho biết lịch query dự kiến, chưa đo tác động của thay lịch trong vòng kín.
- T7–T10 writer diagnostic không tạo fact trên các demo được audit; detector audit T7 chỉ đo hộp 2D. Chưa có cơ sở chấp nhận fidelity identity/relation/containment trên toàn họ task.
- T8 raw/refined action và initial state khớp; robot-state vượt dung sai chẩn đoán từ bước 219 và replay không terminal. Đó vẫn là sai khác chưa giải thích, không phải bằng chứng xác định nguyên nhân dynamics.
- PDF 57 trang có record build/layout QA, nhưng các câu thiếu F1 trong nội dung đã lỗi thời. Không nên xem PASS kỹ thuật tài liệu là nghiệm thu toàn bộ nội dung khoa học.

## Trạng thái công việc sau review

| Gói | Trạng thái hợp lý hiện tại | Việc tiếp theo |
|---|---|---|
| P5 artifact F0/F1 | Có cặp export đối sánh, hash/split/writer contract PASS | Ghi receipt cặp và dùng đường dẫn cụ thể trong các lệnh đánh giá |
| E0 T1/T3 | Có thể thực hiện với checkpoint hiện có | Pilot F0/F1/F1-off trên seen, giữ state/seed/roster, ghi đủ trace và video cần cho annotation |
| D1 | Không còn vướng thiếu F1 | So text/shadow với cùng RNG, prompt, chunk và action; kiểm instrumentation không làm đổi output |
| D2 | Chưa nghiệm thu | Sau D1 và competence/writer gate, kiểm event/periodic trên seen, actual query budget, prefix cancellation và thời gian |
| E2 paired unseen | Chưa có kết quả | Freeze checkpoint-selection và protocol sau development rồi chạy roster unseen đã khóa; không tune trên unseen |
| D3/D4 | Chưa đủ nhãn/bộ đọc đề xuất/recall/parity để mở quyền reject/select | Thu nhãn độc lập và đo sai số; không mở quyền can thiệp bằng suy đoán từ delta action |
| E6 retention SmolVLA | Checkpoint đã có; chưa có nghiệm thu retention | Hoàn thiện/kiểm harness đúng adapter và roster base đối sánh |
| D5 T7–T10, MiniVLA | Các gate riêng còn mở | Tiếp tục audit writer/evaluator/official contract theo từng nhánh |
| P7 bản viết | Cần đồng bộ trạng thái mới | Cập nhật collector → ledger/snapshot → nội dung → PDF/QA; giữ giới hạn claims cho các rollout chưa chạy |

Các nhánh cần **training mới** như C0/L0 hoặc head/encoding mới vẫn nằm ngoài phạm vi đã loại trừ. Việc đánh giá cặp F0/F1 hiện hữu không thuộc phần loại trừ đó.

## Kiểm tra trong review

- Hash và `validate_adapter` cho hai export; `validate_matched_adapters`; so code/versions/base/dataset/split/memory-index provenance: PASS.
- `python docs/thesis/tools/validate_draft.py`: PASS, 20 citations, 17 labels, 14 claims/55 sources; hạn chế về claim hiện hành được ghi ở finding 2.
- Lệnh export validator ở policy env: FAIL do thiếu `h5py`, trước khi nạp checkpoint.
- Runtime check qua evaluator: đang chạy; sẽ chốt bằng run manifest tại `experiments/evaluation/review-f0-f1-runtime-20261115` và receipt của review.

Review không sửa implementation, recipe hoặc weights. Các finding ở trên chỉ rõ những file cần được đồng bộ; nguồn bản thảo/PDF của lượt goal trước được giữ để đối chiếu.
