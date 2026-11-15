# Bản thảo khóa luận cập nhật ngày 15/11/2026

**Nâng cao khả năng xử lý tác vụ chuỗi dài cho mô hình VLA trong điều khiển Robot**  
**Sinh viên:** Lưu Quang Linh, 23020392, K68A-AI2  
**Hướng dẫn:** PGS.TS. Nguyễn Việt Hà; PGS.TS. Hoàng Văn Xiêm  
**Ngành:** Trí tuệ nhân tạo, Trường Đại học Công nghệ, Đại học Quốc gia Hà Nội

Thông tin bìa lấy từ hàng 31, trang 4 của quyết định trong `docs/format_reference` và đã được sinh viên xác nhận trong phiên làm việc. Không sử dụng tên sinh viên/năm 2025 còn sót trong mẫu LaTeX.

Bản thảo khảo sát tiến độ thao tác dưới quan sát một phần, đề xuất bộ nhớ sự kiện có bằng chứng và phân tích vai trò của thông tin lịch sử trong các chính sách VLA nhỏ gọn. Ba câu hỏi nghiên cứu xoay quanh giá trị thông tin, độ tin cậy của evidence chưa đầy đủ và khả năng tái sử dụng. Bản biên tập học thuật tách pilot lịch sử, artifact V2 và replay offline. Chênh lệch B0/F1 mô tả hai cấu hình quan sát được, còn tác động riêng của lịch sử cần đối chứng bổ sung.

## Đọc theo thứ tự

1. [Roadmap và trạng thái các gate](IMPLEMENTATION_AND_WRITING_ROADMAP_2026-11-13.md).
2. [Ma trận hoàn thiện, bằng chứng và phần phải hoãn](STATUS_AND_CLAIMS.md).
3. [Kế hoạch viết toàn bộ khóa luận](WRITING_PLAN.md).
4. [Ghi chú đọc tài liệu và lựa chọn định dạng](REFERENCE_REVIEW.md).
5. [Nguồn LaTeX](latex/main.tex), [thư mục bằng chứng](evidence/snapshot.json), [sổ claim](evidence/claim_ledger.json), [danh mục tài liệu](latex/references.bib).

Nguồn nội dung chính là LaTeX. Bản Markdown là bản lịch sử. Công cụ xuất Markdown chỉ ghi vào thư mục thử nghiệm mới. Khung bìa, định dạng và phụ lục LaTeX nằm riêng trong `latex/`.

[Bàn giao P0–P2 và lệnh chạy tiếp](P0_P2_HANDOFF_2026-11-13.md).
[Bàn giao P3–P4 và lệnh audit, chuẩn bị dữ liệu](P3_P4_HANDOFF_2026-11-13.md).
[Bàn giao P5–P6 theo dung lượng hiện có](P5_P6_HANDOFF_2026-11-13.md).
[Roadmap và trạng thái các gate](IMPLEMENTATION_AND_WRITING_ROADMAP_2026-11-13.md).
[Kế hoạch tích hợp memory vào quyết định, ngày 14/11/2026](MEMORY_DECISION_INTEGRATION_PLAN_2026-11-14.md).
[Bàn giao triển khai, bằng chứng và lệnh chạy](MEMORY_DECISION_HANDOFF_2026-11-14.md).

## Quy ước sử dụng số liệu

- Tách V1 không memory khỏi V2 memory; không sửa baseline đã khóa để phù hợp kết quả mới.
- Các kết quả local mang nhãn **LIBERO-derived / NON-COMPARABLE_OFFICIAL_PAPER**. T1/T3 local không đại diện toàn bộ LIBERO-Mem.
- Số liệu rollout chính lấy từ `summary.json` từng episode của run tháng 11, kiểm tra `status.json` và SHA-256. Đây là tên artifact của runner mới, khác `episode.json` của runner cũ.
- Kết quả writer replay ngày 26/10 được báo riêng: sáu demonstration, ba initial state, 12/12 event ghép trong dung sai ±20 frame. Đây là kiểm tra chẩn đoán trên demonstration đã xem, không phải ước lượng tổng quát event accuracy. Audit relation T7 ngày 14/11 là phép đo riêng: 425/425 prompt không có fact memory và relation status đều unknown. T8 refined source đã hash-check local; audit `demo_1` có 613/613 prompt không có fact memory và relation status unknown. T9 refined source cũng đã hash-check local; audit `demo_1` có 356/356 prompt không có fact memory và relation state unknown. T10 raw/refined sources đã xác minh local; writer `demo_1` có 309/309 prompt không có fact memory, relation unknown và 927/927 identity observations unknown. Audit detector T7 đối chiếu 425 frame với exo-box annotations phát hành (IoU 0,5): bowl P/R 0,893/0,747, plate 0,971/0,646; đây chỉ là detection trên một demo, chưa có nhãn blind identity/relation/containment.
- Số 77/160 trong báo cáo V1 được lưu ở phần lịch sử: đường raw được báo cáo tham chiếu hiện không có episode để kiểm lại. Không lấy run khác thay vào cùng tên.
- Checkpoint F1 trong pilot là artifact lịch sử, không được train trên bộ shards V2. Không dùng nó để thay adapter tương thích writer relation.
- Split V2 có 698 demo, shards 11,83 GiB và alignment audit đã pass trên 180.447 hàng state/action cùng 1.396 mẫu parity. Artifact F0 V2 no-memory 5.000 bước đã tồn tại trước cập nhật này; inference/runtime check qua trên T1/T3. F1 V2 chưa có, vì vậy chưa có cặp F0/F1 matched. Bốn episode pilot B0/F0 trên seen state 0 đã hoàn tất với 0/4 success; kết quả custom development này được giữ riêng, không gộp vào bảng lịch sử.
- Nguồn raw/refined T7/T8/T9/T10 có receipt pin/hash local; writer audit một demo mỗi task không qua gate đầu ra. Collector snapshot bao gồm kiểm tra hash/index T7/T8/T9/T10. Không tuyên bố full-suite relation/occlusion performance.
- Author-action runtime replay T8/T9/T10 một demo/task dùng helper/evaluator upstream đã hash-check; T9/T10 báo terminal success. Với T8, raw/refined `demo_1` action và initial state trùng chính xác, nhưng replay robot-state vượt dung sai `1e-5` lần đầu tại bước 219 và không terminal. Nguyên nhân vẫn mở; đây là development diagnostic, không phải policy score.
- “Đã viết”, “đã có code”, “đã có artifact”, “đã kiểm chứng giả thuyết” là các mức khác nhau. Không quy đổi thành một tỷ lệ phần trăm hoàn thành.

## Tái tạo tài liệu

Từ thư mục gốc repository, dùng Python cho công cụ tài liệu; không cài thêm package vào environment simulator hoặc policy:

```powershell
python -X utf8 docs/thesis/tools/collect_evidence.py
python -X utf8 docs/thesis/tools/make_figures.py
python -X utf8 docs/thesis/tools/validate_draft.py
```

Collector chỉ đọc artifact và ghi snapshot trong thư mục này, không chạy simulator, policy hay training. Snapshot bao gồm split/alignment V2, replay quyết định, T7/T8/T9/T10 writer audit, author-action runtime diagnostics và T8 state-parity diagnostic. Các lần kiểm tra sau có thể nhìn thấy working tree khác; giữ snapshot đã dùng cho bản nộp cùng Git revision và hashes.

Từ thư mục gốc repository, chạy XeLaTeX và BibTeX trong thư mục chứa `main.tex`:

```powershell
Set-Location docs/thesis/latex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
```

Bản biên tập học thuật gồm 57 trang, đã được biên dịch bằng MiKTeX và render toàn bộ trang; rà contact sheets của toàn bộ 57 trang và kiểm độ phân giải cao các trang 35, 37, 48, 55–57 sau khi bổ sung T8 state-parity; không thấy clipping hoặc overflow. [Mở bản PDF](../../output/pdf/khoa-luan-luu-quang-linh-ban-thao.pdf). SHA-256: `220A5123D933A881BBBAAF882917174877F965A71422CECAF396D298B10CEB81`. Reference venv `.venv_libero_reference_3813` dùng Python 3.8.13 cùng các package pin; strict validator pass và simulator smoke pass 14/14 với provenance assertion đã siết. EGL báo fallback sang `kms_swrast`. Đây chỉ là xác minh môi trường và simulator, không phải policy score. Lời cảm ơn và lời cam đoan vẫn cần tác giả đọc và xác nhận trước khi nộp.

Lời cảm ơn và lời cam đoan là nội dung cá nhân cần tác giả tự hoàn thiện, đọc và xác nhận trước khi nộp; bản thảo không tạo chữ ký hoặc xác nhận thay tác giả.
