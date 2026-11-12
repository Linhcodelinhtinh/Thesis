# Bản thảo khóa luận cập nhật ngày 12/11/2026

**Nâng cao khả năng xử lý tác vụ chuỗi dài cho mô hình VLA trong điều khiển Robot**  
**Sinh viên:** Lưu Quang Linh, 23020392, K68A-AI2  
**Hướng dẫn:** PGS.TS. Nguyễn Việt Hà; PGS.TS. Hoàng Văn Xiêm  
**Ngành:** Trí tuệ nhân tạo, Trường Đại học Công nghệ, Đại học Quốc gia Hà Nội

Thông tin bìa lấy từ hàng 31, trang 4 của quyết định trong `docs/format_reference` và đã được sinh viên xác nhận trong phiên làm việc. Không sử dụng tên sinh viên/năm 2025 còn sót trong mẫu LaTeX.

Đây là **bản thảo đang phát triển**, không phải bản nộp hay tuyên bố đã hoàn thành mục tiêu nâng cao hiệu năng. Snapshot bằng chứng mới nhất được tạo ngày 12/11/2026; snapshot này chỉ tái tổng hợp artifact đã lưu, không chạy thí nghiệm mới. Luận điểm hiện bảo vệ được là: đã xây dựng, huấn luyện và chạy thử một hệ thống bộ nhớ sự kiện thị giác cho SmolVLA; đã có bằng chứng thành phần trên demonstration và rollout policy trong hai task; chưa cô lập được lợi ích riêng của thông tin lịch sử.

## Đọc theo thứ tự

1. [Kế hoạch toàn bộ khóa luận và ranh giới đóng góp](WRITING_PLAN.md).
2. [Ma trận hoàn thiện, bằng chứng và phần phải hoãn](STATUS_AND_CLAIMS.md).
3. [Ghi chú đọc tài liệu và lựa chọn định dạng](REFERENCE_REVIEW.md).
4. [Bản thảo nội dung 5 chương](THESIS_DRAFT.md).
5. [Nguồn LaTeX](latex/main.tex), [thư mục bằng chứng](evidence/snapshot.json), [danh mục tài liệu](latex/references.bib).

`THESIS_DRAFT.md` là nguồn nội dung chính; LaTeX được sinh từ nguồn này bởi `tools/build_latex.py`. Chỉ chỉnh nội dung ở Markdown rồi sinh lại để tránh hai bản khác nhau. Khung bìa, định dạng và phụ lục LaTeX nằm riêng trong `latex/`.

## Quy ước sử dụng số liệu

- Tách V1 không memory khỏi V2 memory; không sửa baseline đã khóa để phù hợp kết quả mới.
- Các kết quả local mang nhãn **LIBERO-derived / NON-COMPARABLE_OFFICIAL_PAPER**. T1/T3 local không đại diện toàn bộ LIBERO-Mem.
- Số liệu rollout chính lấy từ `summary.json` từng episode của run tháng 11, kiểm tra `status.json` và SHA-256. Đây là tên artifact của runner mới, khác `episode.json` của runner cũ.
- Kết quả writer replay ngày 26/10 được báo riêng: sáu demonstration, ba initial state, 12/12 event ghép trong dung sai ±20 frame. Đây là kiểm tra chẩn đoán trên demonstration đã xem, không phải ước lượng tổng quát event accuracy. Các hash evidence được đưa vào `input_hashes.json`.
- Số 77/160 trong báo cáo V1 được lưu ở phần lịch sử: đường raw được báo cáo tham chiếu hiện không có episode để kiểm lại. Không lấy run khác thay vào cùng tên.
- Có checkpoint F1 được train trên dữ liệu tác giả. Không gọi nó là checkpoint đã train trên demonstration derived mới.
- `data/libero_mem_derived` có summary replay T3; output `data/libero_mem_derived_v1` và chuỗi lock/index/shards được xác nhận đầy đủ vẫn chưa có trong snapshot. Metadata replay cũ không chứng nhận dataset version mới.
- “Đã viết”, “đã có code”, “đã có artifact”, “đã kiểm chứng giả thuyết” là các mức khác nhau. Không quy đổi thành một tỷ lệ phần trăm hoàn thành.

## Tái tạo tài liệu

Từ thư mục gốc repository, dùng Python cho công cụ tài liệu; không cài thêm package vào environment simulator hoặc policy:

```powershell
python -X utf8 docs/thesis/tools/collect_evidence.py
python -X utf8 docs/thesis/tools/make_figures.py
python -X utf8 docs/thesis/tools/build_latex.py
python -X utf8 docs/thesis/tools/validate_draft.py
```

Collector chỉ đọc artifact và ghi snapshot trong thư mục này, không chạy simulator, policy hay training. Các lần kiểm tra sau có thể nhìn thấy working tree khác; giữ snapshot đã dùng cho bản nộp cùng Git revision và hashes.

Từ thư mục gốc repository, chạy XeLaTeX và BibTeX trong thư mục chứa `main.tex`:

```powershell
Set-Location docs/thesis/latex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
```

Bản thảo đã được biên dịch bằng MiKTeX 26.5 và rà phân trang. [Mở bản PDF](../../output/pdf/khoa-luan-luu-quang-linh-ban-thao.pdf). Nếu PowerShell đang mở từ trước khi cài MiKTeX, hãy mở phiên mới để nhận `PATH` của Windows.

Lời cảm ơn và lời cam đoan là nội dung cá nhân cần tác giả tự hoàn thiện, đọc và xác nhận trước khi nộp; bản thảo không tạo chữ ký hoặc xác nhận thay tác giả.
