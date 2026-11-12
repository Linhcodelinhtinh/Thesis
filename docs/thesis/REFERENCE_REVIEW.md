# Ghi chú đọc tài liệu tham chiếu

Tài liệu gốc giữ nguyên trong `docs/format_reference`. Đã trích text của cả năm PDF để tra cứu, đọc phần hướng dẫn, quyết định, phương pháp và thực nghiệm của các paper; kiểm tra trực quan trang quy cách hình/bảng, hàng đề tài và các hình kiến trúc liên quan. Appendix và danh mục tham khảo của paper dùng để tra cứu, không mặc nhiên coi mọi tuyên bố trích dẫn gián tiếp là đã được xác minh tại nguồn gốc.

SHA-256 của PDF và source project có trong [input_hashes.json](evidence/input_hashes.json). Phiên bản paper dưới đây theo chính metadata/nội dung PDF được cung cấp, không theo tên file suy đoán.

## 1. Hướng dẫn viết khóa luận năm 2023

**Nguồn:** Nguyễn Việt Hà, *Một số gợi ý về trình bày khóa luận tốt nghiệp*, 13 trang.

Đây là tài liệu hướng dẫn tham khảo; ngay trang 1 yêu cầu tuân thủ người hướng dẫn và quy định trường. Không biến mọi gợi ý thành quy định hành chính bắt buộc.

| Nội dung đã đọc | Áp dụng vào bản thảo |
|---|---|
| Trang1–2: cấu trúc 5 chương, tương ứng IMRAD | Giữ mở đầu, lý thuyết, phương pháp, thực nghiệm/thảo luận, kết luận |
| Trang3–4: mở đầu cần lập luận và nguồn; nên 3–5 trang | Đưa trực tiếp bài toán lịch sử bát–đĩa vào động lực, tránh mở đầu chung chung về AI |
| Trang4: lý thuyết phục vụ phần sau, không quá10 trang | Chỉ giữ VLA/partial observability, memory, LoRA/loss, benchmark/metric |
| Trang5–6: định nghĩa bài toán, ví dụ, ý tưởng trước chi tiết, phân biệt kế thừa/đề xuất | Hình thức hóa causal memory và nêu rõ detector/backbone/loss/LoRA được kế thừa |
| Trang6–8: mô tả thí nghiệm đã làm, dữ liệu, công cụ, kiểm thử và phân tích | Chỉ viết quá khứ cho run có artifact; không lấy software tests thay hiệu quả policy |
| Trang8–9: kết luận phải rút ra điều có ý nghĩa, không chỉ liệt kê việc đã làm | Kết luận về khoảng cách offline/closed-loop và điều kiện cần để kiểm chứng memory |
| Trang10: hình/bảng nhất quán, có lời dẫn, vẽ lại khi kế thừa ý tưởng | Hình kiến trúc và biểu đồ tự tạo từ source/logs; không chép hình paper vào thesis |
| Trang10–11: trích đúng nơi, tham khảo sau kết luận trước phụ lục; gợi ý10–20 nguồn | BibTeX đánh số theo xuất hiện, dẫn nguồn ngay cạnh ý dùng; không đưa source chưa sử dụng cho đủ số |
| Trang11–13: văn phong khoa học, thời nhất quán, hạn chế bullet, khởi–thừa–chuyển–kết | Bản thảo chính viết thành đoạn; checklist chỉ ở tài liệu quản lý viết |

## 2. Mẫu LaTeX

Mẫu `format latex/main.tex` xác định: A4, lề trên2,5cm/dưới3cm/trái3cm/phải2cm; thụt đầu dòng1cm; cách đoạn6pt; giãn dòng1,3; số trang giữa chân trang; caption bảng phía trên và hình phía dưới; có bìa/phụ bìa, cảm ơn, tóm tắt, cam đoan và danh mục trước 5 chương.

Hai chỗ phải xử lý khi dùng mẫu:

1. `\documentclass[13pt]{book}` không bảo đảm 13pt vì lớp `book` chuẩn chỉ nhận10/11/12pt. Khung mới dùng KOMA `scrbook` với `fontsize=13pt` và XeLaTeX để chỉ định cỡ/font rõ. Đây là sửa kỹ thuật của template, chưa phải chứng nhận PDF cuối.
2. Tên sinh viên, năm2025 và các tiêu đề placeholder trong mẫu không phải metadata của đề tài này. Đã thay bằng thông tin người dùng xác nhận; giữ bản mẫu gốc nguyên vẹn.

Mẫu dùng `plain` cho BibTeX (sắp theo tác giả), còn hướng dẫn gợi ý thứ tự xuất hiện cho trích dẫn số. Bản thảo chọn `unsrt` nhất quán với lựa chọn sau; có thể đổi theo yêu cầu chính thức của người hướng dẫn. Không suy ra số trang toàn khóa luận bắt buộc từ mẫu.

## 3. Quyết định giao đề tài

PDF năm2026 có 5 trang. Hàng31/trang 4 ghi đề tài, sinh viên và hai cán bộ hướng dẫn đúng thông tin đã xác nhận. Quyết định xác định đợt bảo vệ tháng12/2026. Không tự điền số quyết định/ngày ký bị thiếu trong text extraction, không dùng ngày sinh vào bìa và không suy deadline nộp cụ thể. Đây là nguồn hành chính cho metadata, không phải tài liệu nền tảng kỹ thuật để trích trong chương 2.

## 4. MEM: Multi-Scale Embodied Memory for Vision Language Action Models

**Bản đọc:** Torne và cộng sự, arXiv:2603.03596v2, 2026, 15 trang. [Nguồn tác giả](https://arxiv.org/abs/2603.03596v2).

Paper tách nhu cầu lịch sử thành hai thang: lịch sử ảnh ngắn phục vụ động học/che khuất và tóm tắt ngôn ngữ dài phục vụ sự kiện ngữ nghĩa. Kiến trúc phối hợp high-level policy cập nhật memory/subtask với low-level policy sinh action. Video encoder xen attention không gian và thời gian, nén lịch sử trước VLM; language memory được học cập nhật thay vì nối vô hạn mọi instruction.

Điểm quan trọng khi đọc thực nghiệm là các tác giả dùng ablation loại từng dạng memory, so sánh kiểu nén và phân biệt pretraining memory với chỉ post-training. Bản thân chuỗi subtask instruction không đồng nghĩa sự kiện đã xảy ra; failure lặp lúc inference tạo khác biệt so với demonstration. Bài học phù hợp cho khóa luận là ghi sự kiện có bằng chứng, kiểm tra train–test shift và có đối chứng thông tin, không chỉ nối text vào prompt.

Khóa luận hiện không triển khai video encoder, high-level language memory policy hoặc π0.6-MEM. Không mượn kết quả robot thật/horizon 15 phút của paper cho hệ thống local. Chỉ dùng paper để định vị lựa chọn event memory và thiết kế ablation.

## 5. MemoryVLA

**Bản đọc:** Shi và cộng sự, *MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation*, arXiv:2508.19236v2; bản PDF ghi ICLR2026; 36 trang. [Nguồn tác giả](https://arxiv.org/abs/2508.19236v2).

Phương pháp xây dựng PCMB gồm biểu diễn perceptual và cognitive; current working memory truy xuất lịch sử bằng attention có thời gian, kết hợp qua gate và hợp nhất các mục liền kề tương đồng khi đầy. Action expert điều kiện hóa trên các biểu diễn này. Đây là memory trong không gian đặc trưng được học, khác bộ đếm sự kiện và giao diện văn bản bên ngoài SmolVLA.

Các ablation ở mục4.6 so sánh loại memory, độ dài, temporal positional encoding, fusion và consolidation. Điều đáng học là một kiến trúc memory cần kiểm định từng lựa chọn; dung lượng lớn hơn không tự bảo đảm kết quả tốt hơn. Evaluation setup mô tả model, input, train/validation và số lượt chạy; so sánh đúng phải giữ những khác biệt đó trong tầm nhìn.

Không đặt SR của MemoryVLA và SR local cùng bảng xếp hạng: khác model, dữ liệu, camera, training, task coverage và protocol. Khóa luận cũng không tuyên bố đã tái hiện PCMB hoặc huấn luyện theo recipe 8×A100 của paper.

## 6. A Balanced Data Diet

**Bản đọc:** Zhang và cộng sự, *A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control*, arXiv:2610.12465v1; PDF ghi CoRL2026; 21 trang. [Nguồn tác giả](https://arxiv.org/abs/2610.12465v1).

SGS điều chỉnh xác suất chọn cấu hình huấn luyện theo success gần đây, ưu tiên các cấu hình ở mức khó vừa phải bằng weighting dạng beta. Paper khảo sát RL với nhiều môi trường song song, đối chứng uniform/PLR, nhiều seed và chuyển từ state policy sang RGB policy. Tác giả không nhận nguyên lý curriculum hoàn toàn mới và nêu giới hạn của tập cấu hình hữu hạn, nguồn reset và sim-to-real gap.

Liên quan trực tiếp với khóa luận này ở mức phương pháp luận: phân phối kinh nghiệm và đối chứng ngân sách ảnh hưởng ý nghĩa kết quả. **SGS không phải replay50/50**, không phải supervised fine-tuning và không phải lý do để kết luận50/50 là tỷ lệ tốt nhất. Giữ một đoạn ngắn ở nghiên cứu liên quan; bỏ chương RL/SGS riêng vì không có implementation hoặc thí nghiệm tương ứng trong thesis.

## 7. Nguồn sơ cấp bổ sung cho phần sẽ viết

| Citation key | Nguồn/phiên bản | Vai trò |
|---|---|---|
| `libero2023` | Liu và cộng sự, arXiv:2306.03310v1 | Benchmark gốc, nguồn task và phạm vi |
| `smolvla2025` | Shukor và cộng sự, arXiv:2506.01844v1 | VLM nhỏ, action expert, chunking |
| `liberomem2025` | Chung và cộng sự, arXiv:2511.11478v3 | Task cần lịch sử thực thể, Embodied-SlotSSM; phân biệt paper version với code pin |
| `groundingdino2024` | Liu và cộng sự, arXiv:2303.05499v5 | Detector văn bản–ảnh được kế thừa |
| `lora2021` | Hu và cộng sự, arXiv:2106.09685v1 | Thích ứng hạng thấp |
| `flowmatching2023` | Lipman và cộng sự, arXiv:2210.02747v2 | Cơ sở loss sinh action; không tự sửa implementation policy |
| `smolvlaCheckpoint` | HF `lerobot/smolvla_libero` @ `31d453f7edd78c839a8bbc39744a292686daf0de` | Contract thực tế của checkpoint local |
| `liberoSource` | Official LIBERO @ `8f1084e3132a39270c3a13ebe37270a43ece2a01` | Nguồn phần baseline |
| `liberomemSource` | `libero-mem/libero-mem` @ `0eee0defa4024e51511b6e15a79f30e9620209de` | Nguồn T1/T3; không dùng short pin cũ6df130c của docs tháng10 |
| `artifact2026` | Snapshot nội bộ11/11/2026 và run manifests | Bằng chứng số liệu local; không phải công bố đã phản biện |

Metadata paper bổ sung được kiểm trên arXiv; cấu hình sử dụng trong thực nghiệm lấy từ local manifest/code đã pin, không lấy bản mới nhất trên mạng thay nguồn đã chạy. Mọi citation trong bản thảo có entry BibTeX; giữ full hashes trong evidence để tránh nhồi chuỗi hash vào mạch lập luận.
