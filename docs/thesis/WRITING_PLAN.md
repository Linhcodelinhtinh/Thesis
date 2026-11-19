# Kế hoạch viết toàn bộ khóa luận

> Các trạng thái gate bên dưới ghi theo ngày của tài liệu. Xem [đối chiếu cập nhật 19/11/2026](EXPERIMENT_CONFIGURATION_REGISTER.md) cho export F1 V2, kết quả E2/E6 và replay Relation V3.


Mốc biên tập và snapshot bằng chứng: **15/11/2026**. Snapshot chỉ đọc artifact đã tồn tại; rollout/training lịch sử giữ ngày chạy riêng. Đề tài chính thức: **Nâng cao khả năng xử lý tác vụ chuỗi dài cho mô hình VLA trong điều khiển Robot**. Giữ nguyên tên đã duyệt; luận văn trình bày phương pháp chung và giới hạn đánh giá hiện tại ở SmolVLA, bộ nhớ sự kiện trong một episode, cùng protocol local T1/T3.

## 1. Lập luận nghiên cứu

Một quan sát có thể phù hợp với nhiều mức tiến độ khác nhau trong tác vụ lặp. Khóa luận nghiên cứu liệu các chuyển tiếp của cùng một cặp bát–đĩa có thể cung cấp bằng chứng về tiến độ cho SmolVLA. Các công trình liên quan cho thấy lợi ích của bộ nhớ phụ thuộc tác vụ và chính sách có thể học tương quan giữa thời gian và tiến độ. Hai khoảng trống được khảo sát là cách xác nhận tiến độ từ quan sát khả dụng và cách xác định giá trị riêng của nội dung lịch sử sau khi kiểm soát thích ứng.

Chương phương pháp liên kết giả thuyết với định nghĩa sự kiện, hai tính chất có chứng minh và thiết kế đối chứng. Chương thực nghiệm đối chiếu mốc chú giải, độ nhạy của mô hình trên dữ liệu cố định và hành vi vòng kín. Chi tiết phiên bản thư viện, phép kiểm mã và nguồn kết quả V1 được đưa vào phụ lục để người đọc có thể kiểm tra phép đo.

## 2. Câu hỏi nghiên cứu và mức trả lời hiện tại

| Mã | Câu hỏi | Có thể trả lời hiện tại | Bằng chứng còn thiếu |
|---|---|---|---|
| RQ1 | Dưới cùng ngân sách ngữ cảnh và mức thích ứng, biểu diễn sự kiện có bằng chứng thay đổi quyết định phụ thuộc lịch sử và thành công đến đâu so với current-only hoặc lịch sử phi cấu trúc? | Pilot B0/F1 thăm dò, loss F1-offline và replay D0–D2 trên trajectory cố định | F0 đối sánh từ corpus V2, can thiệp thông tin, sweep budget và rollout matched trên holdout độc lập |
| RQ2 | Khi lịch sử thiếu, trễ hoặc mâu thuẫn, kiểm tra validity và biểu diễn unknown ảnh hưởng thế nào đến lỗi quyết định? | Writer có unknown/validity; T1/T3 audit lịch sử; T7/T8/T9/T10 mỗi task có một demo writer diagnostic không tạo fact; author-action replay T9/T10 đạt terminal còn T8 lệch terminal dù raw/refined action/state đầu vào trùng, với robot-state drift từ bước 219 | Nhãn độc lập gồm policy failure, precision/recall/timing, decision-window labels, risk–coverage và can thiệp availability; giải thích dynamics discrepancy T8 |
| RQ3 | Cơ chế dùng chung duy trì lợi ích qua dạng phụ thuộc lịch sử và policy nhỏ nào, đồng thời ảnh hưởng kỹ năng cơ sở và chi phí ra sao? | Profile OM/OS/OR/OO có khai báo; data V2, writer diagnostics T7/T8/T9/T10 và replay runtime một demo/task | Audit T7–T10 trên nhãn độc lập, retention của đúng adapter, latency toàn hệ và ít nhất một backbone khác đã xác minh contract |

Các câu hỏi đều chưa có phép kiểm matched đủ điều kiện. Không viết “chứng minh tăng hiệu năng”, “không quên kỹ năng” hoặc “đạt real-time” trong tóm tắt hiện tại.

## 3. Đóng góp, cái kế thừa và giá trị học thuật

| Đóng góp khả dĩ | Thành quả cụ thể của repository | Giá trị | Giới hạn tuyên bố |
|---|---|---|---|
| C1. Phát biểu tiến độ dưới quan sát một phần | Xác định số chu kỳ và phân biệt tiến độ cần đạt, tiến độ bộ ghi xác nhận, tiến độ dùng để chấm | Gắn nội dung lịch sử với quyết định tiếp tục hay dừng | Lập luận về nhập nhằng là lý do thiết kế. Tần suất các trạng thái nhập nhằng chưa được đo |
| C2. Bộ nhớ sự kiện có bằng chứng | Thực thể tham chiếu, xác nhận theo cặp, thời điểm khả dụng, lịch sử chưa đầy đủ, hai tính chất được chứng minh | Cho phép kiểm tra dữ kiện chính sách nhận được trước mỗi quyết định | Các tính chất có điều kiện. Độ đúng của sự kiện là câu hỏi thực nghiệm |
| C3. Khảo sát từ bản ghi đến quyết định | Đối chiếu 12/12 mốc trên demonstration đã xem; replay trigger/query D0–D2; pilot vòng kín B0/F1; audit relation T7/T8/T9/T10 có kết quả âm tính | Tách chất lượng ghi sự kiện, thông tin tới policy và hành vi điều khiển | Development-only; D0–D2 replay cố định không phải policy rollout; T7/T8/T9/T10 một demo mỗi task và không có nhãn độc lập; chưa đo hiệu quả riêng hoặc retention |

Định vị phương pháp ở việc xác nhận cặp sự kiện có ngữ nghĩa trên SmolVLA. Cách đánh chỉ mục bộ nhớ theo sự kiện đã được nghiên cứu trong công trình về lối tắt thời gian. Phần đề xuất của khóa luận là biểu diễn quan hệ bát–đĩa có mốc khả dụng và trạng thái lịch sử chưa đầy đủ, cùng phép đánh giá phù hợp với giả thuyết. SmolVLA, Grounding DINO, LoRA và flow matching là các phương pháp kế thừa.

## 4. Cấu trúc 5 chương và phần viết ngay

Các số trang sau là ngân sách biên tập đề xuất, không phải số trang đã viết hay quy định mới của trường. Hướng dẫn tham khảo khuyên chương mở đầu 3–5 trang và cơ sở lý thuyết không quá 10 trang.

| Phần | Nội dung đến cấp mục | Ngân sách | Viết hiện tại | Hoãn |
|---|---|---:|---|---|
| Phần đầu | Bìa, phụ bìa, cảm ơn, tóm tắt, cam đoan, mục lục, danh mục hình/bảng, ký hiệu | Theo mẫu | Bìa và tóm tắt có giới hạn kết luận; danh mục tự sinh | Cảm ơn/cam đoan cá nhân và tóm tắt kết quả cuối |
| 1. Mở đầu | Động cơ, ba RQ, phạm vi và đóng góp | 3–5 | Động cơ, ba RQ và phạm vi bằng chứng T1/T3 hiện có | Phát biểu về mức cải thiện chưa được kiểm định |
| 2. Cơ sở lý thuyết và nghiên cứu liên quan | Quan sát một phần, các cách biểu diễn lịch sử, lối tắt thời gian, thích ứng, tác vụ và phép đo | Tối đa 10 | Tổng hợp nguồn sơ cấp để định vị biểu diễn và phép đối chứng | Tổng quan rộng ngoài phạm vi giả thuyết |
| 3. Bộ nhớ sự kiện có bằng chứng cho policy thao tác | Phát biểu tiến độ, schema chung, nguồn evidence, validity, projection, hai chứng minh, đối chứng và độ đo | 9–16 | Sơ đồ kiến trúc, timeline khái niệm, giả thiết, profile T1/T3 và giới hạn | Identity/containment trên nhãn độc lập; T7 writer hiện không qua gate |
| 4. Thực nghiệm và thảo luận | Artifact V2, writer audit, replay quyết định, pilot lịch sử, điều kiện và giới hạn | 10–18 | Bằng chứng định lượng có nguồn, mẫu số và nhãn custom | Matched V2 rollout, risk–coverage, retention, intervention policy và labels độc lập |
| 5. Kết luận | Trả lời theo câu hỏi nghiên cứu, ý nghĩa của đóng góp và hướng kiểm định | 2–3 | Kết luận theo phạm vi của bằng chứng | Kết luận về mức cải thiện do lịch sử |
| Sau nội dung | Tài liệu tham khảo; phụ lục provenance, contract, lệnh tái tạo | Theo nhu cầu | BibTeX và ánh xạ claim–artifact | Phụ lục video/case chưa kiểm tra trực quan |

Mục chờ thực nghiệm được ghi trong kế hoạch/backlog, không dựng bảng với số 0 giả, không dùng số ước lượng thay kết quả. Bản thảo chính đã có nội dung cho cả 5 chương; phần còn thiếu được diễn đạt thành giới hạn hoặc nghiên cứu tiếp, không giả làm đã hoàn tất.

## 5. Kỹ thuật đáng đi sâu

| Kỹ thuật | Lý do cần giải thích | Cách trình bày/kiểm chứng |
|---|---|---|
| Partial observability và state aliasing | Phân biệt “nhiều bước” với “thực sự cần lịch sử” | Ví dụ ảnh bát tại cùng đĩa sau 0/1/2 chu kỳ; nói đây là lập luận thiết kế, chưa là test aliasing được đo |
| Action chunk H và execution horizon s | Đổi cadence có thể đổi hành vi dù cùng weights | H=50, s=10 ở Mem; phân biệt V1 s=50; không gộp điểm |
| Contract checkpoint | State 8D, hai camera, action 7D, normalizer và gripper quyết định ý nghĩa input/output | Bảng mapping từ manifest và adapter; dấu -1 mở/+1 đóng, không tự đảo |
| Perception qua worker riêng | Tránh dependency ảnh hưởng simulator/policy; giữ frame identity | Diagram IPC, frame hash, timeout/request ID, allowlist |
| Anchor theo đĩa gốc | Đĩa gần bát hiện tại có thể không phải đĩa cần quay về | Bbox tương đối, gating danh tính, ambiguity và unknown |
| FSM với hysteresis | Đếm sự kiện chuyển trạng thái thay vì số frame/lệnh | Pseudo-code và sơ đồ trạng thái; writer dùng 3 frame, khác counter evaluator |
| Thời điểm có bằng chứng | Một sự kiện xảy ra trước khi có thể xác nhận nó | `occurred_at <= available_at <= query_time`; không backfill prompt bằng nhãn tương lai |
| Incomplete history | Mất detection không đồng nghĩa không có sự kiện | Giữ số đã xác nhận và cờ thiếu lịch sử; “at least” chỉ là phát biểu của estimator, không cận dưới ground truth đã chứng minh |
| Token budget và privilege gate | Tránh đổi instruction hoặc rò rỉ progress từ evaluator | Dùng tokenizer thật, 48 token cho toàn prompt; overflow instruction phải dừng |
| Grouped initial-state split | T1/T3 có thể dùng chung state; frame split dễ rò rỉ | Hash toàn vector, giữ group qua task; phân biệt holdout local và exposure của pretrained model |
| Action-expert LoRA và replay | Điều kiện hóa memory trong khi hạn chế thay đổi mô hình | Công thức LoRA, loss kế thừa, batch 50/50; không nói replay đã bảo đảm retention |
| Phép so sánh matched | B0–F1 trộn nhiều thay đổi | Ma trận B0/F0/F1/F1-off; current-only/sham để tách lịch sử khỏi perception/token effects |
| Metric success/SCR/overshot | SCR=1 vẫn có thể fail nếu thực hiện dư chu kỳ | Công thức endpoint; một case overshot lấy từ log |
| Provenance và uncertainty | Commit không đủ khi tree dirty, T1/T3 không độc lập | Source snapshot/hash; paired theo task/state/seed, cluster theo initial state |

## 6. Thứ tự hoàn thiện theo dependency

| Giai đoạn | Công việc | Đầu ra có thể kiểm tra | Điều kiện chuyển bước |
|---|---|---|---|
| W0: viết ngay | Chốt nguồn/claims, viết chương 1–3, kết quả writer replay và rollout policy đã kiểm được ở chương 4, kết luận tạm | Bộ tài liệu hiện tại | Không có claim mạnh hơn evidence; tách writer audit khỏi policy SR |
| E1: đóng đường dữ liệu | Raw/refined conventions, source/split lock, indexes/shards, metadata/provenance | Split V2 698 demo, indexes T1/T3, shards và alignment audit đã qua | Không dùng artifact khác writer/split làm source cho adapter tương thích |
| E2: đối chứng | F0/F1 cùng dữ liệu, split, ngân sách, checkpoint-selection; F1-off/base rõ ràng | Thiết kế protocol, artifact dữ liệu và F0 V2 no-memory adapter đã có; B0/F0 runtime và pilot development đã chạy | Chưa có F1 V2; không thể đóng phép so F0/F1 hoặc suy hiệu ứng riêng của memory khi model training nằm ngoài phạm vi hiện tại |
| E3: phát triển trên validation | Kiểm writer fidelity, labels và decision-window metrics; chọn candidate theo protocol | Writer T1/T3 cũ và replay D0–D2; T7 one-demo detector-vs-exo-box audit có recall còn thiếu; T7/T8/T9/T10 relation writer chưa qua gate | Chưa có nhãn blind identity/relation hoặc adapter tương thích để mở policy intervention |
| E4: đánh giá cuối | Holdout có lịch sử truy cập rõ; state/seed matched; đo retention và chi phí | Thiết kế protocol và roster đã có | Rollout/retention chưa chạy; không loại failure hoặc đổi split theo kết quả |
| W1: viết kết quả | Bổ sung bảng/hình/case và kết luận theo evidence | Chương 1–5 được cập nhật, abstract được giữ hash-lock | Không sửa abstract nếu tác giả chưa duyệt bản nộp |
| W2: biên tập/nộp | Kiểm PDF, trích dẫn, hình/bảng; tác giả duyệt cam đoan; trao đổi người hướng dẫn | Bản nộp đã đọc toàn bộ | Không còn placeholder, tràn trang, citation thiếu, claim chưa đủ bằng chứng |

Không gán ngày chạy training/benchmark hay hạn nộp khi chưa có lịch xác nhận. Quyết định chỉ xác nhận đợt bảo vệ tháng 12/2026, không cung cấp deadline nộp cụ thể. Viết các phần độc lập tiếp tục trong khi E1–E4 được thực hiện ở công việc nghiên cứu riêng.

Nếu không đủ thời gian hoàn tất E2–E4, bản cuối phải giữ phạm vi **thiết kế, triển khai và đánh giá sơ bộ**, trình bày thẳng các hạn chế. Không đổi tên đề tài đã duyệt hoặc tự nhận đạt mục tiêu nâng cao hiệu năng; trao đổi với người hướng dẫn về mức kết quả phù hợp cho bản nộp.
