# Kế hoạch viết toàn bộ khóa luận

Mốc biên tập và snapshot bằng chứng: **12/11/2026**. Snapshot chỉ đọc lại artifact đã tồn tại; rollout/training được trình bày trong chương thực nghiệm mang ngày chạy riêng. Đề tài chính thức: **Nâng cao khả năng xử lý tác vụ chuỗi dài cho mô hình VLA trong điều khiển Robot**. Giữ nguyên tên đã duyệt; trong nội dung xác định phạm vi thực hiện là SmolVLA, bộ nhớ sự kiện thị giác trong một episode và đánh giá T1/T3 dưới protocol local.

## 1. Câu chuyện khoa học

Một quan sát robot hiện tại có thể không cho biết một thao tác lặp đã được thực hiện bao nhiêu lần. Khóa luận khảo sát cách biến bằng chứng thị giác theo thời gian thành bộ nhớ sự kiện ngắn gọn, rồi đưa bộ nhớ vào một VLA nhỏ bằng giao diện văn bản và thích ứng action expert. Muốn biết phương pháp có ích, trước hết phải bảo đảm nguồn thông tin không rò rỉ từ bộ chấm điểm, trajectory huấn luyện phù hợp runtime, và phép so sánh tách adaptation khỏi memory.

Đây là câu chuyện về **thiết kế và đánh giá có kiểm soát một phương pháp cụ thể**, không phải danh sách module đã viết. Baseline V1 cung cấp nền tảng; đóng góp trung tâm nằm ở V2. Kết quả âm hoặc chưa rõ lợi ích vẫn có thể có giá trị khi giới hạn được đo và lý giải bằng chứng, nhưng không tự biến thiếu kết quả thành đóng góp mới.

## 2. Câu hỏi nghiên cứu và mức trả lời hiện tại

| Mã | Câu hỏi | Có thể trả lời hiện tại | Bằng chứng còn thiếu |
|---|---|---|---|
| RQ1 | Có thể nối bộ nhớ sự kiện từ RGB vào SmolVLA mà giữ ranh giới thông tin và contract policy không? | Đã có implementation, checkpoint và rollout thật trong phạm vi T1/T3; các cơ chế causal/provenance mô tả được | Đánh giá fidelity của sự kiện và kiểm tra leak trên đường dữ liệu hoàn chỉnh |
| RQ2 | Thông tin lịch sử giúp policy giải quyết T3 tốt hơn khi kiểm soát adaptation không? | Có B0/F1 sơ bộ; chưa cô lập được memory | F0 cùng data/budget, current-only hoặc sham cùng token budget, F1-off có mode được audit, đủ state/seed |
| RQ3 | Thích ứng memory có bảo toàn kỹ năng LIBERO base không? | Có cơ chế replay và một số run fine-tune cũ; chưa có bằng chứng matched cho adapter hiện tại | Đo chính F1 trên base với cùng protocol/checkpoint identity; margin retention xác định trước |
| RQ4 | Chất lượng writer, quy ước ảnh, runtime và chi phí ảnh hưởng kết quả thế nào? | Có trace, audit và quan sát thất bại để mô tả | Event precision/recall/delay với nhãn độc lập; profiling end-to-end; can thiệp có đối chứng để kết luận nguyên nhân |

RQ2 và RQ3 là câu hỏi mở. Không viết “chứng minh tăng hiệu năng”, “không quên kỹ năng” hoặc “đạt real-time” trong tóm tắt hiện tại.

## 3. Đóng góp, cái kế thừa và giá trị học thuật

| Đóng góp khả dĩ | Thành quả cụ thể của repository | Giá trị | Giới hạn tuyên bố |
|---|---|---|---|
| C1. Hệ thống đánh giá có truy vết | Tách simulator/policy/perception; khóa source, controller, split; log action, prompt, evidence, status | Giúp phát hiện và phân biệt lỗi tài nguyên, interface, thí nghiệm; tạo điều kiện kiểm chứng lại | Đóng góp thực nghiệm/kỹ thuật; không phải một benchmark chính thức mới hay chứng minh tái lập tuyệt đối |
| C2. Bộ nhớ sự kiện theo thực thể | Tracker bát–đĩa gốc, FSM departure/return, hysteresis, `occurred_at`/`available_at`, incomplete history, renderer có budget; audit demo ghép 12/12 return trong ±20 frame | Biểu diễn lịch sử có thể kiểm tra; có kiểm tra ban đầu về thứ tự/timing event | Chỉ sáu demonstration, ba state, milestone không phải contact label; chưa chứng minh mới toàn cầu hoặc fidelity tổng quát |
| C3. Quy trình thích ứng gọn | LoRA q/v action expert, backbone gốc được bảo toàn, replay base/Mem, split theo initial-state hash, export unmerged có hash | Cho phép khảo sát memory conditioning trên mô hình nhỏ với dữ liệu và chi phí kiểm soát | LoRA, flow matching, Grounding DINO, replay là kỹ thuật kế thừa; tỷ lệ 50/50 chưa được chứng minh tối ưu |
| C4. Bằng chứng thực nghiệm và phân tích giới hạn | Checkpoint 3.000 bước; B0/F1 trên cùng roster T1/T3; phân biệt loss, SCR, success, overshot | Cho thấy offline loss và khả năng vận hành không đủ thay thế closed-loop success; xác định phép kiểm định còn cần | Chưa có kết quả xác nhận đóng góp riêng của memory; không quy failure thành “quên” hoặc “attention dilution” |

Điểm khác với MEM và MemoryVLA là lựa chọn **bộ nhớ sự kiện tường minh bên ngoài policy**, kiểm soát nguồn thông tin và thích ứng nhẹ cho một backbone nhỏ. Đây là khác biệt thiết kế có thể mô tả. Muốn nâng thành đóng góp phương pháp đã được xác nhận cần ablation và chứng minh lợi ích/chi phí tương ứng. Không nhận video encoder, PCMB, slot-state-space model hay SGS là thành phần tự phát triển.

## 4. Cấu trúc 5 chương và phần viết ngay

Các số trang sau là ngân sách biên tập đề xuất, không phải số trang đã viết hay quy định mới của trường. Hướng dẫn tham khảo khuyên chương mở đầu 3–5 trang và cơ sở lý thuyết không quá 10 trang.

| Phần | Nội dung đến cấp mục | Ngân sách | Viết hiện tại | Hoãn |
|---|---|---:|---|---|
| Phần đầu | Bìa, phụ bìa, cảm ơn, tóm tắt, cam đoan, mục lục, danh mục hình/bảng, ký hiệu | Theo mẫu | Bìa và tóm tắt có giới hạn kết luận; danh mục tự sinh | Cảm ơn/cam đoan cá nhân và tóm tắt kết quả cuối |
| 1. Mở đầu | 1.1 Bối cảnh; 1.2 bài toán; 1.3 mục tiêu/RQ; 1.4 phạm vi/đóng góp; 1.5 bố cục | 3–5 | Viết thành văn, có trích dẫn và ví dụ lặp bát | Tuyên bố mức cải thiện chưa đo |
| 2. Cơ sở lý thuyết và nghiên cứu liên quan | 2.1 VLA/quan sát một phần; 2.2 SmolVLA/chunk; 2.3 memory; 2.4 detector/event; 2.5 LoRA/loss/replay; 2.6 benchmark và độ đo | 7–10 | Các khái niệm sẽ dùng ở chương 3–4; so sánh cơ chế với MEM/MemoryVLA/LIBERO-Mem | Tổng quan RL dài, SGS như một thuật toán đang dùng, lý thuyết phổ thông không phục vụ bài toán |
| 3. Bộ nhớ sự kiện thị giác cho SmolVLA | 3.1 hình thức hóa; 3.2 kiến trúc/ranh giới; 3.3 tracker; 3.4 FSM; 3.5 causal renderer; 3.6 dữ liệu/split; 3.7 thích ứng; 3.8 protocol; 3.9 giới hạn | 12–16 | Phương pháp đã implement; công thức, pseudo-code, diagram, ví dụ khái niệm có nhãn | Derived export chưa chứng nhận, multi-object/OO/OR chưa làm |
| 4. Thực nghiệm và thảo luận | 4.1 RQ/thiết kế; 4.2 runtime/provenance; 4.3 data/training; 4.4 B0/F1; 4.5 case/giới hạn; 4.6 phép đo còn thiếu | 12–18 | Bảng số liệu kiểm được, loss, local protocol, giới hạn thống kê | Kết quả F0/F1-off, retention F1, multi-seed, event accuracy, paper leaderboard |
| 5. Kết luận | 5.1 thành quả; 5.2 điều rút ra; 5.3 tồn tại; 5.4 hướng tiếp | 2–3 | Kết luận tạm theo evidence hiện tại | Kết luận cải thiện cuối cùng |
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
| E1: đóng đường dữ liệu | Audit source thay đổi ngày 11/11; chạy export version mới; convention raw/refined, resume, terminal, metadata, provenance | Dataset derived đầy đủ có manifest, parent lock, indices/shards, validation ảnh thật | Mỗi artifact hoàn tất; không dùng pilot cũ thay version mới |
| E2: đối chứng | F0/F1 cùng dữ liệu, split, ngân sách, checkpoint-selection; hỗ trợ F1-off/base rõ ràng | Hai training runs + export đã xác minh | Không lấy checkpoint cũ khác data làm matched F0 |
| E3: phát triển trên validation | Kiểm writer fidelity, rollout pilot, chọn candidate theo metric đặt trước | Validation report và quyết định khóa | Không chọn checkpoint trên test đã xem |
| E4: đánh giá cuối | Holdout có lịch sử truy cập rõ; state/seed matched; đo retention và chi phí | Episode logs, báo cáo per-task, uncertainty theo cluster | Không loại failure hoặc đổi split vì kết quả policy |
| W1: viết kết quả cuối | Bổ sung bảng/ablation/case, sửa abstract và conclusion | Chương 4–5 phản ánh đúng mức chứng minh | Số liệu truy được đến logs; kết quả âm vẫn được báo |
| W2: biên tập/nộp | Kiểm PDF, trích dẫn, hình/bảng; tác giả duyệt cam đoan; trao đổi người hướng dẫn | Bản nộp đã đọc toàn bộ | Không còn placeholder, tràn trang, citation thiếu, claim chưa đủ bằng chứng |

Không gán ngày chạy training/benchmark hay hạn nộp khi chưa có lịch xác nhận. Quyết định chỉ xác nhận đợt bảo vệ tháng 12/2026, không cung cấp deadline nộp cụ thể. Viết các phần độc lập tiếp tục trong khi E1–E4 được thực hiện ở công việc nghiên cứu riêng.

Nếu không đủ thời gian hoàn tất E2–E4, bản cuối phải giữ phạm vi **thiết kế, triển khai và đánh giá sơ bộ**, trình bày thẳng các hạn chế. Không đổi tên đề tài đã duyệt hoặc tự nhận đạt mục tiêu nâng cao hiệu năng; trao đổi với người hướng dẫn về mức kết quả phù hợp cho bản nộp.
