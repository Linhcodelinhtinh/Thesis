# Định vị lại nghiên cứu và kế hoạch hoàn thiện luận văn

Ngày: 13/11/2026. Tài liệu tư vấn nghiên cứu và kế hoạch làm việc, không phải chương luận văn hay báo cáo kết quả mới. Đã đối chiếu bản LaTeX đang được biên tập, mã nguồn, bảng bằng chứng ngày 12/11 và nguồn nghiên cứu sơ cấp. Không chạy thêm huấn luyện hoặc benchmark trong lần rà soát này. Giữ nguyên toàn bộ nội dung abstract do tác giả sửa.

Kế hoạch triển khai cụ thể, vị trí code và điều kiện nghiệm thu nằm trong [kế hoạch thực thi](IMPLEMENTATION_AND_WRITING_ROADMAP_2026-11-13.md). Các đường dẫn kỹ thuật trong tài liệu làm việc đó không được đưa sang luận văn.

## 1. Chẩn đoán: vấn đề nằm ở cấu trúc lập luận

Các nhận xét yêu cầu một thay đổi về đối tượng nghiên cứu. Đối tượng nên là **cơ chế lưu giữ và sử dụng lịch sử tương tác cho chính sách robot một tay, đặc biệt các VLA nhỏ có ngân sách ngữ cảnh và tính toán hạn chế**. SmolVLA là một hiện thực để kiểm chứng. LIBERO-Mem là môi trường đo. T1/T3 là hai trường hợp chẩn đoán ban đầu.

Trong bản thảo hiện tại, tên chương phương pháp, phát biểu mục tiêu, RQ1/RQ2, phạm vi đóng góp và các định nghĩa toán học đều đi từ bát–đĩa–SmolVLA. Do đó, chỉ đổi tên chương hoặc thêm câu “có thể tổng quát hóa” sẽ không giải quyết được vấn đề. Cần thay cả chuỗi **vấn đề → câu hỏi → giả thuyết → thành phần phương pháp → đối chứng → kết luận**.

Tuy vậy, mở rộng cách phát biểu không tự mở rộng bằng chứng. Theo tập bằng chứng hiện có, writer RGB và renderer vẫn chuyên biệt cho T1/T3; harness đang chạy B0/F1; bảng vòng kín có B0 0/20 và F1 1/20 trên 10 nhóm trạng thái, một seed. Chưa có F0 được thích ứng tương đương, chưa có đánh giá toàn bộ mười tác vụ hoặc chuyển sang backbone khác. Đây là mốc phát triển để thiết kế nghiên cứu tiếp, không phải căn cứ cho kết luận phương pháp đã tổng quát.

Một vấn đề đặc biệt cần lưu ý: abstract đang dùng các phát biểu khẳng định hiệu quả. Tôn trọng yêu cầu giữ nguyên file; chuyển những phát biểu ấy thành **nghĩa vụ bằng chứng** cho thực nghiệm cuối. Nếu bằng chứng sau cùng không hỗ trợ, phải báo tác giả về bất nhất, không tự tạo số liệu hay xem abstract là bằng chứng.

### Những điểm nên đồng ý và những điểm cần làm chặt hơn

| Nhận xét | Hướng xử lý | Điều kiện để lập luận có sức nặng |
|---|---|---|
| Nhấn mạnh model nhỏ | Đưa ngân sách tài nguyên vào ngay phát biểu bài toán | Đo cả policy, perception, memory và giao tiếp; không chỉ đếm tham số LoRA |
| Memory là hệ sự kiện–bằng chứng | Định nghĩa trạng thái có cấu trúc, nguồn chứng cứ, thời điểm khả dụng và mức chưa biết | Các trường này phải thay đổi cách cập nhật/chọn thông tin, không chỉ nằm trong log |
| Text là projection | Tách trạng thái nhớ khỏi biểu diễn cấp cho policy | Có phép kiểm semantic fidelity và token budget của projection |
| Không đóng khung T1/T3 | Xây biểu diễn cho chu kỳ, danh tính, quan hệ và containment | Có thử nghiệm ngoài đếm chu kỳ; mở rộng code thật tới T1–T10 |
| Method cần novelty | Chọn cơ chế cụ thể và ablation trực tiếp | Không nhận tính mới cho FSM, text memory hoặc timestamp nói chung |
| Tăng liên kết văn phong | Sửa quan hệ giữa các luận điểm trước khi thêm từ nối | Mỗi đoạn có một câu hỏi, một lập luận và một hệ quả cho đoạn tiếp |
| Không dẫn file local trong thesis | Thay bằng bảng protocol, nguồn công khai, version và định danh thực nghiệm | Tài liệu tự đủ để đọc; repository/DOI chỉ dẫn khi đã tồn tại thật |

## 2. Định vị trong nghiên cứu hiện hành

Không nên mở đầu bằng mệnh đề “VLA chưa có bộ nhớ”. Vấn đề phù hợp hơn là **biểu diễn nào giữ được lịch sử cần cho quyết định khi bộ nhớ cũng sai, thiếu và bị giới hạn chi phí**.

Các tài liệu dưới đây làm thay đổi trực tiếp cách đặt câu hỏi. Chúng là nguồn sơ cấp, được kiểm tra trực tuyến ngày 13/11/2026; mã phiên bản paper được giữ trong liên kết. Không lấy điểm công bố của chúng để so ngang với protocol cục bộ.

| Công trình | Liên hệ trực tiếp | Hệ quả đối với luận văn |
|---|---|---|
| [LIBERO-Mem / Embodied-SlotSSM, v3](https://arxiv.org/abs/2511.11478v3) | Lịch sử theo thực thể; chuỗi, quan hệ, chuyển động và che khuất | “Nhớ vật thể theo thời gian” đã có tiền lệ; cần chỉ rõ vai trò của bằng chứng và projection |
| [MEM, v2](https://arxiv.org/abs/2603.03596v2) | Kết hợp lịch sử video ngắn với bộ nhớ ngôn ngữ dài | Text memory hoặc nén lịch sử đơn thuần không đủ làm novelty |
| [MemoryVLA, v2](https://arxiv.org/abs/2508.19236v2) | Ngân hàng đặc trưng tri giác–nhận thức, truy xuất và hợp nhất | Cần đối chiếu thông tin giữ lại, chi phí và khả năng kiểm tra của biểu diễn |
| [RoboMME, v3](https://arxiv.org/abs/2603.04639v3) và [RMBench, v4](https://arxiv.org/abs/2603.01229v4) | Đánh giá nhiều loại nhu cầu bộ nhớ và thiết kế policy | Không đồng nhất horizon dài với memory khó; báo kết quả theo loại phụ thuộc lịch sử |
| [SmoLSTM, v1](https://arxiv.org/abs/2609.22854v1) | VLA nhỏ dùng trạng thái hồi quy duy trì trong episode | “Memory cho VLA nhỏ” đã là hướng có kết quả; cần so với lựa chọn trạng thái tiềm ẩn có chi phí hữu hạn |
| [Is Your Robot’s Memory Just a Clock?, v1](https://arxiv.org/abs/2611.14488v1) | Temporal shortcut và đánh chỉ mục theo sự kiện | Không nhận novelty cho event indexing; dùng null-event intervention như một phép kiểm kế thừa |
| [Beyond Success Rate / IMEP, v1](https://arxiv.org/abs/2610.23372v1) | Thay dữ kiện mục tiêu, đối chứng giữ nguyên ý nghĩa, phân biệt quyết định và thực thi | Can thiệp memory không tự là đóng góp mới; áp dụng protocol với thông tin tường minh của hệ đề xuất |
| [Reward Machines, ICML 2018](https://proceedings.mlr.press/v80/icarte18a.html) | Biểu diễn tiến trình bằng máy trạng thái | Có cơ sở lý thuyết cho tiến độ theo sự kiện; phải phân biệt máy trạng thái mục tiêu với bộ nhớ suy từ cảm biến |

Hai bài SmoLSTM và IMEP đặc biệt cần bổ sung vào bản thảo hiện tại. Với Clock, cần đọc cả giới hạn: một tiêu chuẩn dựa trên việc cảnh trở lại trạng thái cũ có thể xóa nhầm chu kỳ hợp lệ. Như vậy, sự kiện cần mang **ngữ nghĩa chuyển tiếp**, không chỉ đánh dấu ảnh mới khác ảnh cũ.

**Khoảng trống đề xuất — còn cần kiểm chứng qua rà soát sâu:** các thành phần memory hiệu quả, mô hình nhỏ và kiểm định sử dụng memory đã được nghiên cứu. Điểm luận văn có thể khảo sát là việc kết hợp một **trạng thái sự kiện có điều kiện bằng chứng** với **projection bảo toàn ngữ nghĩa chưa biết dưới ngân sách nhỏ**, rồi xác định miền mà thiết kế này cải thiện quyết định vòng kín. Không khẳng định “đầu tiên” hoặc các công trình khác không xử lý uncertainty chỉ từ việc đọc abstract.

### Tên đề tài và luận điểm trung tâm

Tên học thuật đề xuất: **Bộ nhớ sự kiện có bằng chứng cho chính sách thao tác robot nhỏ gọn trong điều kiện quan sát một phần**.

Nếu tên đề tài đã đăng ký phải giữ nguyên, dùng câu trên làm định vị ở mở đầu và tên chương phương pháp; không cần tự đổi tên trên bìa.

Luận điểm để kiểm định:

> Với các tác vụ mà quyết định phụ thuộc vào lịch sử tương tác, một bộ nhớ có cấu trúc gắn sự kiện với thực thể, bằng chứng và mức độ khả dụng có thể giúp chính sách nhỏ sử dụng lịch sử hiệu quả hơn trong cùng ngân sách đầu vào. Lợi ích này phụ thuộc vào chất lượng quan sát, cách biểu diễn điều chưa biết và việc giữ lại đúng dữ kiện tại thời điểm ra quyết định.

“Có thể” là giả thuyết. Kết luận phải xác định nơi hiệu quả, nơi thất bại và chi phí, thay vì tuyên bố memory luôn có lợi.

## 3. Bộ câu hỏi nghiên cứu đề xuất

### Câu hỏi bao trùm

> **Làm thế nào biểu diễn và cung cấp lịch sử tương tác có bằng chứng để các chính sách VLA nhỏ gọn đưa ra quyết định đúng trong tác vụ thao tác phụ thuộc lịch sử, khi ngữ cảnh đầu vào, tài nguyên tính toán và độ đầy đủ của quan sát đều bị giới hạn?**

Câu hỏi này xác định đối tượng là memory và năng lực quyết định của policy. Model, tokenizer và benchmark xuất hiện khi vận hành hóa, không nằm trong bản thân câu hỏi.

Khuyến nghị dùng ba RQ chính. Khả năng duy trì kỹ năng, tính tái lập và độ đúng của protocol là điều kiện đánh giá xuyên suốt; không cần tách mỗi yêu cầu kỹ thuật thành một RQ.

### RQ1 — Giá trị của biểu diễn dưới giới hạn ngữ cảnh

> **Trong cùng ngân sách đầu vào và thích ứng mô hình, biểu diễn lịch sử bằng trạng thái sự kiện–bằng chứng có cải thiện quyết định phụ thuộc lịch sử và tỷ lệ hoàn thành tác vụ so với quan sát hiện tại hoặc lịch sử phi cấu trúc không; lợi ích thay đổi thế nào khi ngân sách ngữ cảnh giảm?**

Đây là câu hỏi về **chọn thông tin để giữ**, không phải “có xây được module memory hay không”. Nó quan trọng vì tăng số khung hình hoặc văn bản không phải lựa chọn miễn phí đối với policy nhỏ.

Giả thuyết H1: tại các ngân sách nhỏ đã đăng ký trước, trạng thái có cấu trúc giữ được biến lịch sử quyết định hành động tốt hơn một bản liệt kê gần nhất. Tuy nhiên, nếu tác vụ cần chi tiết hình học mà schema bỏ mất, lịch sử thị giác có thể tốt hơn; kết quả ấy phải được giữ lại.

Phép kiểm bắt buộc:

- So F1 với F0 cùng dữ liệu, ngân sách và lịch chọn checkpoint để đo lợi ích của cấu hình dùng memory sau thích ứng.
- Thêm current-only dùng cùng perception để tách lợi ích lịch sử khỏi lợi ích có thêm detector.
- Thêm lịch sử phẳng từ cùng dòng thông tin; giữ trần token và quy trình thích ứng tương đương.
- Đo đường hiệu năng theo ngân sách, tỷ lệ dữ kiện cần thiết còn hiện diện đúng lúc quyết định, lỗi lặp/dừng sớm/chọn sai đối tượng, và SR toàn tác vụ.
- Can thiệp một dữ kiện tiến độ/danh tính, có đối chứng đổi cách diễn đạt nhưng giữ nghĩa. Thay đổi hành động đơn thuần chưa chứng minh sử dụng đúng thông tin.

H1 không được hỗ trợ nếu lợi ích mất khi đối sánh perception và training, nếu chỉ giảm loss nhưng không cải thiện quyết định/vòng kín, hoặc nếu advantage chỉ xuất hiện ở một mức ngân sách được chọn hậu nghiệm.

Độ quan trọng đối với field: kết quả có thể chỉ ra một nguyên tắc phân bổ ngữ cảnh cho policy bị giới hạn, thay vì chỉ cho thêm một bảng điểm của một model.

### RQ2 — Độ tin cậy khi lịch sử không đầy đủ

> **Khi bằng chứng bị thiếu, chậm hoặc mâu thuẫn, cơ chế duy trì tính hợp lệ của sự kiện và biểu diễn điều chưa biết ảnh hưởng thế nào đến lỗi quyết định; mức bảo thủ nào giúp giảm khẳng định sai mà vẫn giữ đủ thông tin hữu ích cho điều khiển?**

Câu hỏi này nghiên cứu một đánh đổi thật: xác nhận quá dễ gây tiến độ ảo; xác nhận quá chặt làm memory im lặng hoặc đến muộn. “Có provenance” chưa trả lời được đánh đổi đó.

Giả thuyết H2: giữ uncertainty theo từng dữ kiện và theo thời điểm sẽ giảm lỗi hành động do thông tin sai/cũ tốt hơn hai cực đoan: luôn khẳng định một trạng thái duy nhất, hoặc đánh dấu toàn bộ lịch sử là unknown sau một lần mất dấu.

Phép kiểm bắt buộc:

- Cùng chuỗi detection, so cơ chế đầy đủ với phiên bản bỏ kiểm bằng chứng, bỏ xử lý mâu thuẫn hoặc ép unknown thành giá trị chắc chắn. Ablation phải đổi logic online; xóa metadata log không phải ablation của phương pháp.
- Can thiệp dropout ngắn/dài, ambiguity danh tính, delayed delivery, dữ kiện hết hiệu lực. Tách corruption nhân tạo ở tầng evidence khỏi kiểm trên che khuất/nhiễu thực tế.
- Đo precision/recall sự kiện, false-confirmation, độ trễ đến lúc policy thực sự nhận, tỷ lệ unknown, và lỗi quyết định tương ứng.
- Vẽ risk–coverage và hiệu năng theo độ trễ. Khi hệ không khẳng định gì, risk có thể thấp nhưng coverage bằng không; không coi đó là thành công.
- Khoảng chờ không tạo tiến độ phải được phân biệt với chu kỳ thật quay về cùng cảnh. Dwell có thể hoàn tất trong lúc chờ; chỉ bất biến phần tiến độ khi không có chuyển tiếp hợp lệ mới.

H2 không được hỗ trợ nếu độ chính xác tăng chỉ nhờ loại gần hết dữ kiện, nếu bảo thủ gây lỗi dừng/chọn vật nhiều hơn, hoặc nếu flags uncertainty không làm thay đổi cách chọn/diễn đạt thông tin và hành vi policy.

Độ quan trọng đối với field: memory trở thành một phép suy diễn có thể sai và cần quản lý độ tin cậy; không còn là nơi lưu lịch sử được giả định đúng hoàn toàn.

### RQ3 — Tái sử dụng giữa tác vụ và chính sách

> **Một lõi bộ nhớ dùng chung cho sự kiện, danh tính và quan hệ có thể duy trì lợi ích trên những dạng phụ thuộc lịch sử nào và trên các backbone VLA nhỏ khác nhau, khi chỉ thay giao diện tích hợp và giữ nguyên nguyên tắc cập nhật; chi phí và mức suy giảm kỹ năng cơ sở đi kèm là bao nhiêu?**

H3: một schema và cơ chế cập nhật dùng chung có thể bao phủ nhiều dạng lịch sử; từng model chỉ cần bộ chiếu/adapter phù hợp. Đây là khả năng chuyển thiết kế, không mặc nhiên là chuyển trọng số hay zero-shot.

Phép kiểm bắt buộc:

- Đánh giá toàn bộ mười tác vụ LIBERO-Mem ở revision đã khóa, báo theo từng task và họ memory; T1/T3 chỉ là pilot.
- Trên ít nhất hai backbone, giữ lõi memory và ngữ nghĩa fact; khóa phần nào được điều chỉnh theo model, số tham số học thêm và tuning budget.
- Giữ riêng ba mức: chạy cùng loại task trên state mới; chuyển sang loại quan hệ/che khuất khác; chuyển backbone. Không dùng thành công ở mức thứ nhất để kết luận hai mức sau.
- Đánh giá adapter đang dùng memory trên tác vụ cơ sở. Chuyển lại weights gốc bằng router không phải kiểm retention.
- Đo tổng latency, VRAM, thời gian huấn luyện, storage memory và chi phí định nghĩa primitive/profile. Nếu mỗi task cần nhiều quy tắc riêng, phải báo chi phí đó.

H3 không được hỗ trợ nếu phải viết lại quy tắc cho từng task, nếu chỉ một backbone có lợi, hoặc nếu chi phí bổ sung làm mất mục tiêu triển khai trong ngân sách đã định.

Độ quan trọng đối với field: phân biệt một thủ thuật riêng cho model/benchmark với một thành phần hệ thống có thể tái sử dụng, đồng thời công khai giới hạn của sự tái sử dụng đó.

### Những cách hỏi nên tránh

| Cách hỏi yếu | Vì sao yếu | Cách chuyển |
|---|---|---|
| Có thể xây memory cho SmolVLA không? | Nghiêng về khả năng tích hợp | RQ1: giá trị thông tin dưới ngân sách đối sánh |
| Memory có tăng performance không? | Không chỉ rõ loại lỗi, đối chứng hoặc điều kiện | Gắn SR với quyết định phụ thuộc lịch sử và baseline có cùng thông tin hiện tại |
| Có thể giải quyết POMDP không? | Quá rộng, không phù hợp một hệ cảm biến hữu hạn | Nêu tập biến lịch sử cần giữ và lớp tác vụ được đo |
| Memory có tổng quát không? | Không định nghĩa chiều tổng quát | RQ3: state, họ task, backbone; tách thích ứng khỏi zero-shot |
| Có thể chứng minh memory đúng không? | Lẫn tính đúng của chương trình với đúng vật lý | Chứng minh invariant có điều kiện; đo độ đúng cảm biến bằng nhãn độc lập |

## 4. Phát biểu bài toán tổng quát và các ràng buộc

Với chỉ dẫn \(\ell\), quan sát được phép \(o_t\) và hành động đã thực thi \(a_{t-1}\), đặt:

\[
h_t=(o_0,a_0,\ldots,a_{t-1},o_t),\qquad
M_t=U(M_{t-1},o_t,a_{t-1}),
\]
\[
c_t=P_{\psi}(\ell,o_t,M_t;B),\qquad
A_t\sim\pi_{\theta,\phi}(\cdot\mid o_t,c_t).
\]

Trong đó, \(M_t\) là trạng thái nhớ có cấu trúc; \(P_\psi\) là phép chọn và chiếu thông tin; \(\theta\) là policy nền; \(\phi\) là tham số thích ứng nếu có; \(B\) là ngân sách đầu vào. Chính sách có thể sinh đoạn hành động, nhưng chỉ hành động đã thực thi được nhập vào lịch sử.

Mục tiêu nghiên cứu là tăng kết quả điều khiển trong ràng buộc \(\operatorname{tokens}(c_t)\le B\), giới hạn VRAM/latency và ngân sách thích ứng đã công bố. Không viết như thể code đang giải một bài toán tối ưu toàn cục nếu chỉ dùng quy tắc heuristic. Không đồng nhất \(M_t\) với belief state đầy đủ hoặc thống kê đủ của POMDP.

Một trường hợp lý tưởng giải thích sự cần thiết của memory: hai lịch sử có cùng toàn bộ quan sát mà policy nhận và cùng chỉ dẫn nhưng đòi hỏi hai quyết định không tương thích. Policy chỉ dùng hiện tại không thể phân biệt hai trường hợp đó. Trong thực nghiệm, quan sát “trông giống nhau” có thể còn cue qua proprioception, dấu thời gian hoặc tư thế; cần kiểm soát chúng thay vì khẳng định lịch sử luôn bắt buộc.

**Nhỏ gọn phải được định lượng.** Báo số tham số toàn hệ, tham số huấn luyện, bộ nhớ GPU và phần cứng mục tiêu. Kích thước VLM backbone không phải số tham số toàn policy. Không có một ngưỡng “rất nhỏ” phổ quát; định nghĩa nhóm mô hình được chọn trong phần thực nghiệm, tránh suy rộng từ một model.

**48 token là contract checkpoint cụ thể.** Config công khai của [checkpoint SmolVLA đang dùng](https://huggingface.co/lerobot/smolvla_libero/blob/31d453f7edd78c839a8bbc39744a292686daf0de/config.json) đặt `tokenizer_max_length=48`, `n_obs_steps=1` và padding tới chiều dài tối đa. Vì vậy, không mô tả 48 như giới hạn kiến trúc bất biến của mọi SmolVLA, cũng không suy ít token thực dùng sẽ tự giảm FLOPs khi tensor vẫn pad bằng nhau. Đây là phép thử rõ cho một kênh văn bản nhỏ.

Memory lưu trong hệ thống và memory cấp cho policy là hai ngân sách khác nhau. Audit log có thể tăng theo số sự kiện; context vẫn hữu hạn. Muốn tuyên bố storage \(O(1)\) phải có cơ chế giới hạn/thải bỏ cụ thể, không suy từ prompt ngắn.

## 5. Phương pháp: từ bộ đếm sang trạng thái sự kiện–bằng chứng

### 5.1. Những thành phần bắt buộc

Đề xuất cấu trúc:

\[
M_t=(\mathcal I_t,\mathcal R_t,\mathcal E_t,\mathcal U_t,\mathcal D_t),
\]

với danh tính thực thể \(\mathcal I_t\), quan hệ hiện hành \(\mathcal R_t\), lịch sử sự kiện \(\mathcal E_t\), tình trạng chưa biết/mâu thuẫn \(\mathcal U_t\) và liên kết bằng chứng \(\mathcal D_t\). Đây là thiết kế cần hoàn thiện, không phải mô tả toàn bộ năng lực hiện hữu.

Một sự kiện nên chứa loại chuyển tiếp, thực thể và đối tượng liên quan, thời điểm xảy ra ước lượng, thời điểm khả dụng, danh sách bằng chứng và trạng thái xác nhận. Ví dụ gồm rời quan hệ đỡ, quay về, thay quan hệ vị trí, đưa vật vào vật chứa, mất khả năng xác định danh tính. Không suy “đã grasp” chỉ từ lệnh đóng gripper; không suy “container rỗng” chỉ từ việc không thấy vật.

Tách ba cấp: **ứng viên** đang tích lũy evidence; **đã xác nhận theo quy tắc quan sát**; **chưa xác định/mâu thuẫn/hết hiệu lực**. Nếu chưa hiệu chuẩn xác suất, dùng nhãn trạng thái và score thô, không gọi score detector là xác suất sự kiện đúng.

Sự kiện lịch sử và quan hệ hiện tại có ngữ nghĩa khác nhau. Việc mất dấu sau khi đã quan sát một thao tác không xóa thao tác ấy. Ngược lại, “đang ở trên giá” có thể hết hiệu lực khi có chuyển tiếp mới. Unknown nên gắn từng fact/entity; một lần mất dấu không nhất thiết làm mất mọi thông tin đã xác nhận.

### 5.2. Lõi dùng chung và phần phụ thuộc miền

Giữ lõi schema, kiểm thời gian, consistency, truy xuất và projection dùng chung. Các primitive tri giác có thể chuyên biệt theo quan hệ vật lý; phải khai báo rõ phần kiến thức thiết kế thủ công.

Không xây mười FSM riêng rồi gọi là generic. Tổ chức thành các primitive tái sử dụng như chuyển quan hệ, duy trì identity, containment và đếm chuỗi chuyển tiếp. Task profile chỉ liên kết các thực thể quan sát được với ngữ nghĩa chỉ dẫn; không cung cấp vị trí đặc quyền hay tiến độ từ evaluator.

Không cấp máy trạng thái đáp án đầy đủ từ BDDL cho nhánh predicted nếu nhánh baseline chỉ có instruction. Nếu cần task specification thủ công cho một thí nghiệm, phải khai báo nguồn thông tin và cung cấp cùng đặc quyền cho đối chứng. Bộ ghi sự kiện có thể độc lập với số lần yêu cầu; bộ truy xuất được dùng chỉ dẫn để chọn fact liên quan.

Một điểm khó với nhiều vật giống nhau: ID nội bộ như “bowl A” không có ý nghĩa đối với VLA nếu chưa gắn nó với mô tả quan sát được. Cần liên kết identity theo thời gian với mô tả vị trí/quan hệ hiện tại hoặc anchor ban đầu, ghi ambiguity khi không thể phân biệt. Phép chiếu không được tự tạo danh tính chắc chắn để tiết kiệm token.

### 5.3. Projection thành văn bản

Projection nhận trạng thái, chỉ dẫn và budget; trả tập fact đã chọn, văn bản, fact bị bỏ và lý do. Text là giao diện của một nhánh hiện thực, không phải toàn bộ memory.

Quy trình đề xuất:

1. Lọc fact chưa khả dụng, nguồn không được phép hoặc phụ thuộc bằng chứng bị invalid.
2. Tạo nhóm nguyên tử gồm fact + identity grounding + qualifier uncertainty cần thiết.
3. Xếp mức liên quan dựa trên instruction và trạng thái quan sát được. Không dùng action chuyên gia tương lai, reward hay oracle goal progress để xếp hạng online.
4. Chọn các nhóm không mâu thuẫn trong ngân sách tokenizer thật; ưu tiên thông tin phân biệt lựa chọn đang đặt ra. Bắt đầu bằng rule công khai, chưa cần thêm một LLM lớn.
5. Bảo toàn instruction; khi không đủ chỗ phải bỏ nguyên nhóm và log, không cắt qualifier khiến unknown thành chắc chắn.

Có thể mô tả lựa chọn bằng tối ưu hữu hạn:

\[
S_t^*\in\arg\max_{S\subseteq\mathcal F_t}\sum_{f\in S}w(f;\ell,o_t,M_t)
\quad\text{s.t.}\quad
\operatorname{tokens}(\ell\oplus P(S))\le B,
\]

với ràng buộc fact hợp lệ, đủ các phụ thuộc và không mâu thuẫn. Nếu code dùng greedy thì gọi đúng là heuristic cho mục tiêu này; trọng số được chốt từ development, không phải “độ hữu ích tối ưu” đã được biết. Token cost phải tính toàn chuỗi vì tokenizer không nhất thiết cộng tuyến tính theo fact.

Đây là nơi có thể có đóng góp phương pháp: **chọn thông tin theo mức liên quan nhưng vẫn bảo toàn điều kiện bằng chứng và ngữ nghĩa chưa biết**. Phải so với chronological truncation, recent-only và projection bỏ qualifier để chứng minh đóng góp.

### 5.4. Các mệnh đề có thể chứng minh và giới hạn của chúng

| Tính chất | Điều kiện | Ý nghĩa và giới hạn |
|---|---|---|
| Không nhìn trước | Cùng tiền tố quan sát/hành động đã thực thi, cùng cấu hình/ngẫu nhiên; chỉ evidence đã khả dụng | Chứng minh ngữ cảnh không phụ thuộc tương lai; không chứng minh nội dung cảm biến đúng |
| Không đếm trùng | Mỗi return tiêu thụ đúng một departure chưa ghép | Đúng với chuỗi sự kiện của writer; không suy count bằng số lần vật lý thật |
| Fact có chuỗi bằng chứng | Mỗi fact được chiếu truy về các evidence hợp lệ và phụ thuộc đầy đủ | Truy xuất được căn cứ suy diễn; hash chỉ bảo toàn định danh dữ liệu, không bảo đảm bằng chứng đáng tin tuyệt đối |
| Bảo toàn qualifier | Fact uncertainty và identity context được chọn/bỏ nguyên nhóm | Projection không tự nâng mức khẳng định; vẫn phải đo VLA có hiểu qualifier hay không |
| Bất biến tiến độ với quan sát không có chuyển tiếp hợp lệ | Bộ xác nhận không phát sự kiện mới | Không tạo tiến độ từ clock; không yêu cầu timestamp/coverage/toàn bộ memory bất biến |

Đặc biệt, “đã quan sát k sự kiện” không đồng nghĩa “thực tế ít nhất k lần” nếu detector có false positive. Count là cận dưới của sự thật chỉ khi từng xác nhận đúng và không đếm trùng. Cách diễn đạt nên phản ánh điều kiện đó; cần kiểm trên failure trajectories thay vì dựa vào sáu quỹ đạo thành công.

Không nên trang trí chương phương pháp bằng các định lý không liên quan kết quả. Hai invariant hiện có là nền tảng hợp lệ; novelty cần đến từ thiết kế và bằng chứng thực nghiệm vượt chúng.

## 6. Bản đồ novelty và các phản biện phải chuẩn bị

| Đóng góp dự kiến | Cái được kế thừa | Phần cần tự phát triển/kiểm định |
|---|---|---|
| Biểu diễn sự kiện–bằng chứng có thể dùng chung | Object-centric memory, FSM, provenance | Quy tắc trạng thái uncertainty, invalidation và quan hệ nhiều thực thể hoạt động nhất quán |
| Projection trong ngân sách bảo toàn ý nghĩa | Text conditioning, retrieval, token limit | Chọn fact kèm phụ thuộc/qualifier; đo thông tin quyết định sống sót qua nén |
| Đánh giá giới hạn hiệu lực cho policy nhỏ | Benchmark và interventional evaluation | Kết quả đối sánh về ngân sách–độ tin cậy–hành vi, trên nhiều họ task/backbone |

Các câu hỏi phản biện cần trả lời trước khi viết “đóng góp mới”:

- Nếu bỏ evidence metadata mà policy vẫn nhận đúng cùng văn bản, kết quả có thay đổi không? Nếu không, metadata là chức năng audit; chỉ logic gating/uncertainty có tác động mới là phần cơ chế.
- Vì sao không dùng một trạng thái hồi quy như SmoLSTM? Trả lời bằng yêu cầu tái sử dụng policy và khả năng kiểm tra fact, rồi đo trade-off; không giả định phương pháp tường minh tốt hơn latent memory.
- Có phải hệ đang biến thành một planner ra lệnh cho VLA? Primary memory chỉ mô tả lịch sử/trạng thái có căn cứ; câu kiểu “hãy cầm vật X tiếp theo” thuộc subgoal planning và phải tách thành nhánh riêng.
- Có phải chỉ thêm detector mạnh hơn? Current-only và các nhánh dùng chung perception trả lời câu này.
- Có phải đổi simulator/protocol rồi tăng số liệu? Giữ source lock, báo custom protocol; không so trực tiếp với điểm paper khi chưa conform.
- Có thật phù hợp thiết bị hạn chế khi detector mất khoảng một giây mỗi frame? Phải tối ưu/đo toàn hệ trước claim triển khai. Không lấy thời gian replay cache làm latency online.
- Có thể gọi model-agnostic khi có hai model không? Chỉ được nói đã minh họa khả năng tái sử dụng trên hai model, không suy cho mọi VLA.

## 7. Thiết kế thực nghiệm để trả lời các RQ

### 7.1. Trục coverage

Đích chính là toàn bộ T1–T10 của LIBERO-Mem ở code revision `0eee0defa4024e51511b6e15a79f30e9620209de`. Mapping dưới đây dựa trên task map/BDDL của bản mã đã khóa; không sao chép tên minh họa khi khác release.

| Nhóm | Tác vụ | Năng lực cần kiểm tra |
|---|---|---|
| T1/T2 | Nhấc–đặt bát/chai một lần | Phân biệt chưa làm và đã làm khi cảnh gần như cũ |
| T3/T4 | Ba chu kỳ với bát/chai | Tiến độ lặp và chuyển primitive giữa loại vật |
| T5/T6 | Bát năm/bảy chu kỳ | Lịch sử dài hơn, sai số tích lũy, dừng đúng số lần |
| T7/T8 | Hoán đổi hai bát/quay vòng ba bát | Identity xuyên tương tác, origin binding và quan hệ nhiều vật |
| T9/T10 | Đưa cream cheese vào giỏ, chọn giỏ đã chứa/giỏ rỗng | Containment, object permanence và bằng chứng phủ định |

T7/T8 là phép thử bắt buộc để vượt khỏi “counter”. T9/T10 buộc giải quyết “không thấy” khác “không có”. Cần audit Sequence/Or và call schedule evaluator riêng, không bê nguyên protocol counter T1/T3 sang mọi task.

Bao phủ mười task trong cùng benchmark kiểm tính đa dạng yêu cầu memory. Nó chưa chứng minh tổng quát ngoài benchmark. Một tập chuyển giao ở benchmark thứ hai là mục tiêu mở rộng sau core; không bắt buộc nhân đôi toàn bộ ma trận trước khi chốt các RQ chính.

### 7.2. Các điều kiện đối chứng

| Điều kiện | Huấn luyện/thích ứng | Thông tin cấp cho policy | Phép so sánh chính |
|---|---|---|---|
| B0 | Checkpoint nền | Instruction + observation chính thức | Mốc năng lực trước thích ứng |
| F0 | Cùng dữ liệu/budget với F1 | Không history | F1–F0: lợi ích hệ có memory sau thích ứng |
| C0 | Thích ứng đối sánh | Tóm tắt hiện tại từ cùng perception, không lịch sử | F1–C0: phần giá trị lịch sử |
| L0 | Thích ứng đối sánh | Danh sách sự kiện/lịch sử phẳng từ cùng nguồn, cùng trần token | F1–L0: giá trị tổ chức và projection |
| F1 | Phương pháp đầy đủ | Trạng thái memory được chiếu theo ngân sách và evidence | Điều kiện đề xuất |
| F1-off / fact interventions | Cùng weights F1 | Bỏ/đổi có kiểm soát đầu vào memory | Chẩn đoán cơ chế, không thay F0 |
| O-diag | Khai báo riêng nguồn đặc quyền | Fact chuẩn chỉ dùng trong chẩn đoán | Tách vấn đề tạo memory với đọc memory; không tính vào điểm hệ deployed |

Bốn nhánh thích ứng F0/C0/L0/F1 là cấu hình nghiên cứu khuyến nghị cho model chính. Một baseline recent-observation window và một baseline memory học được là phần mở rộng hữu ích nếu có code/checkpoint hợp lệ và ngân sách; không dùng mock để điền bảng. L0 dùng cùng candidate event stream và quy tắc xây dữ liệu đã công bố, tránh thêm một LLM tạo summary với chi phí không đối sánh.

Với các memory arm, chốt cùng chiến lược dropout và ngân sách tuning. F1 hiện có dropout bằng 0; F1-off hiện tại có thể ngoài phân phối. Khi huấn luyện phiên bản mới với dropout, đó là recipe mới, không đổi nhãn kết quả lịch sử.

Oracle không mặc nhiên là cận trên thành công: policy có thể chưa được huấn luyện đọc định dạng oracle hoặc thất bại vận động. Chỉ dùng nó để chẩn đoán theo contract riêng; memory thực vẫn không nhận simulator state.

### 7.3. Ba lớp đánh giá bổ trợ nhau

**Lớp A — chất lượng trạng thái nhớ.** Audit dữ liệu có cả thành công và thất bại, identity ambiguity, che khuất, false return, wrong support và thao tác dở dang. Chú giải độc lập; định nghĩa event khác contact vật lý; matching một-một và dung sai đặt trước. Báo sai danh tính, precision/recall sự kiện, trạng thái quan hệ, unknown và delay. Không dùng prompt coverage thay accuracy.

**Lớp B — quyết định phụ thuộc lịch sử.** Dùng những thời điểm có lựa chọn quan sát được như tiếp tục/dừng, chọn đúng vật hoặc chọn giỏ. Nếu action chunk chưa biểu lộ lựa chọn rõ ràng, ghi undecidable hoặc đánh giá tiếp đoạn rollout đã định trước; không suy quyết định từ attention. Các phép thay fact phải đến đúng query và có exact-repeat/sham cùng đường cài đặt. Theo cách phân biệt của [IMEP](https://arxiv.org/abs/2610.23372v1), độ nhạy đầu ra cần được đối chiếu với tính đúng và tính đặc hiệu của thông tin.

**Lớp C — điều khiển vòng kín.** Chạy toàn task, success chính thức theo protocol được khai báo, tiến độ từng phần và loại lỗi. Đánh giá độ bền với noise/delay và retention. Không thay success toàn task bằng metric thuận lợi hơn khi kết quả thấp.

### 7.4. Không để failure vận động che khuất câu hỏi memory

Nếu F0/F1 đều gần 0 và phần lớn thất bại trước quyết định cần nhớ, tăng số lượt hoặc thêm task chưa giải quyết nút thắt. Trước hết phải xác minh:

1. Hành động quỹ đạo chuyên gia replay đúng alignment và giải được task trong runtime đang dùng.
2. Policy hiện tại thực hiện được primitive cơ sở trong cùng runtime.
3. Fact đúng và được cấp đúng lúc có thể ảnh hưởng đến lựa chọn liên quan.
4. Writer thực tạo được fact đó trên quỹ đạo policy tự sinh.

Đây là cây chẩn đoán, không cho phép tự thay tài nguyên benchmark. Nếu thiếu checkpoint, action contract hoặc asset chính thức, dừng nhánh bị ảnh hưởng và báo đúng blocker.

### 7.5. Ngân sách, độ trễ và nhịp điều khiển

Khảo sát budget theo phần còn lại sau instruction, với tokenizer từng model. Có thể đặt các mức 0, 1/3, 2/3 và toàn bộ dung lượng memory khả dụng, sau khi kiểm mức đó chứa được fact tối thiểu. Giữ không đổi instruction và không âm thầm tăng giới hạn checkpoint. Báo số token thực và padding; không gọi đó là FLOP-matched nếu tính toán khác.

Ghi cả thời điểm ảnh được chụp, evidence hoàn tất, event được xác nhận, query nhận memory và action bắt đầu thi hành. Độ trễ đến quyết định có thể gồm xác nhận + xử lý + IPC + chờ hết action chunk; dùng đo thực thay vì giả định các thành phần luôn cộng khi chạy bất đồng bộ.

Chế độ simulator dừng chờ inference và chế độ chịu deadline thực phải được phân biệt. Chạy control frequency 20 Hz trong mô phỏng không đồng nghĩa hệ xử lý 20 frame/s theo thời gian thực. Nếu đổi cadence detector, tracker hoặc chunk schedule để giảm chi phí, khóa thành cấu hình mới và đánh giá lại.

### 7.6. Thống kê và tính độc lập

- Khóa development/validation/confirmatory trước tuning. State đã được xem để sửa writer không còn là blind holdout. Seed mới trên cùng state không khôi phục tính độc lập đó.
- Dùng initial-state group và lineage của dữ liệu để tránh chia cùng cảnh/quỹ đạo cha sang train và test. Không giả định seed khởi tạo là hash của trạng thái vật lý.
- Ghép state, task, rollout seed và recipe giữa các điều kiện. Phân biệt seed huấn luyện với seed suy luận; một checkpoint chạy nhiều seed không đo variance do training.
- Primary endpoint: chênh lệch macro SR giữa F1 và F0 trên tập đã khóa. F1–C0, F1–L0 và các metric quyết định là các phép kiểm giải thích cơ chế. Chọn cách xử lý nhiều phép thử trước confirmatory.
- CI/paired bootstrap phải giữ cụm state chia sẻ giữa task và các seed. Chỉ dùng McNemar đơn giản khi các cặp đủ độc lập; không xem frame hoặc policy query là các mẫu độc lập.
- Chọn cỡ mẫu bằng độ rộng CI hoặc power cho hiệu ứng nhỏ nhất có ý nghĩa; không gọi 20 state × 3 seed là bảo đảm đủ. Nếu tài nguyên không đủ độ phân giải, báo giới hạn đó.
- Retention: đăng ký biên suy giảm chấp nhận được và kiểm CI của chênh lệch adapter–B0. “Không có ý nghĩa thống kê” không chứng minh tương đương hoặc không suy giảm.
- Runtime abort, thiếu artifact hoặc timeout phải nằm trong flow report; chốt cách tính trước, không loại lỗi hệ thống để tăng SR.

Một dự toán minh họa: 10 task × 20 state hợp lệ × 3 rollout seed × 4 nhánh thích ứng = 2.400 lượt **cho một seed huấn luyện của mỗi nhánh và một backbone**. Ba seed huấn luyện là 7.200 lượt cho phần này; B0, retention và can thiệp tính riêng. Chỉ dùng 20 state nếu nguồn đã khóa thực sự cung cấp đủ holdout hợp lệ. Chốt số thực tế sau khi đo chi phí pilot, không sinh state thay thế hoặc coi dự toán là yêu cầu của benchmark.

## 8. Viết lại từng chương

### Chương 1 — động cơ, khoảng trống, câu hỏi và phạm vi

Mở từ nhu cầu robot một tay thực hiện chuỗi tương tác mà hiện tại không tiết lộ lịch sử, rồi đến khó khăn của policy nhỏ: kênh ngữ cảnh hẹp, inference và adaptation có ngân sách, lịch sử cảm biến không hoàn hảo. Dùng ba ví dụ ngắn: thao tác lặp, vật giống nhau được di chuyển, vật bị che trong container. Sau đó mới tổng hợp prior work và nêu khoảng trống cụ thể.

Đưa ba RQ và các đóng góp dự kiến vào cuối lập luận. Chỉ một đoạn phạm vi nêu model/benchmark được chọn. Không mở đầu bằng kiến trúc SmolVLA; không định nghĩa biến mục tiêu chỉ với N ∈ {1,3}.

Đoạn mẫu về giọng văn, chưa thay vào bản thảo:

> Trong nhiều tác vụ thao tác, cấu hình mà robot đang quan sát không phản ánh đầy đủ những tương tác đã xảy ra. Một vật có thể trở lại vị trí cũ sau khi được xử lý, hoặc khuất khỏi camera sau khi được đưa vào vật chứa; vì vậy, lựa chọn hành động còn phụ thuộc vào lịch sử của từng thực thể. Yêu cầu này đặt ra khó khăn rõ rệt cho các chính sách VLA nhỏ gọn, bởi lưu lại toàn bộ chuỗi quan sát làm tăng chi phí trong khi một bản tóm tắt quá ngắn có thể bỏ mất dữ kiện quyết định. Hơn nữa, lịch sử được lưu cũng không hoàn toàn đáng tin khi cảm biến bị che khuất hoặc nhận dạng sai. Từ đó, nghiên cứu đặt vấn đề xây dựng một biểu diễn vừa giữ được thông tin cần cho quyết định, vừa thể hiện căn cứ và giới hạn của thông tin ấy trong ngân sách đầu vào hữu hạn.

### Chương 2 — nền tảng và related work

Đề cương đề xuất:

1. Quan sát một phần, observational aliasing và biến lịch sử liên quan quyết định.
2. Các biểu diễn memory: cửa sổ lịch sử, latent/recurrent, object-centric, symbolic/text và multi-scale.
3. **Trừu tượng hóa lịch sử và biểu diễn tiến trình theo sự kiện**: belief state, state abstraction, automata/reward machines, các điều kiện mất thông tin.
4. Tính hợp lệ của evidence, unknown, tính không nhìn trước và chi phí chậm xác nhận.
5. Kiểm định tác dụng memory: thông tin hiện tại, temporal shortcut, can thiệp dữ kiện và performance.

Mục 2.3 thay nội dung “khoảng trống trong tác vụ lặp”; không lặp mục 1.2. Bảng related work so theo biểu diễn, nguồn thông tin, xử lý missingness, cách conditioning, chi phí và bằng chứng hành vi. Ô nào paper chưa báo rõ ghi “chưa xác định từ nguồn đã đọc”, không tự điền “không hỗ trợ”.

### Chương 3 — Bộ nhớ sự kiện có bằng chứng cho chính sách thao tác robot

Trình tự: bài toán/giả thiết → schema → evidence và identity → cập nhật/uncertainty → retrieval/projection → tích hợp policy và thích ứng → tính chất/giới hạn → nối với giả thuyết.

Ví dụ đi xuyên chương phải có ít nhất hai loại ngoài count: quan hệ nhiều vật và containment. Định nghĩa chung trước, minh họa sau. Giữ SmolVLA ở đoạn hiện thực cuối nếu thật cần; chuyển config model, thông số bbox và schema BDDL sang chương thực nghiệm/phụ lục.

Mỗi quyết định thiết kế phải trả lời: vấn đề gì buộc phải có nó, giả định nào nó dùng, failure nào nó ngăn, và ablation nào kiểm tra. Thông số ngưỡng thực nghiệm không được biến thành hằng số mang tính lý thuyết.

### Chương 4 — Thực nghiệm theo RQ

Đặt protocol, nguồn và backbone trước; tiếp theo kết quả theo RQ1–RQ3. Tách bảng writer, quyết định và closed-loop. Bảng chính có coverage đầy đủ; bảng ablation được bố trí theo cơ chế. Thảo luận failure gắn với câu hỏi, không trở thành nhật ký debug.

Các kết quả T1/T3 cũ thuộc pilot/historical development. Nếu kết quả mới chưa có, ghi trạng thái kế hoạch trong tài liệu làm việc; không viết phần kết quả mới bằng thì quá khứ.

### Chương 5 — câu trả lời, giới hạn và ý nghĩa

Mỗi đoạn kết luận trả lời một RQ bằng kết quả, độ bất định và phạm vi áp dụng. Tránh kết luận chỉ liệt kê các module đã xây. Tách khả năng cải thiện quyết định khỏi reliability toàn tác vụ và chi phí.

### Phụ lục và văn phong

Phụ lục nên chứa protocol đủ chi tiết, version nguồn công khai, bảng cấu hình, schema, chứng minh đầy đủ, tiêu chí chú giải và ví dụ lỗi. Không dẫn ổ đĩa, thư mục nội bộ, tên log, tên script hay “xem file trên máy”. Dùng định danh thí nghiệm do luận văn định nghĩa, số bảng/hình và URL/DOI của tài nguyên đã công bố. Bản ghi nội bộ chưa public vẫn có thể được tổng hợp thành bảng trong thesis; không tạo URL giả.

“Demonstration” dịch theo vai trò: **quỹ đạo chuyên gia** nếu có căn cứ về nguồn chuyên gia; **quỹ đạo mẫu/dữ liệu thao tác mẫu** nếu chỉ là dữ liệu học; **dữ liệu điều khiển thủ công** khi nói cách thu. **Ground truth/nhãn tham chiếu** chỉ dùng cho nhãn mục tiêu hoặc đáp án được xác định, không phải từ đồng nghĩa của mọi demonstration.

Sửa liên kết ở cấp ý: câu chủ đề → căn cứ/ví dụ → hệ quả → cầu nối. Dùng “tuy nhiên”, “vì vậy”, “đồng thời”, “do đó” khi đúng quan hệ; không rải liên từ vào các câu vẫn không cùng tuyến lập luận. Mỗi đoạn nên làm rõ điều mới so với đoạn trước. Tránh lặp “khóa luận” ở đầu mọi câu và tránh cấu trúc checklist trong abstract/mở đầu/kết luận.

## 9. Điều kiện hoàn thành và các quyết định cần chốt

Mục tiêu đầy đủ của kế hoạch là: phương pháp chung được triển khai; toàn bộ mười task được đánh giá; có đối chứng tách adaptation/perception/history; có đánh giá uncertainty và budget; có backbone thứ hai với contract được xác minh; có retention và số liệu chi phí; bản viết trả lời từng RQ bằng evidence tương ứng.

Các quyết định nên chốt trước chạy lớn: primary endpoint, ngưỡng cải thiện có ý nghĩa thực tế, biên retention, ngân sách phần cứng, primitive/profile được phép thiết kế thủ công, và roster holdout chưa dùng phát triển. Đây là nội dung đăng ký thực nghiệm, không phải lý do trì hoãn việc sửa cấu trúc lập luận và hoàn thiện harness.

Nếu chưa đạt cả mười task, vẫn giữ phạm vi phương pháp tổng quát như mục tiêu nhưng phải công khai coverage đạt được. Không ghi “đã tổng quát” để làm bài trông nhiều việc hơn. Nếu không có gain đáng tin cậy, kết quả âm chỉ có giá trị mạnh khi kiểm soát được competence floor, information delivery và thống kê; khi đó luận văn cần kết luận về giới hạn cơ chế thay vì tuyên bố cải thiện.

Ưu tiên cao nhất là đóng chuỗi **memory biết gì → fact nào tới policy → quyết định đổi thế nào → task có thành công hơn không → phải trả bao nhiêu chi phí**. Chuỗi này làm phương pháp có ý nghĩa khoa học và giúp mọi phần code, thực nghiệm và văn bản phục vụ cùng một luận điểm.
