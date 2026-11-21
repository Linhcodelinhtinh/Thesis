TỔNG QUAN ĐÁNH GIÁ
Ưu điểm lớn về Logic: Dàn ý có nền tảng toán học và mô hình hóa rất vững:
Bài toán đồ chơi (hai hộp kín X, Y) làm rõ bản chất POMDP và giới hạn của chính sách Markovian.
Cấu trúc bản ghi 7 trường 
𝑓
=
(
𝑘
,
𝑣
,
𝜎
,
𝜏
,
𝜈
,
𝐼
,
𝑃
)
f=(k,v,σ,τ,ν,I,P) chặt chẽ, đặc biệt việc tách biệt 
𝜏
τ (thời điểm vật lý) và 
𝜈
ν (thời điểm xác nhận nhận thức, 
𝜈
≥
𝜏
ν≥τ) ngăn chặn triệt để rò rỉ nhìn trước (look-ahead leak).
Bất biến trạng thái hữu hạn FSM (
𝐽
𝑡
−
𝑁
𝑡
−
𝐴
𝑡
=
𝜒
𝑡
J
t
	​

−N
t
	​

−A
t
	​

=χ
t
	​

) và công thức trễ phản ứng ngắt đoạn hành động (
𝑞
′
=
min
⁡
(
𝑞
+
𝐾
,
max
⁡
(
𝑞
+
𝑚
,
𝑟
+
𝑝
−
1
)
)
q
′
=min(q+K,max(q+m,r+p−1))) có giá trị lý thuyết cao.
Những điểm cần chỉnh sửa trọng tâm:
Tránh biến Chương 3 thành "bộ lọc đo điểm của một tác vụ cụ thể": Mục 3.2.4 hiện tại đi quá sâu vào các ngưỡng tọa độ pixel hộp bao của bài gắp bát (
𝐿
𝑡
≥
0,40
L
t
	​

≥0,40, 
0,055
0,055 m...), khiến phương pháp trông như một heuristic thị giác cục bộ thay vì một khung bộ nhớ tổng quát cho VLA. Cần trình bày nguyên lý trừu tượng trước, sau đó đưa bài gắp bát vào như một Case Study minh họa.
Loại bỏ triệt để tên file/thư mục nội bộ: Dòng 121 (results_f1_author_split_v2, train_memory_sft.py, replay_sampler.py) và dòng 158 (biên bản D2 ngày 20/11/2026) vi phạm trực tiếp nguyên tắc bảo mật và quy chuẩn khóa luận (người ngoài không thể và không nên thấy các đường dẫn local); thay thế bằng các nội dung được mô tả đầy đủ.
Hoàn thiện và thay thế các "câu lệnh chỉ đạo người viết" (Drafting Meta-Instructions): Nhiều câu vẫn viết dưới dạng nhắc việc cho bản thân ("Viết ba phép biến đổi", "Trình bày từ một mẫu...", "Mô tả VLM...", "Thử cắt hai quỹ đạo...") thay vì ngôn ngữ học thuật của một đề cương/luận văn.
Bổ sung liên từ và tăng tính kết nối: Các câu văn đơn ngắn bị ngắt quãng, thiếu liên từ logic (Do đó, Tuy nhiên, Cụ thể, Nhờ vậy), tạo cảm giác máy móc.
Làm nổi bật tính chất "Mô hình VLA nhỏ" và "Text là một bộ chiếu (Projection)": Cần nhấn mạnh ngay từ đầu lý do VLA nhỏ không thể nhét chuỗi video dài vào context vì nghẽn token và chi phí biên toán trên robot, từ đó chứng minh vai trò chiếu nén của bộ đọc ngôn ngữ.
PHẦN 1: KIỂM TRA LOGIC & TÍNH NHẤT QUÁN KHOA HỌC
1. Mục 3.1: Phát biểu bài toán và Động cơ lý thuyết
Logic bài toán đồ chơi 3.1.1:
Hiện trạng: Nêu tình huống hai hộp X, Y với cùng ảnh cuối đã đóng.
Góp ý Logic: Rất tốt về mặt phản ví dụ POMDP (
𝑜
𝑡
(
1
)
≈
𝑜
𝑡
(
2
)
o
t
(1)
	​

≈o
t
(2)
	​

 nhưng 
𝑎
𝑡
(
1
)
≠
𝑎
𝑡
(
2
)
a
t
(1)
	​


=a
t
(2)
	​

). Tuy nhiên, cần bổ sung lý giải: Tại sao không đưa toàn bộ lịch sử video vào VLA? Đối với các mô hình VLA nhỏ gọn chạy trên cánh tay robot biên (edge computing), việc nhận chuỗi khung hình dài là bất khả thi do giới hạn token đầu vào và độ trễ tính toán. Đây là tiền đề sống còn để dẫn dắt đến giải pháp bộ nhớ có cấu trúc được chiếu thành văn bản ngắn gọn.
Phân biệt biến lịch sử 
𝑍
𝑡
Z
t
	​

, quyết định 
𝑌
𝑡
Y
t
	​

 và trạng thái bộ nhớ 
𝑀
𝑡
M
t
	​

 (Dòng 31-33):
Hiện trạng: "
𝑍
𝑡
Z
t
	​

 cho đại lượng lịch sử có liên quan... 
𝑌
𝑡
Y
t
	​

 là quyết định cấp tác vụ... VLA vẫn sinh đoạn lệnh chuyển động."
Góp ý Logic: Cần phân định rõ quan hệ phân cấp: 
𝑍
𝑡
Z
t
	​

 là biến trạng thái tiềm ẩn thực tế của môi trường (Ground-truth Latent State); 
𝑀
𝑡
M
t
	​

 là biểu diễn bộ nhớ có bằng chứng do hệ thống duy trì; 
𝐶
𝑞
C
q
	​

 là hình chiếu ngôn ngữ có ngân sách; còn 
𝑌
𝑡
Y
t
	​

 là nhãn quyết định phân tích độc lập. Việc làm rõ này giúp người đọc không nhầm lẫn giữa bộ nhớ mô hình hóa (
𝑀
𝑡
M
t
	​

) và chân lý ẩn của môi trường (
𝑍
𝑡
Z
t
	​

).
Công thức xác suất toàn phần về sai số (Dòng 36):
Công thức: 
𝐸
[
err
]
=
𝑐
𝜖
+
(
1
−
𝑐
)
/
2
E[err]=cϵ+(1−c)/2.
Đánh giá: Toán học hoàn toàn chính xác. Khi có thông tin (
𝑐
c), sai số là 
𝜖
ϵ; khi không có thông tin (
1
−
𝑐
1−c), chọn ngẫu nhiên đồng xác suất cho sai số 
1
/
2
1/2.
Ý nghĩa khoa học: Cần nhấn mạnh luận điểm: Khi 
𝜖
<
0,5
ϵ<0,5, việc tăng độ phủ 
𝑐
c sẽ giảm sai số toàn phần. Ngược lại, nếu bộ nhớ "ảo giác" (
𝜖
>
0,5
ϵ>0,5), việc trả lời bừa còn tệ hơn chọn ngẫu nhiên. Điều này chứng minh vì sao trạng thái "chưa xác định" (abstain) là một thiết kế an toàn vượt trội so với ảo giác.
2. Mục 3.2: Khung phương pháp bộ nhớ có bằng chứng
Đơn vị bản ghi 7 trường (Mục 3.2.1, Dòng 43-47):
Công thức: 
𝑓
=
(
𝑘
,
𝑣
,
𝜎
,
𝜏
,
𝜈
,
𝐼
,
𝑃
)
f=(k,v,σ,τ,ν,I,P).
Đánh giá: Rất chặt chẽ. Điểm sáng lớn nhất là sự phân tách giữa 
𝜏
τ (thời điểm diễn ra biến cố vật lý) và 
𝜈
ν (thời điểm tích lũy đủ bằng chứng quan sát, 
𝜈
≥
𝜏
ν≥τ).
Góp ý Logic: Cần khẳng định rõ đây là cơ sở để giải quyết vấn đề rò rỉ thời gian (look-ahead leakage) trong huấn luyện offline và lập lịch suy luận online.
Nguy cơ cục bộ hóa trong Mục 3.2.4 (Dòng 62-74):
Vấn đề: Bắt đầu bằng "Ở bộ ghi RGB Cycle lịch sử..." và liệt kê bảng ngưỡng pixel: 
𝐿
𝑡
≥
0,40
L
t
	​

≥0,40, 
𝐷
𝑡
≤
0,30
D
t
	​

≤0,30, mở ngón 
≥
0,055
≥0,055 m.
Rủi ro logic: Vi phạm định hướng của khóa luận (không biến luận văn thành bài toán riêng cho T1/T3 hay LIBERO-Mem). Nếu phản biện đọc vào, họ sẽ đánh giá đây là "các mẹo heuristic thủ công cho cái bát và cái đĩa" chứ không phải một phương pháp khoa học tổng quát.
Giải pháp cấu trúc lại:
Bước 1 (Tổng quát): Định nghĩa toán tử trích xuất bằng chứng tổng quát: Vị từ không gian - hình học 
𝑃
𝑘
(
𝑜
𝑡
)
P
k
	​

(o
t
	​

) trên các thực thể, cơ chế khớp định danh (identity tracking), và bộ lọc trễ thời gian (temporal hysteresis / persistence window qua 
𝑝
p bước) để chuyển trạng thái từ ứng viên sang xác nhận.
Bước 2 (Trường hợp cụ thể hóa - Case Study): Đặt tiêu đề là "Minh họa thực nghiệm: Bộ theo dõi chu kỳ thao tác vật thể". Trình bày rõ đây là một hiện thực mẫu của Task Profile nhằm minh họa cách ánh xạ tọa độ cảm biến thành vị từ sự kiện.
Bất biến kiểm tra hiện thực (Dòng 100):
Công thức: 
𝐽
𝑡
−
𝑁
𝑡
−
𝐴
𝑡
=
𝜒
𝑡
J
t
	​

−N
t
	​

−A
t
	​

=χ
t
	​

 với 
𝜒
𝑡
∈
{
0
,
1
}
χ
t
	​

∈{0,1}.
Đánh giá: Rất tốt và chuẩn mực theo lý thuyết máy trạng thái (FSM). Nên đóng khung thành một Đặc tính kiểm định (Invariant Property) để làm nổi bật tính chặt chẽ trong cài đặt phần mềm.
3. Mục 3.3: Quy trình huấn luyện thích ứng Action Expert
Xử lý các định danh thử nghiệm cục bộ (Dòng 108, 111, 119):
Hiện trạng: "Manifest F1 V2 ghi 
𝑟
=
16
r=16...", "Cặp V2 dùng 598 quỹ đạo...", "văn bản của Relation V1...".
Góp ý: Dàn ý Chương 3 là phần Phương pháp luận. Những cách gọi như "Cặp V2", "F1 V2", "Relation V1", "Manifest" là thuật ngữ nhật ký thí nghiệm. Cần chuyển ngữ sang ngôn ngữ khoa học:
Thay "Manifest F1 V2 ghi 
𝑟
=
16
r=16" 
→
→ "Cấu hình thích ứng sử dụng LoRA với hạng 
𝑟
=
16
r=16, tương ứng với 645.120 tham số tối ưu hóa trên Action Expert."
Thay "Cặp V2 dùng 598 quỹ đạo..." 
→
→ Trình bày nguyên lý Lấy mẫu cân bằng đa miền (Dual-Domain Balanced Sampling): Giải thích cơ sở lý thuyết vì sao phải phối trộn 1:1 giữa quỹ đạo tác vụ cơ sở và quỹ đạo có bộ nhớ nhằm ngăn chặn hiện tượng quên thảm khốc (catastrophic forgetting).
Công thức hàm mất mát (Dòng 113-115):
Đánh giá: Công thức cực tiểu hóa kỳ vọng kép 
min
⁡
𝜙
1
2
𝐸
b
a
s
e
𝐿
f
l
o
w
+
1
2
𝐸
m
e
m
𝐿
f
l
o
w
min
ϕ
	​

2
1
	​

E
base
	​

L
flow
	​

+
2
1
	​

E
mem
	​

L
flow
	​

 là chuẩn mực cho bài toán Multi-task / Domain Balancing.
4. Mục 3.4: Suy luận và Quyền can thiệp đoạn hành động
Cơ chế ngắt đoạn hành động (Dynamic Chunk Preemption - 3.4.4):
Công thức: 
𝑞
′
=
min
⁡
(
𝑞
+
𝐾
,
inf
⁡
{
𝑡
>
𝑞
:
𝑡
−
𝑞
≥
𝑚
,
𝐷
𝑡
=
1
}
)
q
′
=min(q+K,inf{t>q:t−q≥m,D
t
	​

=1}).
Đánh giá: Rất xuất sắc. Đây là đóng góp phương pháp luận rất hay cho bài toán điều khiển VLA dạng Action Chunking. Các mô hình VLA thông thường thực thi khối 
𝐾
K hành động một cách mù quáng (open-loop). Việc có bộ điều phối phát hiện thay đổi ngữ nghĩa bền vững (
𝐷
𝑡
=
1
D
t
	​

=1) qua 
𝑝
p bước và vượt ngưỡng quán tính tối thiểu (
𝑡
−
𝑞
≥
𝑚
t−q≥m) để tái truy vấn sớm là hoàn toàn hợp lý về mặt lý thuyết điều khiển.
Độ trễ phản ứng (Mục 3.4.5, Dòng 156):
Công thức: 
𝑞
′
=
min
⁡
(
𝑞
+
𝐾
,
max
⁡
(
𝑞
+
𝑚
,
𝑟
+
𝑝
−
1
)
)
q
′
=min(q+K,max(q+m,r+p−1)).
Đánh giá: Chứng minh logic cực kỳ chặt chẽ: biến cố xuất hiện tại 
𝑟
r, cần 
𝑝
p bước để xác nhận 
→
→ hoàn tất tại 
𝑟
+
𝑝
−
1
r+p−1; đồng thời phải đảm bảo đã chạy tối thiểu 
𝑚
m bước 
→
𝑞
+
𝑚
→q+m. Do đó điểm ngắt sớm nhất là 
max
⁡
(
𝑞
+
𝑚
,
𝑟
+
𝑝
−
1
)
max(q+m,r+p−1), chặn trên bởi chu kỳ định kỳ 
𝑞
+
𝐾
q+K.
PHẦN 2: KIỂM TRA CÂU TỪ, VĂN PHONG & QUY CHUẨN
1. Các vi phạm quy chuẩn nghiêm ngặt cần sửa ngay
Vị trí	Lỗi vi phạm	Lý do & Cách sửa
Dòng 8	pdftitle={... — ...} dùng gạch ngang dài em-dash —	Vi phạm quy tắc thesis_notation.txt dòng 19 ("tuyệt đối không dùng gạch ngang lạ, dài... chỉ dùng gạch ngắn đơn '-'"). Đổi thành -.
Dòng 41, 103, 124	\head{3.2\quad ...} chứa \quad trong \pdfbookmark	Gây lỗi biên dịch: Package hyperref Warning: Token not allowed in a PDF string (Unicode): \hskip 10.95pt. Cần sửa định nghĩa lệnh \head hoặc tách bookmark plain-text.
Dòng 121	\texttt{results_f1_author_split_v2}, \texttt{train_memory_sft.py}, \texttt{replay_sampler.py}	Vi phạm trực tiếp quy tắc dòng 15 thesis_notation.txt ("không mention đến các tên file local"). Cắt bỏ toàn bộ dòng này hoặc thay bằng mô tả phương pháp.
Dòng 158	Nguồn: biên bản D2 ngày 20/11/2026.	Tham chiếu nhật ký/biên bản nội bộ. Chuyển thành: "Kết quả phân tích từ thực nghiệm đánh giá cơ chế truy vấn sớm."
Dòng 105, 114	A_q^{\rm demo}, \mathcal L_{\rm flow}	Cú pháp LaTeX lỗi thời (\rm), dễ gây lỗi font trong XeLaTeX/LuaLaTeX. Sửa thành A_q^{\mathrm{demo}} và \mathcal L_{\mathrm{flow}}.
2. Rà soát các câu lệnh chỉ đạo (Meta-Instructions) và đề xuất viết lại

Nhiều câu trong dàn ý đang viết theo dạng ghi chú công việc của tác giả thay vì đề cương khoa học. Cần biên tập lại như sau:

Dòng 33:
Hiện tại: Viết ba phép biến đổi. Bộ ghi cập nhật...
Sửa lại: Khung kiến trúc đề xuất vận hành dựa trên ba thành phần xử lý chính: Bộ ghi cập nhật trạng thái...
Dòng 50:
Hiện tại: Viết $M_t=(\mathcal O_t,\mathcal R_t,\mathcal E_t,\mathcal P_t)$ gồm các thực thể...
Sửa lại: Trạng thái bộ nhớ tại bước $t$ được hình thức hóa thành bộ bốn $M_t=(\mathcal O_t,\mathcal R_t,\mathcal E_t,\mathcal P_t)$, bao gồm...
Dòng 92:
Hiện tại: Thử cắt hai quỹ đạo ở $q$ rồi thay phần tiếp diễn để kiểm tra thuộc tính này.
Sửa lại: Tính không nhìn trước có thể được kiểm chứng thực nghiệm bằng giao thức hoán đổi phần tiếp diễn (suffix swap test) giữa hai quỹ đạo có chung tiền tố.
Dòng 105:
Hiện tại: Trình bày từ một mẫu $(o_q,C_q,A_q^{\rm demo})$. Chạy bộ ghi theo thứ tự thời gian...
Sửa lại: Quá trình huấn luyện sử dụng các bộ ba mẫu $(o_q, C_q, A_q^{\mathrm{demo}})$, trong đó bộ ghi vận hành tuần tự theo thời gian...
Dòng 108:
Hiện tại: Mô tả VLM mã hóa ảnh và văn bản, còn action expert dự đoán...
Sửa lại: Kiến trúc mô hình phân tách rõ hai vai trò: phần nền tảng VLM đảm nhiệm mã hóa quan sát hình ảnh và văn bản ngữ cảnh, trong khi module chuyên trách hành động (action expert) dự đoán đoạn điều khiển...
Dòng 111:
Hiện tại: Viết mục tiêu theo hai kỳ vọng dưới phân phối lấy mẫu thực tế:
Sửa lại: Hàm mục tiêu tối ưu hóa được thiết lập nhằm cân bằng kỳ vọng mất mát trên hai phân phối dữ liệu:
Dòng 153:
Hiện tại: Trình bày thuật toán bằng ba thao tác. Thi hành một bước và...
Sửa lại: Quy trình thực thi và điều phối được chuẩn hóa thành ba giai đoạn kế tiếp: thực thi một bước hành động, kiểm tra điều kiện kích hoạt ngắt, và tái truy vấn chính sách khi phát hiện thay đổi ngữ nghĩa.
3. Cải thiện tính liên kết câu văn (Cohesion & Connectors)

Một số đoạn văn bị ngắt câu quá vụn (staccato style). Ví dụ tại Mục 3.1.1 (Dòng 28):

Bản gốc:

"Robot cần lấy đồ vật A từ một trong hai hộp kín có danh tính phân biệt được. Ở lượt thứ nhất robot từng thấy A được đặt vào hộp X, ở lượt thứ hai A được đặt vào Y. Hai lượt kết thúc bằng cùng cảnh hai hộp đã đóng. Giả sử các hộp không bị mở hay tráo đồ vật trong khoảng còn lại. Cùng một ảnh và chỉ dẫn dẫn đến hai lựa chọn hộp khác nhau. Dùng tình huống này xuyên suốt chương, rồi nới giả thiết bảo toàn khi bàn về che khuất."

Đề xuất viết lại mượt mà, liên kết chặt chẽ:

"Xét một bài toán điều khiển điển hình: robot có nhiệm vụ lấy đồ vật A từ một trong hai hộp kín có danh tính phân biệt được. Trong kịch bản thứ nhất, robot đã quan sát thấy đồ vật A được đặt vào hộp X; ngược lại, ở kịch bản thứ hai, A được đưa vào hộp Y. Cả hai kịch bản đều kết thúc bằng trạng thái hai hộp đã đóng kín. Dưới giả định vật lý bảo toàn (các hộp không bị di chuyển hay tráo đổi đồ vật trong thời gian che khuất), tại thời điểm quyết định cuối cùng, robot phải đối mặt với hai quan sát hình ảnh và chỉ dẫn hoàn toàn giống nhau nhưng hành động đúng lại dẫn tới hai hộp khác nhau. Tình huống phản ví dụ này minh họa rõ ràng sự thất bại tất yếu của các chính sách Markovian thuần túy khi thiếu bộ nhớ lịch sử, đồng thời là động lực xuyên suốt để xây dựng cơ chế bộ nhớ có bằng chứng trong chương này."