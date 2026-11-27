# Đề xuất rút gọn và định vị lại Chương 3 (Phương pháp)

Ngày: 2026-11-26. Nguồn đã khảo sát: `D:\Thesis_26\docs\thesis\latex\content\chapter3.tex` (sửa lần cuối 26/11 14:14), đối chiếu với `chapter2.tex`, `chapter4.tex`, `chapter1.tex` và bản build `main.pdf` (15:50 cùng ngày). Đây là **đề xuất**, chưa sửa file nào.

Số dòng (Lx) trong báo cáo là số dòng của `chapter3.tex` hiện tại. Ước lượng trang dựa trên mật độ thực của chương: khoảng 600 từ văn bản một trang, mỗi hình khoảng 0,5 đến 0,7 trang.

---

## 1. Tóm tắt

- **Hiện trạng:** Chương 3 chiếm 16 trang (tr. 17 đến 32), khoảng 7.100 từ, có 5 hình, 2 bảng, 20 phương trình và 4 mệnh đề kèm chứng minh. Ch3 dài hơn Ch2 (10 trang) và gần bằng Ch4 (17 trang) dù Ch4 chứa toàn bộ kết quả.
- **Nguyên nhân chính khiến chương dài:** (a) nhắc lại lập luận POMDP và bùng nổ token của Ch2; (b) mỗi công thức được giải thích hai lần, một lần ngay sau công thức và một lần nữa ở đoạn "ý nghĩa"; (c) chi tiết hiện thực (ngưỡng, số tham số, bảng 10 tác vụ) nằm ngay trong phần phương pháp; (d) 4 mệnh đề có chứng minh dài trong khi 2 mệnh đề gần như là định nghĩa.
- **Nếu làm hết các đề xuất:** Ch3 còn khoảng **9 đến 10 trang** (bớt khoảng 6,5 đến 7 trang). Phụ lục mới "Chứng minh và chi tiết hiện thực" thêm khoảng 3 trang. Tổng luận văn giảm khoảng 3,5 đến 4 trang.

---

## 2. Danh sách cắt hoặc gộp cụ thể

Cột "Tiết kiệm" là số trang ước lượng ở Ch3, sau khi đã trừ phần chuyển sang phụ lục nếu có.

### 2.1. Mục 3.1: Phát biểu bài toán và tổng quan (hiện tại khoảng 2 trang)

| # | Vị trí | Vấn đề | Đề xuất | Tiết kiệm |
|---|---|---|---|---|
| A1 | L7 (170 từ) | Nhắc lại gần như nguyên văn §2.1 và §2.2: quan sát giống nhau dù tiến độ khác, lịch sử thô làm bùng nổ token. Phương trình (2.3) `eq:event-projection-policy` của Ch2 cũng đã nêu $C_t = R_B(\ell, M_t)$, $\lvert C_t\rvert \le B$. | Rút còn 2 câu: "Theo Mục 2.2, … Phương pháp thay lịch sử $h_t$ bằng trạng thái bộ nhớ $M_t$ và ngữ cảnh $C_q$ có độ dài bị chặn:" rồi đưa ngay (3.1). | 0,2 |
| A2 | L18 và L28 (276 từ) | Đoạn L28 diễn giải lại từng ký hiệu và 4 bước vận hành, trong khi (3.1) và Hình 3.1 đã thể hiện đủ. Đoạn L18 mô tả màu mũi tên, việc này nên nằm trong chú thích hình. | Gộp thành 1 đoạn khoảng 80 từ: định nghĩa $A_q$, $H$, rồi một câu về bộ điều phối. Chuyển mô tả mũi tên vào `\caption`. Xóa dòng chú thích `%` ở L27. | 0,3 |
| A3 | L30 đến L37, phương trình (3.2) `eq:error-analysis` (217 từ) | Đây là mô hình đồ chơi (hai trạng thái đồng xác suất), không được Ch4 dùng lại và không có `\ref` từ chương khác. Cụm "Tính an toàn của cơ chế…" cũng dùng sai từ. Kết luận "chỉ giảm khi và chỉ khi $\epsilon<1/2$" là hiển nhiên. | Giữ một câu nguyên tắc thiết kế: "bộ nhớ trả *chưa xác định* thay vì đoán khi thiếu căn cứ". Đưa (3.2) và lập luận xuống phụ lục, hoặc bỏ hẳn. | 0,4 |

### 2.2. Mục 3.2.1: Cấu trúc bản ghi (hiện tại khoảng 2 trang)

| # | Vị trí | Vấn đề | Đề xuất | Tiết kiệm |
|---|---|---|---|---|
| A4 | L50 (228 từ) | Một câu dài định nghĩa 7 trường. Câu khó đọc và có chi tiết thừa ("với sự kiện lịch sử thì $t_{end}=\infty$…"). | Dùng `itemize` ngắn, mỗi trường một dòng khoảng 10 từ. Bỏ chi tiết $\mathcal P$ (định danh khung ảnh, tọa độ), vì ví dụ (3.4) đã cho thấy. | 0,15 |
| A5 | L52 đến L61 (340 từ) | Ví dụ $f_{ex}$ được giải thích bằng 2 đoạn. Ý "$\tau$ khác $\nu$, cần $p=3$ quan sát, không rò rỉ tương lai" xuất hiện **4 lần**: L59, L61, (3.8) cùng L95, và Mệnh đề 1 cùng Hình 3.3. Đoạn L61 còn tự gọi đây là "một đóng góp thiết kế quan trọng". | Giữ (3.4). Gộp L59 và L61 thành 1 đoạn khoảng 90 từ. Ý "không rò rỉ" chỉ nêu một lần tại Mệnh đề 1. | 0,4 |
| A6 | L79 (197 từ) | Đoạn phân biệt $\mathcal R_t$ với $\mathcal E_t$ đã được Hình 3.2 minh họa. Có cụm "nhận thức học" không cần thiết. | Rút còn khoảng 80 từ: định nghĩa hai tập và quy tắc "$\mathcal R_t$ chuyển *chưa xác định* khi che khuất, $\mathcal E_t$ thì không". | 0,2 |

### 2.3. Mục 3.2.2: Trích xuất và kiểm định bằng chứng (hiện tại khoảng 2 trang)

| # | Vị trí | Vấn đề | Đề xuất | Tiết kiệm |
|---|---|---|---|---|
| A7 | L83 đến L91, (3.6) `eq:scale-invariant-distance` | Đây là dạng tổng quát của đúng công thức được viết lại ở (3.11) `eq:case-study-geometry`, chỉ đổi ký hiệu $\mathbf x\to\mathbf b$, $\mathbf w\to(w_b,h_b)$. Hai lần định nghĩa cùng một đại lượng. | Bỏ (3.6) và đoạn L91. Định nghĩa $D_t$, $L_t$ một lần ở phần hiện thực, kèm một câu giải thích tính bất biến tỷ lệ. | 0,3 |
| A8 | L93 (141 từ) | Đoạn theo dõi danh tính khi che khuất không có công thức, không có ngưỡng và không được đánh giá, vì T7 đến T10 không chạy. | Rút còn 1 câu ("thực thể bị che khuất giữ danh tính và chuyển sang *occluded*"). Có thể nêu ở Ch5 như hướng mở rộng. | 0,2 |
| A9 | L102 đến L109 (200 từ) | Bất biến FSM (3.9) được giải thích 2 lần: L109 nói "cho biết…" rồi "được kiểm tra bằng quy nạp…". | Giữ (3.9) và một câu ý nghĩa. Ý quy nạp có thể chuyển xuống phụ lục. | 0,2 |
| A10 | L111 đến L127, Mệnh đề 1 cùng chứng minh (245 từ) | Chứng minh 144 từ cho một sự kiện hiển nhiên: $U$ và $R_B$ là hàm xác định của tiền tố. Đoạn L127 lặp lại ý. | Giữ phát biểu (Mệnh đề 3.5 dùng nó). Chứng minh rút còn 1 câu ngay sau đó, hoặc chuyển xuống phụ lục. | 0,2 |

### 2.4. Mục 3.2.3: Trường hợp cụ thể (hiện tại khoảng 2 trang)

| # | Vị trí | Vấn đề | Đề xuất | Tiết kiệm |
|---|---|---|---|---|
| A11 | L131 đến L156, Bảng 3.1 (10 tác vụ T1 đến T10) | Chỉ T1 và T3 được đánh giá. Bảng tạo cảm giác phương pháp đã được kiểm chứng trên 10 tác vụ. Câu L156 viết "khẳng định tính tổng quát" là overclaim. Nội dung này chồng với §2.6, nơi giới thiệu LIBERO-Mem. | Thay bằng 2 câu: "giao thức $T_{spec}=(\Phi,\Delta,\mathcal F)$ có thể mô tả các lớp OM/OS/OR/OO của LIBERO-Mem (Phụ lục …); khóa luận hiện thực và đánh giá lớp OM/OS qua T1, T3". Chuyển bảng xuống phụ lục hoặc bỏ. | 0,8 |
| A12 | L167 đến L187, Bảng 3.2 (7 ngưỡng) | Đây là chi tiết hiện thực, không phải phương pháp. Ch3 giữ (3.12) là đủ. **Cần kiểm tra lại**: bảng ghi "Trạng thái ổn định: độ dịch tâm ≤ 0,15", còn (3.12) dùng $L_t \le 0{,}15$ (độ nâng), hai điều kiện này khác nhau. | Chuyển Bảng 3.2 xuống phụ lục (hoặc §4.1), gộp chung với các ngưỡng proprioception của A19 thành **một bảng ngưỡng duy nhất**. Khi chuyển thì sửa cho khớp với (3.12). | 0,6 |

### 2.5. Mục 3.3: Huấn luyện thích ứng (hiện tại khoảng 2 trang)

| # | Vị trí | Vấn đề | Đề xuất | Tiết kiệm |
|---|---|---|---|---|
| A13 | L207 đến L220 | Có 3 chỗ trùng. (1) Công thức LoRA $W+\frac{\alpha}{r}B_LA_L$ đã có ở §2.5. (2) Lý do trộn dữ liệu cân bằng đã có ở §2.5 (Balanced Data Diet). (3) "645.120 tham số" và "11,87 triệu" xuất hiện lại ở §4.1, cả trong văn bản lẫn Bảng 4.3. Câu "loại bỏ hoàn toàn nguy cơ rò rỉ" (L209) lặp ý Mệnh đề 1. | Giữ ba bước, giữ (3.13) `eq:training-loss` vì tỷ lệ 1:1 là lựa chọn riêng của khóa luận. Bỏ công thức LoRA, chỉ ghi "LoRA (Mục 2.5) với $r=16$, $\alpha=32$ trên $W_q, W_v$". Số tham số chỉ để ở §4.1. | 0,3 |
| A13b | Hình 3.4 `smolvla-lora` | Hình đẹp nhưng thuộc về cấu hình mô hình. | Có thể chuyển sang §4.1 cạnh Bảng 4.3, không bắt buộc. | (0,5 nếu chuyển) |

### 2.6. Mục 3.4: Suy luận và điều phối (hiện tại khoảng 2 trang)

| # | Vị trí | Vấn đề | Đề xuất | Tiết kiệm |
|---|---|---|---|---|
| A14 | L231 đến L240 (207 từ) | Đoạn mở có các cụm "hai yêu cầu khắt khe", "nghiêm ngặt". Ý "$B=48$ nên số token cố định" được nói 2 lần. | Rút còn khoảng 70 từ. | 0,15 |
| A15 | L242 (215 từ) | Giải thích lại action chunking, đã có ở §2.5. Ví dụ "8 hành động còn lại" thì hữu ích. | Rút còn khoảng 100 từ: định nghĩa $e_t$, $k_{\min}$ rồi đến (3.15). | 0,2 |
| A16 | L258 đến L276, Mệnh đề 2 cùng hai phương trình và chứng minh (360 từ) | (3.16) suy ra trực tiếp từ (3.15) bằng cách thế $e_t=\mathbf 1[t\ge r+p-1]$, nên thực chất là hệ quả của định nghĩa. L258 lại gọi nó là "định lý". Chứng minh có câu "Trừ $r$ … ta được … phương trình sau" trong khi phương trình nằm *trước*. Con số "4 đến 30 bước" (thực nghiệm) lọt vào chứng minh và **mâu thuẫn với Ch4 L191**, nơi ghi "từ 7 bước đến hàng chục bước". | Hạ thành **Nhận xét**: giữ (3.17) `eq:latency-bound` kèm một câu diễn giải. Bỏ (3.16) và chứng minh, hoặc đưa xuống phụ lục. Chuyển số đo độ trễ về Ch4 và thống nhất một con số. | 0,5 |
| A17 | L278 (60 từ) | Đoạn suy luận bất đồng bộ nằm ngoài phạm vi khóa luận. | Chuyển sang Ch5, mục hướng phát triển. | 0,1 |

### 2.7. Mục 3.5: Cơ chế dừng (hiện tại khoảng 3,5 trang)

| # | Vị trí | Vấn đề | Đề xuất | Tiết kiệm |
|---|---|---|---|---|
| A18 | L283 (193 từ) | Phần động cơ về overshoot trùng với phần mở §4.3/§4.4 và §2.2 (ví dụ bát lần thứ ba). | Rút còn khoảng 90 từ: giả định ngầm "VLA tự dừng" và vì sao demo không dạy hành vi dừng. Ý "ba hành động: tiếp tục, truy vấn lại, dừng" là ý hay, nên đưa lên làm khung cho cả mục (xem 4.2). | 0,2 |
| A19 | L292 | Các ngưỡng 0,08 m, 0,03 m, 0,06 m là chi tiết hiện thực. | Giữ nguyên tắc "mỗi pha kẹp tối đa 1 tín dụng; hợp hai nguồn". Đưa ngưỡng vào bảng ngưỡng chung ở phụ lục (A12). | 0,1 |
| A20 | L294 đến L305, mô hình xác suất cùng Mệnh đề 3 (650 từ) | Phần giá trị nhất là phân rã $\Delta$ (3.19): "cứu được trừ làm hỏng". Hai cận $f\le\bar m\varepsilon$ và $r\ge c^K(1-\bar m\varepsilon)\kappa_s$ cần giả định độc lập giữa các chu kỳ, nhưng **giả định này chưa được nêu**. Cụm "dưới các giả định về tính nhân quả…" quá mơ hồ để làm chứng minh. | Ở thân chương chỉ giữ định nghĩa $p_c, p_o, r, f, \rho$, phương trình (3.19) và một đoạn diễn giải, khoảng 0,8 trang. Chuyển hai cận, danh sách giả định đầy đủ (bản nháp G1 đến G6 trong `halt_closed_loop/thesis/ch3_halt_method.tex`) và chứng minh xuống phụ lục. | 0,8 |
| A21 | L307 đến L316, Mệnh đề 4 (330 từ) | Bất đẳng thức $V(\text{thời gian})\le V(\text{bộ nhớ})$ là hệ quả tầm thường của việc lọc thông tin lồng nhau. Đoạn L316 kết luận "bằng chứng bộ nhớ vượt trội hơn các quy tắc định thời gian". **Rủi ro:** phân tích phản thực tế trước đây cho thấy bước dừng cố định tốt nhất nhìn lại (bước 120) cũng đạt 9/40 trên E2, bằng H3. Hội đồng có thể hỏi đúng điểm này. | Chuyển Mệnh đề 4 và chứng minh xuống phụ lục. Ở thân chương giữ 2 câu với phạm vi trung thực: "lợi thế của bộ nhớ so với quy tắc theo thời gian kỳ vọng xuất hiện khi $T_c$ phân tán (đa chu kỳ); trên T1, một bước cố định được tinh chỉnh có thể đạt mức tương đương". | 0,6 |

**Tổng các mục A1 đến A21:** khoảng **6,9 trang** (chưa tính A13b).

### 2.8. Gộp hình (tùy chọn)

| # | Đề xuất | Tiết kiệm |
|---|---|---|
| B1 | Gộp Hình 3.2 (sự kiện khi che khuất) và Hình 3.3 (trục $\tau,\nu,q$) thành một hình hai panel (a) và (b), vì cả hai minh họa ngữ nghĩa thời gian của bản ghi. | 0,4 |

---

## 3. Mệnh đề và chứng minh

| Mệnh đề | Vai trò trong lập luận | Đề xuất |
|---|---|---|
| **MĐ1: Không nhìn trước** | Bắt buộc: bảo đảm dữ liệu huấn luyện không rò rỉ và $\tau$ của cơ chế dừng là thời điểm dừng hợp lệ. | **Giữ phát biểu** ở thân chương. Chứng minh rút còn 1 câu. |
| **MĐ2: Chặn độ trễ** | Hệ quả trực tiếp của định nghĩa (3.15). | **Hạ thành Nhận xét**, giữ (3.17). Chứng minh đưa xuống phụ lục hoặc bỏ. |
| **MĐ3: Phân rã lợi ích dừng** | Có giá trị: giải thích *vì sao* dừng giúp X-VLA ($p_o\approx1$, $p_c$ khá lớn) mà không giúp SmolVLA ($p_c$ nhỏ, vì $\Delta\propto p_c$). Ch4 nên dùng lại để đọc kết quả. | **Giữ phương trình (3.19) và diễn giải** ở thân chương. **Hai cận độ tin cậy và chứng minh** chuyển xuống phụ lục kèm giả định đầy đủ. |
| **MĐ4: Giá trị của bằng chứng** | Phần đầu tầm thường. Phần chặn quy tắc mù đúng nhưng chưa được Ch4 kiểm chứng trực tiếp. | **Chuyển hẳn xuống phụ lục**, giữ 2 câu nhận xét có phạm vi (xem A21). |
| Phân tích sai số (3.2) | Mô hình đồ chơi. | Phụ lục hoặc bỏ. |
| Quy nạp bất biến FSM | Hiển nhiên. | Phụ lục, 3 dòng. |

**Lưu ý kỹ thuật cho phụ lục:** `main.tex` hiện không render phụ lục nào (L108: "Phụ lục được lưu giữ cục bộ…, không render ra bản in theo yêu cầu"). Muốn chuyển chứng minh xuống phụ lục thì phải thêm `\appendix` và một file mới, ví dụ `appendices/proofs.tex`, chỉ chứa chứng minh và bảng ngưỡng, còn provenance vẫn để ẩn. Nên dùng `amsthm` (`\newtheorem{proposition}`, `\begin{proof}`) thay cho `\textbf{Mệnh đề k}` gõ tay. Cách này cho số tự động theo chương (3.1, 3.2), `\ref` được từ Ch4 và trông gọn hơn. Hiện `main.tex` chưa nạp `amsthm`.

---

## 4. Cấu trúc mới đề xuất cho Ch3 (khoảng 9 đến 10 trang)

```
3  PHƯƠNG PHÁP
3.1 Bài toán và tổng quan (~1 tr)
    (3.1) + Hình kiến trúc + 1 đoạn "khác biệt so với công trình trước" (xem Mục 6)
3.2 Bộ nhớ sự kiện có bằng chứng (~2,5 tr)
    bản ghi f (itemize) + ví dụ; M_t = (O,R,E,P); xác nhận trễ p; bất biến FSM;
    Mệnh đề 3.1 (không nhìn trước); Hình 3.2 gộp (a)(b)
3.3 Hiện thực cho tác vụ lặp chu kỳ (~1 tr)
    D_t, L_t (một lần) + quy tắc sự kiện (3.x) + nguồn proprioception;
    bảng ngưỡng -> Phụ lục A
3.4 Huấn luyện thích ứng (~1,2 tr)
    3 bước + (loss 1:1) + Hình LoRA (hoặc chuyển sang 4.1)
3.5 Bộ điều phối: tiếp tục, truy vấn lại, dừng (~3 tr)   <- gộp 3.4 cũ và 3.5 cũ
    3.5.1 Chiếu ngữ cảnh dưới ngân sách B
    3.5.2 Truy vấn sớm theo bằng chứng + Nhận xét độ trễ
    3.5.3 Dừng theo bằng chứng: quy tắc τ, hợp nhất nguồn, Mệnh đề 3.2 (phân rã Δ)
...
Phụ lục A  Chứng minh và chi tiết hiện thực (~3 tr)
    A.1 phân tích sai số; A.2 quy nạp FSM; A.3 chứng minh độ trễ;
    A.4 giả định G1–G6 + cận f, r; A.5 Mệnh đề giá trị bằng chứng; A.6 bảng ngưỡng; A.7 bảng T1–T10
```

Gộp 3.4 cũ và 3.5 cũ thành một bộ điều phối ba hành động là thay đổi quan trọng nhất về mặt trình bày. Cách gộp này biến hai cơ chế rời rạc thành **một** ý tưởng nhất quán: bộ nhớ không chỉ là chữ đưa vào VLA, mà còn là tín hiệu điều khiển việc *khi nào* gọi lại và *khi nào* dừng. Ch4 cũng đã kể theo trình tự này (§4.3 truy vấn sớm thất bại, dẫn đến §4.4 dừng).

---

## 5. Câu chữ phóng đại nên bỏ (vừa ngắn hơn vừa khoa học hơn)

Các từ sau xuất hiện nhiều trong Ch3 và làm chương vừa dài hơn vừa kém thuyết phục với người chấm. Đây là một nguyên nhân khiến supervisor thấy chương "nhiều chữ":

| Dòng | Cụm hiện tại | Gợi ý |
|---|---|---|
| L43 | "khắc phục triệt để hiện tượng mất dấu lịch sử và ảo giác" | 
| L61 | "một đóng góp thiết kế quan trọng", "tuyệt đối không được phép", "ngăn chặn triệt để" | bỏ/giữ một số câu mô tả |
| L79 | "khác biệt căn bản về mặt nhận thức học", "ba dạng truy vấn trọng yếu" | "khác nhau ở chỗ…" |
| L83 | "loại bỏ hoàn toàn sự phụ thuộc vào góc đặt camera" | "giảm phụ thuộc vào…" (chuẩn hóa theo kích thước không loại bỏ hoàn toàn phối cảnh) |
| L127 | "loại bỏ khả năng mô hình học vẹt" | MĐ1 chỉ bảo đảm không rò rỉ tương lai; nó **không** loại trừ đường tắt thời gian. Nên bỏ vế này. |
| L209, L211 | "nghiêm ngặt", "loại bỏ hoàn toàn", "ngăn chặn" catastrophic forgetting | "giảm"; việc trộn 1:1 *nhằm* hạn chế quên, Ch4 đo được F1-off còn 23/50, nên không thể nói "ngăn chặn" |
| L231 | "hai yêu cầu khắt khe" | bỏ |
| L258 | "được chứng minh qua định lý" | "Nhận xét sau…" |
| L292 | "triệt tiêu hoàn toàn nguy cơ đếm trùng" | "tránh đếm trùng trong cùng một pha kẹp" |
| L305 | "khẳng định vai trò quyết định" | "gợi ý rằng…" |
| L316 | "làm sáng tỏ nền tảng lý thuyết vì sao … vượt trội", "bảo đảm … diễn ra chính xác" | xem A21 |

---

## 6. Định vị lại đóng góp để phương pháp đọc ra "mới" hơn

### 6.1. Vì sao hiện tại đọc ra "không mới"

Ch3 hiện trình bày theo dạng *xây kiến trúc*: bản ghi 7 trường, FSM, cửa sổ xác nhận, LoRA, chiếu văn bản. Người chấm sẽ nhận ra từng mảnh đều đã có sẵn: FSM / reward machine, hysteresis, Grounding DINO, LoRA, memory-as-text như MEM, kết hợp sự kiện với quỹ đạo như EventSync. Ch1 (L57) cũng liệt kê đóng góp 2 theo kiểu "xây dựng… tích hợp Grounding DINO và LoRA", tức là kể ra công cụ thay vì phát hiện. Khi đóng góp được mô tả bằng các thành phần, người đọc sẽ đánh giá độ mới theo từng thành phần, và thành phần nào cũng đã có.

### 6.2. Gợi ý: kể một luận điểm, không kể một hệ thống

Kết quả thực nghiệm của luận văn thực ra ủng hộ một câu chuyện sắc hơn kiến trúc:

1. **Phát hiện:** trên tác vụ lặp, VLA nhỏ học từ demo **hoàn thành được chu kỳ nhưng không biết dừng**. X-VLA đạt SCR 0,83 đến 0,92 nhưng SR = 0 vì overshoot. Demo kết thúc ngay ở mốc cuối nên chính sách chưa bao giờ thấy trạng thái "đứng yên sau khi xong".
2. **Phát hiện phụ:** đưa cùng thông tin tiến độ vào lời nhắc (F1, truy vấn sớm) **không** sửa được lỗi này. Ch4 cho thấy `Memory: …` không mang tính cưỡng chế, E4/E5 chỉ nhạy cục bộ, và truy vấn sớm không đổi SR.
3. **Đề xuất tối giản:** dùng *cùng* bộ nhớ đó làm **thời điểm dừng nhân quả bên ngoài chính sách**. Cách này không cần huấn luyện lại, không đổi lời nhắc và áp dụng được cho bất kỳ VLA nào chạy theo đoạn hành động.
4. **Bằng chứng có đối chứng:** 0 lên 9/40 trên E2 vòng kín. Placebo dừng ngẫu nhiên đạt trung bình 3,7/40, cho thấy *bằng chứng* chứ không chỉ *hành động dừng* tạo ra phần chênh. Seen 12 cặp đạt 0 lên 6/12, $p=0{,}031$.
5. **Lý thuyết phục vụ luận điểm:** MĐ3 dự đoán khi nào cơ chế dừng có lợi ($p_o$ cao, $p_c$ đủ lớn). Dự đoán này khớp với việc X-VLA được lợi còn SmolVLA thì không.

Viết lại một câu cho phần mở Ch3 theo hướng này:

> "Chương này đề xuất dùng bộ nhớ sự kiện có bằng chứng theo hai vai trò: làm ngữ cảnh ngôn ngữ cho chính sách, và làm *tín hiệu điều phối* quyết định khi nào truy vấn lại và khi nào dừng. Vai trò thứ hai xuất phát từ quan sát rằng chính sách học từ minh họa thiếu khả năng kết thúc tác vụ lặp, và thông tin tiến độ đặt trong lời nhắc không đủ để chính sách dừng."

Liên hệ lý thuyết cần nêu một cách trung thực. Quy tắc dừng của luận văn chính là **điều kiện kết thúc của option** trong khung options (Sutton, Precup và Singh, 1999), và gần với bài toán **dừng tối ưu (optimal stopping)**. Nói rõ điều này *tăng* độ tin cậy chứ không làm giảm độ mới. Đóng góp nằm ở chỗ hiện thực hóa điều kiện kết thúc đó từ bằng chứng có kiểm chứng cho VLA chạy theo đoạn hành động, chứng minh tính nhân quả và đo lợi ích có đối chứng. (Hiện `references.bib` chưa có tài liệu options; nếu đồng ý thì cần thêm.)


### 6.4. Sửa Ch1 cho khớp (ngoài phạm vi Ch3, chỉ ghi nhận)

Đóng góp 2 ở Ch1 L57 nên viết lại theo luận điểm ở 6.2 (bộ nhớ làm tín hiệu dừng), bỏ phần liệt kê "tích hợp Grounding DINO và LoRA". Nên nhấn mạnh cơ chế dừng hơn 1 chút nữa, vì đây là kết quả dương rõ nhất của luận văn.

---

## 7. Lỗi nhỏ phát hiện khi khảo sát (sửa khi rút gọn)

1. Mâu thuẫn độ trễ bộ ghi: Ch3 L274 ghi "4 đến 30 bước", Ch4 L191 ghi "7 bước đến hàng chục bước".
2. Bảng 3.2 và (3.12) không khớp ở điều kiện "ổn định" và $L_t\le0{,}15$ (xem A12).
3. Chứng minh MĐ2 nhắc tới "phương trình sau" nhưng phương trình nằm trước.
4. Ký hiệu $K$ ở §3.5 (số chu kỳ mục tiêu) trùng với $K$ của Ch2 (số khung hình trong cửa sổ, (2.2)), và với $N$ ở §4.1 (số chu kỳ yêu cầu). Nên thống nhất dùng $N$.
5. Ký hiệu $\tau$ vừa là thời điểm vật lý trong bản ghi $f$ (3.3), vừa là thời điểm dừng (3.18). Nên đổi thời điểm dừng thành $\tau_h$ hoặc $T_h$.
6. Ký hiệu $r$ vừa là thời điểm biến cố (MĐ2), vừa là xác suất dừng kịp (MĐ3), vừa là hạng LoRA (§3.3).
7. Ký hiệu $c$ vừa là độ bao phủ ở (3.2) và MĐ3, vừa là điều kiện đầu vào ở (2.x) flow matching.

---

## 8. Bước tiếp theo đề xuất

Làm theo thứ tự sau:
1. Áp dụng nhóm cắt an toàn A1 đến A10, A13 đến A15, A17, A18 và Mục 5. Nhóm này chỉ cắt chữ, không đổi cấu trúc, tiết kiệm khoảng 3,5 trang.
2. Tạo `appendices/proofs.tex` cùng `amsthm`, rồi chuyển A3, A11, A12, A16, A19 đến A21. Bước này tiết kiệm thêm khoảng 3,4 trang.
3. Gộp §3.4 và §3.5 thành bộ điều phối ba hành động, viết đoạn mở và bảng khác biệt (Mục 6).
4. Sửa Ch1 (đóng góp) và quyết định về baseline bước cố định (6.5).
