<!-- # Tóm tắt

Trong thao tác robot ở chuỗi tác vụ dài, các quan sát cảm biến thu được sau khi hoàn tất một chu kỳ thao tác thường tương tự với quan sát ban đầu, trong khi quyết định tiếp tục hay dừng chuỗi hành động lại phụ thuộc vào số chu kỳ và mốc tiến độ đã hoàn tất. Nhằm giải quyết hiện tượng đa nghĩa quan sát này, khóa luận đề xuất một kiến trúc bộ nhớ theo sự kiện có khả năng biểu diễn mạch lạc thông tin tiến độ và hỗ trợ các mô hình thị giác - ngôn ngữ - hành động (VLA) thực hiện các tác vụ chuỗi cần bộ nhớ. Nghiên cứu tập trung giải quyết bài toán trên bộ tiêu chuẩn thao tác chuỗi dài LIBERO-Mem, đặc biệt hướng tới việc nâng cao năng lực cho các mô hình VLA nhỏ gọn vốn chịu nhiều ràng buộc nghiêm ngặt về độ dài ngữ cảnh và tài nguyên tính toán.

Phương pháp đề xuất biểu diễn tiến độ bằng cách xác định trạng thái vật thể và robot hiện tại, mô tả lại sự kiện đã xảy ra một cách thích ứng với tác vụ và quan sát. Mỗi sự kiện đều được lưu vết đầy đủ bằng chứng và thời điểm để sử dụng nhằm đảm bảo độ tin cậy và tính nhân quả. Ngay cả khi lịch sử chưa được quan sát đầy đủ, bộ nhớ vẫn có thể biểu diễn tiến độ một cách thích ứng. Hai tính chất của hệ bộ nhớ đề xuất được chứng minh trong phạm vi khóa luận bao gồm: Bộ đếm và ghi nhận sự kiện chỉ thay đổi khi xuất hiện một sự kiện kích hoạt hợp lệ; Nội dung cấp cho chính sách tại một thời điểm chỉ phụ thuộc vào dữ liệu sẵn có đến thời điểm đó. Bộ nhớ sự kiện này sau đó được chiếu (project) thành một bản tóm tắt ngôn ngữ tự nhiên gọn nhẹ, đưa vào đầu vào điều kiện hóa của mô hình VLA với phần sinh hành động đã được huấn luyện thích ứng.

Hiệu quả của phương pháp được kiểm chứng thông qua các thực nghiệm trên bộ tiêu chuẩn thao tác chuỗi dài LIBERO-Mem, bao gồm cả đánh giá ngoại tuyến về năng lực tri giác và đánh giá điều khiển vòng kín trên mô hình VLA. Kết quả cho thấy bộ theo dõi thị giác có khả năng phát hiện và đồng bộ chính xác các mốc sự kiện vật lý từ dòng quan sát thô mà không cần tiếp cận trạng thái đặc quyền của môi trường mô phỏng. Khi tích hợp vào vòng điều khiển kín, thông tin bộ nhớ sự kiện cung cấp tín hiệu tiến độ then chốt giúp mô hình phân giải tính đa nghĩa của quan sát thị giác, định hướng chuỗi hành động đúng tiến độ và ngăn ngừa các lỗi lặp lại chu kỳ hoặc kết thúc sớm. Đồng thời, các phân tích chuyên sâu làm sáng tỏ mối quan hệ phụ thuộc giữa độ trễ tích lũy bằng chứng, độ hoàn thiện của lịch sử quan sát và độ ổn định của chính sách điều khiển. Nghiên cứu qua đó xác lập một khung phương pháp luận có tính khả kiểm cao, đặt nền tảng vững chắc cho việc mở rộng kiến trúc bộ nhớ sự kiện - bằng chứng cho các bài toán thao tác đa bước phức tạp trên các mô hình robot học sâu.

**Từ khóa:** mô hình thị giác - ngôn ngữ - hành động, SmolVLA, bộ nhớ sự kiện, quan sát một phần, thao tác robot.

# Chương 1. Mở đầu

## 1.1. Động cơ nghiên cứu

Mô hình thị giác–ngôn ngữ–hành động (Vision–Language–Action, VLA) học cách ánh xạ quan sát và chỉ dẫn ngôn ngữ thành hành động của robot. Với các mô hình như SmolVLA, biểu diễn thị giác và ngôn ngữ được kết hợp với một mô-đun chuyên biệt để sinh chuỗi hành động. Khả năng này cho phép cùng một mô hình thực hiện nhiều thao tác được mô tả bằng ngôn ngữ tự nhiên. Trong tác vụ nhiều giai đoạn, điều kiện để chọn đúng hành động còn bao gồm tiến độ tích lũy của quá trình tương tác. Một chuỗi hành động chính xác ở từng giai đoạn vẫn có thể dẫn đến thất bại nếu robot lặp lại giai đoạn đã hoàn tất hoặc dừng trước khi đạt đủ số lần thao tác. [@smolvla2025; @liberomem2025]

Xét một robot được yêu cầu nhấc bát khỏi đĩa gốc và đặt lại đúng ba lần. Sau một lần đặt lại, quan hệ hình học giữa bát và đĩa có thể gần giống lúc khởi đầu. Khi đã thực hiện một hoặc hai chu kỳ, robot cần tiếp tục. Khi chu kỳ thứ ba đã hoàn tất, robot cần tránh thực hiện thêm một chu kỳ. Khác biệt cần thiết cho quyết định nằm ở lịch sử tương tác. Trong cách nhìn của bài toán quyết định dưới quan sát một phần, trạng thái phục vụ quyết định có thể bao gồm tiến độ mà cảm biến hiện tại chỉ phản ánh gián tiếp. Trường hợp này cho thấy vì sao thời lượng thực thi và mức phụ thuộc vào lịch sử cần được xem xét như hai thuộc tính riêng của tác vụ. [@pomdp1998; @liberomem2025; @robomme2026]

Khóa luận chọn thao tác lặp vì nó cho phép chỉ ra cụ thể thông tin nào bị thiếu và kiểm tra thông tin đó qua các mốc quan sát được. Một chu kỳ có hai mốc có thứ tự là bát rời đĩa và bát trở lại đúng đĩa. Số chu kỳ đã xác nhận là một giả thuyết cụ thể về phần lịch sử liên quan đến quyết định tiếp tục hay dừng. Việc khảo sát yêu cầu một lần và ba lần giúp quan sát cùng dạng thao tác ở hai mức yêu cầu tiến độ. Hai tác vụ vẫn có thể khác nhau về phân phối quỹ đạo và độ khó điều khiển, vì vậy chênh lệch điểm giữa chúng chỉ được diễn giải trong phạm vi cấu hình đã đo. SmolVLA được chọn làm mô hình nền để khảo sát một giao diện văn bản cho lịch sử trong khi giữ kiến trúc dự đoán hành động của mô hình. [@smolvla2025; @smolvlaCheckpoint; @liberomemSource]

## 1.2. Khoảng trống nghiên cứu và định vị đề tài

Các nghiên cứu về bộ nhớ cho VLA đã khảo sát nhiều cách biểu diễn lịch sử. MEM kết hợp lịch sử video ngắn với bản tóm tắt ngôn ngữ dài. MemoryVLA truy xuất ngữ cảnh từ bộ nhớ đặc trưng tri giác và nhận thức. LIBERO-Mem nhấn mạnh diễn tiến của từng thực thể, còn RoboMME cho thấy hiệu quả của các cách tích hợp bộ nhớ phụ thuộc vào loại tác vụ. Các kết quả này đặt ra câu hỏi cụ thể hơn cho mô hình và tác vụ đang khảo sát: nội dung lịch sử nào liên quan trực tiếp đến quyết định, nội dung đó được suy ra từ bằng chứng nào, và khi nào có thể cấp nó cho chính sách? [@mem2026; @memoryvla2026; @liberomem2025; @robomme2026]

Khoảng trống thứ nhất liên quan đến cách biểu diễn tiến độ. Một chính sách có thể học tương quan giữa thời gian đã trôi qua và số thao tác thường hoàn thành trong quỹ đạo trình diễn. Khi robot chờ hoặc thao tác chậm, tương quan này có thể sai lệch. Zhang và An ghi nhận hiện tượng tiến độ ảo trong một số thí nghiệm đếm của RoboMME và khảo sát cách đánh chỉ mục bộ nhớ theo sự kiện. Kết quả của họ củng cố động cơ xem xét bằng chứng thay đổi trạng thái của vật thể. Câu hỏi hẹp của khóa luận là cách biểu diễn trực tiếp mốc rời và trở lại đĩa, ghi rõ thời điểm có thể xác nhận và biểu thị phần lịch sử chưa biết khi theo dõi gián đoạn. Thiết kế này cho phép kiểm tra nội dung nào đã hiện diện trong đầu vào ngay trước một quyết định của SmolVLA. [@clockvla2026; @liberomem2025]

Khoảng trống thứ hai liên quan đến cách đánh giá cơ chế. Sự kiện có thể được nhận diện đúng trên trình diễn nhưng bị bỏ sót khi chính sách tự tạo quỹ đạo có che khuất hoặc thao tác dở dang. Một chính sách cũng có thể giảm sai số dự đoán hành động trên dữ liệu cố định mà vẫn gặp lỗi tích lũy khi điều khiển vòng kín. Nếu bộ nhớ và trọng số chính sách thay đổi đồng thời, chênh lệch thành công còn bao gồm tác động của thích ứng. Vì vậy, nghiên cứu cần đối chiếu độ phù hợp của bản ghi sự kiện, độ nhạy của mô hình với lịch sử và kết quả điều khiển bằng các phép đo tương ứng. Sự phân biệt này đặc biệt quan trọng khi đánh giá một biểu diễn gọn trên một mô hình nền đã huấn luyện. [@mem2026; @memoryvla2026; @robomme2026]

## 1.3. Mục tiêu và câu hỏi nghiên cứu

Mục tiêu của khóa luận là khảo sát một biểu diễn bộ nhớ sự kiện thị giác cho tiến độ thao tác lặp và xác định điều kiện để đo giá trị của biểu diễn đó đối với SmolVLA. Gọi $o_t$ là quan sát mà chính sách được phép nhận tại bước $t$, $N\in\{1,3\}$ là số chu kỳ theo chỉ dẫn và $\hat z_t$ là số chu kỳ do bộ ghi xác nhận. Tiến độ do bộ chấm độc lập xác định được ký hiệu riêng với $\hat z_t$. Sự phân biệt này cho phép xem độ phù hợp của bộ nhớ là một đại lượng cần đo thay vì một giả định của phép đánh giá.

**RQ1. Biểu diễn và độ phù hợp của bằng chứng.** Có thể xây dựng từ quan sát khả dụng một bộ nhớ sự kiện biểu diễn tiến độ nhấc và đặt lại bát cho SmolVLA như thế nào? Các sự kiện ghi được phù hợp với mốc tiến độ chú giải ở mức nào trên những quỹ đạo khảo sát?

**RQ2. Giá trị đối với điều khiển.** Khi giữ tương đương dữ liệu và ngân sách thích ứng, thông tin lịch sử có cải thiện tỷ lệ hoàn thành tác vụ lặp trong điều khiển vòng kín hay không? T3 là tác vụ trọng tâm vì quyết định tiếp tục hay dừng phụ thuộc vào tiến độ qua nhiều chu kỳ.

**RQ3. Duy trì năng lực cơ sở.** Sau khi thích ứng với dữ liệu tác vụ phụ thuộc lịch sử, chính sách còn duy trì khả năng hoàn thành các tác vụ LIBERO cơ sở đến mức nào dưới cùng giao thức đánh giá?

**RQ4. Điều kiện hiệu lực của kết quả.** Sự thiếu hụt hoặc chậm trễ của bằng chứng sự kiện, khác biệt giữa trình diễn và quỹ đạo do chính sách sinh ra, cùng các điều kiện thực thi ảnh hưởng thế nào đến diễn giải kết quả?

Các câu hỏi xác định những đơn vị bằng chứng khác nhau. RQ1 cần đối chiếu bản ghi với mốc chú giải. RQ2 và RQ3 cần đo hành vi vòng kín với điều kiện đối chứng tương ứng. RQ4 cần phân tích những trường hợp bộ ghi và bộ chấm bất đồng, cùng các yếu tố có thể giải thích bất đồng. Chương 4 trình bày phần bằng chứng đã thu được cho từng câu hỏi và chỉ ra những phép kiểm còn thiếu để nhận diện tác động của lịch sử.

## 1.4. Phạm vi và đóng góp

Phạm vi thực nghiệm giới hạn ở quan hệ giữa một chiếc bát và đĩa gốc trong mô phỏng. T1 yêu cầu một chu kỳ và T3 yêu cầu ba chu kỳ. Giao thức nghiên cứu sử dụng nguồn LIBERO và LIBERO-Mem đã khóa phiên bản, với thiết lập thực thi và tiêu chí kết thúc nêu trong Chương 3. Các điểm báo cáo thuộc hai tác vụ theo giao thức riêng của khóa luận và được đánh dấu là không so sánh trực tiếp với điểm công bố của benchmark. Thử nghiệm trên nhiều thực thể, chuyển giao sang mô hình nền khác và đánh giá trên robot thật nằm ngoài phạm vi bằng chứng hiện tại. [@liberoSource; @liberomemSource]

**Đóng góp thứ nhất là phát biểu bài toán tiến độ dưới quan sát một phần.** Khóa luận sử dụng thao tác lặp để chỉ ra một trường hợp mà cùng dạng quan sát có thể tương ứng với yêu cầu tiếp tục hoặc dừng. Từ đó, nghiên cứu xác định nội dung lịch sử cần khảo sát là các chu kỳ có bằng chứng, phân biệt số chu kỳ bộ nhớ xác nhận với số chu kỳ dùng để chấm điểm. Phát biểu này tạo cơ sở để phân tích tính phù hợp của biểu diễn, mức độ chính sách sử dụng thông tin và khả năng duy trì kỹ năng sau thích ứng.

**Đóng góp thứ hai là biểu diễn bộ nhớ sự kiện có bằng chứng và thời điểm khả dụng.** Phương pháp gắn các chuyển tiếp rời và trở lại với quan hệ bát–đĩa gốc, ghi nhận thời điểm có thể xác nhận và biểu thị khi lịch sử chưa được quan sát đầy đủ. Khóa luận phát biểu và chứng minh hai tính chất trong phạm vi mô hình máy trạng thái và giả thiết về nguồn dữ liệu: quy tắc đếm theo cặp sự kiện và tính không nhìn trước của nội dung được cấp cho chính sách. SmolVLA, bộ phát hiện thị giác và LoRA được kế thừa để khảo sát biểu diễn này trong một mô hình điều khiển. [@smolvla2025; @groundingdino2024; @lora2021]

**Đóng góp thứ ba là khảo sát thực nghiệm từ bản ghi đến hành vi vòng kín.** Khóa luận đối chiếu các sự kiện được xác nhận với mốc chú giải trên quỹ đạo trình diễn, đo độ nhạy của sai số dự đoán đối với đầu vào lịch sử, và phân tích kết quả vòng kín trên cùng tập trạng thái khởi tạo. Một bất đồng quan sát được là lượt duy nhất được chấm thành công vẫn có bản ghi lịch sử chưa đầy đủ. Trường hợp này cho thấy việc chấm đúng tiến độ và việc chính sách nhận được một mô tả đầy đủ về tiến độ là hai vấn đề thực nghiệm cần kiểm tra. Các kết quả hiện tại cho phép mô tả tính khả thi của phương pháp và xác định điều kiện cho phép kiểm định giả thuyết cải thiện, với tác động riêng của bộ nhớ còn là câu hỏi mở. [@artifact2026]

## 1.5. Bố cục khóa luận

Chương 2 trình bày cơ sở về quyết định dưới quan sát một phần và đối chiếu các cách biểu diễn lịch sử trong VLA. Chương 3 xây dựng bộ nhớ sự kiện, phát biểu các tính chất và thiết kế phép đối chứng. Chương 4 phân tích độ phù hợp của sự kiện, kết quả ngoại tuyến và hành vi vòng kín theo các câu hỏi nghiên cứu. Chương 5 tổng hợp các kết luận trong phạm vi bằng chứng và xác định các thí nghiệm cần thiết để kiểm định giả thuyết còn mở.

# Chương 2. Cơ sở lý thuyết và nghiên cứu liên quan

## 2.1. Quyết định phụ thuộc lịch sử dưới quan sát một phần

Một tác vụ có thể kéo dài qua nhiều bước nhưng vẫn cho phép suy ra tiến độ từ cấu hình hiện tại của cảnh. Ngược lại, một tác vụ lặp tương đối ngắn có thể đòi hỏi bộ nhớ nếu vật thể quay về vị trí cũ sau mỗi chu kỳ. Khi đó, quyết định có nên bắt đầu chu kỳ tiếp theo phụ thuộc vào những lần tương tác đã hoàn thành. Vì thế, độ dài quỹ đạo và mức phụ thuộc lịch sử là hai đặc trưng cần phân biệt khi đánh giá mô hình thị giác–ngôn ngữ–hành động.

Gọi $x_t$ là trạng thái đầy đủ, $o_t$ là quan sát mà chính sách nhận được và $a_t$ là hành động tại bước $t$. Với tác vụ lặp, trạng thái dùng để mô hình hóa mục tiêu bao gồm cả trạng thái vật lý và số chu kỳ đã hoàn thành $z_t$. Cảm biến có thể cung cấp thông tin về vị trí vật thể và tư thế robot, trong khi $z_t$ là thuộc tính của diễn biến trước đó. Mô hình quá trình quyết định Markov quan sát một phần (POMDP) cho phép mô tả sự khác biệt này. Ký hiệu $h_t$ cho lịch sử và $b_t$ cho phân phối niềm tin ứng với chỉ dẫn $\ell$ [@pomdp1998].

$$
\begin{aligned}
h_t &= (o_0,a_0,\ldots,a_{t-1},o_t),\\
b_t(B) &= \Pr(x_t\in B\mid h_t,\ell),\qquad B\subseteq\mathcal X,
\end{aligned}
$$

trong đó $B$ là một tập đo được của không gian trạng thái $\mathcal X$. Trong một POMDP với mô hình chuyển trạng thái và quan sát đã biết, phân phối niềm tin là thống kê đủ của lịch sử cho bài toán ra quyết định với mô hình đó. Khóa luận khảo sát một đại diện hẹp hơn, $m_t=f(h_t)$, nhằm giữ lại bằng chứng về các chu kỳ tương tác đã quan sát được. Biểu diễn này chỉ được thiết kế để hỗ trợ phân biệt tiến độ trong một họ tác vụ cụ thể. Việc tính được $m_t$ chưa thiết lập rằng nó là thống kê đủ cho điều khiển robot.

Xét hai lịch sử có cùng chỉ dẫn và cùng quan sát tức thời, nhưng có hai tập hành động thích hợp rời nhau. Một chính sách chỉ điều kiện hóa theo $(o_t,\ell)$ sẽ sinh cùng một phân phối hành động trong cả hai trường hợp. Phân phối này không thể đồng thời gán xác suất một cho hai tập rời nhau. Ví dụ, khi bát đã được đặt lại hai lần, mục tiêu ba chu kỳ vẫn yêu cầu một lần nhấc nữa. Sau lần đặt lại thứ ba, việc khởi phát thêm chu kỳ có thể khiến tác vụ thất bại. Đây là lập luận về sự nhập nhằng của quan sát trong trường hợp lý tưởng. Trên dữ liệu thực nghiệm, ảnh, trạng thái robot hoặc nhịp thao tác có thể còn mang dấu vết của tiến độ. Mức chính sách dựa vào các dấu vết ấy là một vấn đề thực nghiệm.

## 2.2. Biểu diễn lịch sử trong các mô hình VLA

Các phương pháp bộ nhớ cho VLA khác nhau ở thông tin được giữ và cách thông tin ấy điều kiện hóa hành động. MEM của Torne và cộng sự phân biệt hai thang thời gian. Bộ mã hóa video giữ ngữ cảnh thị giác gần để hỗ trợ nhận biết diễn biến, còn phần tóm tắt bằng ngôn ngữ giữ thông tin ngữ nghĩa qua các giai đoạn dài hơn của tác vụ. Nhận ra chuyển động đang diễn ra và nhớ một việc đã hoàn thành vì thế có thể cần những khoảng lịch sử khác nhau [@mem2026].

MemoryVLA của Shi và cộng sự lưu các đặc trưng tri giác và nhận thức trong một ngân hàng bộ nhớ. Tại thời điểm quyết định, mô hình truy xuất, hợp nhất đặc trưng lịch sử với đặc trưng hiện tại rồi điều kiện hóa mô-đun sinh hành động. Các thí nghiệm loại bỏ thành phần trong công trình này cho thấy kết quả phụ thuộc vào cách truy xuất, hợp nhất và dung lượng bộ nhớ trong từng thiết lập khảo sát. Những lựa chọn về cách tổ chức lịch sử vì vậy cần được đánh giá cùng yêu cầu của tác vụ [@memoryvla2026].

Chung và cộng sự chuyển trọng tâm sang lịch sử của thực thể trong LIBERO-Mem. Bộ tác vụ làm lộ các tình huống mà quyết định phụ thuộc vào chuyển động, thứ tự, quan hệ hoặc sự che khuất của vật thể. Embodied-SlotSSM biểu diễn diễn biến của các thực thể bằng các trạng thái tiềm ẩn gắn với từng đối tượng. RoboMME mở rộng khảo sát có đối chứng về kiến trúc bộ nhớ bằng 16 tác vụ và 14 biến thể trên cùng mô hình nền $\pi_{0.5}$. Dai và cộng sự nhận thấy tính hữu ích của một biểu diễn phụ thuộc vào loại tác vụ. Kết quả này tạo cơ sở để lựa chọn nội dung bộ nhớ theo biến tiến độ cần suy ra, đồng thời giới hạn việc suy rộng kết quả giữa các họ tác vụ [@liberomem2025; @robomme2026].

| Phương pháp | Nội dung lịch sử | Cách tác động đến quyết định | Câu hỏi liên quan đến khóa luận |
|---|---|---|---|
| MEM | Video ngắn và tóm tắt ngôn ngữ | Điều kiện hóa theo hai thang thời gian | Thông tin nào cần giữ qua nhiều giai đoạn? |
| MemoryVLA | Đặc trưng tri giác và nhận thức | Truy xuất và hợp nhất với ngữ cảnh hiện tại | Chính sách sử dụng biểu diễn lưu trữ như thế nào? |
| Embodied-SlotSSM | Diễn biến theo từng thực thể | Sinh hành động từ trạng thái tiềm ẩn của đối tượng | Quan hệ nào của vật thể quyết định tiến độ? |
| Phương pháp của khóa luận | Sự kiện rời và trở về đĩa gốc | Đưa tiến độ đã xác nhận vào chỉ dẫn bằng văn bản | Sự kiện nào có đủ bằng chứng để dùng tại lúc ra quyết định? |

Các biểu diễn video và đặc trưng giữ được nhiều tín hiệu hữu ích cho điều khiển. Khi câu hỏi nghiên cứu tập trung vào một lần tương tác cụ thể, bản ghi sự kiện tường minh cho phép kiểm tra quan sát nào đã được dùng để xác nhận sự kiện và từ lúc nào chính sách mới có thể sử dụng thông tin này. Cách nén đó dựa trên giả định rằng sự kiện được định nghĩa phù hợp với mục tiêu tác vụ. Nếu chọn sai sự kiện hoặc làm mất bằng chứng quan trọng, bản ghi ngắn vẫn có thể dẫn đến quyết định sai.

## 2.3. Khoảng trống nghiên cứu trong tác vụ lặp

Việc một chính sách có bộ nhớ hoàn thành tác vụ lặp chưa cho biết nó đã suy ra tiến độ từ sự kiện hay từ tương quan sẵn có trong dữ liệu. Nếu các quỹ đạo trình diễn có nhịp gần như cố định, thời gian trôi qua có thể dự báo số chu kỳ dù chính sách chưa nhận diện được một chu kỳ đã hoàn tất. Zhang và An kiểm tra trực tiếp cơ chế này trong bản tiền ấn phẩm trên các tác vụ đếm của RoboMME. Khi thêm khoảng dừng hoặc chuyển động không làm thay đổi tiến độ nhưng vẫn đưa các quan sát đó vào bộ nhớ, chính sách được khảo sát có thể dừng sớm vì hành xử như thể tiến độ đã tăng. Tổ chức chỉ mục bộ nhớ theo sự kiện khôi phục một phần hiệu năng trong các điều kiện được thử nghiệm. Công trình cũng chỉ ra rằng phép gộp các đoạn quay về cấu hình cũ có thể vô tình loại cả một lần lặp hợp lệ [@clockvla2026].

Kết quả ấy làm rõ câu hỏi về ngữ nghĩa của một sự kiện. Khi bát trở về đĩa, hai quan sát trước và sau chu kỳ có thể gần nhau, nhưng phần diễn biến giữa chúng quyết định liệu đó là một lần lặp cần ghi nhận. Một khoảng chờ trên đĩa không mang cùng ý nghĩa với chuỗi rời đĩa rồi quay lại. Khóa luận khảo sát bộ nhớ chuyên biệt giữ lại chuỗi bằng chứng này trên SmolVLA. Mỗi lần hoàn thành được xác nhận từ diễn biến của cùng một cặp bát–đĩa. Trạng thái chưa biết được bảo lưu nếu quan sát bị gián đoạn. Thời điểm xảy ra ước lượng và thời điểm đủ bằng chứng để xác nhận được ghi riêng để tránh đưa thông tin có được sau này vào quyết định trước đó.

Khoảng trống thứ nhất trong phạm vi khảo sát là **khả năng xác lập tiến độ sự kiện từ quan sát trực tuyến có thể kiểm tra lại**. Điều này đòi hỏi bằng chứng cho cả nhận dạng thực thể lẫn chuyển tiếp quan hệ với vật đỡ. Grounding DINO cung cấp hộp giới hạn từ ảnh và mô tả ngôn ngữ [@groundingdino2024]. Mỗi hộp cho một ước lượng vị trí đối tượng trên ảnh tại một thời điểm. Diễn giải việc nhấc hoặc đặt lại cần liên kết đối tượng qua thời gian và áp dụng một quy tắc xác nhận. Chương 3 cụ thể hóa yêu cầu đó bằng máy trạng thái và điều kiện về thời điểm sử dụng sự kiện. Độ đúng, độ trễ và mức thiếu quan sát là những thuộc tính thực nghiệm riêng của bản ghi.

Khoảng trống thứ hai là **mức đóng góp của nội dung lịch sử đối với quyết định sau khi đã kiểm soát thích ứng chính sách**. Một biểu diễn có thể ghi đúng tiến độ mà chính sách vẫn bỏ qua, hoặc chính sách có thể được lợi từ dữ liệu tinh chỉnh ngay cả khi thiếu bản ghi lịch sử. Đối chiếu sự kiện với mốc tham chiếu giúp đo chất lượng bản ghi. Đánh giá hành vi khi thay đổi thông tin lịch sử dưới các điều kiện đối chứng phù hợp giúp xác định giá trị sử dụng của bản ghi. Các can thiệp vào nhịp thực hiện để kiểm tra lối tắt theo thời gian là một phép thử tiếp theo đối với phương pháp của khóa luận.

## 2.4. SmolVLA và vai trò của thích ứng

SmolVLA kết hợp mô hình thị giác–ngôn ngữ với mô-đun sinh hành động (action expert). Biểu diễn từ ảnh, trạng thái robot và chỉ dẫn điều kiện hóa việc dự đoán một đoạn hành động liên tục. Mỗi lần dự đoán có thể sinh $H$ hành động, trong khi chính sách nhận quan sát mới sau khi thực thi $s$ hành động đầu. Cơ chế này tạo điều kiện cho điều khiển liên tục với phản hồi định kỳ. Lịch sử các lần tương tác đã xác nhận là biến bổ sung, khác với số bước dự đoán hoặc thực thi giữa hai lần nhận quan sát. Khóa luận chọn SmolVLA vì có thể khảo sát việc truyền biến bổ sung này qua đầu vào ngôn ngữ của một chính sách đã học thao tác. Thực nghiệm sử dụng checkpoint SmolVLA cho LIBERO đã khóa phiên bản [@smolvla2025; @smolvlaCheckpoint].

Việc bổ sung bản ghi vào chỉ dẫn làm thay đổi phân phối đầu vào của mô hình đã huấn luyện. Khóa luận sử dụng thích ứng hạng thấp LoRA để khảo sát khả năng điều kiện hóa hành động theo biểu diễn mới, đồng thời giữ cố định trọng số nền [@lora2021]. Với một ma trận nền $W_0$, trọng số hiệu dụng có dạng $W=W_0+(\alpha/r)BA$, trong đó $A,B$ là hai ma trận được học, $r$ là hạng của phép cập nhật và $\alpha$ là hệ số điều chỉnh. Cập nhật hạng thấp tạo một cách thích ứng với phạm vi tham số xác định. Tác động của cập nhật vẫn cần được kiểm soát khi ước lượng tác động của thông tin lịch sử.

Mục tiêu huấn luyện của SmolVLA kế thừa flow matching [@flowmatching2023; @smolvla2025]. Ở mức khái quát, hàm mục tiêu đo sai lệch giữa trường vector học được $v_{\theta,\phi}$ và trường mục tiêu $u_\tau$ dọc theo đường xác suất nối nhiễu với đoạn hành động tham chiếu.

$$
\mathcal L_{\mathrm{FM}}
= \mathbb E_{\tau,A,\epsilon,c}\!
\left[\left\|v_{\theta,\phi}(A_\tau,\tau,c)-u_\tau(A,\epsilon)\right\|_2^2\right].
$$

Trong biểu thức này, $\tau$ là tham số thời gian của phép nội suy, $A_\tau$ là mẫu trên đường xác suất và $c$ là điều kiện đầu vào. Sai số ngoại tuyến trên quỹ đạo đã ghi phản ánh chất lượng dự đoán theo phân phối quan sát trong dữ liệu. Trong điều khiển vòng kín, hành động của chính sách làm thay đổi quan sát tiếp theo, đồng thời ảnh hưởng đến bằng chứng mà bộ nhớ có thể thu nhận. Do đó, giảm sai số dự đoán và tăng tỷ lệ thành công là hai giả thuyết phải được kiểm tra bằng những phép đo tương ứng.

Thành phần dữ liệu cũng ảnh hưởng đến việc quy kết kết quả. Nghiên cứu *A Balanced Data Diet* cho thấy cách chọn kinh nghiệm trong học tăng cường có thể làm thay đổi năng lực thu được dưới một ngân sách huấn luyện [@balanceddiet2026]. Đối với bài toán tinh chỉnh có giám sát của khóa luận, hệ quả về thiết kế thí nghiệm là cần khớp dữ liệu và ngân sách giữa các cấu hình so sánh. Việc phối hợp dữ liệu thao tác cơ sở và dữ liệu tác vụ phụ thuộc lịch sử tạo điều kiện khảo sát sự đánh đổi giữa thích ứng và duy trì kỹ năng. Khả năng duy trì kỹ năng cần được đo trực tiếp trên nhóm tác vụ cơ sở.

## 2.5. Tác vụ nghiên cứu và ý nghĩa của các phép đo

LIBERO cung cấp tập tác vụ mô phỏng phục vụ nghiên cứu học chuyển giao ở robot [@libero2023]. LIBERO-Mem đưa thêm các yêu cầu tiến độ và thứ tự gắn với tương tác của vật thể [@liberomem2025]. Trong khóa luận, hai biến thể T1 và T3 giữ cùng dạng thao tác nhấc–đặt lại bát và thay đổi số chu kỳ yêu cầu từ một lên ba. Cặp tác vụ cho phép quan sát cả năng lực thực hiện thao tác cơ sở lẫn hành vi khởi phát, lặp và kết thúc thao tác. T3 cũng kéo dài quỹ đạo và tạo thêm cơ hội tích lũy sai số điều khiển. Vì vậy, chênh lệch T1–T3 tạo tín hiệu về độ khó của tác vụ, còn tác động của lịch sử phải được ước lượng bằng so sánh trong cùng tác vụ.

Ở mức hành vi, tỷ lệ thành công (SR) đo phần các lượt chạy đạt điều kiện hoàn tất. Tỷ lệ hoàn thành mốc (SCR) bổ sung thông tin về tiến độ trước khi thất bại. Việc thực hiện dư chu kỳ phản ánh một dạng sai lệch khác với thất bại khi nhấc hoặc đặt vật. Để diễn giải các chỉ số này, cần đối chiếu tiến độ do bộ đánh giá ghi nhận với tiến độ mà bộ nhớ quan sát được và với chuỗi hành động thực thi. Trên cùng một trạng thái khởi tạo, các cấu hình chính sách được so sánh theo cặp. Khi hai tác vụ chia sẻ trạng thái khởi tạo, phân tích độ bất định cần xét sự phụ thuộc theo nhóm trạng thái. Nhiều khung hình trong cùng một lượt chạy là các quan sát của cùng một quỹ đạo.

Bộ nhớ sự kiện tường minh cho SmolVLA cho phép quan sát những bất đồng giữa tiến độ của tác vụ, tiến độ đã ghi và hành vi của chính sách. Một sự kiện có thể được ghi nhận chậm, bị bỏ lỡ hoặc được ghi đúng nhưng không dẫn đến quyết định phù hợp. Việc phân biệt các khả năng này định hướng thiết kế biểu diễn ở Chương 3 và phân tích bằng chứng thực nghiệm ở Chương 4.

# Chương 3. Bộ nhớ sự kiện thị giác cho SmolVLA

## 3.1. Tiến độ tác vụ và giả thuyết biểu diễn

Tác vụ T1 và T3 yêu cầu robot nhấc bát và đặt lại trên đĩa ban đầu lần lượt một và ba lần. Gọi $N\in\{1,3\}$ là số chu kỳ được yêu cầu và $z_t$ là số chu kỳ đã thực sự hoàn tất đến bước $t$ theo định nghĩa tác vụ. Một cấu hình nhìn thấy bát ở trên đĩa có thể xuất hiện ở nhiều giá trị $z_t$. Trong trường hợp hai lịch sử dẫn đến cùng quan sát tại lúc bát nằm trên đĩa, một chính sách chỉ điều kiện hóa trên quan sát đó và cùng chỉ dẫn sẽ sinh cùng phân phối hành động. Tuy nhiên, vai trò của một chu kỳ tiếp theo thay đổi theo tiến độ. Khi $z_t<N$, chu kỳ tiếp theo có thể đưa tác vụ đến đích. Khi $z_t=N$, một chu kỳ mới có thể gây vượt mục tiêu. Ví dụ này giải thích vì sao số chu kỳ là một biến lịch sử có ý nghĩa cho thiết kế. Tần suất các quan sát tương đương trong các lượt chạy chưa được định lượng.

Nghiên cứu đề xuất biểu diễn lịch sử bằng các sự kiện thay đổi quan hệ giữa bát đích và đĩa gốc. Số chu kỳ được bộ ghi xác nhận, ký hiệu $\hat z_t$, là một đại lượng ước lượng từ quan sát. Biểu diễn còn giữ danh tính cặp vật, tình trạng chu kỳ đang diễn ra và mức đầy đủ của lịch sử. Điều này cho phép mô tả hai trường hợp có cùng $\hat z_t$ nhưng khác nhau về mức độ tin cậy của bằng chứng. Giả thuyết nghiên cứu là biểu diễn sự kiện có thể cung cấp tín hiệu tiến độ hữu ích cho chính sách khi quan sát hiện tại không phân giải được lịch sử. Khả năng ghi nhận sự kiện và khả năng chính sách sử dụng sự kiện là hai vấn đề thực nghiệm riêng.

Tại bước $t$, đặt $o_t=(I_t^{\mathrm{ext}},I_t^{\mathrm{wrist}},p_t)$, với hai ảnh RGB và trạng thái robot $p_t\in\mathbb{R}^8$. Gọi $\ell$ là chỉ dẫn và $a_{t-1}$ là hành động gần nhất đã thực thi. Bộ nhớ $m_t$ cập nhật ở từng bước quan sát, còn chính sách sinh đoạn hành động $A_t$ tại các bước truy vấn.

$$
\begin{aligned}
m_t &= U(m_{t-1},o_t,a_{t-1}),\\
c_t &= R(\ell,m_t),\qquad
A_t\sim \pi_{\theta,\phi}(\cdot\mid o_t,c_t).
\end{aligned}
$$

Trong biểu thức này, $U$ là quy tắc ghi sự kiện, $R$ là phép biểu diễn bộ nhớ thành văn bản, $\theta$ là trọng số SmolVLA gốc và $\phi$ là các tham số LoRA được học. Bước khởi tạo chỉ có quan sát đầu tiên. Ở mỗi bước sau, bộ ghi nhận quan sát thu được sau hành động đã thực thi. Hành động còn nằm trong đoạn dự đoán chưa thuộc lịch sử quan sát.

## 3.2. Ranh giới giữa quan sát và đánh giá

Một yêu cầu để kiểm tra giả thuyết biểu diễn là giữ độc lập nguồn thông tin của chính sách với nguồn thông tin dùng để chấm điểm. Bộ ghi chỉ sử dụng ảnh và tín hiệu trạng thái robot có tại thời điểm quan sát để xác nhận sự kiện. Bộ chấm đọc vị từ và tiến độ từ môi trường mô phỏng, dùng chúng để xác định kết quả của quỹ đạo. Nhánh chấm điểm đứng ngoài đường thông tin đi vào chính sách như minh họa ở Hình 3.1. Vì vậy, chênh lệch giữa $\hat z_t$ và tiến độ do môi trường ghi nhận có thể được dùng để phân tích sai số của phép đo.

![Đường thông tin của phương pháp. Bộ nhớ được suy từ quan sát đã nhận, còn tiến độ do môi trường ghi nhận được dành cho đánh giá.](figures/architecture.png)

Sự phân tách này đặc biệt quan trọng khi sinh mẫu huấn luyện từ quỹ đạo ghi sẵn. Một quỹ đạo đầy đủ có cả trạng thái sau thời điểm tạo mẫu và kết quả cuối lượt. Khi xây dựng ngữ cảnh cho một mẫu tại bước $t$, bộ ghi được cập nhật theo thứ tự thời gian và chỉ dựa trên tiền tố quan sát đến $t$. Thời điểm đủ bằng chứng cho từng sự kiện được lưu riêng để kiểm tra quy tắc này.

## 3.3. Ước lượng quan hệ bát và đĩa gốc

Giả định đo lường là trong bố trí của T1 và T3, một số thay đổi quan hệ bát và đĩa có thể được phát hiện từ chuyển động tương đối của các hộp giới hạn trong ảnh camera ngoài. Camera cổ tay cung cấp bằng chứng bổ trợ về khả năng quan sát. Đĩa gốc được neo bằng quan sát ở đầu lượt khi quan hệ bát trên đĩa đủ rõ và tay kẹp đã mở. Các quan sát sau phải ghép được với đĩa đã neo. Việc cố định tham chiếu theo thời gian giúp phép đo bám vào ngữ nghĩa “đặt về đĩa ban đầu”, kể cả khi bát xuất hiện gần đĩa khác.

Gọi $\mathbf b_t$ và $\mathbf q_t$ là tâm mép dưới hộp giới hạn của bát và đĩa ở bước $t$. Gọi $\mathbf b_0,\mathbf q_0$ là tọa độ tại lúc xác lập mốc tham chiếu, còn $w_b,h_b$ là chiều rộng và chiều cao hộp của bát tại thời điểm đó. Bộ theo dõi dùng ba đại lượng.

$$
\begin{aligned}
d_t &= (\mathbf b_t-\mathbf q_t)-(\mathbf b_0-\mathbf q_0),\\
D_t &= \left\|d_t\oslash(w_b,h_b)\right\|_2,\qquad
L_t = -d_{t,y}/h_b .
\end{aligned}
$$

Ký hiệu $\oslash$ là phép chia theo từng thành phần. $D_t$ đo độ lệch khỏi quan hệ hình học lúc khởi đầu, đã chuẩn hóa theo kích thước biểu kiến của bát. $L_t$ đo thành phần dịch chuyển hướng lên trên mặt ảnh. Bộ theo dõi kết hợp $D_t$, $L_t$, tỷ lệ kích thước và độ ổn định theo thời gian. Độ mở đo từ hai khớp ngón kẹp bổ sung căn cứ cho việc xác nhận trạng thái nghỉ. Các ngưỡng trong cấu hình nghiên cứu được trình bày ở Bảng 3.1.

| Đại lượng | Quy tắc trong cấu hình | Vai trò trong phép đo |
|---|---|---|
| Điểm phát hiện | $s_{\mathrm{det}}\ge 0{,}15$ | Lọc các hộp đề xuất theo điểm của bộ phát hiện |
| Tín hiệu nâng | $L_t\ge 0{,}40$ và nằm ngoài vùng gần đĩa gốc | Tạo ứng viên rời đĩa |
| Gần đĩa gốc | $D_t\le 0{,}30$. Tỷ lệ rộng từ $0{,}6$ đến $1{,}5$, cao từ $0{,}3$ đến $1{,}5$ | Tạo ứng viên quay về theo hình học ảnh |
| Ổn định | Độ dịch chuẩn hóa của tâm mép dưới $\le 0{,}15$ | Xác định bát ổn định so với quan sát gần nhất |
| Tay kẹp mở | Tổng trị tuyệt đối vị trí hai khớp ngón $\ge 0{,}055$ m | Bổ sung căn cứ để xác nhận bát đã được thả |
| Xác nhận theo thời gian | Ba quan sát phù hợp liên tiếp | Giảm xác nhận do nhiễu ở một khung ảnh |
| Ghép đĩa gốc | Độ dịch chuẩn hóa theo kích thước hộp đĩa neo $\le 0{,}45$ | Duy trì cùng đối tượng tham chiếu |
| Thu nhận mốc đĩa | Trong 20 bước đầu | Giới hạn thời điểm xác lập quan hệ ban đầu |

Các phép đo trên mặt ảnh là chỉ dấu gián tiếp của quan hệ vật lý trong không gian ba chiều. Điểm phát hiện được dùng như tiêu chí chọn hộp, với ý nghĩa xác suất chưa được hiệu chuẩn. Trong trường hợp có nhiều hộp đề xuất, quan sát gần nhất có thể hỗ trợ ghép danh tính trong một khoảng thời gian giới hạn. Việc xác nhận ở bước hiện tại vẫn đòi hỏi quan sát phù hợp tại chính bước đó. Khi không định danh được bát hoặc không ghép được đĩa gốc, quan hệ ở bước hiện tại được ghi là chưa xác định. Các đặc điểm này giới hạn phạm vi áp dụng ở những cảnh có bố trí và góc nhìn tương tự.

## 3.4. Ghi sự kiện và bất biến đếm chu kỳ

Bộ ghi biểu diễn mỗi chu kỳ bằng một sự kiện rời đĩa và một sự kiện quay về theo đúng thứ tự. Quan hệ ban đầu được xác nhận khi bát ở gần đĩa gốc, ổn định và tay kẹp mở qua đủ số quan sát. Sau đó, tín hiệu nâng kéo dài tạo một sự kiện rời đĩa. Khi đã xác nhận sự kiện này, bộ ghi mới xét chuỗi quan sát bát về gần đĩa gốc, ổn định và tay kẹp mở để xác nhận sự kiện quay về. Mỗi lần xác nhận quay về kết thúc chu kỳ đang mở và tăng $\hat z_t$ một đơn vị.

Bộ ghi dùng trạng thái ứng viên để tích lũy bằng chứng. Khi bước quan sát bị bỏ qua hoặc không đủ căn cứ thị giác, ứng viên chưa xác nhận được xóa và cờ lịch sử chưa đầy đủ được đặt. Các sự kiện đã xác nhận được giữ. Nếu một chu kỳ đã mở trước khoảng trống, nó vẫn mở cho đến khi bộ ghi có bằng chứng phù hợp để xác nhận quay về. Số khoảng trống phản ánh mức độ gián đoạn của quan sát. Khi một khoảng trống có thể chứa sự kiện chưa được ghi nhận, từ $\hat z_t$ chưa thể khôi phục chính xác $z_t$. Nếu mọi lần quay về được xác nhận đều tương ứng đúng với các sự kiện thật phân biệt, $\hat z_t$ có thể được diễn giải như một cận dưới của số chu kỳ. Điều kiện đó cần được kiểm tra bằng dữ liệu có nhãn phù hợp.

**Mệnh đề 1 (bất biến đếm chu kỳ).** Trong một lượt thực thi với bộ ghi khởi tạo mới và các chuyển trạng thái hoàn tất theo quy tắc trên, mỗi lần tăng $\hat z_t$ đóng đúng một sự kiện rời đĩa đã xác nhận và chưa ghép cặp. Giữa hai lần tăng liên tiếp phải có một sự kiện rời đĩa mới.

**Chứng minh.** Gọi $J_t$ là số sự kiện rời đĩa đã xác nhận và $\chi_t\in\{0,1\}$ là cờ cho biết có một chu kỳ đang mở. Ở trạng thái khởi tạo, cả ba biến đều bằng không. Xét đẳng thức bất biến

$$
J_t-\hat z_t=\chi_t.
$$

Chuyển xác nhận rời đĩa chỉ xảy ra khi $\chi_t=0$ trước chuyển và làm tăng $J_t$ một đơn vị đồng thời đặt $\chi_t=1$. Chuyển xác nhận quay về chỉ xảy ra khi $\chi_t=1$ trước chuyển và làm tăng $\hat z_t$ một đơn vị đồng thời đặt $\chi_t=0$. Các chuyển còn lại giữ nguyên $J_t$, $\hat z_t$ và $\chi_t$. Do đó đẳng thức đúng sau mọi chuyển theo quy nạp. Vì mỗi lần tăng $\hat z_t$ cần một chu kỳ đang mở và đóng chu kỳ đó, cùng một sự kiện rời đĩa chỉ ghép được với một lần tăng. Bất biến này ràng buộc cấu trúc của dãy sự kiện do bộ ghi tạo ra. Độ đúng của từng sự kiện so với thao tác vật lý còn phụ thuộc phép đo ở Mục 3.3.

## 3.5. Tính không nhìn trước và biểu diễn văn bản

Một sự kiện $e_j$ có thời điểm bắt đầu ước lượng $\tau_j$ và thời điểm đủ bằng chứng $\upsilon_j$. Chẳng hạn, nếu bộ ghi dùng ba khung ảnh liên tiếp từ bước 10 để xác nhận quay về, nó có thể gán $\tau_j=10$ và $\upsilon_j=12$. Sự kiện chỉ được sử dụng từ bước 12 trở đi. Tại bước truy vấn $q$, điều kiện thời gian và nguồn bằng chứng được biểu diễn bằng

$$
\mathcal{E}_q =
\{e_j \mid \tau_j\le \upsilon_j\le q,\,
  \operatorname{source}(e_j)=\mathrm{RGB}\}.
$$

Nhãn nguồn $\mathrm{RGB}$ trong biểu thức chỉ các sự kiện do bộ theo dõi thị giác dự đoán và có căn cứ truy vết được. Bên cạnh thời điểm khả dụng, phép tạo văn bản kiểm tra sự nhất quán giữa danh tính bát, danh tính đĩa, số lần quay về và các sự kiện được tham chiếu. Dữ liệu lấy từ trạng thái đặc quyền của mô phỏng hoặc nhãn gán tay thuộc nguồn đánh giá. Phép tạo ngữ cảnh ở điều kiện F1 từ chối ảnh chụp bộ nhớ mang các nguồn này.

Chỉ dẫn gốc được giữ nguyên ở đầu ngữ cảnh. Phần bộ nhớ có thể mô tả số lần quay về đã xác nhận, chu kỳ đang mở hoặc lịch sử chưa đầy đủ. Ví dụ, ở bước truy vấn sau khi ghi nhận hai lần quay về trong T3, phần bổ sung có thể nêu rằng bát đã được quan sát quay lại đĩa gốc hai lần. Với lịch sử gián đoạn, cách diễn đạt phải thể hiện tính chưa đầy đủ của bằng chứng. Ngân sách là 48 token cho toàn bộ chỉ dẫn và phần bổ sung theo tokenizer của checkpoint, kể cả ký hiệu xuống dòng và token đặc biệt. Bộ dựng ngữ cảnh bỏ mệnh đề vượt ngân sách và ghi nhận lần bỏ đó. Chỉ dẫn tự vượt ngân sách làm dừng lượt chạy. Phạm vi của 48 token là đầu vào văn bản, còn số bằng chứng lưu trong suốt lượt có thể tăng theo độ dài quỹ đạo.

**Mệnh đề 2 (tính không nhìn trước của ngữ cảnh).** Cố định chỉ dẫn, cấu hình, định danh lượt và mọi nguồn ngẫu nhiên dùng để tạo bộ nhớ. Giả sử bộ ghi bắt đầu ở trạng thái mới, mỗi lần cập nhật chỉ nhận quan sát đến bước hiện tại và các hành động đã thực thi, đồng thời mỗi bằng chứng ảnh được ghép với đúng bước quan sát của nó. Giả sử phép tạo văn bản chỉ nhận trạng thái bộ nhớ hiện hành, kiểm tra $\upsilon_j\le q$ và từ chối các nguồn đặc quyền như trên. Khi đó, hai quỹ đạo có cùng tiền tố quan sát và hành động đã thực thi đến bước $q$ sẽ sinh cùng ngữ cảnh $c_q$, bất kể phần tiếp diễn của chúng.

**Chứng minh.** Cùng quan sát đầu và cùng cấu hình dẫn đến cùng trạng thái bộ nhớ ban đầu. Giả sử hai trạng thái bộ nhớ ở bước $t-1$ bằng nhau với $t\le q$. Hai tiền tố bằng nhau cho cùng $o_t$ và $a_{t-1}$, vì vậy quy tắc cập nhật $U$ cho cùng $m_t$. Quy nạp cho $m_q$ bằng nhau. Phép $R$ áp dụng cùng phép kiểm tra nguồn, thời điểm và ngân sách lên hai bản ghi giống nhau nên trả cùng $c_q$. Tính chất này có điều kiện trên việc cấp đúng bằng chứng quan sát theo thời gian. Nhãn nguồn và thời điểm được dùng để kiểm tra sự tuân thủ điều kiện đó. Độ đúng của nội dung sự kiện và mức độ chính sách sử dụng nội dung cần được xác định qua đánh giá thực nghiệm.

## 3.6. Dữ liệu và thích ứng chính sách

Dữ liệu thích ứng ghép các quỹ đạo mẫu LIBERO-10 với các quỹ đạo T1 và T3 từ bản phát hành của tác giả. Các quỹ đạo có cùng vector trạng thái khởi tạo được xếp vào cùng một nhóm trước khi chia tập. Định danh nhóm được tính từ vector đầy đủ ở biểu diễn float64 little-endian. Các nhóm từng dùng để phát triển bộ ghi được xếp vào tập huấn luyện. Cách chia này kiểm soát trường hợp cùng trạng thái khởi tạo xuất hiện dưới nhiều tên tác vụ hoặc mã quỹ đạo khác nhau. Lịch sử sử dụng tập kiểm tra trong phát triển và khả năng giao thoa với dữ liệu tiền huấn luyện vẫn ảnh hưởng đến diễn giải khả năng khái quát hóa.

Quy ước ảnh được xác định theo nguồn. Đối với hai camera, ảnh tinh chỉnh từ bản phát hành LIBERO-Mem được lật dọc một lần để trở về quy ước ảnh thô, còn ảnh kết xuất thô từ mô phỏng được giữ nguyên trước bộ tiền xử lý SmolVLA. Việc chuẩn hóa hướng ảnh giữ nguyên các phép biến đổi ảnh và quy tắc giải mã hành động của checkpoint. Adapter dùng trong Chương 4 được huấn luyện trên các quỹ đạo từ bản phát hành. Bộ dữ liệu phát lại theo cấu hình mô phỏng cục bộ còn cần hoàn tất kiểm chứng trước khi dùng để đánh giá một adapter mới.

Thích ứng được thực hiện bằng LoRA tại các phép chiếu query và value của attention trong mô-đun sinh hành động. Các trọng số nền được giữ cố định, cùng với bộ chuẩn hóa trạng thái và phép giải chuẩn hành động của checkpoint. Mỗi lô tối ưu hóa gồm dữ liệu thao tác cơ sở và dữ liệu T1/T3. Bộ lấy mẫu cân bằng theo tác vụ trong từng miền và chọn quỹ đạo trước khi lấy khung ảnh, nhằm hạn chế ảnh hưởng của chênh lệch độ dài giữa các quỹ đạo. Trộn dữ liệu cơ sở tạo điều kiện khảo sát khả năng duy trì kỹ năng sau thích ứng. Kết luận về khả năng duy trì đòi hỏi đánh giá chính adapter trên các tác vụ cơ sở dưới cùng quy trình.

## 3.7. Thiết kế đối chứng

Các đối chứng được xác định theo câu hỏi mà từng phép so sánh có thể trả lời. B0 là mốc tham chiếu với trọng số gốc. Để đo thay đổi do thích ứng, F0 cần dùng cùng tập quỹ đạo, cách chia, cấu hình, ngân sách huấn luyện và quy tắc chọn checkpoint như F1, với ngữ cảnh huấn luyện và suy luận chỉ chứa chỉ dẫn. F1 bổ sung biểu diễn sự kiện trong cả thích ứng lẫn suy luận. F1-off giữ nguyên trọng số F1 và bỏ phần sự kiện ở thời điểm suy luận.

| Điều kiện | Trọng số | Ngữ cảnh suy luận | Mục đích đánh giá | Bằng chứng hiện có |
|---|---|---|---|---|
| B0 | Checkpoint gốc | Chỉ dẫn | Mốc tham chiếu trước thích ứng | 20 lượt T1/T3 |
| F0 | Adapter huấn luyện đối sánh | Chỉ dẫn | Đối chứng cho thay đổi do thích ứng | Chưa có lượt chạy đối sánh |
| F1 | Adapter học với bộ nhớ | Chỉ dẫn và sự kiện | Đánh giá phương pháp đề xuất | 20 lượt T1/T3 |
| F1-off | Cùng adapter F1 | Chỉ dẫn | Đo độ nhạy với việc bỏ phần sự kiện | Chưa có lượt chạy vòng kín |

So sánh hiện có B0 với F1 đo chênh lệch khi thay đồng thời trọng số thích ứng và nguồn ngữ cảnh. So sánh F0 với F1 ở cùng ngân sách sẽ thu hẹp câu hỏi về giá trị của việc bổ sung nhánh bộ nhớ. Để nghiên cứu riêng giá trị của thông tin lịch sử, một đối chứng bổ sung có thể cung cấp tóm tắt quan sát hiện tại bằng cùng bộ phát hiện. Một đối chứng khác có thể giữ ngân sách token tương đương và dùng phần bổ sung không mang tín hiệu tiến độ. F1-off có ý nghĩa chẩn đoán độ nhạy của cùng trọng số với ngữ cảnh. Do phần bộ nhớ chưa được bỏ ngẫu nhiên khi huấn luyện F1, việc bỏ phần này lúc suy luận có thể đưa ngữ cảnh ra ngoài phân phối huấn luyện. Các đối chứng này xác định hướng kiểm tra giả thuyết mà các lượt B0 và F1 hiện có chưa phân giải được.

## 3.8. Quy trình đánh giá và độ đo

Hai cấu hình đã chạy dùng robot Panda với bộ điều khiển OSC_POSE ở 20 Hz, hai ảnh kích thước $256\times256$ và cùng danh sách trạng thái khởi tạo đã khóa. Mỗi lượt có 20 bước làm nóng theo quy trình trước khi tính các bước của chính sách. SmolVLA dự đoán đoạn $H=50$ hành động và thực thi $s=10$ hành động trước khi truy vấn lại. Bộ ghi cập nhật sau từng hành động, vì sự kiện có thể được xác nhận giữa hai lần truy vấn.

Mỗi lượt có giới hạn 1.000 hành động do chính sách sinh và dừng sớm khi bộ chấm ghi nhận vượt mục tiêu. Một trạng thái hoàn tất xuất hiện giữa lượt vẫn phải được duy trì theo quy tắc chấm ở thời điểm kết thúc. Quy tắc này giữ yêu cầu kết thúc đúng tiến độ, vốn là phần quan trọng của tác vụ lặp. Bộ chấm đọc tiến độ và vị từ từ bản mã đánh giá LIBERO-Mem đã cố định phiên bản. Kết quả được diễn giải trong phạm vi quy trình mô phỏng cục bộ đã khai báo.

Với lượt $i$, gọi $K_i$ là số mốc được bộ chấm xác nhận, $K_i^\star$ là số mốc cần đạt, $P_i$ là vị từ đích tại thời điểm kết thúc và $O_i$ là cờ vượt mục tiêu. Gọi $M$ là số lượt hoàn tất đánh giá trong nhóm đang xét. Các độ đo gồm chỉ báo thành công $S_i$, tỷ lệ hoàn thành mốc $\mathrm{SCR}_i$ và tỷ lệ thành công thực nghiệm $\widehat{\mathrm{SR}}$.

$$
\begin{aligned}
S_i &= \mathbf{1}[K_i=K_i^\star\land P_i\land\neg O_i],\\
\mathrm{SCR}_i &= K_i/K_i^\star,\qquad
\widehat{\mathrm{SR}} = \frac{1}{M}\sum_{i=1}^{M}S_i.
\end{aligned}
$$

$\mathrm{SCR}_i$ phản ánh tiến độ từng phần. $S_i$ còn yêu cầu trạng thái đích hợp lệ ở điểm cuối và dừng trong giới hạn số lần lặp. Hai tác vụ có cùng số lượt trong mỗi cấu hình đã chạy, nên trung bình tỷ lệ thành công theo tác vụ bằng tỷ lệ tính trên tất cả các lượt của cấu hình. Trạng thái khởi tạo được ghép giữa B0 và F1, đồng thời được dùng lại giữa T1 và T3. Cách ghép này cho phép đối chiếu từng trường hợp nhưng tạo phụ thuộc giữa các quan sát chung trạng thái. Việc phân tích vì vậy cần giữ thông tin nhóm trạng thái, số seed và điều kiện dừng của từng lượt.

# Chương 4. Thực nghiệm và thảo luận

## 4.1. Phạm vi suy luận thực nghiệm

Để hỗ trợ điều khiển, một biểu diễn tiến độ cần thỏa mãn ba điều kiện: có thể suy ra từ quan sát khả dụng, đến được chính sách đúng lúc và chứa thông tin mà chính sách có thể sử dụng để chọn hành động. Khóa luận xem xét ba loại bằng chứng: sự phù hợp của sự kiện với mốc tiến độ trong quỹ đạo trình diễn, độ nhạy của sai số dự đoán hành động đối với đầu vào bộ nhớ, và kết quả điều khiển vòng kín. Đối chiếu ba phép đo giúp xác định mắt xích nào đã có bằng chứng và mắt xích nào cần một phép can thiệp riêng.

Nguồn phân tích chính gồm phép phát lại sáu quỹ đạo trình diễn ngày 26/10/2026, nhật ký xác thực của mô hình F1 và 40 quỹ đạo điều khiển hoàn tất trong đợt đánh giá ngày 09/11/2026. Với đợt đánh giá vòng kín, số liệu được tính lại từ từng bản tóm tắt quỹ đạo, sau khi đối chiếu tổng kiểm, trạng thái hoàn tất, tác vụ, trạng thái khởi tạo và hạt giống. Danh mục dữ liệu, phiên bản và tổng kiểm được lưu cùng bản ghi thực nghiệm ngày 12/11/2026 [@artifact2026].

Đối chứng của phân tích vòng kín là B0 trên chính T1/T3. Kết quả V1 trong phụ lục chỉ cung cấp bối cảnh về chính sách gốc trên các tác vụ khác và với nhịp truy vấn khác. [@artifact2026]

## 4.2. Mức phù hợp giữa sự kiện thị giác và mốc tiến độ

Để khảo sát RQ1, bộ ghi sự kiện từ RGB được áp dụng trên sáu quỹ đạo trình diễn T1/T3, gồm 1.276 khung hình và ba trạng thái khởi tạo. Với mỗi trạng thái đều có quỹ đạo của cả T1 lẫn T3. Hai quỹ đạo được dùng để phát triển phương pháp và các quỹ đạo bổ sung cũng đã được quan sát trước khi tổng hợp. Phép đo này vì vậy có phạm vi chẩn đoán trên những quỹ đạo đã biết. Nhãn tham chiếu mô tả thời điểm tiến độ tích lũy đạt mốc, nên phù hợp để đối chiếu sự kiện hoàn thành chu kỳ, còn thời điểm tiếp xúc hoặc nhả vật cần một loại nhãn khác.

Mỗi sự kiện đặt lại do bộ ghi xác nhận được ghép với nhiều nhất một mốc tham chiếu, trong dung sai 20 khung hình. Tất cả 12 sự kiện xác nhận đều ghép được với 12 mốc. Độ trễ xác nhận có trung vị 9 khung hình, trung bình xấp xỉ 9,92 khung hình và nằm trong khoảng 6–20 khung hình. Kết quả cho thấy có thể trích xuất một biến tiến độ hữu ích để mô tả những quỹ đạo đã xem từ bằng chứng thị giác. Độ trễ quan sát được cho thấy giữa thời điểm tiến độ thay đổi và thời điểm chính sách có thể nhận bằng chứng về thay đổi đó tồn tại một khoảng chờ cần tính đến.

| Đại lượng khảo sát | Kết quả quan sát | Phạm vi diễn giải |
|---|---:|---|
| Quỹ đạo trình diễn / trạng thái / khung hình | 6 / 3 / 1.276 | Các quỹ đạo hoàn thành tác vụ đã được quan sát |
| Mốc tham chiếu / sự kiện xác nhận / cặp ghép | 12 / 12 / 12 | Ghép một-một trong dung sai ±20 khung hình |
| Sự kiện dư / mốc bị bỏ sót | 0 / 0 | Tính trên sáu quỹ đạo trình diễn |
| Độ trễ xác nhận | Trung vị 9, khoảng 6–20 khung hình | So với mốc tham chiếu |
| Phát hiện bát có IoU $\geq0{,}5$ | 1.169/1.172 (99,74%) | Mẫu số chỉ gồm phát hiện đã được bộ theo dõi chấp nhận |
| Có định vị vật / có văn bản bộ nhớ | 1.172/1.276 / 1.258/1.276 khung hình | Tỷ lệ có đầu ra ở từng khâu |
| Lời nhắc dài nhất | 31 trên giới hạn 48 token | Đo bằng bộ tách token đã chọn |
| Độ trễ xử lý RGB | Khoảng 999–1.058 ms/khung hình | Độ trễ tính toán ở lần đo riêng |

Tỷ lệ khớp hộp giới hạn đo chất lượng định vị có điều kiện trên các phát hiện được chấp nhận. Tỷ lệ này bỏ qua các khung mà đối tượng bị bỏ sót hoặc bị từ chối. Tương tự, việc tạo được văn bản chỉ chứng tỏ biểu diễn có đầu ra tại khung hình đó. Muốn ước lượng mức phù hợp của nội dung bộ nhớ trên phân phối điều khiển, cần cả quỹ đạo thành công lẫn thất bại và nhãn độc lập cho những tình huống như nhấc hụt, đặt lệch đĩa hoặc che khuất kéo dài.

Phụ lục ghi nguồn dữ liệu, phương pháp đối chiếu và các phép kiểm tính nhất quán của quá trình tạo bản ghi. [@artifact2026]

## 4.3. Điều kiện đối chiếu trong mô phỏng

Việc so sánh vòng kín giữa các chính sách đòi hỏi giữ ổn định động lực học, điều kiện quan sát và tiêu chí chấm điểm. Mức độ quy kết còn phụ thuộc vào những thành phần được thay đổi trong phép đối chứng. Hai lần chạy B0/F1 có cùng tổng kiểm của mô hình gốc, cấu hình đánh giá và danh sách trạng thái. Bản ghi nguồn của hai lần chạy khác nhau ở ba tập lệnh khởi chạy hoặc chẩn đoán. Các điểm khớp giúp thu hẹp diễn giải về biến gây nhiễu, còn khác biệt nguồn đặt giới hạn cho phát biểu về độ tương đương của hai lần chạy. Phụ lục chỉ ra vị trí các bản ghi có thể đối chiếu [@artifact2026].

| Điều kiện | Giá trị dùng trong thực nghiệm |
|---|---|
| Mô hình gốc | SmolVLA cho LIBERO, cùng phiên bản ở B0 và F1 |
| Nguồn tác vụ và bộ chấm | LIBERO-Mem tại phiên bản đã khóa |
| Mô phỏng | Franka Panda trong robosuite 1.4.0, cùng thiết lập động lực học |
| Đầu vào chính sách | Hai camera 256×256, trạng thái 8 chiều, tiền xử lý theo checkpoint gốc |
| Điều khiển | OSC_POSE ở 20 Hz trong mô phỏng, dự đoán $H=50$, thực thi $s=10$ |
| Khởi tạo và điểm cuối | Trạng thái từ danh sách đã khóa, 20 bước làm ấm, tối đa 1.000 hành động chính sách, dừng sớm khi vượt số chu kỳ |

Huấn luyện sử dụng hai GPU Tesla T4. Phép đánh giá dùng NVIDIA GeForce RTX 4060 Laptop, với mô phỏng chạy trong WSL và suy luận chính sách trên Windows. Những điều kiện này xác định môi trường áp dụng của kết quả. Chu kỳ danh định 20 Hz của bộ điều khiển tương ứng 50 ms mỗi bước mô phỏng. Riêng thời gian xử lý một khung hình RGB trong phép đo chẩn đoán đã xấp xỉ một giây. Thời gian phát lại máy trạng thái từ dữ liệu đã lưu khoảng 1–2 ms mỗi khung hình phản ánh chi phí của một thành phần khác, sau khi đã có kết quả nhận dạng vật. Khi mô phỏng tạm dừng để chờ suy luận, tiến độ mô phỏng và thời gian thực trôi qua là hai đại lượng khác nhau. Một kết luận về khả năng vận hành thời gian thực cần đo trọn chu trình trên cấu hình đã xác định.

## 4.4. Mẫu nghiên cứu và điều kiện thích ứng

Tập dữ liệu huấn luyện kết hợp quỹ đạo thao tác cơ sở LIBERO-10 với quỹ đạo T1/T3 từ nguồn LIBERO-Mem của tác giả. Bảng dưới ghi số quỹ đạo trong danh mục dữ liệu được dùng để huấn luyện F1. Các quỹ đạo có cùng mã băm trạng thái khởi tạo được phân vào cùng một tập. Trong tập kiểm tra T1/T3, 20 tổ hợp tác vụ và trạng thái của mỗi điều kiện tương ứng 10 nhóm trạng thái dùng chung giữa hai tác vụ. Tập này đã được sử dụng trong quá trình phát triển. Vì vậy, số liệu vòng kín phù hợp cho phân tích thăm dò và hình thành giả thuyết kiểm chứng.

| Nhóm dữ liệu | Huấn luyện | Xác thực | Kiểm tra | Tổng |
|---|---:|---:|---:|---:|
| Thao tác cơ sở LIBERO-10 | 400 | 50 | 50 | 500 |
| Mem T1 | 80 | 10 | 10 | 100 |
| Mem T3 | 78 | 10 | 10 | 98 |
| Tổng | 558 | 70 | 70 | 698 |

Lấy mẫu hỗn hợp nhằm đặt ra bài toán duy trì kỹ năng cơ sở (RQ3) trong cùng quá trình thích ứng với tác vụ phụ thuộc lịch sử. Trong mỗi lô cập nhật, số mẫu của hai miền được phân bổ theo tỷ lệ đã chọn. Trong từng miền, tác vụ được chọn luân phiên, sau đó lấy mẫu đều theo quỹ đạo rồi chọn khung hình. Cách này làm giảm ảnh hưởng của độ dài quỹ đạo lên xác suất được chọn. Tỷ lệ lấy mẫu 50/50 là một điều kiện huấn luyện đã quan sát, nên giá trị của việc trộn dữ liệu và lựa chọn tỷ lệ vẫn cần đối chứng riêng.

| Hạng mục | Cấu hình đã sử dụng | Vai trò nghiên cứu | Giới hạn suy luận |
|---|---|---|---|
| Vị trí thích ứng | LoRA tại phép chiếu query/value của khối sinh hành động, hạng 16, 645.120 tham số được cập nhật | Giữ mô hình nền cố định khi thích ứng | Chỉ khảo sát một vị trí và một hạng |
| Tối ưu hóa | AdamW, tốc độ học ban đầu $2\times10^{-5}$, làm ấm 100 bước, 3.000 bước cập nhật, chặn chuẩn gradient 1,0 | Cố định ngân sách của F1 | Chưa có khảo sát siêu tham số |
| Lấy mẫu hỗn hợp | Lô hiệu dụng 32, tỷ lệ dữ liệu cơ sở 0,5, hạt giống 42 | Đặt kỹ năng cơ sở và thích ứng vào cùng bài toán huấn luyện | Mức độ duy trì cần đo bằng điều khiển vòng kín |
| Bộ nhớ khi huấn luyện | Tỷ lệ bỏ bộ nhớ bằng 0, xác thực và lưu mô hình mỗi 500 bước | Xác định phân phối đầu vào của F1 | Bỏ bộ nhớ khi suy luận có thể thay đổi phân phối |
| Mô hình được đánh giá | Trọng số LoRA lưu riêng, dùng checkpoint ở bước 3.000 | Cố định phiên bản dùng cho các phép đo | Chưa chọn checkpoint theo tỷ lệ thành công vòng kín |

Tổng kiểm của danh mục dữ liệu khớp với bản ghi huấn luyện, và bốn tệp được liệt kê trong bản xuất F1 có tổng kiểm khớp bản kê. Phép đối chiếu giới hạn ở các tệp đã nêu. Dữ liệu Mem của F1 dùng các quỹ đạo do tác giả công bố, với ảnh được đưa về quy ước gốc trước khi áp dụng tiền xử lý của chính sách. Một đợt tái tạo dữ liệu tại chỗ ghi nhận 73 trên 78 quỹ đạo phát lại T3 đạt điều kiện thành công của đợt phát lại. Các quỹ đạo đó là một nhánh nghiên cứu riêng. Chuỗi xác thực dữ liệu dẫn xuất, lập chỉ mục và huấn luyện trên phiên bản derived-v1 chưa hoàn tất, nên phân tích F1 căn cứ vào tập dữ liệu của tác giả [@artifact2026].

## 4.5. Độ nhạy ngoại tuyến đối với thông tin bộ nhớ

Tại bước 3.000, trên 4.299 mẫu khung hình của tập xác thực Mem, hàm mất mát flow matching của cùng F1 là 0,23023575 khi có bộ nhớ và 0,33553660 khi bỏ bộ nhớ. Trên 13.525 mẫu khung hình của tập xác thực cơ sở, hai chế độ đều cho giá trị 0,42984212. Các khung hình được lấy từ quỹ đạo và có thể phụ thuộc theo thời gian. Phép so sánh giữ cố định trọng số F1 và thay đổi cách cung cấp bộ nhớ ở đầu vào.

![Hàm mất mát flow matching trên tập xác thực theo số bước huấn luyện. Hai chế độ có và bỏ bộ nhớ được đo trên cùng checkpoint ở mỗi bước.](figures/validation_loss.png)

Chênh lệch trên tập Mem là bằng chứng rằng cách cung cấp đầu vào bộ nhớ có liên hệ với chất lượng dự đoán hành động trên phân phối xác thực. Do F1 được huấn luyện với tỷ lệ bỏ bộ nhớ bằng 0, phép bỏ bộ nhớ có thể đồng thời loại thông tin tiến độ và đưa đầu vào ra ngoài phân phối huấn luyện. Để tách hai cách giải thích này, một phép thử có kiểm soát cần duy trì định dạng và dung lượng lời nhắc, đồng thời thay đổi có chủ đích thông tin lịch sử mà nó chứa. Mặt khác, một chính sách có sai số dự đoán nhỏ trên quỹ đạo trình diễn vẫn có thể thực thi kém khi các hành động của chính nó làm thay đổi phân phối quan sát.

Đối với RQ3, hai giá trị hàm mất mát bằng nhau trên tập cơ sở chỉ phản ánh tính ổn định giữa hai chế độ đầu vào của F1 trong phép đo này. Chẳng hạn, các mẫu cơ sở có thể có cùng đầu vào ở hai chế độ nếu chúng đều không mang hậu tố bộ nhớ. Để ước lượng mức duy trì kỹ năng, phép so sánh cần đặt chính F1 và B0 vào cùng các tác vụ cơ sở, với tiêu chí thành công và ngân sách điều khiển giống nhau.

## 4.6. Kết quả điều khiển vòng kín

RQ2 đặt vấn đề liệu lịch sử có giúp chính sách hoàn thành tác vụ khi đã kiểm soát thích ứng mô hình. Dữ liệu hiện có đối chiếu B0, tức checkpoint gốc, với F1, tức cấu hình có cả thích ứng và bộ nhớ quan sát trực tuyến. Đây là phép so sánh sơ bộ hai cấu hình nghiên cứu. Mỗi điều kiện dùng cùng 20 tổ hợp tác vụ và trạng thái, với hạt giống chính sách 42. Kết quả áp dụng cho giao thức T1/T3 của khóa luận, với nhãn `CUSTOM_V2_LIBERO_MEM` và `NON-COMPARABLE_OFFICIAL_PAPER`.

| Nhiệm vụ | Điều kiện | Thành công / số lượt | SR | SCR trung bình | Vượt số chu kỳ / số lượt |
|---|---|---:|---:|---:|---:|
| T1 | B0 | 0/10 | 0% | 0,1000 | 1/10 |
| T1 | F1 | 0/10 | 0% | 0,1000 | 1/10 |
| T3 | B0 | 0/10 | 0% | 0,0000 | 0/10 |
| T3 | F1 | 1/10 | 10% | 0,1667 | 0/10 |
| Hai tác vụ | B0 | 0/20 | 0% | 0,0500 | 1/20 |
| Hai tác vụ | F1 | 1/20 | 5% | 0,1333 | 1/20 |

![Tỷ lệ thành công trên T1 và T3 theo giao thức của khóa luận. Mỗi cột biểu diễn 10 quỹ đạo. Hai tác vụ dùng chung các nhóm trạng thái khởi tạo và cùng một hạt giống chính sách.](figures/mem_success.png)

Gọi $S_{c,j}$ là chỉ báo thành công của điều kiện $c$ trên tổ hợp tác vụ và trạng thái thứ $j$. Chênh lệch tỷ lệ thành công quan sát được tính theo

$$
\widehat{\Delta}_{\mathrm{SR}}
=\frac{1}{M}\sum_{j=1}^{M}\left(S_{F1,j}-S_{B0,j}\right),
\qquad M=20.
$$

Có 19 cặp có kết quả $(0,0)$ và một cặp có kết quả $(0,1)$ theo thứ tự $(B0,F1)$, nên $\widehat{\Delta}_{\mathrm{SR}}=0{,}05$, tức 5 điểm phần trăm. Toàn bộ chênh lệch đến từ một lần thực thi T3. Hai tác vụ chia sẻ 10 nhóm trạng thái và thí nghiệm chỉ dùng một hạt giống, nên báo cáo giới hạn ở thống kê mô tả. Việc ước lượng độ bất định cần thêm biến thiên qua trạng thái và hạt giống, đồng thời giữ cấu trúc ghép cặp.

SCR ghi nhận mức hoàn thành từng phần. Trên T1, cả B0 và F1 đều có một quỹ đạo đạt mốc tiến độ nhưng tiếp tục đến mức vượt số chu kỳ. Trên T3, SCR trung bình của F1 là 0,1667 trong khi của B0 bằng 0, cho thấy bộ đánh giá ghi nhận thêm tiến độ ở một số lần chạy. Diễn giải này bổ sung cho tỷ lệ thành công, vì tiến độ từng phần và hoàn thành ở điểm cuối là hai tiêu chí khác nhau. Đối chứng F0 thích ứng trên cùng dữ liệu nhưng không có bộ nhớ và phép đánh giá F1 khi thay đổi có kiểm soát đầu vào lịch sử vẫn cần thiết để trả lời RQ2.

## 4.7. Bất đồng giữa tiến độ quan sát và tiến độ được chấm

Khi chính sách tự sinh quỹ đạo, nó có thể đưa hệ thống đến những tình huống ít gặp trong dữ liệu trình diễn. Trên 20 quỹ đạo F1, 19 trạng thái bộ nhớ cuối báo lịch sử chưa đầy đủ. Trong 1.927 lần gọi chính sách, 1.894 lời nhắc chứa hậu tố bộ nhớ và 1.738 lời nhắc chứa cụm `history incomplete`. Hai số đếm ở cấp lời nhắc lần lượt phản ánh tần suất cung cấp bộ nhớ và tần suất báo lịch sử chưa đầy đủ. Chúng là các phép đo lặp theo thời gian trên cùng quỹ đạo. Độ đúng của từng sự kiện đòi hỏi đối chiếu với nhãn độc lập.

Quỹ đạo T3 thành công duy nhất, `T3_F1_test_state2` khởi tạo từ `demo_29`, là một trường hợp chẩn đoán cho RQ4. Tại bước 1.000, bộ đánh giá ghi đủ ba chu kỳ, điều kiện cuối được thỏa mãn và số chu kỳ không bị vượt. Bộ ghi thị giác cuối chỉ xác nhận hai lần đặt lại và đánh dấu lịch sử chưa đầy đủ. Bất đồng này làm suy yếu cách giải thích rằng chính sách dừng khi bộ đếm thị giác ghi đủ ba chu kỳ. Kết quả vẫn tương thích với nhiều cơ chế, chẳng hạn sử dụng một phần bằng chứng lịch sử, phản ứng với quan sát hiện tại, hoặc hành vi đã thay đổi qua thích ứng trọng số.

Ở `T1_F1_test_state7`, bộ đánh giá ghi một mốc và SCR bằng 1, nhưng dừng quỹ đạo tại bước 266 do vượt số chu kỳ. Bộ ghi thị giác cuối ghi nhận 0 lần đặt lại và lịch sử chưa đầy đủ. B0 cũng có một quỹ đạo T1 vượt số chu kỳ, ở trạng thái 3 và bước 290. Những trường hợp này chỉ ra hai vấn đề khác nhau: hệ thống có thể thực hiện thao tác đạt mốc nhưng không duy trì đúng điểm cuối, và bộ ghi thị giác có thể thiếu bằng chứng để xác nhận mốc mà bộ đánh giá đã ghi nhận. Muốn quy nguyên nhân cần nhãn độc lập theo thời gian và can thiệp riêng lên nội dung bộ nhớ trong khi giữ các điều kiện còn lại ổn định.

## 4.8. Hiệu lực của kết luận và thiết kế kiểm chứng tiếp theo

Đối với RQ1, sự phù hợp trong sáu quỹ đạo trình diễn cung cấp bằng chứng về tính khả thi của việc biểu diễn tiến độ bằng sự kiện thị giác. Dấu thời điểm khả dụng cho phép kiểm tra dữ kiện nào đã có thể đưa vào lời nhắc tại từng bước. Khoảng cách giữa kết quả ghép mốc trên trình diễn và tỷ lệ lịch sử chưa đầy đủ trên quỹ đạo F1 đặt ra giới hạn về tính đại diện của phép kiểm chẩn đoán. Một tập nhãn độc lập gồm cả tình huống thất bại, với quy tắc ghép và dung sai đặt trước, cho phép ước lượng riêng xác nhận sai, bỏ sót, độ trễ và tỷ lệ tình huống chưa xác định.

Đối với RQ2, cần giữ dữ liệu, phân hoạch, ngân sách huấn luyện và quy tắc chọn checkpoint giống nhau khi so F0 với F1. Phép thử can thiệp lúc suy luận cần có điều kiện chỉ dùng thông tin hiện tại và điều kiện có cùng định dạng nhưng thay đổi giá trị lịch sử. Hai phép so sánh trả lời các câu hỏi khác nhau: thích ứng với bộ nhớ có tạo lợi ích vòng kín hay không, và chính sách đã thích ứng nhạy thế nào với nội dung cụ thể của bộ nhớ. Một tập trạng thái xác nhận độc lập với phát triển và nhiều hạt giống sẽ cho phép ước lượng chênh lệch ghép cặp cùng độ bất định ở cấp nhóm trạng thái. Chỉ đổi hạt giống trên các trạng thái đã dùng để phát triển không tạo một tập xác nhận độc lập.

Đối với RQ3, đánh giá chính F1 trên các tác vụ cơ sở ghép cặp với B0 là cách đo trực tiếp mức duy trì. Nếu mục tiêu là kiểm tra tính không kém hơn, cần xác định trước biên suy giảm có thể chấp nhận được và cỡ mẫu đủ để diễn giải biên đó. Cơ chế chọn lại trọng số B0 cho tác vụ cơ sở chỉ đo chất lượng của một quy tắc định tuyến và không định lượng kỹ năng của F1.

Đối với RQ4, ưu tiên can thiệp vào những yếu tố có thể đổi giá trị hoặc thời điểm khả dụng của bằng chứng, gồm quy ước ảnh, độ trễ xác nhận và số hành động giữa hai lần gọi chính sách. Có thể dùng cùng một quỹ đạo để kiểm tra tác động riêng lên chất lượng quan sát trước khi thay đổi các yếu tố đó trong vòng kín. Cần báo cáo cả thời gian mô phỏng và thời gian thực đo trên phần cứng. Thiết kế như vậy gắn mỗi phép đo với một cơ chế sai lệch cụ thể, từ sai số của biểu diễn cho đến hạn chế về cách sử dụng biểu diễn trong điều khiển.

# Chương 5. Kết luận

## 5.1. Kết luận theo câu hỏi nghiên cứu

Khóa luận nghiên cứu tiến độ thao tác như một thành phần của trạng thái quyết định mà quan sát hiện thời có thể không biểu hiện đầy đủ. Trong bài toán nhấc và đặt lại bát, sự kiện rời đĩa rồi trở lại đĩa gốc cho phép diễn đạt thông tin lịch sử bằng một đơn vị có ý nghĩa đối với yêu cầu tiếp tục hay dừng. Cách phát biểu này liên kết yêu cầu của tác vụ, bằng chứng có thể quan sát và hành vi cần đánh giá ở chính sách điều khiển.

Với **RQ1**, bộ nhớ ghi các cặp sự kiện đã xác nhận, thời điểm khả dụng và tình trạng thiếu lịch sử. Hai tính chất về đếm chu kỳ và tính không nhìn trước được thiết lập dưới các giả thiết của phương pháp. Trên sáu quỹ đạo trình diễn, 12 sự kiện trở lại đĩa ghép được với 12 mốc chú giải trong dung sai ±20 khung hình. Độ đúng trên quỹ đạo thất bại cần được đo riêng.

Với **RQ2**, điều kiện có bộ nhớ cho sai số flow matching trên tập xác thực Mem thấp hơn chế độ bỏ thông tin lịch sử của cùng adapter. Trong đánh giá vòng kín, cấu hình kết hợp adapter và bộ nhớ hoàn thành 1/20 lượt, còn SmolVLA gốc hoàn thành 0/20 lượt. Hai điều kiện khác nhau đồng thời về thích ứng trọng số và đầu vào lịch sử, nên chênh lệch quan sát được chỉ mô tả hai cấu hình trên tập trạng thái đã khảo sát. Giả thuyết rằng lịch sử sự kiện tự nó cải thiện khả năng hoàn thành T3 còn mở.

Với **RQ3**, việc lấy mẫu hỗn hợp tạo điều kiện để chính sách tiếp tục quan sát các thao tác cơ sở khi thích ứng với tác vụ phụ thuộc lịch sử. Khả năng duy trì kỹ năng cần được xác định bằng kết quả thực thi các tác vụ cơ sở của chính adapter đã đánh giá, trên cùng tập trạng thái và giao thức với mô hình gốc. Kết quả ngoại tuyến hiện tại chỉ mô tả sai số dự đoán trên dữ liệu cố định.

Với **RQ4**, bộ ghi báo lịch sử chưa đầy đủ ở cuối 19/20 lượt của cấu hình có bộ nhớ. Trong lượt duy nhất thành công, bộ ghi xác nhận hai chu kỳ và bộ chấm ghi nhận ba. Bất đồng này xác định một hạn chế cần khảo sát khi chuyển từ trình diễn sang quỹ đạo do chính sách sinh ra. Ảnh hưởng của sai lệch bản ghi đến hành vi điều khiển cần được đo bằng can thiệp riêng.

## 5.2. Ý nghĩa và phạm vi của đóng góp

Đóng góp học thuật của khóa luận nằm ở cách liên kết biểu diễn tiến độ với bằng chứng có thể kiểm tra. Khi một bản ghi gắn với mốc quan sát và thời điểm khả dụng, có thể xác định một chính sách đã được cung cấp thông tin gì trước quyết định. Khi bộ chấm hoạt động độc lập với bản ghi, có thể quan sát các trường hợp ước lượng tiến độ và kết quả của tác vụ bất đồng. Hai đặc điểm này làm cho giả thuyết về vai trò của bộ nhớ trở thành một đối tượng có thể khảo sát bằng dữ liệu và đối chứng.

Độ khớp với mốc chú giải đánh giá bản ghi trên một tập quỹ đạo. Sai số ngoại tuyến đo mức phù hợp của dự đoán với hành động tham chiếu. Tỷ lệ thành công vòng kín đo kết quả khi chính sách tự làm thay đổi quan sát qua hành động. Ba phép đo xác định những khâu khác nhau cần kiểm tra khi kết quả không đồng nhất. Kết luận hiện áp dụng cho quan hệ bát–đĩa và giao thức mô phỏng đã khảo sát.

## 5.3. Hướng kiểm định tiếp theo

Ưu tiên đầu tiên là kiểm tra bộ ghi trên tập quỹ đạo độc lập có cả thao tác thành công và thất bại. Nhãn cần phân biệt nhấc hụt, che khuất, đặt sai đĩa và quay lại đúng đĩa. Việc báo cáo sự kiện bỏ sót, xác nhận nhầm và độ trễ xác nhận theo loại tình huống sẽ cho biết điều kiện nào làm thông tin tiến độ thiếu tin cậy. Các phép thử chèn khoảng chờ hoặc thay đổi nhịp thao tác có thể khảo sát thêm mức độ tách biệt giữa tiến độ đã xác nhận và thời gian đã trôi qua.

Ưu tiên thứ hai là đánh giá các chính sách trên cùng tập trạng thái giữ riêng và nhiều seed. Một adapter huấn luyện trên cùng dữ liệu và ngân sách nhưng không nhận bộ nhớ cho phép kiểm tra giá trị gia tăng của cấu hình sử dụng lịch sử. Can thiệp vào đầu vào của cùng một adapter, với nội dung đối chứng và ngân sách token tương đương, giúp khảo sát mức độ chính sách dựa vào lịch sử tại thời điểm ra quyết định. Trên các tác vụ cơ sở, cần xác định trước mức suy giảm chấp nhận được và thực hiện so sánh theo cặp với SmolVLA gốc. Khi hiệu lực của các phép đo này được xác lập, có thể kiểm tra khả năng chuyển phương pháp sang nhiều thực thể, mô hình nền khác và robot thật. -->
