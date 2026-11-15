**Kế hoạch tích hợp memory vào quyết định và thực thi hành động**

Ngày 14/11/2026. Đây là thiết kế V2. Tiến độ D0–D2, replay và các gate D3–D6 được cập nhật trong [bàn giao quyết định](MEMORY_DECISION_HANDOFF_2026-11-14.md). Thiết kế triển khai dùng lớp metadata bên ngoài renderer để khớp source hash của writer. Tài liệu đi cùng [roadmap](IMPLEMENTATION_AND_WRITING_ROADMAP_2026-11-13.md), [bàn giao P5–P6](P5_P6_HANDOFF_2026-11-13.md) và [protocol phân tích v2](evidence/mem_analysis_protocol_v2.json).

**1. Phương án được khuyến nghị**

Xây tầng điều phối thực thi đọc trực tiếp trạng thái memory có bằng chứng. Bản đầu theo dõi các dữ kiện liên quan tại mỗi query và yêu cầu query lại khi chúng thay đổi về nghĩa. Bản tiếp theo kiểm tra vật đích và hiệu ứng dự kiến của đề xuất hành động. Việc chọn giữa nhiều chunk, thu nhận thêm bằng chứng và conditioning bằng vùng ảnh được mở theo lỗi đã đo.

Thứ tự là quan sát thụ động → query lại theo sự kiện → kiểm tra đề xuất → chọn giữa các đề xuất → thu nhận thông tin khi cần. T1/T3 kiểm tiến trình và phản ứng đúng thời điểm. T7 kiểm danh tính cùng quan hệ. T9/T10 kiểm vật chứa và che khuất. Đây là scope đang được chọn trong bàn giao hiện tại. T2/T4 nằm ngoài đợt chính.

Hai giả định cần sửa trong đề xuất trước. Snapshot hiện có chưa đủ để kết luận action chunk sắp gắp vật nào. Cần thêm bộ đọc đề xuất, được đánh giá trên hành động thật. Ngoài ra, nhiều lần gọi policy trên cùng ảnh có thể cập nhật hàng đợi quan sát của LeRobot nhiều lần. Cơ chế sinh ứng viên phải quản lý trạng thái policy và RNG trước khi dùng kết quả để rerank.

**2. Đánh giá các hướng đã khảo sát**

| Hướng | Giá trị đối với thesis | Chi phí hoặc thiếu sót quan trọng | Quyết định |
|---|---|---|---|
| Text memory hiện tại | Đường tích hợp đã có, thích hợp làm đối chứng và đo token budget | Policy có thể đọc sai hoặc bỏ qua fact; prompt chỉ đổi tại query | Hoàn tất cặp P5 và dùng F1 làm nền cho các đối sánh cơ chế |
| Điều phối thực thi theo bằng chứng | Kiểm trực tiếp thời điểm memory gây thay đổi quyết định; dùng được cùng checkpoint | Requery tăng compute, có thể gây đổi mode hoặc giật chuyển động | Ưu tiên triển khai với một chunk mỗi query |
| Kiểm và chọn chunk theo lịch sử | Dùng count, identity, relation để loại lựa chọn sai | Cần bộ đọc ý định/hiệu ứng, ứng viên đủ đa dạng và kiểm trạng thái policy | Bước tiếp theo khi có bằng chứng về khả năng nhận diện đề xuất |
| Conditioning bằng ROI hoặc object tokens | Truyền danh tính lịch sử đến đúng vùng ảnh hiện tại | Cần thay interface action expert, train adapter và kiểm lỗi grounding | Mở khi ứng viên đúng hiếm do nhầm vật |
| Critic hướng dẫn flow hoặc world model | Có thể đổi trực tiếp quá trình sinh hành động | Cần rollout đa dạng, calibration và kiểm lỗi ngoài phân phối; gradient có thể khai thác lỗi critic | Nhánh nghiên cứu sau khi baseline chọn chunk có kết quả |
| Active perception bằng chuyển động | Có thể giải mơ hồ ảnh hưởng lựa chọn | Cần kỹ năng quan sát khả thi, chi phí thời gian và dữ liệu riêng | Khởi đầu bằng xử lý bổ sung trên camera sẵn có, sau đó mới xem xét chuyển động |
| Thay control head bằng recurrent/latent memory | Biểu diễn gọn và học trực tiếp từ lịch sử | Thay đổi lớn về training và confound kiến trúc | Dùng làm đối chiếu related work, mở nếu sau này có tài nguyên |

MemoryVLA truy xuất và kết hợp token nhớ để điều kiện hóa action expert. Embodied-SlotSSM dùng danh tính slot và biểu diễn quan hệ cho action decoding. SmoLSTM dùng trạng thái hồi quy xuyên episode. Các công trình này đặt nền cho nhánh học biểu diễn và cho thấy cần đo thêm lợi ích của cách sử dụng bằng chứng tường minh. [MemoryVLA](https://arxiv.org/abs/2508.19236), [Embodied-SlotSSM](https://arxiv.org/html/2511.11478v3), [SmoLSTM](https://arxiv.org/abs/2609.22854v1).

Task graph cùng procedural memory đã được đề xuất để chọn subgoal và xác minh chuyển tiếp. AAC điều chỉnh chunk theo entropy. AFI chấm trajectory bằng affordance hình học. GAF học critic để hướng dẫn flow của policy đóng băng. Vì vậy tính mới cần được xác lập ở cơ chế dùng danh tính và validity của bằng chứng lịch sử, cùng thực nghiệm can thiệp tương ứng. Đây là nhận định thiết kế từ survey, cần tiếp tục đối chiếu khi viết claim novelty. [Neuro-symbolic procedural reasoning](https://arxiv.org/html/2609.05369v1), [AAC](https://arxiv.org/abs/2604.04161v2), [AFI](https://arxiv.org/abs/2512.07472), [GAF](https://arxiv.org/abs/2607.02092).

**3. Câu hỏi và giả thuyết có thể bác bỏ**

Giữ câu hỏi bao trùm về memory cho VLA nhỏ trong ngữ cảnh, compute và quan sát bị giới hạn. Nhánh tích hợp cụ thể hóa RQ1–RQ2 bằng câu hỏi sau.

> Với cùng VLA nhỏ và cùng nguồn bằng chứng lịch sử, điều kiện nào khiến việc dùng tính hợp lệ của memory để điều phối query và lựa chọn hành động cải thiện quyết định phụ thuộc lịch sử so với text conditioning, khi đồng thời giới hạn lỗi quyết định, số lần suy luận và tỷ lệ không hoàn thành?

| Câu hỏi phụ | Giả thuyết | Bằng chứng cần có | Kết quả bác bỏ hoặc thu hẹp claim |
|---|---|---|---|
| Khi nào memory phải tác động lại vào điều khiển? | Query theo thay đổi ngữ nghĩa giảm hành động dựa trên fact cũ ở cùng ngân sách gọi policy | Đối chứng lịch query độc lập với memory, delay tới query/action, lỗi tiến trình | Gain biến mất khi đối sánh số query hoặc wall-clock |
| Bất định nào cần được giải trước hành động? | Chỉ xác minh ambiguity làm đổi quyết định giữ coverage tốt hơn dừng ở mọi unknown | Risk–coverage, thời gian xác minh, mơ hồ liên quan và không liên quan | Lỗi giảm chủ yếu do bỏ cuộc hoặc vượt thời gian |
| Danh tính lịch sử có đổi lựa chọn vật đúng cách? | Cùng ảnh hiện tại và cùng tập đề xuất, thay fact liên quan đổi lựa chọn đúng hướng | T7 và sau đó T9/T10, sửa fact có kiểm soát, đổi tên ID nhất quán | Chỉ nhạy với token/ID bất kỳ, hoặc chỉ tốt ở đếm T3 |

Primary contrast của protocol P6 v2 là F1–F0. Các phép so supervisor thuộc phân tích cơ chế đăng ký riêng. Nếu phát triển cơ chế sau khi đã xem kết quả unseen, các state đó trở thành development đối với claim mới. Roster mới hoặc một báo cáo exploratory phải thể hiện đúng mức độc lập còn lại.

**4. Khoảng cách từ code đang có tới thiết kế**

| Thành phần hiện có | Khả năng đã thấy trong code | Phần cần bổ sung |
|---|---|---|
| `src/memory/relation_memory.py` | Snapshot bất biến, event/evidence IDs, current/historical relation, observed returns, gap/conflict handling | Validity có version theo fact, phụ thuộc của suy luận, khả năng truy lại evidence cho sự kiện cũ khi cần |
| `src/perception/relation_tracking.py` | Association bảo thủ, origin binding, vị trí hiện tại, ambiguous/unknown | Tập phương án danh tính giới hạn cho T7; xử lý phần giả thuyết bị cắt bằng residual unknown |
| `src/interfaces/relation_memory_text.py` | Chiếu fact cùng qualifier và kiểm ngân sách 48 token | Trả thêm định danh/version của fact đã chiếu, phụ thuộc và facts dropped theo cấu trúc |
| `src/evaluation/libero_mem_rollout.py` | Writer cập nhật giữa các action; mỗi query thực thi tối đa 10 action của chunk trong protocol hiện tại | Tầng supervisor tại query và giữa bước; các quyết định query lại có nguyên nhân và revision |
| `src/models/smolvla/adapter.py` | API trả chunk qua pre/postprocessor chính thức | Capability cho candidate sampling với một lần cập nhật observation state và RNG có kiểm soát |
| `src/evaluation/mem_conditions.py` và CLI | B0/F0/F1/F1-off, manifest treatment, runtime guard | Execution treatment riêng và contract về quyền đọc fact, thông số query, bộ đọc đề xuất, parser instruction |
| `src/evaluation/mem_eval_io.py` | JSONL observation/query/action/memory, chunk và trajectory NPZ | Nhật ký quyết định, episode từ chối theo thuật toán, candidate/chunk ID và reason code |
| `scripts/aggregate_mem_pilot.py` | Ghép cặp F0/F1, kiểm identity và số lượt | Tổng hợp cơ chế, tỷ lệ từ chối và clustering với nhiều decision window trong cùng state |

Writer giữ tám recent events cùng một số event cần cho reducer, rồi thu gọn evidence. Một phân tích can thiệp lịch sử dài cần journal có thể truy vết các dependency đã rời working state. Counter `returns` hiện tăng theo sự kiện đã xác nhận và không có đường thu hồi sự kiện sai. Trong bản đầu, phát hiện bằng chứng mâu thuẫn phải khiến kết luận count trở thành unknown. Mở rộng tiếp theo mới hỗ trợ tính lại count từ journal đã sửa validity. Một chuỗi quan sát liên tiếp có tương quan và điểm detector chưa được hiệu chuẩn thành xác suất event đúng.

**5. Kiến trúc và hợp đồng dữ liệu**

Luồng online dự kiến là quan sát cùng hành động đã thi hành → writer hiện tại → trạng thái memory → projection cho VLA và dữ kiện cho supervisor → chunk thật → kiểm tra thực thi → controller → quan sát mới. Evaluator cung cấp nhãn cho recorder và phân tích. Quyết định online chỉ nhận các trường đã định nghĩa trong observation/memory contract.

Tái sử dụng `RelationMemorySnapshot`, `V2MemoryEvent`, `ObjectIdentity`. Bổ sung ít kiểu dữ liệu trong một module contract và một module supervisor. Tách thêm module khi có triển khai thực tế đủ lớn.

| Kiểu dữ liệu dự kiến | Trường tối thiểu | Công dụng |
|---|---|---|
| `DecisionContext` | snapshot revision, query ID, instruction intent, selected fact IDs/versions, relevant fact keys, availability watermark | Ghi chính xác memory mà quyết định phụ thuộc vào, gồm những fact cần theo dõi dù đang unknown |
| `FactValidity` | fact ID, version, supported/unknown/conflicted/retracted, dependency IDs, available_at, lý do đổi | Giữ riêng sự kiện quá khứ, quan hệ hiện tại và liên kết danh tính |
| `ActionProposal` | proposal ID, raw chunk hash, policy state identity, sampling seed, tiền tố dự định thực thi | Gắn mọi action đã chạy với một chunk thật |
| `ProposalAssessment` | tập vật đích, loại hiệu ứng có thể có, thời điểm cam kết ước lượng, unknown/score đã hiệu chuẩn, provenance của bộ đọc | Cho biết phần nào của đề xuất có thể kiểm tra được |
| `ExecutionDecision` | continue/requery/reject/request_evidence/abstain, proposal ID, prefix length, fact IDs/versions, lý do, thời gian xử lý | Tác động vào runner và phục vụ can thiệp nhân quả |

Thời gian phải có observation index, thời điểm thu ảnh, available_at của evidence, query và bước thực thi. Một fact mới đến muộn chỉ có thể đổi quyết định từ lúc nó sẵn có. Khi requery, snapshot revision và observation state phải nhất quán. Chế độ đồng bộ là điểm bắt đầu phù hợp với runtime hiện tại.

Task semantics được tạo từ instruction bằng parser có phạm vi rõ hoặc annotation chỉ dựa vào instruction. Parser lỗi trả unsupported và lưu log. Mục tiêu số lượt, thao tác swap và yêu cầu chọn giỏ được tách khỏi writer ghi sự kiện. Các task profile và số vật mà hệ được cung cấp phải có provenance và được khai báo như prior. Việc đối sánh dùng cùng đặc quyền đó. BDDL, hidden object ID, reward, `_sub_goal_progress` và simulator ground truth phục vụ audit ngoài đường quyết định.

**6. Thuật toán triển khai theo mức**

**Mức đầu tiên dùng một đề xuất và query theo bằng chứng.** Tại mỗi query, lưu các fact/version thực sự chiếu vào prompt cùng tập khóa fact liên quan đến instruction. Sau mỗi action đã chạy, writer cập nhật. Khi có đổi count, relation, identity binding hoặc validity liên quan, supervisor đánh giá xem quyết định cũ có còn dựa trên cùng tiền đề. Revision đổi do thêm timestamp hoặc evidence xác nhận lại cùng nghĩa sẽ không tự kích hoạt query.

Requery hủy phần chunk chưa chạy, dùng observation mới nhất và policy state hợp lệ. Mỗi revision đã xử lý được ghi nhận để tránh lặp query mãi ở cùng bước. Có trần số query, giới hạn requery liên tiếp, hysteresis theo bằng chứng và cách xử lý hết ngân sách. Các thông số được chọn trên seen rồi khóa. Bản này đo giá trị của phản ứng theo sự kiện. Kiểm chứng ngữ nghĩa của từng action được bổ sung ở mức sau.

**Mức tiếp theo kiểm đề xuất theo ba giá trị.** Với tập giả thuyết lịch sử phù hợp `H_t`, phép kiểm `V(h, a)` nhận supported, violated hoặc unknown. Một đề xuất được chấp nhận khi tiền điều kiện cần kiểm được hỗ trợ trong mọi giả thuyết liên quan còn xét. Nếu mọi giả thuyết xác định cùng vi phạm thì reject. Các trường hợp còn lại cần evidence hoặc một lựa chọn khác được hỗ trợ. Khi giới hạn số giả thuyết, phần chưa biểu diễn tiếp tục đóng góp unknown vào quyết định.

Tính chất của quy tắc là nhất quán đối với tập giả thuyết và bộ kiểm tra đã xây dựng. Khả năng trạng thái thật bị loại khỏi tập và lỗi nhận diện hiệu ứng phải được đo riêng. Các score hiệu chuẩn trên development giúp chọn ngưỡng nhưng không bảo đảm chính xác ngoài phân phối. Thiết kế kế thừa nguyên tắc lập kế hoạch trên trạng thái niềm tin và chỉ giữ phần bất định cần cho lựa chọn hiện tại. [Belief-space planning](https://dspace.mit.edu/entities/publication/a08d58a1-ff3e-4839-b03e-66a67f86a7c7).

Ví dụ T3 có 2 observed returns nhưng history incomplete. Count có thể là 2 hoặc lớn hơn, tùy evidence. Chọn chính xác tập `{2, 3}` cần căn cứ về khoảng mất quan sát. Tăng count theo thời gian hoặc lệnh grasp sẽ tạo tiến độ chưa được quan sát. Một ảnh mới có thể xác nhận relation hiện tại nhưng không khôi phục số lượt đã mất. Với T7, giữ các phương án gán bát sau mất dấu. Nếu chúng cùng cho một thao tác hợp lệ thì tiếp tục. Với T9/T10, suy ra giỏ rỗng cần giả thiết về inventory và bằng chứng chứa còn hợp lệ.

**Đọc đề xuất trước khi cho nó quyền loại action.** Dự đoán ngắn hạn từ RGB, proprioception và chunk, trước hết tập trung vào tiếp cận vật, đóng kẹp gần vật và rời vùng đỡ. Dấu gripper command và tổng delta OSC đều chưa đủ để chứng nhận grasp hoặc trajectory thế giới. Với phép đo hình học cần calibration và contract controller đã audit. Các đoạn chưa biểu lộ vật đích trả unknown.

Nếu quy tắc hình học không đủ coverage, huấn luyện head nhỏ từ cửa sổ rollout development có nhãn độc lập. Nhãn gồm target entity/region, loại hiệu ứng, bước cam kết và ambiguous. Đánh giá false accept trên đề xuất sai, false reject trên đề xuất đúng, coverage, calibration và độ sớm của dự đoán. Dữ liệu hậu quả của action được dùng làm nhãn training sau khi chạy. Input online kết thúc tại thời điểm quyết định.

Ground truth hậu quả chỉ có cho chunk thực sự đã thi hành. Một candidate chưa chạy có thể được chú giải về ý định quan sát được, với nhãn ambiguous khi cần. Đánh giá khả năng candidate đạt success cần rollout nhánh có reset/state đã audit, ghi là chẩn đoán custom với chi phí riêng. Việc lấy hậu quả của quỹ đạo chuyên gia để gán thành công cho một chunk khác sẽ làm sai supervision. Nhãn target recall và recall của hành động thành công cần được báo đúng định nghĩa của từng phép đo.

**Chỉ mở nhiều ứng viên khi đã có khả năng kiểm tra.** Bắt đầu `K=2`, mở tới `K=4` nếu ích lợi và chi phí đo được. Sinh các ứng viên từ cùng observation/prompt và policy state, dùng noise có seed riêng. Thực hiện pre/postprocessing chính thức. Kiểm code LeRobot đang cài cho thấy `predict_action_chunk` gọi `populate_queues` và nhận noise tùy chọn. Adapter của repo hiện chưa cung cấp contract sampling này. Cần kiểm parity `K=1`, snapshot/restore queue, xử lý cache, RNG và telemetry trước khi đánh giá.

Xếp hạng ứng viên đã được kiểm hợp lệ bằng tiêu chí tiến triển được hỗ trợ, continuity chuyển động và chi phí. Khi mọi ứng viên bị loại, cho phép số lần lấy mẫu lại hữu hạn rồi xuất trạng thái abstain theo protocol. Nếu thiếu ứng viên đúng thường xuyên, chuyển công sức sang grounding hoặc năng lực policy. Một scorer chỉ có thể chọn trong các hành động đã được sinh.

**Thu nhận evidence bổ sung có mục tiêu.** Chọn phép quan sát dự kiến phân biệt hai giả thuyết đang dẫn tới hai lựa chọn khác nhau. Bản đầu chỉ dùng lại camera hiện có, crop chất lượng cao hoặc đối chiếu cache cục bộ. Đo thời gian, frame availability và chất lượng thông tin thu được. Xử lý lại cùng pixel sẽ không khôi phục một sự kiện quá khứ hoàn toàn không được ghi. Điều khiển cổ tay để nhìn vào giỏ cần kỹ năng đã kiểm và protocol riêng về bước vật lý. Giá trị thông tin là hướng tối ưu hóa, còn bản đầu dùng quy tắc quyết định có thể kiểm toán.

Khi hết ngân sách query/evidence hoặc không giải được ambiguity, hệ ghi `decision_abstention`. Protocol mới phải định nghĩa việc kết thúc lượt hoặc hành vi giữ tư thế đã audit. Mọi action gửi controller cần có nguồn hợp lệ. Trạng thái abstain được tính vào failure hoàn thành ở endpoint hệ thống. Lỗi hạ tầng được giữ trong sổ các lượt thiếu và bounds. Việc memory tự cho rằng xong tác vụ không thay quyết định success của evaluator.

**7. Gói triển khai và điều kiện chuyển bước**

Effort dưới đây là dự toán kỹ thuật cho một người, chưa gồm GPU, thời gian gán nhãn và sửa lỗi được phát hiện. Các gate là quan hệ phụ thuộc kỹ thuật. Chốt giá trị ngưỡng từ pilot chi phí và mức lỗi chấp nhận trước khi đánh giá unseen.

| Gói | Công việc và module | Artifact nghiệm thu | Gate và effort dự kiến |
|---|---|---|---|
| D0 | Khóa protocol supervisor, fact exposure, nghĩa abstain, budget và chỉ dẫn task. Định nghĩa schema nhãn decision window | Protocol có hash, hướng dẫn annotation, roster seen và phân bổ train/calibration theo nhóm state | Làm ngay cùng P5, 1–2 ngày |
| D1 | Thêm structured fact references vào renderer, `DecisionContext`, diff ngữ nghĩa và journal liên quan. Theo dõi song song, chỉ ghi đề xuất requery | Trace quyết định liên kết query/action/evidence, báo các trigger đúng/sai/muộn | Smoke thật với F1, parity của observer, 2–3 ngày |
| D2 | Runner gọi supervisor trước query và giữa action. Thêm control lịch query đối sánh, version protocol, vòng lặp hữu hạn và logging abstain | Seen pilot K=1, bảng latency/query/coverage/decision errors/SR | Writer và competence đủ cho mục tiêu đo; trigger đúng về thời điểm, 2–4 ngày |
| D3 | Bộ đọc đề xuất, nhãn target/commit, validity và dependency invalidation. Chạy đánh giá offline rồi shadow trên rollout | Báo false accept/false reject/unknown, accuracy tại điểm quyết định và lead time | Chỉ cho quyền reject khi sai số và coverage đạt mức đã khóa, 3–6 ngày cộng nhãn |
| D4 | API sinh K ứng viên, matched candidate controls, bounded resampling và selector | Parity K=1, candidate recall, selection accuracy và paired rollout | D3 đạt gate và candidate recall có dư địa cải thiện, 2–4 ngày cộng GPU |
| D5 | T7 source lock và evaluator Or/runtime; T8/T9/T10 sources, writer diagnostics và author-action runtime replay một demo/task đã có. T9/T10 báo terminal success; T8 ba milestone nhưng không terminal. Raw/refined T8 demo_1 action/state đầu vào trùng chính xác; robot-state replay vượt dung sai 1e-5 đầu tiên tại bước 219, nguyên nhân chưa rõ | Theo từng task có hash, smoke, writer/identity audit, closed-loop và failure cases | D2/D3 cùng runtime từng họ, nhãn độc lập, 1–2 tuần hoặc hơn |
| D6 | Can thiệp fact và evidence, đối chứng compute/information, tổng hợp theo cụm state, biểu đồ risk–coverage–cost | Báo cáo đã khóa, bảng/hình theo RQ và ledger claims | Chỉ bắt đầu xác nhận sau freeze, 3–5 ngày cộng rollout |

D1 có thể chuẩn bị bằng trace hiện có. D2 trở đi cần F1 có writer quan hệ đã qua nghiệm thu. Gói T7 có thể mở song song về nguồn/runtime trong phạm vi dung lượng, nhưng phải kiểm competence của policy trên họ mới trước khi quy lỗi cho memory. Nếu cần SFT T7, tạo version corpus mới và train các arm đối sánh trên cùng dữ liệu đó.

Sản phẩm tối thiểu đủ để chuyển sang thực nghiệm cơ chế là D0–D2 cùng một báo cáo seen có đối chứng query budget. D3–D4 mở theo kết quả nhận diện đề xuất. T7 là điều kiện thực nghiệm để nâng claim từ tiến trình lên danh tính. Các nhánh ROI, critic và chuyển động quan sát chỉ được chọn khi failure analysis xác định đúng nhu cầu đó.

**8. File, script và kiểm thử cần triển khai**

| Điểm tích hợp | Thay đổi cụ thể dự kiến |
|---|---|
| `src/interfaces/relation_memory_text.py` | Dùng kết quả renderer và lớp metadata ngoài source lock để đối chiếu fact IDs/versions cùng dependency refs. Tập fact dropped và instruction vượt budget có trạng thái riêng. |
| `src/memory/decision_context.py` · mới | Contract cùng bộ chiếu snapshot thành dữ kiện quyết định. Reuse kiểu identity/event đã có. Version validity, khóa fact liên quan và journal phục vụ hồi cứu. |
| `src/evaluation/memory_supervisor.py` · mới | Pure decision logic, K=1 scheduler đầu tiên. Cấu hình hysteresis, budget, availability, reason codes và các phép kiểm theo họ quan hệ. |
| `src/evaluation/libero_mem_rollout.py` | Tích hợp supervisor theo execution treatment, giữ liên kết từng action với candidate và prefix. Chỉ reset state khi reset episode hoặc theo contract policy. |
| `src/evaluation/mem_conditions.py`, `scripts/evaluate_mem_adapter.py` | Tách training condition F1 khỏi execution treatment. Thay các nhánh kiểm tên `F1` bằng capability phù hợp, kiểm đầy đủ factory, binder, detector, trace. |
| `src/training/mem_adapter_artifact.py`, `src/memory/relation_pipeline.py` | Thêm decision contract/hash ngoài writer contract. Nếu thay file thuộc writer source lock, tạo version receipt/index mới khi cần. Mọi phép tương thích phải qua kiểm parity. |
| `src/models/smolvla/adapter.py` | Chỉ ở D4 mới thêm sampling API. Candidate thứ nhất khớp đường K=1 với cùng noise/state. Cache và queue được audit từ code chính thức. |
| `src/evaluation/proposal_assessment.py` · mới, D3 | Giao diện bộ đọc đề xuất với unknown rõ ràng, hỗ trợ rule hoặc head có checkpoint provenance. |
| `src/evaluation/mem_eval_io.py` | Stream `decisions.jsonl`, candidate metadata/NPZ, fact versions và thời gian. Ghi final trạng thái đầy đủ khi lỗi hoặc abstain. |
| `scripts/audit_memory_decisions.py` · mới | So trace với nhãn độc lập, metric trigger/lead time/target/coverage và stratification theo failure. |
| `scripts/aggregate_memory_decisions.py` · mới | Ghép theo protocol/task/state/seed, kiểm code/checkpoint/hash, xử lý missing và bootstrap đúng cụm. Reuse phần kiểm provenance hiện có. |
| `scripts/intervene_memory_decisions.py` · mới | Fact-changing, fact-preserving, ID rename, exact-repeat, delayed evidence và null-event theo roster development. |

Kiểm thử cần bắt lỗi cụ thể gồm episode reset sạch, prefix bất biến khi thêm future evidence, invalidation chỉ lan theo dependency, unknown không bị đổi thành false, renaming ID nhất quán, late evidence không sửa quá khứ, semantic no-op không gây requery, một revision không gây vòng lặp vô hạn, observer không đổi RNG/policy output, candidate selection không cập nhật queue nhiều lần, policy seed độc lập với can thiệp và resume không làm mất lượt lỗi. Kiểm simulator xác nhận lịch evaluator không đổi khi thêm quan sát/logging. Real smoke mới chứng nhận đường model → controller → evaluator.

Các test quan hệ, rollout, CLI, adapter và supervisor kiểm từng tầng. Lệnh hiện hành và đầu vào của các script development nằm trong [bàn giao triển khai](MEMORY_DECISION_HANDOFF_2026-11-14.md).

**9. Ma trận đối chứng và phép đo**

| Arm mô tả | Đặc điểm cần khóa | Phép so trả lời |
|---|---|---|
| F1 text hiện tại | Cùng writer, checkpoint, prompt và execution horizon hiện hành | Mốc tham chiếu |
| F1 cùng observer thụ động | Tính supervisor và ghi trace, execution như F1 | Hiệu ứng instrument/logging và kiểm parity |
| F1 query theo lịch độc lập | Ngân sách query và runtime đối sánh với event schedule | Lợi ích từ số query so với chọn đúng thời điểm |
| F1 query theo evidence | Một chunk mỗi query, semantic trigger có version | Hiệu quả sử dụng memory tại thời điểm thực thi |
| F1 selector current-only | Cùng K ứng viên, prompt, parser, detector và bộ đọc đề xuất, selector chỉ đọc scene hiện tại | Lợi ích của quyền đọc lịch sử ở tầng lựa chọn |
| F1 selector lịch sử và validity | Cùng K, thêm fact cùng dependency hợp lệ | Giá trị của lịch sử tường minh và uncertainty |
| F1 selector bỏ validity | Dùng cùng fact nhưng thay đổi logic xử lý stale/conflict/unknown một cách có version | Tác động thuật toán của uncertainty |

Trong so sánh current-only selector, policy F1 vẫn có text history, vì mục tiêu là đo lịch sử bổ sung tại selector. So sánh tác dụng của toàn bộ history thuộc F1–F0 và các đối chứng của P6. Mỗi bảng phải ghi rõ tập thông tin đến từng module.

Một supervisor đọc toàn snapshot có nhiều dữ kiện hơn prompt 48 token. Cần hai phép kiểm. Trong phép kiểm quyền thực thi, supervisor chỉ đọc semantic fact và qualifier đã tới policy, cùng dependency metadata được khai báo. Trong phép kiểm hệ thống, supervisor đọc thêm snapshot và phải báo số fact/evidence bổ sung. Chi phí input tính cả token, object slots, detector calls, proposal count và VRAM của toàn hệ.

Offline cùng observation có thể dùng đúng một tập chunk đã lưu để so selector. Closed-loop chỉ bảo đảm cùng state ban đầu và seed đã định, các đường chạy sẽ khác sau khi chọn khác. Tách RNG sampling policy, detector và intervention để số lần query bổ sung không vô tình thay toàn bộ chuỗi noise. Đối chứng lịch query dùng số lần và tổng chi phí được khống chế, báo chênh lệch thực đo. Replay lịch event sau khi đường hành động đã tách được ghi là phân tích phụ.

Các metric bắt buộc gồm task-macro success, wrong-object/sequence/extra-cycle errors tại cửa sổ quyết định, tỷ lệ không quyết định được, tỷ lệ abstain, trigger precision/recall và lead time tới bước cam kết. Báo query count, detector count, thời gian writer/supervisor/policy, wall-clock episode và jitter. Với K>1 báo tỷ lệ tập ứng viên chứa lựa chọn đúng và tỷ lệ selector chọn đúng khi đã có lựa chọn đúng. Nhãn đúng dựa trên audit ngoại tuyến và hậu quả quan sát được.

Bootstrap theo initial-state group, giữ task, mọi decision window và rollout seed trong cùng cụm. Báo kết quả conditional trên cửa sổ đạt được cùng tỷ lệ episode đã tới cửa sổ đó. Dùng cửa sổ được khóa theo ngữ nghĩa để tránh một arm có nhiều cửa sổ thuận lợi hơn. Mẫu số success gồm mọi lượt hợp lệ theo roster, kể cả abstain. Lỗi hạ tầng ghi riêng, giữ bounds khi thiếu lượt. Freeze protocol, thresholds và checkpoint trước unseen.

Fact-changing phải làm thay đúng thành phần quyết định theo logic task. Fact-preserving và exact-repeat đi cùng đường cài đặt. Thay dependency cần tính lại validity của các kết luận liên quan. Phiên bản chỉ xóa metadata đo auditability, còn phiên bản gây stale/unknown hợp lệ đo cơ chế điều khiển. Null events cần được kiểm là không tạo subgoal hay hoàn tất dwell. Các nguyên tắc này bám IMEP và chẩn đoán temporal shortcuts. [IMEP](https://arxiv.org/abs/2610.23372v1), [Memory Clock](https://arxiv.org/html/2611.14488v1).

**10. Dung lượng và compute**

Ổ D còn khoảng 15,74 GiB khi kiểm ngày 14/11. Dự toán RGB shards cho khóa author v2 trong roadmap là 11,70 GiB, cần đo lại khi chạy. D1/D2 dùng lại index/shards và checkpoint P5. Mở rộng metadata làm thay đổi source hash của renderer/writer cần receipt hoặc index version mới dù nội dung text giống nhau. Đo ảnh hưởng trước khi tạo lại dữ liệu hàng loạt.

Lưu decision trace dạng JSONL, fact IDs/version và các đoạn cần audit. Candidate chunk float32 có 50 × 7 giá trị, khoảng 1.400 byte mỗi ứng viên. Với K=4, 100 query và 40 episode, dữ liệu chunk thô khoảng 21,4 MiB, chưa tính metadata và các arm. Dung lượng RGB/feature cache phụ thuộc số cửa sổ chọn, kích thước và compression, đo trước khi mở rộng. Giữ cache có giới hạn và source references truy lại được.

T7 được tải từng nguồn chính thức sau kiểm free space và receipt. Chuẩn bị metadata/labels theo quỹ đạo, tránh nhân bản RGB riêng cho từng arm. T9/T10 chỉ mở sau khi nguồn, runtime và dung lượng phù hợp. Việc dọn dữ liệu thực hiện theo danh sách đã kiểm và authorization tương ứng của tác giả.

Training thêm bắt buộc là cặp P5 đã có trong kế hoạch. D1/D2 không cần cập nhật weights. D3 chỉ cần train head nếu đọc hiệu ứng bằng quy tắc không đủ. D4 tăng inference theo K và overhead, cần đo từng thành phần trước khi chốt budget. ROI conditioning cần adapter riêng và corpus version tương ứng. Các nhánh critic/world model có chi phí rollout/training lớn, được quyết định sau báo cáo D4.

**11. Công việc tác giả cần chuẩn bị hoặc chạy nặng**

| Công việc | Đầu vào và điều kiện | Kết quả cần giữ |
|---|---|---|
| Hoàn tất E0/E1 và P5 | Writer audit, competence development, index refined v2, inference environment | F0 V2 artifact tồn tại; alignment, strict reload và B0/F0 runtime smoke đã qua. F1 V2 vẫn thiếu nên cặp F0/F1 và memory competence chưa đóng. |
| Gán nhãn cửa sổ quyết định | Seen rollout, video/crop hoặc observation có thể phát lại, hướng dẫn D0 | Target entity, tiền điều kiện, bước cam kết, ambiguity, lỗi thực thi, nguồn và hash |
| Chạy D1/D2 development | F1 mới, protocol supervisor, output mới và ngân sách đã chọn | Trace decision/action, chi phí và failure taxonomy |
| Train bộ đọc đề xuất nếu cần | Nhãn train/calibration tách theo state, các chunk đúng/sai trên dữ liệu thật | Checkpoint/recipe, calibration, false accept và coverage |
| Tải và chạy T7 | Nguồn chính thức, evaluator Or và runtime qua smoke, competency task mới | Identity/relation audit và các lượt vòng kín có provenance |
| Chạy đối chứng D4/D6 | Cấu hình đã freeze trên seen, independent roster đủ dùng, dung lượng log | Paired rollouts, bootstrap, can thiệp và báo cáo theo RQ |

Agent có thể triển khai contract, logging, scheduler, tests, script tổng hợp và tài liệu nghiệm thu. Thời gian gán nhãn, GPU cho full indexing/training và chiến dịch nhiều episode cần được bố trí theo nguồn lực thực tế. Chỉ chạy lệnh của các script đã có trong [bàn giao P5–P6](P5_P6_HANDOFF_2026-11-13.md) ở thời điểm lập kế hoạch này.

**12. Điều kiện đóng góp được hỗ trợ**

Kết quả T1/T3 có thể hỗ trợ claim về phản ứng với tiến trình và evidence validity. Claim danh tính cần T7. Claim chứa vật qua che khuất cần T9/T10. Claim tái sử dụng giữa backbone cần kiểm một policy thứ hai và công khai phần adapter, rule, tham số học thêm.

Khi cơ chế chỉ làm giảm lỗi bằng cách tăng compute hoặc abstain, báo đúng đường đánh đổi. Nếu candidate recall thấp, ưu tiên policy/grounding. Nếu false accept cao, cải thiện bộ đọc và writer trước mở quyền điều phối sâu hơn. Nếu writer chỉ mất lịch sử mà quan sát mới không thể phân xử, ghi giới hạn khả quan sát của tác vụ.

Bản viết chương 3 sẽ trình bày state, validity, dependency và quy tắc quyết định, kèm giả thiết để suy ra tính nhất quán. Chương 4 báo cả mức thông tin đã tới module, tác động lên quyết định, kết quả điều khiển và compute. Tất cả hiệu quả của cơ chế mới trong tài liệu này đang là giả thuyết cần thí nghiệm.

## Cập nhật gate ngoài huấn luyện, 15/11/2026

- P5 split/index/shards T1/T3 đã qua alignment audit. Artifact F0 V2 no-memory 5.000 bước đã có; inference environment Python 3.12.3/CUDA 12.8 đã verify, B0/F0 runtime check qua 4/4 task-condition checks. Pilot E0 B0/F0 hoàn tất bốn episode (một state seen/task, 1.000 action/episode), 0/4 thành công, SCR 0, overshot 0. Đây là development custom, không phải paired memory evaluation.
- Raw/refined T7/T8/T9/T10 source receipts đã xác minh local SHA-256. Audit writer RGB trên demo 1 hoàn tất: T7 425 mẫu, T8 613, T9 356 và T10 309; cả bốn task có 0 mẫu memory, relation state `unknown` toàn bộ. T7 identity observations: 92 `tracked`, 648 `ambiguous`, 1.385 `unknown`; T10 có 927 identity observations `unknown`. Phép audit detector riêng đối chiếu đủ 425 frame hash với exo-box annotations của nguồn ở IoU 0,5: bowl P/R 0,893/0,747, plate 0,971/0,646. Đây là trạng thái writer trên bốn demo train, không phải accuracy identity/relation/containment hoặc nhãn blind độc lập; writer fidelity gate vẫn mở.
- Schema gán nhãn cửa sổ quyết định đã được viết trong `evidence/decision_annotation_guide.md`; chưa có label rows. D3 chưa được cấp quyền reject action trong runner. D4 chưa có candidate sampler với parity K=1/queue/cache/RNG đã chứng minh.
- D5 evaluator Or T7 đã được replay với RGB trên action tác giả ở hai nhánh. Một author-action replay/task T8/T9/T10 đã dùng helper/state/action/evaluator upstream qua hash-checked source: T9/T10 `demo_1` đạt terminal return/done; T8 ghi ba milestone nhưng không terminal dù refined metainfo ghi success cho cùng state. Audit bổ sung xác nhận raw/refined T8 action và initial state trùng chính xác; robot-state replay vượt dung sai `1e-5` ở bước 219 (tối đa 0,008815). Nguyên nhân chưa xác định. Đây là một demo/task, không có nhãn độc lập hoặc policy rollout; runtime/evaluator gate rộng hơn và writer fidelity vẫn mở.
- Không chạy model training theo phạm vi người dùng. F1 V2 chưa có; D1/D2 closed-loop, E2 paired policy, E3–E5 policy interventions, và D6 risk–coverage/cost giữ trạng thái chưa nghiệm thu. Không coi B0/F0 pilot hoặc checkpoint F1 writer cũ là giá trị thay thế cho phép so sánh memory.
- Audit `experiments/audits/smolvla-image-preprocessing-parity-20261115/report.json` không thấy lệch hướng ảnh train/eval ở transform đã khai báo: canonical raw OpenGL ở train và raw simulator ở eval cùng đi qua `LiberoProcessorStep` LeRobot 0.6.1. F0 lưu version package chứ không lưu source hash phía training, nên parity được xác nhận theo version/call path; không đổi preprocessing và không gán nguyên nhân E0 cho hướng ảnh.
