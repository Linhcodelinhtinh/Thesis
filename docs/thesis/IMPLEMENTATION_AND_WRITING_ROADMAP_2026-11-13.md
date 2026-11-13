# Kế hoạch thực thi: bản viết, phương pháp và thực nghiệm

Ngày 13/11/2026. Đi kèm [định vị nghiên cứu và bộ RQ](RESEARCH_REFRAME_2026-11-13.md). Đây là backlog nội bộ: tên module/đường dẫn dùng để giao việc và review code, không được chép vào nội dung luận văn hoặc phụ lục.

Phạm vi của lượt làm việc hiện tại là phân tích và lập kế hoạch. Chưa sửa code, chưa chạy training/benchmark, chưa chỉnh bản LaTeX. V1 đã khóa vẫn được giữ làm baseline lịch sử; toàn bộ phần memory của kế hoạch thuộc V2 theo yêu cầu nghiên cứu hiện tại.

## 1. Hiện trạng đã đối chiếu và hệ quả

| Thành phần | Hiện trạng thấy trong nguồn/tập bằng chứng | Việc tiếp theo |
|---|---|---|
| Bản viết | Các chương 1, 2, 4 và abstract có chỉnh sửa chưa commit | Bảo toàn chỉnh sửa; không sinh lại từ Markdown cũ |
| Nguồn nội dung | README coi Markdown là nguồn chính; LaTeX đang có biên tập riêng | Hợp nhất có kiểm soát và chọn một nguồn chuẩn, tránh mất nội dung |
| Generator LaTeX | `build_latex.py` ghi lại abstract và cả năm chương | Tách regenerate khỏi compile; mặc định không ghi abstract |
| Validator thesis | `validate_draft.py` khóa số liệu pilot cụ thể như 40 summaries, 12 events, 19 unknown | Chuyển sang kiểm theo evidence manifest/schema; giữ audit lịch sử riêng |
| Core memory | Có entity, relation, evidence IDs, timestamps, snapshot bất biến | Mở rộng ngữ nghĩa uncertainty/relation; tránh tạo stack song song không tương thích |
| Writer | `LiberoMemEventWriter` và tracker gắn với một bát–đĩa | Tách primitive quan hệ khỏi cycle reducer; xây multi-object và containment |
| Renderer | `LiberoMemTextRenderer` chủ yếu chiếu cycle count; câu chữ dùng plate | Tạo projection chung và profile từ ngữ; bảo toàn qualifier, identity và budget |
| Runner | Có nhãn F0/O1 ở lớp runner nhưng CLI chỉ B0/F1; chunk bị kiểm cố định `(50, 7)` | Xây capability/condition contract xuyên suốt CLI–artifact–policy–runner |
| Adapter | Runtime guard ràng buộc writer, detector và renderer của F1 | Thêm các mode hợp lệ theo manifest; không bỏ guard để làm ablation |
| Protocol | Chỉ T1/T3, custom horizon/termination | Audit đầy đủ task map/BDDL/evaluator cho T1–T10 |
| Dữ liệu derived | Audit đã ghi source được sửa sau snapshot; chuỗi export thật chưa được chứng nhận | Kiểm export/resume/terminal/ảnh trên simulator thật trước train |
| Kết quả | B0 0/20; F1 1/20; F1 chưa có đối chứng thích ứng tương đương | Giữ là pilot, không coi 5 điểm phần trăm là causal effect |
| Chất lượng memory | Sáu quỹ đạo mẫu khớp 12 mốc trong tolerance; 19/20 F1 có lịch sử cuối chưa đầy đủ | Cần nhãn độc lập trên quỹ đạo do policy tự sinh và failure cases |
| Chi phí | Audit writer live khoảng 999–1.058 ms/frame trên phần cứng đã báo | Đo lại toàn hệ; không gọi 20 Hz là throughput thực |
| Backbone thứ hai | Có code MiniVLA và manifest ứng viên, chưa có kết quả được nhận vào thesis | Audit official contract và real smoke trước kế hoạch so sánh |

Các số liệu trên tham chiếu `STATUS_AND_CLAIMS.md` và các bảng evidence ngày 12/11; lần lập kế hoạch này chưa xác minh lại toàn bộ weights, HDF5 và từng raw artifact. Các đoạn “implementation update” mới hơn phải được ưu tiên hơn blocker lịch sử khi review.

## 2. Thứ tự phụ thuộc

`P0 bảo toàn bản viết + chốt protocol/RQ → P1 runtime/data → P2 condition harness + P3 core memory → P4 full-suite perception/projection → P5 dữ liệu/train đối sánh → P6 thực nghiệm xác nhận → P7 tổng hợp/bản viết cuối`.

Sửa chương 1–2 và cấu trúc chương 3 có thể tiến hành trong khi hoàn thiện code. Chương 4–5 chỉ chốt sau khi đóng bằng chứng. Không huấn luyện toàn bộ ma trận khi còn lỗi data contract hoặc phần lớn failure xảy ra trước quyết định cần memory.

## 3. P0 — Bảo toàn nội dung và chốt thiết kế nghiên cứu

**Đầu ra:** nguồn viết thống nhất, abstract được bảo vệ, bảng RQ–hypothesis–metric–experiment và protocol có version.

Việc bản viết:

- Lưu bản biên tập hiện tại bằng commit/checkpoint công việc theo quy trình người dùng; giữ các thay đổi có sẵn, không dùng reset/checkout đè lên.
- Dùng diff giữa LaTeX và Markdown để thu nhận các biên tập thủ công. Chọn LaTeX làm nguồn biên tập chính cho phiên bản hiện tại, hoặc đồng bộ lại Markdown trước khi cho phép generator chạy. Không áp dụng một quy tắc “Markdown là nguồn” cũ để ghi đè bản mới.
- Tách công cụ sinh nội dung khỏi công cụ compile PDF. Abstract do tác giả sở hữu: mặc định loại khỏi đường regenerate, kiểm hash trước/sau mọi build.
- Sửa validator để kiểm citation, label, cross-reference, evidence linkage và trạng thái claims; không khóa số liệu pilot thành điều kiện vĩnh viễn của một luận văn hợp lệ.
- Kiểm `references.bib`, caption, bảng và phụ lục để loại nguồn chỉ dẫn đến file local. Thay bằng bảng tự đủ, publication/version công khai, hoặc định danh nghiên cứu được định nghĩa trong thesis.

Việc thiết kế:

- Chốt ba RQ trong tài liệu định vị; chuyển retention và validity thành tiêu chí xuyên suốt.
- Đăng ký primary contrast F1–F0, contrasts về perception và representation, metric quyết định, CI, xử lý abort, budget và retention margin.
- Phân loại dữ liệu đã xem thành development. Chọn holdout còn hợp lệ từ nguồn chính thức; giữ quan hệ nhóm và nguồn cha. Không chuyển các state thất bại sang train rồi gọi cùng test là độc lập.
- Lập resource lock đầy đủ task map, BDDL, init state/dataset, runtime, checkpoint, tokenizer, preprocessing và action decoder theo từng backbone.

**Nghiệm thu:** chạy công cụ viết trên bản thử mà abstract không đổi byte; không mất các biên tập đang có; mọi claim dự kiến có phép kiểm và nguồn dữ liệu; V1/V2 cùng official/custom được tách rõ.

## 4. P1 — Runtime và dữ liệu thực sự đáng dùng

**Đầu ra:** một chuỗi real data → preprocess → policy → controller → evaluator được xác minh; manifest dữ liệu đủ để train đối sánh.

| Gói việc | Module liên quan | Kiểm chứng và điều kiện nghiệm thu |
|---|---|---|
| Protocol toàn suite | `src/evaluation/libero_mem_protocol.py`, `src/simulator/libero_mem_loader.py`, cấu hình benchmark mới có version | Mười task resolve từ canonical name/index/BDDL hash; test Sequence/Or, reset, warmup, overshoot và call schedule đúng source |
| Runtime parity | `src/simulator/libero_mem_env.py`, bridge và simulator worker | Camera/controller/robot đúng contract; trace một trajectory thật qua từng họ task; không gọi evaluator thêm để log |
| Observation/action alignment | HDF5 loader, `src/training/mixed_sft.py`, các converter | Đối chiếu index trên dữ liệu thật, raw/refined ảnh đúng một lần; state/action đúng source; không tự điền trường thiếu |
| Derived pipeline | Generator, split lock, preparation | Export thật có parent/source/runtime/code hashes; không tiếp tục step sau terminal; resume không mất failure; không publish output một phần |
| Validation export | `scripts/validate_memory_sft_export.py`, validation dữ liệu mixed | Roster đủ; hashes, timestamps, image convention và sample parity đúng; validator đọc artifact thực chứ không chỉ fixture |

Dùng V2 environment riêng; không cài dependencies vào V1 reference để tiện chạy. Mọi khác biệt so runtime tác giả phải khai báo. Không thay vật, mesh, controller hoặc init state bằng tương đương thị giác.

Với dữ liệu dẫn xuất, giữ hai định danh: nguồn tác giả và bản replay local. Quỹ đạo replay thất bại không còn là action target chuyên gia hợp lệ chỉ vì action gốc tốt. Có thể dùng failure để đánh giá/huấn luyện nhận diện evidence nếu nhãn hợp lệ; không trộn nó thành imitation target thành công. Ghi tỷ lệ loại theo task và nguyên nhân; không sửa evaluation roster theo success của replay/policy.

**Gate năng lực:** trước khi mở rộng training, cần policy thực làm được thao tác cơ sở trên development và xuất hiện các điểm quyết định cần memory. Ngưỡng competence và số lượt được chốt sau pilot chi phí, trước confirmatory; nếu quá thấp, ưu tiên image/action/data/runtime diagnostics. Không kết luận mọi failure là “quên”.

## 5. P2 — Harness điều kiện thực nghiệm và backbone

**Đầu ra:** các điều kiện trong RQ chạy được bằng cùng đường điều khiển, manifest ghi đầy đủ treatment.

Định nghĩa một condition spec tách `weight_identity`, nguồn memory, kiểu projection, can thiệp, ngân sách và quyền đọc oracle. Runner đọc spec; artifact validator kiểm spec phù hợp training. F0/F1-off/C0/L0 phải là mode được hỗ trợ có chủ đích, không phải sửa cờ để qua guard.

Các vị trí cần review: `scripts/evaluate_mem_adapter.py`, `scripts/mem_eval_runtime.py`, `src/evaluation/libero_mem_rollout.py`, `src/training/mem_adapter_artifact.py`, `src/models/smolvla/adapter.py`, `src/models/base.py` và registry.

Thay assumption `(50,7)` ở tầng dùng chung bằng contract đã audit của policy và controller. Không có nghĩa tự đổi action dimension hoặc chunk của model. Model sở hữu preprocessing, decoder, normalization và chunk; controller adapter chỉ thực hiện mapping được xác minh. Execution schedule khác giữa backbone phải được báo và đối sánh theo thời gian điều khiển nơi khả thi, không ép một model vào contract của model kia.

MiniVLA là ứng viên thứ hai vì repo có sẵn hướng tích hợp; chưa coi là sẵn chạy. Cần kiểm official config/code cho action polarity, tokenizer/VQ, chunk và camera/state. Chú thích dấu gripper ở candidate config hiện cần được đối chiếu với contract simulator trước sử dụng; không sửa bằng suy đoán. Nếu checkpoint/interface không xác định, dừng riêng nhánh này và báo blocker; không dùng fallback.

**Kiểm chứng cần có:** same-condition repeat, đúng weights cho từng arm, reset sạch mọi state/queue, không lọt oracle, F1-off không gọi nhầm B0, artifact reload khớp prediction trong tolerance đã định. Test với fixture chỉ chứng minh plumbing; mọi condition được nhận vào bảng phải chạy thật với checkpoint.

## 6. P3 — Lõi memory và evidence

**Đầu ra:** schema trạng thái chung và primitive cập nhật được dùng cho nhiều họ task.

Tái sử dụng `src/memory/models.py`, event writer, store/updater/retriever sau khi kiểm đường nào thực sự được rollout dùng. Không xây thêm một API đẹp nhưng bỏ ngoài đường chạy.

Các thay đổi trọng tâm:

- Tách cycle reducer khỏi perception bát/đĩa. Một chuỗi chuyển quan hệ tạo cycle; cùng reducer dùng cho đối tượng khác sau khi primitive được audit.
- Lưu identity, origin, current relation, visibility và event history riêng; bằng chứng tham chiếu tới camera/frame/model version thực.
- Thêm uncertainty theo fact, invalidation của quan hệ hiện tại và conflict handling. Không đặt một `history_complete=False` toàn episode làm ngữ nghĩa duy nhất của mọi fact.
- Xác định trạng thái candidate/confirmed/unknown/conflicted. Score chưa calibration không được trình bày như probability.
- Mỗi fact chiếu được phải có dependency closure và thời điểm khả dụng; mất evidence hỗ trợ phải dẫn tới đúng thay đổi validity.
- Phân biệt audit log tích lũy với working state hữu hạn. Đo tăng storage theo sự kiện nếu vẫn giữ log không giới hạn.

**Kiểm thử có giá trị:** không đếm trùng khi đứng yên; không tiến độ từ lệnh chưa thực thi; một khoảng trống không sinh sự kiện; cùng tiền tố cho cùng snapshot; replay duplicate idempotent; event không sửa quá khứ đã phục vụ policy; evidence tương lai/khác hash/khác camera/nguồn oracle bị từ chối. Kiểm persistence và invalidation bằng các trace thật đã chú giải. Invariant pass không thay thế đánh giá perception.

## 7. P4 — Toàn bộ LIBERO-Mem và projection trong ngân sách

**Đầu ra:** một lõi dùng chung với các primitive/profile được ghi rõ, hỗ trợ cả mười tác vụ.

| Chặng | Công việc thực chất | Gate trước policy evaluation |
|---|---|---|
| A: T1–T6 | Generalize support departure/return cho bát/chai; bỏ phụ thuộc target count trong writer; kiểm đếm dài | Precision/recall/timing trên state chưa dùng tune, cả failure và boundary cases; đúng mục tiêu số lần theo instruction |
| B: T7/T8 | Multi-instance tracking, origin binding, identity qua giao cắt, relation transitions và grounding tên trong prompt | Phân biệt được các entity bằng evidence; ambiguous được giữ; báo ID switches và relation accuracy |
| C: T9/T10 | Containment qua quá trình đưa vật vào giỏ, continuity của container, evidence về giỏ chưa chứa/rỗng | Không đồng nhất invisible với absent; nhãn độc lập; target-object và container binding đúng qua che khuất |

Các module tri giác hiện có gồm `src/perception/object_tracker.py`, `visual_evidence.py`, detector và `src/memory/visual_pipeline.py`. Những primitive mới cần source riêng có phạm vi rõ; không đổi tên bowl thành object rồi coi đã tổng quát.

Thêm interface projection dùng chung trên `src/interfaces/`, giữ `LiberoMemTextRenderer` hiện tại làm profile tương thích để tái lập pilot. Projection mới phải:

- Chọn fact theo instruction và state đã quan sát; không gọi evaluator hay đọc nhãn tương lai.
- Chọn/bỏ nguyên nhóm fact–identity–uncertainty.
- Kiểm tokenizer chính thức của từng model, toàn prompt kể special tokens/newline; giữ nguyên instruction.
- Báo facts considered/included/dropped, dependency failure và budget exhaustion tại từng query.
- Có baseline chronological/recent projection cùng nguồn thông tin.
- Xác minh prompt tạo offline từ prefix và prompt online tại cùng prefix có cùng nội dung khi cấu hình/ngẫu nhiên bằng nhau.

Trước huấn luyện, thống kê instruction length của cả mười task. Nếu một task gần hết 48 token, không được cắt lệnh hoặc tăng context âm thầm. Đó là điều kiện của RQ1 cần báo rõ. Nếu thử một cách encoding ngắn khác, khóa recipe, huấn luyện tương ứng và giữ đối chứng đủ công bằng.

**Tối ưu latency:** đo từng stage rồi chọn detector cadence/ROI reuse/caching giữa frame theo evidence thật. Không coi nội suy hoặc tracker prediction là detection mới. Giảm tần suất phải đo lại missed-event, delay và quyết định; không chỉ throughput. Triển khai async chỉ sau khi có watermark thời điểm khả dụng và xử lý kết quả cũ/đến sai thứ tự.

## 8. P5 — Dữ liệu và huấn luyện đối sánh

**Đầu ra:** manifest và checkpoint cho F0/C0/L0/F1, có training provenance, dùng cùng corpus và split.

Các vị trí: `src/training/libero_mem_data.py`, `memory_dataset.py`, `mixed_sft.py`, `split_lock.py`, `replay_sampler.py`, các script prepare/validate/train/export.

Chốt một corpus đã được validation; tạo các **view thông tin** trên cùng sample indices, images, states và action targets. F1 có memory prefix, C0 chỉ fact hiện tại, L0 lịch sử phẳng, F0 chỉ instruction. So checksum sample roster trước chạy. Không thêm dữ liệu chất lượng cao riêng cho F1 rồi quy gain cho representation.

Kiểm train/validation/test ở cấp initial-state group và lineage nguồn. Nếu writer dùng demo để tune threshold, nhóm đó không được nằm ở confirmatory. Dữ liệu pretrained có thể không hoàn toàn kiểm soát; khai báo giới hạn thay vì khẳng định chưa từng thấy.

Training matched: cùng checkpoint xuất phát, số bước, sampler, base replay fraction, augmentations, effective batch, số tham số được mở, optimizer và tiêu chí checkpoint. Với backbone khác, so within-backbone trước; không gọi các backbone khác capacity là compute-identical chỉ vì số bước bằng nhau. Giữ tuning budget đối sánh.

Smoke trên dataset thật nhỏ: load/export/reload, loss hữu hạn, memory facts causal và adapter tạo action đúng contract. Sau smoke mới train đủ budget. Memory dropout nếu dùng phải có seed/rate trong manifest và được đối sánh giữa memory arms. Không mặc định tăng budget đến khi đạt số mong muốn.

Đánh giá retention chạy chính các adapter trên task cơ sở. Đo B0 và F0 để phân biệt tác động thích ứng chung với tác động recipe memory. Không chọn nhánh B0 qua router để báo “không quên”.

## 9. P6 — Các chiến dịch thực nghiệm

| Chiến dịch | Mục tiêu/RQ | Đầu ra bắt buộc |
|---|---|---|
| E0: competence và runtime | Loại measurement confound | Replay thật, action/image audit, success primitive và blocker list |
| E1: writer trên dữ liệu độc lập | RQ2 | Event/identity/relation metrics, delay, unknown; failure labels |
| E2: full-suite matched evaluation | RQ1, RQ3 | B0 cùng F0/C0/L0/F1; per-task counts, macro SR, paired CI |
| E3: budget sweep | RQ1 | Hiệu năng theo budget, decision-fact survival, token/padding/chi phí |
| E4: evidence/noise/delay ablation | RQ2 | Risk–coverage, false-confirmation, lỗi hành vi, latency |
| E5: can thiệp thông tin | RQ1, RQ2 | Fact-changing, fact-preserving và exact-repeat cùng installation path; delivery audit |
| E6: backbone thứ hai + retention | RQ3 | Within-backbone contrasts, phần tích hợp thay đổi, kỹ năng cơ sở và chi phí |

E2 là bảng chính. E3–E5 có thể bắt đầu bằng subset đại diện được chọn trước theo loại memory, sau đó mở rộng khi có tương tác đáng kiểm chứng. Không cần full factorial mọi budget × noise × backbone × seed ngay từ đầu. Tập subset chẩn đoán không thay coverage toàn suite của E2.

Thiết kế can thiệp:

- Null event: thêm khoảng chờ hoặc chuyển động không tạo tiến độ theo task; kiểm chính intervention không đổi success condition hoặc đưa hệ ra ngoài protocol. Các thí nghiệm can thiệp mang nhãn custom.
- Fact-changing: đổi count/entity/container cụ thể; đảm bảo cấu trúc và token budget hợp lệ.
- Fact-preserving: đổi cách diễn đạt hoặc thứ tự các fact độc lập trong khi giữ nghĩa cần kiểm; không hoán đổi order của event nếu order chính là thông tin.
- Exact-repeat: thực hiện cùng đường cài đặt nhưng giữ nguyên nội dung để phát hiện hiệu ứng do can thiệp phần mềm.
- Delay: giữ nội dung, trì hoãn availability có kiểm soát; không chuyển timestamp để giả vờ có evidence sớm hơn.
- Current-only: bỏ lịch sử nhưng giữ detector/current fact, kiểm phần này không ngầm chứa counter/cached history.

Nhãn oracle chỉ đi vào chiến dịch chẩn đoán riêng. Không đưa số O-diag vào bảng thành tích predicted memory. Không coi failed oracle diagnostic tự động chứng minh memory không hữu ích: trước hết kiểm khả năng đọc định dạng và năng lực vận động.

## 10. P7 — Báo cáo và sửa luận văn

**Đầu ra:** tập bảng/hình tự đủ, bản viết theo RQ, nguồn công khai và evidence ledger nội bộ.

Bảng/hình dự kiến:

1. Sơ đồ kiến trúc tách observation, memory, projection, policy và evaluator.
2. Một timeline cho mỗi họ memory, có occurred/available/consumed time và uncertainty.
3. Bảng full-suite SR + counts + CI; aggregate theo family và macro task.
4. Đường budget–decision accuracy–SR cùng dữ kiện bị bỏ.
5. Risk–coverage và latency–performance; tránh nhiều trục khó đọc nếu có thể tách panel.
6. Bảng đối chứng thông tin/can thiệp và tỷ lệ delivery thực.
7. Bảng retention, chi phí toàn hệ và mức tái sử dụng giữa backbone.
8. Các case study được chọn theo taxonomy trước, có cả success lẫn failure; không chỉ chọn video đẹp.

Checklist sửa bản viết:

- Chương 1: đổi mục tiêu/RQ/phạm vi; thêm constraint của model nhỏ; giữ model và benchmark ở vai trò thực nghiệm.
- Chương 2: thay mục 2.3 lặp gap bằng nền tảng trừu tượng hóa lịch sử; thêm SmoLSTM, IMEP, Clock, reward-machine connections; kiểm claims của từng nguồn.
- Chương 3: viết từ schema và giả thiết chung, cho thấy uncertainty/projection là thuật toán; chuyển chi tiết bát–đĩa/config sang phần hiện thực.
- Chương 4: tái bố trí theo RQ; tách pilot cũ với confirmatory mới; giữ mẫu số, nguồn và labels custom rõ.
- Chương 5: trả lời RQ theo evidence, có effect size/uncertainty/giới hạn; không lặp danh sách module.
- Phụ lục/bibliography/caption: loại mọi tham chiếu đến file local, thêm bảng protocol và nguồn public thật; không xóa provenance mà chuyển nó sang dạng độc giả truy cập được.
- Từ ngữ: rà “trình diễn”, “demonstration”, “ground truth”, “chứng minh”, “tổng quát”, “nhẹ”, “thời gian thực”; dùng chính xác theo nguồn và evidence.
- Abstract: không sửa. Khi chốt bản nộp, báo tác giả nếu kết quả chưa đủ cho một phát biểu đã có.

Nghiệm thu bản viết: citation/label đầy đủ; không có local path trong rendered thesis; đọc liên tục mở đầu và kết luận để kiểm flow; compile và render PDF rà bảng, phương trình, hình, phân trang. Không tuyên bố visual QA nếu mới chạy kiểm text hoặc compile.

## 11. Ước lượng tiến độ theo mốc

Đây là ước lượng sơ bộ cho một người làm chính, phụ thuộc độ khó tri giác và thời gian GPU; không phải cam kết lịch. Chốt lại sau E0 và đo chi phí một episode. Kế hoạch đầy đủ nhiều khả năng cần nhiều tuần, đặc biệt identity/containment và backbone thứ hai.

| Mốc | Effort kỹ thuật/biên tập dự kiến, chưa gồm toàn bộ compute | Kết quả để chuyển mốc |
|---|---|---|
| M0 | 1–2 ngày | P0, RQ/protocol/scope lock, nguồn viết bảo toàn |
| M1 | 3–5 ngày | P1, real export và runtime/data contract |
| M2 | 4–7 ngày | P2/P3, điều kiện matched và memory core |
| M3 | 1–2 tuần hoặc hơn nếu identity khó | P4, T1–T10 và projection qua audit |
| M4 | 3–5 ngày chuẩn bị + compute đo thực | P5, checkpoint matched và smoke |
| M5 | 1–2 tuần điều phối/phân tích + compute | P6, main evaluation và thí nghiệm cơ chế |
| M6 | 4–7 ngày | P7, bản viết và PDF hoàn chỉnh |

Ưu tiên giữ nếu phải giảm ngân sách: baseline đối sánh, information controls, coverage cả mười task trên model chính, test chất lượng evidence, và kết luận trung thực. Có thể giảm full factorial, số biến thể renderer hoặc benchmark thứ hai trước. Không tự bỏ backbone thứ hai rồi giữ nguyên claim generality; nếu không làm kịp phải giới hạn claim đó.

## 12. Định nghĩa hoàn thành

- Các RQ có hypothesis, metric, đối chứng, điều kiện bác bỏ và kết luận tương ứng.
- Full suite được resolve từ nguồn chính thức; không có task thay thế, asset giả hoặc policy fallback.
- Mọi kết quả primary đến từ real rollout có identity và protocol; custom results được ghi đúng phạm vi.
- Core method có logic dùng chung thật; evidence/uncertainty tham gia quyết định cập nhật và projection.
- Gain được đối chiếu adaptation, perception và representation; dùng CI đúng đơn vị cụm và có holdout hợp lệ.
- Cost bao gồm mọi thành phần online; retention chạy đúng adapter.
- Chương phương pháp độc lập với tên model/benchmark; chương thực nghiệm giải thích cụ thể các lựa chọn đó.
- Thesis và phụ lục đọc được mà không cần truy cập máy tác giả; abstract giữ nguyên theo yêu cầu.

Không một mục nào được đánh dấu hoàn thành chỉ vì đã có tên class, enum condition, tài liệu thiết kế hoặc unit test pass.
