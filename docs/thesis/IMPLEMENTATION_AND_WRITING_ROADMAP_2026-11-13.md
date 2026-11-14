# Kế hoạch thực thi: bản viết, phương pháp và thực nghiệm

Ngày 13/11/2026. Đi kèm [định vị nghiên cứu và bộ RQ](RESEARCH_REFRAME_2026-11-13.md). Đây là backlog nội bộ: tên module/đường dẫn dùng để giao việc và review code, không được chép vào nội dung luận văn hoặc phụ lục.

Trạng thái sau khi chuẩn bị P5–P6 ngày 13/11/2026. V1 là baseline lịch sử đã khóa. Các thay đổi về memory, dữ liệu đối sánh và đánh giá trong kế hoạch này thuộc V2. Lệnh theo thứ tự phụ thuộc và vị trí artifact nằm trong [bàn giao P5–P6](P5_P6_HANDOFF_2026-11-13.md).

## 1. Hiện trạng đã đối chiếu và hệ quả

| Thành phần | Hiện trạng thấy trong nguồn/tập bằng chứng | Việc tiếp theo |
|---|---|---|
| P0 · Bản viết và protocol | LaTeX là nguồn biên tập chính. Đường build mặc định bảo toàn abstract. Protocol phân tích có version, roster xác nhận còn trống. | Khóa holdout và tiêu chí phân tích trước khi xem kết quả xác nhận. |
| P1 · Dữ liệu và runtime | Khóa v2 xác minh 198 demo Mem refined và 500 demo LIBERO cơ sở. Mỗi roster raw seen/unseen có 20 episode/task. Giao thức vòng kín T1/T3 mang nhãn `NON-COMPARABLE_OFFICIAL_PAPER`. | Audit alignment sau khi tạo shards. Dùng seen để phát triển và khóa checkpoint, seed, phương pháp phân tích trước lượt unseen. |
| P2 · Điều kiện và adapter | CLI có F0 và F1-off. Smoke thật hai action đã chạy B0/F1/F1-off trên T1/T3. Export lịch sử 5.000 bước qua kiểm artifact và dùng `rgb_cycle_v3`. | Chạy smoke với F0/F1 mới; đóng competence gate bằng lượt hoàn tất. |
| P3–P4 · Memory và projection | Writer `rgb_relations_v1` và profile có cho cả mười task. T1 GPU ghi 1 lần đặt lại, T3 từ cache ghi 2/3, parity IPC 4 frame T1. T2/T4 có HDF5 đã kiểm hash. | Audit writer T3/T2/T4 trên GPU với mốc độc lập. Mở rộng theo họ task khi có dữ liệu và evaluator. |
| P5 · Dung lượng | Ổ D còn 16,55 GiB lúc đo. Lấy 96 frame mỗi nguồn của khóa v2 ước tính RGB trong shards cần 11,70 GiB với gzip lossless. | Dùng chung một bộ shards. Đo lại sau khi tạo index và ghi checkpoint/ZIP sang volume đủ chỗ. |
| P6 · Đánh giá | B0 đã qua smoke hai action thật trên raw seen của T1 và T3 theo khóa v2. Chưa có cặp checkpoint mới. Pilot lịch sử B0 0/20 và F1 1/20 vẫn là mốc chẩn đoán. | Chạy E0/E1 trước lô chính. Báo bounds của lượt thiếu và CI theo cụm khi đã kiểm phương pháp phân tích. |
| Backbone thứ hai | Có code MiniVLA và manifest ứng viên. | Kiểm contract chính thức, gồm gripper, rồi smoke thật trước khi đưa vào so sánh. |

Các mốc thực thi trước nằm trong [bàn giao P0–P2](P0_P2_HANDOFF_2026-11-13.md), [bàn giao P3–P4](P3_P4_HANDOFF_2026-11-13.md) và [sổ claim lịch sử](STATUS_AND_CLAIMS.md).

### 1.1. Giao thức split V2

Refiner tác giả duyệt tối đa 100 vị trí đầu sau natural sort và chỉ ghi replay thành công. Hai evaluator lấy 20 vị trí đầu cho seen và 20 vị trí cuối cho unseen khi chạy 20 lượt. T1 có 100 demo refined. T3 có 98, thiếu `demo_48` và `demo_81`. Cả 198 demo đều khớp hash state ở đúng vị trí raw. [Refiner](https://github.com/libero-mem/libero-mem/blob/0eee0defa4024e51511b6e15a79f30e9620209de/scripts/mem_3_refine_demonstration.py#L748-L768), [evaluator GT](https://github.com/libero-mem/libero-mem/blob/0eee0defa4024e51511b6e15a79f30e9620209de/scripts/mem_5_run_evaluation_env_gt.py#L755-L783), [evaluator Pred](https://github.com/libero-mem/libero-mem/blob/0eee0defa4024e51511b6e15a79f30e9620209de/scripts/mem_6_run_evaluation_env_pred.py#L1030-L1058).

| Mục đích | T1 | T3 | LIBERO-10 cơ sở |
|---|---:|---:|---:|
| SFT train | 100 refined | 98 refined | 400 replay |
| Offline flow-loss validation | 0 | 0 | 50 replay |
| Base test đã khóa cho kiểm tra retention | 0 | 0 | 50 replay |
| Rollout seen, dùng cho development | 20 raw, vị trí 0–19 | 20 raw, vị trí 0–19 | — |
| Rollout unseen, dùng cho đánh giá đã khóa | 20 raw, vị trí 100–119 | 20 raw, vị trí 100–119 | — |

Khóa v2 lưu roster raw riêng khỏi inventory SFT. Unseen có 20 cụm initial-state hash dùng chung giữa hai task. Hash của các cụm này không giao với 100 vị trí raw đầu hoặc các episode base. ID HDF5 là `demo_101`–`demo_120`. Các tập val/test của split Mem v1 thuộc development. Overlap với dữ liệu pretraining của checkpoint nền vẫn chưa được xác minh ở cấp state. [Khóa v2](../../resources/manifests/libero_mem_t1_t3_base10_split_v2.json), [protocol phân tích v2](evidence/mem_analysis_protocol_v2.json).

| Module bị ảnh hưởng | Thay đổi |
|---|---|
| Nguồn và split | `download_memory_training_sources.py` tải raw theo revision/SHA-256. `lock_memory_sft_split.py` xác minh raw/refined, chuyển toàn bộ demo refined về train và giữ assignment base. |
| Chỉ mục, shards và Kaggle | `prepare_libero_mem_sft.py`, loader HDF5, `prepare_mem_adapter_data.py`, `prepare_mixed_memory_sft.py`, validator và notebook dùng inventory train-only cho Mem. |
| Derived replay | Generator và lock derived nhận roster refined train-only từ khóa cha. Khi cần nhánh derived, ghi exclusions theo từng replay. |
| Training và export | F0/F1 đọc cùng 598 demo train. Validation offline chỉ đo base. Export ghi hash corpus và khóa dùng cho rollout. |
| Rollout và tổng hợp | Evaluator, simulator worker, launcher và aggregator đọc seen/unseen từ khóa v2, ghép cặp theo task/state/seed và gom cụm theo hash state. |

## 2. Thứ tự phụ thuộc

`Khóa v2 đã kiểm → E0/E1 development → index refined T1/T3 → shards và alignment audit → smoke train/reload F0/F1 → train hai arm từ SmolVLA gốc → seen ghép cặp → đóng băng protocol → E2 unseen → P7`.

E1 cho T2/T4 có thể chạy độc lập và đo khả năng chuyển từ bát sang chai. Khi đủ chỗ cho một HDF5, ưu tiên audit offline T7 rồi T10 để kiểm identity và containment. E2 xác nhận cần roster độc lập đã khóa và kế hoạch phân tích được duyệt. Đánh giá vòng kín ngoài T1/T3 cần evaluator và runtime tương ứng. Có thể sửa chương 1–3 song song; chương 4–5 chốt theo bằng chứng đã hoàn tất.

## 3. P0 — Bảo toàn nội dung và chốt thiết kế nghiên cứu

**Đầu ra:** nguồn viết thống nhất, abstract được bảo vệ, bảng RQ–hypothesis–metric–experiment và protocol có version.

Việc bản viết:

- Lưu bản biên tập hiện tại bằng commit/checkpoint công việc theo quy trình người dùng; giữ các thay đổi có sẵn, không dùng reset/checkout đè lên.
- Chọn LaTeX làm nguồn biên tập chính cho phiên bản hiện tại.
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

Thay assumption `(50,7)` ở tầng dùng chung bằng contract đã audit của policy và controller. Không có nghĩa là đổi action dimension hoặc chunk của model. Model sở hữu preprocessing, decoder, normalization và chunk; controller adapter chỉ thực hiện mapping được xác minh. Execution schedule khác giữa backbone phải được báo và đối sánh theo thời gian điều khiển nơi khả thi, không ép một model vào contract của model kia.

MiniVLA là ứng viên thứ hai vì repo có sẵn hướng tích hợp; chưa coi là sẵn chạy. Cần kiểm official config/code cho action polarity, tokenizer/VQ, chunk và camera/state. Chú thích dấu gripper ở candidate config hiện cần được đối chiếu với contract simulator trước sử dụng; không sửa bằng suy đoán. Nếu checkpoint/interface không xác định, dừng riêng nhánh này và báo blocker; không dùng fallback.

**Kiểm chứng cần có:** same-condition repeat, đúng weights cho từng arm, reset sạch mọi state/queue, không lọt oracle, F1-off không gọi nhầm B0, artifact reload khớp prediction trong tolerance đã định. Test với fixture chỉ chứng minh plumbing; mọi condition được nhận vào bảng phải chạy thật với checkpoint.

## 6. P3 — Lõi memory và evidence

**Đầu ra:** schema trạng thái chung và primitive cập nhật được dùng cho nhiều họ task.

Tái sử dụng `src/memory/models.py`, event writer, store/updater/retriever sau khi kiểm đường nào thực sự được rollout dùng. Không xây thêm một API đẹp nhưng bỏ ngoài đường chạy. Đảm bảo memory ghi đúng và chính xác, có thế được sử dụng trên nhiều task khác nhau trong LIBERO-MEM.

Các thay đổi trọng tâm:

- Tách cycle reducer khỏi perception bát/đĩa. Một chuỗi chuyển quan hệ tạo cycle; cùng reducer dùng cho đối tượng khác sau khi primitive được audit. Quan sát các tác vụ LIBERO-MEM để đưa ra quyết định hợp lý
- Lưu identity, origin, current relation, visibility và event history riêng; bằng chứng tham chiếu tới camera/frame/model version thực.
- Thêm uncertainty theo fact, invalidation của quan hệ hiện tại và conflict handling. Không đặt một `history_complete=False` toàn episode làm ngữ nghĩa duy nhất của mọi fact.
- Xác định trạng thái candidate/confirmed/unknown/conflicted. Score chưa calibration không được trình bày như probability.
- Mỗi fact chiếu được phải có dependency closure và thời điểm khả dụng; mất evidence hỗ trợ phải dẫn tới đúng thay đổi validity.
- Phân biệt audit log tích lũy với working state hữu hạn. Đo tăng storage theo sự kiện nếu vẫn giữ log không giới hạn.
- Vẫn đảm bảo ngân sách 48 tokens của smolvla. với minivla có ngân sách input token dư giả hơn, có thể thiết kế memory chi tiết hơn.

**Kiểm thử có giá trị:** không đếm trùng khi đứng yên; không tiến độ từ lệnh chưa thực thi; một khoảng trống không sinh sự kiện; cùng tiền tố cho cùng snapshot; replay duplicate idempotent; event không sửa quá khứ đã phục vụ policy; evidence tương lai/khác hash/khác camera/nguồn oracle bị từ chối. Kiểm persistence và invalidation bằng các trace thật đã chú giải. Invariant pass không thay thế đánh giá perception.

## 7. P4 — Toàn bộ LIBERO-Mem và projection trong ngân sách

**Đầu ra:** một lõi dùng chung với các primitive/profile được ghi rõ, hỗ trợ cả mười tác vụ.

LIBERO-Mem có tổng cộng 10 manipulation tasks, được chia thành 4 nhóm năng lực bộ nhớ (object-centric memory):
1. OM – Object Motion: Nhớ lịch sử chuyển động và trạng thái tương tác của từng object.
2. OS – Object Sequence: Nhớ số lần tương tác và thứ tự các thao tác.
3. OR – Object Relations: Nhớ quan hệ, vị trí và thứ tự tương tác giữa nhiều object.
4. OO – Object Occlusion: Nhớ danh tính và vị trí object khi chúng bị che khuất.

Danh sách 10 tasks
Task	Nhóm	Mô tả tác vụ	Memory cần thiết
T1	OM	Nhấc bowl lên rồi đặt lại lên plate	Nhớ trạng thái lift → place
T2	OM	Nhấc bottle lên rồi đặt xuống plate	Theo dõi trạng thái tương tác của bottle
T3	OM + OS	Nhấc bowl và đặt lại lên plate 3 lần	Đếm số lần pick-and-place
T4	OM + OS	Nhấc bottle và đặt xuống plate 3 lần	Đếm interaction history
T5	OM + OS	Nhấc bowl và đặt lại lên plate 5 lần	Long-term interaction counting
T6	OM + OS	Nhấc bowl và đặt lên plate 7 lần	Duy trì bộ đếm qua chuỗi dài
T7	OM + OR	Hoán đổi 2 bowls giữa các plates theo rotation rule	Theo dõi object identity và thứ tự di chuyển
T8	OM + OR	Hoán đổi 3 bowls giữa các plates theo rotation rule	Multi-object relational tracking
T9	OM + OO	Đặt bowl vào basket gần nhất, sau đó di chuyển basket chứa bowl vào giữa	Nhớ basket nào đang chứa bowl
T10	OM + OO	Đặt bowl vào basket gần nhất, sau đó di chuyển basket rỗng vào giữa	Phân biệt basket đã chứa bowl và basket chưa chứa bow
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

**Phạm vi triển khai theo dung lượng hiện tại:** T1/T3 cho vòng kín. T2/T4 đã có HDF5 và dùng cho audit offline; chỉ mục no-memory T4 đọc được 303 frame và xác nhận đường đọc nguồn. T5–T10 chưa có nguồn local. Khi có chỗ, dùng từng HDF5 chính thức và lưu hash, metadata, manifest, trace, nhãn và báo cáo trước khi chuyển nguồn sang ổ khác. Ưu tiên T7 rồi T10 để kiểm hai họ còn thiếu. Profile hoặc audit offline chỉ hỗ trợ claim ở cấp phương pháp/writer tương ứng. Để báo kết quả policy trên họ mới cần thêm khóa trạng thái, smoke runtime, evaluator đúng ngữ nghĩa và xử lý Or.

## 8. P5 — Dữ liệu và huấn luyện đối sánh

Mốc trước mắt dùng 198 refined demo của tác giả và 500 demo base đã khóa. F0/F1 dùng chung corpus và sampler, F1 dùng writer quan hệ. C0/L0 là đối chứng cơ chế tiếp theo. Chọn author-refined cho bảng chính vì derived có thêm exclusions do tiêu chí replay local.

| Gate | Trạng thái | Việc và minh chứng cần có |
|---|---|---|
| Năng lực và writer | Mở. Lượt competence cũ lỗi sau 5/40 episode. T3 cache ghi 2/3. | E0 cần lượt hoàn chỉnh và log đạt primitive/điểm quyết định. E1 cần audit GPU T1/T3 theo ngưỡng đã ghi. |
| Chỉ mục memory từ refined | Mở. Các index v1 có lineage khác. | Chạy `prepare_libero_mem_sft.py --source-kind author_refined` với writer quan hệ và khóa v2. Lưu manifest, hash, trace, coverage, overflow và provenance cho cả hai task. |
| Một bộ shards lossless | Smoke gzip trên 6 episode thật đã đối chiếu 1.219 hàng state/action, 18 frame RGB mỗi camera và 12 mẫu parity đầu vào adapter. Audit bộ đầy đủ có memory còn mở. | Đo lại với `estimate_mem_shard_storage.py`, chạy `prepare_mixed_memory_sft.py` dùng gzip với reserve. Kiểm đầy đủ bằng `audit_mixed_memory_sft.py --policy-parity`; lưu hash roster và báo cáo. |
| Training và reload | Mở. `.venv_training` thiếu các dependencies của recipe. Export 5k lịch sử dùng writer cũ. | Chuẩn bị môi trường theo `envs/mem_adapter_training.txt`. Smoke 5 bước cả hai arm bằng `train_mem_adapter.py --export-only`, xác thực từng export với `validate_memory_sft_export.py`, rồi smoke policy thật. |
| Huấn luyện đối sánh | Mở. Chưa có F0 và F1 mới. | Train từ base với cùng corpus, recipe, seed, sampler, replay fraction, effective batch, optimizer và bước chọn checkpoint. Ghi checkpoint ra volume đủ chỗ. Lưu training config/metrics, export, hashes và runtime receipt hai arm. |

F0 đọc instruction, F1 đọc prompt có memory prefix. C0 cần view chỉ có fact hiện tại, L0 cần view lịch sử phẳng với cùng nguồn fact; cả bốn dùng cùng images, states, action targets và sample indices. Kiểm checksum roster trước train. Memory dropout nếu dùng phải khai báo seed/rate và áp dụng nhất quán giữa memory arms. So within-backbone trước khi so khác backbone.

| Training loop | Quyết định |
|---|---|
| F0 no-memory pilot 7-task `no_memory-20261029-175154` | Lưu lịch sử 438 train demo, trong đó 158 Mem. Train F0 mới từ checkpoint gốc để dùng cùng 598 demo và recipe với F1. |
| F1 full 10-task `mem-adapter-20261105-083543` | Lưu lịch sử 558 train demo, trong đó 158 Mem. Train F1 mới từ cùng checkpoint gốc với writer quan hệ. |
| F1 derived v1 `results_mem_derived/mem-adapter-v1/output` | Lưu lịch sử 552 train demo, trong đó 152 Mem. Chỉ train lại nếu cần đối chứng derived riêng. Khi đó tạo lock/manifest mới và train cả F0/F1 cho nhánh derived. |
| B0 frozen SmolVLA và baseline V1 | Dùng nguyên checkpoint. Chạy rollout B0 trên raw unseen nếu đưa B0 vào bảng mới. |
| F1-off, C0 và L0 | F1-off nạp F1 mới và tắt memory lúc inference. C0/L0 chỉ train khi triển khai đối chứng cơ chế, trên cùng corpus. |

Recipe chính giữ 3.000 bước, seed 42 và batch hiệu dụng 32 cho cặp đầu tiên. Smoke 5 bước mỗi arm, kiểm strict reload, rồi train từ base trong output mới. `--resume` chỉ dùng với manifest của chính run v2. Base-only validation theo dõi flow loss. Chốt checkpoint cuối trước khi dùng unseen. [Lệnh và thứ tự chạy P5](P5_P6_HANDOFF_2026-11-13.md#2-dự-toán-dung-lượng-và-p5).

## 9. P6 — Các chiến dịch thực nghiệm

| Chiến dịch | Mốc hiện tại | Gate, code hoặc evidence còn thiếu |
|---|---|---|
| E0 · competence/runtime | B0 v2 chạy hai action thật trên một state seen mỗi task. Một run cũ dừng detector `MemoryError` sau 5/40; lượt 5k ghi `running`, chưa có summary lúc rà soát. | Chạy lượt development T1/T3 hoàn tất, ghi số primitive và điểm quyết định đạt được, latency toàn hệ, taxonomy lỗi và tiêu chí go/no-go. Streaming giảm RAM của trace ở runner, detector vẫn cần chẩn đoán nếu tái lỗi. |
| E1 · writer (RQ2) | Có script chấm mốc đặt lại T1–T4 từ nguồn tác giả. | GPU T3, T2, T4; roster và nhãn độc lập cho quỹ đạo do policy sinh, gồm failure. Báo precision/recall, độ trễ, unknown; thêm nhãn identity/relation/containment cho T7/T10 khi có nguồn. |
| E2 · đối sánh vòng kín (RQ1/RQ3) | Khóa đã xác minh 40 lượt raw unseen mỗi arm trên 20 cụm state. Wrapper và aggregator hỗ trợ hai roster. | Chạy smoke và seen với cặp F0/F1 P5. Chốt phân tích rồi chạy 40 task/state mỗi arm mỗi seed. Báo counts, macro delta, latency và lượt thiếu. Cần triển khai bootstrap 20 cụm và xem lại cỡ mẫu trước claim xác nhận. |
| E3 · ngân sách token (RQ1) | Projection có budget và telemetry. | Thiếu runner sweep budget, roster cố định và báo cáo decision-fact survival, SR, token/padding/chi phí. Giữ nguyên instruction và tokenizer. |
| E4 · evidence/noise/delay (RQ2) | Writer có uncertainty và nguồn evidence. | Thiếu code can thiệp availability/noise có log delivery; cần risk–coverage, false confirmation, lỗi hành vi, latency. |
| E5 · can thiệp thông tin (RQ1/RQ2) | F1-off đã có để chẩn đoán. | Thiếu fact-changing/fact-preserving/exact-repeat, kiểm delivery và nhánh C0/L0 đối sánh. |
| E6 · retention/backbone (RQ3) | Có dữ liệu replay và ứng viên MiniVLA. | Thiếu harness retention chạy chính B0/F0/F1 trên tác vụ cơ sở, roster ghép cặp, cỡ mẫu và biên retention đã chốt. MiniVLA còn audit gripper/checkpoint, smoke thật và so within-backbone. |

E2 unseen T1/T3 là bảng vòng kín ưu tiên. Seen phục vụ smoke và development. Chốt checkpoint, seed, horizon, estimator và cách xử lý lỗi trước khi chạy unseen. `run_mem_pilot_batch.ps1` tắt video và ghi trace theo luồng. `aggregate_mem_pilot.py` kiểm cặp episode, manifest và hash, đồng thời báo bounds khi thiếu lượt. Bootstrap theo state giữ hai task và mọi seed trong cùng cụm. E3–E5 có thể bắt đầu từ tập development đã chọn trước. [Lệnh chạy P6](P5_P6_HANDOFF_2026-11-13.md#3-p6-theo-lô-nhỏ).

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
3. Bảng T1/T3 ghép cặp có mẫu số, SR và lượt thiếu. Bổ sung CI khi có chiến dịch xác nhận. Bảng toàn suite theo family và macro task chỉ mở khi evaluator và nguồn của các họ đã qua audit.
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

Với dung lượng hiện tại, ưu tiên cặp đối sánh, audit chất lượng evidence và một phép kiểm thông tin trên T1/T3. T2/T4 giúp kiểm writer với vật thể khác. T7/T10 là mốc audit offline tiếp theo khi có dung lượng. M3 toàn suite và các đối sánh backbone vẫn là các mốc mở rộng. Giới hạn claim policy theo task, backbone và giao thức đã đo.

## 12. Định nghĩa hoàn thành

- Các RQ có hypothesis, metric, đối chứng, điều kiện bác bỏ và kết luận tương ứng.
- Mốc P5–P6 cần cặp F0/F1 từ 198 refined demo, alignment audit, smoke thật trên seen và lượt ghép cặp unseen T1/T3 có mẫu số đầy đủ hoặc trạng thái thiếu được công bố. E1 báo riêng phạm vi writer trên T2/T4 và các họ bổ sung.
- Mốc toàn suite cần nguồn chính thức, khóa trạng thái, evaluator/runtime và lượt hợp lệ cho từng task. Nguồn, asset và policy phải qua kiểm provenance.
- Mọi kết quả primary đến từ real rollout có identity và protocol; custom results được ghi đúng phạm vi.
- Core method có logic dùng chung thật; evidence/uncertainty tham gia quyết định cập nhật và projection.
- F1–F0 ước lượng đóng góp của recipe memory so với adaptation cùng corpus. Quy kết tác dụng riêng của perception hoặc representation cần E3–E5. Claim xác nhận cần holdout và CI đúng đơn vị cụm.
- Cost bao gồm mọi thành phần online; retention chạy đúng adapter.
- Chương phương pháp độc lập với tên model/benchmark; chương thực nghiệm giải thích cụ thể các lựa chọn đó.
- Thesis và phụ lục đọc được mà không cần truy cập máy tác giả; abstract giữ nguyên theo yêu cầu.

Không một mục nào được đánh dấu hoàn thành chỉ vì đã có tên class, enum condition, tài liệu thiết kế hoặc unit test pass.
