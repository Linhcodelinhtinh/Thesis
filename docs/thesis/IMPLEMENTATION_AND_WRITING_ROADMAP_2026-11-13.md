# Kế hoạch thực thi: bản viết, phương pháp và thực nghiệm

> Các trạng thái gate bên dưới ghi theo ngày của tài liệu. Xem [đối chiếu cập nhật 19/11/2026](EXPERIMENT_CONFIGURATION_REGISTER.md) cho export F1 V2, kết quả E2/E6 và replay Relation V3.


Ngày 13/11/2026. Đi kèm [định vị nghiên cứu và bộ RQ](RESEARCH_REFRAME_2026-11-13.md). Đây là backlog nội bộ: tên module/đường dẫn dùng để giao việc và review code, không được chép vào nội dung luận văn hoặc phụ lục.

Trạng thái sau khi chuẩn bị P5–P6, bổ sung kế hoạch tích hợp quyết định ngày 14/11/2026. V1 là baseline lịch sử đã khóa. Các thay đổi về memory, dữ liệu đối sánh và đánh giá trong kế hoạch này thuộc V2. Lệnh theo thứ tự phụ thuộc và vị trí artifact nằm trong [bàn giao P5–P6](P5_P6_HANDOFF_2026-11-13.md). Thiết kế, interface và nghiệm thu cơ chế mới nằm trong [kế hoạch tích hợp memory vào quyết định](MEMORY_DECISION_INTEGRATION_PLAN_2026-11-14.md).

## Cập nhật thực thi ngày 15/11/2026

Người dùng yêu cầu hoàn tất các mục tiêu ngoài huấn luyện model. Trong cập nhật này không chạy training hoặc tạo checkpoint mới. P5 phần dữ liệu đã qua audit alignment; D0–D2 có replay offline trên trace cố định; T7/T8/T9/T10 raw và refined sources đã xác minh local. Artifact F0 V2 no-memory 5.000 bước đã tồn tại trước lượt cập nhật này; manifest, split lock và weight identity được xác nhận, sau đó adapter nạp được trong inference environment đã khóa. B0/F0 qua bốn runtime checks (hai action mỗi task-condition). Pilot E0 B0/F0 hoàn tất bốn episode seen: một state/task, 1.000 action mỗi episode, 0/4 thành công, SCR trung bình 0, overshot 0; đây là bằng chứng development custom, không phải LIBERO score hoặc memory effect. Audit preprocessing mới đối chiếu image contract, code collator và LeRobot 0.6.1: không thấy lệch hướng ảnh ở đường transform đã khai báo; F0 không lưu riêng package source hash phía training nên không tuyên bố đồng nhất byte-level. Writer RGB quan hệ trên một demo train T7/T8/T9/T10 đều không qua gate đầu ra: lần lượt 0/425, 0/613, 0/356 và 0/309 prompt có memory; relation state unknown xuyên suốt cả bốn demo, gồm 927/927 quan sát identity unknown trên T10. Audit hộp ảnh T7 độc lập với đầu ra detector đã đối chiếu đủ 425 hash ảnh với exo-box annotations phát hành: tại IoU 0,5, bowl precision/recall 0,893/0,747 và plate 0,971/0,646; đây là một demo train, mô tả detection theo frame, không phải accuracy identity/relation hoặc nhãn người thứ hai. Audit MiniVLA xác minh hash của hai checkpoint local và pin source `openvla-mini` ở `0822b36227b5a771be4eb2680e34c559734c8fdc`, nhưng adapter khác loader/prompt/ảnh/action path chính thức nên đã bị khóa fail-fast; chưa có eval checkpoint selection. F1 V2 còn thiếu; vì vậy paired F0/F1, D1/D2 vòng kín, E2 unseen, can thiệp policy, retention và phân tích quyết định vẫn chưa nghiệm thu.

Bổ sung cùng ngày: một author-action replay cho mỗi T8/T9/T10 dùng helper môi trường, state initializer, action transform và evaluator từ commit upstream đã pin. T9/T10 ghi nhận terminal success; T8 ghi ba mốc sequence nhưng không đạt terminal success, dù metadata refined của `demo_1` ghi `success=true`. Audit đối chiếu tiếp xác nhận action sau transform và initial state raw/refined trùng chính xác; robot-state replay lần đầu vượt dung sai chẩn đoán $10^{-5}$ ở bước 219, tối đa 0,008815, nhưng vẫn không terminal. Điều này loại trừ khác biệt action/state đầu vào của demo_1, chưa xác định nguyên nhân runtime/dynamics. T8 vẫn là gate mở. Đây là smoke một demo/task, không có policy.

| Gói | Hoàn tất được ngoài training | Còn mở |
|---|---|---|
| P0 | LaTeX canonical; abstract SHA-256 lock; build script kiểm hash; Chương 1–5 đã biên tập theo 3 RQ; validator và evidence ledger pass | Không còn gate biên tập/kỹ thuật; nội dung cá nhân cần tác giả đọc và xác nhận trước khi nộp |
| P1 | Split V2, dữ liệu nguồn và alignment audit; T7 branch replay; T8/T9/T10 one-demo author-action replay qua upstream setup/evaluator; strict validator và simulator smoke 14/14 | EGL fallback sang `kms_swrast`; T8 input action/state raw-refined khớp nhưng robot-state drift đầu tại bước 219 và chưa terminal; smoke không là policy score; MiniVLA còn bị chặn contract |
| P2–P4 | Điều kiện D0–D2 và profile task đã hiện thực; T7/T8/T9/T10 receipts, writer audit và replay nguồn một demo/task có artifact; T7 detector-vs-exo-box audit cũng đã có | Writer T7/T8/T9/T10 không tạo memory trên demo đã kiểm; nguyên nhân T8 trajectory drift chưa rõ; independent labels và accuracy còn mở |
| P5 | 698 demo, 180.447 cặp obs/action, shards 11,83 GiB; alignment pass với 1.396 mẫu parity; artifact F0 V2 no-memory hiện hữu được nạp thật; audit preprocessing không thấy mismatch ở transform ảnh khai báo | F1 V2, export/reload matched và cặp F0/F1 chưa có; training mới bị loại khỏi phạm vi; package source hash phía training không lưu |
| P6 | Pilot lịch sử B0/F1 đủ 40 summary; D0–D2 replay offline 198 demo; E0 B0/F0 runtime check/pilot; E1 T7–T10 source-action/evaluator diagnostics và writer audit đã có | E0 memory competence/decision point, nhãn blind, E2 matched/unseen, E3–E5 interventions và E6 retention/backbone chưa qua gate; T8 input actions/state khớp nhưng replay dynamics mismatch chưa giải thích |
| Backbone thứ hai | MiniVLA checkpoint bytes và source revision đã kiểm provenance; report ghi official input/action contract và sai khác adapter nội bộ | Eval checkpoint chưa được chọn; adapter không khớp official loader/prompt/image/gripper path; Candidate VQ thiếu checkpoint/codebook; không chạy rollout |
| P7 | Nguồn LaTeX, RQ, giới hạn claim, provenance và hình đã cập nhật; full build ngày 15/11 pass; PDF 57 trang đã render và rà trực quan toàn bộ sau khi thêm T8 state-parity diagnostic | Không còn gate layout; kiểm tra cuối bởi tác giả trước khi nộp |

Các mốc được đánh dấu “còn mở” giữ nguyên trạng thái mở. Không đổi kết quả offline hoặc unit test thành nghiệm thu runtime/policy.

## 1. Hiện trạng đã đối chiếu và hệ quả

| Thành phần | Hiện trạng thấy trong nguồn/tập bằng chứng | Việc tiếp theo |
|---|---|---|
| P0 · Bản viết và protocol | LaTeX là nguồn biên tập chính; abstract có SHA-256 lock. Protocol v2 khóa roster unseen 20 cụm state T1/T3; D0 development protocol riêng đã có. | Chạy validator, build/render PDF và rà cuối; chỉ sửa abstract khi tác giả duyệt bản nộp. |
| P1 · Dữ liệu và runtime | Khóa v2 xác minh 198 demo Mem refined, 500 demo LIBERO cơ sở; alignment toàn corpus đã pass. T7–T10 nguồn raw/refined đã khóa; author-action runtime diagnostics cho T7–T10 đã chạy trên demo development. Strict validator pass và simulator smoke 14/14 trong Python 3.8.13. | EGL fallback sang `kms_swrast`; T8 chưa terminal trong replay một demo, dù metadata refined ghi success; các lượt này không phải policy score; MiniVLA chưa qua runtime gate. |
| P2 · Điều kiện và adapter | CLI có F0 và F1-off. B0/F0 V2 đã qua runtime check thật, bốn task-condition checks; F0 V2 no-memory adapter nạp thành công. | F1 V2 không có; E0 pilot không đạt subgoal nên competence cho cơ chế memory vẫn mở. |
| P3–P4 · Memory và projection | `rgb_relations_v1` và task profile có trên mã nguồn; T1/T3 có audit writer lịch sử. Demo T7/T8/T9/T10 lần lượt cho 0/425, 0/613, 0/356 và 0/309 frame có memory; relation unknown. T9/T10 source replay đạt terminal ở một demo; T8 có discrepancy chưa giải thích. | Không tuyên bố writer T7/T8/T9/T10 đạt; cần nhãn độc lập và kiểm identity, relation, containment. T2/T4 vẫn ngoài luồng chính. |
| P5 · Dữ liệu | Bộ shards V2 lossless 11,83 GiB; 698 demo/180.447 cặp obs-action; alignment pass 2.094 frame mỗi camera và 1.396 mẫu parity. | Tạo checkpoint/export/reload bị loại khỏi phạm vi hiện tại; không chạy rollout matched khi chưa có adapter tương ứng. |
| P6 · Đánh giá | Pilot lịch sử B0 0/20 và F1 1/20 đủ 40 summary; D0–D2 replay offline 198 demos/42.857 frame; T8–T10 author-action diagnostics đã có. | E0/E1 policy+annotation, E2 matched rollout, E3–E5 intervention và E6 retention/backbone chưa qua gate; không báo thành score mới. |
| Backbone thứ hai | MiniVLA checkpoint bytes và source commit đã pin; audit contract chính thức ghi lại các khác biệt của adapter. | Chưa chọn eval checkpoint; local adapter sai loader/prompt/image/gripper path; Candidate VQ thiếu codebook. Không đưa vào so sánh. |

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

Loại bỏ T2/T4 khỏi luồng train/evaluate chính của mem-derived: khi môi trường mô phỏng bất định dẫn đến tư thế của chính vật cần gắp (chai rượu) bị nghiêng/đổ khác biệt sau reset/warmup, thất bại của policy bị chi phối bởi khả năng thích nghi tiếp xúc vật lý (grasp-physics adaptation) thay vì phản ánh hiệu quả của bộ nhớ. Ưu tiên trực tiếp chuyển sang T7 (hoán đổi 2 bát giữa các đĩa: hình học bát tiếp xúc ổn định, trực tiếp kiểm tra Object Relations và multi-instance tracking). Sau khi ưu tiên T7, giải phóng bộ nhớ và dung lượng lưu trữ (xóa/loại bỏ HDF5 T2/T4) để hướng đến T9/T10 (kiểm tra Object Occlusion và containment reasoning). E2 xác nhận cần roster độc lập đã khóa và kế hoạch phân tích được duyệt. Có thể sửa chương 1–3 song song; chương 4–5 chốt theo bằng chứng đã hoàn tất.

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

MiniVLA là ứng viên thứ hai vì repo có sẵn hướng tích hợp; chưa coi là sẵn chạy. Audit đã pin official code, xác nhận action dimension/tokenizer/statistics, default primary camera và action wrapper LIBERO, đồng thời ghi hai hash checkpoint. Checkpoint eval chưa được chọn; candidate adapter không tái dùng official loader, prompt builder, backbone image transform hoặc đầy đủ gripper postprocess. Adapter bị fail-fast cho đến khi được nối bằng runtime chính thức và kiểm parity. Candidate VQ tiếp tục bị chặn vì thiếu checkpoint/codebook có provenance. [Báo cáo audit](../../experiments/audits/minivla-contract-audit-20261115/report.json).

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
- Vẫn đảm bảo ngân sách 48 tokens của SmolVLA. Không giả định MiniVLA còn dư token cho memory; chốt ngân sách theo tokenizer, prompt và backbone transform chính thức sau khi runtime MiniVLA qua gate.

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
T7	OM + OR	Hoán đổi 2 bowls giữa các plates qua đĩa trống	Theo dõi object identity và thứ tự di chuyển
T8	OM + OR	Hoán đổi 3 bowls giữa các plates theo rotation rule	Multi-object relational tracking
T9	OM + OO	Đặt hộp cream cheese vào basket gần nhất, sau đó di chuyển basket chứa hộp vào giữa	Nhớ basket nào đang chứa hộp
T10	OM + OO	Đặt hộp cream cheese vào basket gần nhất, sau đó di chuyển basket rỗng vào giữa	Phân biệt basket chứa hộp và basket rỗng
| Chặng | Công việc thực chất | Gate trước policy evaluation |
|---|---|---|
| A: T1, T3 (loại T2/T4) | Duy trì đếm chu kỳ bát (T1, T3); loại trừ T2/T4 khỏi luồng chính vì tính bất định vật lý của chai gây sai lệch tư thế ban đầu và chi phối kết quả bởi thích ứng grasp cơ học; kiểm soát sai số tích lũy trên chuỗi lặp dài (T5/T6 mở rộng sau) | Precision/recall/timing trên state chưa dùng tune, cả failure và boundary cases; đúng mục tiêu số lần theo instruction |
| B: T7 (ưu tiên hàng đầu) / T8 | Multi-instance tracking, origin binding, identity qua giao cắt và hoán đổi đĩa (T7: 2 bowls/3 plates qua đĩa trống); kiểm riêng lỗi danh tính và lỗi thực thi grasp trên rollout thật | Phân biệt được các entity bằng evidence; ambiguous được giữ; báo ID switches và relation accuracy |
| C: T9/T10 (sau T7) | T9/T10 sources đã được hash-check local và writer diagnostic `demo_1` đã chạy; cả hai không tạo fact memory. Cần kiểm containment, continuity của container, phân biệt giỏ chứa vật vs giỏ rỗng | Không đồng nhất invisible với absent; cần nhãn độc lập, target-object và container binding đúng qua che khuất |

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

**Phạm vi triển khai và giải phóng dung lượng:** T1/T3 cho vòng kín hiện tại. T2/T4 bị loại bỏ khỏi luồng train/evaluate chính của mem-derived do rủi ro bất định mô phỏng làm lệch tư thế gắp chai (biến gây nhiễu vật lý). Dữ liệu và chỉ mục T2/T4 được dọn dẹp để giải phóng bộ nhớ RAM và dung lượng đĩa cục bộ (~16.55 GiB). Dung lượng giải phóng được ưu tiên tuyệt đối cho việc tải và thiết lập T7 (Object Relations với 2 bát). Sau T7, tiếp tục giải phóng tài nguyên trung gian để hướng đến T9/T10 (Object Occlusion / containment reasoning). Profile hoặc audit offline chỉ hỗ trợ claim ở cấp phương pháp/writer tương ứng. Để báo kết quả policy trên họ mới cần thêm khóa trạng thái, smoke runtime, evaluator đúng ngữ nghĩa và xử lý Or.

## 8. P5 — Dữ liệu và huấn luyện đối sánh

Mốc trước mắt dùng 198 refined demo của tác giả và 500 demo base đã khóa. F0/F1 dùng chung corpus và sampler, F1 dùng writer quan hệ. C0/L0 là đối chứng cơ chế tiếp theo. Chọn author-refined cho bảng chính vì derived có thêm exclusions do tiêu chí replay local.

| Gate | Trạng thái | Việc và minh chứng cần có |
|---|---|---|
| Năng lực và writer | E0 B0/F0 T1/T3 đã hoàn tất bốn rollout; cả bốn đạt horizon nhưng 0 subgoal, SCR 0 và không overshot. F0 runtime nạp đạt; không có F1 V2 để kiểm competence khi memory được dùng. | E0 memory gate còn mở vì chưa quan sát được điểm quyết định/primitive memory. E1 cần audit GPU T1/T3 và T7 cùng nhãn độc lập. |
| Chỉ mục memory từ refined | Hoàn tất cho T1/T3 từ nguồn refined theo khóa split v2; hai manifest index cùng phủ 42.857 frame Mem và được giữ làm lineage của bộ shards. | Không có việc còn mở trong phạm vi dữ liệu hiện tại; T7–T10 là diagnostics writer riêng, không nhập vào tập train T1/T3. |
| Một bộ shards lossless | Hoàn tất cho corpus 698 demo: 598 Mem/base train, tổng 180.447 cặp observation–action; 11,83 GiB. | Alignment audit toàn bộ đã qua: 180.447 hàng state/action, 2.094 RGB frame mỗi camera và 1.396 mẫu parity; phân loại QA dữ liệu, không phải policy score. |
| Training và reload | Có F0 V2 no-memory artifact từ run trước, strict reload đạt và runtime thật đã kiểm; F1 V2 chưa có. | Training mới bị loại khỏi phạm vi. Không dùng adapter F1 lịch sử với writer khác để đóng gate; giữ matched F0/F1 mở. |
| Huấn luyện đối sánh | Cặp F0/F1 mới chưa có; artifact F0 hiện hữu không tạo thành cặp. | Mục này ngoài phạm vi hiện tại. Nếu được mở lại ở giai đoạn khác, cần cùng corpus, recipe, seed, sampler, optimizer, checkpoint-selection và runtime receipts hai arm. |

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
| E0 · competence/runtime | Runtime check B0/F0 qua 4/4 checks (2 action/check). Pilot B0/F0 hoàn tất 4/4 full-horizon episodes: T1/T3 × B0/F0, state seen 0, seed 42, 1.000 policy actions/episode; 0/4 success, SCR 0, overshot 0. Mean policy inference 596–708 ms/query; wall time 202–208 s/episode. | Pilot chỉ có một state/task, B0/F0 đều không dùng memory và hoàn thành 0 subgoal; không đóng memory competence, decision-point hoặc E2. Không gộp với run lịch sử. Artifact: `experiments/evaluation/e0-f0-runtime-ext4-20261115` và `experiments/evaluation/e0-b0-f0-pilot-20261115`. |
| E1 · writer (RQ2) | Có script chấm mốc đặt lại T1/T3; T7/T8/T9/T10 writer diagnostics trên một demo train mỗi task; T7 detector được đối chiếu exo-box annotations. | Writer T7/T8/T9/T10 hiện không cấp memory fact; roster và nhãn độc lập cho quỹ đạo do policy sinh (kể cả failure) còn mở. Báo precision/recall, độ trễ, unknown và relation accuracy sau khi có nhãn. |
| E2 · đối sánh vòng kín (RQ1/RQ3) | Khóa đã xác minh 40 lượt raw unseen mỗi arm trên 20 cụm state. Wrapper và aggregator hỗ trợ hai roster. | Chạy smoke và seen với cặp F0/F1 P5. Chốt phân tích rồi chạy 40 task/state mỗi arm mỗi seed. Báo counts, macro delta, latency và lượt thiếu. Cần triển khai bootstrap 20 cụm và xem lại cỡ mẫu trước claim xác nhận. |
| E3 · ngân sách token (RQ1) | Projection có budget và telemetry. | Thiếu runner sweep budget, roster cố định và báo cáo decision-fact survival, SR, token/padding/chi phí. Giữ nguyên instruction và tokenizer. |
| E4 · evidence/noise/delay (RQ2) | Writer có uncertainty và nguồn evidence. | Thiếu code can thiệp availability/noise có log delivery; cần risk–coverage, false confirmation, lỗi hành vi, latency. |
| E5 · can thiệp thông tin (RQ1/RQ2) | F1-off đã có để chẩn đoán. | Thiếu fact-changing/fact-preserving/exact-repeat, kiểm delivery và nhánh C0/L0 đối sánh. |
| E6 · retention/backbone (RQ3) | Có dữ liệu replay; MiniVLA source/checkpoint provenance đã audit và ghi blocker. | Thiếu harness retention chạy chính B0/F0/F1 trên tác vụ cơ sở, roster ghép cặp, cỡ mẫu và biên retention đã chốt. MiniVLA còn eval checkpoint selection, official-runtime parity, môi trường inference, smoke thật và so within-backbone; VQ thiếu resource codebook. |

E2 unseen T1/T3 là bảng vòng kín ưu tiên. Seen phục vụ smoke và development. Chốt checkpoint, seed, horizon, estimator và cách xử lý lỗi trước khi chạy unseen. `run_mem_pilot_batch.ps1` tắt video và ghi trace theo luồng. `aggregate_mem_pilot.py` kiểm cặp episode, manifest và hash, đồng thời báo bounds khi thiếu lượt. Bootstrap theo state giữ hai task và mọi seed trong cùng cụm. E3–E5 có thể bắt đầu từ tập development đã chọn trước. [Lệnh chạy P6](P5_P6_HANDOFF_2026-11-13.md#3-p6-theo-lô-nhỏ).

Thiết kế can thiệp:

- Null event: thêm khoảng chờ hoặc chuyển động không tạo tiến độ theo task; kiểm chính intervention không đổi success condition hoặc đưa hệ ra ngoài protocol. Các thí nghiệm can thiệp mang nhãn custom.
- Fact-changing: đổi count/entity/container cụ thể; đảm bảo cấu trúc và token budget hợp lệ.
- Fact-preserving: đổi cách diễn đạt hoặc thứ tự các fact độc lập trong khi giữ nghĩa cần kiểm; không hoán đổi order của event nếu order chính là thông tin.
- Exact-repeat: thực hiện cùng đường cài đặt nhưng giữ nguyên nội dung để phát hiện hiệu ứng do can thiệp phần mềm.
- Delay: giữ nội dung, trì hoãn availability có kiểm soát; không chuyển timestamp để giả vờ có evidence sớm hơn.
- Current-only: bỏ lịch sử nhưng giữ detector/current fact, kiểm phần này không ngầm chứa counter/cached history.

Nhãn oracle chỉ đi vào chiến dịch chẩn đoán riêng. Không đưa số O-diag vào bảng thành tích predicted memory. Không coi failed oracle diagnostic tự động chứng minh memory không hữu ích: trước hết kiểm khả năng đọc định dạng và năng lực vận động.

### 9.1. Tích hợp cơ chế quyết định từ memory

Cập nhật ngày 14/11/2026. D0–D2 đã có đường chạy development trong CLI và runner. Replay đã chạy trên 198 quỹ đạo mẫu T1/T3, gồm 42.857 frame. P5 đã dựng và audit bộ shards chung 11,83 GiB. Smoke vòng kín còn phụ thuộc checkpoint F1 dùng writer quan hệ từ P5. [Thiết kế và ma trận đối chứng](MEMORY_DECISION_INTEGRATION_PLAN_2026-11-14.md). [Bàn giao triển khai và lệnh chạy](MEMORY_DECISION_HANDOFF_2026-11-14.md).

| Gói | Việc kế tiếp | Gate còn mở |
|---|---|---|
| D0–D1 | Protocol seen, schema nhãn, fact refs và shadow đã có. Kiểm parity prompt/chunk/action/RNG bằng unit fixture. | F1 quan hệ cho smoke thật và nhãn điểm quyết định |
| D2 | Requery một đề xuất, lịch đối chứng, giới hạn query và trace abstention đã tích hợp. | Pilot vòng kín để đo trigger, số query, chi phí và success |
| D3 | Có phép kiểm offline ba giá trị cho intent đã gán nhãn. | Bộ đọc vật đích/hiệu ứng, nhãn độc lập và ngưỡng false accept/false reject/coverage trên development |
| D4 | Sinh K ứng viên và chọn theo memory | Kiểm queue/cache/RNG của policy, parity K=1 và khả năng tập ứng viên chứa hành động đúng |
| D5 | T7 raw đã kiểm SHA-256, khóa 120 state; replay kiểm hai nhánh Or có RGB. T8/T9/T10 mỗi task có một author-action replay qua helper/evaluator upstream; T9/T10 terminal success, T8 ba milestone nhưng không terminal. Raw/refined T8 demo_1 action/state đầu vào trùng chính xác; robot-state replay vượt dung sai $10^{-5}$ đầu tiên tại bước 219, nguyên nhân chưa rõ. Writer audit T7/T8/T9/T10 vẫn có 0 memory frames. T7 detector được đối chiếu 425 ảnh với exo-box annotations: IoU 0,5, bowl P/R 0,893/0,747; plate 0,971/0,646. | Chỉ là source-action diagnostics trên demo development; exo-box không phải nhãn blind identity/relation. T8 dynamics discrepancy, independent labels, policy competence và protocol vòng kín vẫn mở. |
| D6 | Probe offline, aggregator ghép cặp, bounds lượt thiếu và bootstrap theo state. | Nhãn, risk–coverage–cost vòng kín và protocol xác nhận trước unseen |

Primary F1–F0 của P6 đo recipe memory đối sánh. Các arm supervisor đo tác động của thời điểm và quyền thực thi với cùng F1. Khi supervisor đọc toàn snapshot, báo quyền truy cập thông tin bổ sung so với prompt. Thêm phép kiểm dùng cùng tập fact cho cả policy và supervisor. ROI conditioning hoặc critic được mở theo kết quả phân tích lỗi.

D1/D2 dùng lớp chiếu metadata riêng, đã kiểm hash writer/renderer với hai index hiện có. Lịch theo evidence có 1.864 query dự kiến ở T1 và 3.546 ở T3. Lịch cố định 8 action cho 1.874 và 3.550 query trên cùng trace. Chênh lệch compute thực tế cần đo lại trên từng arm. Hai thiếu hụt kỹ thuật chính ở D3/D4 là bộ đọc hiệu ứng action và parity queue/cache/RNG khi sinh nhiều đề xuất. Protocol development ghi abstention là thất bại hoàn thành và dùng evaluator hiện có cho success.

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

Với dung lượng hiện tại, ưu tiên cặp đối sánh F0/F1, audit chất lượng evidence và kiểm tra thông tin trên T1/T3. Loại bỏ hoàn toàn T2/T4 khỏi luồng chính vì tính bất định vật lý của chai làm sai lệch tư thế gắp, khiến thí nghiệm bị chi phối bởi độ thích nghi grasp hơn là hiệu quả bộ nhớ. Ưu tiên cao nhất cho T7 để kiểm tra quan hệ/danh tính đa vật thể với hình học bát ổn định. Giải phóng bộ nhớ RAM và dung lượng đĩa cục bộ sau khi dọn dẹp T2/T4, sau đó hướng đến T9/T10 để khảo sát năng lực suy luận che khuất/vật chứa. M3 toàn suite và các đối sánh backbone vẫn là các mốc mở rộng. Giới hạn claim policy theo task, backbone và giao thức đã đo.

## 12. Định nghĩa hoàn thành

- Các RQ có hypothesis, metric, đối chứng, điều kiện bác bỏ và kết luận tương ứng.
- Mốc P5–P6 cần cặp F0/F1 từ 198 refined demo, alignment audit, smoke thật trên seen và lượt ghép cặp unseen T1/T3 có mẫu số đầy đủ hoặc trạng thái thiếu được công bố. E1 loại bỏ T2/T4 và chuyển trọng tâm sang T7 (quan hệ đa bát), tiếp nối bởi T9/T10 sau khi giải phóng tài nguyên.
- Mốc toàn suite cần nguồn chính thức, khóa trạng thái, evaluator/runtime và lượt hợp lệ cho từng task. Nguồn, asset và policy phải qua kiểm provenance.
- Mọi kết quả primary đến từ real rollout có identity và protocol; custom results được ghi đúng phạm vi.
- Core method có logic dùng chung thật; evidence/uncertainty tham gia quyết định cập nhật và projection.
- F1–F0 ước lượng đóng góp của recipe memory so với adaptation cùng corpus. Quy kết tác dụng riêng của perception hoặc representation cần E3–E5. Claim xác nhận cần holdout và CI đúng đơn vị cụm.
- Cost bao gồm mọi thành phần online; retention chạy đúng adapter.
- Chương phương pháp độc lập với tên model/benchmark; chương thực nghiệm giải thích cụ thể các lựa chọn đó.
- Thesis và phụ lục đọc được mà không cần truy cập máy tác giả; abstract giữ nguyên theo yêu cầu.

Không một mục nào được đánh dấu hoàn thành chỉ vì đã có tên class, enum condition, tài liệu thiết kế hoặc unit test pass.
