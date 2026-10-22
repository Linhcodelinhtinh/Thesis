# Audit mức hoàn thiện memory LIBERO-Mem — 22/10/2026

**Kết luận: CHƯA HOÀN THIỆN, kể cả khi loại toàn bộ training/fine-tuning khỏi phạm vi.** Implementation hiện có là khung phần mềm với một số logic chạy được trên unit fixtures. Chưa phải pipeline memory dùng được để đánh giá trên LIBERO-Mem thật. Các blocker nằm ở simulator, perception/writer, evaluator, chuẩn bị dữ liệu và tích hợp rollout; không phải chỉ thiếu checkpoint sau SFT.

Commit kiểm tra: `ac584d299601083b4c271145eea8c6062bca3b26`. Working tree sạch khi bắt đầu. Đối chiếu [implementation plan](V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md), source hiện tại, tests hiện tại và source tác giả tại revision `0eee0defa4024e51511b6e15a79f30e9620209de`.

## 1. Phạm vi và ý nghĩa bằng chứng

- Đã đọc simulator wrapper, evaluator, tracker/writer, schema, oracle, renderer, data processor/prepare CLI, rollout runner và tests liên quan.
- Đã chạy unit tests và các probe bổ sung. Mọi ảnh/observation/action tổng hợp trong probes là **unit fixtures**, không phải demonstration hoặc benchmark run.
- Không chạy training/fine-tuning, không tải weights/dataset, không chạy benchmark policy thật, không sửa production implementation.
- Chuẩn bị dữ liệu và causal annotation vẫn thuộc phạm vi: đây là điều kiện hoạt động của memory, dù nằm dưới `src/training/`.
- Các probe không đo accuracy, success rate hoặc writer precision trên task thật. Chúng chứng minh API hiện chấp nhận những trường hợp sai hoặc thiếu bằng chứng.
- Không đưa phần training/retention sau SFT vào tỷ lệ hoàn thành. Audit này không dùng một phần trăm tùy ý để gộp các module có mức độ quan trọng khác nhau.

Artifacts: [script tái hiện](../experiments/audits/libero_mem_completeness_20261022/reproduce_gaps.py), [kết quả cùng SHA256 source](../experiments/audits/libero_mem_completeness_20261022/probe_results.json). Thư mục `experiments/audits` đang bị gitignore; cần đóng gói riêng nếu muốn lưu evidence cùng thesis/review ở máy khác.

## 2. Ma trận hoàn thiện ngoài training

| Thành phần | Đã có | Kết luận |
|---|---|---|
| Source/config | Revision và task names trong YAML/manifest | **Một phần**: chưa có runtime resource lock/replay; task ID chưa phân biệt local index với upstream index |
| Simulator Mem | Class nhận action và trả observation | **Blocker**: mô phỏng kinematic tự viết, ảnh zero, không load LIBERO-Mem |
| Evaluator | Local lift/return counter và SCR | **Blocker**: tự dựng semantics khác BDDL/evaluator tác giả |
| RGB perception | Interface `detected_objects` | **Chưa có detector/segmenter**; không đọc hai camera hoặc thực hiện association/re-identification |
| Writer count | FSM với dwell, occurred/available timestamps | **Một phần, chưa trung thực**: có thể confirm chu kỳ từ EEF khi không thấy vật |
| Identity / relations / OO | Dataclass fields, enum và câu render | **Chưa hoạt động end-to-end**: một target, fixed support; không tracking nhiều vật, containment/occlusion thực |
| Memory schema | Object, relation, event, cycle, snapshot | **Một phần**: có validation cơ bản; thiếu temporal consistency/provenance guard và immutable containers |
| Renderer | Count/in-progress/containment text | **Blocker với contract hiện tại**: estimate token, mất identity/partial-history; không có privilege gate |
| Oracle | Writer gắn source oracle vào events | **Một phần/sai tích hợp**: data stream không gọi oracle; snapshot sau làm mất containment |
| Data preparation | In-memory chunking và JSONL CLI | **Chưa có loader dataset thật**, split/hash/shards/QA; thiếu field được thay bằng dữ liệu mặc định |
| Online runner | Update writer mỗi step, query theo horizon | **Một phần**: có mock policy fallback; thiếu artifact/provenance/action validation |
| Conformance audit | Unit traces tự kiểm | **Không đủ** để chứng nhận upstream/evaluator/data readiness |
| Base compatibility | Regression tests phần memory cũ | Chỉ kiểm tương thích phần mềm; chưa chứng minh giữ performance base khi bật memory mới |

## 3. Phát hiện chặn sử dụng

### F01 — Simulator production thực chất là một mock

**Mức độ: BLOCKER.** [libero_mem_env.py](../src/simulator/libero_mem_env.py) dòng 65–85, 97–138, 163–175.

Class không khởi tạo MuJoCo/robosuite/LIBERO, không đọc BDDL hoặc initial-state arrays. `step()` cộng delta vào EEF, cập nhật object/gripper bằng quy tắc tự viết. Hai camera là `np.zeros`. `reset(initial_state_id)` không sử dụng ID. Các hằng `OFFICIAL_*` không tạo hoặc kiểm chứng robot/controller thật; horizon 1000 cũng chưa được chứng minh là protocol Mem.

Probe `black_cameras_and_unused_init_id`: reset ID 0 và 9 trả tất cả observation giống nhau; cả hai ảnh toàn zero. Đây là kiểm tra reset của class, không phải rollout benchmark.

**Hệ quả:** không thể đánh giá VLA, memory hoặc retention qua wrapper này. Nhãn `CUSTOM_V2_LIBERO_MEM`/`NON-COMPARABLE` không hợp thức hóa việc thay simulator bằng mock theo AGENTS.md.

**Điều kiện đóng:** chuyển fixture sang test-only; production load checkout Mem đã pin, assets/BDDL/init states thật; validate controller/state/camera contract; fail-fast nếu thiếu resource; có real-demo replay và provenance.

### F02 — Writer xác nhận sự kiện mà không có bằng chứng object

**Mức độ: BLOCKER cho tính trung thực của memory.** [object_tracker.py](../src/perception/object_tracker.py) dòng 81–95, 132–180; [event_writer.py](../src/memory/event_writer.py) dòng 77–102, 127–145.

Tracker không đọc ảnh. Support mặc định là tọa độ cố định; target được đặt sẵn lên support. Khi thiếu detection và `is_held=True`, object position được thay bằng EEF, visibility vẫn `visible`; sau đó khoảng cách EEF–object bằng zero tiếp tục củng cố held. Mất detection/slip không làm belief chuyển sang unknown. Không có cross-camera tracking, identity association hoặc detector checkpoint.

Writer còn tạo relation `confirmed_by_writer` ngay trong constructor, trước observation đầu tiên, kèm tên evidence `obs_0_plate_A` chưa được ingest. Các event phát sau đó được gắn `predicted_rgb_tracker` nhưng không có frame evidence IDs.

Probe `memory_without_object_evidence`: chỉ đưa EEF/gripper, không ảnh và không detection, writer vẫn đếm **1 confirmed cycle**, xuất hai event `predicted_rgb_tracker`, evidence lists rỗng và visibility `visible`. Probe `initial_fact_without_observation` xác nhận relation đã tồn tại trước mọi input.

**Điều kiện đóng:** object/support phải được quan sát/ground theo sensor contract; prediction khi occluded phải giữ source/uncertainty, không thành observation mới; chỉ confirm transition khi có evidence đủ; bỏ initial fact giả; đo writer trên held-out real recordings gồm missed grasp/slip/occlusion. Học perception, nếu cần, là dependency ngoài audit này; dù không xét việc huấn luyện nó, đường chạy inference perception hiện cũng chưa có.

### F03 — Runner tự tạo action khi policy interface không hợp lệ

**Mức độ: BLOCKER.** [libero_mem_rollout.py](../src/evaluation/libero_mem_rollout.py) dòng 120–127.

Nếu object truyền làm policy không có `predict_action_chunk`, runner tạo `(50,7)` zeros và mở gripper. Nhánh này không bị giới hạn bởi test-only mode hoặc fixture dependency.

Probe `invalid_policy_is_accepted` dùng protocol spy trong unit test: `policy_adapter=object()` không raise; runner vẫn gọi `step` với `[0,0,0,0,0,0,-1]`. Không có checkpoint/policy thật được chạy trong probe.

**Điều kiện đóng:** xóa fallback khỏi production; validate interface/load status, chunk shape/finite values/horizon trước execution; mock policy chỉ truyền tường minh từ tests. Không được ghi kết quả missing-policy thành một episode failure bình thường.

### F04 — Evaluator không tuân thủ semantics của bản phát hành

**Mức độ: BLOCKER cho score.** [libero_mem_metrics.py](../src/evaluation/libero_mem_metrics.py) dòng 62–89, 115–151.

Local evaluator tự sinh hai subgoal lift/return cho mỗi cycle, không parse BDDL. T3 do đó có 6 subgoal. BDDL tác giả ở revision đã pin chứa 3 predicate `On` trong Sequence; evaluator tác giả dùng live/nonlive counters và overshoot semantics riêng. [BDDL gốc](https://github.com/libero-mem/libero-mem/blob/0eee0defa4024e51511b6e15a79f30e9620209de/libero/libero/bddl_files/libero_mem/KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times.bddl), [evaluator gốc](https://github.com/libero-mem/libero-mem/blob/0eee0defa4024e51511b6e15a79f30e9620209de/libero/libero/envs/problems/libero_kitchen_tabletop_manipulation.py).

Probe `evaluator_accepts_two_single_frames`: hai frame boolean lift rồi return đủ khiến local T1 completion true. Đây không phải conformance test với upstream. Không có xử lý OR cho T10; truyền task name khác vẫn xây lift/return từ `target_cycles`.

Ngoài ra wrapper dừng ngay khi `full_success` đầu tiên (`libero_mem_env.py:136–138`), nên đường rollout bình thường không tiếp tục tới frame overshoot mà unit test evaluator gọi trực tiếp. `termination_reason` của evaluator vẫn `in_progress` khi runner hết horizon.

**Điều kiện đóng:** adapter dùng semantics/call schedule từ upstream; đối chiếu traces trên replay thật; tách diagnostic lift/place khỏi official subgoal progress; khóa full success, terminal, reset và overshoot trước dùng score. Nếu sửa scoring, giữ patch/hash và metric custom riêng.

### F05 — Prepare CLI chưa đọc được dataset thật; processor âm thầm bù observation

**Mức độ: BLOCKER cho data readiness.** [prepare_libero_mem_sft.py](../scripts/prepare_libero_mem_sft.py) dòng 24–106; [libero_mem_data.py](../src/training/libero_mem_data.py) dòng 98–141.

CLI chỉ có đường sinh `--mock-episodes`; không có input HDF5/metadata/raw-dir. Mặc định không sinh sample nào nhưng vẫn ghi manifest. Không thực hiện split theo initial-state hash như docstring, không kiểm SHA256/upstream revisions, không lưu image/state/action shards hoặc writer event audit; JSONL chỉ chứa prompt và index.

Processor nhận ảnh/state thiếu bằng zeros/default quaternion. Probe `missing_observations_are_fabricated` đưa hai dict rỗng vẫn nhận hai samples có camera 128×128 và state mặc định, thậm chí prompt count. Không kiểm finite/range/contract đầy đủ, không xác minh image orientation hoặc real-data alignment. Đường `obs_action_offset=1` đổi target slice nhưng `last_action` vẫn lấy `actions[t-1]`, nên chưa đồng bộ lịch sử theo semantics base raw.

**Điều kiện đóng:** loader resource thật; verify source/split/contracts trước prepare; không dựng ảnh/state thiếu; source-specific mapping cả history lẫn target; lưu provenance/event evidence/shards đủ tái lập. Đây là chuẩn bị dữ liệu, không phải yêu cầu chạy training.

### F06 — Nhánh oracle bị ghi nhãn sai và mất state

**Mức độ: HIGH.** [libero_mem_data.py](../src/training/libero_mem_data.py) dòng 88–95, 123–132; [libero_mem_oracle.py](../src/memory/libero_mem_oracle.py) dòng 76–79, 140–155; [libero_mem_rollout.py](../src/evaluation/libero_mem_rollout.py) dòng 103–112, 144–146.

`ORACLE_MEMORY` trong data processor render cùng snapshot của `LiberoMemEventWriter`; không gọi oracle writer hoặc đọc annotation. Probe xác nhận predicted/oracle cùng prompt dù không cung cấp oracle signals. Đây là bằng chứng nhánh bị nối sai từ code, không chỉ việc hai phương pháp ngẫu nhiên cho cùng kết quả.

Oracle `step()` truyền containment/visibility vào snapshot trả ngay, nhưng không lưu state đó. `get_snapshot()` lần sau mặc định `container_id=None`, `is_occluded=False`; runner lại dùng lần gọi này khi dựng prompt. Probe: `basket_A/occluded` thành `None/visible`. Nếu thiếu `oracle_signals`, default `is_on_support=True` vẫn được chấp nhận, có thể gây return giả sau departure.

**Điều kiện đóng:** đúng oracle adapter với timestamp/provenance; lưu container belief và visibility qua snapshot; thiếu signal phải fail-fast; không dùng hardcoded state ban đầu; source privilege phải được duy trì tới renderer/report.

### F07 — Renderer chưa giữ thông tin cần cho OR/OO và thiếu guard causal

**Mức độ: HIGH.** [libero_mem_text.py](../src/interfaces/libero_mem_text.py) dòng 104–126; [models.py](../src/memory/models.py) dòng 426–432.

Renderer bỏ `history_complete`, origin anchors và identity ambiguity; không render `relations`. `object_id.split('_')[0]` biến `cream_cheese_A` thành `cream`, đồng thời biến cả `basket_A`/`basket_B` thành `basket`. Hai memory khác container do đó cho cùng prompt, mất thông tin quyết định của OO. Partial history vẫn được render như “zero times”.

Schema không kiểm `event.available_at <= snapshot.as_of_observation`; renderer không kiểm event evidence/availability khi xuất count. Probe dựng snapshot as-of 0 tham chiếu event available-at 10 vẫn render “once”. Renderer cũng nhận oracle-sourced snapshot mà không cần opt-in.

**Giới hạn nhận định:** hai probe cuối chứng minh guard ở API còn thiếu, chưa chứng minh writer hiện tự đọc tương lai hoặc một benchmark F1 thật đã bị oracle leakage. Tuy nhiên bất biến causal/privilege trong plan chưa được enforce.

**Điều kiện đóng:** giữ aliases/identity binding, qualifiers và partial-history status; retrieval theo as-of và evidence source; reject hoặc bỏ fact không hợp lệ có audit; explicit opt-in oracle. Thêm tests multi-object, source isolation và future-event rejection.

### F08 — Token budget chưa dùng contract tokenizer của model

**Mức độ: HIGH.** [libero_mem_text.py](../src/interfaces/libero_mem_text.py) dòng 40–80; call sites mặc định ở runner dòng 64 và data processor dòng 67.

Default không có tokenizer, `strict_tokenizer=False`, dùng estimate ký tự/từ. Nếu có tokenizer thì `encode(..., add_special_tokens=False)` và không tái hiện newline preprocessing. File preprocessor local của SmolVLA có newline step và tokenization `max_length=48`, `truncation=true`; vì vậy estimate không thể bảo đảm memory vào model nguyên vẹn.

Probe cho instruction dài: renderer tự báo 132 tokens trong budget 48 nhưng không raise ở no-memory path. Con số 132 là **estimate của renderer**, không phải số token thật đã đo. `.strip()` cũng làm off-path không còn exact pass-through. Chưa có kiểm chứng bằng official tokenizer trong các tests mới.

**Điều kiện đóng:** bắt buộc tokenizer đã pin, đếm đúng toàn pipeline; instruction quá dài hoặc mandatory memory không vừa phải báo lỗi theo protocol; exact pass-through khi off; log đầy đủ token/fact audit tại query.

### F09 — Schema có fields nhưng chưa cung cấp audit trail đủ mạnh

**Mức độ: MEDIUM/HIGH tùy dùng để chứng minh causality.** [models.py](../src/memory/models.py) dòng 374–444; [event_writer.py](../src/memory/event_writer.py) dòng 127–145.

`frozen=True` không làm các dictionary trong snapshot immutable: probe `snapshot.objects.clear()` thành công. `V2MemoryEvent.to_dict()` bỏ `details`, làm mất cycle information gắn vào event. Event không có frame evidence, object/target association đầy đủ hoặc model/config hash; relation retractions không nối tới một relation ID có thể truy nguyên. Writer chưa có cơ chế correction/recount, history gap và calibrated uncertainty hoạt động.

**Điều kiện đóng:** immutable/deep-copy snapshot contract; serialization round-trip không mất metadata; provenance từ sensor frame tới event tới count; kiểm tra consistent event references; version/threshold/perception hash và correction log.

### F10 — Audit “PASS” hiện chỉ xác nhận logic tự viết

**Mức độ: HIGH vì có thể tạo cảm giác readiness sai.** [audit_libero_mem.py](../scripts/audit_libero_mem.py) dòng 27–112, 116–175.

Chạy script hiện tại trả `overall_conformance=PASS`. Nhưng script chỉ kiểm YAML, boolean traces cho local evaluator và detection coordinates tự đặt cho writer; không load upstream environment, BDDL, init states, resource hashes hoặc demonstration thật. `cycle_counter_accuracy=verified` không phải accuracy đo trên recordings. Unit tests config còn assert 6 subtask T3, tức bảo vệ chính định nghĩa custom đã khác BDDL.

**Điều kiện đóng:** phân loại rõ `unit_contract_pass` và `upstream_conformance_pass`; runtime gate fail hoặc not-run khi chưa có resource/replay evidence. Không dùng output hiện tại để thông qua P1/P4.

## 4. Các phần còn thiếu khác

- `CausalObjectTracker` chỉ giữ một `target`; không cập nhật `container_id`, không chuyển sang lost/ambiguous/occluded theo ảnh. Vì vậy OR/OO chưa thể được coi là đã triển khai chỉ vì có enum/schema.
- Config thêm T10 nhưng runner luôn khởi tạo `bowl_A/plate_A`; evaluator vẫn lift/return. Local task IDs 1/2 cho T3/T10 chưa có upstream index riêng và resolution gate.
- Runner không gọi `policy.reset()`, không validate condition, execution horizon hoặc chunk shape/finite values; không có allowlist observation tại writer boundary. Các chỗ này cần protocol tests trước nối policy thật.
- Không có output writer precision/recall, count exactness, ID-switch/occlusion metrics, calibration hoặc streaming/offline parity trên dữ liệu thật.
- `RolloutResult` mới lưu prompt history, mean policy latency và summary. Chưa ghi full action/observation trajectory, video, writer evidence, tokenizer audit, writer latency/age, package/source/checkpoint hashes hay comparability tag trong result.
- Search trong checkout thấy `external/LIBERO` base; không thấy checkout/env Mem riêng trong cấu trúc repository đã kiểm tra. Planning manifest vẫn đánh dấu các runtime readiness flags là false. Không khẳng định người dùng không có resource ở một filesystem khác chưa kiểm tra.
- Phần base regression không thay thế paired rollout trên base với writer mới. Không có cơ sở kết luận performance base giữ nguyên chỉ từ tests phần mềm.

## 5. Verification đã chạy

| Kiểm tra | Kết quả | Phạm vi thực sự |
|---|---|---|
| 8 files `tests/unit/test_libero_mem_*.py` + `tests/test_project_structure.py` | **27 passed** (20 Mem + 7 structure), 2 SWIG deprecation warnings | Unit/config/import, không simulator thật |
| 15 probes bổ sung | **15/15 tái hiện hành vi thiếu guard/sai yêu cầu** | Không phải 15 lỗi độc lập hoặc 15 benchmark failures |
| `scripts/audit_libero_mem.py` | In `overall_conformance=PASS` | Chỉ local synthetic conformance như F10 |
| 5 files regression memory cũ | **38 passed**, 2 SWIG deprecation warnings | Tương thích phần mềm ở phạm vi tests đã chạy |
| `scripts/validate_project.py` | **PASS** toàn bộ kiểm tra cấu trúc/import | Không phải kiểm chứng simulation/performance |

Môi trường probes: Windows, Python 3.12.2, NumPy 2.2.6. Không phải official LIBERO reference stack. Không chạy `system_info.py`/environment smoke để chứng nhận Mem, vì wrapper mới chưa có backend thật; không thay thế bằng smoke của LIBERO base.

Lệnh tái lập phần audit không training:

```powershell
$memUnitTests = @(Get-ChildItem -LiteralPath tests/unit -Filter 'test_libero_mem_*.py' | ForEach-Object FullName)
python -m pytest -q @memUnitTests tests/test_project_structure.py
python experiments/audits/libero_mem_completeness_20261022/reproduce_gaps.py
python scripts/audit_libero_mem.py
```

Tổng cộng **65 tests hiện có pass**. Kết quả này đồng thời tồn tại với 15 probe tái hiện thiếu sót, vì tests hiện tại chưa bao phủ các điều kiện nghiệm thu quan trọng. Đây không phải kết luận các unit tests đều vô ích: chúng đã kiểm tra một số happy paths và invariants cơ bản.

Các probe chủ ý ghi nhận **hành vi lỗi hiện tại** để cung cấp evidence. Sau sửa lỗi, cần chuyển thành regression tests đòi hành vi đúng; không sửa code để làm các probe “gap_reproduced” tiếp tục true.

## 6. Thứ tự hoàn thiện trước training

1. **Cô lập mocks/fallbacks:** đưa synthetic environment vào test fixtures, bỏ auto-action fallback khỏi runner; sửa readiness labels để chưa có upstream replay không thể PASS runtime.
2. **Nối backend/evaluator thật:** resource lock, isolated env, BDDL/init-state mapping và camera/state/action contract; replay demo và kiểm reset/terminal/sequence/OR semantics.
3. **Làm writer trung thực:** perception inference từ RGB, grounding object/support, unknown khi thiếu evidence, causal evidence IDs và count FSM không dùng EEF thay object observation.
4. **Hoàn thiện schema/renderer/oracle:** immutable as-of snapshots, source filtering, identity/containment persistence, exact tokenizer và partial-history handling.
5. **Data/online parity:** loader HDF5 thật, split/hash/offset/transform audit, predicted/oracle tách đúng; replay từng prefix cho cùng memory với streaming.
6. **Chứng minh runtime không training:** shadow writer trên demo/rollout thật; error metrics và calibration; frozen-policy integration có trajectory/video/provenance. Sau đó mới có cơ sở xác nhận phần non-training ready.

Các dataclass, FSM đơn giản và skeleton query loop có thể giữ lại sau sửa. **Không nên tiếp tục từ giả định rằng chỉ còn thiếu fine-tuning.**
