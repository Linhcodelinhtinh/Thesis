# Re-audit implementation memory LIBERO-Mem — 23/10/2026

**Kết luận: CHƯA HOÀN THIỆN ngoài training/fine-tuning.** Có tiến triển thực sự về loader, tài nguyên pilot và một số contract của memory. Tuy nhiên, chưa thể xác nhận hệ thống memory hoạt động trung thực trên LIBERO-Mem thật, hoặc giữ performance LIBERO base. Không nên dùng nhãn `READY` hiện tại của script audit làm điều kiện bắt đầu thí nghiệm chính.

Đây là lần kiểm tra lại [audit 22/10](V2_LIBERO_MEM_COMPLETENESS_AUDIT_2026-10-22.md), đối chiếu [implementation plan](V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md). Phạm vi gồm perception/writer, schema, oracle, renderer, chuẩn bị dữ liệu, simulator/evaluator và rollout. **Không chạy training/fine-tuning, không chạy benchmark bằng policy giả, không sửa production code.**

## 1. Phạm vi source và bằng chứng

- HEAD: `f9f577962119b75bf0b94d54648602d5bc485955`.
- Audit cả working tree: ba file đã sửa nhưng chưa commit (`prepare_libero_mem_sft.py`, `libero_mem_env.py`, `libero_mem_data.py`) và loader/tests/data/config/upstream/pilot artifacts chưa được Git track. Kết luận không chỉ áp dụng cho HEAD.
- Upstream checkout có HEAD đúng `0eee0defa4024e51511b6e15a79f30e9620209de`; checkout có thêm `libero/__init__.py` do loader tạo. Pin đúng chưa chứng minh runtime dùng đúng module hoặc simulator tương đương.
- **Đã đọc toàn bộ bytes và xác minh SHA256 của T1, T3, `metainfo.json`: cả ba khớp nguồn đã pin** `libero-mem/LIBERO-Mem@6df130c05b82e09294fa9534e84a069687b72d33`. Tổng 28.814.124.298 bytes. Provenance của ba file tải về đã được xác minh; vấn đề hiện tại là cách sử dụng dữ liệu.
- Probe chỉ dùng synthetic fixtures để kiểm tra contract. Probe evaluator thực thi chính `_eval_goal` trích AST từ source upstream trên cùng predicate trace; đây chưa phải replay simulator hay kiểm tra toàn bộ lịch gọi evaluator.
- Không sửa hoặc ghi đè evidence ngày 22/10. SHA256 của 13 file production đang audit nằm trong artifact mới.

Artifacts cục bộ:

- [Probe script](../experiments/audits/libero_mem_completeness_20261023/verify_contracts.py) và [kết quả 20 contract probes](../experiments/audits/libero_mem_completeness_20261023/contract_results.json).
- [Unit/regression results](../experiments/audits/libero_mem_completeness_20261023/core_results.txt), [JUnit](../experiments/audits/libero_mem_completeness_20261023/core_results.xml).
- [HDF5 light tests](../experiments/audits/libero_mem_completeness_20261023/hdf5_light_results.txt).
- [Model-import collection failure](../experiments/audits/libero_mem_completeness_20261023/unit_results.txt), [project validation](../experiments/audits/libero_mem_completeness_20261023/project_validation.txt).
- [Output script audit sẵn có](../experiments/audits/libero_mem_completeness_20261023/builtin_audit_report.json).
- [Data evidence: SHA256, split overlap, pilot prompts, image/bbox contract](../experiments/audits/libero_mem_completeness_20261023/data_evidence.json).
- [Verification summary và phiên bản môi trường](../experiments/audits/libero_mem_completeness_20261023/verification_summary.json).

`experiments/audits` bị gitignore: cần đóng gói riêng các artifacts này nếu chuyển evidence sang máy khác.

## 2. Những gì đã sửa được từ lần trước

| Gap cũ | Tiến triển được xác minh | Trạng thái hiện tại |
|---|---|---|
| F01 simulator chỉ là mock | Có `LiberoMemBDDLLoader`, source upstream và đường tạo MuJoCo env | **Chưa đóng:** lỗi backend vẫn bị đổi thành fixture |
| F02 writer tự xác nhận từ proprioception | Không còn tạo initial relation trong constructor; thiếu grounding ban đầu không tăng count | **Một phần:** sau held vẫn thay vị trí vật bằng EEF; thiếu RGB perception |
| F03 policy fallback | Runner từ chối adapter thiếu callable `predict_action_chunk`; kiểm shape `(50,7)` và finite | **Đã sửa fallback action**; các gap rollout khác còn mở |
| F04 evaluator sáu subgoals cho T3 | Local evaluator đổi thành ba milestones và có dwell | **Một phần:** semantics vẫn khác upstream; YAML/test config còn sáu subtasks |
| F05 chỉ có mock data | Có parser HDF5 thật, pilot JSONL và deterministic episode split; observation rỗng bị từ chối | **Một phần:** grounding/oracle sai, split chưa group theo initial state, output chưa đủ multimodal |
| F06 oracle sai nhánh/mất containment | Processor đã gọi `LiberoMemOracleWriter`; containment/occlusion được giữ qua snapshot | **Đã sửa hai lỗi cụ thể**; nguồn oracle và online renderer vẫn sai |
| F07 renderer mất identity/future guard | Giữ `cream cheese A`/`basket B`, có partial-history rendering, schema từ chối future event/relation | **Một phần:** writer không đánh dấu history thiếu; privilege gate còn lọt facts |
| F08 prompt off bị strip | Memory-off giữ nguyên instruction; instruction vượt budget bị từ chối | **Một phần:** chưa dùng exact policy tokenizer trong đường mặc định |
| F09 serialize/mutability | Serialize giữ event `details`; mapping objects/cycles dùng `MappingProxyType` | **Một phần:** nested `details`/metadata vẫn có thể thay đổi |
| F10 readiness chỉ dựa unit fixtures | Tách tên unit contract và upstream readiness | **Chưa đóng:** upstream readiness vẫn chỉ kiểm tra thư mục tồn tại |

Các sửa đổi trên là hữu ích, nhưng không thể suy ra toàn pipeline đã ready từ số file hoặc số unit test pass.

## 3. Các blocker hiện tại

### R01 — Simulator vẫn fallback sang fixture khi được yêu cầu chạy thật

**P0 — chặn benchmark.** [libero_mem_env.py](../src/simulator/libero_mem_env.py) dòng 65–98.

`use_real_sim=True` vẫn bị bao bởi `except Exception`, ghi warning rồi đặt `use_real_sim=False`. Probe ép loader ném lỗi xác nhận constructor vẫn trả object hoạt động ở fixture mode. Khi thiếu checkout, auto mode cũng chọn fixture. Ảnh fixture hiện là màu đồng nhất 128/64; việc đổi ảnh zero thành nonzero không tạo ra quan sát camera thật.

Loader còn áp dụng compatibility shims cho robot/controller giữa robosuite versions, tạo file trong checkout và không kiểm revision/hash/version tại runtime ([libero_mem_loader.py](../src/simulator/libero_mem_loader.py), dòng 30–100). Host đang có robosuite 1.5.2, MuJoCo 3.12.0, bddl 3.6.0, robomimic 0.3.0 và NumPy 2.2.6; không phải reference stack được pin trong AGENTS.md. Chưa có conformance replay cho các thay đổi này.

**Đóng gap:** production constructor phải fail nếu backend/resource/contract không hợp lệ; fixture chỉ nằm ở tests hoặc entrypoint unit-test riêng và không thể qua benchmark runner. Pin runtime, xác nhận assets/controller/camera/state trước khi smoke test thật.

### R02 — Evaluator và reset lifecycle chưa tương đương upstream

**P0 — chưa dùng được metric để kết luận.** [libero_mem_env.py](../src/simulator/libero_mem_env.py) dòng 115–119, 149–178; [libero_mem_metrics.py](../src/evaluation/libero_mem_metrics.py) dòng 64–89, 113–163.

- Real env path bỏ qua `sim_reward`, `sim_done`, `sim_info`, tự tính quan hệ theo khoảng cách và proximity EEF rồi chạy local evaluator.
- Local mặc định cần hai bước lifted và hai bước on-support. Upstream `_eval_goal` có `_sub_goal_live_time > 5` và `_sub_goal_nonlive_time > 15`, cùng trạng thái sequence/OR/overshoot riêng.
- Trên trace `[off, off, on, on]`, probe local trả success/completed=1; source upstream trả false/completed=0. Đây là bằng chứng cụ thể rằng thêm dwell=2 chưa sửa conformance.
- Wrapper dừng ngay khi local full-success, nên không còn bước sau để quan sát overshoot. Không nên đồng nhất success tạm thời với full-sequence validity.
- Reset có initial state chỉ gọi `set_init_state`, không reset upstream timestep/progress. Refiner tác giả gọi `reset()`, `set_init_state(...)`, rồi `reset_subgoal_progress()`. Thiếu metadata thì wrapper đổi sang random reset.
- Chọn target/support qua substring đầu tiên trong raw observations có thể nhầm nhiều plate/basket. Config T3 vẫn ghi sáu subtasks; BDDL thật tại dòng 126–129 có ba `On akita_black_bowl_1 plate_1`. Tách decomposition của writer khỏi evaluator milestones và upstream task index.

**Đóng gap:** dùng upstream evaluator/state theo call schedule đã kiểm chứng; replay T1/T3 với reset lặp, intermediate success, repeated predicates, OR và overshoot. Không tự thay bằng local geometric counter.

### R03 — Predicted writer chưa có perception online và còn xác nhận quá mức bằng chứng

**P0 — chặn đóng góp memory tự chủ.** [object_tracker.py](../src/perception/object_tracker.py) dòng 81–95, 146–159, 203–245; [event_writer.py](../src/memory/event_writer.py) dòng 78–81, 149–166.

Tracker chỉ đọc `detected_objects`; không inference từ `agentview_image`/wrist. Support vẫn mặc định `[-0.05, 0.20, 0.85]`, chưa được grounding theo scene. Observation thật từ wrapper không chứa detections; runner cũng không gọi detector trước writer.

Probe với detection ở lúc departure, sau đó mất detection ở hai frame return, vẫn tạo count=1, visibility=`visible`, event source=`predicted_rgb_tracker`. Nguyên nhân: held được suy từ proximity+gripper, rồi vị trí vật được thay bằng EEF khi mất detection. Đây không phải xác nhận vật đã trở về support. Các signal thành công còn bỏ `has_grounding_evidence`/`evidence_id` đã tính; writer thay bằng True/generic ID, làm mất phân biệt visual evidence với kinematic inference.

`history_complete=True` và `uncertain_count=0` không được cập nhật khi mất quan sát. Probe bỏ đoạn từ step 1 tới 500 rồi thấy lại vật vẫn render exact zero/no-cycle. Trước khi quan sát được vật, snapshot cũng có thể render zero với vị trí “on plate”. Không có pipeline association/re-identification nhiều vật, cập nhật containment hay xử lý object đổi chỗ thực tế.

**Đóng gap:** detector/tracker phải chạy được từ input online; ground target lẫn support; phân biệt observed/inferred/unknown; không dùng kinematic extrapolation làm xác nhận return. Khoảng mất quan sát phải ảnh hưởng độ chắc chắn của count/history. Đánh giá precision/recall, count error và false confirmation trên demo thật; đây là kiểm tra ngoài policy training.

### R04 — Có data thật nhưng memory pilot chưa phản ánh sự kiện

**P0 — chặn sử dụng dataset memory.** [libero_mem_hdf5_loader.py](../src/training/libero_mem_hdf5_loader.py) dòng 101–105, 161–204; [libero_mem_data.py](../src/training/libero_mem_data.py) dòng 79–83, 124, 160–169; [prepare_libero_mem_sft.py](../scripts/prepare_libero_mem_sft.py) dòng 119–147.

- Loader đưa raw `exo_boxes` vào `detected_objects`. Metadata thật dùng key `bowl 1`/`plate 1`, value gồm segmentation ID, normalized 2D bbox và subgoal string; tracker lại đòi `bowl_A -> tọa độ 3D`. Đổi key đơn thuần không giải quyết khác biệt representation hay provenance. Không đưa subgoal string vào nhánh predicted.
- `oracle_signals` được dựng từ EEF/gripper threshold, không phải simulator predicate/contact. Hai demo đầu T1/T3 có EEF z tối thiểu khoảng 0.9365/0.9260, đều cao hơn điều kiện support z≤0.90 của loader; do đó nhánh này không xác nhận return. Gắn tên oracle và xác suất 1 không biến heuristic thành ground truth.
- Hai pilot artifacts hiện có, mỗi nhánh **303 samples / 2 episodes**, đều có `has_memory=true` 303 lần nhưng chỉ **một prompt duy nhất**. Predicted luôn “zero times; no cycle in progress”; oracle luôn “zero times; one cycle in progress”. Đây là kiểm tra artifact đã có, chưa phải đánh giá writer accuracy.
- Parser hash episode ID có task name, chưa group initial-state hash. Đối chiếu 198 demo T1/T3, 100 initial states riêng biệt, thấy 18 nhóm initial state đi qua cả train/val với seed 42, val ratio 0.1. Ví dụ `demo_11` cùng state hash `1406bab655ccd9c8893fd4fad1588dbd56a867c010653dbefd61d334bb110e58`: T1 thuộc train, T3 thuộc val. Điều này vi phạm mục tiêu held-out initial states chung của pilot; test “không trùng episode ID” chưa đủ.
- CLI chỉ lưu prompt/index JSONL: bỏ split, camera/state/action chunks/mask và lineage của memory. Chưa có consumer được xác minh để reconstruct đầy đủ từ source index/hash. Đây chưa phải artifact multimodal hoàn chỉnh.
- Processor vẫn thay camera thiếu bằng zero và quaternion thiếu bằng default; probe input chỉ có EEF/gripper vẫn tạo sample 9D. Adapter SmolVLA đang yêu cầu state 8D `[pos, axis-angle, gripper]`; chưa có cầu chuyển đổi được kiểm chứng từ field 9D `[gripper,pos,quat]` này. Loader còn có nhánh padding axis-angle thành quaternion, không phải phép chuyển rotation hợp lệ.
- Offset 0 của Mem có căn cứ từ thứ tự recorder trước `env.step`. Nhưng ảnh lưu qua `get_visual` trong refiner bị lật ngang (`[:, ::-1]`, dòng 141), parser giữ nguyên ảnh lưu còn wrapper trả raw observation. Chưa có test parity cho transform offline/online. Với base offset +1, processor vẫn truyền `actions[t-1]` cho writer; alignment action history cần source-specific verification.

**Đóng gap:** chuyển annotation sang diagnostic channel đúng provenance; dùng writer online để tạo predicted memory; oracle cần predicate/contact thật hoặc annotation được định nghĩa và kiểm chứng. Group split theo initial state, khóa manifest/hash, lưu đủ payload hoặc nguồn tái tạo kiểm chứng được, và so sánh offline/streaming từng prefix.

### R05 — Privilege gate, oracle và snapshot mới chỉ sửa một phần

**P1.** [libero_mem_text.py](../src/interfaces/libero_mem_text.py) dòng 112–159; [libero_mem_oracle.py](../src/memory/libero_mem_oracle.py) dòng 91–96; [models.py](../src/memory/models.py), các class `V2MemoryEvent`/`SchemaV2EpisodeMemory`.

Renderer lọc source ở events, nhưng containment facts lấy trực tiếp từ ObjectIdentity không có source gate. Count `in_progress` chưa có return event cũng lọt. Probe `allow_oracle=False` vẫn render “one cycle in progress” và “placed in basket B; now hidden” từ oracle. Nếu cycle có cả event allowed/disallowed, điều kiện `any(...)` cũng chưa chứng minh tổng count chỉ dựa trên allowed evidence.

Oracle writer nhận `{}` sau departure và mặc định `is_on_support=True`, tạo return/count=1 không có ground truth. Persistence đã sửa nhưng contract đầu vào chưa fail closed.

Mapping ngoài cùng đã immutable, nhưng sửa `snapshot.events[-1].details['cycle_count']` vẫn đổi cả snapshot sau của writer. Temporal guard mới áp dụng cho event/relation không chứng minh count/object summary có đủ evidence hợp lệ. Cần source/causal provenance ở mọi fact được render và deep immutable snapshot.

### R06 — Online rollout chưa đồng nhất với data preparation và chưa chứng minh base compatibility

**P1.** [libero_mem_rollout.py](../src/evaluation/libero_mem_rollout.py) dòng 64, 73–104, 154–181; [libero_mem_text.py](../src/interfaces/libero_mem_text.py) dòng 47–77.

- O1 online dùng renderer mặc định `allow_oracle=False`; offline processor lại dùng renderer riêng `allow_oracle=True`. Probe online thấy prompt lúc đầu có zero/in-progress, sau một oracle return đã xác nhận lại trở thành instruction trần. Chưa có offline/online parity.
- O1 step 0 tự đưa `is_on_support=True`, không đọc initial predicate thật.
- Runner không gọi `policy.reset()` mỗi episode; adapter hiện có API này. Thiếu `eval_result` vẫn dựng kết quả timeout với total_subgoals=2, thay vì báo thiếu evaluator contract.
- Renderer mặc định không có tokenizer, dùng estimate. Khi có tokenizer, nhánh encode bỏ special tokens và chưa khớp newline/preprocessor policy. Chưa chứng minh ngân sách 48 token thật hoặc không bị model cắt mất fact.
- Runner chỉ giữ prompt history và policy latency; chưa đủ event/evidence trace, tokenizer audit, state/action trajectory, video và resource provenance để review thí nghiệm. Condition/target selection chưa nối task spec; mặc định bowl/plate không dùng được cho T10.

Memory-off pass-through đã được probe xác nhận, nhưng chỉ chứng minh chuỗi instruction không đổi trong case đó. Nó không thay thế kiểm tra preprocessing/action equality, reset, paired rollouts và retention trên LIBERO-10. Không có success rate base/Mem mới được xác minh ở lần audit này.

### R07 — Script audit báo READY quá sớm; environment/test suite chưa green

**P1.** [audit_libero_mem.py](../scripts/audit_libero_mem.py) dòng 174–183, 201–202.

`audit_upstream_environment()` chỉ kiểm tra directory tồn tại rồi trả PASS. Lần chạy này script báo `overall_runtime_readiness=READY` dù chưa tạo simulator, chưa xác minh imports/assets/evaluator, và các probe R01–R06 thất bại. Đây là false-positive readiness gate.

Môi trường host hiện tại chặn import `src.models`: `transformers` yêu cầu `huggingface-hub>=1.5.0,<2.0`, nhưng đang có `2.2.0`. `validate_project.py` thất bại; nhóm regression có `test_rollout_memory.py` dừng ở collection. Không tự nâng/hạ package trong audit vì cần giữ môi trường cô lập và version contract.

## 4. Kết quả kiểm thử lần này

| Kiểm tra | Kết quả và giới hạn |
|---|---|
| Mem core + 4 nhóm memory regression cũ + project structure không import model | **57 passed, 1 deselected**; đặt `LIBERO_MEM_FORCE_FIXTURE=1` chỉ cho unit tests, không phải benchmark |
| HDF5 path/split hash light tests | **2 passed, 1 failed, 2 deselected**; test resolve dùng sai tên T3 (stove/frying pan thay vì task bowl ba lần) |
| 20 probe contract bổ sung | **8 đạt, 12 không đạt**; không phải tỷ lệ hoàn thiện hay success rate |
| Suite có old rollout regression | **Collection error** do dependency mismatch; không tính các test chưa chạy là pass |
| `validate_project.py` | **FAIL** ở import `src.models`; structural checks trước đó pass |
| Script audit sẵn có | In `READY`, nhưng gate không hợp lệ như R07 |
| Real simulator / frozen SmolVLA rollout | **Chưa xác minh**; không tạo benchmark bằng fixture để thay thế |

Lượt chạy gộp ban đầu gặp áp lực bộ nhớ/`MemoryError` trong lúc các tests đọc metadata lớn. Không dùng lượt này để công bố tổng pass/fail vì không có summary hoàn chỉnh. Sau đó tách các kiểm tra nhẹ để có kết quả xác định. Parser và simulator loader đều `json.load` toàn bộ `metainfo.json` khoảng 714 MB; bản Python mở rộng trong RAM lớn hơn file nhiều lần. Cần đọc/index metadata có giới hạn thay vì dựa vào máy nhiều RAM.

Test parser hiện còn assert RGB `(128,128,3)` trong khi HDF5 thật lưu `(256,256,3)`; đây là mismatch được xác minh qua file, chưa tính thành một pytest failure riêng trong lần chạy nhẹ. Test real env cũng `set_init_state()` rồi `reset()`, nên dù pass vẫn không chứng minh initial state đã được giữ.

## 5. Gate phải đóng trước khi gọi phần non-training là hoàn thiện

1. **Backend/evaluator:** bỏ fallback production, pin môi trường cô lập, xác minh resource và exact init state, replay source evaluator với reset/terminal/overshoot. Readiness phải dựa trên evidence artifact, không phải path exists.
2. **Writer thật:** RGB → identity/relations → event evidence; target/support grounded, missing observation thành unknown, count không tăng chỉ nhờ EEF. Kiểm định event/count trên demo thật và lưu lỗi cùng timestamp/frame.
3. **Data/online parity:** predicted và oracle có nguồn đúng, render cùng contract, group split theo initial state, payload đủ, camera/state/action alignment được kiểm chứng. Pilot prompt cần thay đổi theo sự kiện có thể truy vết.
4. **Contract hoàn chỉnh:** deep immutable, source gate ở mọi fact, oracle thiếu key phải fail, actual tokenizer + budget, reset policy và task-specific runner, artifacts đủ để replay/debug.
5. **Chứng minh ngoài training:** shadow writer và frozen-policy integration trên T1/T3; paired memory-off chạy cùng đường preprocessing/action với base. Chỉ sau đó mới đánh giá memory-on và retention theo protocol; không suy ra performance từ unit tests.

**Đánh giá cuối:** nguồn và loader đã tiến lên đáng kể; một số lỗi API của audit cũ đã đóng. Phần còn thiếu tập trung ở tính đúng và trung thực của memory, conformance của simulator/evaluator, dữ liệu và tích hợp runtime. Đây vẫn là các việc cần làm trước training, không thể giải quyết chỉ bằng fine-tuning.
