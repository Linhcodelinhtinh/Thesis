# Rà soát LIBERO-Mem-derived — 11/11/2026

**Kết luận:** Có cơ sở triển khai một benchmark derived để so sánh nội bộ. Hệ thống hiện có evaluator/rollout T1/T3 và kết quả policy thật; phần tái tạo demonstration và nối vào training chưa hoàn chỉnh. Chưa có cơ sở gọi hệ thống “100% reproducible” hoặc “100% fair”.

Audit HEAD `933ad9e` cùng working tree đang được chỉnh sửa. Snapshot lúc **15:56:21, UTC+7, ngày 11/11/2026** nằm trong [evidence.json](../experiments/audits/libero_mem_derived_20261111_review/evidence.json); hash và bản sao source được giữ cùng thư mục. Đây là inspection source, existing artifacts và software tests; không chạy thêm policy benchmark hoặc training. Thư mục experiments có thể bị gitignore, cần giữ artifact riêng khi chia sẻ báo cáo.

## 1. Ý nghĩa khoa học của hướng derived

Có thể công bố một phiên bản môi trường/task/protocol có version riêng và đo lại các model trực tiếp trong phiên bản đó. Tên phù hợp với phạm vi hiện tại là **LIBERO-Mem-derived T1/T3**, cùng nhãn `NON-COMPARABLE_OFFICIAL_PAPER`. Kết quả chỉ áp dụng cho task, checkpoint, dữ liệu và ngân sách được khai báo; không đại diện toàn bộ mười task hoặc năng lực memory nói chung.

Không có tính tất định tuyệt đối vẫn có thể có đánh giá tái lập về mặt thống kê. Cần khóa simulator/assets/controller, reset/warmup, evaluator, observation, action interface, horizon, lịch thực thi action, danh sách state và quy tắc kết thúc. Chạy lặp để đo biến động; dùng cùng danh sách state cho mọi model, báo kết quả từng task và chênh lệch theo cặp. Bootstrap/khoảng bất định phải tính theo nhóm initial state, không coi các lần lặp của cùng state hoặc T1/T3 chung state là mẫu độc lập.

Mỗi model giữ preprocessing/postprocessing đúng checkpoint của nó và nhận cùng nguồn thông tin được phép. Khi so sánh fine-tuning giữa kiến trúc, dùng cùng dữ liệu, split và ngân sách lựa chọn hyperparameter trên validation; không buộc cùng learning rate rồi coi đó là đủ công bằng. So sánh memory trong cùng backbone cần đối chứng được train không memory trên cùng demonstration và ngân sách.

Repo tác giả [ghi nhận replay có thể khác dù cố định initial state và seed](https://github.com/libero-mem/libero-mem#7--refine-with-object-centric-labels). Điều này không chứng minh toàn benchmark không thể tái lập. Evidence local cụ thể hơn: [diagnostic author runtime](../scratch/libero_mem_author_candidate/bottle_warmup/diagnostic.json) dùng MuJoCo 3.2.2, robosuite vendored 1.4.1, Python 3.10.22 do project chọn; restore state sai số 0 nhưng chai sau warmup không khớp ảnh refined của `demo_33`. [Lịch sử runtime](LIBERO_MEM_AUTHOR_RUNTIME.md) chưa xác định được nguyên nhân. Không được suy từ đó rằng chắc chắn tác giả sửa geometry hoặc dùng cấu hình bí mật.

## 2. Fine-tune với dữ liệu có chai đổ

Nếu môi trường derived giữ cơ chế settling đó, training demonstration thu/replay trong cùng môi trường là lựa chọn hợp lý để giảm lệch phân phối. Giữ ba nhánh: B0 frozen; F0 fine-tuned không memory; F1 fine-tuned có memory, F0/F1 khởi đầu cùng checkpoint. B0–F0 đo adaptation; F0–F1 đo tác động của hệ thống memory khi dữ liệu/ngân sách được kiểm soát. B0–F1 hiện chỉ đo tổng thay đổi của adaptation và memory/perception.

Replay action cũ chỉ tạo demonstration phù hợp nếu trajectory mới thật sự giải quyết task. Khi physics/contact khác, action cũ có thể trở thành target không phù hợp: phải kiểm tra outcome, alignment, các lần overshoot và thống kê toàn bộ episode bị loại theo quy tắc cố định. Nếu phải thu lại demonstration, lưu provenance rõ. Không dùng outcome của policy ứng viên để chọn test states.

Tạo lại RGB, proprioception, action alignment và predicted memory từ chính trajectory derived. Annotation hoặc prompt memory của trajectory gốc không tự động đúng cho trajectory mới. Kiểm tra chai không vô tình trở thành tín hiệu thời gian/tiến độ hoặc vật cản làm thay đổi câu hỏi về memory; đây là kiểm tra cần thực hiện, chưa phải lỗi đã chứng minh.

Giữ test ngoài chọn checkpoint, tuning và tạo demonstration train. Test hiện tại đã được xem nhiều lần; nếu tiếp tục thiết kế dựa trên các failure đó, nên mô tả là evaluation phục vụ phát triển và dành thêm holdout chưa xem cho kết luận cuối. Không tự phân lại split cũ để có kết quả đẹp.

## 3. Những gì đã được xác minh

- `LiberoMemEnv` dùng simulator thật, từ chối fixture, khôi phục vector từ split lock, warmup 20 bước, dùng upstream predicates/progress. Runtime khóa robosuite 1.4.0 / MuJoCo 3.2.3, 20 Hz. Protocol dùng 1.000 policy actions, chunk 50, execution horizon 10. Đây là lựa chọn local được ghi rõ.
- Generator mới đã bổ sung `robot_states` 9D, companion `metainfo.json`, validation schema và success gồm đủ milestones + final predicate + không overshot. Hai lỗi thiếu schema và success quá lỏng trong lần rà trước đã được sửa trong source; điều đó chưa chứng minh export hoàn tất hoặc training đọc đúng ảnh.
- Khi đọc trạng thái WSL, tiến trình `generate_derived_demonstrations.py --tasks T3 --split train` đang chạy, ghi vào `/tmp/libero_mem_staging`. Thư mục cuối `data/libero_mem_derived` đang trống ở mốc snapshot. Không đọc file HDF5 đang mở để kết luận corruption. Artifact T1 78/80 ở lần rà trước không còn là bản xuất cuối hiện tại được xác nhận.
- Có bộ test **đã hoàn tất** tại `experiments/evaluation/results_mem_2/step5_test_20261109-093744`, không chỉ smoke trong thư mục tên derived. Đã đọc đủ summary của 20 episode mỗi condition và đối chiếu protocol hash, split hash, roster, checkpoint gốc, policy/simulator environment, policy implementation và processor hash. Các mục đó khớp. Snapshot source khác ở ba script launcher/diagnostic, được liệt kê trong evidence.

| Condition | N task-episodes | Success | Macro SR | Macro mean SCR | Overshot |
|---|---:|---:|---:|---:|---:|
| B0 | 20 | 0 | 0% | 0.0500 | 1/20 |
| F1 | 20 | 1 | 5% | 0.1333 | 1/20 |

Chi tiết: T1 B0/F1 đều 0/10 success và SCR 0.1; T3 B0 0/10, SCR 0; T3 F1 1/10, SCR 1/6. Hai task dùng chung **10 initial-state hashes**, policy seed 42. Đây là kết quả lịch sử một seed, chưa chứng minh cải thiện đáng tin cậy hoặc hiệu quả riêng của memory.

Hash `dataset-manifest.json` của training khớp `data_manifest_sha256` trong adapter. Manifest trỏ tới HDF5 tác giả đã pin và convention refined. Vì vậy F1 hiện có **chưa được fine-tune trên dataset derived đang tạo**. Rà soát này không hash lại toàn bộ hàng chục GB source/shards/weights.

Run `libero_mem_derived_20261110-225618` riêng có B0 hoàn tất 2/2 state-0 episodes và F1 còn manifest `running`, không có summary hoàn tất. Không thấy tiến trình evaluate tương ứng trong snapshot WSL; trạng thái JSON `running` không đủ chứng minh job còn sống.

## 4. Các vấn đề còn mở

**P1 — Convention ảnh giữa generator và training chưa khớp.**

[Generator](../scripts/generate_derived_demonstrations.py#L225) ghi nguyên ảnh raw từ env. `canonical_raw_rgb(images, "mem")` trong [mixed_sft.py](../src/training/mixed_sft.py#L15) lại lật dọc vì giả định input là author-refined. `prepare_libero_mem_sft.py` cũng khóa writer ở `libero_mem_refined`, trong khi online writer dùng `libero_raw`. Probe thực thi chính hàm conversion trên mảng bất đối xứng xác nhận ảnh raw bị lật. Cần convention explicit theo nguồn, giữ ảnh đến model và detector nhất quán offline/online, cùng regression trên ảnh thật. HDF5 load được không đồng nghĩa preprocessing đúng.

**P1 — Source lock và đường preparation chưa hỗ trợ dataset derived.**

[prepare_mixed_memory_sft.py](../scripts/prepare_mixed_memory_sft.py#L65) kiểm hash source theo lock tác giả trước khi tạo shard. [lock_memory_sft_split.py](../scripts/lock_memory_sft_split.py#L18) cũng chỉ nhận đúng hash release tác giả; `prepare_mem_adapter_data.py` mặc định/hardcode nguồn original. Chỉ đổi thư mục input sẽ bị từ chối; bypass kiểm hash sẽ mất provenance. Cần manifest derived riêng có parent hashes, generator/protocol/runtime/asset hashes, trạng thái hoàn tất và episode inventory, giữ nguyên assignment state của lock gốc. Cần quyết định rõ artifact train-only hay đủ ba split vì preparation hiện yêu cầu mọi episode trong lock.

**P1 — Resume chưa bảo đảm provenance và metadata đầy đủ.**

[Nhánh resume](../scripts/generate_derived_demonstrations.py#L195) chỉ kiểm demo có trong HDF5 và sidecar, rồi skip. Không kiểm hash input/protocol/code, schema hay `require-success` mới. Companion metadata chỉ được ghi cuối task, trong khi sidecar được ghi từng episode. Nếu dừng giữa task trước lần ghi metadata đầu, khi resume các demo đã có bị skip mà không dựng lại entry trong `metainfo.json`. Artifact có thể đủ HDF5 nhưng thiếu metadata cho loader. Staging dùng tên task, không có version/config identity; các lần chạy khác cấu hình có thể trộn dữ liệu.

**P1 — Không xử lý terminal giữa demonstration.**

[Vòng replay](../scripts/generate_derived_demonstrations.py#L222) chạy hết action và bỏ qua `d`. Env đánh dấu terminal khi overshot hoặc đạt horizon, sau đó từ chối action tiếp theo. Với demo overshoot trước action cuối, batch sẽ ném lỗi trước khi ghi failure và chuyển demo. `--require-success` chỉ chạy sau vòng lặp nên không xử lý được case đó. Cần ghi failure/partial outcome rõ, giới hạn action theo episode lifecycle và không lặng lẽ cắt trajectory rồi coi đủ dữ liệu.

**P2 — Một số trường và trạng thái hoàn tất chưa trung thực.**

Env observation allowlist không chứa `robot0_joint_pos`; generator vì thế ghi `joint_states=np.zeros(7)`. Đây không phải joint state được đo, dù model hiện dùng robot state khác. Generator còn skip source demo thiếu bằng warning, có fallback instruction bắt mọi exception, không kiểm hash source bytes trước replay, không lưu đầy đủ parent/runtime/code identity vào manifest. `flush()` không biến HDF5 update và JSON sidecar thành một transaction; copy ra tên cuối cũng chưa atomic. Cần fail/incomplete rõ và chỉ publish artifact sau validation.

**P2 — Spec chưa khớp protocol và evidence.**

[LIBERO_MEM_DERIVED_SPEC.md](LIBERO_MEM_DERIVED_SPEC.md) vẫn ghi horizon 150/300 thay vì 1.000; T1 hai subgoals thay vì một return milestone; T3 mô tả swap nhiều bowl thay vì nhấc/đặt cùng bowl ba lần. Dấu gripper bị đảo: source Panda ghi `-1 => open, 1 => closed`. Counter upstream là cumulative giữa milestones, không phải consecutive dwell như spec. State lock được build từ metadata HDF5, không có chứng minh chúng được lấy trực tiếp từ `.pruned_init` như spec nói. Các tuyên bố 100% và quy nguyên nhân mismatch cho cấu hình/asset không công bố cũng chưa được evidence hỗ trợ.

Table 4 của [paper v1](https://arxiv.org/html/2511.11478v1#S5.T4) ghi average lần lượt 5.0%, 0%, 5.0%, 14.8%; bảng historical context trong spec đang ghi sai SlotVLA h=1 và h=8. Đây là macro trên mười task của paper, không so ngang với macro T1/T3 local.

**P2 — Chưa là harness nhiều model.**

CLI `evaluate_mem_adapter.py` chỉ nhận B0/F1 và tạo SmolVLA adapter. MiniVLA/OpenVLA trong spec mới là candidate; runner chưa có đối chứng F0 hoàn chỉnh cho nhánh adapter riêng và chưa có training artifact derived. Phần này cần triển khai trước khi gọi là bảng so sánh nhiều model.

## 5. Kiểm tra đã chạy và bước tiếp theo

Trong `.venv_memory_wsl`, đã chạy năm file unit tests: `test_derived_generator_unit`, `test_libero_mem_evaluator`, `test_mem_evaluation_cli`, `test_mixed_memory_sft`, `test_libero_mem_rollout`. Kết quả **35 passed, 2 skipped**, lưu tại [pytest.txt](../experiments/audits/libero_mem_derived_20261111_review/pytest.txt) và [JUnit](../experiments/audits/libero_mem_derived_20261111_review/pytest.xml). Không cài/thay dependencies.

Hai test skip là encode/decode video do thiếu `imageio`, và Accelerate sharding/resume do thiếu `accelerate` trong môi trường kiểm tra. Source hashes không thay đổi từ snapshot đến khi kết thúc audit.

Tests derived hiện tự dựng HDF5 fixture và tự tính biểu thức success; chúng không gọi generator thật. Chúng không phát hiện mismatch convention ảnh, mất metadata khi resume hoặc bỏ qua terminal trong batch. Vì vậy không dùng kết quả unit test để chứng nhận dataset hay simulator reproducibility.

Ưu tiên hoàn thiện: (1) sửa và khóa spec/protocol; (2) sửa convention ảnh, source manifest và preparation derived; (3) làm generator có resume/terminal/completion đúng rồi kiểm export thật; (4) tái tạo predicted memory và train F0/F1 cùng dữ liệu; (5) mở harness model, chạy paired multi-seed và báo uncertainty/retention. Kết quả B0/F1 cũ giữ vai trò evidence phát triển, không đổi nhãn thành kết quả của model train trên data derived.

## 6. Implementation update sau snapshot audit

Các bước code cần thiết cho giai đoạn chuẩn bị data đã được sửa trong working tree sau snapshot trên:

- RGB raw từ replay và RGB author-refined giờ có convention tường minh trong memory writer, HDF5 source manifest và mixed shard preparation. Derived training input giữ raw OpenGL; author-refined được lật về raw đúng một lần.
- `generate_derived_demonstrations.py` dùng task roster trong protocol (parent lock không có `task_alias`), kiểm tra runtime/source hashes, lưu provenance code và runtime, xử lý terminal trước hết trajectory, giữ nguyên trạng thái replay đã bị loại khi resume, và yêu cầu HDF5/metainfo khớp roster trước khi publish. Output mặc định được cách ly ở `data/libero_mem_derived_v1` cùng staging v1.
- `lock_libero_mem_derived_split.py` xác nhận đủ sáu roster đầy đủ, `--require-success`, hash artifact cuối, khớp action nguồn và initial state với parent lock, rồi chỉ đưa replay thành công vào lock training mới. Parent lock evaluation không bị sửa.
- `prepare_libero_mem_sft.py` và `prepare_mixed_memory_sft.py` nhận nguồn derived raw theo lock mới; dependency `ijson` được pin trong simulator environment để đọc metainfo nguồn lớn theo streaming.
- Spec đã sửa nội dung task, dấu gripper, horizon, convention ảnh và tình trạng score/harness. Fine-tuning derived chưa chạy.

Đây là trạng thái **implement trong source**, không phải xác nhận export hay benchmark. Chưa chạy generator trên simulator, chưa tạo derived lock/index/shards, chưa fine-tune và chưa chạy policy evaluation. Chỉ kiểm tra tĩnh cú pháp sau các sửa đổi; không chạy lại test suite trong lượt này. `data/libero_mem_derived_v1` chưa có manifest khi rà soát này kết thúc. Vì vậy benchmark vẫn cần một lần export thật được audit trước khi dùng để train hoặc so model. Harness hiện cũng chỉ hỗ trợ B0/F1; F0 và nhiều kiến trúc, multi-seed paired evaluation vẫn là các bước nghiên cứu kế tiếp.
