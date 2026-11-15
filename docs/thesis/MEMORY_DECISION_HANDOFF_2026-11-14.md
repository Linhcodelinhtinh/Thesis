# Bàn giao tích hợp quyết định từ memory

**Cập nhật 15/11/2026.** Shards và alignment audit V2 T1/T3 đã hoàn tất. T7/T8/T9/T10 raw/refined sources đều có receipt xác nhận local SHA-256. RGB relation writer trên demo `demo_1` hoàn tất cho bốn task: T7 425 frame, T8 613 frame, T9 356 frame và T10 309 frame; cả bốn trả 0 fact memory và relation state `unknown` toàn bộ. T8/T9/T10 artifacts nằm tại `experiments/audits/relation-e1-20261115-t8-demo1`, `...-t9-demo1` và `...-t10-demo1`. T10 cũng có 927/927 identity observations ở trạng thái `unknown`. T7 identity observations gồm 92 `tracked`, 648 `ambiguous`, 1.385 `unknown`; phép audit detector-vs-exo-box annotations đã hash-check đủ 425 ảnh: tại IoU 0,5, bowl P/R là 0,893/0,747, plate P/R là 0,971/0,646. T7/T8/T9/T10 đều là demo train; chưa có nhãn blind identity/relation/containment. Artifact F0 V2 no-memory 5.000 bước đã tồn tại trước cập nhật này; B0/F0 runtime checks qua 4/4 lượt, và pilot development B0/F0 hoàn tất 4/4 full-horizon episodes trên một state/task, tất cả 0/4 success. Report giữ nhãn `NON-COMPARABLE_OFFICIAL_PAPER`. F1 V2 tương thích writer relation chưa có, do đó D1/D2 closed-loop vẫn chưa thể nghiệm thu; model training nằm ngoài phạm vi.

Bổ sung runtime 15/11: một author-action replay mỗi T8/T9/T10 dùng environment/state/action/evaluator helpers đã đối chiếu với commit upstream `0eee0defa4024e51511b6e15a79f30e9620209de`, qua camera runtime và cùng reference venv. T9/T10 `demo_1` ghi hai mốc và terminal success; T8 `demo_1` ghi ba mốc sequence nhưng terminal return/done đều false, trái với `success=true` trong metainfo refined cùng state hash. Audit tiếp xác nhận action sau transform và initial state raw/refined trùng chính xác; robot-state replay lần đầu vượt dung sai chẩn đoán `1e-5` tại bước 219, với sai số tuyệt đối lớn nhất 0,008815. Sai khác T8 chưa rõ nguyên nhân và không được coi là nghiệm thu terminal evaluator. Lượt thử T8 ban đầu dùng `ControlEnv` trực tiếp và thiếu wrapper/action transform upstream đã được loại khỏi kết quả; dùng report hậu tố `upstream` làm record duy nhất.

Audit preprocessing ảnh `experiments/audits/smolvla-image-preprocessing-parity-20261115/report.json` đối chiếu manifest/code của F0 với runtime: ảnh Mem refined được hoàn nguyên về raw OpenGL khi chuẩn bị shards; train collator và eval adapter đều gọi `LiberoProcessorStep` LeRobot 0.6.1 để flip H/W. Không thấy sai khác hướng ảnh ở transform đã khai báo. Manifest F0 chỉ ghi version LeRobot, không lưu hash package processor phía train; vì vậy đây là parity theo version/call path, không phải attestation byte-level của hai installation. Không sửa preprocessing hoặc quy nguyên nhân pilot 0/4 cho hướng ảnh.

Giao thức development áp dụng cho F1 trên T1/T3 với khóa seen v2. Giữ P6 F1–F0 là phép so sánh riêng. Mỗi run cơ chế có hash protocol, checkpoint, code, roster và trace. [Giao thức development](../../configs/benchmarks/libero_mem_decision_dev_v1.json).

| Gói | Đã có | Điều kiện còn mở |
|---|---|---|
| D0 | Khóa 20 cụm state seen, 12 development và 8 calibration. Khóa sơ bộ 125 query, tối đa 10 action/chunk, giữ thay đổi 2 quan sát và lịch cố định 8 action. | Đo query, wall-clock và lỗi quyết định trên rollout thật để chốt ngưỡng cho nghiên cứu xác nhận. |
| D1 | Fact tham chiếu kèm version và evidence, tách quyền đọc projected/snapshot, shadow và nhật ký. Unit test so từng prompt, chunk và action với lịch gốc ở cùng RNG. | Kiểm lại parity với F1 thật. Nhãn điểm cam kết trên rollout còn thiếu. |
| D2 | Runner cắt phần chunk chưa chạy và query trên quan sát mới. Lượt hết ngân sách kết thúc với abstention và tính failure. CLI, lịch cố định và tổng hợp paired đã có. | Smoke và pilot vòng kín cần F1 quan hệ P5. Kiểm số query và tổng thời gian thực đo của các arm trước khi diễn giải tác động. |
| D3 | Phép kiểm offline ba giá trị cho intent start/finish đã gán nhãn. Khôi phục uncertainty từ evidence bất hợp lệ trong phép kiểm. | Thiếu nhãn vật đích/hiệu ứng và bộ đọc chunk đã hiệu chuẩn. Chưa cấp quyền reject trong runner. |
| D4 | Đã xác định contract của một chunk và trace từng action. | Cần bộ đọc D3, đánh giá candidate recall và kiểm parity queue/cache/RNG trước khi bật chọn nhiều chunk. |
| D5 | Kiểm hash BDDL T7/T9/T10. T7 raw 46,50 MiB, khóa 120 state; replay RGB chẩn đoán thấy cả hai nhánh Or. T8/T9/T10 raw/refined receipts đã xác minh; author-action replay upstream một demo/task đã chạy. T9/T10 báo terminal success; T8 ba milestone nhưng không terminal. Raw/refined T8 demo_1 có action/state đầu vào trùng chính xác; robot-state drift đầu ở bước 219, nguyên nhân chưa rõ. Writer index vẫn 0 memory, relation unknown; T10 identity unknown 927/927. | T8 terminal/runtime discrepancy, nhãn độc lập identity/containment và protocol vòng kín riêng vẫn mở. |
| D6 | Tổng hợp paired theo state, bounds cho lượt thiếu, bootstrap mô tả trên roster đủ và probe can thiệp quy tắc đếm. | Cần trace đã gán nhãn, dữ liệu vòng kín và protocol xác nhận riêng. |

Metadata quyết định nằm trong một lớp chiếu riêng. Hash của toàn bộ writer/renderer khớp hai manifest index đã replay. Quyền `projected` dùng các fact thực sự có mặt trong prompt tại thời điểm query và các fact sẽ vào projection nếu query lại trên ảnh mới. Quyền `snapshot` đọc thêm trạng thái quan hệ, độ đầy đủ lịch sử và danh tính. Vì vậy replay snapshot có phạm vi thông tin rộng hơn arm projected.

### Kiểm tra offline đã chạy

Replay dùng 198 quỹ đạo mẫu do tác giả phát hành, tổng 42.857 frame. Trace và samples đều được kiểm hash. Báo cáo mốc cũ tham chiếu hash manifest khác nên đã tạo lại mốc từ nhãn gốc và trace hiện tại. Kết quả dưới đây mô tả lịch query giả định khi đường quan sát đã cố định.

| Task | Quỹ đạo | Mốc writer khớp / mốc tác giả | Query lịch gốc | Query theo evidence | Query lịch cố định 8 |
|---|---:|---:|---:|---:|---:|
| T1 | 100 | 68/100 | 1.504 | 1.864 | 1.874 |
| T3 | 98 | 126/294 | 2.845 | 3.546 | 3.550 |

Median writer delay là 9 frame trên các mốc ghép được của cả hai task. T1 có 653 lần query sớm dự kiến và T3 có 1.272. Tín hiệu đổi trạng thái theo dõi xuất hiện trong 367 lần ở T1 và 886 lần ở T3. Đây là các trường hợp cần ưu tiên gán nhãn khi kiểm trigger. Lịch 8 bước là đối chứng sơ bộ theo trace snapshot. Mức tương đương compute trong vòng kín cần được đo trên từng arm, kể cả projected.

Probe offline đổi count vượt ngưỡng, đổi ID và làm mất validity chạy trên 95 fact T1 và 93 fact T3 đủ điều kiện. Quy tắc đếm đổi kết quả khi count đổi, giữ kết quả khi đổi tên ID, và trả unknown khi evidence mất hiệu lực. Các cửa sổ này đến từ replay, với intent start được áp đặt để thử quy tắc. Cần nhãn độc lập và hành động thật để đo dự báo vật đích, lỗi quyết định hoặc success.

Artifact nội bộ nằm trong `experiments/audits/relation-e1-20261113-232401`, các file `decision-dev-v1-final-T1.json`, `decision-dev-v1-final-T3.json` và `decision-rule-v1-*.json`. Thư mục preflight `experiments/evaluation/decision-f1-checkpoint-gate-20261114` lưu lỗi khi đối chiếu writer hash của export lịch sử. Hai export local kiểm được đều dùng `rgb_cycle_v3`. Môi trường training riêng hiện chưa có Torch, LeRobot và PEFT. Máy có một RTX 4060 Laptop 8 GiB. Dùng smoke một GPU hoặc máy training riêng và ghi recipe của cả hai arm.

P5 đã tạo bộ shards lossless chung tại `experiments/training/relations-author-split-v2/shards`. Tổng dung lượng 11,83 GiB, gồm 698 quỹ đạo và 180.447 cặp observation–action. Có 598 quỹ đạo train với 152.802 cặp. Hai index có sẵn đã qua kiểm nguồn, writer, tokenizer và phủ đúng khóa v2 trước khi tạo shards. Audit alignment qua 180.447 cặp state/action, 2.094 frame mỗi camera và 1.396 mẫu parity với adapter. Báo cáo nằm cạnh thư mục shards.

T7 replay 12 quỹ đạo raw trong simulator khóa cho thấy 2 lượt hoàn tất nhánh thứ nhất và 3 lượt hoàn tất nhánh thứ hai khi tắt quan sát RGB để đo evaluator. Bảy lượt chưa hoàn tất. Các mốc ghi được xuất hiện trong `env.step` vì upstream có thể commit sau khi bộ đếm đạt ngưỡng. Hai nhánh cũng đã hoàn tất trên replay có RGB với action tác giả. Kết quả dùng để kiểm reset, tiến trình và nhánh Or. Artifact chi tiết là `experiments/audits/t7-raw-branch-timing-20261114.json` và các báo cáo `t7-raw-rgb-branch*-20261114.json`.

## Replay author-action T8–T10 ngày 15/11

Script `scripts/audit_libero_mem_author_replay.py` dùng raw receipt từng task, giới hạn `demo_1`, xác minh BDDL/revision/package và so sánh file helper/evaluator với commit upstream. Vì reference venv không có TensorFlow, script nạp chính xác định nghĩa AST của `get_libero_env`, wrapper deterministic, dummy action và `set_init_state` từ các file đã hash-check; các import TensorFlow không liên quan tới environment bị bỏ qua. Action handling giữ đúng upstream: bỏ leading no-op theo ngưỡng `1e-4`, thêm 20 vector zero cuối và map gripper `0` sang `-1`.

| Task | State/action | Kết quả evaluator upstream |
|---|---|---|
| T8 | `demo_1`, 613 action sau transform, state SHA `6a07dfff4588ef8fe515f74c0b0c0b9997504d23b3a67a5900f2bf966ff6330d` | Ghi 3 milestone; `last_upstream_return=false`, `done=false`, overshot false. Raw/refined action/state đầu vào trùng; robot-state replay vượt `1e-5` lần đầu ở bước 219. Metainfo refined ghi success true; nguyên nhân drift chưa rõ. |
| T9 | `demo_1`, 356 action, state SHA `610f476016abbc2d0ee8e326f5f746621df2915a13addd43201602811f56f1ce` | Ghi 2 milestone; terminal return/done true; overshot false. |
| T10 | `demo_1`, 309 action, cùng state SHA T9 | Ghi 2 milestone; terminal return/done true; overshot false. |

Reports là `experiments/audits/t8-author-replay-upstream-20261115.json`, `.../t9-author-replay-upstream-20261115.json` và `.../t10-author-replay-upstream-20261115.json`. So sánh T8 robot-state trace nằm trong `experiments/audits/t8-replay-state-parity-20261115.json`. Đây chỉ là một author-action trajectory mỗi task, không kiểm tra policy, perception, relation/containment accuracy hay memory effect. T8 vẫn mở cho tới khi nguyên nhân được xác định; không thay action, state, controller hoặc evaluator để làm kết quả khớp metainfo.

### Nhãn cần thu trên seen

Mỗi dòng JSONL gắn `task`, `initial_state_sha256`, `policy_seed`, `decision_arm`, `episode_id`, `query_id`, `chunk_sha256`, `observation_step` và `annotation_source_sha256`. Người gán nhãn xem camera, instruction và prefix được đề xuất. Ghi `intent` với giá trị start_cycle, finish, approach, grasp, release, other hoặc unknown. Ghi `target` theo vật/vùng quan sát được hoặc null, `commit_step` theo bước sớm nhất có thể đánh giá hoặc null, `label_status` với clear, ambiguous hoặc unavailable. Dùng `executed` để phân biệt chunk có hậu quả quan sát được và ứng viên chỉ được gán nhãn ý định. Lưu `observed_effect` và nguồn nhãn hậu quả khi đã thi hành. Đối với trigger, ghi cửa sổ quan sát trong đó dữ kiện mới thực sự ảnh hưởng tới lựa chọn và phân loại lỗi tiến trình, danh tính hoặc thực thi. Giữ nhãn độc lập với dự đoán writer/selector. Giữ nguyên cụm state qua T1/T3 và qua các seed khi chia dữ liệu.

### Việc chạy tiếp theo

1. Trong phạm vi hiện tại, giữ nguyên artifact F0 V2 đã có và không chạy training. Bằng chứng B0/F0 E0 được lưu ở `experiments/evaluation/e0-f0-runtime-ext4-20261115` và `experiments/evaluation/e0-b0-f0-pilot-20261115`; cả bốn full-horizon episode hoàn tất nhưng không ghi nhận subgoal. F1 V2 và parity giữa F0/F1 vì vậy vẫn là gate mở. Nếu giai đoạn sau cho phép training, bàn giao P5–P6 có recipe và thứ tự smoke/reload tương ứng. [Bàn giao P5–P6](P5_P6_HANDOFF_2026-11-13.md).
2. Sau khi có F1 đạt preflight, chạy từng arm trên một state mỗi task. So trace text/shadow và kiểm điểm cắt prefix của event. Sau đó mở các lô seen đã khóa để gán nhãn, đo trigger, coverage, success và compute. Với các lượt dành cho gán nhãn, thay `--no-save-video` bằng `--save-video` đồng nhất giữa các arm.
3. Dùng T7 raw để kiểm evaluator trên quỹ đạo tác giả. Helper upstream lọc bằng quan hệ tập con của predicate và nhánh có thể cập nhật bộ đếm trong một lần gọi Or. Cần xác minh timing cả hai nhánh trước khi dùng một quy tắc kết thúc riêng. Tải T7 refined khi cần audit RGB. [Bàn giao nguồn](P0_P2_HANDOFF_2026-11-13.md). [Metadata hai bản T7](../../resources/manifests/libero_mem_t7_refined_source.json). [Khóa 120 trạng thái raw](../../resources/manifests/libero_mem_t7_raw_states_v1.json).

```powershell
$Policy = (Get-Command python).Source
$Detector = (Resolve-Path .venv_memory/Scripts/python.exe).Path
$Sim = '/mnt/d/Thesis_26/.venv_libero_reference/bin/python'
$F1 = Read-Host 'Đường dẫn pretrained_model của F1 huấn luyện bằng writer quan hệ'
$Campaign = "experiments/evaluation/decision-dev-$(Get-Date -Format yyyyMMdd-HHmmss)"
$Runs = @()
foreach ($Mode in @('text','shadow','event','periodic')) {
    $Run = "$Campaign/$Mode"
    & $Policy scripts/evaluate_mem_adapter.py --use-wsl --sim-python $Sim --detector-python $Detector --adapter-dir $F1 --conditions F1 --tasks T1 T3 --split seen --states 0 --policy-seed 42 --decision-mode $Mode --decision-exposure projected --no-save-video --stream-memory-trace --output-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Decision rollout failed in $Run" }
    $Runs += $Run
}
& $Policy scripts/aggregate_memory_decisions.py --runs $Runs --tasks T1 T3 --states 0 --seeds 42 --output "$Campaign/paired.json"
if ($LASTEXITCODE -ne 0) { throw 'Decision aggregation failed' }
```

Mở rộng snapshot bằng lượt event và shadow riêng, cùng seed và roster. Truyền tên arm tương ứng cho aggregator. Các arm cùng checkpoint và provenance. Giữ thư mục lỗi và dùng thư mục mới cho lượt thử lại. Với P6, dùng CLI theo bàn giao P5–P6 với protocol gốc.

## Chẩn đoán writer T7 ngày 14/11

Trong `demo_1`, detector trả hai plate mỗi frame trên 20 observation đầu, còn task profile yêu cầu neo ba plate. `RelationTracker` chỉ tạo anchor khi thấy đủ instance cùng lớp trước observation 20; do đó anchor plate không khởi tạo ở demo này. Không điều chỉnh threshold hoặc tracker theo suy đoán: cần nhãn độc lập hoặc một ROI/camera khác đã được khóa rồi mới đánh đổi recall và false positive. Snapshot collector giữ per-frame detection counts, state trace, prompt samples, source receipt và hashes.
