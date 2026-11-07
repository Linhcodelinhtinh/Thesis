# Kế hoạch test checkpoint `results_mem_2`

Lập plan: 2026-11-06. Cập nhật CLI: 2026-11-07.
Trạng thái: **CLI đã triển khai và qua software tests; chưa có paired rollout hoàn tất**.
Phạm vi: custom V2; giữ nguyên baseline V1. Mọi rollout Mem mang nhãn
`CUSTOM_V2_LIBERO_MEM` và `NON-COMPARABLE_OFFICIAL_PAPER`.

Kiểm tra sửa CLI ngày 2026-11-07: 12 test CLI/artifact, 3 test runner và 7 test
SmolVLA adapter đã pass qua các lượt chạy WSL; 7 test cấu trúc và
`validate_project.py` pass. Test video có encode/decode artifact. Đây là software
tests, không dùng fixture policy để tạo benchmark score. Lượt WSL thật
`runtime_t3_final_20261107` qua preflight nhưng tiến trình thoát mã 1 tại phase
`Loading policy runtime libraries`, không traceback và không episode hoàn tất.
Nguyên nhân lần thoát này chưa được xác định; không tính runtime gate là pass.

Cách nạp và đưa memory vào policy được ghi tại
[hướng dẫn sử dụng Mem](MEM_ADAPTER_V1.md#cách-sử-dụng-mem--cập-nhật-2026-11-06)
và [API wiring](MEM_ADAPTER_V1.md#api-wiring-cho-adapter-riêng).
Memory là suffix do renderer tạo từ causal RGB writer, reset từng episode;
không nhập count thủ công hoặc lấy progress của evaluator làm prompt.
Ví dụ API là hướng dẫn tích hợp; dùng CLI bên dưới để chạy các gate và lưu artifacts.

## 1. Checkpoint và bằng chứng đã kiểm tra

Candidate đầu tiên:

```text
experiments/training/results_mem_2/
  mem-adapter-20261105-083543/pretrained_model/
```

Đây là **unmerged PEFT LoRA adapter**, không phải full model. Nạp cùng
`resources/checkpoints/smolvla_libero`, revision gốc
`31d453f7edd78c839a8bbc39744a292686daf0de`.

| Mục | Kết quả inspection local |
|---|---|
| Training | Run kết thúc tại step 3000; có training checkpoints 500/1000/1500/2000/2500/3000 |
| Adapter | Attention q/v expert LoRA, rank 16, 645,120 tham số; 64 tensor FP32 |
| Artifact integrity | SHA256 của cả 4 file export khớp `mem_adapter_manifest.json` |
| Adapter weights SHA256 | `da0f89ea1228cdb698ef531b77ff084f7060ce91c9f85cbde7965b6f2f565c1b` |
| Base weights SHA256 | `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`, khớp file local |
| Normalizers | Hai file normalizer/unnormalizer local khớp manifest; state 8, action 7 |
| Provenance | Dataset manifest hash, 6 implementation hashes được run ghi lại và writer code hashes khớp file local tại lúc inspection |
| Training data | 10 task LIBERO-10 + Mem T1/T3; base/Mem sampling 50/50; predicted RGB writer v3; không memory dropout |
| Split | 698 demos: 558 train, 70 validation, 70 test; grouping theo hash toàn bộ initial-state vector |
| Mem split | T1: 80/10/10; T3: 78/10/10; thứ tự train/val/test |
| Base split | 400/50/50 trên đủ 10 task |
| Validation step 3000 | Mem flow loss: memory on **0.23024**, memory off **0.33554**; base **0.42984** |

Loss trên chỉ là offline diagnostic của cùng model, **không phải success rate**,
không phải đối chứng F0 được train riêng và không chứng minh step 3000 tốt nhất.
Chưa kiểm tra toàn bộ dữ liệu raw/shard hoặc thực hiện export/reload inference.
T1/T3 dùng cùng 10 state hashes ở mỗi held-out split: 20 task-episodes test chỉ
tương ứng **10 initial states độc lập**. Validation demo IDs là
`33,35,58,59,75,91,93,94,95,99`; test là `8,15,29,34,53,60,61,74,78,82`.

Nguồn local: `run_config.json`, `metrics.jsonl`, `checkpoint-003000/progress.json`,
`pretrained_model/{adapter_config,mem_adapter_manifest}.json` trong run;
`mem-adapter-v1/{dataset-manifest,dataset-split.lock,source-verified}.json`.
Working tree đang có thay đổi của người dùng; lưu diff/source hashes khi triển khai,
không chỉ ghi HEAD `1978f05b3519fa9af63e1dcc5b02e715083ff6ef` làm identity của code.

## 2. Các việc phải hoàn tất trước khi chạy batch

1. **CLI, bridge và logging đã được triển khai; vẫn phải qua runtime/conformance gates.**
   `scripts/evaluate_mem_adapter.py` gọi `LiberoMemRolloutRunner`; các launcher
   `run_mem_validation_and_test.sh/.ps1` gọi CLI đó. Instruction lấy từ BDDL
   đã khóa hash; F1 nhận explicit base checkpoint. Có logging
   action/video/memory trace/provenance và failure status. `scripts/run_benchmark.py` và
   `scripts/run_episode.py` vẫn dùng pipeline base/legacy memory; không thể chỉ
   đổi `--checkpoint` thành adapter này để chạy F1.
2. **Cần bridge giữa simulator và policy.** Mem loader khóa numpy 1.22.4,
   robosuite 1.4.0, bddl 1.0.1, robomimic 0.2.0, mujoco 3.2.3; policy có dependency
   khác. Đã có detector subprocess và simulator–policy IPC mới
   (`libero_mem_sim_worker.py`, `process_libero_mem_env.py`), chưa được xác minh
   end-to-end trên model và simulator thật.
   Dùng process riêng cho simulator, policy và perception; transport giữ nguyên
   dtype/shape/camera/timestamp/action và chỉ đưa allowlist observations đến policy.
   Không gộp dependency training vào simulator hoặc sửa protocol để né version check.
3. **Validator phải đúng loại export.** Dùng
   `src.training.mem_adapter_artifact.validate_adapter(adapter_path, base_path)`;
   `scripts/validate_memory_sft_export.py` dành cho merged export với
   `export_manifest.json`, không phải artifact này.
4. **F1-off và adapter-on base chưa được hỗ trợ.**
   `SmolVLAAdapter.preprocess()` chặn adapter chưa bind writer/renderer;
   runner chỉ bind ở condition F1. Muốn làm ablation phải bổ sung mode tường minh,
   giữ kiểm tra artifact/base/tokenizer, test instruction pass-through và reset.
   Không gán cờ verified thủ công, không dùng writer giả để vượt guard.
5. **Runtime phải được xác minh bằng inference thật.** Training ghi
   torch `2.11.0+cu130`, PEFT `0.21.2`, accelerate `1.15.0`, h5py `3.14.0`,
   khác một số pin trong `envs/mem_adapter_training.txt`. Ghi resolved versions,
   official policy source hash, GPU/driver/dtype; không coi môi trường có sẵn là tương đương.
   Metadata trên đĩa của `.venv_smolvla` còn dùng LeRobot 0.4.4, torch 2.5.1+cu121,
   transformers 4.57.6; PEFT 0.18.0 đã được bổ sung để nạp adapter. Metadata
   package không thay thế runtime probe.
   `environment_packages.txt` của training liệt kê cả NumPy 1.26.4 và 2.1.3,
   cần xác định version thực được import, không cài nguyên freeze mâu thuẫn.
6. **Training assets lock chưa được mang về đầy đủ.** Bundle không có `assets.json`
   có hash `938b2b074fc36742c9d05022bcdf6628dd722f9e5ed21bb954c3ad338e0d0491`
   được run tham chiếu; assets lock local gần đó có hash khác. Base weights và
   normalizers đã khớp độc lập, nhưng vẫn phải đối chiếu config/backbone/processors
   với nguồn đã pin trước inference. Không coi toàn bộ preprocessing đã được
   chứng nhận chỉ từ hai file normalizer.

## 3. Ma trận so sánh

| Nhánh | Weights | Prompt khi rollout Mem | Mục đích / điều kiện |
|---|---|---|---|
| B0 | SmolVLA gốc | Instruction gốc, memory off | Đối chứng bắt buộc, cùng protocol với F1 |
| F1-3000 | Base + adapter export mới | Predicted RGB writer v3 + renderer đã pin | Candidate bắt buộc |
| B1-predicted | SmolVLA gốc | Cùng writer/renderer với F1 | Đối chứng bổ sung có thể dùng API F1; tách ảnh hưởng thêm adapter khi cùng memory pipeline |
| F1-off | Cùng adapter F1-3000 | Instruction gốc | Ablation bổ sung sau khi có mode được audit; OOD vì train memory_dropout=0 |
| F0 matched | Model train riêng không memory | Instruction gốc | Chỉ thêm nếu xác minh cùng data/split/budget; không tự coi checkpoint cũ là matched |
| Routed-base | Model gốc trên LIBERO-10 | Instruction gốc | Regression của hệ thống routing theo `MEM_ADAPTER_V1.md` |
| Adapter-base | Base + cùng adapter trên LIBERO-10 | Memory off | Retention của chính checkpoint; chưa được hỗ trợ, cần mode audit riêng |

B0 → F1 đo lợi ích của **toàn hệ thống adaptation + perception + memory**.
Nếu đủ ngân sách, ưu tiên B1-predicted trước việc bổ sung mode F1-off: thêm 20 val
và 20 test episodes. Cùng writer contract không có nghĩa prompt giống nhau sau khi
các trajectory bắt đầu khác nhau.
F1 → F1-off đo độ phụ thuộc vào memory của cùng checkpoint, không thay thế F0.
Routed-base không chứng minh adapter giữ kỹ năng base. Không dùng O1 trong kế hoạch
tối thiểu: real Mem environment hiện từ chối privileged-oracle mode.

## 4. Trình tự thực hiện và gate

### P0 — Freeze và preflight

- Tạo thư mục mới `experiments/evaluation/results_mem_2/<run_id>/`; không ghi vào
  training outputs. Ghi candidate/base hashes, code revision + dirty diff,
  protocol/split hashes, environments, detector/tokenizer hashes và episode roster.
- Re-run artifact verification; xác minh tokenizer revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, detector/code/config và source bytes.
  Không sửa recorded hashes để cho qua validation.
  F1 renderer phải nhận `tokenizer=policy.tokenizer` đúng cùng object;
  runtime còn kiểm tra object identity, không chỉ tokenizer hash.
- Pin Mem source `0eee0defa4024e51511b6e15a79f30e9620209de`; kiểm tra checkout sạch,
  BDDL/assets, controller đầy đủ và data paths. Base giữ LIBERO
  `8f1084e3132a39270c3a13ebe37270a43ece2a01` trong process riêng.
- Làm adapter reload trên observation thật có provenance: state8, action chunk
  `(50,7)` hữu hạn, official processors/normalizers, direct gripper polarity,
  không merge LoRA hoặc clip/scale action thêm. So sánh load/reload với cùng RNG.
- Giữ camera `agentview` rồi wrist, ảnh 256×256. Offline refined RGB và raw simulator
  RGB dùng đúng source-specific convention. Không thêm flip do quan sát cảm tính.
- Ghi rõ discrepancy config state6 với runtime normalizer state8; dựa trên official
  processor + stats đã audit, không tự đổi feature semantics.
- Hoàn tất CLI/bridge/logger ở mục 2; kiểm tra request ID, reset, dtype/shape,
  stale frames/actions, worker crash và invalid-action rejection. Unit fixtures
  chỉ phục vụ software tests; benchmark bắt buộc model và simulator thật.

**Gate:** integrity, interface, real reload và isolation đều pass; mọi missing
weights/assets, unknown state/action/camera/controller hoặc hash mismatch đều dừng
bước phụ thuộc ngay và ghi nguyên nhân, không fallback.

### P1 — Simulator/evaluator conformance và smoke

- Chạy replay T1/T3 bằng demo train đã khóa (bắt đầu demo 1), kiểm tra exact reset,
  progress reset, inc cadence, robot/controller/camera/alignment. Đây là
  demonstration replay, không tính vào policy score.
  Đã có historical reports pass ở
  `experiments/audits/mem_training_v1/real_replay_demo1/report.json` và
  `real_replay_demo5_31/report.json`; chúng không chứng nhận runtime hiện tại.
- Giữ protocol `configs/benchmarks/libero_mem_t1_t3_protocol.json`:
  **20 Hz**, warmup **20**, horizon **1000 policy actions**, chunk **50**,
  execute **10**, simulator seed **0**.
- Không early-stop vì subgoal/full progress tạm thời. Dừng ở horizon hoặc irreversible
  overshot; full success ở terminal phải đủ milestones, final predicate true và
  không overshot. Logging chỉ đọc evaluator result, không gọi thêm check_success.
- Chạy B0 và F1 trên **validation split index 0 của từng T1/T3**: 4 episodes.
  Khóa cùng policy seed đề xuất **42**; seed policy process tách khỏi detector.
  `initial_state_id` của Mem là index trong filtered split, không phải official
  base init ID và không phải số `demo_n`.
- Xem video hai camera, action/state traces, exact prompt/token count ≤48,
  writer/evaluator counts tách riêng, nguồn thông tin causal và reset giữa episodes.

**Gate:** chạy đủ protocol hoặc dừng hợp lệ, artifact đầy đủ, không NaN/crash/leakage,
writer đúng contract. Episode thất bại nhiệm vụ vẫn là dữ liệu hợp lệ; không bắt buộc
smoke đạt SR cao. Từ wall time smoke ước lượng batch bằng thời gian episode đo được;
không giả định 20 Hz simulation là tốc độ chạy thực tế.

### P2 — Validation paired pilot

- Chạy toàn bộ **10 validation states/task × 2 tasks × 2 nhánh = 40 episodes**
  B0/F1, cùng policy seed 42. Bốn smoke episodes được tái sử dụng chỉ nếu toàn bộ
  code/config/protocol giống hệt; nếu có thay đổi, chạy lại tập validation nhất quán.
- Khóa roster trước khi chạy; pair theo task + initial-state hash + policy seed.
  Không thay states thất bại bằng states khác. Crash/incomplete được báo riêng;
  không loại âm thầm khỏi denominator hoặc chỉ chạy lại failures để chọn kết quả đẹp.
- Step 3000 là candidate đầu tiên. Nếu cần chọn giữa checkpoints, chỉ dùng validation;
  predeclare tập so sánh, ưu tiên 2000/2500/3000. Intermediate `checkpoint-*` là
  training snapshots, phải export riêng bằng cơ chế PEFT có provenance và reload
  verification; không đưa trực tiếp folder đó vào `SmolVLAAdapter.load`.
- Ranking đề xuất: macro full-task SR trên T1/T3, rồi SCR, rồi overshoot thấp hơn;
  tie cuối ưu tiên checkpoint sớm hơn. Không chọn theo flow loss đơn độc.
- Nếu thêm F1-off: thêm 20 val episodes sau khi mode audit pass. Chưa có F0 matched
  thì giới hạn kết luận ở whole-system gain. Không mở rộng T2/T4/T7… trong lượt này.

**Gate:** freeze checkpoint + writer + renderer + code + selection rule trước test.
Không sửa thresholds hoặc protocol dựa trên final test. Pilot không cải thiện vẫn
được báo cáo; không đổi định nghĩa success để đạt gate.

### P3 — Test giữ kín và kiểm tra base

- Sau freeze, dùng **10 test states/task × 2 tasks × 2 nhánh = 40 episodes**.
  Đây là custom adaptation holdout; lock không chứng minh disjoint với pretraining
  của model gốc. Không dùng test để chọn lại checkpoint.
- Minimum Mem budget: **80 episodes B0/F1** gộp validation + test nếu smoke tái dùng
  được; 84 nếu phải giữ smoke riêng. F1-off thêm 40 khi chạy cả val và test.
- Nếu mở rộng policy seeds, predeclare trước test, dùng cùng seeds ở các nhánh,
  không thay simulator seed đã khóa. Ví dụ 3 seeds làm tổng Mem budget 240 episodes;
  seed repeats không tạo thêm independent initial states.
- Routed-base regression: đủ 10 LIBERO-10 tasks × 5 official initial states đã khóa
  = 50 episodes với checkpoint gốc, writer off. Label **routing regression**;
  so với baseline chỉ khi cùng device/dtype/protocol/runtime. Báo overlap với
  adaptation split nếu có; không gọi mẫu này là unseen adapter retention.
- Nếu mục tiêu là retention của cùng adapter: sau mode audit, dùng adapter-on và
  base gốc trên cùng base holdout roster đã đối chiếu official initial-state arrays
  với split lock. Chưa xác minh ánh xạ thì chưa có claim held-out retention;
  không thay official states bằng vector tự tạo. 50 episodes/mỗi nhánh chỉ là pilot,
  không tự kết luận non-inferiority 3 điểm phần trăm khi CI còn rộng.

### P4 — Báo cáo

- Primary: full-task SR từng task và macro average; SCR = completed/target,
  overshoot rate, milestones/timestamps và termination reasons.
- Báo paired ΔF1−B0, n, 95% CI; resample theo initial-state group khi tính paired CI,
  giữ các seed repeats cùng group. Với 10 test states/task, ghi rõ độ bất định lớn.
- Writer diagnostics: count agreement/overcount/undercount, missing/uncertain memory,
  latency và prompt coverage. Không dùng writer count làm ground-truth success.
- Runtime: policy/writer latency riêng mean/p95, total wall time, throughput,
  peak VRAM theo process/device. Đồng hồ bắt đầu/kết thúc phải bao gồm reset/warmup
  ở metric total, phân biệt với policy-only timing. GPU timing cần synchronization.
- Failure review: reach/grasp/drop/placement/count/overshoot/timeout và technical
  failures; không ép taxonomy khi video/trace chưa đủ bằng chứng.
- Giữ `run_manifest.json`, `episode_roster.json`, từng `episode.json`,
  `trajectory.npz` (actions raw/executed, states, step IDs), `video.mp4`, prompt/memory
  traces và evaluator records; sinh `summary.csv`, `failure_distribution.csv`,
  `report.md`. Không log oracle information vào policy input.
- Chọn tối đa 3 success và 3 failure videos từ run thật; nếu không đủ loại nào thì
  ghi số thực có, không dựng hoặc lặp video để đủ quota.

## 5. Commands đã có và phần chưa có

Các lệnh sau là kế hoạch chạy trong đúng môi trường tương ứng, **chưa chạy ở lượt
lập plan**. `<SIM_PYTHON>` là executable trong isolated Mem simulator, không phải
một executable đã được xác minh ở lượt này. Policy tests dùng environment riêng
có dependencies cần thiết; skip không được tính là đã xác minh real checkpoint.

```bash
# Governance (repository environment)
python -m pytest -q tests/test_project_structure.py
python scripts/validate_project.py
python scripts/system_info.py

# Policy/artifact + writer/runner software checks (isolated policy test environment)
python -m pytest -q tests/unit/test_mem_adapter_artifact.py tests/unit/test_smolvla_adapter_unit.py tests/unit/test_libero_mem_rollout.py tests/unit/test_libero_mem_text_renderer.py tests/unit/test_libero_mem_tracker_writer.py tests/model_adapters/test_smolvla_shape_contracts.py

# Simulator/evaluator software checks (isolated simulator environment)
<SIM_PYTHON> -m pytest -q tests/unit/test_libero_mem_loader.py tests/unit/test_libero_mem_config_unit.py tests/unit/test_libero_mem_evaluator.py

# Official base simulator smoke (isolated LIBERO reference environment)
python scripts/validate_environment.py
python -m pytest -v tests/environment/test_libero_env.py

# Real replay; use a fresh output directory and verified local raw HDF5 files
<SIM_PYTHON> scripts/audit_libero_mem.py --split-lock experiments/training/results_mem_2/mem-adapter-v1/dataset-split.lock.json --tasks T1 T3 --demo-ids 1 --output experiments/evaluation/results_mem_2/<run_id>/conformance
```

Command CLI hiện hành nằm ở [hướng dẫn đánh giá checkpoint](MEM_ADAPTER_V1.md#đánh-giá-checkpoint-bằng-cli).
Chạy `preflight`, rồi `smoke --runtime-check` (hai action thật, không score), sau
đó smoke đầy đủ và validation. Stage test là thao tác riêng sau khi khóa candidate.
Không dùng `run_benchmark.py --task-suite libero_mem` để thay cho CLI này.

## 6. Tiêu chí hoàn tất

1. Checkpoint thật được load/reload đúng contract, mọi gate integrity và simulator pass.
2. Có paired validation/test B0/F1 T1/T3, seed/state provenance và đầy đủ artifacts.
3. Báo SR/SCR/overshoot/runtime/failures, kể cả kết quả không cải thiện.
4. Phân biệt rõ whole-system gain, memory ablation, routed-base regression và
   same-checkpoint retention; chỉ claim những gì đã thực sự đo.

Tài liệu liên quan: [Mem adapter workflow](MEM_ADAPTER_V1.md),
[V2 research protocol](V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md),
[Mem protocol](../configs/benchmarks/libero_mem_t1_t3_protocol.json),
[model audit checklist](../.agents/skills/vla_model_audit.md),
[evaluation checklist](../.agents/skills/policy_evaluation.md).
Generic checklist `env.check_success()`/stop-on-done của base không thay thế
stateful Mem evaluator protocol; đây là track V2 custom đã ghi rõ.
