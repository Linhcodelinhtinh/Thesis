# V2 isolated Mem adapter, expanded data

> Hướng dẫn dưới đây ghi lại adapter RGB v3 với split lịch sử. Cặp F0/F1 mới
> dùng toàn bộ refined T1/T3 và roster raw seen/unseen. [Lệnh P5–P6 hiện hành](thesis/P5_P6_HANDOFF_2026-11-13.md).

This is a custom V2 training system, not an official LIBERO score. V1 baseline
execution still loads the original checkpoint without an adapter. No training
or rollout improvement is claimed by the implementation alone.

## Cách sử dụng Mem — cập nhật 2026-11-06

Đây là hướng dẫn hiện hành cho **adapter Mem riêng + RGB writer v3**, thuộc V2.
Các notebook SFT cũ dùng merged checkpoint hoặc writer legacy không thay thế
workflow này. Đọc [writer T1/T3](MEMORY_WRITER_V3_T1_T3.md) để hiểu evidence và
[kế hoạch test results_mem_2](RESULTS_MEM_2_TEST_PLAN.md) trước khi chạy rollout.

| Nhu cầu | Checkpoint / prompt | Cách dùng hiện tại |
|---|---|---|
| Baseline LIBERO thông thường | Checkpoint gốc, instruction gốc | Không nạp Mem adapter; giữ protocol baseline đã khóa |
| Học Mem T1/T3 | Base gốc + LoRA attention q/v riêng | Notebook `kaggle_mem_adapter_v1.ipynb`, dataset full LIBERO-10 + T1/T3 |
| Rollout Mem F1 | Base gốc + unmerged adapter + predicted memory | API `LiberoMemRolloutRunner`, writer/renderer được bind và kiểm tra mỗi episode |
| Adapter với memory off | Cùng adapter, chỉ instruction | Chưa có mode inference được hỗ trợ; không bỏ qua runtime guard |

Chọn nhánh bằng cấu hình task/experiment đã biết trước khi chạy; chưa có learned
router. Dùng base gốc cho task thường bảo toàn đường chạy baseline về thiết kế,
không chứng minh chính adapter giữ được kỹ năng base hoặc làm tăng success rate.

### Memory đi vào model như thế nào

Memory là **text thêm vào instruction**, không phải tensor state mới, hidden state
huấn luyện riêng hay một adapter tự tạo ký ức. Luồng hiện tại:

```text
RGB + proprioception quan sát được tại t, action đã thực thi tại t-1
  -> RGB detector / causal tracker -> event writer -> memory snapshot
  -> LiberoMemTextRenderer(instruction, snapshot) -> SmolVLA + Mem LoRA
```

Detector chỉ nhận RGB; tracker dùng thêm proprioception quan sát được. Offline
và online dùng chung factory, writer/config, detector đã pin và tokenizer.
Tại `obs[t]`, memory chỉ chứa evidence đã có tới `t`; action target và annotation
tương lai không được dùng để tạo prompt. Writer reset cho từng demonstration
hoặc episode. Online cập nhật sau **mỗi action thực thi**, kể cả giữa hai lần query
policy; với execution horizon 10, prompt mới được gửi ở query kế tiếp.

Renderer nối bằng dấu cách, ví dụ minh họa khi evidence đủ:

```text
lift the bowl and place it back on the plate 3 times Memory: bowl A: 2 returns to original plate. Cycle underway.
```

Không nối thêm suffix thủ công. Budget **48 token tính cả instruction** bằng
tokenizer đã pin; renderer bỏ nguyên fact nếu vượt budget. Instruction tự nó
vượt budget được đánh dấu overflow, không được coi là prompt hợp lệ chỉ vì đã
bỏ memory. Khi history thiếu, dùng `at least N` / `return count unknown`; khi
không có fact hợp lệ, giữ instruction gốc. Không điền count từ evaluator.

Trong training, Mem frames dùng predicted prompt tương ứng; base replay dùng
instruction gốc. `memory_dropout=0` không có nghĩa mọi frame đều có memory:
frame thiếu bằng chứng vẫn có thể không có suffix. Data lock hiện hành gồm
698 demos (558 train / 70 val / 70 test), trong đó base 400/50/50 và Mem
158/20/20. T1/T3 chia sẻ initial states; không đếm chúng như các state độc lập.

## Scope and source coverage

Expand replay from seven to all ten official LIBERO-10 tasks. Keep all author
T1/T3 demonstrations, with train/validation/test initial-state separation.
The old split assignments are preserved when extending the lock. The RGB v3
writer supports bowl/original-plate only; T2/T4 and other object profiles are
not enabled by this version. Never substitute demonstrations for missing files.

Preparation produces predicted memory for every locked Mem frame using the
same visual_writer_factory and LiberoMemTextRenderer as evaluation. Every
prompt uses only current/past observations and the last executed action.
Source hashes, detector settings, writer code/config, tokenizer and index hashes
are verified. Empty/uncertain memory is retained; coverage alone is not accuracy.

## Preferred workflow: prepare directly on Kaggle

Upload only the small `mem_adapter_v1_code.zip` and import
`notebooks/kaggle_mem_adapter_v1.ipynb`. Enable Internet, select two T4 GPUs,
and keep `DATA_MODE = 'prepare_on_kaggle'`. Local memory generation, local
mixed-shard preparation and `mem_adapter_v1_data.zip` are no longer prerequisites.
Rebuild the code package after updating this repository:

```powershell
python scripts/package_memory_sft.py --output experiments/training/mem_adapter_v1_code.zip
```

The notebook performs these steps in order:

1. **Identify data sources before processing.** Write `source-plan.json` with
   repository, pinned revision, SHA256, expected bytes and existing/attached/download
   origin for every required file. Exact raw files already attached under
   `/kaggle/input` are reused through symlinks after full hash verification.
   Missing files are downloaded from the reviewed author repositories directly
   to Kaggle. Duplicate candidates and corrupt/incomplete files fail loudly.
2. **Check capacity and split.** The locked input inventory is 42,544,733,202 bytes
   (42.54 GB / 39.62 GiB). The notebook checks free scratch disk against missing
   downloads plus a configurable 60 GiB processing allowance. This allowance is
   a planning estimate, not a measured peak or a guarantee. The bundled
   `libero_mem_t1_t3_base10_split_v1.json` preserves the existing lock exactly:
   698 demos, 558 train, 70 validation, 70 test; all previous state assignments
   remain unchanged. No split is regenerated on Kaggle.
3. **Install isolated perception and training environments.** Fetch pinned model,
   tokenizer and detector assets. Verify source bytes and write
   `source-verified.json`; record both environments' resolved dependencies.
4. **Generate T1/T3 causal memory and build mixed shards on Kaggle.** The existing
   writer runs sequentially on one GPU. Two-GPU training begins afterward.
   Retain all locked episodes; the 50/50 base/Mem mixture is applied by the
   training sampler, not by deleting source episodes.
5. **Validate and train directly from the generated dataset.** No data ZIP is
   created or uploaded. Writer, tokenizer, split, lineage and shard integrity
   checks remain mandatory.

Raw files, models, virtual environments and large shards live under
`/kaggle/temp/mem-adapter-v1` and are session-local. Memory indices, provenance
receipts and split locks live under `/kaggle/working/mem-adapter-v1`; training
outputs also use `/kaggle/working`. Kaggle documents a 20 GB saved notebook-output
limit, distinct from runtime scratch capacity; check actual total output size
before saving. Completed notebook outputs can be attached to later notebooks
([official documentation](https://www.kaggle.com/docs/notebooks)). Persist large
shards separately on Kaggle if they need to survive the session; this notebook
does not publish a Kaggle dataset automatically.

Completed memory indices can be reused by attaching a saved output and copying
its `memory_T1` / `memory_T3` directories into the notebook's `PREP` directory.
The mixed-data validator checks the resulting indices against the lock and
writer contract. The current orchestrator passes `--resume` to the prompt-index
builder: recovery uses the saved samples/trace/detections records and may rewrite
their incomplete tail. Back up partial indices before retrying; keep the same
sources, split, writer, detector and tokenizer. Recovery is not a certificate of
episode completeness or provenance; final dataset validation remains mandatory.
There is no frame-level tracker-state resume. An index without a completed
manifest is not a training dataset; incomplete mixed shards require inspection
and a fresh output location.

Status: notebook wiring and source/split contracts are locally checked; full
Kaggle preparation/training and runtime are still unmeasured. This change does
not mean the unfinished local T1/T3 indices or mixed dataset are complete.

## Optional local preparation / attached prepared dataset

```powershell
.venv_memory/Scripts/python.exe scripts/prepare_mem_adapter_data.py --download --output-root experiments/training/mem_adapter_full10_t1t3_v1 --perception-python .venv_memory/Scripts/python.exe --tokenizer-path experiments/training/results/no_memory-20261029-175154/pretrained_model/backbone
```

This requires space for downloaded sources and a new immutable dataset; it does
not overwrite the previous F0 dataset. Prompt-index recovery follows the resume
rules above; inspect a failed stage before retrying or using a new output root.

After preparation completes, package the verified data with:

```powershell
.venv_memory/Scripts/python.exe scripts/package_mem_adapter_data.py --data-dir experiments/training/mem_adapter_full10_t1t3_v1/dataset --output experiments/training/mem_adapter_v1_data.zip
python scripts/package_memory_sft.py --output experiments/training/mem_adapter_v1_code.zip
```

For this optional route, set `DATA_MODE = 'attached_prepared'` and attach the code
ZIP and completed data ZIP plus its JSON checksum
sidecar (or the extracted dataset directory). Import
notebooks/kaggle_mem_adapter_v1.ipynb, enable Internet and select two T4 GPUs.
The notebook validates the expanded memory dataset before starting training.
Do not upload a `.partial` archive or an unfinished memory index.

## Training

Use the isolated training environment / Kaggle runtime with the existing pinned
training dependencies. Do not install training packages in the perception or
simulator environment. Fetch pinned assets with memory_sft_assets.py first.
The primary package pins are in envs/mem_adapter_training.txt, taken from the
previous successful training run; actual resolved dependencies are recorded
again by the trainer. Use a separate virtual environment for a local run.

```bash
python scripts/memory_sft_assets.py fetch /kaggle/temp/sft-assets
python scripts/train_mem_adapter.py --data-dir DATASET --assets-dir /kaggle/temp/sft-assets --output-dir OUTPUT --gpus 2
```

The launcher consumes configs/training/mem_adapter_v1.json: memory enabled,
attention-only expert LoRA rank 16, LR 2e-5, batch 32, 3,000 updates, 50/50
replay, checkpoints every 500. One GPU uses accumulation 16; two use 8.
These are pilot settings, not a validated optimum. Select later deployment
checkpoints using held-out Mem success/progress/overshoot, not flow loss alone.

The final pretrained_model directory contains an unmerged PEFT adapter and
mem_adapter_manifest.json, not a replacement base model. Base weights and
normalizer hashes bind it to the original checkpoint. Source training
checkpoints remain available; the final export is not automatically best.

## Inference

SmolVLAAdapter.load(adapter_directory, base_checkpoint=original_checkpoint)
loads the original policy/processors and attaches the frozen adapter. The
default base is resources/checkpoints/smolvla_libero. LiberoMemRolloutRunner
condition F1 binds the online writer and strict renderer to the training
contract before inference. Use the adapter's pinned tokenizer for the renderer.
The runtime rejects different detector/config/code and unverified memory paths.
Image conventions differ between refined offline HDF5 and raw simulator RGB;
use the explicit source-specific conventions in the shared factory.

For baseline tasks, load the original checkpoint, not the adapter directory.
No routing decision reads success predicates, hidden progress, or future state.
This version has explicit caller-selected modes; it does not claim learned routing.

### API wiring cho adapter riêng

Đoạn dưới minh họa kết nối API; CLI triển khai kết nối này được mô tả bên dưới.
`env` phải là môi trường Mem thật hoặc bridge đã xác minh, theo protocol khóa;
`adapter_dir`, `base_dir`, `perception_python`, `detector_dir` là đường dẫn local
đã kiểm tra. Không dùng fixture trong benchmark.

```python
from src.models.smolvla.adapter import SmolVLAAdapter
from src.perception.process_detector import ProcessVisualDetector
from src.perception.object_tracker import TrackerConfig
from src.memory.visual_pipeline import visual_writer_factory
from src.interfaces.libero_mem_text import LiberoMemTextRenderer
from src.evaluation.libero_mem_rollout import LiberoMemRolloutRunner

policy = SmolVLAAdapter()
policy.load(str(adapter_dir), base_checkpoint=str(base_dir), device="cuda")
contract = policy.mem_adapter_manifest["memory_provenance"][0]
renderer = LiberoMemTextRenderer(
    tokenizer=policy.tokenizer, strict_tokenizer=True,
    max_total_tokens=48, allow_oracle=False,
)
detector = ProcessVisualDetector(perception_python, detector_dir, device="cuda")
try:
    factory = visual_writer_factory(
        detector, image_convention="libero_raw",
        config=TrackerConfig(**contract["writer_config"]),
    )
    runner = LiberoMemRolloutRunner(
        env, policy, condition="F1", renderer=renderer, writer_factory=factory,
        execution_horizon=10, max_steps=1000,
    )
    # Khi chạy runner.run_episode(...), runner reset policy, tạo writer mới,
    # rồi bind_memory_runtime(writer, renderer) trước query đầu tiên.
finally:
    detector.close()
```

Renderer phải nhận **cùng object `policy.tokenizer`**, không load một tokenizer
object khác dù bytes giống nhau. `policy.reset()` xóa trạng thái runtime đã xác
minh; runner F1 bind lại cho mỗi episode. Offline refined HDF5 dùng
`libero_mem_refined`, online simulator dùng `libero_raw`; chuyển quy ước ảnh nằm
trong perception, không tự flip thêm ảnh policy hoặc đổi action decoding.

Artifact cần `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`
và `mem_adapter_manifest.json`, cùng checkpoint base gốc. Kiểm tra bằng
`src.training.mem_adapter_artifact.validate_adapter(adapter_dir, base_dir)`;
`scripts/validate_memory_sft_export.py` dành cho merged export cũ. Không merge
adapter vào baseline, sửa hashes hoặc tự đặt cờ verified để vượt lỗi contract.

Tại ngày cập nhật, `run_benchmark.py --memory-condition text_only` là luồng legacy;
chỉ đổi `--checkpoint` sang adapter không tạo được F1 ở trên. CLI riêng
`scripts/evaluate_mem_adapter.py` lấy instruction từ BDDL đã khóa hash, truyền
base checkpoint cho F1 và lưu trajectory/video/memory trace. Các gate thực nghiệm
trong [kế hoạch results_mem_2](RESULTS_MEM_2_TEST_PLAN.md) vẫn áp dụng. Việc có training artifact
không đồng nghĩa đã đo closed-loop success. Ghi rõ custom V2, protocol, split,
seed, hashes, prompt/memory trace, SR/SCR/overshoot và runtime khi báo kết quả.

### Lỗi config khi chạy từ WSL

`DecodingError: The fields type are not valid for SmolVLAConfig` xuất hiện nếu
đường nạp backbone local gọi `SmolVLAConfig.from_pretrained()` trực tiếp.
`type` là discriminator hợp lệ của checkpoint LeRobot; adapter phải gọi
`PreTrainedConfig.from_pretrained()` rồi kiểm tra `isinstance(..., SmolVLAConfig)`.
Đường nạp này đã được sửa; không xóa `type` khỏi config hoặc sửa weights để né lỗi.

Launcher Bash dùng `POLICY_PYTHON` (mặc định Python đang activate),
`SIM_PYTHON` (mặc định `.venv_libero_reference/bin/python`) và
`DETECTOR_PYTHON` (mặc định `.venv_memory_wsl/bin/python`). Nếu chưa có environment
tương ứng, launcher dừng trước khi load model. `.venv_memory/Scripts/python.exe`
là environment Windows, không phải Linux venv có thể activate trong WSL.

Sau khi ba environment và các gate evaluation đã được xác minh, cấu hình lệnh
smoke theo mẫu sau (đường dẫn detector là placeholder, cần thay bằng env thật):

```bash
SIM_PYTHON="$PWD/.venv_libero_reference/bin/python" \
DETECTOR_PYTHON=/absolute/path/to/linux-perception-env/bin/python \
bash scripts/run_mem_validation_and_test.sh smoke
```

Không đặt `SIM_PYTHON` hay `DETECTOR_PYTHON` bằng Python của policy để cho qua
preflight. Sửa lỗi đọc config và chọn đúng executable chưa chứng nhận rollout.

## Đánh giá checkpoint bằng CLI

Chạy từ repo trong WSL với policy venv đã activate. Simulator mặc định dùng
`.venv_libero_reference`, detector dùng `.venv_memory_wsl`; có thể đổi bằng các
environment variables ở trên. Policy environment cần PEFT để nạp adapter và
`imageio`/`imageio-ffmpeg` để ghi video. Không cài các dependency này vào simulator.

```bash
# Kiểm tra hashes, source, split và các environment; không chạy policy.
bash scripts/run_mem_validation_and_test.sh preflight

# Model + simulator + writer thật, hai action mỗi task/condition; KHÔNG tính score.
bash scripts/run_mem_validation_and_test.sh smoke --runtime-check

# 4 episodes: B0/F1, T1/T3, validation state index 0; đủ horizon theo protocol.
bash scripts/run_mem_validation_and_test.sh smoke

# 40 episodes validation để chọn checkpoint.
bash scripts/run_mem_validation_and_test.sh validation

# Chỉ gọi sau khi chốt candidate: 40 episodes test.
bash scripts/run_mem_validation_and_test.sh test
```

Stage `all` chỉ chạy smoke rồi validation, không tự chạy test. CLI không có mode
F1-off hoặc adapter-on base. Các số 40 ở trên áp dụng cho split hiện hành; `--states
all` đọc số state thực từ split lock, không hard-code 10. State index là vị trí
trong split, không phải demo ID. Run mới phải có output directory mới; không tự
ghi đè hoặc bỏ qua episode cũ.

Đổi candidate hoặc thu hẹp probe qua option chuyển tiếp:

```bash
bash scripts/run_mem_validation_and_test.sh smoke \
  --adapter-dir /absolute/path/to/another/pretrained_model \
  --tasks T3 --conditions F1 --runtime-check
```

`--adapter-dir` nhận **export unmerged có `mem_adapter_manifest.json`**. Thư mục
Accelerate `checkpoint-000500` là training resume state; phải export đúng adapter
và provenance trước khi đánh giá, không đổi tên thư mục để giả thành export.
`--no-save-video` tắt video có ghi rõ trong run manifest; mặc định ghi cả hai camera.

Output gồm `run_manifest.json`, `split.lock.json`, adapter manifest, `code.patch`,
source hashes và `code_snapshot.zip` (kể cả file chưa commit), `summary.csv`, `report.md` và mỗi episode:

- `status.json`: running/failed/complete hoặc diagnostic_complete, hashes artifacts.
- `initial_observation.npz`, `final_observation.npz`: RGB/proprioception lossless.
- `observations.jsonl`: step, raw proprioception, RGB hashes, evaluator info đọc sẵn.
- `actions.jsonl`, `queries.jsonl`: action thực thi, toàn chunk, prompt, token/render audit.
- `memory_trace.jsonl`: snapshot/evidence/signals ở mỗi observation F1, kể cả giữa query.
- `trajectory.npz`: action/chunk/query steps và raw proprioception theo thời gian.
- Hai MP4 raw camera: để xem; không dùng frame nén làm đầu vào policy hoặc writer.

Logging chỉ đọc `eval_result`, không gọi evaluator thêm lần nữa. Lỗi worker, hash,
request ID, stale observation hoặc ghi artifact làm run failed; report ghi số episode
đã hoàn tất và không coi phần còn thiếu là failure/success. SR/SCR/overshoot chỉ
có ở rollout đầy đủ; `--runtime-check` không tạo score. Report paired theo task +
initial-state hash; delta mang tính mô tả, chưa tự tính confidence interval.

Trên Linux, CLI tạo `detector-python.sh` trong run để giữ đúng venv executable
khi Python là symlink. Wrapper chỉ `exec` Python đã chọn; writer/perception code
đã khóa hash được giữ nguyên. Simulator stdout dành riêng cho IPC; diagnostic
prints chuyển sang stderr. Worker kiểm request ID, step, split/state identity
và allowlist observations, không đưa evaluator progress vào policy observation.
Trên WSL với repo ở `/mnt`, launcher ưu tiên `git.exe` nếu có để tránh quét NTFS
chậm; `MEM_EVAL_GIT` / `--git-executable` chọn explicit binary. Revision, dirty
source và hash checks vẫn giữ nguyên; lỗi/timeout không bị bỏ qua.

`run_manifest.json` cập nhật `phase` và `updated_at` trước mỗi lần nạp policy,
detector, simulator và episode. Nếu tiến trình bị ngắt từ ngoài mà không chạy
được exception handler, trạng thái cuối vẫn là chưa hoàn tất; không tính run đó
là pass. `total_wall_time_s` của episode bao gồm policy reset và simulator
reset/warmup; thời gian nạp model nằm ngoài episode.
Hai launcher lưu stdout/stderr và exit code trong
`experiments/evaluation/results_mem_2/launcher_logs/`, kể cả khi Python thoát
trước khi ghi được traceback vào manifest. Đường dẫn log được in khi bắt đầu.
