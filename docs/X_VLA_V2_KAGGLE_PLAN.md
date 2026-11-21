# X-VLA F0/F1 trên split V2: kiểm chứng dữ liệu và kế hoạch chạy Kaggle

Ngày 21/11/2026. Tài liệu này kiểm lại bảng so sánh dữ liệu SmolVLA và X-VLA, rồi đưa ra đường chạy cụ thể trên Kaggle.
Mọi file mới nằm trong `scripts/xvla_v2/` và `notebooks/kaggle_xvla_v2_f0_f1.ipynb`. Không file cũ nào bị sửa.

## 1. Kết quả kiểm chứng bảng

| Nhận định trong bảng | Kết luận | Bằng chứng |
|---|---|---|
| Action: SmolVLA delta 7D, X-VLA absolute EEF-6D 10D | **Đúng**, nhưng kết luận "không dùng được shard" **sai một phần** | Shard V2 lưu `actions [n,7]` và `robot_states [n,9]` (gripper 2, eef_pos 3, quat xyzw 4), đã căn sẵn theo hàng: base `obs[i] -> actions[i+1]`, Mem `obs[i] -> actions[i]` (`scripts/prepare_mixed_memory_sft.py` dòng 145–183). Từ hai mảng này tính được đúng target mà `rel2abs.py` thu từ `controller.goal_pos/goal_ori`, không cần simulator. Ảnh và prompt trong shard dùng lại nguyên vẹn. Chỉ riêng cột action là phải đổi. `libero_mem_derived_v1` là dữ liệu pilot cũ, không thuộc lock V2, nên không dùng. |
| Ảnh: 224×224, agentview xoay 180°, wrist giữ nguyên | **Đúng** | Client `libero_client.py` chỉ `_flip_agentview` (lật 2 trục) cho agentview, wrist gửi nguyên. Resize 224 bicubic, ColorJitter và chuẩn hóa ImageNet được làm sẵn trong `datasets/dataset.py` của X-VLA, nên giữ ảnh 256 gốc trong shard. Notebook cũ ghi "sẽ được flip 180° tự động". Điều này sai: `LiberoHandler` không lật ảnh. |
| HDF5: `LiberoHandler` đọc `abs_action_6d [T,10]` và bỏ frame đầu | **Đúng**, thiếu hai chi tiết | `metas.json` gồm `dataset_name`, `datalist` (mỗi phần tử là **một file HDF5 cho một quỹ đạo**), `observation_key`, `language_instruction_key`. Instruction chỉ được đọc **một lần cho cả episode** (`base.py`, `read_instruction`). Việc bỏ frame đầu ghép ảnh hàng `idx+1` với proprio `abs[idx]`, và chỉ đúng khi ảnh hàng i là quan sát *trước* action i. Shard V2 thỏa điều kiện này. |
| Tokenizer: SmolVLM2 và Florence-2 | **Đúng**, nhưng không mất gì | Shard lưu chuỗi prompt (`prompts`, dtype string), không lưu token id. X-VLA dùng BartTokenizer, `language_max_length = 50`, có truncation. Trên 27 prompt khác nhau lấy từ audit index, dài nhất là 29 token. Script dựng dữ liệu kiểm toàn bộ prompt V2 và dừng nếu có prompt vượt 50 token. |
| Split lock V2: 598 quỹ đạo | **Đúng** | `resources/manifests/libero_mem_t1_t3_base10_split_v2.json`: train = 10 task × 40 base + T1 100 + T3 98 Mem. Policy là `libero_mem_refined_train_raw_eval_v2`, lock sha `8275e468…`. Tên `AUTHOR_MEM_SPLIT` là hằng số trong `src/training/split_lock.py`. |
| Dữ liệu thô `libero_mem_raw` + replay `rel2abs.py` | **Sai** | Theo lock V2, Mem train lấy từ bản *refined* `data/libero_mem` (sha `1737dd…`, `bffe4e…`). `libero_mem_raw` là nguồn roster đánh giá (seen `demo_1…`, unseen `demo_101–120`). Replay raw để train sẽ trộn dữ liệu đánh giá vào. Bảng cũng bỏ sót 400 quỹ đạo base, vốn cũng cần abs action. Ngoài ra `rel2abs.py` upstream chạy 10 bước no-op trước khi replay, nên quỹ đạo lệch khỏi state đã ghi. Cách tính giải tích từ shard tránh được cả ba vấn đề. |
| Prompt sự kiện bộ nhớ dùng lại, ghép vào từng timestep | **Đúng về dữ liệu, sai về cơ chế** | Prompt theo từng frame nằm **trong shard** (`prompts`, `has_memory`). `samples.jsonl` chỉ có chỉ số và cờ. Writer là Relation V1 (manifest ghi `predicted_rgb_relations_v1`), trong khi `configs/training/xvla_mem_adapter_v1.json` ghi `predicted_rgb_v3`, hai chỗ này không khớp. `LiberoHandler` upstream không nhận prompt theo timestep, nên cần handler riêng. Do bỏ frame đầu, prompt phải lấy `prompts[idx+1]`, tức prompt của chính frame đang hiển thị. |

### Hai điểm bảng chưa nêu nhưng làm hỏng training nếu sai

1. **Frame quay.** `robot0_eef_quat` của robosuite 1.4 là quaternion của *body* `right_hand`. Controller lại dùng *site* `grip_site`, lệch Rz(−90°) (`panda_gripper.xml`: `right_gripper quat="0.707107 0 0 -0.707107"`). Nếu không bù, mọi target quay lệch đúng 90°.
2. **Bố cục 6D.** Client LIBERO dùng `concat(R[:,0], R[:,1])` (hai cột đầu). Hàm `quat_to_rotate6d` trong `datasets/utils.py` dùng bố cục xen kẽ theo hàng, không khớp checkpoint LIBERO.

**Đã kiểm trên robosuite 1.4.0 thật** (Panda OSC_POSE, 300 bước ngẫu nhiên, có hàng xoay bằng 0 và hàng bị clip), file `scripts/xvla_v2/tests/test_sim_formula.py`:

| Frame | Sai số vị trí | Sai số quay |
|---|---|---|
| `panda_rz_m90` (đang dùng) | max 0,78 mm, median 0,28 mm | median 0,08°, max 0,13° |
| `identity` (không bù) | như trên | ≈ 90° |

Bố cục 6D trùng hoàn toàn với `Mat_to_Rotate6D` của client. Sai số dưới 1 mm đến từ việc quan sát robosuite chậm một substep so với lúc controller đặt goal.

### Lỗi trong `notebooks/kaggle_xvla_mem_finetune_f0_f1.ipynb` (đã sửa 21/11)

- `meta_f0` và `meta_f1` giống hệt nhau. Điều kiện `"instruction_memory" in meta_f0` luôn sai, nên F1 không có memory. `datalist[:400]` cắt mất 198 quỹ đạo Mem.
- `peft_train.py` không có gradient accumulation. "batch 2 × acc 8 × 2 GPU = 32" thực tế chỉ còn batch hiệu dụng 4.
- Auto-resume chỉ tìm checkpoint, không nạp lại. `peft_train.py` không có tham số resume, nên mỗi phiên train lại từ 0.
- Phần chuyển dữ liệu (`RUN_DATA_PREP=True`) chưa được viết.

Notebook này giờ chạy cùng pipeline với `kaggle_xvla_v2_f0_f1.ipynb`, và phần đầu có bảng ghi từng lỗi cùng cách sửa. Bản cũ vẫn còn trong git (commit `27ea215`).

## 2. Thành phần mới

| File | Việc |
|---|---|
| `scripts/xvla_v2/abs_actions.py` | Delta → abs EEF-6D theo OSC_POSE robosuite 1.4, có bù frame site, bố cục 6D của client, giữ goal_ori khi delta quay bằng 0, kèm chẩn đoán tracking. |
| `scripts/xvla_v2/build_xvla_v2_dataset.py` | Đọc shard V2, kiểm lock, số episode, độ dài, cờ memory và ngân sách token. Ghi `xvla_abs_actions_v2.h5` (~15 MB), `metas_f0/`, `metas_f1/` (mỗi arm có `base.json` và `mem.json`, upstream tự trộn 50/50), cùng `build_receipt.json`. |
| `scripts/xvla_v2/shard_handler.py` | Handler X-VLA đọc thẳng shard. Sampling giống `LiberoHandler` (bỏ frame đầu, 30 bước, freq 30, bỏ đoạn đứng yên). Lật agentview, giữ wrist. F1 lấy `prompts[idx+1]`. Đăng ký lúc chạy, không sửa code upstream, `domain_id = 3`. |
| `scripts/xvla_v2/train_xvla_mem.py` | Cấu trúc PEFT của tác giả (LoRA r8/α16 all-linear, modules_to_save, `build_optimizer`/`update_group_lrs` upstream) với lịch LR ở mục 4, thêm gradient accumulation thật, resume (adapter + optimizer + step), `--max-hours`, log jsonl (loss, LR, tỷ lệ mẫu Mem, tỷ lệ prompt có memory, VRAM). |
| `scripts/xvla_v2/check_xvla_alignment.py` | Cho checkpoint gốc dự đoán trên frame base rồi so với target. Thử thêm các biến thể: frame body, không lật ảnh, lệch 1 frame. Dùng làm cổng trước khi train. |
| `scripts/xvla_v2/verify_abs_actions_sim.py` | Tùy chọn, chạy local trên WSL: replay demo LIBERO thật, so với `controller.goal_*`, ghi receipt. |
| `scripts/xvla_v2/tests/` | Test công thức trên simulator và test pipeline trên shard giả lập qua `InfiniteDataReader` upstream. Cả hai đã chạy qua. |
| `notebooks/kaggle_xvla_v2_f0_f1.ipynb` | Toàn bộ quy trình trên Kaggle. |
| `configs/training/xvla_mem_adapter_v2.json` | Siêu tham số đã chốt cho X-VLA (mục 4). |

Đã chạy được: test simulator, test pipeline giả lập (ảnh lật đúng, proprio `abs[idx]` ghép với ảnh `idx+1`, prompt đúng frame, `domain_id = 3`, batch `[B,3,3,224,224]`, action `[B,30,20]`), và vòng train với model giả (accumulation, lưu, resume, dừng theo giờ).
**Chưa chạy:** dữ liệu V2 thật, checkpoint thật, GPU T4. Vì vậy số giây mỗi update, VRAM và tổng thời gian chỉ biết sau bước smoke.

## 3. Các bước chạy

### Bước 0, trên máy (một lần)

Đóng gói code mới thành Kaggle dataset `xvla-v2-code` (PowerShell, trong `D:\Thesis_26`):

```powershell
New-Item -ItemType Directory -Force kaggle_upload\xvla-v2-code | Out-Null
Copy-Item -Recurse -Force scripts\xvla_v2 kaggle_upload\xvla-v2-code\
'{"title":"xvla-v2-code","id":"<kaggle-username>/xvla-v2-code","licenses":[{"name":"CC0-1.0"}]}' |
  Set-Content -Encoding utf8 kaggle_upload\xvla-v2-code\dataset-metadata.json
kaggle datasets create -p kaggle_upload\xvla-v2-code --dir-mode zip
# lần sau: kaggle datasets version -p kaggle_upload\xvla-v2-code -m "update" --dir-mode zip
```

Shard V2 đã có trên Kaggle: `luquanglinh/libero-mem-t1-t3-shards-v2` (`experiments/training/kaggle_dataset_shards_v2/dataset-metadata.json`). Nếu tài khoản đang chạy notebook không có quyền đọc dataset này, upload lại zip (12,4 GB) bằng cùng lệnh.

Tùy chọn, lấy receipt replay trên simulator LIBERO:

```powershell
wsl -- bash -lc "cd /mnt/d/Thesis_26 && .venv_libero_reference/bin/python scripts/xvla_v2/verify_abs_actions_sim.py --hdf5 resources/demonstrations/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 --demos demo_0 demo_1 --out experiments/audits/xvla_abs_sim_receipt_v2.json"
```

### Bước 1–7, trên Kaggle (`notebooks/kaggle_xvla_v2_f0_f1.ipynb`)

Cấu hình: GPU T4 ×2, Internet ON, gắn shard V2 và `xvla-v2-code`.

1. **Cài đặt:** transformers 4.51.3, peft 0.17.1, accelerate 1.2.1, mmengine 0.10.5 cùng các thư viện X-VLA cần.
2. **Chuẩn bị:** clone X-VLA và checkout `6c4ecfb`, tải `2toINF/X-VLA-Libero@129e714…` vào `/kaggle/tmp`, rồi tìm shard (nếu dataset còn ở dạng zip thì tự giải nén).
3. **Dựng dữ liệu (CPU, vài phút).** Phải thấy: `train_episodes = {base: 400, mem: 198}`, `token_check.over_limit = 0`, tracking median ở mức mm. Đây là bước "tạo memory": prompt của Relation V1 đã nằm sẵn trong shard, và F1 X-VLA dùng **đúng** chuỗi mà SmolVLA F1 V2 đã dùng. Không sinh lại, để hai backbone so sánh được.
4. **Cổng căn chỉnh (GPU, ~5 phút).** `rotation_frame_ok` và `flip_ok` phải là `true`. Sai số bước đầu của `reference` phải nhỏ hơn rõ so với `body_frame` và `no_flip`. Nếu không đạt, dừng lại và gửi `alignment_base.json`.
5. **Smoke.** 20 update, resume lên 30. Đọc `sec_per_update`, `peak_vram_gib`, `frac_has_memory` (F1 > 0, F0 = 0). Nếu OOM, đặt `BATCH_PER_GPU = 2`, `GRAD_ACCUM = 8`. Thời gian mỗi arm ≈ `sec_per_update × 5000 / 3600` giờ.
6. **Train** F0 rồi F1, mỗi arm 5.000 update, checkpoint mỗi 500. Khi chạm `MAX_HOURS` (10,5 giờ), script tự lưu. Vì `/kaggle/working` bị xóa ở phiên mới, hãy Save Version để giữ output, rồi ở phiên sau gắn output đó qua **Add Input → Notebook output**. Cell 2b sẽ chép `runs/` cũ về và train tự resume.
7. **Đóng gói** `xvla_v2_f0_step5000.zip` và `xvla_v2_f1_step5000.zip`: adapter, `export_manifest.json` (hash, revision, lock, hyperparameter), `build_receipt.json`, `alignment_base.json`, `train_log.jsonl`.

## 4. Siêu tham số cho X-VLA (đã chốt, giống hệt cho F0 và F1)

File: `configs/training/xvla_mem_adapter_v2.json` (file mới, thay cho các giá trị trong v1). Đây cũng là mặc định của `train_xvla_mem.py` và notebook.

Số tham số đọc từ header `model.safetensors` của checkpoint:

| | SmolVLA V2 (đã chạy) | X-VLA-Libero |
|---|---|---|
| Tổng tham số | khoảng 0,45 B | 879,7 M (vision 360,6 M, transformer 302,3 M, language 207,9 M) |
| Tham số được train | 645.120 (LoRA r16 attention của expert, VLM đóng băng) | khoảng 8,45 M LoRA r8 all-linear (ước lượng) và 3,84 M train toàn phần (soft prompt 0,98 M, action encoder 2,24 M, action decoder 0,61 M) |
| LR / warmup / batch / step | 2e-5 / 100 / 32 / 5.000 | xem bảng dưới |

Model X-VLA lớn khoảng gấp đôi, nhưng số tham số được train lớn hơn khoảng 19 lần, và có 3,84 M tham số đầu ra action được cập nhật trực tiếp. Mức gấp 19 lần này mới là yếu tố quyết định lịch LR.

| Tham số | Upstream PEFT | Dùng | Lý do |
|---|---|---|---|
| Base LR (transformer LoRA + action heads) | 1e-4 | **5e-5** | Checkpoint đã được fine-tune trên LIBERO (domain 3), nên đây là tinh chỉnh tiếp chứ không phải học domain mới. Action heads được train toàn phần và quyết định trực tiếp action, nên LR thấp hơn giúp giữ kỹ năng LIBERO-10 (E6). |
| `learning_coef` (VLM LoRA + soft prompts) | 1,0 | **0,5** → 2,5e-5 | VLM chiếm 568 M tham số và là nơi đọc chuỗi memory. Mức 2,5e-5 tương đương LR SmolVLA (2e-5), nên đường đọc ngôn ngữ được chỉnh ở tốc độ tương đương, trong khi phần nhận thức ảnh vẫn ổn định. |
| `freeze_steps` | 1000 | **0** | Pha đóng băng của tác giả dùng để khởi động soft prompt và action head cho embodiment mới. Domain 3 đã học sẵn. Với upstream, 20% ngân sách bị dành cho việc chỉ train heads. |
| Warmup | 2000 | **250** (5%) | 2.000/5.000 quá dài. 5% là đủ để LoRA all-linear (khởi tạo B = 0) và Adam ổn định. SmolVLA dùng 2%, X-VLA nhiều tham số hơn nên dùng dài hơn. |
| Lịch | hằng số | **cosine về 0,1×** tại step 5.000 | Step 5.000 là endpoint đã định trước. LR thấp ở cuối cho checkpoint ổn định hơn, ít phụ thuộc vào vài batch cuối. |
| Batch hiệu dụng / step / clip / weight decay | 16 GPU-batch / – / 1,0 / 0 | **32 / 5.000 / 1,0 / 0** | Giống SmolVLA V2: cùng 160.000 mẫu với 50% base, nên so sánh giữa hai backbone là theo cùng lượng dữ liệu. |
| LoRA | r8, α16, all-linear | giữ nguyên | Đây là công thức tác giả, và đã cho 8,45 M tham số. Tăng rank sẽ càng làm lệch xa SmolVLA. |

Lưu ý: các giá trị này là lựa chọn có lý do, chưa có sweep. Nếu muốn có số liệu đỡ cho lựa chọn, có thể chạy hai pilot ngắn (1.000 update, F0, LR 5e-5 và 1e-4), rồi so loss base trên frame val của lock và kết quả bước căn chỉnh. Việc chọn lịch LR phải xong trước khi xem bất kỳ kết quả T1/T3 hay E6 nào. `state.json` của mỗi checkpoint ghi số tham số được train thực tế theo từng phần, nên con số 8,45 M ước lượng sẽ được thay bằng số đo.

## 5. Rủi ro còn lại

- fp16 trên T4 (T4 không có bf16). Script train tự dừng nếu loss non-finite 20 update liên tiếp. Khi đó chuyển sang `--mixed_precision no` (fp32), giảm batch, rồi resume.
- Dataloader upstream lấy lần lượt từng episode trong mỗi worker, nên một micro-batch thường đến từ một episode. Accumulation 4 lần qua 4 worker giúp trộn lại. Cách này giữ nguyên hành vi upstream.
- Resume khôi phục trọng số, optimizer và step, nhưng không khôi phục vị trí trong luồng dữ liệu (seed được đổi theo step). Cần ghi điều này trong luận văn nếu một arm bị cắt ngang phiên.
- **Đã sửa:** `scripts/run_x_vla_libero.py` nhận `--adapter <ckpt> --arm F0|F1-off`. Runner kiểm `export_manifest.json` (revision checkpoint, commit X-VLA, SHA-256 adapter), truyền `--LoRA_path` cho `deploy.py` của tác giả, yêu cầu log có "LoRA weights applied successfully", rồi ghi adapter vào `run.json`. Môi trường inference cần thêm `peft==0.17.1` (đã thêm vào `envs/x_vla_inference.txt`). Ví dụ: `python scripts/run_x_vla_libero.py --suite libero_10 --arm F0 --adapter <giải nén>/xvla_v2_f0_step5000 --inference-python .venv_x_vla/bin/python --output-dir experiments/results/<run>`. Runner này chỉ đo E6, nơi F1 luôn chạy không có memory (F1-off). E2 trên T1/T3 với writer cần thêm phần nối X-VLA vào evaluator Mem, việc này chưa làm.
- `configs/training/xvla_mem_adapter_v1.json` đang ghi `memory_source: predicted_rgb_v3`, khác shard (`predicted_rgb_relations_v1`). **Đã sửa:** v1 được đổi `memory_source` thành `predicted_rgb_relations_v1` và đánh dấu `SUPERSEDED` → v2.
