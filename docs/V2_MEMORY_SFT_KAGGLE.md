# V2 memory SFT trên Kaggle 2×T4

> **Hướng phát triển LIBERO-Mem (22/10/2026):** xem
> [implementation plan mới](V2_LIBERO_MEM_IMPLEMENTATION_PLAN.md) cho writer dựa trên
> object/events, causal memory, mixed base replay và retention gates. Các lệnh bên dưới
> vẫn là pipeline base/interface smoke hiện có, chưa hỗ trợ LIBERO-Mem. Không kế thừa
> observation/action offset hoặc memory dropout 0.2 sang Mem mà chưa audit.

Đây là supervised adaptation dùng **loss flow matching chính thức của SmolVLA**,
không phải RL và không phải một kết quả LIBERO official. Checkpoint baseline không bị sửa.

## Các file chính

- `scripts/prepare_memory_sft.py`: xác minh demonstration, tạo prompt causal, split và HDF5 shards.
- `scripts/train_memory_sft.py`: Accelerate/DDP, FP16, expert LoRA hoặc full expert, validation và resume.
- `scripts/kaggle_memory_sft.sh`: cài môi trường riêng, tải assets, prepare và launch 2 GPU.
- `scripts/memory_sft_assets.py`: tải checkpoint/tokenizer đã pin; sửa đường dẫn khi chuyển model đã export.
- `notebooks/kaggle_memory_sft.ipynb`: các cell chạy trên Kaggle.

## Nguồn dữ liệu và giới hạn

- Author dataset: `yifengzhu-hf/LIBERO-datasets`, revision
  `f13aa24a3da8c43c7225569f28c562979fa0e35a`.
- LIBERO source: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Policy: `lerobot/smolvla_libero`, revision `31d453f7edd78c839a8bbc39744a292686daf0de`.
- Backbone/tokenizer metadata: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Python dependencies: `envs/kaggle_memory_sft.txt`. Đây là môi trường training riêng;
  metadata của source LeRobot đã kiểm tra ghi 0.4.4, khác nhãn 0.6.1 trong manifest cũ.
  Không nâng cấp môi trường simulation/evaluation theo môi trường này.

Torch 2.7.1/torchvision 0.22.1 CUDA 11.8 theo [bản cài chính thức PyTorch](https://pytorch.org/get-started/previous-versions/),
torchcodec 0.5 được pin theo [bảng tương thích TorchCodec](https://github.com/meta-pytorch/torchcodec).

**Dataset raw có ảnh 128×128 trong file đã kiểm tra.** Không coi ảnh này là ảnh gốc 256×256
của dataset converted mà checkpoint từng dùng. Loader áp dụng đúng `_resize_rgb_image`
của adapter (PIL bilinear tới 256), sau đó official `LiberoProcessorStep` xoay 180°,
rồi preprocessing bên trong policy. Đây là lựa chọn dữ liệu được khai báo rõ;
cần đối chứng no-memory cùng dữ liệu để tách hiệu ứng data adaptation khỏi memory.

`external/LIBERO/scripts/create_dataset.py` ở revision trên gọi `env.step(action)`
trước khi lưu observation. Converter vì vậy dùng **obs[i] → actions[i+1:i+1+50]**.
Bỏ action đầu không có observation trước nó và bỏ observation cuối không có target kế tiếp.
Không tự dựng lại ảnh hoặc action còn thiếu. `robot_states` có thứ tự
`[gripper_qpos(2), eef_pos(3), quaternion_xyzw(4)]`; official processor chuyển thành state8.
Đối chiếu các trường gripper/position trong HDF5 trước khi chấp nhận schema.

Memory dùng đúng writer/retriever/compact renderer hiện tại, tokenizer thật và giới hạn 48 token.
Tại obs[i], chỉ các transition đã hoàn thành tới obs[i] được dùng. Writer reset riêng từng demo.
Các frame đầu đã bị bỏ trong source dataset không được tái dựng thành lịch sử.
Nhãn memory là **ước lượng của observation writer, không phải ground truth**.

Writer hiện tại không xác định được danh tính vật trong nhiều task multi-object và renderer bỏ
`unidentified_object`. Vì vậy nhiều task có thể có **0 memory frames**. Manifest báo tỷ lệ
theo task/split; trainer dừng nếu toàn train split không có memory. Không tạo fact giả để lấp chỗ trống.
Muốn học completion/recovery theo danh tính vật cần nguồn annotation/teacher đáng tin cậy hơn.

## Chuẩn bị dữ liệu

LIBERO-10 raw ở revision đã pin có tổng dung lượng **13.73 GB (decimal)**. Cộng thêm prepared
shards, dependencies và checkpoint có thể vượt dung lượng làm việc của một phiên Kaggle.
Ưu tiên prepare trên máy có đủ đĩa rồi upload thư mục prepared thành Kaggle Dataset riêng.
Nếu prepare trực tiếp trên Kaggle, kiểm tra dung lượng trước; có thể pilot một task bằng `--task-names`.
Không tự động xoá raw data hoặc checkpoint.

Từ repository đã copy vào thư mục ghi được:

```bash
bash scripts/kaggle_memory_sft.sh install
bash scripts/kaggle_memory_sft.sh assets
bash scripts/kaggle_memory_sft.sh prepare
```

Mặc định tạo `/kaggle/working/libero10-memory`. Có thể đổi `TRAIN_ENV`, `ASSETS_DIR`, `RAW_DIR`,
`DATA_DIR`, `OUTPUT_DIR` bằng environment variables. `prepare` tải đúng các file trong
revision đã pin và đối chiếu SHA256 từ LFS của upstream; có thể dùng `--source-lock` để kiểm tra offline.
Environment, assets và raw download mặc định ở `/kaggle/temp`, không nằm trong output cần lưu.
Chúng không được kỳ vọng tồn tại sang phiên mới: cài/tải lại trước khi resume. Prepared data và run outputs
ở `/kaggle/working` hoặc prepared data được attach tại `/kaggle/input`.

Ví dụ pilot một task có khả năng có prompt memory với writer hiện tại:

```bash
DATA_DIR=/kaggle/working/libero10-memory-pilot bash scripts/kaggle_memory_sft.sh prepare \
  --task-names STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
```

Output:

```text
libero10-memory/
  manifest.json           # completion marker, source hashes, split, camera/state contract, memory coverage
  samples.jsonl           # task/demo/index, prompt, facts, token count, source observation/action indices
  shards/<task>.hdf5      # images, robot_states, aligned actions, prompts, has_memory per demo
```

Split mặc định 80/10/10 theo nhóm initial-state hash trong mỗi task; không chia ngẫu nhiên frame.
Hai demo có cùng initial-state hash ở cùng split. `test` không được đọc trong training/validation.
Demo IDs không phải official evaluation initial-state IDs. Checkpoint gốc có thể đã thấy các demo này
khi pretrain: đây là held-out split cho **adaptation**, không chứng minh generalization chưa từng thấy.
Không thêm rollout benchmark đã dùng để đánh giá vào train rồi báo lại score trên chính episode đó.

## Chạy training

Chọn Kaggle accelerator **GPU T4 ×2**, Internet bật khi cài/tải assets. Mỗi T4 chứa một bản model;
DDP không gộp VRAM thành một GPU 32 GB. Không dùng BF16, FlashAttention bắt buộc hay lượng tử hoá.

Trước tiên chạy smoke 2 optimizer steps, dùng output riêng:

```bash
OUTPUT_DIR=/kaggle/working/sft-smoke bash scripts/kaggle_memory_sft.sh train \
  --steps 2 --warmup-steps 0 --validation-max-batches 2 --save-every 1
```

Sau khi smoke thành công:

```bash
OUTPUT_DIR=/kaggle/working/sft-memory bash scripts/kaggle_memory_sft.sh train \
  --condition memory --adaptation expert_lora --steps 3000
```

Preset khởi đầu, chưa phải hyperparameter đã tối ưu:

| Tham số | Giá trị |
|---|---|
| Backbone VLM | Frozen |
| Expert | LoRA q/v và state/action/time projections theo official targets |
| LoRA rank / alpha | 16 / 16 |
| Precision | FP32 weights + FP16 AMP/GradScaler |
| Batch mỗi GPU | 2 |
| Gradient accumulation | 8 |
| Effective batch | 2 × 2 × 8 = 32 |
| Learning rate | 1e-4; 100 warmup updates, cosine tới 0.1×LR |
| Action chunk | 50, mask cuối episode theo official `actions_id_pad` |
| Memory dropout | 0.2 trên train; deterministic theo seed/epoch/sample |
| Save / validation | Mỗi 500 optimizer updates và bước cuối |

Nếu OOM, giảm `--batch-size 1 --accumulation 16` để giữ effective batch 32.
Không giảm image resolution/chunk size hay đổi preprocessing để chữa OOM một cách âm thầm.
Trainer mặc định `--workers 0`; tăng workers chỉ sau khi đo RAM và startup overhead.

Đối chứng với cùng data và ngân sách:

```bash
OUTPUT_DIR=/kaggle/working/sft-no-memory bash scripts/kaggle_memory_sft.sh train \
  --condition no_memory --adaptation expert_lora --steps 3000
```

Có `--adaptation expert_full` để mở toàn action expert trong khi giữ VLM frozen;
cấu hình này dùng nhiều VRAM/optimizer state hơn. Chưa triển khai partial VLM LoRA trong preset đầu.

Validation đo flow-matching loss trên cùng frame/noise cho prompt memory và instruction gốc.
Đó **không phải success rate**, và loss giảm chưa chứng minh policy sử dụng memory hữu ích.
Mặc định chạy toàn validation split; `--validation-max-batches >0` chỉ dành cho smoke/đo nhanh,
được ghi rõ trong metrics. Final export là checkpoint cuối, không tự chọn best success rate.

## Normalization, checkpoint và resume

Trainer nạp **nguyên pre/postprocessor từ checkpoint**, chỉ override device và đường dẫn tokenizer.
Không tính lại stats từ dataset mới. 8D state được xác minh từ tensor stats, dù config gốc ghi shape6.
Không sửa config/checkpoint baseline. Toàn checkpoint được strict-load trước khi gắn LoRA;
`load_vlm_weights=False` tránh tải lại backbone weights riêng, không có nghĩa expert được random-init.

Output gồm `run_config.json`, `environment_packages.txt`, `metrics.jsonl`,
`checkpoint-NNNNNN/` (model/optimizer/scaler/scheduler/RNG + vị trí data) và `pretrained_model/`.
Không tự upload Hugging Face, Kaggle hay bật W&B.

```bash
OUTPUT_DIR=/kaggle/working/sft-memory bash scripts/kaggle_memory_sft.sh train \
  --condition memory --adaptation expert_lora --steps 3000 \
  --resume /kaggle/working/sft-memory/checkpoint-001000
```

Giữ nguyên schedule, batch, seed và các argument training khi resume. Không đổi tổng `--steps`
để nối một smoke run thành long run. Giữ checkpoint cùng dataset và phiên bản code đã chuẩn bị;
hash mismatch sẽ dừng. Resume phục hồi state và vị trí batch; không tuyên bố bit-identical giữa
phần cứng, phiên bản thư viện hay worker configurations khác nhau.

`pretrained_model/` chứa weights đã merge LoRA, normalizers gốc và metadata/tokenizer backbone.
Weights export là FP32 (khoảng 1.8 GB cho checkpoint này); dành đủ đĩa cho cả các checkpoint resume.
Sau khi tải thư mục này về máy đánh giá, sửa **chỉ đường dẫn local**:

```bash
python scripts/memory_sft_assets.py relocate /absolute/path/to/pretrained_model
python scripts/validate_memory_sft_export.py --checkpoint /absolute/path/to/pretrained_model \
  --data-dir /absolute/path/to/prepared_data
```

Sau đó dùng đường dẫn ấy làm checkpoint cho adapter hiện có, với execution horizon 10 và protocol
đánh giá đã khoá. Kiểm tra export/reload trước khi chạy benchmark dài. So sánh frozen baseline,
fine-tuned no-memory, fine-tuned memory trên cùng task/initial states; ghi rõ là custom V2 experiment.

## Kiểm tra local

```bash
pytest -q tests/unit/test_memory_sft_data.py
MEMORY_SFT_DATA=/path/to/prepared/libero_object_smoke \
  pytest -q tests/integration/test_memory_sft_pipeline.py
```

Integration test đối chiếu ảnh/state với adapter và nhãn action với demonstration thật.
Test fixture nhân tạo chỉ dùng trong unit tests, không phải benchmark data.

## Kết quả kiểm chứng local ngày 2026-10-16

- Dữ liệu thật: 50 demo của `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket`.
  Tạo 7,758 points: train 6,244 / validation 786 / test 728; tổng 3,947 points có memory (50.9%).
  Đây là dataset smoke, **không phải LIBERO-10 đã được tải đầy đủ**.
- 9 unit tests mới; 15 regression/structure tests liên quan; 2 integration tests trên demonstration thật: passed.
  `scripts/validate_project.py`: passed trong môi trường WSL có torch.
- GPU: RTX 4060 Laptop, một GPU, batch1, accumulation1; không phải hai T4.
  Torch 2.5.1+cu121, LeRobot 0.4.4, transformers 4.57.6, Accelerate 1.15.0, PEFT 0.18.0.
- Hai optimizer updates LoRA thật, 742,656 trainable parameters; loss hữu hạn.
  `torch.cuda.max_memory_allocated()` ở bước đầu khoảng 2.38 GiB (không phải tổng VRAM driver/reserved).
- Save checkpoint, resume từ step1 tới step2 và merge/export: passed.
- Export metadata giữ `type=smolvla` để tránh serializer bỏ discriminator của LeRobot.
  Strict reload qua adapter hiện tại, hash normalizers gốc và action chunk hữu hạn `(50,7)`: passed.
- Launcher bash, notebook Python cells và archive integrity: checked.

**Chưa chạy trên Kaggle 2×T4 hoặc toàn bộ dependency stack CUDA 11.8 đã pin; chưa đo success rate sau training.**
Notebook yêu cầu chạy smoke trên đúng máy Kaggle trước long run. Checkpoint smoke chỉ dùng kiểm chứng pipeline.
