# Kế hoạch setup và benchmark X-VLA trên LIBERO

## Runner đã triển khai

`scripts/run_x_vla_libero.py` dùng trực tiếp `ClientModel`, `LiberoAbsActionProcessor` và chuỗi reset của client gốc tại revision bên dưới. Runner tự khởi động `deploy.py` từ env inference, tạo config LIBERO riêng và lưu trajectory 20D, 10D, 7D cùng video, timing và receipt. Entrypoint mặc định chọn `libero_10`, 50 init states mỗi task và 900 bước policy. Dùng `--preflight-only` để kiểm tra asset, reset và render độc lập với GPU inference.

Trong WSL2/Linux, cài env inference từ `envs/x_vla_inference.txt` với Python 3.10. Có thể dùng env `xvla-stable` từ `external/X-VLA/environment.yml` với `transformers==4.51.3`. Bổ sung `json-numpy==2.1.0` vào env simulator Python 3.8.13 nếu dùng env cũ. Chạy từ thư mục gốc repo với env simulator đã kích hoạt. `XVLA_PYTHON` là đường dẫn tới Python của env X-VLA.

```bash
uv venv --python 3.10 .venv_x_vla
uv pip install --python .venv_x_vla/bin/python --index-strategy unsafe-best-match \
  -r envs/x_vla_inference.txt
export XVLA_PYTHON="$PWD/.venv_x_vla/bin/python"
```

```bash
MUJOCO_GL=egl python scripts/run_x_vla_libero.py --preflight-only \
  --task-ids 0 --initial-state-ids 0 --output-dir experiments/results/x_vla_libero_10_preflight

MUJOCO_GL=egl python scripts/run_x_vla_libero.py --inference-python "$XVLA_PYTHON" \
  --task-ids 0 --initial-state-ids 0 --output-dir experiments/results/x_vla_libero_10_smoke
```

Bỏ hai bộ lọc ID và dùng thư mục output mới để chạy 500 episode. Khi WSL đọc weights chậm qua `/mnt/d`, có thể sao chép snapshot sang filesystem Linux và truyền `--checkpoint` với đường dẫn bản sao. Runner kiểm tra cùng SHA-256 trong cả hai trường hợp. Runner này dùng full checkpoint `2toINF/X-VLA-Libero`. Long PEFT có lệnh upstream riêng ở phần 6.

Chạy `2toINF/X-VLA-Libero` với weights đóng băng và instruction gốc của từng task. Ưu tiên `libero_10` cho task dài. Dùng `libero_object` cho các bài kiểm tra pick-and-place, rồi mở rộng sang `libero_spatial` và `libero_goal` nếu cần đủ bốn suite. [Hướng dẫn đánh giá của X-VLA](https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/README.md)

## 1. Checkpoint, mã nguồn và tài nguyên

| Thành phần | Revision dùng cho kế hoạch | Trạng thái kiểm tra |
| --- | --- | --- |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | `external/LIBERO` đang ở đúng commit. Xác minh lại đường dẫn package và tài nguyên được tiến trình thực tế import. |
| X-VLA | `6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2` | `external/X-VLA` đang ở commit này và remote trỏ tới repo tác giả. |
| `2toINF/X-VLA-Libero` | `129e71460678b7236cee6fc9707f09d9fa0c3590` | Có snapshot cục bộ. SHA-256 của weights khớp upstream. Cần kiểm tra phần còn lại của snapshot theo revision. |
| `2toINF/X-VLA-libero-long-peft` | `b6af8e31b153c0c67998ee0bd81bffd45321220d` | Có adapter cục bộ. SHA-256 của adapter weights khớp upstream. |
| `2toINF/X-VLA-Pt` | `c1c4a64a7e03ac5b95c468bf1578f3d03651b53b` | Snapshot tác giả được chọn để thử PEFT. Chưa có trong `resources/checkpoints/`. Config PEFT khai báo `revision: null`, nên revision được chọn cần ghi rõ trong báo cáo. |

Nguồn cho các revision và weights. [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01) · [Client X-VLA đã pin](https://raw.githubusercontent.com/2toinf/X-VLA/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/evaluation/libero/libero_client.py) · [X-VLA-Libero](https://huggingface.co/2toINF/X-VLA-Libero/tree/129e71460678b7236cee6fc9707f09d9fa0c3590) · [SHA-256 của full checkpoint](https://huggingface.co/2toINF/X-VLA-Libero/blob/main/model.safetensors) · [Long PEFT](https://huggingface.co/2toINF/X-VLA-libero-long-peft/commit/b6af8e31b153c0c67998ee0bd81bffd45321220d) · [X-VLA-Pt](https://huggingface.co/2toINF/X-VLA-Pt/commit/c1c4a64a7e03ac5b95c468bf1578f3d03651b53b) · [SHA-256 của X-VLA-Pt](https://huggingface.co/2toINF/X-VLA-Pt/blob/main/model.safetensors)

| File weights | SHA-256 |
| --- | --- |
| `x_vla_libero/model.safetensors` | `260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8` |
| `x_vla_libero_long_peft/adapter_model.safetensors` | `36476fc608ee5808939482fa522c580e5ca24fd0f0d137e5320b50f2b8e9c1e8` |
| `x_vla_pt/model.safetensors` | `433acffc992f457b1737d35ae20f81a895521f38eadaf03c736b6c0b0af1c93e` theo upstream |

- Xác minh hash của weights, config, tokenizer, processor và mã model đi kèm theo snapshot đã pin. Ghi vào `resources/manifests/models/` trước khi load. Dùng đường dẫn local cho server và lưu log load, gồm các key thiếu hoặc dư. Dừng khi model/processor không load được hoặc checkpoint không khớp.
- Tạo model ID riêng `x_vla_libero_long_peft` nếu chạy LoRA. Load `X-VLA-Pt` và processor của base rồi gắn `--LoRA_path` theo `deploy.py`. Xác minh cả LoRA weights lẫn `modules_to_save` của soft prompt, action encoder và action decoder được load. Chỉ chạy adapter Long trên `libero_10` trong kế hoạch này. [Cấu hình Long PEFT](https://huggingface.co/2toINF/X-VLA-libero-long-peft/commit/b6af8e31b153c0c67998ee0bd81bffd45321220d) · [Mã deploy](https://raw.githubusercontent.com/2toinf/X-VLA/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/deploy.py)
- Lấy task ID, BDDL và instruction theo registry LIBERO chính thức. Audit 50 init states của từng task, BDDL, XML và meshes theo manifest. `configs/libero/config.yaml` hiện trỏ vào `external/libero-mem` và dữ liệu memory. Tạo `config.yaml` trong thư mục runtime riêng qua `LIBERO_CONFIG_PATH`, với đường dẫn Linux đến BDDL, init states và assets của LIBERO chính thức. Kiểm tra cả `libero.__file__` và kết quả `get_libero_path(...)`. [Cơ chế chọn đường dẫn của LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/__init__.py)

## 2. Hai môi trường độc lập

1. **Simulator và client** dùng Linux hoặc WSL2 với `envs/libero_reference.yml`. Giữ Python 3.8.13, robosuite 1.4.0, MuJoCo 3.2.3, bddl 1.0.1, robomimic 0.2.0, NumPy 1.22.4 và stack Torch của reference. Kiểm tra `requests`, `json_numpy==2.1.0`, `tqdm`, `imageio` và FFmpeg trong môi trường này. Lock dependency bổ sung sau khi kiểm tra chúng tương thích với các pin của simulator. [LIBERO setup của tác giả X-VLA](https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/README.md) · [Tương thích Python của json-numpy 2.1.0](https://pypi.org/project/json-numpy/2.1.0/)
2. **Inference server** dùng env riêng, dự kiến `envs/x_vla.yml`. Bắt đầu từ `external/X-VLA/environment.yml` ở commit đã pin. Python 3.10, Torch 2.1 dòng CUDA 12.1, torchvision 0.16, Transformers 4.51.3 theo metadata full checkpoint và PEFT 0.17.1 cho thử nghiệm LoRA. Chốt patch và lock sau khi smoke test giải quyết dependency thành công. Khởi chạy trên CUDA với FP32 như `deploy.py`, ghi `pip check`, lock/freeze, driver, CUDA runtime, device và dtype thực tế. [Hướng dẫn cài đặt của tác giả](https://github.com/2toinf/X-VLA#-quick-start-installation--deployment) · [Mã deploy](https://raw.githubusercontent.com/2toinf/X-VLA/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/deploy.py)
3. **Preflight GPU và render.** Máy hiện có RTX 4060 Laptop GPU, 8188 MiB VRAM theo `nvidia-smi` trên Windows. Kiểm tra CUDA lại trong WSL2 hoặc máy Linux chạy thực tế. Kiểm tra offscreen render của cả hai camera, backend OpenGL thực tế và FFmpeg. Manifest đã ghi một lần EGL rơi xuống `kms_swrast`, nên phải ghi backend thực tế cho run mới. Chạy từng episode tuần tự lúc đầu. Đo riêng VRAM peak, thời gian inference và thời gian render trước khi tăng concurrency. Khi FP32 không vừa bộ nhớ, chạy cùng cấu hình trên GPU nhiều VRAM hơn.

Nếu cần tạo env mới, chạy các lệnh sau từ thư mục gốc Thesis_26. Sau khi kiểm chứng và khóa dependency, dùng `envs/x_vla.yml` cho các lần setup tiếp theo.

```bash
conda env create --file envs/libero_reference.yml
conda env create --file external/X-VLA/environment.yml
```

Chạy các kiểm tra sau trong env simulator. Lưu output và `pip check` cho cả hai env trong thư mục run.

```bash
python -m pytest -q tests/test_project_structure.py
python scripts/validate_project.py
python scripts/system_info.py
python scripts/validate_environment.py --strict
python -m pytest -v tests/environment/test_libero_env.py
python -m pip check
```

## 3. Hợp đồng action và vòng đời controller

| Ranh giới | Shape | Ý nghĩa |
| --- | --- | --- |
| `generate_actions` → HTTP `/act` | `(30, 20)` sau khi bỏ batch dimension | Action head `ee6d`, 30 action mỗi lần predict. Mỗi block 10 chiều mô tả một tay. |
| HTTP client → action Panda | `(30, 10)` | Lấy `action[:, :10]`. `0:3` là XYZ tuyệt đối theo frame của controller, `3:9` là rotation EEF6D, `9` là giá trị gripper sau sigmoid. Client chỉ sử dụng block đầu. |
| Bộ chuyển đổi → `env.step` | `(7,)` mỗi step | `[abs_x, abs_y, abs_z, axis_angle_x, axis_angle_y, axis_angle_z, gripper_cmd]` kiểu `float32`, với `OSC_POSE` ở chế độ absolute. |

- Rotation 6D mã hóa **hai cột đầu của ma trận rotation**, theo thứ tự `R[:, 0]` rồi `R[:, 1]`. Dùng đúng Gram–Schmidt của client và đổi ma trận qua quaternion sang axis-angle. XYZ đi thẳng vào OSC theo hệ tọa độ controller, đơn vị mét, axis-angle theo radian. Bỏ kẹp chuẩn hóa `[-1, 1]` của delta path đối với sáu trường pose tuyệt đối. Giữ các giới hạn vật lý và impedance mặc định của OSC. [Client X-VLA](https://raw.githubusercontent.com/2toinf/X-VLA/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/evaluation/libero/libero_client.py) · [OSC của robosuite v1.4.0](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.0/robosuite/controllers/osc.py)
- `generate_actions` đã sigmoid gripper. Client dùng `g > 0.5` thành `+1` đóng, còn `g <= 0.5` thành `-1` mở. Kiểm tra cả trường hợp bằng `0.5` và tránh áp dụng sigmoid lần hai. `ee6d` dùng pose trực tiếp theo đường inference chính thức. [Action head X-VLA](https://raw.githubusercontent.com/2toinf/X-VLA/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/models/action_hub.py) · [Client X-VLA](https://raw.githubusercontent.com/2toinf/X-VLA/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/evaluation/libero/libero_client.py) · [PandaGripper v1.4.0](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.0/robosuite/models/grippers/panda_gripper.py)
- Mỗi episode thực hiện `reset` → `set_init_state` → **10 step settling ở delta mode** với `[0, 0, 0, 0, 0, 0, -1]` → đặt `robot.controller.use_delta = False` trên controller hiện tại → bắt đầu policy. Assert mode trước và sau lần chuyển. Áp dụng lại mỗi reset vì controller có thể được dựng lại. Settling do simulator sở hữu để tránh cộng thêm settling của `rollout_episode`. [Client X-VLA](https://raw.githubusercontent.com/2toinf/X-VLA/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/evaluation/libero/libero_client.py) · [Issue #2](https://github.com/2toinf/X-VLA/issues/2) · [Issue #15](https://github.com/2toinf/X-VLA/issues/15)
- Giữ Franka Panda, gripper mặc định, `OSC_POSE`, impedance cố định theo config gốc, control 20 Hz và env horizon 1000. Absolute pose là **setpoint** được OSC bám qua physics, nên pose quan sát có thể còn sai số sau một step. Log riêng target của controller và pose đo được. Dùng `env.step` trong toàn bộ rollout. [LIBERO env tại commit pin](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py) · [OSC mặc định](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.0/robosuite/controllers/config/osc_pose.json) · [Issue #15](https://github.com/2toinf/X-VLA/issues/15)

## 4. Hợp đồng observation và chunk

- Render RGB `uint8` **256×256** từ `agentview` và `robot0_eye_in_hand`. Gửi `image0` là `agentview` lật hai trục ảnh, tương đương xoay 180°, và `image1` là wrist nguyên hướng. Processor của checkpoint resize xuống **224×224**, dùng normalization của checkpoint, tokenizer đi kèm và mask `[True, True, False]` cho ba slot ảnh. Kiểm tra bằng ảnh thật có mốc bất đối xứng để bắt lỗi đổi view hoặc flip hai lần. [Client X-VLA](https://raw.githubusercontent.com/2toinf/X-VLA/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/evaluation/libero/libero_client.py) · [Snapshot processor](https://huggingface.co/2toINF/X-VLA-Libero/tree/129e71460678b7236cee6fc9707f09d9fa0c3590)
- Với request đầu, dựng proprio `(20,)` từ `controller.ee_pos`, hai cột đầu của `controller.ee_ori_mat`, một giá trị gripper bằng `0` và 10 số `0` padding. Từ request thứ hai, client chính thức cập nhật `proprio[:9]` bằng XYZ và rotation 6D của **hàng cuối chunk dự đoán trước**. Giữ giá trị gripper proprio và padding như lúc khởi tạo. Reset proprio và hàng đợi mỗi episode. [Client X-VLA](https://raw.githubusercontent.com/2toinf/X-VLA/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/evaluation/libero/libero_client.py)
- Gửi task language từ registry, `domain_id=3`, `steps=10` cho flow matching. Execute tuần tự cả 30 action qua một hàng đợi rồi lấy observation để predict lại. Dừng sớm khi có success hoặc hết ngân sách step. `generate_actions` lấy mẫu bằng `torch.randn` phía server, nên seed client chỉ kiểm soát phía simulator. Khi đối chiếu parity, kiểm soát và ghi seed phía inference với cùng payload và thứ tự request. [Client X-VLA](https://raw.githubusercontent.com/2toinf/X-VLA/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/evaluation/libero/libero_client.py) · [Inference model](https://github.com/2toinf/X-VLA/blob/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/models/modeling_xvla.py)

## 5. Phương án tích hợp vào runner chung

Entrypoint hiện tại ở đầu tài liệu. Bảng dưới đây ghi lại các điểm cần xử lý nếu đưa X-VLA vào runner dùng chung với các policy delta.

| Lớp | Việc cần làm |
| --- | --- |
| Model config và manifest | Hoàn thiện `configs/models/x_vla_libero.yaml` và manifest model. Khai báo raw output 20, `action_dim: 10` của policy, simulator action 7, cùng `action_mode`, chunk size 30, execution horizon 30, kích thước camera và model input, frame, normalization, gripper threshold và nguồn proprio. `src/models/model_manifest.py` hiện kiểm cứng `action_dim == 7`. Xác thực theo giao diện từng model và từng ranh giới. |
| Model adapter | Thêm client X-VLA dưới `src/models/`, đăng ký model ID. Adapter mã hóa request và trả chunk 10 chiều đã cắt block Panda. Cho phép timeout với lỗi rõ ràng. Xác thực shape gốc `(30, 20)`, kiểu dữ liệu, giá trị hữu hạn, gripper và response schema trước khi tạo action chunk. Ghi các chiều theo `action_spec` của X-VLA. |
| Controller | Đặt decoder 10→7 trong `src/controllers/`, kiểm thử theo implementation tác giả. Validate rotation suy biến và gripper ngoài contract trước `env.step`. Truyền phán quyết lỗi nguyên nhân cụ thể vào artifact. |
| Simulator | Mở chế độ `absolute_pose` theo opt-in trong `src/simulator/libero_env.py`. Simulator cung cấp kinematics từ đúng controller, chịu trách nhiệm reset/settling và ghi `use_delta` thực tế trong metadata. Kiểm tra kiểu action khớp mode trước step. Diagnostics nhận action 7 chiều đã gửi simulator. |
| Runner | Dùng `scripts/run_episode.py` cho smoke và `scripts/run_benchmark.py` cho batch sau tích hợp. Dùng chung một nguồn config X-VLA. Tách `env.horizon=1000` khỏi ngân sách rollout 800/900 của tác giả. Bắt lỗi misconfig giữa checkpoint, chunk, execution horizon, camera và controller. Khi bật chế độ X-VLA, lưu action head 20D ở cấp chunk, action policy 10D và action thực thi 7D ở cấp step. Chốt đường run instruction-only với `memory_condition=off`. |

Viết kiểm thử hồi quy cho tổ hợp model–controller, reset nhiều episode, số step settling, chunk FIFO, rotation round-trip, polarity gripper, hai request proprio liên tiếp và shape hợp lệ của ba lớp telemetry. Dùng regression của SmolVLA để kiểm tra delta mode và action spec mặc định. So sánh đường render của `LiberoEnv` với client gốc sau reset, gồm camera geometry, geomgroup và pixel payload. Mọi sai khác ảnh hoặc physics phải có nguyên nhân được xác định trước khi dùng runner tích hợp để chạy số liệu theo giao thức tác giả.

## 6. Kiểm chứng rồi mở rộng benchmark

| Giai đoạn | Phạm vi và tiêu chí qua |
| --- | --- |
| 1. Replay controller | Dùng demo `libero_object` alphabet soup đã có và khớp SHA-256 trong `resources/manifests/demonstrations_manifest.yaml`. Từ cùng task, demo initial state và settling, replay delta action gốc để thu `goal_pos` và `goal_ori`. Reset lại cùng state, chạy absolute action đã chuyển qua EEF6D và decoder 10→7. Kiểm tra goal vị trí, góc sai số SO(3), gripper, trajectory pose và success theo từng step. Báo max/P95 sai lệch theo dung sai khai báo trước dựa trên repeatability của simulator. |
| 2. Inference với observation thật | Một state chính thức, đọc hai camera và proprio theo client. Đo tải FP32 và VRAM. Kiểm tra `(30, 20) → (30, 10) → (30, 7)` và đối chiếu ít nhất hai lần predict để kiểm tra proprio từ chunk trước. So sánh với client gốc trên cùng payload và seed inference. |
| 3. Smoke closed-loop | Runner tích hợp chạy một task `libero_object` với state `0`, sau đó một task `libero_10` với state `0`. Xác minh cả hai đi hết luồng, thông số effective và artifacts đúng, action khớp decoder, reset sạch và mọi lỗi runtime được phát hiện. Client tác giả `--eval_time 1` là một episode **mỗi task**, tức 10 episode cho một suite. |
| 4. Pilot | `libero_object` và `libero_10`, 10 task × 10 states `0..9` mỗi suite, 200 episode tổng. Báo SR theo task, timing và nhóm lỗi. Dự trù thời gian và dung lượng cho full run bằng số đo của pilot. |
| 5. Full | 50 states chính thức `0..49` × 10 task mỗi suite. Hai suite ưu tiên là 1.000 episode. Full bốn suite gồm `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` là 2.000 episode. Chạy PEFT Long trong job riêng trên 500 episode của `libero_10` khi base và adapter đã qua kiểm chứng. |

Replay kiểm chứng dùng `controller.goal_pos` và `controller.goal_ori` thu được khi chạy delta. Mỗi mục tiêu được đưa qua `env.step` lúc replay absolute, với trạng thái simulator đặt lại ở đầu demo. [Issue #2](https://github.com/2toinf/X-VLA/issues/2) · [Issue #15](https://github.com/2toinf/X-VLA/issues/15) · [Mã rel2abs của tác giả](https://github.com/2toinf/X-VLA/blob/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/evaluation/libero/rel2abs.py)

### Lệnh tham chiếu trong WSL2 hoặc Linux

Sau khi tạo hai env và xác minh snapshot, đặt `THESIS_ROOT` về đường dẫn checkout trong mỗi terminal, ví dụ `/mnt/d/Thesis_26`. Đặt `LIBERO_CONFIG_PATH` trỏ thư mục config LIBERO đã audit và `MUJOCO_GL` theo backend đã kiểm chứng trong terminal client. Mỗi lần chạy dùng `XVLA_RUN_DIR` mới vì `deploy.py` kiểm tra `info.json` đã tồn tại.

Server trong env X-VLA.

```bash
cd "$THESIS_ROOT/external/X-VLA"
HF_HUB_OFFLINE=1 python -m deploy \
  --model_path "$THESIS_ROOT/resources/checkpoints/x_vla_libero" \
  --host 127.0.0.1 --port 8000 --device cuda --disable_slurm \
  --output_dir "$XVLA_RUN_DIR/server"
```

Client trong env LIBERO. Thử trước với `--eval_time 1`, dùng thư mục output mới cho mỗi run.

```bash
cd "$THESIS_ROOT/external/X-VLA"
python evaluation/libero/libero_client.py \
  --server_ip 127.0.0.1 --server_port 8000 \
  --task_suites libero_object --eval_time 1 \
  --init_seed 42 --act_type abs --output_dir "$XVLA_RUN_DIR/upstream_smoke"
```

Để chạy upstream full, dùng `--task_suites libero_spatial libero_object libero_goal libero_10 --eval_time 50`. Ngân sách rollout tác giả là **800 step** cho spatial, object và goal, **900 step** cho `libero_10`, bên cạnh 10 step settling. Seed simulator của client là `42 + ep + 100` theo mặc định và chỉ số init state là `ep % 50`. Upstream log là đối chứng. Dùng runner đã tích hợp và kiểm chứng parity để xuất đầy đủ artifact của luận văn. [Client X-VLA](https://raw.githubusercontent.com/2toinf/X-VLA/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/evaluation/libero/libero_client.py)

Khi bắt đầu nhánh Long PEFT, tải base đã pin và xác minh theo bảng hash trước khi deploy. Dùng thư mục run riêng và chạy client với `--task_suites libero_10`. [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#download-a-specific-revision)

```bash
hf download 2toINF/X-VLA-Pt \
  --revision c1c4a64a7e03ac5b95c468bf1578f3d03651b53b \
  --local-dir "$THESIS_ROOT/resources/checkpoints/x_vla_pt"

cd "$THESIS_ROOT/external/X-VLA"
HF_HUB_OFFLINE=1 python -m deploy \
  --model_path "$THESIS_ROOT/resources/checkpoints/x_vla_pt" \
  --LoRA_path "$THESIS_ROOT/resources/checkpoints/x_vla_libero_long_peft" \
  --host 127.0.0.1 --port 8000 --device cuda --disable_slurm \
  --output_dir "$XVLA_RUN_DIR/server"
```

## 7. Báo cáo và điều kiện dừng

- Log `episode.json`, `trajectory.npz`, `model_output.jsonl`, `timing.json`, diagnostics và video. Trajectory ghi init state ID, seed, target pose, pose đạt được, gripper, index chunk và index step. Dùng `env.check_success()` làm chuẩn cho SR, kiểm tra parity với `done` của LIBERO pin, ghi rõ timeout và lỗi runtime. Khi sharding hoặc resume, giữ ID episode duy nhất theo protocol, model, suite, task, initial state và seed. Chỉ tổng hợp episode đã hoàn tất và công bố đủ số success, failure, lỗi và số episode còn thiếu. [Success predicate trong LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/bddl_base_domain.py#L779)
- Xuất `config.yaml`, receipt provenance, `report.md`, `summary.csv`, `failure_distribution.csv` và video đại diện theo quy định repo khi có đủ episode mỗi nhóm. Ghi SR theo task/suite, grasp và placement dưới dạng diagnostics, thời gian đến success, mean/P95 latency theo chunk, latency end-to-end và peak VRAM. Tính SR từ `episode.json` cho job đã đủ mẫu.
- Chạy theo contract tác giả được báo là **kết quả tái lập cục bộ bằng protocol X-VLA** sau khi đạt parity, kèm commit, protocol, weights, số mẫu và backend. Các số công bố của tác giả như 98,1% cho bản full và 83,2% cho Long PEFT chỉ dùng làm mốc tham chiếu. Đối sánh nội bộ với SmolVLA cần khóa cùng task/init state và ngân sách phù hợp, ghi controller và giao thức của từng policy. Giữ nhãn `NON-COMPARABLE_OFFICIAL_PAPER` theo ADR-0016 cho run khác protocol hoặc có stack dẫn xuất. Nếu cần so sánh trực tiếp theo cùng ngân sách, lên cấu hình đối sánh riêng với giới hạn đó được khai báo trước cho cả hai policy. [Bài báo, bảng 13](https://arxiv.org/html/2510.10274) · [Bảng checkpoint và PEFT của tác giả](https://github.com/2toinf/X-VLA#-quick-start-installation--deployment)
- Dừng và báo cụ thể khi sai hash, thiếu checkpoint/mesh, camera hoặc state sai contract, rotation/gripper không hợp lệ, digest config controller khác, HTTP/inference lỗi hoặc replay không đạt parity. Phân biệt policy failure hợp lệ với lỗi hạ tầng trong mẫu số. Trước batch mới, chốt trạng thái working tree và ghi hash của mã thực thi, cấu hình và lockfiles.

## Kết quả kiểm chứng triển khai

- Preflight trên WSL xác minh BDDL và 50 init states cho từng task của `libero_10`. Task 0, state 0 render hai camera RGB 256×256 với `llvmpipe`.
- Replay demo alphabet soup qua bộ chuyển đổi của tác giả chạy 148 bước. Cả hai chế độ delta và absolute đều đạt success. Sai lệch XYZ mục tiêu tối đa bằng 0, sai lệch phần tử rotation 6D tối đa `5.23e-7`.
- Smoke với full checkpoint trên RTX 4060 đạt success ở `libero_10` task 0, state 0 sau 308 bước policy. Trace gồm 11 chunk `(30, 20)`, 308 action policy 10D và 308 action thực thi 7D. Lượt nạp trực tiếp qua `/mnt/d` hết thời gian chờ. Lượt thành công dùng bản sao đã xác minh hash trên filesystem Linux qua `--checkpoint`.
- Receipt, video và trajectory nằm trong `experiments/results/x_vla_libero_10_cuda_smoke_20261119/`. Phạm vi kiểm chứng policy là một episode với nhãn `NON-COMPARABLE_OFFICIAL_PAPER`.
