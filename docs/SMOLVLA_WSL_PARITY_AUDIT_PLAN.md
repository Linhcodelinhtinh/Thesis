# SmolVLA Windows–WSL: đánh giá bằng chứng và kế hoạch kiểm tra

Ngày kiểm tra: 2026-10-13. Phạm vi: raw policy, không memory. Đây là đánh giá artifact đã lưu và kế hoạch; chưa chạy thêm rollout hoặc thay đổi runtime trong lượt đánh giá này.

## Kết luận hiện tại

WSL có thể hoàn thành task soup với một chuỗi action từ Windows. Policy chạy closed-loop trên WSL vẫn thất bại trong probe seed 1042. Hai quan sát này chưa xác định được nguyên nhân là observation, stochastic noise, model runtime hay tương tác controller/contact. Chưa đủ bằng chứng loại trừ simulator hoặc chứng nhận toàn bộ WSL stack tương đương reference.

## Điều chỉnh các nhận định trong báo cáo A/B

| Nhận định | Bằng chứng và giới hạn |
|---|---|
| Replay loại trừ simulator | Replay một trajectory đạt success lần đầu ở index 341; mean EEF error 3.560 mm, max 8.860 mm, max lift 227.225 mm. Chứng minh khả năng hoàn thành trajectory này, không chứng minh contact/object dynamics tương đương. Episode Windows nguồn kết thúc thành công sau 695 action. Cần đối chiếu success predicate và object pose theo thời gian. |
| `final_success=true` chứng minh thành công ở cuối replay | Script tính `first_success_step is not None`; giá trị thực tế là **ever_success**, không phải success tại action cuối. Index 341 tương ứng action thứ 342. |
| EEF trước settling trùng nhau loại trừ init state | Chỉ xác nhận các trường đã đo. Cần kiểm tra toàn bộ qpos/qvel, object pose, controller state và compiled model. |
| Sau settling lệch khoảng 3.5 mm | Các tọa độ được trích trong báo cáo là EEF sau action policy đầu tiên. `settling_snapshot.post_settling` của hai episode hiện có cho chênh lệch **2.220 mm**. Đây là Windows seed 1042 và clean-reset seed 42; không thay thế snapshot đầu vào bị thiếu của probe B seed 1042. |
| Action đầu tiên có L2=1.031 | 1.031350 là norm của ma trận **10×7**. Riêng vector action đầu tiên: **0.470574**. Max absolute difference 0.451546. Đây là đơn vị action, không phải khoảng cách mm; không phải toàn bộ chunk 50 action. |
| Cùng seed xác nhận model parity | Chưa cố định observation và tensor noise. Khác CPU/CUDA, phiên bản thư viện hoặc thời điểm reseed có thể thay đổi noise. Cần so sánh với cùng tensor đầu vào và noise được lưu nguyên giá trị. |
| Sai lệch settling gây phân kỳ flow matching | Giả thuyết hợp lý nhưng chưa được can thiệp kiểm chứng. Runtime và noise còn là biến gây nhiễu. |
| GRASP failure chứng minh lệch góc kẹp vài mm | Diagnostics cho biết reach threshold được vượt qua, không ghi nhận grasp, max lift 4.7 mm. Khoảng cách EEF–object 36.9 mm không đo sai số fingertip/contact hoặc góc kẹp. Chưa chứng minh slip hay nguyên nhân góc kẹp. |
| Nhóm task 100% chứng minh giả thuyết hình học | Snapshot `baseline_s10_clean_reset` có 28 episode/10 task; các task được nêu 100% chủ yếu là **2/2**. Chưa đủ suy rộng toàn bộ 40 task hay phân nhóm theo hình dạng/friction. |

Tọa độ post-settling từ artifact:

- Windows: `[-0.1542952609, -0.0065243248, 0.2470945286]`.
- WSL clean reset: `[-0.1525524638, -0.0062767265, 0.2484475461]`.
- Orientation và gripper cũng phải được đối chiếu; chỉ xét translation là chưa đầy đủ.

## Giới hạn về khả năng tái lập probe

- `scratch/test_seed_1042_rollout.py` hiện kết thúc ở `return 0` dòng 101, sau CPU parity print. Phần GPU/B1 và ghi JSON bên dưới không chạy; còn tham chiếu biến không được định nghĩa trong bản hiện tại. File được sửa sau artifact kết quả, nên không dùng bản hiện tại để tuyên bố đã tái lập nguyên vẹn B1/B2.
- `scratch/probe_b_results.json` có GPU comparison và B1; không lưu toàn bộ CPU comparison để kiểm chứng độc lập con số CPU L2=0.988.
- `scratch/replay_trajectory_probe.py` dùng giá trị zero mặc định nếu thiếu object position. Probe tiếp theo phải báo lỗi nếu thiếu observation bắt buộc; chưa có bằng chứng fallback này đã kích hoạt trong run cũ.
- Probe thiếu snapshot script, input tensors, noise, full runtime provenance và chuỗi contact/object state. Giữ nguyên các artifact cũ; bổ sung run mới trong thư mục riêng.

## Reference và khả năng so sánh benchmark

`reset -> set_init_state -> 10 dummy steps` phù hợp code LeRobot v0.4.4. Robosuite 1.4.0 phù hợp baseline được repository yêu cầu. Tuy nhiên hai điều này chưa đủ chứng nhận toàn bộ stack, camera/controller hoặc protocol.

LeRobot v0.4.4 đặt giới hạn mặc định 280 action cho object/spatial, 300 cho goal và 520 cho libero_10. Replay đạt success ở action 342 và probe dùng 1000 bước là chẩn đoán với thời lượng riêng, không được tính thành official comparable score theo các giới hạn mặc định đó. Cần chốt protocol của kết quả công bố cụ thể trước khi so điểm; không có một score “LIBERO” duy nhất độc lập với checkpoint và evaluation protocol.

Checkpoint official, checkpoint dùng trong paper, và checkpoint của bên thứ ba là ba đối tượng cần xác minh riêng bằng revision/hash. Số đo gần công bố không tự chứng nhận run Windows hợp lệ hơn; số đo thấp cũng không tự chứng minh WSL sai.

## Kế hoạch thực hiện theo thứ tự

### C0 — Đóng băng dữ liệu và cấu hình

Tạo run ID mới, lưu hash script, git HEAD và dirty diff, command, Python executable, package versions/import paths, model/processor/tokenizer/backbone revisions và hashes. Lưu actual GL renderer/vendor/version riêng với model device/dtype; không suy luận GPU rendering từ CUDA hoặc biến môi trường.

Ghi task BDDL hash, init-state file hash/index, episode seed tường minh, XML/compiled model, camera names/order/resolution/pose, controller config/frequency, reset/settling, chunk size và execution horizon. Giữ horizon 10 cho chẩn đoán cặp hiện tại; không âm thầm đổi protocol để tăng điểm. Nếu gặp controller/interface/asset không xác định, dừng nhánh kiểm tra và báo theo AGENTS.md.

Đầu ra: manifest và artifact đã kiểm hash. Không ghi đè run cũ hoặc thay môi trường global.

### C1 — Offline parity: cùng observation, cùng noise (ưu tiên đầu tiên)

1. Xuất observation ngay trước policy call: hai ảnh raw uint8, toàn bộ proprio được adapter sử dụng, instruction và thông tin camera. Lưu cả trạng thái trước/sau settling. Nếu không có ảnh/noise của run Windows lịch sử, tạo snapshot Windows mới và ghi rõ không phải tái dựng chính xác lần chạy cũ.
2. Tạo và lưu một tensor noise với shape/dtype suy ra từ official model config/sampler. Dùng cùng giá trị trên mọi runtime. Cả hai bản LeRobot đang cài đều có `predict_action_chunk(batch, noise=...)`; dùng API này trong diagnostic harness, không sửa sampler hoặc benchmark preprocessing.
3. Chạy Windows CPU và WSL CPU trước; sau đó WSL CUDA. Mỗi trường hợp lặp ít nhất hai lần với cùng input/noise, model eval và dtype được ghi rõ. Kiểm tra cấu hình autocast/precision/attention backend.
4. So sánh từng tầng: raw input → processor output (images/state/token IDs/masks) → model với cùng processed batch → official postprocessing. Lưu full action chunk trước/sau unnormalization, sai số theo từng chiều và từng action; không trộn norm action với mm.
5. Dùng một panel nhỏ gồm snapshot đầu episode và gần grasp, nhiều tensor noise cố định. Một snapshot có thể tìm ra lỗi nhưng không đủ chứng nhận toàn bộ model.

Điều kiện chuyển bước: xác định tầng phân kỳ đầu tiên hoặc chứng minh output gần nhau trong panel dưới cùng input/noise. Đặt tolerance số học trước khi đánh giá, dựa trên dtype và kiểm tra lặp; không dùng ngưỡng tùy ý `max_diff < 0.05` để chứng nhận parity. Lưu mọi residual; parity số học không đồng nghĩa rollout chắc chắn thành công.

Nếu processor khác: kiểm tra camera mapping/flip/resize, state representation, tokenization và normalization. Nếu model khác với cùng batch/noise: kiểm tra weights thực được load, missing/unexpected keys, LeRobot/Transformers implementation và precision. Nếu chỉ action sau postprocessor khác: kiểm tra stats/action decoding. Chỉ sửa nguyên nhân đã xác định trong môi trường cô lập.

### C2 — Tách ảnh và proprio khi model parity đã được kiểm soát

Giữ cố định runtime và noise. So observation Windows với WSL; tiếp theo thay riêng image bundle, riêng proprio, rồi từng camera để đo ảnh hưởng lên action chunk. Các đầu vào lai chỉ dùng offline sensitivity analysis, ghi **NON-COMPARABLE** và không đưa vào benchmark. Không diễn giải đây là trạng thái vật lý thực hoặc bằng chứng đủ về success rate.

Đo cả orientation (xử lý tính tương đương dấu quaternion trong metric), gripper và pose của camera wrist; không giới hạn giả thuyết ở translation 2–3 mm. Nếu muốn kiểm tra render backend, giữ cùng simulator state và model/noise rồi mới đổi renderer.

### C3 — Mở rộng replay để kiểm tra controller/contact

Chọn trước 5 task × 3 initial states = 15 trajectory, gồm soup/cream-cheese/salad, ketchup và bowl; dùng trajectory có sẵn và ghi rõ success/failure của nguồn, không chỉ chọn các trajectory thành công.

Lưu theo từng physics/control step phù hợp: EEF position/orientation, gripper qpos/qvel, object pose, contact pairs/forces, controller targets và success predicates. So mean/P95/max, đặc biệt cửa sổ trước/sau contact; phân biệt `first_success_step`, `ever_success`, `final_success`.

Điều kiện kết luận: chỉ giới hạn mức sai khác trong các trajectory đã đo. Nếu cùng action nhưng object/contact khác đáng kể, tiếp tục audit compiled model/controller; chưa loại trừ simulator chỉ vì replay có thể success.

### C4 — Pilot paired rollout trước full benchmark

Sau khi giải quyết các khác biệt lớn chưa giải thích được, chọn trước 5 task đại diện cả nhóm giảm điểm và nhóm đối chứng × 5 official init states × 3 policy seeds = 75 episode mỗi điều kiện. Mỗi cặp giữ task/state/seed/reset/horizon/model cố định và đổi một yếu tố đã xác định. Seed được gắn với episode identity, không phụ thuộc thứ tự chọn task.

Báo paired success/failure, step-to-success, failure phase và khoảng bất định; seed lặp trong một task/state không phải các task độc lập. Nếu cần common noise theo từng replan để kiểm tra nhân quả, đặt thành diagnostic protocol riêng, không âm thầm thay RNG của benchmark chuẩn.

Chỉ mở rộng full 40 task sau khi manifest đầy đủ, các nhánh parity quan trọng đã được giải thích và protocol mục tiêu đã được chốt. Dùng đủ episode/seed theo nguồn so sánh đã chọn. Báo riêng internal LIBERO-derived score và score có protocol tương thích; không dùng tolerance, horizon hay renderer như tham số tối ưu trên test tasks để đạt số công bố.

## Nguồn kiểm tra

- Local: `scratch/probe_b_results.json`, `scratch/replay_probe_results.json`, hai script probe, `experiments/results/baseline_full_40/libero_object_task0/init_0/episode.json`, `experiments/results/baseline_s10_clean_reset/libero_object_task0/init_0/episode.json` và `model_output.jsonl` tương ứng.
- [LeRobot v0.4.4 LIBERO environment: reset, settling và episode limits](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/envs/libero.py).
- [PyTorch reproducibility: khác release/platform và CPU/GPU có thể không tái lập dù cùng seed](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md).
- Source of truth cho LIBERO tại repository này: `8f1084e3132a39270c3a13ebe37270a43ece2a01`, theo `AGENTS.md`.
