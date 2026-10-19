# Đánh giá mức sẵn sàng cho thesis memory — 18/10/2026

**Kết luận:** repository đã là một nền tảng thực nghiệm hữu ích, đủ để tiến hành pilot có đối chứng. Phần chứng minh giá trị của memory còn ở giai đoạn đầu. Điểm cần đầu tư tiếp là chất lượng fact, danh tính vật, semantics của tiến độ và thiết kế thí nghiệm; tăng quy mô training chưa giải quyết được các khoảng trống này.

Phạm vi: đọc source, cấu hình, tests và artifacts hiện có; đối chiếu kết quả với raw episode records; kiểm tra nguồn nghiên cứu gốc. Không chạy benchmark/training mới, không thay policy, controller, dependency hoặc dữ liệu. Những điểm dưới đây phân biệt code hiện tại, artifact lịch sử và đề xuất. Các file kế hoạch đang bị xóa trong working tree được giữ nguyên.

Kiểm chứng trong lần audit này: **44 tests passed trong190.13 giây** ở môi trường WSL hiện có, gồm memory core, writers, prompt grounding, paired comparison, SFT data và project structure. Lệnh: `.venv_smolvla/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_memory_core.py tests/unit/test_memory_writers.py tests/unit/test_memory_prompt_grounding.py tests/unit/test_compare_memory_runs.py tests/unit/test_memory_sft_data.py tests/test_project_structure.py`. Đây là component tests, không có live GPU rollout hoặc kiểm chứng semantic accuracy trên tập annotation.

## 1. Mức độ hoàn thiện theo bằng chứng

| Thành phần | Mức sẵn sàng | Bằng chứng và giới hạn |
|---|---|---|
| LIBERO + SmolVLA closed-loop | Khá, dùng được cho pilot | Có rollout thật, official task/init/success, official policy/processors; chưa có một runtime/protocol lock được kiểm tra đầy đủ |
| Logging và phân tích | Khá | Episode/trajectory/prompt/action/video, paired tools, audits; provenance thiếu dirty diff và đối chiếu toàn bộ contract |
| Memory store/update/retrieve | Khá ở mức component | Episode isolation, causal filtering, evidence source, confidence, token budget; các test không chứng minh fact đúng trong simulator |
| Observation writer | Sớm; nút thắt chính | FSM từ proprioception/action, chưa có object tracking, chưa sinh completion fact đáng tin cậy |
| Oracle writer | Chưa đủ làm chuẩn tham chiếu | Một target FSM, placement heuristic và mismatch với renderer |
| Memory conditioning | Có bằng chứng gây suy giảm; chưa có lợi ích dương | Dynamic mới trên hai task không đưa memory vào policy |
| SFT/data | Có pipeline và smoke | Một task libero_object, hai optimizer updates local, export/reload; chưa có kết quả training dài hoặc Kaggle 2×T4 |
| Generalization/novelty | Chưa được thiết lập | Chưa có test thiết kế riêng để chứng minh thông tin lịch sử thay đổi quyết định hữu ích |

Không quy các mức này thành phần trăm hoàn thành thesis: chất lượng kết luận không tỷ lệ với số module đã viết.

## 2. Kết quả đang có thực sự nói gì?

Run `experiments/results/v2_memory/prompt_shoren_obs_text` đã hoàn tất **28/100** episode. Các con số 1/15, 7/28 trong trao đổi trước là quan sát khi run chưa xong.

Đối chiếu raw `episode.json` với `experiments/results/libero-10-wsl` trên phần giao nhau gồm 10 task × init 0..4:

| Điều kiện | Thành công trên cùng 50 episode |
|---|---:|
| No-memory WSL | 35/50 (70%) |
| Memory run cũ | 13/50 (26%) |

Có 12 cặp cùng thành công, 23 cặp chỉ baseline thành công, 1 cặp chỉ memory thành công, 14 cặp cùng thất bại. Task/state/seed/execution horizon khớp trên phần giao nhau. Không so 70% trên 50 episode với 28% trên 100 episode như một paired effect. Đây là dấu hiệu regression mạnh của cấu hình memory cũ; dirty source/runtime provenance chưa đủ để coi mọi khác biệt đều đã được cô lập.

Ablation mới, hai task đầu × init 0..4:

| Điều kiện | Success | Diễn giải |
|---|---:|---|
| Original instruction | 7/10 | Đối chứng |
| Shadow | 7/10 | Cùng mẫu thành công/thất bại với original |
| Static suffix | 2/10 | 5 cặp regression, 0 cặp improvement |
| Validated dynamic | 7/10 | 515 query có context rỗng; policy nhận instruction gốc |

Nguồn: `experiments/results/v2_memory/ablation_*/benchmark_summary.json`, raw episode artifacts và `paired_original_vs_*/paired_memory_summary.json`.

Ở trace static task0 đã kiểm tra, tổng prompt 32 token so với 13 token của instruction gốc, dưới budget48. Text bổ sung chứa các assertion về detection nhưng không có detector tương ứng. Thí nghiệm gợi ý nhạy cảm với text bổ sung; chưa xác định attention dilution, sai semantics hay cơ chế nào là nguyên nhân. Static condition đi ngoài đường budget/fact accounting chung nên cũng cần sửa instrumentation trước lần đo tiếp.

N=10 chỉ gồm hai task đã chọn. Với 5 cặp discordant đều regression, exact two-sided McNemar/binomial p=0.0625; không kết luận có ý nghĩa thống kê chắc chắn từ pilot này. Dynamic khôi phục 7/10 bằng cách không đưa fact vào prompt, không phải bằng chứng model sử dụng memory.

## 3. Kiến trúc và mức tái sử dụng official implementation

Điểm tốt cần giữ:

- `src/simulator/libero_env.py` dùng official registry/BDDL/init states/OffScreenRenderEnv và success predicate.
- `src/models/smolvla/adapter.py:122` gọi official SmolVLAPolicy, LiberoProcessorStep và pre/postprocessors. Đây là hướng adapter mỏng phù hợp để tái sử dụng và giảm sai lệch.
- `src/memory/models.py`, `store.py`, `updater.py`, `retriever.py` tách dữ liệu, vòng đời, cập nhật và truy xuất; có provenance và causal guards.
- `src/evaluation/rollout.py` truy xuất trước inference, cập nhật từ transition đã hoàn thành cho lần replan tiếp theo; có off/shadow/text-only và prompt trace.
- Training dùng loss chính thức, giữ normalizers, có chunk padding/mask, split theo episode/state group, export/reload qua adapter.

Ranh giới cần làm rõ:

- Trong SmolVLA, `adapter.py:224` thay missing gripper qpos bằng giá trị mặc định. Normal LIBERO có field này, nên đây là lỗ hổng fail-fast, chưa phải nguyên nhân được chứng minh của regression hiện có.
- `ARCHITECTURE.md` mô tả detector/segmenter/tracker, spatial memory và các file chưa tồn tại. Cần ghi rõ proposed versus implemented.
- Một số diagnostic truy cập simulator ground truth là hợp lệ để đo lường. Nếu thông tin ấy đi vào policy thì phải thành treatment privileged riêng, không được coi là observation-only.

Không cần viết lại hệ thống. Nên thu nhỏ đường thực nghiệm chuẩn về một simulator/protocol, một checkpoint SmolVLA và các module memory có thể thay thế.

## 4. Vì sao memory hiện tại chưa khớp bài toán long-horizon?

`observation_writer.py:119–150` suy ra grasp từ gripper gap, action và EEF height. Có một target thì gán obstruction cho target đó với confidence cố định0.70; vẫn có thể đang chạm distractor hoặc fixture. Có nhiều target thì dùng `unidentified_object`; retriever/renderer loại đối tượng này khỏi prompt. Confidence hiện là tham số heuristic, chưa có bằng chứng calibration.

Writer sinh grasped/lifted/contact_lost/released, chưa xác nhận từng vật đã vào đúng container. Ví dụ thesis “A đã xong nên chuyển sang B” hiện chưa có nguồn fact observation-based tương ứng. Cấu trúc memory cũng chủ yếu chứa textual state, chưa có object identity tracking, visibility hay quan hệ không gian.

Oracle chưa khắc phục được vấn đề này:

- `oracle_writer.py:235` ưu tiên target đầu tiên và một FSM chung.
- `:290` dùng khoảng cách8cm hoặc episode success để suy placement, không phải từng goal predicate.
- Các missing-state fallback có thể biến giả định thành fact confidence1.
- Oracle `lifted_in_air` bị renderer `text_memory.py:183` phân loại thành completed state do substring `in_`; probe trực tiếp tạo `Memory: Bowl is already in the air.`
- Oracle `dropped_on_table` không qua recovery filter chỉ chấp nhận state `contact_lost` tại `text_memory.py:147`.

Ưu tiên typed states/events và mapping tường minh, per-object lifecycle, official predicate oracle cho annotation/diagnostic, rồi mới thay bằng estimator từ quan sát. Oracle vẫn chỉ là tham chiếu privileged, không bảo đảm upper bound thực nghiệm nếu policy hiểu sai interface.

## 5. Độ tin cậy thực nghiệm

Các vấn đề còn tồn tại trong source hiện tại:

1. `run_benchmark.py:492` gọi hashlib trước local import tại498, exception bị nuốt. Các run kiểm tra có dirty tree nhưng `git_diff_sha256=null`. Cần lưu cả source snapshot/diff và file mới, không chỉ hash.
2. Lock validator ở `:510–536` chỉ kiểm model name, dimensions, wait và Git; chưa enforce H/s, camera, max_steps, model bytes hoặc dependencies dù YAML khai báo.
3. `compare_memory_runs.py:210` kiểm seed/horizon/coverage, chưa so toàn bộ checkpoint/preprocessor/runtime/code fingerprint. Paired ID khớp chưa đủ khẳng định chỉ một treatment khác nhau.
4. `envs/smolvla.yml` chứa lower bounds và thiếu LeRobot; model manifest ghi0.6.1 trong khi training/test docs ghi0.4.4. Cần lock resolved environment thực sự chạy.
5. `tests/smoke/test_rollout_smoke_gpu.py:23` dùng constructor/API cũ, adapter chưa load và thuộc tính task_description không còn đúng. Test này chưa bảo vệ đường closed-loop hiện tại.
6. `thesis_completeness.md` còn ghi s50, chỉ có oracle, frozen-only và các kết luận quá mạnh về deterministic/completeness. Baseline lịch sử không được dùng thay baseline s10 mới.

Audit render ngày14/10 là bằng chứng lịch sử quan trọng. Source hiện đã có reset/set_init_state và geomgroup fix; không suy rằng bug cũ vẫn đang làm hỏng tất cả run mới. Cũng chưa chạy runtime probe trong lần audit này để chứng nhận môi trường hiện tại. Đóng việc này bằng một kiểm tra có artifact là đủ; không kéo dài thành dự án tái triển khai simulator.

Unit tests chứng minh các invariant đã mã hóa, không chứng minh semantic grounding đúng. Các parity/replay audit cũ hữu ích nhưng có phạm vi hữu hạn: C1 hai snapshot×hai noise không bit-identical; C3 có13/15 outcome trùng và max EEF error110.37mm. Không dùng chúng để tuyên bố runtime tương đương tuyệt đối.

## 6. Câu hỏi nghiên cứu và thiết kế đề xuất

Một giả thuyết đủ hẹp để triển khai:

> Memory sự kiện theo từng vật, có kiểm soát độ tin cậy và xử lý fact lỗi thời, giúp policy nhỏ chọn bước tiếp theo hoặc phục hồi tốt hơn trong tình huống quan sát hiện tại thiếu thông tin lịch sử, với ngân sách tính toán cố định.

Đây là định hướng đề xuất, chưa phải đóng góp đã chứng minh. Long-horizon không tự đồng nghĩa với history-dependent: scene hiện tại có thể đã thể hiện mọi tiến độ. LIBERO-10 nên là regression anchor; bổ sung một tập thí nghiệm custom có lịch sử cần thiết, ghi rõ thay đổi/provenance và không gọi là official LIBERO score.

Chọn một nhu cầu thông tin trước: completion khi vật bị che khuất, ghi nhớ lần grasp thất bại, hoặc goal cue chỉ quan sát ở đầu episode. Mỗi tình huống cần kiểm chứng rằng observation hiện tại thật sự không đủ; đừng thêm perturbation chỉ làm perception khó hơn rồi gọi đó là memory benchmark.

Các đối chứng có giá trị:

| Đối chứng | Tách được yếu tố |
|---|---|
| No memory / shadow | Side effects của instrumentation |
| Current-observation summary / historical memory | Thông tin trạng thái hiện tại so với thông tin quá khứ |
| Oracle current-state / oracle history | Extra privileged perception so với giá trị của lịch sử |
| Estimated history / oracle history | Giới hạn writer/perception |
| Real history / shuffled, stale hoặc irrelevant history cùng budget | Policy có sử dụng nội dung đúng hay chỉ chịu thêm token |
| SFT no-memory / SFT memory cùng data và compute | Lợi ích adaptation chung so với memory conditioning |

Giữ accuracy/success chính thức ở cấp episode; thêm event precision/recall, identity errors, false completion, detection delay, stale-fact duration, memory coverage, subgoal success, recovery success, repeat-completed-subgoal count và end-to-end latency. Failure phase chỉ là mô tả trừ khi có intervention xác minh nguyên nhân.

Pair theo task/state/policy seed và kiểm contract; báo per-task cùng aggregate, uncertainty phù hợp với clustering. Tách development/validation/test từ trước, không chọn checkpoint tốt nhất trên tập test. Task chưa dùng trong adaptation vẫn có thể đã được checkpoint gốc học; công bố đúng phạm vi holdout.

## 7. Training và thứ tự phát triển

Dataset thật hiện có: 50 demo một task libero_object,7,758 points; chưa phải LIBERO-10 đầy đủ. SFT có expert LoRA/full expert, frozen VLM, FP16/Accelerate, resume/export. Kiểm chứng trước đây mới hai updates trên một GPU local và finite action chunk sau reload; chưa có success sau SFT hoặc validation Kaggle2×T4.

Reusing demonstrations với prompt mới là thí nghiệm adaptation hợp lệ. Nó có thể dạy model bỏ qua memory. Để học completion/recovery cần labels và trajectory chứa quyết định đúng tương ứng; success-only rollouts của chính policy dễ thiếu các tình huống cần sửa. Raw128px và training dataset converted256px cũng là một thay đổi dữ liệu cần đối chứng.

Thứ tự ưu tiên và tiêu chí chuyển giai đoạn:

1. **P0 — Chốt đường thực nghiệm:** resolved runtime/protocol/source snapshot; sửa provenance/lock/smoke; chạy panel nhỏ off/shadow. Đạt khi contract được kiểm, logs tái dựng được và side effects được định lượng.
2. **P1 — Làm đúng thông tin memory:** typed semantics, per-object oracle, test event/render, annotation và đo chất lượng observation writer. Đạt khi có fact thật sự liên quan quyết định và coverage đủ trong các task đã chọn; ngưỡng precision/coverage phải đặt trước thí nghiệm.
3. **P2 — Kiểm tra giá trị lịch sử:** fixed-input probes với cùng noise và closed-loop nhỏ theo các đối chứng trên. Đạt khi true history có hành vi hợp lý và được so với current-state summary/sham; action thay đổi đơn thuần chưa đủ.
4. **P3 — Adaptation có kiểm soát:** pilot expert LoRA trên dữ liệu đã kiểm; same-budget no-memory SFT. Chỉ mở một phần VLM sau ablation cho thấy action-expert adaptation chưa đủ; đây không phải yêu cầu mặc định.
5. **P4 — Mở rộng bằng chứng:** nhiều initial states/seeds, task/composition holdout có mô tả exposure, robustness/cost trade-off; thêm backbone thứ hai sau khi official parity được xác nhận.

Chưa ưu tiên RL, full LIBERO download, vector database, spatial-memory framework lớn hoặc UR3 transfer trước P1/P2. Khả năng tối ưu quan trọng nhất hiện là chất lượng thông tin và treatment validity. Tối ưu throughput có thể thực hiện sau bằng profiling; không âm thầm đổi chunk/camera/precision vì chúng thay đổi điều kiện thí nghiệm.

## 8. Vị trí trong nghiên cứu hiện có

Ý tưởng memory cho VLA đã có nhiều hướng. [MemoryVLA](https://arxiv.org/abs/2508.19236) nghiên cứu perceptual/cognitive memory và memory-conditioned action generation; [official code](https://github.com/shihao1895/MemoryVLA) có training/evaluation riêng và recipe8×A100. Đây là nguồn tham khảo thiết kế/ablation, không phải dependency có thể thay trực tiếp vào SmolVLA2×T4.

[OnEvoMemory](https://arxiv.org/abs/2608.08749) nghiên cứu chọn/lưu kinh nghiệm từ demonstrations và online rollouts. [LIBERO-Plus](https://arxiv.org/abs/2510.13626) cho thấy benchmark success và robustness trước thay đổi observation/instruction cần được phân biệt. Các nguồn này không chứng minh phương pháp của repository sẽ hiệu quả, và đây chưa phải systematic literature review.

Một thesis có thể đóng góp bằng memory theo sự kiện với chi phí thấp, quản lý độ không chắc chắn, hoặc phân tích nhân quả rõ ràng về khi nào memory giúp/hại policy nhỏ. Chỉ nối thêm text và chạy LoRA chưa đủ thiết lập novelty. Kết quả âm vẫn có giá trị nếu tách được lỗi grounding, interface và training bằng đối chứng có thể tái lập.
