# PROVENANCE LEDGER — SỔ KIỂM ĐỊNH NGUỒN VÀ MÃ BĂM THỰC NGHIỆM

Tài liệu này lưu trữ toàn bộ các mã băm Git commit, mã băm SHA-256 và định danh Git LFS OID của tất cả các thành phần dữ liệu, mã nguồn chuẩn mực, trọng số mô hình và báo cáo kiểm định phục vụ nghiên cứu khóa luận.

Việc lưu trữ tập trung tại tệp nội bộ này nhằm:
1. Đảm bảo tính khả kiểm, liêm chính học thuật và khả năng tái lập 100% của toàn bộ dữ liệu thực nghiệm.
2. Giữ cho văn phong của bản thảo khóa luận chính thức được mạch lạc, trang trọng, tập trung vào bản chất khoa học thay vì liệt kê các chuỗi ký tự băm kỹ thuật.

---

## 1. Nguồn chuẩn mực công khai (Upstream Public Repositories & Checkpoints)

| Thành phần | Loại định danh | Giá trị định danh (Git Commit / Revision) | Phạm vi & Vai trò |
| :--- | :--- | :--- | :--- |
| **LIBERO Benchmark** | Git Commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | Mã nguồn kho lưu trữ chính thức của LIBERO benchmark trên GitHub. |
| **LIBERO-Mem** | Git Commit | `0eee0defa4024e51511b6e15a79f30e9620209de` | Kho lưu trữ chính thức của LIBERO-Mem trên GitHub; chốt task map, logic evaluator và định nghĩa BDDL vật thể `cream_cheese_1` cho T9/T10. |
| **LIBERO-Mem Refined Data** | Git Revision / Commit | `6df130c05b82e09294fa9534e84a069687b72d33` | Bộ dữ liệu thao tác mẫu refined do tác giả LIBERO-Mem phát hành trên Hugging Face. |
| **LIBERO-Mem Raw Data** | Git Revision / Commit | `61917674055e8b1cbbfd22c1b54c28d20c1a1584` | Bộ dữ liệu trạng thái thô (raw states) do tác giả LIBERO-Mem phát hành trên Hugging Face. |
| **SmolVLA Checkpoint** | Git Revision / Commit | `31d453f7edd78c839a8bbc39744a292686daf0de` | Mô hình nền SmolVLA tiền huấn luyện trên LIBERO do Hugging Face / LeRobot công bố. |

---

## 2. Mã băm Git LFS OID của các tệp dữ liệu thô và mẫu công khai

Các mã băm dưới đây là SHA-256 Git LFS OID chính thức của các tệp HDF5 từ tác giả benchmark, có thể tra cứu và xác minh trực tiếp từ máy chủ lưu trữ công khai:

| Tệp dữ liệu / Tác vụ | Loại tệp | Git LFS SHA-256 OID | Mô tả |
| :--- | :--- | :--- | :--- |
| **T7 refined source** | HDF5 | `8cbe1c2e7b5d1664fdc7e08c773aa0beabbc4ba5079bf47d4a10f9e6c9190757` | 88 quỹ đạo mẫu refined của tác vụ T7. |
| **T8 raw source** | HDF5 | `77dcad3d4f04b36441a70dc95b4cd2131189c5ebfa105bf92ce5d2a8125a16af` | Tệp dữ liệu raw HDF5 của tác vụ T8. |
| **T8 refined source** | HDF5 | `399061494f4a0b4ac8a5ce646aa5755a6698ec65fbd4e6843143de5e95c1c555` | Tệp dữ liệu refined HDF5 của tác vụ T8. |
| **T9 refined source** | HDF5 | `f825fa2f8094f5ec45994e354dcee0ded16e8c6f9d046d4a0e2b919441246748` | Tệp dữ liệu refined HDF5 của tác vụ T9. |
| **T10 raw source** | HDF5 | `53afeac724e9ead7eedf289e9fc985f87b071aa9146b3694d6c25e16760bc72c` | Tệp dữ liệu raw HDF5 của tác vụ T10. |
| **T10 refined source** | HDF5 | `bcb9357581d17a6312a34c1f7bd855b9bb7357ca3d5a6aa1c5be2d2a58e1d85b` | Tệp dữ liệu refined HDF5 của tác vụ T10. |

---

## 3. Mã băm SHA-256 của các Artifacts, Môi trường và Báo cáo kiểm định nội bộ

Các mã băm SHA-256 dưới đây tương ứng với các tệp manifest cấu hình, bản kê dữ liệu và báo cáo kiểm định thực nghiệm được tạo lập và lưu trữ trong cây thư mục của đề tài:

| Mục kiểm định / Artifact | Định dạng | Mã băm SHA-256 | Ý nghĩa kiểm định |
| :--- | :--- | :--- | :--- |
| **Môi trường suy luận (Inference env)** | Package Manifest | `94d1fc4f7988beba96d62a98a8f97a4f9b8119c101036987287656aa93a9ea10` | Bản kê phiên bản thư viện Python 3.12.3, PyTorch 2.11.0, LeRobot 0.6.1 trên WSL/CUDA. |
| **Khóa phân chia dữ liệu Split V2** | Manifest | `5ac5eb8c8d5498b4422d010394337d0300030204c58df65a789561ba32f82ab6` | Khóa danh mục 598 train, 50 val, 50 retention và 20 nhóm raw unseen. |
| **Đối chuẩn dữ liệu (Alignment audit V2)** | Audit Report | `79ca820e542785f11cd820bfa246a32848cd8fb6af17f5858bb6b6648171b269` | Báo cáo kiểm tra tính nhất quán 180.447 cặp observation-action và 1.396 mẫu parity. |
| **Trọng số F0 V2 (No-memory adapter)** | Model Manifest | `04d9f259d21426615d26da0c23b70bfdd7f44df43bcdb3655265e58c4dee236b` | Manifest của adapter F0 V2 sau 5.000 bước huấn luyện. |
| **Kiểm tra runtime E0** | Run Manifest | `797c7ec40aa08dc8396950411a23835cdf1c979547a0d2e56df87dfea3c5cc76` | Bản ghi chạy thử 2 bước hành động trên B0/F0 trên môi trường mô phỏng. |
| **Pilot E0 B0/F0** | Run Manifest | `9a38b6144a5bdfeadd40340be64d73b15d28fdd272a7d9d2993505c68956ab74` | Bản ghi bốn lượt chạy chẩn đoán ban đầu của B0 và F0 trên trạng thái seen. |
| **Giao thức phát triển quyết định** | Protocol Spec | `5e19c8fa9a23bca8613525b39b350fa99977c80c15a7b72e62a036e2fd182e69` | Giao thức `libero_mem_decision_dev_v1` (12 dev states, 8 calibration states). |
| **Danh mục trạng thái thô T7** | State Inventory | `797b51fb196ad5e0fe161dfc40b2e91d23da173d94c5ec1dfb1637ed7e369187` | Bản kê 120 vector trạng thái thô của tác vụ T7. |
| **Báo cáo phát lại hành động tác giả T8** | Report | `a2c15af157c3585036f4ce663123613bac07c19170673500ee48b10a863929b9` | Ghi nhận đạt 3 milestone nhưng không thỏa mãn terminal. |
| **Báo cáo đối sánh trạng thái T8** | Report | `36faa0ac563d1bc49379bc3b70f2e55ea5c974ea482d450037126c4960620fb0` | Xác định độ lệch trạng thái robot bắt đầu vượt ngưỡng $10^{-5}$ từ bước 219. |
| **Mã nguồn kiểm định trạng thái T8** | Python Source | `ca04a99e61bca1e10c1a0b43f2a691ad19919a7b3faa152ae31d85c513a52f62` | Script đối chiếu vector trạng thái robot trước hành động. |
| **Báo cáo phát lại tác giả T9** | Report | `801d68b6737d5ae25d0ee4942a32e52b6cc1362a1311de0e3e55958408e64f18` | Xác nhận phát lại thành công trọn vẹn quỹ đạo mẫu T9. |
| **Báo cáo phát lại tác giả T10** | Report | `fe1c433204e0921a0d1b7256d7b1a8b38eb6fe58f354967da0149e2f00338df7` | Xác nhận phát lại thành công trọn vẹn quỹ đạo mẫu T10. |
| **Mã nguồn kiểm định phát lại tác giả** | Python Source | `f5e7796832749cff20f06288330956484507d5454aa0442b5abc543f1f2ada30` | Script thực thi author actions độc lập với mô hình chính sách. |
| **Bộ ghi quan hệ RGB trên T7** | Run Manifest | `1174a7c6d026c4593c802d29d5e636397ffa87306123f4429e00e9cc9e376428` | Manifest chạy bộ ghi trên 425 mẫu demo 1 của T7. |
| **Bộ ghi quan hệ RGB trên T8** | Run Manifest | `dfb008a673a8ea40f963d7900ea79c632644a056f216aa38a39d69a5651aa143` | Manifest chạy bộ ghi trên 613 mẫu demo 1 của T8. |
| **Bộ ghi quan hệ RGB trên T9** | Run Manifest | `885ad6ee980001f4678480075845a2b25814f3dd495f528363c863876ee311d2` | Manifest chạy bộ ghi trên 356 mẫu demo 1 của T9. |
| **Bộ ghi quan hệ RGB trên T10** | Run Manifest | `1bf30840c20bc043cb80b25d7261103943c079a812b97b61722f044958ba24b3` | Manifest chạy bộ ghi trên 309 mẫu demo 1 của T10. |
| **Báo cáo kiểm định bộ phát hiện T7** | Audit Report | `fceaaed7ba5a7b465293d1cfa5b14a0cb72a81bb339b5ca184b7378f87eb5e8b` | Báo cáo so khớp 425 khung hình ảnh camera ngoài với nhãn hộp bao. |
| **Mã nguồn kiểm định bộ phát hiện T7** | Python Source | `caeb1a12ee4f136b487a1c7c2062dbfb73eedb3468fc48ef1c93d2d03579fa3d` | Script tính toán IoU và độ chuẩn xác/đầy đủ của bộ phát hiện. |
| **Kiểm định tiền xử lý ảnh F0/E0** | Audit Report | `297ec298d24644c088d08ea6a003e98b6fb6bc94449453a835ec55a957aa0ff5` | Báo cáo kiểm định tính nhất quán của pipeline biến đổi ảnh `LiberoProcessorStep`. |
