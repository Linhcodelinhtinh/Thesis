#!/usr/bin/env python
"""Download pinned LIBERO-Mem task demonstrations (raw or refined) with SHA-256 validation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]

PINNED_CONFIG = {
    "raw": {
        "repo": "libero-mem/LIBERO-Mem-Raw",
        "revision": "61917674055e8b1cbbfd22c1b54c28d20c1a1584",
        "default_dir": ROOT / "data/libero_mem_raw",
        "hashes": {
            "T7": ("KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate_demo.hdf5",
                   "257f2a48d7c127c1abe45d0feea0f146e1a01165693be5f535fef3a2dd48d71c", 48758920),
            "T9": ("KITCHEN_SCENE1_9_put_the_cream_cheese_in_the_nearest_basket_and_place_that_basket_in_the_center_demo.hdf5",
                   "16ddf346d85a647d3611cb37b2b68fd82b32e6d456359b332ed0268a4db5cf2a", 25979784),
            "T8": ("KITCHEN_SCENE1_8_rotate_the_3_bowls_on_their_plates_from_left_to_right_using_the_empty_plate_demo.hdf5",
                   "77dcad3d4f04b36441a70dc95b4cd2131189c5ebfa105bf92ce5d2a8125a16af", 74236352),
            "T10": ("KITCHEN_SCENE1_10_put_the_cream_cheese_in_the_nearest_basket_and_place_the_empty_basket_in_the_center_demo.hdf5",
                    "53afeac724e9ead7eedf289e9fc985f87b071aa9146b3694d6c25e16760bc72c", 25280904),
        }
    },
    "refined": {
        "repo": "libero-mem/LIBERO-Mem",
        "revision": "6df130c05b82e09294fa9534e84a069687b72d33",
        "default_dir": ROOT / "data/libero_mem",
        "metainfo": ("metainfo.json", "2afe9bc01f9579d4adeec63a6cf012f8560de3b5a4b7d5a96c0643c94777ad0c", 713618260),
        "hashes": {
            "T7": ("KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate_demo.hdf5",
                   "8cbe1c2e7b5d1664fdc7e08c773aa0beabbc4ba5079bf47d4a10f9e6c9190757", 23366117457),
            "T9": ("KITCHEN_SCENE1_9_put_the_cream_cheese_in_the_nearest_basket_and_place_that_basket_in_the_center_demo.hdf5",
                   "f825fa2f8094f5ec45994e354dcee0ded16e8c6f9d046d4a0e2b919441246748", 20858283876),
            "T8": ("KITCHEN_SCENE1_8_rotate_the_3_bowls_on_their_plates_from_left_to_right_using_the_empty_plate_demo.hdf5",
                   "399061494f4a0b4ac8a5ce646aa5755a6698ec65fbd4e6843143de5e95c1c555", 28219269155),
            "T10": ("KITCHEN_SCENE1_10_put_the_cream_cheese_in_the_nearest_basket_and_place_the_empty_basket_in_the_center_demo.hdf5",
                    "bcb9357581d17a6312a34c1f7bd855b9bb7357ca3d5a6aa1c5be2d2a58e1d85b", 19947570362),
        }
    }
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", choices=["T7", "T8", "T9", "T10"], default=["T7", "T8", "T9", "T10"],
                   help="Tasks to download (default: T7 T8 T9 T10)")
    p.add_argument("--kind", choices=["raw", "refined", "both"], default="refined",
                   help="Download raw (state/action only ~25-75MB) or refined (full RGB ~20-28GB/task)")
    p.add_argument("--out-dir", type=Path, help="Override destination directory")
    p.add_argument("--receipt-dir", type=Path, default=ROOT / "resources/manifests",
                   help="Directory for per-task source receipts")
    args = p.parse_args()

    kinds = ["raw", "refined"] if args.kind == "both" else [args.kind]

    for kind in kinds:
        cfg = PINNED_CONFIG[kind]
        target_dir = args.out_dir if args.out_dir else cfg["default_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=======================================================")
        print(f"📦 BẮT ĐẦU TẢI [{kind.upper()}]: Repo {cfg['repo']} (rev {cfg['revision'][:8]}...)")
        print(f"📂 Thư mục đích: {target_dir}")
        print(f"=======================================================")

        for task in args.tasks:
            filename, expected_sha, expected_size = cfg["hashes"][task]
            dest = target_dir / filename
            size_mb = expected_size / 1e6
            size_gb = expected_size / 1e9

            print(f"\n-> [{task}] {filename}")
            print(f"   Dung lượng dự kiến: {size_gb:.2f} GB ({size_mb:.1f} MB)")

            downloaded = None
            if dest.is_file() and dest.stat().st_size == expected_size:
                print(f"   Đã tồn tại file đúng kích thước. Đang xác thực SHA-256...")
                actual_sha = sha256_file(dest)
                if actual_sha == expected_sha:
                    print(f"   ✓ File đã có sẵn và SHA-256 khớp 100%! Bỏ qua download.")
                    downloaded = dest
                else:
                    print(f"   ⚠️ SHA-256 không khớp, sẽ tải lại.")

            if downloaded is None:
                print(f"   Đang tải từ Hugging Face...")
                downloaded_path = hf_hub_download(
                    repo_id=cfg["repo"],
                    filename=filename,
                    repo_type="dataset",
                    revision=cfg["revision"],
                    local_dir=target_dir,
                )
                downloaded = Path(downloaded_path)
            print(f"   Đang kiểm tra tính toàn vẹn SHA-256...")
            actual_sha = sha256_file(downloaded)
            if actual_sha != expected_sha:
                raise ValueError(f"SHA-256 mismatch! Nhận: {actual_sha}, Mong đợi: {expected_sha}")
            print(f"   ✓ Tải thành công và hash SHA-256 đã khớp ({actual_sha[:12]}...)!")

            receipt = dict(
                schema_version=1,
                classification="AUTHOR_RELEASE_LOCAL_SHA256_RECEIPT",
                task=task,
                source_kind=kind,
                repository=cfg["repo"],
                revision=cfg["revision"],
                source_metadata_url=(
                    f"https://huggingface.co/api/datasets/{cfg['repo']}/paths-info/{cfg['revision']}"
                ),
                sha256={filename: expected_sha},
                local_verification={
                    "task_file_path": str(downloaded.resolve()),
                    "task_file_size_bytes": expected_size,
                    "task_file_sha256_verified": True,
                    "verification_time_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            if kind == "refined":
                metainfo_name, metainfo_sha, metainfo_size = cfg["metainfo"]
                metainfo_path = ROOT / "data/libero_mem" / metainfo_name
                if not metainfo_path.is_file() or metainfo_path.stat().st_size != metainfo_size:
                    raise FileNotFoundError(f"Pinned metainfo is unavailable at expected size: {metainfo_path}")
                if sha256_file(metainfo_path) != metainfo_sha:
                    raise ValueError(f"Pinned metainfo SHA-256 mismatch: {metainfo_path}")
                receipt["sha256"][metainfo_name] = metainfo_sha
                receipt["local_verification"]["metainfo_sha256_verified"] = True
                receipt["local_verification"]["metainfo_size_bytes"] = metainfo_size
            receipt_path = args.receipt_dir / f"libero_mem_{task.lower()}_{kind}_receipt_v1.json"
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            print(f"   Receipt: {receipt_path}")

    print("\n🎉 TẤT CẢ CÁC TÁC VỤ ĐÃ HOÀN TẤT VÀ XÁC THỰC TOÀN VẸN!")


if __name__ == "__main__":
    main()
