#!/usr/bin/env python
"""Download pinned LIBERO-Mem task demonstrations (raw or refined) with SHA-256 validation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

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
                   "16ddf346d85a647d3611cb37b2b68fd82b32e6d456359b332ed0268a4db5cf2a", 25983808),
        }
    },
    "refined": {
        "repo": "libero-mem/LIBERO-Mem",
        "revision": "6df130c05b82e09294fa9534e84a069687b72d33",
        "default_dir": ROOT / "data/libero_mem",
        "hashes": {
            "T7": ("KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate_demo.hdf5",
                   "8cbe1c2e7b5d1664fdc7e08c773aa0beabbc4ba5079bf47d4a10f9e6c9190757", 23366117457),
            "T9": ("KITCHEN_SCENE1_9_put_the_cream_cheese_in_the_nearest_basket_and_place_that_basket_in_the_center_demo.hdf5",
                   "f825fa2f8094f5ec45994e354dcee0ded16e8c6f9d046d4a0e2b919441246748", 20863261271),
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
    p.add_argument("--tasks", nargs="+", choices=["T7", "T9"], default=["T7", "T9"],
                   help="Tasks to download (default: T7 T9)")
    p.add_argument("--kind", choices=["raw", "refined", "both"], default="refined",
                   help="Download raw (state/action only ~50MB) or refined (full RGB ~22GB/task)")
    p.add_argument("--out-dir", type=Path, help="Override destination directory")
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

            if dest.is_file() and dest.stat().st_size == expected_size:
                print(f"   Đã tồn tại file đúng kích thước. Đang xác thực SHA-256...")
                actual_sha = sha256_file(dest)
                if actual_sha == expected_sha:
                    print(f"   ✓ File đã có sẵn và SHA-256 khớp 100%! Bỏ qua download.")
                    continue
                else:
                    print(f"   ⚠️ SHA-256 không khớp, sẽ tải lại.")

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
            assert actual_sha == expected_sha, f"SHA-256 mismatch! Nhận: {actual_sha}, Mong đợi: {expected_sha}"
            print(f"   ✓ Tải thành công và hash SHA-256 đã khớp ({actual_sha[:12]}...)!")

    print("\n🎉 TẤT CẢ CÁC TÁC VỤ ĐÃ HOÀN TẤT VÀ XÁC THỰC TOÀN VẸN!")


if __name__ == "__main__":
    main()
