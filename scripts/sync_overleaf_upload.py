#!/usr/bin/env python3
"""
sync_overleaf_upload.py
-----------------------
Dong bo hoa cac phien ban moi nhat cua ban thao Thesis sang:
  output/overleaf_upload/

Muc tieu:
- Giu cho thu muc output/overleaf_upload luon sach se, khong file rac build (.aux, .log, .toc...).
- Tu dong cap nhat cac hinh anh tu docs/thesis/figures sang figures/.
- Tu dong chuan hoa duong dan ../figures/ thanh figures/ de khong gay loi tren Overleaf.
- Ho tro tuy chon mo ngay thu muc tren Windows Explorer sau khi dong bo (--open).
- Ho tro tuy chon nen them file zip (--zip).
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Cac thu muc co so (tinh tu goc repository)
REPO_ROOT = Path(__file__).resolve().parent.parent
LATEX_DIR = REPO_ROOT / "docs" / "thesis" / "latex"
FIGURES_DIR = REPO_ROOT / "docs" / "thesis" / "figures"
TARGET_DIR = REPO_ROOT / "output" / "overleaf_upload"
ZIP_TARGET = REPO_ROOT / "output" / "thesis_overleaf.zip"

ALLOWED_DIRS = {"assets", "figures", "content", "frontmatter", "appendices"}
ALLOWED_ROOT_FILES = {"main.tex", "references.bib"}
IGNORED_EXTS = {
    ".aux", ".log", ".toc", ".lof", ".lot", ".bbl", ".blg",
    ".out", ".fls", ".fdb_latexmk", ".synctex.gz",
    ".ps1", ".sh", ".txt", ".meta"
}
IGNORED_FILES = {"main.pdf", "image_note.txt"}


def sync_figures_to_latex() -> int:
    """Sao chep tat ca figure moi nhat tu docs/thesis/figures sang docs/thesis/latex/figures."""
    if not FIGURES_DIR.exists():
        return 0
    dest = LATEX_DIR / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in FIGURES_DIR.iterdir():
        if item.is_file() and item.name not in IGNORED_FILES:
            dest_file = dest / item.name
            shutil.copy2(item, dest_file)
            count += 1
    return count


def sanitize_tex_paths() -> int:
    """Kiem tra va chuan hoa duong dan hinh anh trong cac file .tex."""
    fixed = 0
    for root, _, files in os.walk(LATEX_DIR):
        for f in files:
            if f.endswith(".tex"):
                fpath = Path(root) / f
                try:
                    text = fpath.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                if "../figures/" in text:
                    text_fixed = text.replace("../figures/", "figures/")
                    fpath.write_text(text_fixed, encoding="utf-8")
                    fixed += 1
    return fixed


def collect_clean_files():
    """Thu thap danh sach tat ca cac file hop le can dong bo sang overleaf_upload."""
    clean_files = []
    for root, dirs, files in os.walk(LATEX_DIR):
        rel_root = Path(root).relative_to(LATEX_DIR)
        parts = rel_root.parts

        if len(parts) == 0:
            for f in files:
                if f in ALLOWED_ROOT_FILES:
                    clean_files.append((Path(root) / f, Path(f)))
        else:
            top_sub = parts[0]
            if top_sub in ALLOWED_DIRS:
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext not in IGNORED_EXTS and f not in IGNORED_FILES:
                        arcname = rel_root / f
                        clean_files.append((Path(root) / f, arcname))
    return sorted(clean_files, key=lambda x: str(x[1]))


def sync_to_destination(files) -> int:
    """Dong bo vao output/overleaf_upload (xoa cac file cu khong con ton tai)."""
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_path, rel_path in files:
        target_path = TARGET_DIR / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, target_path)
        copied += 1
    return copied


def make_zip(files) -> Path:
    """Dong goi them file zip neu co yeu cau."""
    ZIP_TARGET.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_TARGET, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_path, rel_path in files:
            norm_arc = str(rel_path).replace("\\", "/")
            zf.write(src_path, norm_arc)
    return ZIP_TARGET


def open_in_explorer(path: Path) -> None:
    """Mo thu muc tren Windows Explorer."""
    if sys.platform == "win32":
        try:
            os.startfile(str(path))
        except Exception:
            subprocess.run(["explorer", str(path)])
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)])
    else:
        subprocess.run(["xdg-open", str(path)])


def main():
    parser = argparse.ArgumentParser(
        description="Dong bo ban thao khoa luan moi nhat sang output/overleaf_upload."
    )
    parser.add_argument(
        "-o", "--open",
        action="store_true",
        help="Tu dong mo thu muc output/overleaf_upload tren Windows Explorer sau khi dong bo."
    )
    parser.add_argument(
        "-z", "--zip",
        action="store_true",
        help="Tao them file output/thesis_overleaf.zip."
    )
    args = parser.parse_args()

    print("=" * 65)
    print(" DONG BO PHIEN BAN MOI SANG output/overleaf_upload")
    print("=" * 65)

    # 1. Dong bo figures
    num_fig = sync_figures_to_latex()
    print(f"[*] Da dong bo {num_fig} hinh anh tu docs/thesis/figures -> latex/figures.")

    # 2. Chuan hoa tex
    num_fixed = sanitize_tex_paths()
    if num_fixed > 0:
        print(f"[*] Da sua duong dan '../figures/' -> 'figures/' tren {num_fixed} file .tex.")

    # 3. Thu thap & Dong bo sang TARGET_DIR
    files = collect_clean_files()
    copied = sync_to_destination(files)
    print(f"[+] Da dong bo thanh cong {copied} tep nguon vao:")
    print(f"    -> {TARGET_DIR}")

    # Tinh dung luong thu muc
    total_bytes = sum(f[0].stat().st_size for f in files)
    total_mb = total_bytes / (1024 * 1024)
    print(f"[*] Tong dung luong: {total_mb:.2f} MB ({len(files)} teps).")

    # 4. Neu co yeu cau zip
    if args.zip:
        zip_path = make_zip(files)
        zip_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"[+] Da tao them file zip: {zip_path} ({zip_mb:.2f} MB)")

    # 5. Mo Explorer neu co yeu cau
    if args.open:
        print(f"[*] Dang mo Windows Explorer tai: {TARGET_DIR} ...")
        open_in_explorer(TARGET_DIR)

    print("=" * 65)
    print("HUONG DAN CAP NHAT OVERLEAF:")
    print(f"1. Mo thu muc: {TARGET_DIR}")
    print("2. Nhan Ctrl + A de chon tat ca cac tệp va thu muc con.")
    print("3. Keo va tha thang vao cot danh sach tep (File Tree) tren Overleaf.")
    print("4. Chon 'Overwrite' -> Bấm Recompile tren Overleaf.")
    print("=" * 65)


if __name__ == "__main__":
    main()
