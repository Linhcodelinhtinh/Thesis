#!/usr/bin/env python3
"""
package_thesis_overleaf.py
--------------------------
Dong goi ban thao khoa luan LaTeX cho Overleaf.
Ho tro ca 2 che do:
1. Tao thu muc sach (output/overleaf_upload) de keo tha truc tiep vao du an Overleaf hien co.
2. Tao file zip sach (output/thesis_overleaf.zip) de upload tao du an moi.

Cac tinh nang tu dong:
- Dong bo figures tu docs/thesis/figures sang docs/thesis/latex/figures.
- Kiem tra va chuan hoa duong dan hinh anh (khong de ../figures lam loi sandbox Overleaf).
- Loc bo toan bo file trung gian / file rac build (.aux, .log, .toc, .pdf build, script...).
- Tuy chon mo ngay thu muc tren Windows Explorer de keo tha.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Cac thu muc mac dinh (tinh tu goc repository)
REPO_ROOT = Path(__file__).resolve().parent.parent
LATEX_DIR = REPO_ROOT / "docs" / "thesis" / "latex"
FIGURES_DIR = REPO_ROOT / "docs" / "thesis" / "figures"
OUTPUT_DIR = REPO_ROOT / "output"
STAGING_DIR = OUTPUT_DIR / "overleaf_upload"
ZIP_OUTPUTS = [
    OUTPUT_DIR / "thesis_overleaf.zip",
    REPO_ROOT / "docs" / "thesis" / "thesis_overleaf.zip",
]

ALLOWED_DIRS = {"assets", "figures", "content", "frontmatter", "appendices"}
ALLOWED_ROOT_FILES = {"main.tex", "references.bib"}
IGNORED_EXTS = {
    ".aux", ".log", ".toc", ".lof", ".lot", ".bbl", ".blg",
    ".out", ".fls", ".fdb_latexmk", ".synctex.gz",
    ".ps1", ".sh", ".txt", ".meta"
}
IGNORED_FILES = {"main.pdf", "image_note.txt"}


def sync_figures() -> None:
    """Dong bo cac hinh anh tu docs/thesis/figures sang docs/thesis/latex/figures."""
    if not FIGURES_DIR.exists():
        return
    dest_figures = LATEX_DIR / "figures"
    dest_figures.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in FIGURES_DIR.iterdir():
        if item.is_file() and item.name not in IGNORED_FILES:
            dest_file = dest_figures / item.name
            shutil.copy2(item, dest_file)
            count += 1
    print(f"[*] Da dong bo {count} tep hinh anh tu '{FIGURES_DIR.name}' sang '{dest_figures.relative_to(REPO_ROOT)}'.")


def sanitize_tex_paths() -> None:
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
                    print(f"[*] Da sua duong dan '../figures/' -> 'figures/' trong {fpath.name}")
    if fixed > 0:
        print(f"[*] Da chuan hoa duong dan tren {fixed} file .tex de tuong thich Overleaf.")


def collect_clean_files():
    """Thu thap danh sach cac file sach can dong goi."""
    clean_files = []
    for root, dirs, files in os.walk(LATEX_DIR):
        rel_root = Path(root).relative_to(LATEX_DIR)
        parts = rel_root.parts

        if len(parts) == 0:
            # Thu muc goc cua latex/
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


def build_staging_folder(files) -> Path:
    """Tao thu muc staging output/overleaf_upload de keo tha truc tiep."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    for src_path, arcname in files:
        target_path = STAGING_DIR / arcname
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, target_path)

    print(f"[+] Thu muc staging san sang: {STAGING_DIR}")
    return STAGING_DIR


def build_zip_archives(files) -> list:
    """Dong goi file zip sach."""
    created = []
    for zip_path in ZIP_OUTPUTS:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for src_path, arcname in files:
                # Dung dau gach cheo xuoi (forward slash) chuan cho ZIP / Overleaf
                norm_arc = str(arcname).replace("\\", "/")
                zf.write(src_path, norm_arc)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"[+] File zip san sang: {zip_path} ({size_mb:.2f} MB)")
        created.append(zip_path)
    return created


def open_in_explorer(path: Path) -> None:
    """Mo thu muc tren Windows Explorer de thao tac keo tha."""
    if sys.platform == "win32":
        try:
            os.startfile(str(path))
            print(f"[*] Da mo File Explorer tai: {path}")
        except Exception as e:
            subprocess.run(["explorer", str(path)])
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)])
    else:
        subprocess.run(["xdg-open", str(path)])


def main():
    parser = argparse.ArgumentParser(
        description="Dong goi ban thao khoa luan LaTeX cho Overleaf."
    )
    parser.add_argument("--no-zip", action="store_true", help="Bo qua buoc tao file .zip.")
    parser.add_argument("--no-folder", action="store_true", help="Bo qua buoc tao thu muc staging.")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Tu dong mo thu muc staging tren Windows Explorer de keo tha vao Overleaf.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DONG GOI BAN THAO THESIS CHO OVERLEAF")
    print("=" * 60)

    # 1. Dong bo figures & chuan hoa
    sync_figures()
    sanitize_tex_paths()

    # 2. Thu thap file sach
    files = collect_clean_files()
    print(f"[*] Da thu thap {len(files)} tep nguon sach (khong co file build rac).")

    # 3. Tao staging folder
    if not args.no_folder:
        build_staging_folder(files)

    # 4. Tao zip archive
    if not args.no_zip:
        build_zip_archives(files)

    # 5. Mo thu muc neu co yeu cau
    if args.open:
        target = STAGING_DIR if not args.no_folder else OUTPUT_DIR
        open_in_explorer(target)

    print("=" * 60)
    print("HOAN TAT! HUONG DAN SU DUNG:")
    print("1. Neu cap nhat du an Overleaf DA CO:")
    print(f"   -> Mo thu muc: {STAGING_DIR}")
    print("   -> Nhan Ctrl+A, keo tha tat ca vao cay thu muc tren Overleaf, chon Overwrite.")
    print("2. Neu tao du an Overleaf MOI:")
    print(f"   -> Chon 'New Project' -> 'Upload Project' -> Chon file zip: {ZIP_OUTPUTS[0]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
