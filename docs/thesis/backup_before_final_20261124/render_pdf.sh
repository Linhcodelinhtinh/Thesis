#!/usr/bin/env bash
# Bien dich khoa luan LaTeX sang PDF bang XeLaTeX
# Su dung tren Linux, WSL hoac Git Bash (Windows)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${REPO_DIR}/output/pdf"
DEST_PDF="${OUTPUT_DIR}/khoa-luan-luu-quang-linh-ban-thao.pdf"

FULL_BUILD=0
CLEAN_BUILD=0
OPEN_PDF=0
SYNC_OUTPUT=1

# Phan tich tham so truyen vao
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--full)
            FULL_BUILD=1
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=1
            shift
            ;;
        -o|--open)
            OPEN_PDF=1
            shift
            ;;
        --no-sync)
            SYNC_OUTPUT=0
            shift
            ;;
        -h|--help)
            echo "Su dung: ./render_pdf.sh [lua chon]"
            echo "Lua chon:"
            echo "  -f, --full     Chay 4-pass day du (XeLaTeX -> BibTeX -> XeLaTeX -> XeLaTeX)"
            echo "  -c, --clean    Don dep cac file trung gian truoc khi bien dich"
            echo "  -o, --open     Mo file PDF sau khi bien dich thanh cong"
            echo "  --no-sync      Khong copy sang thu muc output/pdf/"
            echo "  -h, --help     Hien thi huong dan nay"
            exit 0
            ;;
        *)
            echo "Tham so khong hop le: $1"
            echo "Dung './render_pdf.sh --help' de xem huong dan."
            exit 1
            ;;
    esac
done

clean_files() {
    echo ">>> Dang don dep file trung gian..."
    find "${SCRIPT_DIR}" -type f \( \
        -name "*.aux" -o \
        -name "*.bbl" -o \
        -name "*.blg" -o \
        -name "*.lof" -o \
        -name "*.lot" -o \
        -name "*.log" -o \
        -name "*.toc" -o \
        -name "*.out" -o \
        -name "*.fls" -o \
        -name "*.fdb_latexmk" -o \
        -name "*.synctex.gz" \
    \) -delete 2>/dev/null || true
    echo ">>> Da don dep xong file trung gian."
}

cd "${SCRIPT_DIR}"

if [[ ${CLEAN_BUILD} -eq 1 ]]; then
    clean_files
fi

if ! command -v xelatex &>/dev/null; then
    echo "Error: Khong tim thay 'xelatex' trong PATH. Vui long cai dat TeX Live / MiKTeX."
    exit 1
fi

START_TIME=$(date +%s)

if [[ ${FULL_BUILD} -eq 1 ]]; then
    echo ">>> [1/4] Chay XeLaTeX pass 1..."
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex

    if command -v bibtex &>/dev/null; then
        echo ">>> [2/4] Chay BibTeX de xu ly tai lieu tham khao..."
        bibtex main
    fi

    echo ">>> [3/4] Chay XeLaTeX pass 2..."
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex

    echo ">>> [4/4] Chay XeLaTeX pass 3 (chot cross-reference & so trang)..."
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
else
    echo ">>> Chay XeLaTeX nhanh (1-pass)..."
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

GENERATED_PDF="${SCRIPT_DIR}/main.pdf"
if [[ ! -s "${GENERATED_PDF}" ]]; then
    echo "Error: Khong tim thay ${GENERATED_PDF} hoac file rong. Kiem tra main.log."
    exit 1
fi

echo "--------------------------------------------------------"
echo ">>> BIEN DICH THANH CONG trong ${DURATION}s!"
echo ">>> File: ${GENERATED_PDF}"

if [[ ${SYNC_OUTPUT} -eq 1 ]]; then
    mkdir -p "${OUTPUT_DIR}"
    cp -f "${GENERATED_PDF}" "${DEST_PDF}"
    echo ">>> Da dong bo sang: ${DEST_PDF}"
fi
echo "--------------------------------------------------------"

if [[ ${OPEN_PDF} -eq 1 ]]; then
    echo ">>> Dang mo file PDF..."
    if command -v start &>/dev/null; then
        start "${GENERATED_PDF}"
    elif command -v xdg-open &>/dev/null; then
        xdg-open "${GENERATED_PDF}" &>/dev/null &
    elif command -v open &>/dev/null; then
        open "${GENERATED_PDF}"
    else
        echo "Khong tim thay trinh mo file mac dinh (start/xdg-open/open)."
    fi
fi
