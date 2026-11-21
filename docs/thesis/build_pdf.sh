#!/usr/bin/env bash
# Official thesis verification and build script (XeLaTeX -> BibTeX)
# Mirrors docs/thesis/build_pdf.ps1

set -e

FULL=0
SKIP_HASH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--full)
            FULL=1
            shift
            ;;
        --skip-hash)
            SKIP_HASH=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

THESIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATEX_DIR="${THESIS_DIR}/latex"
ABSTRACT_PATH="${LATEX_DIR}/content/abstract.tex"
ABSTRACT_HASH_PATH="${THESIS_DIR}/evidence/abstract.sha256"
REPO_DIR="$(cd "${THESIS_DIR}/../.." && pwd)"
OUTPUT_DIR="${REPO_DIR}/output/pdf"
OUTPUT_PDF="${OUTPUT_DIR}/khoa-luan-luu-quang-linh-ban-thao.pdf"

if [[ ${SKIP_HASH} -eq 0 ]]; then
    if [[ ! -f "${ABSTRACT_PATH}" || ! -f "${ABSTRACT_HASH_PATH}" ]]; then
        echo "Error: Abstract or checksum file is missing."
        exit 1
    fi

    EXPECTED_HASH=$(tr -d '[:space:]' < "${ABSTRACT_HASH_PATH}" | tr '[:lower:]' '[:upper:]')
    CURRENT_HASH=$(sha256sum "${ABSTRACT_PATH}" | awk '{print $1}' | tr '[:lower:]' '[:upper:]')

    if [[ "${CURRENT_HASH}" != "${EXPECTED_HASH}" ]]; then
        echo "Error: Abstract differs from its author-approved checksum."
        echo "Review it and update evidence/abstract.sha256 explicitly before building, or use --skip-hash."
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}"

cd "${LATEX_DIR}"

if [[ ${FULL} -eq 1 ]]; then
    echo ">>> Running Full Build (XeLaTeX -> BibTeX -> XeLaTeX -> XeLaTeX)..."
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
    bibtex main
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
else
    echo ">>> Running Fast Build (XeLaTeX single-pass)..."
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
fi

if [[ ! -s "main.pdf" ]]; then
    echo "Error: main.pdf was not generated. Check docs/thesis/latex/main.log for details."
    exit 1
fi

cp -f "main.pdf" "${OUTPUT_PDF}"
echo ">>> PDF successfully built and saved to: ${OUTPUT_PDF}"
