#!/usr/bin/env bash
# Goi script bien dich khoa luan LaTeX tu thu muc goc cua repository

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/docs/thesis/latex/render_pdf.sh"

exec "${TARGET_SCRIPT}" "$@"
