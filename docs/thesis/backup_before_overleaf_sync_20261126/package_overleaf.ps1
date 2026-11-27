<#
.SYNOPSIS
    Dong goi ban thao khoa luan LaTeX cho Overleaf.
.DESCRIPTION
    Script tu dong:
    1. Dong bo toan bo hinh anh tu docs/thesis/figures sang docs/thesis/latex/figures.
    2. Kiem tra va chuan hoa duong dan ../figures/ thanh figures/.
    3. Tao thu muc sach output/overleaf_upload (Ctrl+A keo tha vao du an Overleaf dang mo).
    4. Tao file zip sach output/thesis_overleaf.zip (Upload Project khi tao du an moi).
    5. Loai bo toan bo file rac trung gian (.aux, .log, .toc, .pdf build...).
.PARAMETER Open
    Mo ngay thu muc output/overleaf_upload tren Windows File Explorer de keo tha vao Overleaf.
.PARAMETER NoZip
    Chi tao thu muc staging ma khong nen file zip.
.EXAMPLE
    .\package_overleaf.ps1
    .\package_overleaf.ps1 -Open
    .\package_overleaf.ps1 -NoZip
#>

[CmdletBinding()]
param(
    [switch]$Open = $false,
    [switch]$NoZip = $false
)

$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    $scriptDir = (Get-Item .).FullName
}

# Tim thu muc goc cua repo
$repoRoot = (Get-Item "$scriptDir\..\..\..").FullName
$pythonScript = Join-Path $repoRoot "scripts\package_thesis_overleaf.py"

if (-not (Test-Path $pythonScript)) {
    throw "Khong tim thay script xu ly tai: $pythonScript"
}

# Kiem tra python tren he thong
$pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command "python3" -ErrorAction SilentlyContinue
}
if (-not $pythonCmd) {
    throw "Khong tim thay Python tren PATH de chay script dong goi."
}

$pyArgs = @($pythonScript)
if ($Open) {
    $pyArgs += "--open"
}
if ($NoZip) {
    $pyArgs += "--no-zip"
}

Write-Host ">>> Dang khoi chay tien trinh dong goi Overleaf..." -ForegroundColor Cyan
& $pythonCmd.Source @pyArgs

if ($LASTEXITCODE -ne 0) {
    throw "Tien trinh dong goi that bai (Exit code $LASTEXITCODE)."
}
