<#
.SYNOPSIS
    Dong bo ban thao khoa luan LaTeX sang output/overleaf_upload/.
.DESCRIPTION
    Script tu dong:
    1. Quet va cap nhat moi cac hinh anh tu docs/thesis/figures sang figures/.
    2. Chuan hoa duong dan hinh anh (tranh loi sandbox .. tren Overleaf).
    3. Dong bo cac file sach (.tex, .bib, content, figures, assets, frontmatter, appendices)
       vao thu muc output/overleaf_upload/.
    4. Loai bo 100% cac file rac build trung gian (.aux, .log, .toc, .pdf build...).
.PARAMETER Open
    Tu dong mo thu muc output/overleaf_upload tren Windows File Explorer de keo tha vao Overleaf.
.PARAMETER Zip
    Tao them file zip output/thesis_overleaf.zip phong khi can.
.EXAMPLE
    .\sync_overleaf.ps1
    .\sync_overleaf.ps1 -Open
    .\sync_overleaf.ps1 -Open -Zip
#>

[CmdletBinding()]
param(
    [switch]$Open = $false,
    [switch]$Zip = $false
)

$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    $scriptDir = (Get-Item .).FullName
}

$repoRoot = (Get-Item "$scriptDir\..\..\..").FullName
$pythonScript = Join-Path $repoRoot "scripts\sync_overleaf_upload.py"

if (-not (Test-Path $pythonScript)) {
    throw "Khong tim thay script xu ly tai: $pythonScript"
}

$pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command "python3" -ErrorAction SilentlyContinue
}
if (-not $pythonCmd) {
    throw "Khong tim thay Python tren PATH de chay script dong bo."
}

$pyArgs = @($pythonScript)
if ($Open) {
    $pyArgs += "--open"
}
if ($Zip) {
    $pyArgs += "--zip"
}

Write-Host ">>> Dang dong bo ban thao moi sang output/overleaf_upload..." -ForegroundColor Cyan
& $pythonCmd.Source @pyArgs

if ($LASTEXITCODE -ne 0) {
    throw "Tien trinh dong bo that bai (Exit code $LASTEXITCODE)."
}
