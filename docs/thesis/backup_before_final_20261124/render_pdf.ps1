<#
.SYNOPSIS
    Bien dich khoa luan LaTeX sang PDF bang XeLaTeX.
.DESCRIPTION
    Script ho tro build nhanh (1 pass) hoac build day du co tai lieu tham khao (4 pass: XeLaTeX -> BibTeX -> XeLaTeX -> XeLaTeX).
    Co the chay tu bat ky thu muc nao.
.PARAMETER Full
    Chay che do build day du (cap nhat muc luc, hinh anh, bang bieu va trich dan BibTeX).
.PARAMETER Clean
    Don dep cac file trung gian (.aux, .log, .out, .toc, .lof, .lot, .bbl, .blg).
.PARAMETER Open
    Mo file PDF sau khi bien dich thanh cong bang trinh doc PDF mac dinh cua he thong.
.PARAMETER SyncOutput
    Sao chep file PDF dau ra vao thu muc output/pdf/ theo quy uoc cua du an.
.EXAMPLE
    .\render_pdf.ps1
    .\render_pdf.ps1 -Full
    .\render_pdf.ps1 -Full -Open
    .\render_pdf.ps1 -Clean
#>

[CmdletBinding()]
param(
    [switch]$Full = $false,
    [switch]$Clean = $false,
    [switch]$Open = $false,
    [switch]$SyncOutput = $true
)

$ErrorActionPreference = "Stop"

# Thu muc chua script (chinh la docs/thesis/latex)
$latexDir = $PSScriptRoot
if (-not $latexDir) {
    $latexDir = (Get-Item .).FullName
}

$repoDir = (Get-Item "$latexDir\..\..\..").FullName
$outputDir = Join-Path $repoDir "output\pdf"
$destPdf = Join-Path $outputDir "khoa-luan-luu-quang-linh-ban-thao.pdf"

function Invoke-Clean {
    param([string]$Dir)
    $extensions = @("*.aux", "*.bbl", "*.blg", "*.lof", "*.lot", "*.log", "*.toc", "*.out", "*.fls", "*.fdb_latexmk", "*.synctex.gz")
    Write-Host ">>> Dang don dep file trung gian..." -ForegroundColor Yellow
    foreach ($ext in $extensions) {
        Get-ChildItem -Path $Dir -Filter $ext -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    }
    Write-Host ">>> Da don dep xong file trung gian." -ForegroundColor Green
}

Push-Location $latexDir
try {
    if ($Clean) {
        Invoke-Clean -Dir $latexDir
    }

    if (-not (Get-Command "xelatex" -ErrorAction SilentlyContinue)) {
        throw "Khong tim thay 'xelatex' trong PATH. Vui long cai dat MiKTeX hoac TeX Live va them vao PATH."
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    if ($Full) {
        Write-Host ">>> [1/4] Chay XeLaTeX pass 1..." -ForegroundColor Cyan
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass 1 that bai (Exit code $LASTEXITCODE). Xem main.log de biet chi tiet." }

        if (Get-Command "bibtex" -ErrorAction SilentlyContinue) {
            Write-Host ">>> [2/4] Chay BibTeX de xu ly tai lieu tham khao..." -ForegroundColor Cyan
            & bibtex main
            if ($LASTEXITCODE -ne 0) { throw "BibTeX that bai (Exit code $LASTEXITCODE). Xem main.blg de biet chi tiet." }
        } else {
            Write-Warning "Khong tim thay bibtex, bo qua buoc xu ly trich dan."
        }

        Write-Host ">>> [3/4] Chay XeLaTeX pass 2..." -ForegroundColor Cyan
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass 2 that bai (Exit code $LASTEXITCODE). Xem main.log de biet chi tiet." }

        Write-Host ">>> [4/4] Chay XeLaTeX pass 3 (chot cross-reference & so trang)..." -ForegroundColor Cyan
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass 3 that bai (Exit code $LASTEXITCODE). Xem main.log de biet chi tiet." }
    } else {
        Write-Host ">>> Chay XeLaTeX nhanh (1-pass)..." -ForegroundColor Cyan
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX that bai (Exit code $LASTEXITCODE). Xem main.log de biet chi tiet." }
    }

    $sw.Stop()

    $generatedPdf = Join-Path $latexDir "main.pdf"
    if (-not (Test-Path $generatedPdf) -or (Get-Item $generatedPdf).Length -le 0) {
        throw "Khong tim thay file main.pdf sau khi bien dich. Kiem tra main.log."
    }

    $fileInfo = Get-Item $generatedPdf
    $fileSizeKB = [math]::Round($fileInfo.Length / 1KB, 1)

    Write-Host "--------------------------------------------------------" -ForegroundColor Green
    Write-Host ">>> BIEN DICH THANH CONG trong $($sw.Elapsed.TotalSeconds.ToString('F1'))s!" -ForegroundColor Green
    Write-Host ">>> File: $generatedPdf ($fileSizeKB KB)" -ForegroundColor Green

    if ($SyncOutput) {
        if (-not (Test-Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        }
        Copy-Item -LiteralPath $generatedPdf -Destination $destPdf -Force
        Write-Host ">>> Da dong bo sang: $destPdf" -ForegroundColor Green
    }

    Write-Host "--------------------------------------------------------" -ForegroundColor Green

    if ($Open) {
        Write-Host ">>> Dang mo file PDF..." -ForegroundColor Cyan
        Invoke-Item $generatedPdf
    }
} finally {
    Pop-Location
}
