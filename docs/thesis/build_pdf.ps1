param(
    [switch]$Full = $false,
    [switch]$SkipHash = $false
)

$ErrorActionPreference = "Stop"
$thesisDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$latexDir = Join-Path $thesisDir "latex"
$abstractPath = Join-Path $latexDir "content\abstract.tex"
$abstractHashPath = Join-Path $thesisDir "evidence\abstract.sha256"
$repoDir = Split-Path -Parent (Split-Path -Parent $thesisDir)
$outputDir = Join-Path $repoDir "output\pdf"
$outputPdf = Join-Path $outputDir "khoa-luan-luu-quang-linh-ban-thao.pdf"

if (-not $SkipHash) {
    if (-not (Test-Path $abstractPath) -or -not (Test-Path $abstractHashPath)) {
        throw "Abstract or its author-owned checksum is missing."
    }
    $expectedAbstractHash = (Get-Content -LiteralPath $abstractHashPath -Raw).Trim().ToUpperInvariant()
    if ($expectedAbstractHash -notmatch '^[0-9A-F]{64}$') {
        throw "Invalid SHA-256 in $abstractHashPath"
    }
    $beforeAbstractHash = (Get-FileHash -LiteralPath $abstractPath -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($beforeAbstractHash -ne $expectedAbstractHash) {
        throw "Abstract differs from its author-approved checksum. Review it and update evidence/abstract.sha256 explicitly before building (or use -SkipHash for drafting)."
    }
}

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

Push-Location $latexDir
try {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    if ($Full) {
        Write-Host ">>> Running Full Build (XeLaTeX -> BibTeX -> XeLaTeX -> XeLaTeX)..." -ForegroundColor Cyan
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "First XeLaTeX pass failed with exit code $LASTEXITCODE" }
        & bibtex main | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "BibTeX failed with exit code $LASTEXITCODE" }
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Second XeLaTeX pass failed with exit code $LASTEXITCODE" }
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Final XeLaTeX pass failed with exit code $LASTEXITCODE" }
    } else {
        Write-Host ">>> Running Fast Build (XeLaTeX single-pass)..." -ForegroundColor Cyan
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass failed with exit code $LASTEXITCODE" }
    }
    $sw.Stop()

    if (-not $SkipHash) {
        $afterAbstractHash = (Get-FileHash -LiteralPath $abstractPath -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($afterAbstractHash -ne $expectedAbstractHash) {
            throw "Abstract changed during the build; refusing to publish the PDF."
        }
    }
    if (-not (Test-Path "main.pdf") -or (Get-Item "main.pdf").Length -le 0) {
        throw "main.pdf was not generated. Check docs/thesis/latex/main.log for details."
    }
    Copy-Item -LiteralPath "main.pdf" -Destination $outputPdf -Force
    if (-not (Test-Path $outputPdf) -or (Get-Item $outputPdf).Length -le 0) {
        throw "Built PDF was not copied to $outputPdf"
    }
    Write-Host ">>> PDF successfully built in $($sw.Elapsed.TotalSeconds.ToString('F1')) seconds!" -ForegroundColor Green
    Write-Host ">>> Abstract SHA-256 unchanged: $afterAbstractHash" -ForegroundColor Green
    Write-Host ">>> Saved to: $outputPdf" -ForegroundColor Green
} finally {
    Pop-Location
}
