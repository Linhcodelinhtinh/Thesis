param(
    [switch]$Full = $false
)

$ErrorActionPreference = "Stop"
$thesisDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$latexDir = Join-Path $thesisDir "latex"
$repoDir = Split-Path -Parent (Split-Path -Parent $thesisDir)
$outputDir = Join-Path $repoDir "output\pdf"
$outputPdf = Join-Path $outputDir "khoa-luan-luu-quang-linh-ban-thao.pdf"

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

Push-Location $latexDir
try {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    if ($Full) {
        Write-Host ">>> Running Full Build (XeLaTeX -> BibTeX -> XeLaTeX -> XeLaTeX)..." -ForegroundColor Cyan
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex | Out-Null
        & bibtex main | Out-Null
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex | Out-Null
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex | Out-Null
    } else {
        Write-Host ">>> Running Fast Build (XeLaTeX single-pass)..." -ForegroundColor Cyan
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex | Out-Null
    }
    $sw.Stop()

    if (Test-Path "main.pdf") {
        Copy-Item -Path "main.pdf" -Destination $outputPdf -Force
        Write-Host ">>> PDF successfully built in $($sw.Elapsed.TotalSeconds.ToString('F1')) seconds!" -ForegroundColor Green
        Write-Host ">>> Saved to: $outputPdf" -ForegroundColor Green
    } else {
        Write-Error "main.pdf was not generated. Check docs/thesis/latex/main.log for details."
    }
} finally {
    Pop-Location
}
