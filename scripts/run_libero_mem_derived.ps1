# ==============================================================================
# LIBERO-Mem-Derived Benchmark Suite Runner (v1.0)
# Track: CUSTOM_V2_LIBERO_MEM_DERIVED / NON-COMPARABLE_OFFICIAL_PAPER
# Specification: docs/LIBERO_MEM_DERIVED_SPEC.md
# ==============================================================================
[CmdletBinding()]
param (
    [string]$States = "all",
    [string[]]$Tasks = @("T1", "T3"),
    [string[]]$Conditions = @("B0", "F1"),
    [int]$PolicySeed = 42,
    [string]$OutputDirBase = "experiments/evaluation/libero_mem_derived",
    [string]$TargetDir = "",
    [string]$SimPython = "/mnt/d/Thesis_26/.venv_libero_reference/bin/python",
    [string]$DetectorPython = "D:\Thesis_26\.venv_detector\Scripts\python.exe",
    [string]$AdapterDir = "experiments/training/results_mem_2/mem-adapter-20261105-083543/pretrained_model",
    [string]$BaseCheckpoint = "resources/checkpoints/smolvla_libero",
    [string]$SplitLock = "experiments/training/results_mem_2/mem-adapter-v1/dataset-split.lock.json",
    [switch]$SkipB0,
    [switch]$SkipF1
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  LIBERO-Mem-Derived Benchmark Evaluation Runner                " -ForegroundColor Cyan
Write-Host "  Track: CUSTOM_V2_LIBERO_MEM_DERIVED (Locked Split, 20 States) " -ForegroundColor Cyan
Write-Host "  Conditions: $($Conditions -join ', ') | Tasks: $($Tasks -join ', ') | States: $States" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $TargetDir = "${OutputDirBase}_${Timestamp}"
}

# Ensure output directory exists
if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}

$ResultsDirs = @()

# ------------------------------------------------------------------------------
# 1. Run Baseline B0 (Frozen Base Policy without Memory on Held-Out Test Split)
# ------------------------------------------------------------------------------
if (-not $SkipB0 -and ($Conditions -contains "B0")) {
    $B0Dir = Join-Path $TargetDir "eval_B0"
    Write-Host "`n>>> [1/2] Starting B0 Evaluation (Base SmolVLA) -> $B0Dir" -ForegroundColor Green

    python scripts/evaluate_mem_adapter.py `
        --split test `
        --tasks $Tasks `
        --conditions B0 `
        --states $States `
        --policy-seed $PolicySeed `
        --base-checkpoint $BaseCheckpoint `
        --split-lock $SplitLock `
        --sim-python $SimPython `
        --use-wsl `
        --output-dir $B0Dir

    if ($LASTEXITCODE -ne 0) {
        Write-Error "B0 evaluation failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
    $ResultsDirs += $B0Dir
} elseif ($SkipB0) {
    Write-Host "`n>>> [1/2] Skipping B0 as requested (-SkipB0)." -ForegroundColor Yellow
    $ExistingB0 = Join-Path $TargetDir "eval_B0"
    if (Test-Path $ExistingB0) { $ResultsDirs += $ExistingB0 }
}

# ------------------------------------------------------------------------------
# 2. Run Condition F1 (LoRA Step 3000 + Causal Memory on Held-Out Test Split)
# ------------------------------------------------------------------------------
if (-not $SkipF1 -and ($Conditions -contains "F1")) {
    $F1Dir = Join-Path $TargetDir "eval_F1"
    Write-Host "`n>>> [2/2] Starting F1 Evaluation (LoRA + Causal Memory) -> $F1Dir" -ForegroundColor Green

    python scripts/evaluate_mem_adapter.py `
        --split test `
        --tasks $Tasks `
        --conditions F1 `
        --states $States `
        --policy-seed $PolicySeed `
        --base-checkpoint $BaseCheckpoint `
        --adapter-dir $AdapterDir `
        --split-lock $SplitLock `
        --sim-python $SimPython `
        --detector-python $DetectorPython `
        --use-wsl `
        --output-dir $F1Dir

    if ($LASTEXITCODE -ne 0) {
        Write-Error "F1 evaluation failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
    $ResultsDirs += $F1Dir
} elseif ($SkipF1) {
    Write-Host "`n>>> [2/2] Skipping F1 as requested (-SkipF1)." -ForegroundColor Yellow
    $ExistingF1 = Join-Path $TargetDir "eval_F1"
    if (Test-Path $ExistingF1) { $ResultsDirs += $ExistingF1 }
}

# ------------------------------------------------------------------------------
# 3. Consolidate Evaluation Results
# ------------------------------------------------------------------------------
if ($ResultsDirs.Count -gt 0) {
    Write-Host "`n>>> Generating Consolidated Benchmark Report..." -ForegroundColor Cyan
    $ReportPath = Join-Path $TargetDir "benchmark_consolidated_report.md"
    $CsvPath = Join-Path $TargetDir "benchmark_consolidated_summary.csv"

    python scripts/summarize_mem_evaluation.py `
        $ResultsDirs `
        --title "LIBERO-Mem-Derived Benchmark Evaluation Report" `
        --output-report $ReportPath `
        --output-csv $CsvPath

    Write-Host "`nEvaluation complete! Consolidated report saved to:" -ForegroundColor Green
    Write-Host "  Report: $ReportPath" -ForegroundColor Yellow
    Write-Host "  CSV:    $CsvPath" -ForegroundColor Yellow
}

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "  LIBERO-Mem-Derived Evaluation Complete!                       " -ForegroundColor Green
Write-Host "  Results Root: $TargetDir" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan
