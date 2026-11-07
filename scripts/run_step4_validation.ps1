# ==============================================================================
# Step 4: Validation Paired Pilot (40 Episodes)
# Protocol: CUSTOM_V2_LIBERO_MEM / NON-COMPARABLE_OFFICIAL_PAPER
# Specification: docs/RESULTS_MEM_2_TEST_PLAN.md Section P2
# ==============================================================================
[CmdletBinding()]
param (
    [string]$States = "all",
    [string[]]$Tasks = @("T1", "T3"),
    [string[]]$Conditions = @("B0", "F1"),
    [int]$PolicySeed = 42,
    [string]$OutputDirBase = "experiments/evaluation/results_mem_2/step4_validation",
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
Write-Host "  Step 4: LIBERO-Mem Validation Evaluation (40 Episodes Target)  " -ForegroundColor Cyan
Write-Host "  Conditions: $($Conditions -join ', ') | Tasks: $($Tasks -join ', ') | States: $States" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$TargetDir = "${OutputDirBase}_${Timestamp}"

# Ensure output directory base exists
if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}

$TaskArgs = $Tasks -join " "
$ResultsDirs = @()

# ------------------------------------------------------------------------------
# 1. Run Baseline B0 (Frozen Base Policy without Memory)
# ------------------------------------------------------------------------------
if (-not $SkipB0 -and ($Conditions -contains "B0")) {
    $B0Dir = Join-Path $TargetDir "val_B0"
    Write-Host "`n>>> [1/2] Starting B0 Evaluation (Base Policy) -> $B0Dir" -ForegroundColor Green
    
    python scripts/evaluate_mem_adapter.py `
        --split val `
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
        Write-Error "B0 evaluation failed with exit code $LASTEXITCODE"
    }
    $ResultsDirs += $B0Dir
}

# ------------------------------------------------------------------------------
# 2. Run Candidate F1 (LoRA Step 3000 Adapter + Predicted RGB Memory)
# ------------------------------------------------------------------------------
if (-not $SkipF1 -and ($Conditions -contains "F1")) {
    $F1Dir = Join-Path $TargetDir "val_F1"
    Write-Host "`n>>> [2/2] Starting F1 Evaluation (LoRA Step 3000 + Memory) -> $F1Dir" -ForegroundColor Green
    
    python scripts/evaluate_mem_adapter.py `
        --split val `
        --tasks $Tasks `
        --conditions F1 `
        --states $States `
        --policy-seed $PolicySeed `
        --adapter-dir $AdapterDir `
        --base-checkpoint $BaseCheckpoint `
        --split-lock $SplitLock `
        --sim-python $SimPython `
        --use-wsl `
        --detector-python $DetectorPython `
        --output-dir $F1Dir
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "F1 evaluation failed with exit code $LASTEXITCODE"
    }
    $ResultsDirs += $F1Dir
}

# ------------------------------------------------------------------------------
# 3. Consolidate and Generate Paired Report
# ------------------------------------------------------------------------------
if ($ResultsDirs.Count -gt 0) {
    Write-Host "`n>>> Generating Consolidated Validation Report..." -ForegroundColor Cyan
    $ReportPath = Join-Path $TargetDir "validation_consolidated_report.md"
    $CsvPath = Join-Path $TargetDir "validation_consolidated_summary.csv"
    
    python scripts/summarize_mem_evaluation.py `
        $ResultsDirs `
        --title "Step 4: LIBERO-Mem Validation Paired Pilot Report" `
        --output-report $ReportPath `
        --output-csv $CsvPath
    
    Write-Host "`nValidation complete! Consolidated report saved to:" -ForegroundColor Green
    Write-Host "  Report: $ReportPath" -ForegroundColor Yellow
    Write-Host "  CSV:    $CsvPath" -ForegroundColor Yellow
}
