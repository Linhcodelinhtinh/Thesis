# ==============================================================================
# Batch Replay Runner for LIBERO-Mem-Derived Demonstrations (MuJoCo 3.2.3)
# Protocol: CUSTOM_V2_LIBERO_MEM_DERIVED
# ==============================================================================
[CmdletBinding()]
param (
    [string[]]$Tasks = @("T1", "T3"),
    [string]$Split = "train",
    [string]$OutputDir = "data/libero_mem_derived_v1",
    [string]$StagingDir = "/var/tmp/libero_mem_derived_staging_v1",
    [string]$SplitLock = "experiments/training/results_mem_2/mem-adapter-v1/dataset-split.lock.json",
    [string]$SourceDir = "data/libero_mem",
    [switch]$NoResume,
    [int]$MaxEpisodes = 0,
    [switch]$RequireSuccess,
    [string]$WslDistro = "Ubuntu-24.04",
    [string]$SimPython = "/mnt/d/Thesis_26/.venv_libero_reference/bin/python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  LIBERO-Mem-Derived Training Demonstrations Generator           " -ForegroundColor Cyan
Write-Host "  Engine: MuJoCo 3.2.3 / robosuite 1.4.0 (raw RGB, locked provenance)" -ForegroundColor Cyan
Write-Host "  Tasks: $($Tasks -join ', ') | Split: $Split                     " -ForegroundColor Cyan
Write-Host "  Output Directory: $OutputDir                                    " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# Convert Windows paths to WSL paths
$WslOutputDir = "/mnt/" + $OutputDir.Replace(":\", "/").Replace("\", "/")
if (-not $OutputDir.StartsWith("data/")) {
    $WslOutputDir = "/mnt/d/Thesis_26/$OutputDir"
} else {
    $WslOutputDir = "/mnt/d/Thesis_26/$OutputDir"
}

function Convert-RepoPathToWsl([string]$Path) {
    if ($Path.StartsWith("/")) { return $Path }
    return "/mnt/d/Thesis_26/" + $Path.Replace("\", "/").TrimStart("/")
}

$ArgsList = @(
    "/mnt/d/Thesis_26/scripts/generate_derived_demonstrations.py",
    "--tasks", ($Tasks -join " "),
    "--split", $Split,
    "--split-lock", (Convert-RepoPathToWsl $SplitLock),
    "--source-dir", (Convert-RepoPathToWsl $SourceDir),
    "--output-dir", $WslOutputDir,
    "--staging-dir", $StagingDir
)

if ($NoResume) {
    $ArgsList += @("--no-resume")
}
if ($MaxEpisodes -gt 0) {
    $ArgsList += @("--max-episodes", $MaxEpisodes.ToString())
}
if ($RequireSuccess) {
    $ArgsList += @("--require-success")
}

$WslCmd = "$SimPython " + ($ArgsList -join " ")
Write-Host "`n>>> Executing batch replay in WSL ($WslDistro)..." -ForegroundColor Green
Write-Host "Command: $WslCmd" -ForegroundColor DarkGray

wsl -d $WslDistro -- bash -c "cd /mnt/d/Thesis_26 && export MUJOCO_GL=osmesa && $WslCmd"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Batch replay failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "  Batch Replay Complete!                                        " -ForegroundColor Green
Write-Host "  Derived datasets saved to: $OutputDir                         " -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan
