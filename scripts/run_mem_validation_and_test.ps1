# Custom V2 Mem evaluation. Native policy/detector; optionally WSL simulator.
param (
    [Parameter(Position=0)][ValidateSet('preflight','smoke','validation','test','all')][string]$Stage = 'smoke',
    [Parameter(Position=1)][string]$SimPython = $env:SIM_PYTHON,
    [string]$DetectorPython = $env:DETECTOR_PYTHON,
    [string]$PolicyPython = 'python',
    [string]$AdapterDir = 'experiments/training/results_mem_2/mem-adapter-20261105-083543/pretrained_model',
    [string]$BaseCheckpoint = 'resources/checkpoints/smolvla_libero',
    [string]$SplitLock = 'experiments/training/results_mem_2/mem-adapter-v1/dataset-split.lock.json',
    [switch]$UseWsl,
    [string]$WslDistro = 'Ubuntu-24.04',
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $SimPython -or -not $DetectorPython) {
    throw 'Specify -SimPython and -DetectorPython for the isolated environments.'
}
$Stages = if ($Stage -eq 'all') { @('smoke', 'validation') } else { @($Stage) }
foreach ($CurrentStage in $Stages) {
    $SelectedSplit = if ($CurrentStage -eq 'test') { 'test' } else { 'val' }
    $States = if ($CurrentStage -in @('validation','test')) { 'all' } else { '0' }
    $OutputDir = 'experiments/evaluation/results_mem_2/{0}_{1}_{2}' -f $CurrentStage, (Get-Date -Format 'yyyyMMdd_HHmmss_fff'), $PID
    $LogDir = 'experiments/evaluation/results_mem_2/launcher_logs'
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    $ConsoleLog = Join-Path $LogDir ((Split-Path $OutputDir -Leaf) + '.log')
    Write-Host "Console log: $ConsoleLog"
    $CliArgs = @('scripts/evaluate_mem_adapter.py', '--adapter-dir', $AdapterDir,
        '--base-checkpoint', $BaseCheckpoint, '--split-lock', $SplitLock,
        '--split', $SelectedSplit, '--states', $States, '--tasks', 'T1', 'T3',
        '--conditions', 'B0', 'F1', '--policy-seed', '42', '--output-dir', $OutputDir,
        '--sim-python', $SimPython, '--detector-python', $DetectorPython)
    if ($UseWsl) { $CliArgs += @('--use-wsl', '--wsl-distro', $WslDistro) }
    if ($CurrentStage -eq 'preflight') { $CliArgs += '--preflight-only' }
    $CliArgs += $ExtraArgs
    & $PolicyPython -u @CliArgs 2>&1 | Tee-Object -FilePath $ConsoleLog
    $RunStatus = $LASTEXITCODE
    Add-Content -Path $ConsoleLog -Value "Launcher exit code: $RunStatus"
    if ($RunStatus -ne 0) { throw "Mem $CurrentStage failed (exit $RunStatus). Inspect $ConsoleLog and run_manifest.json." }
    Write-Host "$CurrentStage completed; inspect the selected output directory."
}
if ($Stage -eq 'all') { Write-Host 'Select the candidate on validation before requesting stage test.' }
