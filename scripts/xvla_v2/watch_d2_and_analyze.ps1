$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location $repo
$eventRun = Join-Path $repo 'experiments/evaluation/xvla_d2_event_dev12_t1_t3_20261123_wsl'
$eventManifest = Join-Path $eventRun 'run_manifest.json'
$logPath = Join-Path $repo 'experiments/evaluation/xvla_d3_d5_d2_trace_analysis_20261123/analysis_chain.log'
$d3Output = Join-Path $repo 'experiments/evaluation/xvla_d3_d5_d2_trace_analysis_20261123_results'
$e3Output = Join-Path $repo 'experiments/evaluation/xvla_e3_token_budget_d2_20261123'
$python = (Get-Command python).Source
$e3Python = Join-Path $repo '.venv_memory/Scripts/python.exe'

function Write-ChainLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null
Write-ChainLog 'Watcher started; waiting for completed X-VLA D2 event cohort.'

while ($true) {
    if (Test-Path -LiteralPath $eventManifest) {
        try {
            $manifest = Get-Content -LiteralPath $eventManifest -Raw | ConvertFrom-Json
        } catch {
            Write-ChainLog "Manifest is not readable yet: $($_.Exception.Message)"
            Start-Sleep -Seconds 30
            continue
        }
        if ($manifest.status -eq 'complete') { break }
        if ($manifest.status -in @('failed', 'error', 'aborted')) {
            Write-ChainLog "Event run terminated with status $($manifest.status); analysis not launched."
            exit 2
        }
    }
    Start-Sleep -Seconds 30
}

if ($manifest.completed_episodes -ne 24) {
    Write-ChainLog "Event run says complete but has $($manifest.completed_episodes) episodes; analysis not launched."
    exit 3
}

Write-ChainLog 'Event cohort complete; launching paired D3/D5 trace analysis.'
& $python (Join-Path $repo 'scripts/analyze_xvla_d3_d5_d2.py') --output-dir $d3Output *>> $logPath
if ($LASTEXITCODE -ne 0) {
    Write-ChainLog "D3/D5 analysis failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-ChainLog 'D3/D5 analysis complete; launching E3 token-budget analysis on D2 event traces.'
& $e3Python (Join-Path $repo 'scripts/analyze_xvla_e3_token_budget.py') `
    --source-run 'experiments/evaluation/xvla_d2_event_dev12_t1_t3_20261123_wsl' `
    --output-dir 'experiments/evaluation/xvla_e3_token_budget_d2_20261123' *>> $logPath
if ($LASTEXITCODE -ne 0) {
    Write-ChainLog "E3 analysis failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-ChainLog 'D3/D5 and E3 analyses completed.'
