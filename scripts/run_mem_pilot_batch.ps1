[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PolicyPython,
    [Parameter(Mandatory)][string]$SimPython,
    [Parameter(Mandatory)][string]$DetectorPython,
    [Parameter(Mandatory)][string]$F0,
    [Parameter(Mandatory)][string]$F1,
    [Parameter(Mandatory)][string]$Output,
    [ValidateSet('T1','T3')][string[]]$Tasks = @('T1','T3'),
    [ValidateSet('F0','F1')][string[]]$Conditions = @('F0','F1'),
    [string]$States = '0',
    [int]$Seed = 42,
    [string]$ParentLock = 'resources/manifests/libero_mem_t1_t3_base10_split_v2.json',
    [ValidateSet('seen','unseen','val','test')][string]$Split = 'seen',
    [string]$Distro = 'Ubuntu-24.04',
    [switch]$RuntimeCheck
)

$ErrorActionPreference = 'Stop'
if ($States -notmatch '^\d+(,\d+)*$' -or ($States -split ',').Count -gt 2) {
    throw 'Use one or two explicit state indices per development batch'
}
if ($RuntimeCheck -and ($States -ne '0' -or $Conditions.Count -ne 2 -or $Split -eq 'unseen')) {
    throw 'Use state 0 and both arms for the matched runtime check'
}
$cli = Join-Path $PSScriptRoot 'evaluate_mem_adapter.py'
$cliArgs = @($cli, '--use-wsl', '--wsl-distro', $Distro, '--sim-python', $SimPython,
    '--detector-python', $DetectorPython, '--split-lock', $ParentLock, '--split', $Split,
    '--states', $States, '--policy-seed', [string]$Seed,
    '--f0-adapter-dir', $F0, '--adapter-dir', $F1,
    '--no-save-video', '--stream-memory-trace', '--output-dir', $Output)
$cliArgs += @('--tasks') + $Tasks + @('--conditions') + $Conditions
if ($RuntimeCheck) { $cliArgs += '--runtime-check' }
& $PolicyPython @cliArgs
if ($LASTEXITCODE -ne 0) { throw "Batch failed. Inspect $Output/run_manifest.json" }
