<#
.SYNOPSIS
    Goi script bien dich khoa luan LaTeX tu thu muc goc cua repository.
.EXAMPLE
    .\render_thesis.ps1
    .\render_thesis.ps1 -Full
    .\render_thesis.ps1 -Full -Open
#>
[CmdletBinding()]
param(
    [switch]$Full = $false,
    [switch]$Clean = $false,
    [switch]$Open = $false,
    [switch]$SyncOutput = $true
)

$targetScript = Join-Path $PSScriptRoot "docs\thesis\latex\render_pdf.ps1"
& $targetScript -Full:$Full -Clean:$Clean -Open:$Open -SyncOutput:$SyncOutput
