<#
.SYNOPSIS
    Shortcut de dong goi ban thao khoa luan cho Overleaf tu thu muc goc.
#>

[CmdletBinding()]
param(
    [switch]$Open = $false,
    [switch]$NoZip = $false
)

$scriptDir = $PSScriptRoot
$targetPs1 = Join-Path $scriptDir "..\docs\thesis\latex\package_overleaf.ps1"

$params = @{}
if ($Open) { $params["Open"] = $true }
if ($NoZip) { $params["NoZip"] = $true }

& $targetPs1 @params
