<#
.SYNOPSIS
    Shortcut dong bo ban thao khoa luan sang output/overleaf_upload tu goc repository.
#>

[CmdletBinding()]
param(
    [switch]$Open = $false,
    [switch]$Zip = $false
)

$scriptDir = $PSScriptRoot
$targetPs1 = Join-Path $scriptDir "..\docs\thesis\latex\sync_overleaf.ps1"

$params = @{}
if ($Open) { $params["Open"] = $true }
if ($Zip) { $params["Zip"] = $true }

& $targetPs1 @params
