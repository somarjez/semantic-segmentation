$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $projectRoot 'run.ps1') -Check

if ($LASTEXITCODE -ne 0) {
    throw "PowerShell launcher check failed with exit code $LASTEXITCODE."
}
