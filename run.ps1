param(
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Check) {
    & $venvPython -c "import flask, torch, torchvision, cv2; print('Environment ready.')"
    exit $LASTEXITCODE
}

& $venvPython app.py
exit $LASTEXITCODE
