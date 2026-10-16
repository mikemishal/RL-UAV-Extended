param([switch]$Install)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if ($Install) {
    if (-not (Test-Path -LiteralPath $python)) {
        python -m venv (Join-Path $PSScriptRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the visualizer virtual environment.' }
    }
    & $python -m pip install -r (Join-Path $PSScriptRoot 'backend\requirements-dev.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed.' }
}
if (-not (Test-Path -LiteralPath $python)) { throw 'Run .\start-backend.ps1 -Install first.' }
Push-Location $repo
try {
    & $python -m uvicorn visualizer.backend.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir (Join-Path $PSScriptRoot 'backend')
    if ($LASTEXITCODE -ne 0) { throw 'Backend server failed.' }
} finally {
    Pop-Location
}
