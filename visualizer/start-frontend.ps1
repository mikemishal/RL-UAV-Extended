param([switch]$Install)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot 'frontend')
try {
    if ($Install) {
        npm ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    }
    npm run dev
    if ($LASTEXITCODE -ne 0) { throw 'Frontend server failed.' }
} finally {
    Pop-Location
}
