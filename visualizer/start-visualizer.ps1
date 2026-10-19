param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$visualizer = $PSScriptRoot
$repo = Split-Path -Parent $visualizer
$frontend = Join-Path $visualizer 'frontend'
$envFile = Join-Path $visualizer '.env'
$python = Join-Path $visualizer '.venv\Scripts\python.exe'
$recordFile = Join-Path $visualizer 'data\local\visualizer-processes.json'
$logRoot = Join-Path $visualizer 'data\local\logs'
$owned = [Collections.Generic.List[object]]::new()

function Save-ProcessRecord {
    $owned | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $recordFile -Encoding utf8
}
function Stop-OwnedProcessTree([int]$ProcessId) {
    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $descendants = [Collections.Generic.List[int]]::new()
    $pending = [Collections.Generic.Queue[int]]::new()
    $pending.Enqueue($ProcessId)
    while ($pending.Count -gt 0) {
        $parent = $pending.Dequeue()
        foreach ($child in $all | Where-Object { $_.ParentProcessId -eq $parent }) {
            $descendants.Add([int]$child.ProcessId)
            $pending.Enqueue([int]$child.ProcessId)
        }
    }
    foreach ($id in @($descendants.ToArray() | Select-Object -Unique | Sort-Object -Descending)) {
        $process = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($process) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
    }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($process) { Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue }
}

try {
    if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
        throw "Local configuration is missing: $envFile. Run .\visualizer\setup-visualizer.ps1 and provide the existing scientific Python path."
    }
    $configuration = @{}
    foreach ($line in Get-Content -LiteralPath $envFile) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            $configuration[$Matches[1]] = $Matches[2].Trim('"').Trim("'")
        }
    }
    $required = @('VISUALIZER_RESEARCH_DATA_ROOT', 'VISUALIZER_SIMULATION_PYTHON')
    foreach ($key in $required) {
        if (-not $configuration[$key]) { throw "Missing $key in $envFile. Set it explicitly; no interpreter or data root will be substituted." }
    }
    $dataRoot = $configuration['VISUALIZER_RESEARCH_DATA_ROOT']
    if (-not [IO.Path]::IsPathRooted($dataRoot)) { $dataRoot = Join-Path $visualizer $dataRoot }
    $dataRoot = (Resolve-Path -LiteralPath $dataRoot -ErrorAction Stop).Path
    $simPython = [IO.Path]::GetFullPath($configuration['VISUALIZER_SIMULATION_PYTHON'])
    if (-not (Test-Path -LiteralPath $simPython -PathType Leaf)) { throw "Configured simulation Python does not exist: $simPython" }
    if (-not (Test-Path -LiteralPath (Join-Path $dataRoot 'uav_defend\envs\soldier_env.py') -PathType Leaf)) {
        throw "Configured research data root is not a valid research worktree: $dataRoot"
    }
    $registry = Join-Path $dataRoot 'results\defender_adaptation_model_registry.json'
    if (-not (Test-Path -LiteralPath $registry -PathType Leaf)) { throw "Frozen model registry is unavailable: $registry" }
    try { Get-Content -LiteralPath $registry -Raw | ConvertFrom-Json | Out-Null }
    catch { throw "Frozen model registry is not readable valid JSON: $registry. $($_.Exception.Message)" }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Visualizer API environment is missing at $python. Run .\visualizer\setup-visualizer.ps1 -InstallApiDependencies." }
    if (-not (Test-Path -LiteralPath (Join-Path $frontend 'node_modules\vite\bin\vite.js') -PathType Leaf)) {
        throw "Frontend dependencies are missing. Run .\visualizer\setup-visualizer.ps1 -InstallFrontendDependencies."
    }
    $node = (Get-Command node.exe -ErrorAction Stop).Source
    $probe = & $simPython -c 'import sys, gymnasium, stable_baselines3, torch; print(sys.executable)' 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Configured scientific Python cannot import Gymnasium, Stable-Baselines3, and PyTorch: $($probe -join ' ')" }
    $confirmedPython = [IO.Path]::GetFullPath(($probe | Select-Object -Last 1).Trim())
    if ($confirmedPython -ne $simPython) { throw "Configured Python resolved to '$confirmedPython', not '$simPython'. Correct the explicit .env value." }

    $ports = @(8000, 5173)
    foreach ($port in $ports) {
        $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
        if ($listeners.Count -gt 0) {
            $details = $listeners | ForEach-Object {
                $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $($_.OwningProcess)" -ErrorAction SilentlyContinue
                "port $port PID $($_.OwningProcess) $($owner.Name) $($owner.CommandLine)"
            }
            throw "Required localhost port is already occupied; no process was stopped. $($details -join '; ')"
        }
    }

    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $recordFile) | Out-Null
    if (Test-Path -LiteralPath $recordFile) {
        throw "A prior process ownership record exists at $recordFile. Run .\visualizer\stop-visualizer.ps1 and verify shutdown before starting another instance."
    }
    $env:VISUALIZER_RESEARCH_DATA_ROOT = $dataRoot
    $env:VISUALIZER_SIMULATION_PYTHON = $simPython
    if ($configuration['VISUALIZER_EPISODE_TIMEOUT_SECONDS']) { $env:VISUALIZER_EPISODE_TIMEOUT_SECONDS = $configuration['VISUALIZER_EPISODE_TIMEOUT_SECONDS'] }
    if ($configuration['VISUALIZER_FRONTEND_ORIGIN']) { $env:VISUALIZER_FRONTEND_ORIGIN = $configuration['VISUALIZER_FRONTEND_ORIGIN'] }
    $env:PYTHONPATH = $repo

    $backendOut = Join-Path $logRoot 'backend.stdout.log'
    $backendErr = Join-Path $logRoot 'backend.stderr.log'
    $backend = Start-Process -FilePath $python -ArgumentList @('-m', 'uvicorn', 'visualizer.backend.main:app', '--host', '127.0.0.1', '--port', '8000') -WorkingDirectory $repo -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr -WindowStyle Hidden -PassThru
    $owned.Add([pscustomobject]@{ role = 'backend'; processId = $backend.Id; executable = $python; startedUtc = $backend.StartTime.ToUniversalTime().ToString('o') })
    Save-ProcessRecord

    $viteOut = Join-Path $logRoot 'frontend.stdout.log'
    $viteErr = Join-Path $logRoot 'frontend.stderr.log'
    $vite = Start-Process -FilePath $node -ArgumentList @((Join-Path $frontend 'node_modules\vite\bin\vite.js'), '--host', '127.0.0.1', '--port', '5173', '--strictPort') -WorkingDirectory $frontend -RedirectStandardOutput $viteOut -RedirectStandardError $viteErr -WindowStyle Hidden -PassThru
    $owned.Add([pscustomobject]@{ role = 'frontend'; processId = $vite.Id; executable = $node; startedUtc = $vite.StartTime.ToUniversalTime().ToString('o') })
    Save-ProcessRecord

    $deadline = [DateTime]::UtcNow.AddMinutes(5)
    $apiReady = $false
    $webReady = $false
    while ([DateTime]::UtcNow -lt $deadline -and (-not $apiReady -or -not $webReady)) {
        if (-not (Get-Process -Id $backend.Id -ErrorAction SilentlyContinue)) { throw "FastAPI exited during startup. See $backendErr" }
        if (-not (Get-Process -Id $vite.Id -ErrorAction SilentlyContinue)) { throw "Vite exited during startup. See $viteErr" }
        if (-not $apiReady) {
            try {
                $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 3
                if ($health.status -eq 'ok' -and $health.simulation_enabled) {
                    Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/policies' -TimeoutSec 60 | Out-Null
                    Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/scenarios' -TimeoutSec 30 | Out-Null
                    $apiReady = $true
                }
            } catch { Start-Sleep -Seconds 2 }
        }
        if (-not $webReady) {
            try { $null = Invoke-WebRequest -Uri 'http://127.0.0.1:5173' -TimeoutSec 3; $webReady = $true }
            catch { Start-Sleep -Milliseconds 500 }
        }
    }
    if (-not $apiReady) { throw "FastAPI did not become ready with an accessible policy registry. See $backendErr" }
    if (-not $webReady) { throw "Vite did not become ready. See $viteErr" }
    Write-Host "Simulation Viewer is ready: http://127.0.0.1:5173"
    Write-Host "Backend health: http://127.0.0.1:8000/api/health"
    Write-Host "Owned process IDs: FastAPI $($backend.Id), Vite $($vite.Id)"
    Write-Host 'Stop safely with .\visualizer\stop-visualizer.ps1'
    if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:5173' }
} catch {
    foreach ($item in @($owned.ToArray())) { Stop-OwnedProcessTree -ProcessId $item.processId }
    if (Test-Path -LiteralPath $recordFile) { Remove-Item -LiteralPath $recordFile -Force }
    throw
}
