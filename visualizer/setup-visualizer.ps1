param(
    [string]$SimulationPython,
    [switch]$InstallApiDependencies,
    [switch]$InstallFrontendDependencies
)
$ErrorActionPreference = 'Stop'
$visualizer = $PSScriptRoot
$repo = Split-Path -Parent $visualizer
$envFile = Join-Path $visualizer '.env'
$dataRoot = (Resolve-Path -LiteralPath (Join-Path $repo '..\RL-UAV-Extended-lrppo-clutter') -ErrorAction SilentlyContinue).Path
if (-not $dataRoot) {
    throw "Research worktree not found beside the visualizer. Set VISUALIZER_RESEARCH_DATA_ROOT manually in $envFile."
}

if (-not (Test-Path -LiteralPath $envFile)) {
    if (-not $SimulationPython) {
        $SimulationPython = Read-Host 'Enter the absolute path to the existing research Python executable (the one with Gymnasium, SB3, and PyTorch)'
    }
    if (-not [IO.Path]::IsPathRooted($SimulationPython) -or -not (Test-Path -LiteralPath $SimulationPython -PathType Leaf)) {
        throw "SimulationPython must be the absolute path to an existing executable. No environment was created."
    }
    $probe = & $SimulationPython -c 'import sys, gymnasium, stable_baselines3, torch; print(sys.executable)' 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "The selected Python is missing required scientific packages. No packages were installed. Probe: $($probe -join ' ')"
    }
    @(
        '# Local paths only. The research worktree is read-only to this visualizer.'
        "VISUALIZER_RESEARCH_DATA_ROOT=$dataRoot"
        "VISUALIZER_SIMULATION_PYTHON=$([IO.Path]::GetFullPath($SimulationPython))"
        'VISUALIZER_EPISODE_TIMEOUT_SECONDS=900'
        'VISUALIZER_FRONTEND_ORIGIN=http://127.0.0.1:5173'
    ) | Set-Content -LiteralPath $envFile -Encoding utf8
    Write-Host "Created local configuration: $envFile"
} else {
    Write-Host "Preserved existing configuration: $envFile"
}

if ($InstallApiDependencies) {
    & (Join-Path $visualizer 'start-backend.ps1') -Install
}
if ($InstallFrontendDependencies) {
    & (Join-Path $visualizer 'start-frontend.ps1') -Install
}
Write-Host 'Setup complete. Review visualizer\.env, then run .\visualizer\start-visualizer.ps1 from the repository root.'
