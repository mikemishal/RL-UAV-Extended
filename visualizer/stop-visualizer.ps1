$ErrorActionPreference = 'Stop'
$visualizer = $PSScriptRoot
$recordFile = Join-Path $visualizer 'data\local\visualizer-processes.json'
if (-not (Test-Path -LiteralPath $recordFile -PathType Leaf)) {
    Write-Host 'No Copilot/application process ownership record exists; no processes were stopped.'
    return
}
$records = @(Get-Content -LiteralPath $recordFile -Raw | ConvertFrom-Json)
$ownedById = @{}
foreach ($record in $records) {
    $ownedById[[int]$record.processId] = $record
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($record.processId)" -ErrorAction SilentlyContinue
    if ($process) {
        $pathMatches = $process.ExecutablePath -and ([IO.Path]::GetFullPath($process.ExecutablePath) -eq [IO.Path]::GetFullPath($record.executable))
        $timeMatches = $false
        try {
            $start = (Get-Process -Id $record.processId -ErrorAction Stop).StartTime.ToUniversalTime()
            $recordedStart = [DateTime]$record.startedUtc
            $recordedUtc = if ($recordedStart.Kind -eq [DateTimeKind]::Utc) { $recordedStart }
                elseif ($recordedStart.Kind -eq [DateTimeKind]::Local) { $recordedStart.ToUniversalTime() }
                else { [DateTime]::SpecifyKind($recordedStart, [DateTimeKind]::Utc) }
            $timeMatches = [Math]::Abs(($start - $recordedUtc).TotalSeconds) -lt 5
        } catch {}
        if (-not $pathMatches -or -not $timeMatches) {
            throw "Recorded PID $($record.processId) no longer matches the owned process identity. It was not terminated; inspect manually."
        }
    }
}

# Ask the owned API to cancel active workers before shutting it down.
$backend = $records | Where-Object role -eq 'backend' | Select-Object -First 1
if ($backend -and (Get-Process -Id $backend.processId -ErrorAction SilentlyContinue)) {
    $jobsRoot = Join-Path $visualizer 'data\local\jobs'
    if (Test-Path -LiteralPath $jobsRoot) {
        foreach ($file in Get-ChildItem -LiteralPath $jobsRoot -Filter '*.json' -File) {
            try {
                $job = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
                if ($job.state -in @('queued', 'running')) {
                    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/replay-jobs/$($job.job_id)/cancel" -TimeoutSec 5 | Out-Null
                }
            } catch { Write-Warning "Could not request cancellation for job metadata $($file.Name): $($_.Exception.Message)" }
        }
    }
    $cancelDeadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        $activeJobs = @()
        if (Test-Path -LiteralPath $jobsRoot) {
            foreach ($file in Get-ChildItem -LiteralPath $jobsRoot -Filter '*.json' -File) {
                try {
                    $job = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
                    if ($job.state -in @('queued', 'running')) { $activeJobs += $job }
                } catch { Write-Warning "Could not inspect job metadata $($file.Name): $($_.Exception.Message)" }
            }
        }
        if ($activeJobs.Count -eq 0) { break }
        Start-Sleep -Seconds 1
    } while ([DateTime]::UtcNow -lt $cancelDeadline)
    if ($activeJobs.Count -gt 0) { Write-Warning 'Replay workers did not report a terminal state within 30 seconds; stopping only the verified owned process tree.' }
}

$all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
$ownedTree = [Collections.Generic.List[int]]::new()
foreach ($rootId in $ownedById.Keys) {
    $pending = [Collections.Generic.Queue[int]]::new()
    $pending.Enqueue([int]$rootId)
    while ($pending.Count -gt 0) {
        $parentId = $pending.Dequeue()
        foreach ($child in $all | Where-Object { $_.ParentProcessId -eq $parentId }) {
            $commandLine = [string]$child.CommandLine
            if ($commandLine -and $commandLine.Contains((Split-Path -Leaf $visualizer))) {
                $ownedTree.Add([int]$child.ProcessId)
                $pending.Enqueue([int]$child.ProcessId)
            }
        }
    }
}
foreach ($id in @($ownedTree.ToArray() | Select-Object -Unique | Sort-Object -Descending)) {
    if (Get-Process -Id $id -ErrorAction SilentlyContinue) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
}
foreach ($record in $records) {
    $id = [int]$record.processId
    if (Get-Process -Id $id -ErrorAction SilentlyContinue) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
}
Start-Sleep -Milliseconds 800
$remaining = @($records | Where-Object { Get-Process -Id $_.processId -ErrorAction SilentlyContinue })
$remainingChildren = @($ownedTree.ToArray() | Select-Object -Unique | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort 8000,5173 -ErrorAction SilentlyContinue)
if ($remaining.Count -gt 0 -or $remainingChildren.Count -gt 0 -or $listeners.Count -gt 0) {
    $details = @($remaining | ForEach-Object { "$($_.role) PID $($_.processId)" })
    $details += @($remainingChildren | ForEach-Object { "owned child PID $_" })
    $details += @($listeners | ForEach-Object { "listener port $($_.LocalPort) PID $($_.OwningProcess)" })
    throw "Owned application processes or listeners remain: $($details -join ', ')."
}
Remove-Item -LiteralPath $recordFile -Force
Write-Host 'FastAPI and Vite processes recorded by this launcher have stopped. Existing unrelated listeners were not touched.'
