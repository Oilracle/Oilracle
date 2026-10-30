param([Parameter(Mandatory=$true)][int]$RunnerId)
$ErrorActionPreference = 'Stop'
$runnerProcess = Get-Process -Id $RunnerId
$runnerStarted = $runnerProcess.StartTime
$allowedMask = $runnerProcess.ProcessorAffinity.ToInt64()
$availableCores = @(0..([Environment]::ProcessorCount - 1) | Where-Object { $allowedMask -band (1L -shl $_) })
$assigned = @{}
while ($true) {
    $currentRunner = Get-Process -Id $RunnerId -ErrorAction SilentlyContinue
    if (-not $currentRunner -or $currentRunner.StartTime -ne $runnerStarted) { break }
    $processes = @(Get-CimInstance Win32_Process)
    $descendants = [System.Collections.Generic.HashSet[int]]::new()
    [void]$descendants.Add($RunnerId)
    do {
        $previousCount = $descendants.Count
        foreach ($item in $processes) {
            if ($descendants.Contains([int]$item.ParentProcessId)) { [void]$descendants.Add([int]$item.ProcessId) }
        }
    } while ($descendants.Count -ne $previousCount)
    $workers = @($processes | Where-Object {
        $descendants.Contains([int]$_.ProcessId) -and $_.CommandLine -match 'multiprocessing.spawn'
    } | Sort-Object ProcessId)
    $liveIds = @($workers | ForEach-Object { [int]$_.ProcessId })
    foreach ($oldId in @($assigned.Keys)) {
        if ($oldId -notin $liveIds) { $assigned.Remove($oldId) }
    }
    foreach ($item in $workers) {
        $workerId = [int]$item.ProcessId
        if ($assigned.ContainsKey($workerId)) { continue }
        $freeCores = @($availableCores | Where-Object { $_ -notin @($assigned.Values) })
        if (-not $freeCores.Count) { continue }
        try {
            $workerProcess = Get-Process -Id $workerId
            $workerProcess.ProcessorAffinity = [IntPtr](1L -shl $freeCores[0])
            $assigned[$workerId] = $freeCores[0]
            Write-Output "$(Get-Date -Format o) worker=$workerId core=$($freeCores[0])"
        } catch {
            # A completed worker may exit between the inventory and assignment.
            if (Get-Process -Id $workerId -ErrorAction SilentlyContinue) { throw }
        }
    }
    Start-Sleep -Seconds 3
}
Write-Output "$(Get-Date -Format o) runner=$RunnerId exited; affinity helper stopped"
