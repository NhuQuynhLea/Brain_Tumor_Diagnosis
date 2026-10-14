$folder   = "D:\Project\Brain_Tumor_Diagnosis\LaTeX2e+Proceedings+Template+ZIP"
$target   = Join-Path $folder "main.tex"
$tectonic = "$env:LOCALAPPDATA\tectonic\tectonic.exe"

if (-not (Test-Path $tectonic)) {
    Write-Error "tectonic.exe not found at $tectonic"
    exit 1
}

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $folder
$watcher.Filter = "*.tex"
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]::LastWrite -bor [System.IO.NotifyFilters]::FileName
$watcher.EnableRaisingEvents = $true

Write-Host "Watching $target and sections\*.tex" -ForegroundColor Cyan
Write-Host "Rebuilds main.pdf on every save. Close this window (or Ctrl+C) to stop.`n"

Push-Location $folder
try {
    while ($true) {
        $change = $watcher.WaitForChanged(
            [System.IO.WatcherChangeTypes]::Changed -bor
            [System.IO.WatcherChangeTypes]::Created -bor
            [System.IO.WatcherChangeTypes]::Renamed, 1000)
        if ($change.TimedOut) { continue }

        # debounce: editors often fire several writes per save
        Start-Sleep -Milliseconds 400
        while (-not $watcher.WaitForChanged(
                [System.IO.WatcherChangeTypes]::Changed -bor
                [System.IO.WatcherChangeTypes]::Created -bor
                [System.IO.WatcherChangeTypes]::Renamed, 150).TimedOut) { }

        Write-Host "$(Get-Date -Format 'HH:mm:ss') $($change.Name) changed -> rebuilding..." -ForegroundColor Yellow
        & $tectonic main.tex
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$(Get-Date -Format 'HH:mm:ss') main.pdf rebuilt`n" -ForegroundColor Green
        } else {
            Write-Host "$(Get-Date -Format 'HH:mm:ss') build finished with errors (see above)`n" -ForegroundColor Red
        }
    }
} finally {
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Pop-Location
}
