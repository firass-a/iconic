# Tail the latest (or a specific) training log in real time.
#
# Usage:
#   .\watch_training.ps1              # follow latest log, last 25 lines
#   .\watch_training.ps1 -Tail 40
#   .\watch_training.ps1 -LogFile logs\train_20250601_120000.log

param(
    [int]$Tail = 25,
    [string]$LogFile = ''
)

$Logs = Join-Path $PSScriptRoot "logs"

if ($LogFile) {
    if (-not (Test-Path $LogFile)) {
        Write-Host "ERROR: log not found: $LogFile"
        exit 1
    }
    $target = Resolve-Path $LogFile
} else {
    if (-not (Test-Path $Logs)) {
        Write-Host "No logs/ folder yet. Start training first: .\run_custom_ppo.ps1"
        exit 1
    }
    $latest = Get-ChildItem (Join-Path $Logs "*.log") -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latest) {
        Write-Host "No log files in $Logs yet. Start training first: .\run_custom_ppo.ps1"
        exit 1
    }
    $target = $latest.FullName
}

Write-Host "Watching (Ctrl+C to stop):"
Write-Host "  $target"
Write-Host ""

Get-Content -Path $target -Wait -Tail $Tail
