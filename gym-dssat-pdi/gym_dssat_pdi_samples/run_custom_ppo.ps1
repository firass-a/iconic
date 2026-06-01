# Run custom PC-PPO training in Docker (verbose logging)
#
# Presets (pick ONE — default is STRONG):
#   .\run_custom_ppo.ps1 -Quick           # smoke test   ~2k steps,   ~1 min
#   .\run_custom_ppo.ps1                    # STRONG       500k steps,  ~60 min
#   .\run_custom_ppo.ps1 -Thesis            # THESIS       1M steps,    ~2 hours
#   .\run_custom_ppo.ps1 -Timesteps 200000  # custom
#
#   .\watch_training.ps1                  # 2nd terminal — live log tail
#   .\stop_training.ps1                   # stop if needed

param(
    [switch]$Quick,
    [switch]$Thesis,
    [int]$Timesteps = 0,
    [int]$Steps = 0
)

$env:PATH = "C:\Program Files\Docker\Docker\resources\bin;C:\Program Files\Docker\Docker\resources;$env:PATH"

Write-Host "Checking Docker..."
docker ps | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running. Open Docker Desktop first."
    exit 1
}

$Samples = $PSScriptRoot
$Models  = Join-Path $Samples "models"
$Logs    = Join-Path $Samples "logs"
New-Item -ItemType Directory -Force -Path $Models, $Logs | Out-Null

# Training presets
if ($Quick) {
    $Preset     = "QUICK"
    $Timesteps  = 2048
    $Steps      = 256
    $LogEvery   = 1
    $CkptEvery  = 0
    $EntCoef    = "0.02"
} elseif ($Thesis) {
    $Preset     = "THESIS"
    $Timesteps  = 1000000
    $Steps      = 2048
    $LogEvery   = 10
    $CkptEvery  = 100
    $EntCoef    = "0.015"
} else {
    $Preset     = "STRONG"
    if ($Timesteps -le 0) { $Timesteps = 500000 }
    if ($Steps -le 0)     { $Steps = 1024 }
    $LogEvery   = 5
    $CkptEvery  = 50
    $EntCoef    = "0.02"
}

if ($Timesteps -gt 0 -and ($Quick -or $Thesis)) {
    # preset already set timesteps
} elseif ($Timesteps -gt 0 -and $Steps -le 0) {
    $Steps = 1024
}

$Updates = [math]::Floor($Timesteps / $Steps)
$EstMin  = [math]::Round($Timesteps / 140.0 / 60.0, 0)

Write-Host ""
Write-Host "=== $Preset training ==="
Write-Host "  timesteps     : $($Timesteps.ToString('N0'))"
Write-Host "  n_steps       : $Steps  ($Updates updates)"
Write-Host "  log_every     : $LogEvery updates"
Write-Host "  checkpoint    : every $CkptEvery updates"
Write-Host "  est. runtime  : ~$EstMin min  (at ~140 fps)"
Write-Host ""

$LogName = "train_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$LogHost = Join-Path $Logs $LogName
$LogDocker = "/logs/$LogName"
$Started = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

@"
==========================================================================================
TRAINING LOG — started $Started
Preset: $Preset  |  Timesteps: $Timesteps  |  n_steps: $Steps  |  updates: $Updates
log_every: $LogEvery  |  checkpoint_every: $CkptEvery  |  est_runtime_min: ~$EstMin
Host path: $LogHost
==========================================================================================
"@ | Set-Content -Path $LogHost -Encoding utf8

Write-Host "Model  -> $Models\pc_ppo_custom.pt"
Write-Host "Log    -> $LogHost"
Write-Host ""
Write-Host "REAL-TIME LOG (open a 2nd PowerShell window):"
Write-Host "  cd `"$Samples`""
Write-Host "  .\watch_training.ps1 -LogFile `"logs\$LogName`""
Write-Host ""
Write-Host "Press Ctrl+C to stop training, then run: .\stop_training.ps1"
Write-Host ""

$innerCmd = "export PYTHONUNBUFFERED=1; echo '[setup] installing torch + gymnasium...' | tee -a '$LogDocker'; pip install -q --no-warn-script-location torch gymnasium; echo '[setup] starting 05_pc_ppo_custom_train.py' | tee -a '$LogDocker'; python3 -u 05_pc_ppo_custom_train.py"

docker run --rm `
  -v "${Samples}:/work" `
  -v "${Models}:/models" `
  -v "${Logs}:/logs" `
  -w /work `
  -e PYTHONUNBUFFERED=1 `
  -e TOTAL_TIMESTEPS=$Timesteps `
  -e N_STEPS=$Steps `
  -e MODEL_PATH=/models/pc_ppo_custom.pt `
  -e LOG_FILE=$LogDocker `
  -e ENT_COEF=$EntCoef `
  -e LOG_EVERY=$LogEvery `
  -e CHECKPOINT_EVERY=$CkptEvery `
  --entrypoint bash `
  gym-dssat:bookworm `
  -lc $innerCmd

$exitCode = $LASTEXITCODE
$Finished = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

@"
==========================================================================================
TRAINING FINISHED - $Finished  (exit code: $exitCode)
Log saved: $LogHost
==========================================================================================
"@ | Add-Content -Path $LogHost -Encoding utf8

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "Done. Full log: $LogHost"
} else {
    Write-Host "Training stopped or failed (exit $exitCode). Log: $LogHost"
}
