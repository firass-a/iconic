# Strong PC-PPO evaluation in Docker (preference grid + baselines + CSV)
#
# Usage:
#   .\run_eval.ps1                         # best model, 20 eps/pref
#   .\run_eval.ps1 -Episodes 30            # more episodes
#   .\run_eval.ps1 -Model models\pc_ppo_custom.pt
#   .\run_custom_ppo.ps1 -Thesis            # train first (~2h), then run_eval.ps1

param(
    [string]$Model = "pc_ppo_custom_best.pt",
    [int]$Episodes = 20,
    [int]$RandomPrefs = 20
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
$EvalOut = Join-Path $Models "eval"
New-Item -ItemType Directory -Force -Path $Models, $Logs, $EvalOut | Out-Null

# Patched env_config.yml on host (adds wtnup/cleach/cnox/trnu/tleachd for eval reporting).
# We bind-mount it over the copy baked inside the gym-dssat:torch image.
$PatchedConfig = Join-Path (Split-Path $Samples -Parent) "gym_dssat_pdi\envs\configs\maize\env_config.yml"
$ContainerConfig = "/opt/gym_dssat_pdi/lib/python3.11/site-packages/gym_dssat_pdi/envs/configs/maize/env_config.yml"

$ModelHost = Join-Path $Models $Model
if (-not (Test-Path $ModelHost)) {
    Write-Host "ERROR: model not found: $ModelHost"
    Write-Host "Train first: .\run_custom_ppo.ps1 -Thesis"
    exit 1
}

$GridSize = 7 + $RandomPrefs
$TotalEps = $GridSize * $Episodes + 3 * $Episodes
$EstMin   = [math]::Round($TotalEps * 6.0 / 60.0, 0)

$LogName = "eval_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$LogHost = Join-Path $Logs $LogName
$LogDocker = "/logs/$LogName"

Write-Host ""
Write-Host "=== STRONG EVAL ==="
Write-Host "  model           : $ModelHost"
Write-Host "  episodes/pref   : $Episodes"
Write-Host "  preference grid : $GridSize points"
Write-Host "  total episodes  : $TotalEps"
Write-Host "  est. runtime    : ~$EstMin min"
Write-Host "  output          : $EvalOut\"
Write-Host ""

# Fix CRLF from Windows editor, then run entry script
$innerCmd = "sed -i 's/\r$//' /work/docker_entry_eval.sh && bash /work/docker_entry_eval.sh"

docker run --rm `
  -v "${Samples}:/work" `
  -v "${Models}:/models" `
  -v "${Logs}:/logs" `
  -v "${PatchedConfig}:${ContainerConfig}:ro" `
  -w /work `
  -e PYTHONUNBUFFERED=1 `
  -e MODEL_PATH="/models/$Model" `
  -e EPISODES_PER_PREF=$Episodes `
  -e N_RANDOM_PREFS=$RandomPrefs `
  -e OUTPUT_DIR=/models/eval `
  -e LOG_FILE=$LogDocker `
  --entrypoint bash `
  $(if (docker image inspect gym-dssat:torch 2>$null) { "gym-dssat:torch" } else { "gym-dssat:bookworm" }) `
  -lc $innerCmd

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "Done. Results in: $EvalOut\"
    Write-Host "Log: $LogHost"
} else {
    Write-Host "Eval failed (exit $LASTEXITCODE). Log: $LogHost"
}
