# Start CAPQL API in Docker (DSSAT + actor required)
param(
    [string]$Image = $env:GYM_DSSAT_IMAGE
)

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $Image) {
    $torch = docker image inspect gym-dssat:torch 2>$null
    if ($LASTEXITCODE -eq 0) { $Image = "gym-dssat:torch" }
    else { $Image = "gym-dssat:debian-bookworm" }
}

$actor = Join-Path $root "models\capql_v2\actor.pt"
$weatherCli = Join-Path $root "weather\UFGA.CLI"
$patchedEnvConfig = Join-Path $root "gym-dssat-pdi\gym_dssat_pdi\envs\configs\maize\env_config.yml"
$patchedGymPkg = Join-Path $root "gym-dssat-pdi\gym_dssat_pdi"
$containerEnvConfig = "/opt/gym_dssat_pdi/lib/python3.11/site-packages/gym_dssat_pdi/envs/configs/maize/env_config.yml"
$containerGymPkg = "/opt/gym_dssat_pdi/lib/python3.11/site-packages/gym_dssat_pdi"
if (-not (Test-Path $actor)) {
    Write-Host "ERROR: Missing $actor - train CAPQL v2 first." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $weatherCli)) {
    Write-Host "WARNING: Missing $weatherCli - rainfall may be zero." -ForegroundColor Yellow
    Write-Host "  Generate inside Docker: python3 gym_dssat_pdi_samples/07_weather_setup.py" -ForegroundColor Yellow
}

try {
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "docker not running" }
} catch {
    Write-Host "ERROR: Start Docker Desktop first." -ForegroundColor Red
    exit 1
}

Write-Host "=== Smart Farm API (CAPQL + DSSAT) ===" -ForegroundColor Cyan
Write-Host "Image:  $Image"
Write-Host "API:    http://localhost:8000"
Write-Host "Docs:   http://localhost:8000/docs  (check POST /advisory/session)"
Write-Host "Mount:  $root -> /workspace"
Write-Host ""

# Must use gym_dssat_pdi Python (has torch + gymnasium). System python3 does not.
$bashCmd = @'
export PATH="/opt/gym_dssat_pdi/bin:/opt/dssat_pdi:$PATH"
export GYM_SAMPLES_DIR="/workspace/gym-dssat-pdi/gym_dssat_pdi_samples"
export CAPQL_MODEL_DIR="/workspace/models/capql_v2"
export CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
PY=/opt/gym_dssat_pdi/bin/python3
cd /workspace/interface/api
$PY -m pip install -q fastapi uvicorn pydantic
exec $PY -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /workspace/interface/api --reload-dir /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
'@ -replace "`r`n", "`n"

docker run --rm -it `
    -p 8000:8000 `
    -v "${root}:/workspace" `
    -v "${patchedEnvConfig}:${containerEnvConfig}:ro" `
    -v "${patchedGymPkg}:${containerGymPkg}:ro" `
    -e CAPQL_MODEL_DIR=/workspace/models/capql_v2 `
    -e GYM_SAMPLES_DIR=/workspace/gym-dssat-pdi/gym_dssat_pdi_samples `
    -e CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173 `
    --entrypoint /bin/bash `
    $Image `
    -lc $bashCmd
