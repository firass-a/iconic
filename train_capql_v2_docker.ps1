# Train CAPQL v2 in Docker — saves actor.pt + critic.pt to your PC at:
#   iconic\models\capql_v2_l7afla\
#
# Does NOT touch models\capql_v2\ (interface still uses those old-obs weights).
#
# Prerequisites:
#   1. Docker Desktop running
#   2. Image built:  .\build_docker.ps1  (from interface folder)
#
# Usage:
#   .\train_capql_v2_docker.ps1
#   .\train_capql_v2_docker.ps1 -Image "gym-dssat:debian-bookworm"
#   .\train_capql_v2_docker.ps1 -ModelDir "models\capql_v2_custom"
param(
    [string]$Image = $env:GYM_DSSAT_IMAGE,
    [string]$ModelDir = "models\capql_v2_l7afla"
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Image) { $Image = "gym-dssat:debian-bookworm" }

$outDir = Join-Path $root $ModelDir
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$modelDirUnix = ($ModelDir -replace '\\', '/')

try {
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "docker not running" }
} catch {
    Write-Host "ERROR: Start Docker Desktop first." -ForegroundColor Red
    exit 1
}

docker image inspect $Image 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Image '$Image' not found. Run:  interface\build_docker.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "=== CAPQL v2 training (new obs layout) ===" -ForegroundColor Cyan
Write-Host "Mount:       $root -> /workspace"
Write-Host "Save dir:    $outDir"
Write-Host "  actor.pt   (new weights, farmer-realistic obs)"
Write-Host "  critic.pt"
Write-Host "Untouched:   $root\models\capql_v2\  (interface demo)"
Write-Host "Log:         $root\train_capql_v2.log"
Write-Host "Steps:       1,000,000  (~several hours)"
Write-Host ""

$volume = "${root}:/workspace"
$bashCmd = @"
export PATH="/opt/gym_dssat_pdi/bin:/opt/dssat_pdi:`$PATH"
export CAPQL_TRAIN_MODEL_DIR="/workspace/$modelDirUnix"
cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
mkdir -p "/workspace/$modelDirUnix"
python3 -u 07_capql_train_v2.py 2>&1 | tee /workspace/train_capql_v2.log
"@ -replace "`r`n", "`n"

docker run --rm -it `
    -v "$volume" `
    -w /workspace/gym-dssat-pdi/gym_dssat_pdi_samples `
    -e "CAPQL_TRAIN_MODEL_DIR=/workspace/$modelDirUnix" `
    --entrypoint /bin/bash `
    $Image `
    -lc $bashCmd

Write-Host ""
$actor = Join-Path $outDir "actor.pt"
$critic = Join-Path $outDir "critic.pt"
if ((Test-Path $actor) -and (Test-Path $critic)) {
    Write-Host "SUCCESS: weights saved to $outDir" -ForegroundColor Green
} elseif (Test-Path $actor) {
    Write-Host "Partial: actor.pt found but critic.pt missing — check train_capql_v2.log" -ForegroundColor Yellow
} else {
    Write-Host "Training ended but weights not found — check train_capql_v2.log" -ForegroundColor Yellow
}
