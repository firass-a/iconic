# Generate all PC-PPO thesis figures.
#   1. Record one season's daily actions (in Docker)
#   2. Generate 5 PNGs from training log + eval CSV + daily CSV
#
#   Output: figures/*.png

$ErrorActionPreference = "Stop"
$env:PATH = "C:\Program Files\Docker\Docker\resources\bin;C:\Program Files\Docker\Docker\resources;$env:PATH"

$Samples = $PSScriptRoot
$Models  = Join-Path $Samples "models"
$Figures = Join-Path $Samples "figures"
$PyCache = Join-Path $Samples ".cache\pylibs"
New-Item -ItemType Directory -Force -Path $Figures | Out-Null
New-Item -ItemType Directory -Force -Path $PyCache | Out-Null

Write-Host "Checking Docker..."
docker ps | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running. Open Docker Desktop first."
    exit 1
}

$Image = if (docker image inspect gym-dssat:torch 2>$null) { "gym-dssat:torch" } else { "gym-dssat:bookworm" }
Write-Host "Using image: $Image"
Write-Host ""

$innerCmd = @"
set -e
sed -i 's/\r$//' /work/_record_season.py /work/plot_results.py

echo '[1/2] Recording one season for action profile...'
if ! python3 -c 'import torch, gymnasium' 2>/dev/null; then
  echo '[setup] installing torch + gymnasium (CPU)...'
  pip install -q --default-timeout=300 --no-warn-script-location --index-url https://download.pytorch.org/whl/cpu torch
  pip install -q --default-timeout=300 --no-warn-script-location gymnasium
fi
python3 -u /work/_record_season.py

echo ''
echo '[2/2] Generating plots...'
export PYTHONPATH=/pylibs:`$PYTHONPATH
if ! python3 -c 'import pandas, seaborn, mpltern' 2>/dev/null; then
  echo '[setup] installing pandas + seaborn + mpltern into /pylibs cache (one-time, ~45s)...'
  pip install -q --default-timeout=300 --no-warn-script-location --target /pylibs pandas seaborn mpltern
fi
python3 -u /work/plot_results.py
"@

# strip CRLFs so bash doesn't choke on \r
$innerCmd = $innerCmd -replace "`r", ""

docker run --rm `
  -v "${Samples}:/work" `
  -v "${Models}:/models" `
  -v "${Figures}:/figures" `
  -v "${PyCache}:/pylibs" `
  -w /work `
  -e PYTHONUNBUFFERED=1 `
  --entrypoint bash `
  $Image `
  -lc $innerCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Done. Figures:"
    Get-ChildItem $Figures -Filter "*.png" | Format-Table Name, @{N='KB';E={[math]::Round($_.Length/1KB,1)}} -AutoSize
    Write-Host "Open the folder: $Figures"
} else {
    Write-Host "Plot generation failed (exit $LASTEXITCODE)."
}
