# Build gym-dssat:torch — bakes CPU torch + gymnasium on top of gym-dssat:bookworm.
# Run once. Subsequent training/eval skip pip entirely (instant startup).

$env:PATH = "C:\Program Files\Docker\Docker\resources\bin;C:\Program Files\Docker\Docker\resources;$env:PATH"

Write-Host "Building gym-dssat:torch (this downloads ~200MB once)..."
docker build -f Dockerfile.torch -t gym-dssat:torch .

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Done. Image gym-dssat:torch built."
    Write-Host "Training/eval scripts will now use it automatically and skip pip install."
} else {
    Write-Host "Build failed (exit $LASTEXITCODE)."
}
