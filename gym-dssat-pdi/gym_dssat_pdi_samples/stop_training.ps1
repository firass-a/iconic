# Stop any running gym-DSSAT training container
$env:PATH = "C:\Program Files\Docker\Docker\resources\bin;$env:PATH"

$ids = docker ps -q
if ($ids) {
    Write-Host "Stopping containers: $ids"
    docker stop $ids
    Write-Host "Done."
} else {
    Write-Host "No running containers."
}
