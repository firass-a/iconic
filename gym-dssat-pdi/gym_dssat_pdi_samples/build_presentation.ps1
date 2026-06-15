# Compile thesis_presentation.tex -> PDF using TeX Live in Docker.
# First run downloads the texlive image (~1.5 GB, one-time).
# Subsequent runs take ~30 s.

$ErrorActionPreference = "Stop"
$env:PATH = "C:\Program Files\Docker\Docker\resources\bin;C:\Program Files\Docker\Docker\resources;$env:PATH"

$Samples = $PSScriptRoot
$TexFile = "thesis_presentation.tex"

if (-not (Test-Path (Join-Path $Samples $TexFile))) {
    Write-Host "ERROR: $TexFile not found in $Samples"
    exit 1
}

Write-Host "Checking Docker..."
docker ps | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running. Open Docker Desktop first."
    exit 1
}

# Use the lightweight texlive variant from texlive/texlive (~1.5 GB)
# 'small' has beamer, tikz, amsmath - everything we need.
$Image = "texlive/texlive:latest-small"

Write-Host "Using image: $Image"
Write-Host ""
Write-Host "Building thesis_presentation.pdf (pdflatex x2 for cross-refs)..."

docker run --rm `
  -v "${Samples}:/work" `
  -w /work `
  $Image `
  bash -c "pdflatex -interaction=nonstopmode $TexFile && pdflatex -interaction=nonstopmode $TexFile"

$exit = $LASTEXITCODE
$pdfPath = Join-Path $Samples ($TexFile -replace '\.tex$', '.pdf')

if ($exit -eq 0 -and (Test-Path $pdfPath)) {
    $kb = [math]::Round((Get-Item $pdfPath).Length / 1KB, 1)
    Write-Host ""
    Write-Host "Done. PDF: $pdfPath  ($kb KB)"
    Write-Host ""
    $open = Read-Host "Open the PDF now? [Y/n]"
    if ($open -ne 'n' -and $open -ne 'N') {
        Start-Process $pdfPath
    }
} else {
    Write-Host ""
    Write-Host "Build failed (exit $exit). Check the .log file for details:"
    Write-Host "  $($TexFile -replace '\.tex$', '.log')"
}
