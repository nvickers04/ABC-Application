# TigerBeetle Setup Script for Windows
# Downloads and sets up TigerBeetle for development

param(
    [string]$Version = "0.15.3"
)

Write-Host "Setting up TigerBeetle v$Version for Windows..." -ForegroundColor Green

# Create data directory
$dataDir = Join-Path $PSScriptRoot "..\tigerbeetle-data"
if (!(Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}

# Download TigerBeetle binary
$url = "https://github.com/tigerbeetle/tigerbeetle/releases/download/$Version/tigerbeetle-x86_64-windows.zip"
$zipPath = Join-Path $PSScriptRoot "tigerbeetle.zip"

Write-Host "Downloading TigerBeetle from $url..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $url -OutFile $zipPath
} catch {
    Write-Error "Failed to download TigerBeetle: $_"
    exit 1
}

# Extract binary
Write-Host "Extracting TigerBeetle..." -ForegroundColor Yellow
Expand-Archive -Path $zipPath -DestinationPath $PSScriptRoot -Force

# Clean up zip
Remove-Item $zipPath

# Initialize data file
$tigerbeetleExe = Join-Path $PSScriptRoot "tigerbeetle.exe"
$dataFile = Join-Path $dataDir "tigerbeetle-data-file"

Write-Host "Initializing TigerBeetle data file..." -ForegroundColor Yellow
& $tigerbeetleExe format --cluster=0 --replica=0 --replica-count=1 $dataFile

Write-Host "TigerBeetle setup complete!" -ForegroundColor Green
Write-Host "To start TigerBeetle server:" -ForegroundColor Cyan
Write-Host "  & '$tigerbeetleExe' start --addresses=3000 $dataFile" -ForegroundColor White