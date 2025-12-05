# PowerShell script to start Redis server for ABC-Application
# Run this script to start Redis before running the main application

Write-Host "Starting Redis server for ABC-Application..." -ForegroundColor Green

# Check if Redis is already running
$redisProcess = Get-Process redis-server -ErrorAction SilentlyContinue
if ($redisProcess) {
    Write-Host "Redis is already running (PID: $($redisProcess.Id))" -ForegroundColor Yellow
    exit 0
}

# Start Redis server
$redisPath = Join-Path $PSScriptRoot "..\redis\redis-server.exe"
$configPath = Join-Path $PSScriptRoot "..\redis\minimal.conf"

if (Test-Path $redisPath) {
    Write-Host "Starting Redis server..." -ForegroundColor Cyan
    Start-Process -FilePath $redisPath -ArgumentList $configPath -NoNewWindow

    # Wait a moment for Redis to start
    Start-Sleep -Seconds 2

    # Check if it started successfully
    $redisProcess = Get-Process redis-server -ErrorAction SilentlyContinue
    if ($redisProcess) {
        Write-Host "✅ Redis server started successfully (PID: $($redisProcess.Id))" -ForegroundColor Green
        Write-Host "Redis is now available at localhost:6379" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to start Redis server" -ForegroundColor Red
        Write-Host "Please check that Redis is properly installed" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Redis server not found at $redisPath" -ForegroundColor Red
    Write-Host "Please ensure Redis is installed in the redis/ directory" -ForegroundColor Red
}