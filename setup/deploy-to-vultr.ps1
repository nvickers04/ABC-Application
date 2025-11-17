# [LABEL:DEPLOY:vultr] [LABEL:SCRIPT:powershell] [LABEL:INFRA:vps]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Automated deployment script for ABC Application to Vultr VPS
# Dependencies: SSH, SCP, PowerShell 5.1+
# Related: setup/deploy-vultr.sh, docs/IMPLEMENTATION/deployment.md
#
# ABC Application Deployment to Vultr VPS
# Run this script from your local machine to deploy to Vultr

param(
    [Parameter(Mandatory=$true)]
    [string]$VultrIP,

    [string]$SSHKeyPath = "ABCSSH",

    [string]$Username = "linuxuser"
)

Write-Host "Starting ABC Application deployment to Vultr VPS ($VultrIP)..." -ForegroundColor Green

# Test SSH connection first
Write-Host "Testing SSH connection..." -ForegroundColor Yellow
try {
    $testResult = & ssh -i $SSHKeyPath -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$Username@$VultrIP" echo "SSH connection successful" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "SSH connection failed. Please check your SSH key and IP address." -ForegroundColor Red
        Write-Host "Error: $testResult" -ForegroundColor Red
        exit 1
    }
    Write-Host "SSH connection successful" -ForegroundColor Green
} catch {
    Write-Host "SSH connection failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Create application directory
Write-Host "Creating application directory..." -ForegroundColor Yellow
try {
    $mkdirResult = & ssh -i $SSHKeyPath -o StrictHostKeyChecking=no "$Username@$VultrIP" "sudo mkdir -p /opt/abc-application && sudo chown $Username /opt/abc-application" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create directory: $mkdirResult" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Failed to create directory: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Upload essential files
Write-Host "Uploading environment file..." -ForegroundColor Yellow
try {
    $scpResult = & scp -i $SSHKeyPath -o StrictHostKeyChecking=no ".env" "$Username@$VultrIP`:/opt/abc-application/.env" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to upload .env: $scpResult" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Failed to upload .env: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "Uploading requirements..." -ForegroundColor Yellow
try {
    $scpResult = & scp -i $SSHKeyPath -o StrictHostKeyChecking=no "requirements.txt" "$Username@$VultrIP`:/opt/abc-application/requirements.txt" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to upload requirements.txt: $scpResult" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Failed to upload requirements.txt: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "Uploading deployment script..." -ForegroundColor Yellow
try {
    $scpResult = & scp -i $SSHKeyPath -o StrictHostKeyChecking=no "deploy-vultr.sh" "$Username@$VultrIP`:/opt/abc-application/deploy-vultr.sh" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to upload deploy-vultr.sh: $scpResult" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Failed to upload deploy-vultr.sh: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Upload source code
Write-Host "Uploading source code..." -ForegroundColor Yellow
try {
    $scpResult = & scp -i $SSHKeyPath -o StrictHostKeyChecking=no -r "src" "$Username@$VultrIP`:/opt/abc-application/" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to upload source code: $scpResult" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Failed to upload source code: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Upload configuration
Write-Host "Uploading configuration..." -ForegroundColor Yellow
try {
    $scpResult = & scp -i $SSHKeyPath -o StrictHostKeyChecking=no -r "config" "$Username@$VultrIP`:/opt/abc-application/" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to upload configuration: $scpResult" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Failed to upload configuration: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Make deployment script executable and run it
Write-Host "Making deployment script executable..." -ForegroundColor Yellow
try {
    $chmodResult = & ssh -i $SSHKeyPath -o StrictHostKeyChecking=no "$Username@$VultrIP" "chmod +x /opt/abc-application/deploy-vultr.sh" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to make script executable: $chmodResult" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Failed to make script executable: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "Running deployment script..." -ForegroundColor Yellow
try {
    $deployResult = & ssh -i $SSHKeyPath -o StrictHostKeyChecking=no "$Username@$VultrIP" "cd /opt/abc-application && sudo ./deploy-vultr.sh" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Deployment failed: $deployResult" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Deployment failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Check if service is running
Write-Host "Checking deployment status..." -ForegroundColor Yellow
try {
    $statusResult = & ssh -i $SSHKeyPath -o StrictHostKeyChecking=no "$Username@$VultrIP" "sudo systemctl status abc-application --no-pager -l" 2>&1
    Write-Host "Service Status:" -ForegroundColor Cyan
    Write-Host $statusResult -ForegroundColor White
} catch {
    Write-Host "Could not check service status: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "Deployment completed!" -ForegroundColor Green
Write-Host "Your ABC Application should be running on: http://$VultrIP:8000" -ForegroundColor Cyan
Write-Host "Monitor logs with: ssh -i $SSHKeyPath $Username@$VultrIP 'sudo journalctl -u abc-application -f'" -ForegroundColor Cyan
