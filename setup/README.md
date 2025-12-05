# [LABEL:DOC:setup] [LABEL:DOC:readme] [LABEL:INFRA:deployment]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Documentation for system setup, installation, and deployment
# Dependencies: Python 3.11+, system administration access
# Related: config/.env.template, docs/IMPLEMENTATION/setup-and-development.md
#
# Setup Folder

This folder contains installation files, setup utilities, and deployment scripts for the ABC Application system.

## Installation Files

### Redis
- `redis.msi` - Windows installer for Redis server
- `redis.zip` - Alternative Redis installation package

### Python Package Manager
- `get-pip.py` - Bootstrap installer for pip (Python package manager)

## Deployment Scripts

### Vultr VPS Deployment
- `deploy-to-vultr.ps1` - PowerShell script for deploying to Vultr VPS (run from local machine)
- `deploy-vultr.sh` - Bash script for server-side setup on Vultr VPS

**Deployment Process:**
1. Run `deploy-to-vultr.ps1` from your local machine with the VPS IP
2. The script will upload files and execute `deploy-vultr.sh` on the server
3. Monitor deployment status and logs

## Setup Instructions

### Installing Python pip
If pip is not available on your system:
```bash
python setup/get-pip.py
```

### Installing Redis

#### Option 1: MSI Installer (Recommended for Windows)
1. Run `setup/redis.msi`
2. Follow the installation wizard
3. Start Redis service from Windows Services

#### Option 2: Zip Package
1. Extract `setup/redis.zip`
2. Run `redis-server.exe` from the extracted folder

### Verifying Installation
```bash
redis-cli ping
```
Should respond with `PONG`

## Deployment to Vultr VPS

### Prerequisites
- Vultr VPS with Ubuntu 22.04+
- SSH access configured
- Local machine with PowerShell (Windows) or Bash

### Deployment Steps
1. **Prepare your environment:**
   ```bash
   # Copy environment template
   cp config/.env.template .env
   # Edit .env with your actual API keys
   ```

2. **Run deployment script:**
   ```powershell
   # From PowerShell on Windows
   .\setup\deploy-to-vultr.ps1 -VultrIP "your.vps.ip.address"
   ```

3. **Monitor deployment:**
   The script will show progress and provide connection details when complete.

4. **Access your deployed application:**
   - Web interface: `http://your.vps.ip.address:8000`
   - Logs: `ssh user@your.vps.ip.address 'sudo journalctl -u abc-application -f'`

## Dependencies

The system requires:
- Python 3.11+
- Redis server (for caching and memory management)
- IBKR Trader Workstation or Gateway (for live trading)
- Various Python packages (see requirements.txt)

## Environment Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password
IBKR_ACCOUNT_ID=your_account_id
# Add other API keys as needed
```

3. Install and start Redis server

4. Ensure IBKR TWS/Gateway is running for live trading