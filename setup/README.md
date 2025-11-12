# Setup Folder

This folder contains installation files and setup utilities for the ABC Application system dependencies.

## Installation Files

### Redis
- `redis.msi` - Windows installer for Redis server
- `redis.zip` - Alternative Redis installation package

## Setup Instructions

### Installing Redis

#### Option 1: MSI Installer (Recommended)
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