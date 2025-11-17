# ABC Application Deployment Guide - IBKR Paper Trading

## Overview
This guide covers deploying the ABC Application with Interactive Brokers (IBKR) paper trading on Vultr VPS.

## Prerequisites
- Vultr VPS with Ubuntu 22.04+ (recommended: 4GB RAM, 2 vCPUs)
- IBKR Paper Trading Account (free to create at interactivebrokers.com)
- API keys for LLM services (Grok, OpenAI, etc.)

## IBKR Paper Trading Setup

### 1. Create IBKR Paper Trading Account
1. Visit https://www.interactivebrokers.com/
2. Create a free account
3. Enable paper trading in your account settings
4. Note your paper trading account ID (starts with "DU" followed by numbers)

### 2. Download IBKR Trader Workstation (TWS)
The deployment script will automatically download and install TWS, but you can also do this manually:
```bash
wget https://download2.interactivebrokers.com/installers/tws/latest/tws-latest-linux-x64.sh
chmod +x tws-latest-linux-x64.sh
./tws-latest-linux-x64.sh
```

### 3. Configure TWS for Paper Trading
1. Launch TWS: `xvfb-run -a /opt/ibkr/tws &`
2. Log in with your IBKR paper trading credentials
3. Go to File → Global Configuration → API
4. Enable "Enable ActiveX and Socket Clients"
5. Set "Socket port" to 7497
6. Enable "Create API message log file"
7. Check "Allow connections from localhost only" (for security)
8. Save configuration

## Deployment Steps

### 1. Run the Deployment Script
```bash
# Upload deploy-vultr.sh to your VPS
chmod +x deploy-vultr.sh
sudo ./deploy-vultr.sh
```

### 2. Configure Environment Variables
Create `/opt/abc-application/.env` with your credentials:
```bash
cp .env.template .env
nano .env  # Edit with your actual credentials
```

Required IBKR settings:
```
IBKR_USERNAME=your_ibkr_username
IBKR_PASSWORD=your_ibkr_password
IBKR_ACCOUNT_ID=DUF123456  # Your paper trading account ID
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
```

### 3. Start Services
```bash
# Start IBKR TWS first
sudo systemctl start ibkr-tws

# Wait for TWS to fully load, then start ABC Application
sudo systemctl start abc-application
```

### 4. Verify Installation
```bash
# Check service status
sudo systemctl status abc-application
sudo systemctl status ibkr-tws

# Check logs
journalctl -u abc-application -f
journalctl -u ibkr-tws -f

# Test health endpoint
curl http://localhost:8000/health

# Test IBKR connection
cd /opt/abc-application
python -c "import asyncio; from integrations.ibkr_connector import test_connection; asyncio.run(test_connection())"
```

## IBKR Paper Trading Features

### What is Paper Trading?
- **Virtual Money**: Trade with simulated funds ($1,000,000 default)
- **Real Market Data**: Uses live market prices and conditions
- **Realistic Execution**: Simulates actual trading costs and slippage
- **Risk-Free Testing**: Perfect for strategy development and testing

### Supported Trading Features
- ✅ Market orders (MKT)
- ✅ Limit orders (LMT)
- ✅ Stop orders (STP)
- ✅ Bracket orders (entry + stop loss + take profit)
- ✅ Real-time position monitoring
- ✅ Portfolio P&L tracking
- ✅ Account summary and cash balance
- ✅ Order status and execution tracking

### Risk Management
- Position size limits (configurable)
- Daily loss limits
- Maximum drawdown protection
- Diversification checks
- Real-time risk monitoring

## Troubleshooting

### IBKR Connection Issues
```bash
# Check if TWS is running
ps aux | grep tws

# Check TWS logs
journalctl -u ibkr-tws -f

# Restart TWS
sudo systemctl restart ibkr-tws

# Test connection manually
cd /opt/abc-application
python -c "import asyncio; from integrations.ibkr_connector import test_connection; asyncio.run(test_connection())"
```

### Common Issues
1. **"IBKR credentials not available"**
   - Check .env file has correct IBKR_USERNAME and IBKR_PASSWORD
   - Ensure IBKR account is paper trading enabled

2. **"Connection timeout"**
   - Verify TWS is running: `sudo systemctl status ibkr-tws`
   - Check API settings in TWS (port 7497, API enabled)
   - Restart TWS if needed

3. **"Market is closed"**
   - Paper trading works 24/5 during market hours
   - Check current time and market status

### Log Locations
- ABC Application: `journalctl -u abc-application -f`
- IBKR TWS: `journalctl -u ibkr-tws -f`
- Application logs: `/opt/abc-application/logs/`

## Security Notes
- Paper trading uses virtual funds only
- All API connections are secured
- Credentials are stored in environment variables
- Firewall restricts access to necessary ports only

## Next Steps
1. **Test Trading**: Place small paper trades to verify functionality
2. **Monitor Performance**: Use the dashboard to track paper trading results
3. **Strategy Development**: Refine strategies using paper trading feedback
4. **Live Trading**: Once confident, switch to live IBKR account (port 7496)

## Support
- Check logs for detailed error messages
- Verify all environment variables are set correctly
- Ensure IBKR TWS is properly configured for API access
- Test IBKR connection independently before running full application</content>
<parameter name="filePath">c:\Users\nvick\ABC-Application\IBKR_PAPER_TRADING_DEPLOYMENT.md