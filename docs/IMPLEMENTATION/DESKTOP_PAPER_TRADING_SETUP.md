# Desktop Paper Trading Setup Guide

## Overview
This guide covers setting up the ABC Application for paper trading on a desktop environment. Paper trading allows you to test trading strategies with simulated money without financial risk.

## Prerequisites
- Windows 10/11 with administrative privileges
- Python 3.11+ installed
- Git for version control
- Internet connection for market data

## IBKR Paper Trading Setup

### 1. Create IBKR Paper Trading Account
1. Visit https://www.interactivebrokers.com/
2. Create a free account if you don't have one
3. Enable paper trading in your account settings
4. Note your paper trading account ID (starts with "DU" followed by numbers)

### 2. Download and Install IBKR Trader Workstation (TWS)
1. Download TWS from: https://www.interactivebrokers.com/en/index.php?f=16042
2. Choose the Windows installer
3. Run the installer with administrative privileges
4. Complete the installation wizard

### 3. Configure TWS for Paper Trading
1. Launch TWS from the desktop shortcut
2. When prompted, select "Paper Trading" mode
3. Log in with your IBKR paper trading credentials
4. Go to File → Global Configuration → API
5. Check "Enable ActiveX and Socket Clients"
6. Set "Socket port" to 7497 (paper trading default)
7. Check "Allow connections from localhost only" (recommended for security)
8. Optionally check "Create API message log file" for debugging
9. Click "Apply" then "OK"
10. Restart TWS for changes to take effect

### 4. Verify TWS Connection
1. In TWS, check the bottom right corner - you should see "Connected" status
2. The paper trading account should show $1,000,000 virtual balance
3. Test basic functionality by requesting market data for a symbol like AAPL

## ABC Application Desktop Setup

### 1. Clone and Setup Repository
```bash
# Clone the repository
git clone https://github.com/nvickers04/ABC-Application.git
cd ABC-Application

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:

```bash
# IBKR Paper Trading Configuration
IBKR_USERNAME=your_ibkr_username
IBKR_PASSWORD=your_ibkr_password
IBKR_ACCOUNT_ID=DUF123456  # Your paper trading account ID
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# API Keys (get from respective services)
GROK_API_KEY=your_grok_api_key
OPENAI_API_KEY=your_openai_api_key
NEWSAPI_KEY=your_newsapi_key
FRED_API_KEY=your_fred_api_key

# Redis (optional, for caching)
REDIS_HOST=127.0.0.1
REDIS_PORT=6380

# Logging
LOG_LEVEL=INFO
LOG_FILE=data/logs/abc_application.log
```

### 3. Initialize Required Services

#### Start Redis (Optional but Recommended)
```bash
# Download Redis for Windows from:
# https://github.com/microsoftarchive/redis/releases
# Extract and run:
redis-server.exe --port 6380
```

#### Alternative: Use Redis Portable
The application includes a portable Redis version:
```bash
# From project root
data\redis-portable\redis-server.exe --port 6380
```

### 4. Verify Installation
```bash
# Run dependency check
python check_deps.py

# Test health endpoints
python -c "import asyncio; from health_server import app; print('Health server imports OK')"

# Test IBKR connection (requires TWS running)
python -c "
import asyncio
from src.integrations.ibkr_connector import get_ibkr_connector

async def test_connection():
    connector = get_ibkr_connector()
    result = await connector.connect()
    print(f'IBKR Connection: {result}')

asyncio.run(test_connection())
"
```

## Running Paper Trading

### 1. Start Health Monitoring (Optional)
```bash
# In one terminal
python health_server.py --host 127.0.0.1 --port 8080
```

### 2. Start ABC Application
```bash
# In another terminal
python src/main.py
```

### 3. Monitor Trading Activity
- Check logs in `data/logs/`
- Monitor health at http://localhost:8080/health
- View positions and orders through TWS
- Check Discord notifications if configured

## Troubleshooting

### TWS Connection Issues
**Problem:** "Connection refused" errors
**Solution:**
1. Ensure TWS is running and logged in
2. Verify API settings in TWS (File → Global Configuration → API)
3. Check that port 7497 is not blocked by firewall
4. Try restarting TWS
5. Verify IBKR credentials in .env file

### Python Import Errors
**Problem:** Module not found errors
**Solution:**
1. Ensure virtual environment is activated: `myenv\Scripts\activate`
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python path: `python -c "import sys; print(sys.path)"`

### Redis Connection Issues
**Problem:** Redis not available warnings
**Solution:**
1. Start Redis server on port 6380
2. Or disable Redis features (application will work without caching)
3. Check Redis logs for errors

### Memory Issues
**Problem:** High memory usage or crashes
**Solution:**
1. Close other applications to free memory
2. Reduce concurrent operations in configuration
3. Monitor memory usage via health endpoints
4. The application is optimized for 8GB+ RAM systems

## Performance Optimization

### For Desktop Environments
- **RAM:** 16GB recommended for full feature set
- **CPU:** Multi-core processor for parallel processing
- **Storage:** 10GB free space for logs and data
- **Network:** Stable internet for market data feeds

### Configuration Tuning
```yaml
# In config/risk-constraints.yaml
max_concurrent_operations: 3  # Reduce for lower-end systems
memory_cache_size: 512MB      # Adjust based on available RAM
api_timeout_seconds: 30       # Increase for slower connections
```

## Security Considerations

### Desktop Security
- Keep IBKR credentials secure and never commit to version control
- Use environment variables for sensitive configuration
- Regularly update dependencies for security patches
- Consider using a dedicated user account for trading

### Network Security
- TWS API connections are localhost-only by default
- External API calls use HTTPS encryption
- Sensitive data is encrypted in transit and at rest where possible

## Backup and Recovery

### Data Backup
- Logs: `data/logs/` - Contains trading history and decisions
- Memory: `data/memory/` - Agent learning and state
- Configuration: `config/` - Custom settings and constraints

### Recovery Procedures
1. **Application Crash:** Restart with `python src/main.py`
2. **TWS Disconnect:** Reconnect automatically (circuit breaker pattern)
3. **Network Issues:** Automatic retry with exponential backoff
4. **Data Loss:** Restore from backups or reinitialize

## Monitoring and Maintenance

### Daily Checks
- Verify TWS connection status
- Check health endpoints: http://localhost:8080/health
- Review logs for errors or warnings
- Monitor account balance in TWS

### Weekly Maintenance
- Update dependencies: `pip install -r requirements.txt --upgrade`
- Clear old logs if disk space is limited
- Review trading performance and adjust strategies
- Backup important data directories

### Monthly Maintenance
- Full system restart to clear memory
- Review and update API keys if needed
- Analyze trading performance metrics
- Update IBKR TWS to latest version

## Next Steps

Once paper trading is stable:
1. Monitor performance for 2-4 weeks
2. Validate all integration points work correctly
3. Test error scenarios and recovery mechanisms
4. Consider moving to live trading with small position sizes

## Support

For issues:
1. Check the troubleshooting section above
2. Review logs in `data/logs/`
3. Test individual components with the integration tests
4. Check IBKR TWS logs for API-related issues

Remember: Paper trading is for testing and validation. Always start live trading with small position sizes and proper risk management.