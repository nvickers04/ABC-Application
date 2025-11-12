# ABC Application Continuous Trading System

## Quick Start

### Prerequisites
1. **IBKR TWS Running**: Ensure Trader Workstation is running and connected to your paper trading account
2. **API Enabled**: In TWS, go to File → Global Configuration → API and enable API connections
3. **Paper Trading Account**: Make sure you're logged into your IBKR paper trading account

### Running the System

#### Option 1: Windows Batch File (Recommended)
Double-click `start_continuous_trading.bat` in the project root.

#### Option 2: Manual Python Execution
```bash
python continuous_trading.py
```

## What Happens During Execution

1. **Pre-Flight Checks**: Verifies TWS connection and market hours
2. **Agent Initialization**: Loads all trading agents (Strategy, Risk, Execution)
3. **Continuous Loop**: Runs trading cycles every 30 seconds during market hours
4. **Status Reporting**: Logs progress and P&L every 5 minutes
5. **Graceful Shutdown**: Handles Ctrl+C for clean exit

## Monitoring

- **Live Logs**: Watch the console output for real-time status
- **Log File**: Check `continuous_trading.log` for detailed logs
- **TWS**: Monitor trades in your IBKR Trader Workstation

## Safety Features

- **Paper Trading Only**: All trades are paper trades (simulated)
- **Risk Limits**: Position sizing and loss limits enforced
- **Market Hours**: Only trades during regular market hours (9:30 AM - 4:00 PM ET)
- **Circuit Breakers**: Stops trading if major issues detected

## Stopping the System

- Press `Ctrl+C` in the terminal/console
- The system will gracefully shut down all agents and close connections

## Troubleshooting

### TWS Connection Issues
- Ensure TWS is running and API is enabled
- Check that you're using port 7497 (paper trading)
- Verify you're logged into your paper account

### Python Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ recommended)

### Market Hours
- System only runs during NYSE hours (9:30 AM - 4:00 PM ET)
- Outside hours, it will wait until next market open

## Configuration

Edit `continuous_trading.py` to modify:
- Trading cycle interval (default: 30 seconds)
- Status reporting frequency (default: 5 minutes)
- Risk parameters (see config/risk-constraints.yaml)

## Support

For issues, check the logs and ensure all prerequisites are met. The system is designed to be robust and will retry failed operations automatically.