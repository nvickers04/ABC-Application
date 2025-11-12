# Tools Folder

This folder contains operational tools and utilities for running and monitoring the ABC Application system.

## Operational Tools

### Trading Tools
- `continuous_trading.py` - Automated continuous paper trading system
- `start_continuous_trading.bat` - Windows launcher for continuous trading
- `CONTINUOUS_TRADING_README.md` - Detailed continuous trading documentation

### Monitoring Tools
- `api_health_dashboard.py` - Real-time API health monitoring dashboard

## Usage

### Start Continuous Trading
```bash
# Using the batch file (recommended)
tools/start_continuous_trading.bat

# Or directly with Python
python tools/continuous_trading.py
```

### Monitor API Health
```bash
python tools/api_health_dashboard.py
```

## Features

### Continuous Trading
- Runs during market hours (9:30 AM - 4:00 PM ET)
- Multi-agent orchestration (Strategy → Risk → Execution)
- Real-time position monitoring
- Automatic status reporting
- Graceful shutdown handling

### API Health Monitoring
- Real-time health status for all APIs
- Circuit breaker integration
- Response time tracking
- Success rate monitoring
- Alert system for degraded services