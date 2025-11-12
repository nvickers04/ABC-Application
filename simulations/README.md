# Simulations Folder

This folder contains all simulation and backtesting scripts for the ABC Application system.

## Simulation Scripts

### Historical Simulations
- `comprehensive_historical_simulation.py` - Full historical portfolio simulation using yfinance data
- `comprehensive_ibkr_simulation.py` - Professional-grade simulation using IBKR historical data
- `historical_agent_backtesting.py` - Complete agent-based historical backtesting

## Usage

### Run Historical Simulation (yfinance data)
```bash
python simulations/comprehensive_historical_simulation.py
```

### Run IBKR-Powered Simulation (Professional data)
```bash
python simulations/comprehensive_ibkr_simulation.py
```

### Run Agent Backtesting
```bash
python simulations/historical_agent_backtesting.py
```

## Features

- Multi-asset portfolio backtesting
- Rebalancing strategies (daily/weekly/monthly/quarterly)
- Transaction costs and slippage modeling
- Risk metrics (Sharpe ratio, max drawdown, VaR)
- Performance analytics and reporting
- Agent orchestration validation