# IBKR Premarket Preparation

This directory contains scripts for performing premarket preparation routines with Interactive Brokers (IBKR) integration.

## Overview

The premarket preparation system connects to your IBKR paper trading account and performs comprehensive morning checks including:

- **Account Status**: Cash balance, buying power, account summary
- **Current Positions**: Holdings, P&L, position details
- **Market Data**: Real-time quotes for key symbols
- **Trading Permissions**: Account capabilities and restrictions
- **News & Bulletins**: Market news and exchange announcements
- **System Health**: Memory backends and API connectivity
- **Trading Recommendations**: AI-powered insights based on premarket data

## Files

### `premarket_prep.py`
**Live IBKR Connection Script**
- Requires IBKR Trader Workstation (TWS) or Gateway running
- Connects to real paper trading account
- Performs actual API calls to IBKR
- Use when TWS is available and you want live data

### `premarket_prep_simulated.py`
**Simulated Environment Script**
- No IBKR connection required
- Uses realistic simulated data
- Perfect for development and testing
- Demonstrates full functionality without TWS

## Prerequisites

### For Live Connection (`premarket_prep.py`)

1. **IBKR Account**: Active paper trading account
2. **TWS/Gateway**: Running on port 7497 (paper trading)
3. **API Access**: Enabled in TWS settings
4. **Credentials**: Set in `.env` file:
   ```
   IBKR_USERNAME=noahvickers
   IBKR_PASSWORD=N0a4v1ckers97$
   IBKR_ACCOUNT_ID=DUN976979
   ```

### For Simulation (`premarket_prep_simulated.py`)

- No prerequisites required
- Works in any environment
- Uses realistic simulated market data

## Usage

### Command Line

```bash
# Simulated (recommended for development)
python premarket_prep_simulated.py

# Live connection (requires TWS)
python premarket_prep.py
```

### VS Code Tasks

1. Open Command Palette (`Ctrl+Shift+P`)
2. Select "Tasks: Run Task"
3. Choose:
   - **"IBKR Premarket Preparation (Simulated)"** - No TWS required
   - **"IBKR Premarket Preparation (Live)"** - Requires TWS connection

## Sample Output

```
================================================================================
📊 SIMULATED IBKR PREMARKET PREPARATION SUMMARY
================================================================================
⏰ Time: 2025-11-20T13:45:35.416588+00:00
🎭 Mode: SIMULATION (No live IBKR connection required)

✅ IBKR Connection: SUCCESSFUL (SIMULATED)

💰 ACCOUNT SUMMARY:
   Cash Balance: $98,543.67
   Positions: 3
   Buying Power: $197,087.34
   Total Value: $152,341.89
   Currency: USD

📊 CURRENT POSITIONS:
   SPY: 50 shares @ $450.25 ($22,512.50) | P&L: +$262.50
   AAPL: 25 shares @ $195.60 ($4,890.00) | P&L: +$127.50
   MSFT: 15 shares @ $425.30 ($6,379.50) | P&L: +$142.50
   Total Value: $33,782.00 | Total P&L: +$532.50

📈 KEY MARKET DATA:
   SPY: $450.25 | QQQ: $380.80 | AAPL: $195.60 | MSFT: $425.30
   GOOGL: $175.90 | AMZN: $185.45 | TSLA: $285.70 | NVDA: $145.20
   META: $485.60 | NFLX: $725.80 | AMD: $165.40 | INTC: $22.15
   BA: $185.90 | DIS: $95.25 | V: $305.80 | JPM: $225.60

🏦 ACCOUNT TYPE: PAPER_TRADING
✅ ENABLED TRADING: equities, options, futures, forex, margin, short_selling

💡 RECOMMENDATIONS:
   • 💰 HEALTHY CASH: Good liquidity for trading operations
   • 📊 ACTIVE POSITIONS: Review current holdings for adjustments
   • 📈 MARKET DATA: Retrieved data for 16 symbols
   • 🎓 PAPER TRADING: Practice environment - all features available
   • 📰 NEWS ALERT: 2 market bulletins - review for trading insights
   • ✅ MEMORY HEALTHY: All memory backends operational
   • 🔧 SIMULATION MODE: This is simulated data for development/testing
================================================================================
```

## Features

### Account Analysis
- Real-time cash balance and buying power
- Current position holdings with P&L
- Account type and trading permissions
- Margin utilization and restrictions

### Market Intelligence
- Pre-market quotes for major indices and stocks
- Volume analysis and price movements
- News bulletins from exchanges
- Market condition assessments

### System Diagnostics
- Memory backend health checks
- API connectivity verification
- Trading permission validation
- System recommendation engine

### Trading Readiness
- Automated pre-market checklist
- Risk assessment and position review
- Market condition analysis
- Trading opportunity identification

## Integration

The premarket preparation integrates with:

- **IBKR API**: Live trading data and account information
- **Memory Systems**: Advanced multi-backend memory management
- **Health Monitoring**: Comprehensive API and system health checks
- **AI Agents**: Provides context for trading decisions

## Development Notes

- **Simulated Mode**: Use for development, testing, and demonstrations
- **Live Mode**: Use only when TWS/Gateway is running and stable
- **Error Handling**: Both modes include comprehensive error handling
- **Logging**: Detailed logging for troubleshooting and monitoring

## Next Steps

After running premarket preparation:

1. **Review Recommendations**: Check AI-generated insights
2. **Position Analysis**: Evaluate current holdings
3. **Market Assessment**: Review key symbol movements
4. **Strategy Planning**: Use data for trading decisions
5. **Risk Management**: Verify position sizing and exposure

## Troubleshooting

### Live Connection Issues
- Ensure TWS/Gateway is running on port 7497
- Verify API access is enabled in TWS settings
- Check firewall settings for port access
- Confirm IBKR credentials in `.env` file

### Simulation Issues
- Ensure all Python dependencies are installed
- Check memory backend initialization
- Verify `.env` file is properly loaded

### Common Errors
- **Connection Refused**: TWS not running or wrong port
- **Authentication Failed**: Check IBKR credentials
- **API Not Enabled**: Enable API in TWS settings
- **Memory Backend Issues**: Check dependency installations