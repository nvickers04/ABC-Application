# test_ibkr_historical.py
# Simple test script for IBKR historical data functionality

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.integrations.ibkr_historical_data import IBKRHistoricalDataProvider

async def test_ibkr_historical():
    """Test IBKR historical data provider"""
    print("🧪 Testing IBKR Historical Data Provider")
    print("=" * 45)

    provider = IBKRHistoricalDataProvider()

    # Test connection
    print("\n1. Testing IBKR Connection...")
    connected = await provider.connector.connect()
    if not connected:
        print("❌ IBKR connection failed - TWS may not be running")
        print("Please ensure:")
        print("  - IBKR Trader Workstation is running")
        print("  - API is enabled in TWS")
        print("  - Using paper trading account")
        return

    print("✅ IBKR connection successful")

    # Test historical data retrieval
    print("\n2. Testing Historical Data Retrieval...")
    symbol = 'SPY'
    start_date = '2024-01-01'
    end_date = '2024-01-05'

    print(f"Requesting {symbol} data from {start_date} to {end_date}")

    data = await provider.get_historical_bars(symbol, start_date, end_date, '1 day')

    if data is not None and not data.empty:
        print("✅ Historical data retrieved successfully!")
        print(f"   Bars: {len(data)}")
        print(f"   Date range: {data.index.min().date()} to {data.index.max().date()}")
        print(f"   Sample prices: Open=${data['open'].iloc[0]:.2f}, Close=${data['close'].iloc[0]:.2f}")
        print(f"   Volume: {data['volume'].iloc[0]:,}")
    else:
        print("❌ No historical data retrieved")
        return

    # Test multiple symbols
    print("\n3. Testing Multiple Symbol Retrieval...")
    symbols = ['AAPL', 'MSFT']
    multi_data = await provider.get_multiple_symbols_data(symbols, start_date, end_date, '1 day')

    print(f"✅ Retrieved data for {len(multi_data)}/{len(symbols)} symbols")
    for sym, df in multi_data.items():
        print(f"   {sym}: {len(df)} bars")

    # Test portfolio data
    print("\n4. Testing Portfolio Data...")
    weights = {'AAPL': 0.6, 'MSFT': 0.4}
    portfolio_data = await provider.get_portfolio_historical_data(symbols, weights, start_date, end_date)

    if 'portfolio_returns' in portfolio_data:
        print("✅ Portfolio data retrieved successfully!")
        print(f"   Return periods: {len(portfolio_data['portfolio_returns'])}")
    else:
        print("❌ Portfolio data retrieval failed")

    print("\n🎉 IBKR Historical Data Provider test completed!")
    print("\nYou can now run:")
    print("  python comprehensive_ibkr_simulation.py  # For IBKR-powered backtesting")
    print("  python compare_data_sources.py           # To compare yfinance vs IBKR")

if __name__ == '__main__':
    asyncio.run(test_ibkr_historical())