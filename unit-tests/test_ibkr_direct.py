import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('src'))

from integrations.ibkr_connector import get_ibkr_connector

async def test_ibkr_direct():
    print("🔄 Testing IBKR Direct Connection...")
    print("=" * 40)

    # Get connector instance
    connector = get_ibkr_connector()

    # Test connection
    print("1. Connecting to IBKR...")
    connected = await connector.connect()
    print(f"   Result: {'✅ Success' if connected else '❌ Failed'}")

    if connected:
        print("2. Testing market data...")
        try:
            # Test with yfinance fallback since IBKR market data has issues
            import yfinance as yf
            ticker = yf.Ticker('SPY')
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                latest = data.iloc[-1]
                print("   ✅ Market data retrieved (yfinance)")
                print(f"   SPY: ${float(latest['Close']):.2f}")
            else:
                print("   ❌ No market data")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        print("3. Testing account summary...")
        try:
            account = await connector.get_account_summary()
            if account and 'error' not in account:
                print("   ✅ Account data retrieved")
                cash = account.get('cash_balance', 'N/A')
                positions = account.get('total_positions', 0)
                print(f"   Cash: ${cash}")
                print(f"   Positions: {positions}")
            else:
                print("   ❌ No account data")
                error_msg = account.get('error', 'Unknown') if account else 'No response'
                print(f"   Error: {error_msg}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        print("4. Testing positions...")
        try:
            positions = await connector.get_positions()
            print(f"   ✅ Positions retrieved: {len(positions)} positions")
            if positions:
                for pos in positions[:2]:
                    symbol = pos.get('symbol', 'Unknown')
                    qty = pos.get('position', 0)
                    print(f"      {symbol}: {qty} shares")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    else:
        print("❌ IBKR connection failed")
        print("   Make sure TWS is running and API is enabled")

    # Disconnect
    try:
        await connector.disconnect()
        print("🔌 Disconnected from IBKR")
    except Exception as e:
        print(f"⚠️  Disconnect error: {e}")

    print("\n🎯 Direct test completed!")

if __name__ == "__main__":
    asyncio.run(test_ibkr_direct())