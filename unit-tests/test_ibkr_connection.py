#!/usr/bin/env python3
"""
Quick IBKR Connection Test
Run this after configuring TWS API settings
"""

import asyncio
from integrations.nautilus_ibkr_bridge import get_nautilus_ibkr_bridge

async def test_connection():
    print("🔄 Testing IBKR Connection...")
    print("=" * 40)

    # Get bridge instance
    bridge = get_nautilus_ibkr_bridge()

    # Test initialization
    print("1. Initializing bridge...")
    success = await bridge.initialize()
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")

    # Check connection status
    status = bridge.get_bridge_status()
    print(f"2. Connection Status: {status['ibkr_connected']}")

    if status['ibkr_connected']:
        print("3. Testing live market data...")
        try:
            data = await bridge.get_market_data('SPY')
            if data:
                print("   ✅ SPY data retrieved from IBKR")
                print(f"   Price: ${data.get('close', 'N/A'):.2f}")
            else:
                print("   ❌ No data received")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        print("4. Testing account summary...")
        try:
            account = await bridge.get_account_summary()
            if account:
                print("   ✅ Account data retrieved")
                cash = account.get('TotalCashValue', 'N/A')
                print(f"   Cash: ${cash}")
            else:
                print("   ❌ No account data")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("3. IBKR not connected - check TWS configuration")
        print("   Using yfinance fallback for market data...")

        data = await bridge.get_market_data('SPY')
        if data:
            print("   ✅ Fallback data working")
            print(f"   SPY Price: ${data.get('close', 'N/A'):.2f}")

    print("\n🎯 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_connection())