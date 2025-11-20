#!/usr/bin/env python3
"""
Quick IBKR Connection Test
Run this after configuring TWS API settings
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        print("3. Testing market data...")
        try:
            # Test market data first (most reliable)
            data = await bridge.get_market_data('SPY')
            if data:
                print("   ✅ Market data retrieved")
                price = data.get('close', data.get('price', 'N/A'))
                source = data.get('source', 'unknown')
                print(f"   SPY: ${price} (source: {source})")
            else:
                print("   ❌ No market data received")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        print("4. Testing positions...")
        try:
            positions = await bridge.get_positions()
            print(f"   ✅ Positions retrieved: {len(positions)} positions")
            if positions:
                for pos in positions[:2]:  # Show first 2
                    symbol = pos.get('symbol', 'Unknown')
                    qty = pos.get('position', 0)
                    print(f"      {symbol}: {qty} shares")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        print("5. Testing account summary...")
        try:
            account = await bridge.get_account_summary()
            if account and 'error' not in account:
                print("   ✅ Account data retrieved")
                cash = account.get('TotalCashValue', account.get('cash_balance', 'N/A'))
                print(f"   Cash: ${cash}")
            else:
                print("   ❌ No account data")
                error_msg = account.get('error', 'Unknown') if account else 'No response'
                print(f"   Error: {error_msg}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    else:
        print("3. IBKR not connected - using yfinance fallback")
        try:
            data = await bridge.get_market_data('SPY')
            if data:
                print("   ✅ Fallback data working")
                price = data.get('close', data.get('price', 'N/A'))
                print(f"   SPY Price: ${price}")
            else:
                print("   ❌ Fallback failed")
        except Exception as e:
            print(f"   ❌ Fallback error: {e}")

    # Always disconnect
    try:
        await bridge.disconnect()
        print("🔌 Disconnected from IBKR")
    except Exception as e:
        print(f"⚠️  Disconnect error: {e}")

    print("\n🎯 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_connection())