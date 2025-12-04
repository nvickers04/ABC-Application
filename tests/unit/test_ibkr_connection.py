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

from src.integrations.nautilus_ibkr_bridge import get_nautilus_ibkr_bridge

async def test_connection():
    print("🔄 Testing IBKR Connection...")
    print("=" * 40)

    # Get bridge instance
    bridge = get_nautilus_ibkr_bridge()

    # Test connection by attempting operations
    print("1. Testing IBKR connection via operations...")

    try:
        # Try to get positions to test connection
        positions = await asyncio.wait_for(bridge.get_positions(), timeout=10.0)
        print("   ✅ Connection successful")
        connected = True
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        connected = False

    if connected:
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
        print("2. IBKR not connected - skipping market data test")

    print("\n🎯 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_connection())