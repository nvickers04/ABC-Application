#!/usr/bin/env python3
"""
Simple IBKR Connection Test
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_ibkr():
    """Simple IBKR connection test"""
    try:
        print("🔌 Testing IBKR Connection...")

        from integrations.ibkr_connector import get_ibkr_connector

        connector = get_ibkr_connector()
        print("📡 Attempting to connect...")

        connected = await connector.connect()
        if connected:
            print("✅ IBKR Connected Successfully!")

            # Test basic functions
            account = await connector.get_account_summary()
            print(f"💰 Account: ${account.get('cash_balance', 0):,.2f}")

            positions = await connector.get_positions()
            print(f"📊 Positions: {len(positions)}")

        else:
            print("❌ IBKR Connection Failed")

        await connector.disconnect()
        print("🔌 Disconnected")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ibkr())