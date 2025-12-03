# [LABEL:TEST:ibkr_integration] [LABEL:TEST:integration] [LABEL:FRAMEWORK:asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Integration test for IBKR paper trading connection and functionality
# Dependencies: IBKR connector, paper trading account, asyncio
# Related: config/ibkr_config.ini, integrations/ibkr_connector.py, docs/IMPLEMENTATION/IBKR_PAPER_TRADING_DEPLOYMENT.md
#
#!/usr/bin/env python3
"""
IBKR Paper Trading Connection Test
Tests the IBKR connector with paper trading account
"""
import asyncio
import sys
import os
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

async def test_ibkr_paper_trading():
    """Test IBKR paper trading connection and basic functionality"""
    try:
        print("🧪 Testing IBKR Paper Trading Connection")
        print("=" * 50)

        # Import IBKR connector
        from src.integrations.ibkr_connector import get_ibkr_connector

        # Get connector instance
        connector = get_ibkr_connector()

        # Test 1: Connection
        print("1. Testing connection to IBKR Paper Trading...")
        connected = await connector.connect()

        if not connected:
            print("❌ FAILED: Could not connect to IBKR Paper Trading")
            print("\nTroubleshooting steps:")
            print("- Ensure IBKR TWS/Gateway is running")
            print("- Check that API is enabled in TWS (File > Global Configuration > API)")
            print("- Verify port 7497 is configured for paper trading")
            print("- Confirm IBKR_USERNAME and IBKR_PASSWORD are set in .env")
            return False

        print("✅ SUCCESS: Connected to IBKR Paper Trading")

        # Test 2: Account Summary
        print("\n2. Testing account summary...")
        account = await connector.get_account_summary()

        if 'error' in account:
            print(f"❌ FAILED: {account['error']}")
            return False

        print("✅ SUCCESS: Account summary retrieved")
        print(f"   Account ID: {account.get('account_id', 'N/A')}")
        print(f"   Cash Balance: ${account.get('cash_balance', 0):,.2f}")
        print(f"   Positions: {account.get('total_positions', 0)}")

        # Test 3: Market Data
        print("\n3. Testing market data retrieval...")
        market_data = await connector.get_market_data('SPY')

        if not market_data:
            print("⚠️  WARNING: Could not retrieve market data (may be normal outside market hours)")
        else:
            print("✅ SUCCESS: Market data retrieved")
            print(f"   SPY Price: ${market_data.get('close', 0):.2f}")

        # Test 4: Positions
        print("\n4. Testing positions retrieval...")
        positions = await connector.get_positions()
        print(f"✅ SUCCESS: Retrieved {len(positions)} positions")

        # Test 5: Open Orders
        print("\n5. Testing open orders...")
        open_orders = await connector.get_open_orders()
        print(f"✅ SUCCESS: Retrieved {len(open_orders)} open orders")

        # Disconnect
        await connector.disconnect()
        print("\n✅ SUCCESS: Disconnected from IBKR")

        print("\n🎉 All IBKR Paper Trading tests passed!")
        print("\nNext steps:")
        print("- The ABC Application can now connect to IBKR Paper Trading")
        print("- Monitor logs: journalctl -u abc-application -f")
        print("- Check health: curl http://localhost:8000/health")

        return True

    except Exception as e:
        print(f"❌ ERROR: IBKR test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        print(f"❌ ERROR: IBKR test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ibkr_paper_trading())
    sys.exit(0 if success else 1)