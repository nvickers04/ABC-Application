import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('src'))

from src.integrations.ibkr_connector import get_ibkr_connector

async def debug_ibkr_connection():
    print("🔍 Debugging IBKR Connection...")
    print("=" * 40)

    # Get connector instance
    connector = get_ibkr_connector()

    # Test connection
    print("1. Connecting to IBKR...")
    connected = await connector.connect()
    print(f"   Result: {'✅ Success' if connected else '❌ Failed'}")

    if connected:
        print(f"2. Connection status: {connector.connected}")
        print(f"3. IB object exists: {connector.ib is not None}")

        if connector.ib:
            print(f"4. IB is connected: {connector.ib.isConnected()}")
            print(f"5. IB client exists: {connector.ib.client is not None}")

            if connector.ib.client:
                print(f"6. Client connected: {connector.ib.client.isConnected()}")

        print("7. Testing simple IBKR call...")
        try:
            # Test a simple synchronous call
            accounts = connector.ib.managedAccounts()
            print(f"   Managed accounts: {accounts}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        print("8. Testing account summary...")
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
            print(f"   ❌ Exception: {e}")

        print("9. Testing positions...")
        try:
            positions = await connector.get_positions()
            if positions is not None:
                print(f"   ✅ Positions retrieved: {len(positions)} positions")
                if positions:
                    for pos in positions[:3]:  # Show first 3 positions
                        print(f"      {pos.get('symbol', 'N/A')}: {pos.get('position', 0)} shares")
            else:
                print("   ❌ No positions data")
        except Exception as e:
            print(f"   ❌ Exception: {e}")

    else:
        print("❌ IBKR connection failed")

    # Disconnect
    try:
        await connector.disconnect()
        print("🔌 Disconnected from IBKR")
    except Exception as e:
        print(f"⚠️  Disconnect error: {e}")

    print("\n🎯 Debug completed!")

if __name__ == "__main__":
    asyncio.run(debug_ibkr_connection())