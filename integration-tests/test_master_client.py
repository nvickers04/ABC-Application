#!/usr/bin/env python3
"""
Test IBKR connection with Master Client ID (0)
"""

import asyncio
import time

async def test_master_client():
    """Test connection with client ID 0 (master)"""
    print("🧪 Testing IBKR connection with Master Client ID 0")

    try:
        from integrations.ibkr_connector import IBKRConnector

        # Create connector
        connector = IBKRConnector()
        connector.client_id = 0  # Master client ID

        print(f"Testing connection with client ID: {connector.client_id}")

        start_time = time.time()
        connected = await connector.connect()
        end_time = time.time()

        if connected:
            print(f"✅ Connected successfully in {(end_time - start_time):.2f} seconds!")
            
            # Get account summary to verify
            summary = await connector.get_account_summary()
            if summary:
                print("\n📊 Account Summary:")
                for key, value in summary.items():
                    print(f"  • {key}: {value}")
            
            await connector.disconnect()
            return True
        else:
            print(f"❌ Failed after {(end_time - start_time):.2f} seconds")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_master_client())
    if success:
        print("\n🎉 SUCCESS! Connection works with Master Client ID 0.")
        print("Update your config/ibkr_config.ini:")
        print("client_id=0")
        print("\nThen run: python test_paper_trading.py")
    else:
        print("\n❌ Still failing. The issue is deeper - perhaps reinstall TWS.")