#!/usr/bin/env python3
"""
Test IBKR connection with Client ID 1
"""

import asyncio
import time

async def test_client_id_1():
    """Test connection with client ID 1"""
    print("🧪 Testing IBKR connection with Client ID 1")

    try:
        from integrations.ibkr_connector import IBKRConnector

        # Create connector
        connector = IBKRConnector()
        connector.client_id = 1  # Client ID 1

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
    success = asyncio.run(test_client_id_1())
    if success:
        print("\n🎉 SUCCESS! Connection works with Client ID 1.")
        print("Update your config/ibkr_config.ini:")
        print("client_id=1")
        print("\nThen run: python test_paper_trading.py")
    else:
        print("\n❌ Still failing. Check IBKR account API access.")