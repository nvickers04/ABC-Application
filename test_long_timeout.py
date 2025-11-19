#!/usr/bin/env python3
"""
Test IBKR connection with longer timeout
"""

import asyncio
import time

async def test_with_long_timeout():
    """Test connection with very long timeout"""
    print("🧪 Testing IBKR connection with extended timeout (30 seconds)")

    try:
        from integrations.ibkr_connector import IBKRConnector

        # Create connector
        connector = IBKRConnector()

        print("Setting extended timeout...")
        # Override the timeout in the connector
        connector.timeout = 30  # 30 second timeout

        print(f"Testing connection with timeout: {connector.timeout}s")

        start_time = time.time()
        connected = await connector.connect()
        end_time = time.time()

        if connected:
            print(f"✅ Connected successfully in {(end_time - start_time):.2f} seconds!")
            await connector.disconnect()
            return True
        else:
            print(f"❌ Still failed after {(end_time - start_time):.2f} seconds")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_with_long_timeout())
    if success:
        print("\n🎯 SUCCESS! Connection works with extended timeout.")
        print("The issue was just timeout - TWS needs more time to initialize.")
    else:
        print("\n❌ Still failing even with extended timeout.")
        print("This suggests an API configuration issue in TWS.")