#!/usr/bin/env python3
"""
Test IBKR connection with different client ID
"""

import asyncio
import sys

async def test_different_client_id():
    """Test connection with client ID 2"""
    print("🧪 Testing IBKR connection with Client ID 2")

    try:
        from src.integrations.ibkr_connector import IBKRConnector

        # Create connector with different client ID
        connector = IBKRConnector()
        connector.client_id = 2  # Try client ID 2 instead of 1

        print(f"Testing connection with client ID: {connector.client_id}")

        connected = await connector.connect()
        if connected:
            print("✅ Connected successfully with client ID 2!")
            await connector.disconnect()
            return True
        else:
            print("❌ Still failed with client ID 2")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_different_client_id())
    if success:
        print("\n🎯 SUCCESS! Use client ID 2 in your configuration.")
    else:
        print("\n❌ Still having issues. Check TWS API settings.")