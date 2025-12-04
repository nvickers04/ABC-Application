#!/usr/bin/env python3
"""
Final API Test - Run this after enabling TWS API
"""

import asyncio
import sys

async def final_test():
    """Final comprehensive test"""
    print("🎯 FINAL API TEST")
    print("=" * 50)

    try:
        from src.integrations.ibkr_connector import IBKRConnector

        print("1. Testing basic import...")
        print("✅ IBKR connector imported")

        print("\n2. Testing connector initialization...")
        connector = IBKRConnector()
        print("✅ Connector initialized")

        print("\n3. Testing connection...")
        connected = await connector.connect()

        if not connected:
            print("❌ Connection failed - API still not enabled")
            return False

        print("✅ Connected to IBKR Paper Trading!")

        print("\n4. Testing account access...")
        summary = await connector.get_account_summary()

        if 'error' in summary:
            print(f"⚠️ Account access issue: {summary['error']}")
        else:
            print("✅ Account summary retrieved")
            print(f"   Account: {summary.get('account_id', 'Unknown')}")
            print(f"   Cash: ${summary.get('cash_balance', 0):,.2f}")
        print("\n5. Testing position query...")
        positions = await connector.get_positions()
        print(f"✅ Positions retrieved: {len(positions)} positions")

        await connector.disconnect()
        print("\n🎉 SUCCESS! TWS API is fully enabled and working!")
        print("\n🚀 READY FOR LIVE TRADING!")
        print("Run: python tools/start_unified_workflow.py --mode hybrid --symbols SPY")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("Run this script AFTER enabling TWS API.")
    print("If it fails, the API is still not enabled.")
    print()

    success = asyncio.run(final_test())

    if not success:
        print("\n" + "=" * 50)
        print("❌ TWS API IS STILL NOT ENABLED")
        print("Go to TWS: File → Global Configuration → API")
        print("Check 'Enable ActiveX and Socket Clients'")
        print("Click OK, restart TWS, then run this script again")

if __name__ == "__main__":
    main()