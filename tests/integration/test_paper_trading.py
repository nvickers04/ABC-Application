#!/usr/bin/env python3
"""
Quick Paper Trading Test - Verify IBKR paper trading setup
"""

import asyncio
import sys
import os

async def test_paper_trading():
    """Test paper trading connection and basic functionality"""
    print("🧪 Testing Paper Trading Setup")
    print("=" * 50)

    try:
        # Import the connector
        from src.integrations.ibkr_connector import IBKRConnector

        print("✅ IBKR connector imported successfully")

        # Initialize connector
        connector = IBKRConnector()
        print("✅ Connector initialized")

        # Test connection
        print("\n🔌 Testing connection...")
        connected = await connector.connect()

        if not connected:
            print("❌ Connection failed")
            return False

        print("✅ Connected to IBKR Paper Trading!")

        # Get account info
        print("\n📊 Getting account information...")
        summary = await connector.get_account_summary()

        if summary:
            print("✅ Account summary retrieved:")
            for key, value in summary.items():
                print(f"  • {key}: {value}")
        else:
            print("⚠️ Could not retrieve account summary")

        # Test position query
        print("\n📈 Testing position query...")
        positions = await connector.get_positions()

        if positions is not None:
            print(f"✅ Positions retrieved: {len(positions)} positions")
            if positions:
                for pos in positions[:3]:  # Show first 3
                    print(f"  • {pos}")
        else:
            print("⚠️ Could not retrieve positions")

        # Disconnect
        await connector.disconnect()
        print("\n✅ Test completed successfully!")
        print("\n🎯 Ready for live trading with Discord integration!")
        print("Run: python tools/start_unified_workflow.py --mode hybrid --symbols SPY")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🎯 ABC Application - Paper Trading Test")
    print("Testing IBKR paper trading connection and basic functionality")
    print()

    # Check if we're in the right directory
    if not os.path.exists('integrations/ibkr_connector.py'):
        print("❌ Please run this script from the ABC-Application root directory")
        sys.exit(1)

    # Run the test
    success = asyncio.run(test_paper_trading())

    if success:
        print("\n" + "=" * 50)
        print("🎉 PAPER TRADING IS READY!")
        print("You can now start live trading with Discord.")
    else:
        print("\n" + "=" * 50)
        print("❌ PAPER TRADING SETUP NEEDS ATTENTION")
        print("Please check your TWS configuration and try again.")

if __name__ == "__main__":
    main()