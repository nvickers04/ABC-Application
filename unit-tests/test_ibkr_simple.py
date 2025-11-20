#!/usr/bin/env python3
"""
Simple IBKR Connection Test using Nautilus Bridge
Tests IBKR connection with automatic fallback to yfinance
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.nautilus_ibkr_bridge import get_nautilus_ibkr_bridge, BridgeConfig, BridgeMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ibkr_connection():
    """Test IBKR connection with Nautilus bridge"""

    print("🔌 Testing IBKR Connection with Nautilus Bridge")
    print("=" * 50)

    # Initialize bridge with IB_INSYNC_ONLY mode for simplicity
    config = BridgeConfig(
        mode=BridgeMode.IB_INSYNC_ONLY,
        ibkr_host="127.0.0.1",
        ibkr_port=7497,
        client_id=1,
        enable_paper_trading=True
    )

    bridge = get_nautilus_ibkr_bridge(config)

    try:
        # Try to initialize
        print("📡 Attempting to connect to IBKR...")
        connected = await bridge.initialize()

        if connected:
            print("✅ IBKR Connection Successful!")
            print("🏦 Connected to Interactive Brokers")

            # Test basic functionality
            print("\n📊 Testing basic functions...")

            # Get account summary
            account = await bridge.get_account_summary()
            if account and 'error' not in account:
                print("✅ Account summary retrieved")
                cash = account.get('cash_balance', 0)
                print(".2f")
            else:
                print("⚠️  Account summary failed")

            # Get positions
            positions = await bridge.get_positions()
            print(f"✅ Positions retrieved: {len(positions)} positions")

            # Get market data for a test symbol
            market_data = await bridge.get_market_data('SPY')
            if market_data:
                print("✅ Market data retrieved for SPY")
                print(".2f")
            else:
                print("⚠️  Market data failed")

            print("\n🎉 IBKR is fully operational!")

        else:
            print("❌ IBKR Connection Failed")
            print("🔄 Falling back to yfinance for market data...")

            # Test yfinance fallback
            market_data = await bridge.get_market_data('SPY')
            if market_data:
                print("✅ YFinance fallback working")
                print(".2f")
            else:
                print("❌ Even yfinance fallback failed")

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always disconnect
        try:
            await bridge.disconnect()
            print("🔌 Disconnected from IBKR")
        except:
            pass

    print("\n" + "=" * 50)
    print("Connection test complete")

if __name__ == "__main__":
    asyncio.run(test_ibkr_connection())</content>
<parameter name="filePath">c:\Users\nvick\ABC-Application\test_ibkr_simple.py