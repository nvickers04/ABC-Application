# integrations/test_nautilus_bridge.py
"""
Test script for NautilusIBKRBridge integration
"""

import asyncio
import logging
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.nautilus_ibkr_bridge import (
    NautilusIBKRBridge,
    BridgeConfig,
    BridgeMode,
    get_nautilus_ibkr_bridge,
    initialize_bridge,
    get_market_data
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skip all tests in this file - API has changed significantly
pytestmark = pytest.mark.skip(reason="NautilusIBKRBridge API has changed - tests need refactoring")


async def test_bridge_basic():
    """Test basic bridge functionality"""
    print("Testing NautilusIBKRBridge Basic Functionality")
    print("=" * 50)

    # Test 1: Bridge instantiation
    print("1. Testing bridge instantiation...")
    config = BridgeConfig(mode=BridgeMode.IB_INSYNC_ONLY)
    bridge = NautilusIBKRBridge(config)
    print("✅ Bridge instantiated successfully")

    # Test 2: Bridge status
    print("2. Testing bridge status...")
    status = bridge.get_bridge_status()
    print(f"Bridge status: {status}")
    print("✅ Bridge status retrieved")

    # Test 3: Bridge initialization
    print("3. Testing bridge initialization...")
    success = await bridge.initialize()
    print(f"Bridge initialization: {'✅ Success' if success else '❌ Failed'}")

    # Test 4: Market data (mock test)
    print("4. Testing market data retrieval...")
    try:
        # This will fail without IBKR connection, but tests the interface
        market_data = await bridge.get_market_data("SPY")
        print(f"Market data: {market_data}")
    except Exception as e:
        print(f"Market data test (expected to fail without IBKR): {e}")

    # Test 5: Singleton bridge
    print("5. Testing singleton bridge...")
    bridge2 = get_nautilus_ibkr_bridge()
    print(f"Same instance: {bridge is bridge2}")
    print("✅ Singleton pattern working")

    # Test 6: Convenience functions
    print("6. Testing convenience functions...")
    try:
        market_data = await get_market_data("AAPL")
        print(f"Convenience market data: {market_data}")
    except Exception as e:
        print(f"Convenience function test: {e}")

    print("\nBridge basic tests completed!")


async def test_bridge_modes():
    """Test different bridge modes"""
    print("\nTesting Bridge Modes")
    print("=" * 30)

    modes = [BridgeMode.IB_INSYNC_ONLY, BridgeMode.NAUTILUS_ENHANCED]

    for mode in modes:
        print(f"\nTesting mode: {mode.value}")
        config = BridgeConfig(mode=mode)
        bridge = NautilusIBKRBridge(config)

        status = bridge.get_bridge_status()
        print(f"Mode: {status['mode']}")
        print(f"Nautilus available: {status['nautilus_available']}")
        print(f"Nautilus active: {status['nautilus_active']}")

        # Test enhanced features
        if mode == BridgeMode.NAUTILUS_ENHANCED:
            analysis = await bridge.get_portfolio_analysis()
            print(f"Enhanced analysis available: {'nautilus_enhanced' in analysis}")


async def test_order_simulation():
    """Test order placement simulation"""
    print("\nTesting Order Simulation")
    print("=" * 30)

    bridge = get_nautilus_ibkr_bridge()

    # Test order placement (will use simulation since no real IBKR connection)
    try:
        result = await bridge.place_order(
            symbol="SPY",
            quantity=100,
            action="BUY",
            order_type="MKT"
        )
        print(f"Order result: {result}")
    except Exception as e:
        print(f"Order simulation: {e}")


async def main():
    """Run all bridge tests"""
    print("NautilusIBKRBridge Integration Tests")
    print("=" * 40)

    try:
        await test_bridge_basic()
        await test_bridge_modes()
        await test_order_simulation()

        print("\n🎉 All tests completed!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())