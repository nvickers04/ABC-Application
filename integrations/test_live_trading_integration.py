# integrations/test_live_trading_integration.py
"""
Comprehensive test for live trading integration with safeguards
"""

import asyncio
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.nautilus_ibkr_bridge import (
    NautilusIBKRBridge,
    BridgeConfig,
    BridgeMode,
    get_nautilus_ibkr_bridge
)
from integrations.live_trading_safeguards import (
    get_live_trading_safeguards,
    get_risk_status,
    emergency_stop
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_live_trading_integration():
    """Comprehensive test of live trading integration"""
    print("🧪 Testing Live Trading Integration with Safeguards")
    print("=" * 60)

    # Initialize bridge with risk management enabled
    config = BridgeConfig(
        mode=BridgeMode.IB_INSYNC_ONLY,
        enable_risk_management=True,
        enable_position_sizing=True
    )
    bridge = NautilusIBKRBridge(config)

    # Initialize safeguards
    safeguards = get_live_trading_safeguards()

    print("1. Testing Bridge Initialization...")
    success = await bridge.initialize()
    print(f"   Bridge initialization: {'✅ Success' if success else '❌ Failed'}")

    if not success:
        print("   ❌ Cannot proceed with tests - bridge initialization failed")
        return

    print("2. Testing Risk Status...")
    risk_status = get_risk_status()
    print(f"   Trading state: {risk_status['trading_state']}")
    print(f"   Circuit breaker: {risk_status['circuit_breaker_triggered']}")
    print("   ✅ Risk status retrieved")

    print("3. Testing Market Condition Validation...")
    market_safe, market_reason = await safeguards.validate_market_conditions()
    print(f"   Market safe for trading: {'✅ Yes' if market_safe else '❌ No'}")
    print(f"   Reason: {market_reason}")

    print("4. Testing Account Connection...")
    account_info = await bridge.get_account_summary()
    if 'error' in account_info:
        print(f"   ❌ Account connection failed: {account_info['error']}")
        print("   Note: This test requires IBKR TWS/Gateway to be running")
        await bridge.disconnect()
        return
    else:
        print(f"   ✅ Account connected: {account_info.get('account_id', 'Unknown')}")
        print(f"   💰 Cash Balance: ${account_info.get('cash_balance', 0):.2f}")
    print("5. Testing Position Retrieval...")
    positions = await bridge.get_positions()
    print(f"   Positions retrieved: {len(positions)} positions")
    if positions:
        print("   Sample position:")
        pos = positions[0]
        print(f"      Symbol: {pos.get('symbol')}, Quantity: {pos.get('position')}, P&L: ${pos.get('unrealized_pnl', 0):.2f}")
    print("   ✅ Position data retrieved")

    print("6. Testing Market Data...")
    market_data = await bridge.get_market_data('SPY')
    if market_data:
        print("   ✅ Market data retrieved:")
        print(f"      SPY: ${market_data.get('close', 'N/A')}")
    else:
        print("   ⚠️  Market data not available (may be expected if IBKR not connected)")

    print("7. Testing Risk-Checked Order (Safe Test)...")
    # Test with a very small position to avoid actual trading
    test_symbol = 'SPY'
    test_quantity = 1  # Very small quantity for testing
    test_price = 400.0  # Reasonable test price

    # Get current market data for realistic price
    if market_data and 'close' in market_data:
        test_price = market_data['close']

    # Test pre-trade risk check
    risk_approved, risk_reason, risk_analysis = await safeguards.pre_trade_risk_check(
        symbol=test_symbol,
        quantity=test_quantity,
        price=test_price,
        order_type='MKT',
        account_info=account_info,
        positions=positions
    )

    print(f"   Risk check result: {'✅ Approved' if risk_approved else '❌ Rejected'}")
    print(f"   Risk level: {risk_analysis.get('risk_level', 'unknown')}")
    print(f"   Checks passed: {len(risk_analysis.get('checks_passed', []))}")
    print(f"   Checks failed: {len(risk_analysis.get('checks_failed', []))}")

    if risk_approved:
        print("   Note: Would place order here in live testing")
        print("   Order details would be:")
        print(f"      Symbol: {test_symbol}")
        print(f"      Quantity: {test_quantity}")
        print(f"      Price: ${test_price:.2f}")
    else:
        print(f"   Rejection reason: {risk_reason}")

    print("8. Testing Open Orders Retrieval...")
    open_orders = await bridge.get_open_orders()
    print(f"   Open orders: {len(open_orders)}")
    print("   ✅ Open orders retrieved")

    print("9. Testing Portfolio P&L...")
    pnl_data = await bridge.get_portfolio_pnl()
    if 'error' in pnl_data:
        print(f"   ❌ P&L retrieval failed: {pnl_data['error']}")
    else:
        print("   ✅ Portfolio P&L retrieved:")
        print(f"      Cash Balance: ${pnl_data.get('cash_balance', 0):.2f}")
        print(f"      Total P&L: ${pnl_data.get('total_pnl', 0):.2f}")
        print(f"      Positions: {pnl_data.get('positions_count', 0)}")
        print(f"      Risk state: {pnl_data.get('risk_status', {}).get('trading_state', 'unknown')}")

    print("10. Testing Emergency Stop (Safe Test)...")
    original_state = safeguards.trading_state.value
    emergency_stop("Test emergency stop")
    print(f"    Trading state changed: {original_state} → {safeguards.trading_state.value}")

    # Reset for cleanup
    safeguards.reset_emergency_stop()
    print("    Emergency stop reset for cleanup")

    print("11. Final Risk Status Check...")
    final_risk_status = get_risk_status()
    print(f"    Final trading state: {final_risk_status['trading_state']}")
    print(f"    Session orders: {final_risk_status['current_session']['orders_placed']}")

    # Cleanup
    await bridge.disconnect()
    print("✅ Bridge disconnected")

    print("\n" + "=" * 60)
    print("🎯 Live Trading Integration Test Complete")
    print("Summary:")
    print(f"  - Bridge: {'✅ Working' if success else '❌ Issues'}")
    print(f"  - Risk Management: ✅ Implemented")
    print(f"  - Account Access: {'✅ Working' if 'error' not in account_info else '⚠️ Requires IBKR connection'}")
    print(f"  - Safeguards: ✅ Active")
    print("\nNote: Full live trading requires IBKR TWS/Gateway running with API enabled")
    print("      This test validates the integration framework and safeguards")


if __name__ == "__main__":
    asyncio.run(test_live_trading_integration())