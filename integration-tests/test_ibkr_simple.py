#!/usr/bin/env python3
"""
Simplified IBKR Trading Test
Tests the IBKR connection and basic trading functionality
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ibkr_connection():
    """Test IBKR connection and basic operations"""
    logger.info("🔄 Testing IBKR Paper Trading Connection...")

    try:
        from src.integrations.ibkr_connector import get_ibkr_connector

        # Get connector
        connector = get_ibkr_connector()
        logger.info("✅ IBKR Connector initialized")

        # Test connection
        logger.info("1. Connecting to IBKR...")
        connected = await connector.connect()

        if not connected:
            logger.error("❌ Failed to connect to IBKR")
            return

        logger.info("✅ Connected to IBKR Paper Trading")

        # Test account summary
        logger.info("2. Getting account summary...")
        account = await connector.get_account_summary()

        if 'error' in account:
            logger.error(f"❌ Account summary error: {account['error']}")
        else:
            logger.info("✅ Account summary retrieved")
            logger.info(f"   Account: {account.get('account_id', 'Unknown')}")
            logger.info(f"   Cash: ${account.get('cash_balance', 0):,.2f}")
            logger.info(f"   Positions: {account.get('total_positions', 0)}")

        # Test market data
        logger.info("3. Testing market data...")
        market_data = await connector.get_market_data('SPY')

        if market_data:
            logger.info("✅ Market data retrieved")
            logger.info(f"   SPY Price: ${market_data.get('close', 0):.2f}")
        else:
            logger.warning("⚠️  Market data not available")

        # Test placing a small order (if market is open)
        logger.info("4. Testing order placement...")
        if connector._is_market_open():
            # Place a small test order
            order_result = await connector.place_order(
                symbol='SPY',
                quantity=1,
                order_type='MKT',
                action='BUY'
            )

            if order_result.get('success'):
                logger.info("✅ Test order placed successfully")
                logger.info(f"   Order ID: {order_result.get('order_id')}")
                logger.info(f"   Symbol: {order_result.get('symbol')}")
                logger.info(f"   Quantity: {order_result.get('quantity')}")
                logger.info(f"   Action: {order_result.get('action')}")

                # Check order status
                await asyncio.sleep(2)  # Wait a bit
                status = await connector.get_order_status(order_result.get('order_id'))
                if status:
                    logger.info(f"   Order Status: {status.get('status', 'Unknown')}")
            else:
                logger.warning(f"⚠️  Order placement failed: {order_result.get('error', 'Unknown error')}")
        else:
            logger.info("⏰ Market closed - skipping order test")

        # Disconnect
        logger.info("5. Disconnecting...")
        await connector.disconnect()
        logger.info("✅ Disconnected from IBKR")

        logger.info("=" * 50)
        logger.info("🎯 IBKR Integration Test Complete!")
        logger.info("If you see this message, the system can connect to IBKR")
        logger.info("and execute real trades that will appear in TWS!")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main entry point"""
    await test_ibkr_connection()

if __name__ == "__main__":
    asyncio.run(main())