#!/usr/bin/env python3
"""
Migration Staging Test Script

Tests migration from Direct IBKR Connector to Nautilus Bridge
in a controlled staging environment before production rollout.
"""

import asyncio
import sys
import time
import logging
from typing import Dict, Any
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.integrations.ibkr_connector import IBKRConnector
from src.integrations.nautilus_ibkr_bridge import NautilusIBKRBridge, BridgeConfig, BridgeMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MigrationTester:
    """Tests migration between implementations"""

    def __init__(self):
        self.results = {}

    async def test_direct_connector(self) -> Dict[str, Any]:
        """Test direct IBKR connector"""
        logger.info("Testing Direct IBKR Connector...")
        start_time = time.time()

        try:
            connector = IBKRConnector()

            # Test basic functionality
            status = connector.get_connection_status()
            market_open = connector._is_market_open()

            # Test market data (if connected)
            market_data = None
            if status.get('connected', False):
                try:
                    market_data = await connector.get_market_data('AAPL')
                except Exception as e:
                    logger.warning(f"Market data test failed: {e}")

            end_time = time.time()

            result = {
                'implementation': 'Direct Connector',
                'connection_status': status,
                'market_open': market_open,
                'market_data_success': market_data is not None,
                'execution_time': end_time - start_time,
                'success': True
            }

        except Exception as e:
            end_time = time.time()
            result = {
                'implementation': 'Direct Connector',
                'error': str(e),
                'execution_time': end_time - start_time,
                'success': False
            }

        self.results['direct'] = result
        return result

    async def test_bridge_ib_insync_only(self) -> Dict[str, Any]:
        """Test bridge in IB_INSYNC_ONLY mode"""
        logger.info("Testing Bridge (IB_INSYNC_ONLY mode)...")
        start_time = time.time()

        try:
            config = BridgeConfig(mode=BridgeMode.IB_INSYNC_ONLY)
            bridge = NautilusIBKRBridge(config)

            await bridge.initialize()

            # Test basic functionality
            status = bridge.get_bridge_status()
            # Bridge uses different market validation
            try:
                from src.integrations.live_trading_safeguards import validate_trading_conditions
                market_safe, market_reason = await validate_trading_conditions()
                market_open = market_safe
            except Exception as e:
                market_open = "N/A"

            # Test market data
            market_data = None
            try:
                market_data = await bridge.get_market_data('AAPL')
            except Exception as e:
                logger.warning(f"Market data test failed: {e}")

            await bridge.disconnect()

            end_time = time.time()

            result = {
                'implementation': 'Bridge (IB_INSYNC_ONLY)',
                'connection_status': status,
                'market_open': market_open,
                'market_data_success': market_data is not None,
                'execution_time': end_time - start_time,
                'success': True
            }

        except Exception as e:
            end_time = time.time()
            result = {
                'implementation': 'Bridge (IB_INSYNC_ONLY)',
                'error': str(e),
                'execution_time': end_time - start_time,
                'success': False
            }

        self.results['bridge_ib_insync'] = result
        return result

    async def test_bridge_nautilus_enhanced(self) -> Dict[str, Any]:
        """Test bridge in NAUTILUS_ENHANCED mode"""
        logger.info("Testing Bridge (NAUTILUS_ENHANCED mode)...")
        start_time = time.time()

        try:
            config = BridgeConfig(mode=BridgeMode.NAUTILUS_ENHANCED)
            bridge = NautilusIBKRBridge(config)

            await bridge.initialize()

            # Test basic functionality
            status = bridge.get_bridge_status()
            # Bridge uses different market validation
            try:
                from src.integrations.live_trading_safeguards import validate_trading_conditions
                market_safe, market_reason = await validate_trading_conditions()
                market_open = market_safe
            except Exception as e:
                market_open = "N/A"

            # Test market data
            market_data = None
            try:
                market_data = await bridge.get_market_data('AAPL')
            except Exception as e:
                logger.warning(f"Market data test failed: {e}")

            await bridge.disconnect()

            end_time = time.time()

            result = {
                'implementation': 'Bridge (NAUTILUS_ENHANCED)',
                'connection_status': status,
                'market_open': market_open,
                'market_data_success': market_data is not None,
                'execution_time': end_time - start_time,
                'success': True
            }

        except Exception as e:
            end_time = time.time()
            result = {
                'implementation': 'Bridge (NAUTILUS_ENHANCED)',
                'error': str(e),
                'execution_time': end_time - start_time,
                'success': False
            }

        self.results['bridge_nautilus_enhanced'] = result
        return result

    def print_comparison_report(self):
        """Print detailed comparison report"""
        print("\n" + "="*60)
        print("MIGRATION STAGING TEST REPORT")
        print("="*60)

        for key, result in self.results.items():
            print(f"\n{result['implementation']}:")
            print(f"  Success: {result['success']}")
            print(f"  Execution Time: {result['execution_time']:.2f}s")

            if result['success']:
                print(f"  Connection Status: {result.get('connection_status', 'N/A')}")
                print(f"  Market Open: {result.get('market_open', 'N/A')}")
                print(f"  Market Data Success: {result.get('market_data_success', 'N/A')}")
            else:
                print(f"  Error: {result.get('error', 'Unknown')}")

        # Performance comparison
        if len(self.results) >= 2:
            print("\nPERFORMANCE COMPARISON:")
            successful_results = {k: v for k, v in self.results.items() if v['success']}
            if successful_results:
                fastest = min(successful_results.items(), key=lambda x: x[1]['execution_time'])
                slowest = max(successful_results.items(), key=lambda x: x[1]['execution_time'])

                print(f"  Fastest: {fastest[1]['implementation']} ({fastest[1]['execution_time']:.2f}s)")
                print(f"  Slowest: {slowest[1]['implementation']} ({slowest[1]['execution_time']:.2f}s)")
                print(f"  Performance Ratio: {slowest[1]['execution_time'] / fastest[1]['execution_time']:.2f}x")

        print("\n" + "="*60)

async def main():
    """Run migration staging tests"""
    print("Starting Migration Staging Tests...")
    print("This simulates testing the migration in a staging environment")

    tester = MigrationTester()

    # Test all implementations
    await tester.test_direct_connector()
    await tester.test_bridge_ib_insync_only()
    await tester.test_bridge_nautilus_enhanced()

    # Print report
    tester.print_comparison_report()

    # Determine if migration is ready
    successful_implementations = [r for r in tester.results.values() if r['success']]

    if len(successful_implementations) >= 2:
        print("\n✅ MIGRATION STAGING TEST PASSED")
        print("Multiple implementations are functional")
        print("Ready to proceed with production migration evaluation")
    else:
        print("\n❌ MIGRATION STAGING TEST FAILED")
        print("Not all implementations are functional")
        print("Further investigation required before migration")

if __name__ == "__main__":
    asyncio.run(main())