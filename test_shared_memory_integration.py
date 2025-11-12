#!/usr/bin/env python3
"""
Test script to verify shared memory integration across data and strategy subagents.
Tests the complete data flow from data collection to strategy analysis.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.shared_memory import get_multi_agent_coordinator
from src.agents.data_subs.yfinance_datasub import YfinanceDatasub
from src.agents.data_subs.institutional_datasub import InstitutionalDatasub
from src.agents.data_subs.microstructure_datasub import MicrostructureDatasub
from src.agents.data_subs.marketdataapp_datasub import MarketDataAppDatasub
from src.agents.strategy_subs.flow_strategy_sub import FlowStrategySub

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_shared_memory_integration():
    """Test the complete shared memory data flow."""
    logger.info("Starting shared memory integration test...")

    # Initialize coordinator
    coordinator = get_multi_agent_coordinator()
    logger.info("Shared memory coordinator initialized")

    # Test basic shared memory operations
    test_data = {"test_key": "test_value", "timestamp": "2024-01-01"}
    success = await coordinator.share_memory("test_agent", "test_receiver", "test_namespace", "test_key", test_data)
    logger.info(f"Basic shared memory store: {'SUCCESS' if success else 'FAILED'}")

    # Test retrieval
    retrieved = await coordinator.a2a_protocol.get_namespace("test_namespace").retrieve_shared_memory("test_key", "test_receiver")
    logger.info(f"Basic shared memory retrieve: {'SUCCESS' if retrieved else 'FAILED'}")

    # Initialize data subagents
    logger.info("Initializing data subagents...")
    yfinance_agent = YfinanceDatasub()
    institutional_agent = InstitutionalDatasub()
    microstructure_agent = MicrostructureDatasub()
    marketdataapp_agent = MarketDataAppDatasub()

    # Test data collection and storage
    test_symbol = "AAPL"
    logger.info(f"Testing data collection for {test_symbol}...")

    try:
        # Collect and store yfinance data
        yfinance_result = await yfinance_agent.process_input({"symbols": [test_symbol]})
        logger.info(f"Yfinance data collection: {'SUCCESS' if yfinance_result else 'FAILED'}")

        # Collect and store institutional data
        institutional_result = await institutional_agent.process_input({"symbol": test_symbol})
        logger.info(f"Institutional data collection: {'SUCCESS' if institutional_result else 'FAILED'}")

        # Collect and store microstructure data
        microstructure_result = await microstructure_agent.process_input({"symbol": test_symbol})
        logger.info(f"Microstructure data collection: {'SUCCESS' if microstructure_result else 'FAILED'}")

        # Collect and store marketdataapp data
        marketdataapp_result = await marketdataapp_agent.process_input({"symbol": test_symbol})
        logger.info(f"MarketDataApp data collection: {'SUCCESS' if marketdataapp_result else 'FAILED'}")

    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return False

    # Initialize strategy subagent
    logger.info("Initializing strategy subagent...")
    flow_strategy = FlowStrategySub()

    # Test strategy analysis with shared data
    try:
        strategy_input = {
            "symbols": [test_symbol],
            "timeframes": ["1min"],
            "order_flow": True,
            "institutional": True,
            "dark_pool": True,
            "market_impact": True
        }

        strategy_result = await flow_strategy.process_input(strategy_input)
        logger.info(f"Flow strategy analysis: {'SUCCESS' if strategy_result else 'FAILED'}")

        if strategy_result:
            logger.info("Strategy result keys:")
            for key in strategy_result.keys():
                logger.info(f"  - {key}")

    except Exception as e:
        logger.error(f"Strategy analysis failed: {e}")
        return False

    # Verify shared memory contents
    logger.info("Verifying shared memory contents...")
    namespaces = coordinator.a2a_protocol.get_namespace_info()

    logger.info("Available namespaces:")
    for ns_name, ns_info in namespaces.items():
        logger.info(f"  - {ns_name}: {len(ns_info.get('subscribers', {}))} subscribers")

        # Check if our test symbol data is stored
        ns_obj = coordinator.a2a_protocol.get_namespace(ns_name)
        if ns_obj:
            data = await ns_obj.retrieve_shared_memory(test_symbol, "test_agent")
            if data:
                logger.info(f"    Found {test_symbol} data in {ns_name}")

    logger.info("Shared memory integration test completed!")
    return True

async def test_data_flow_verification():
    """Verify that data flows correctly from data subs to strategy subs."""
    logger.info("Testing data flow verification...")

    coordinator = get_multi_agent_coordinator()

    # Manually store test data in shared memory
    test_market_data = {
        "market_data": {"price": 150.0, "volume": 1000000},
        "llm_analysis": {"trend": "bullish"},
        "timestamp": "2024-01-01T12:00:00",
        "symbol": "TEST"
    }

    test_institutional_data = {
        "institutional_holdings": {"total_institutional": 0.75},
        "llm_analysis": {"concentration": "high"},
        "timestamp": "2024-01-01T12:00:00",
        "symbol": "TEST"
    }

    # Store data
    await coordinator.share_memory("data_agent", "strategy_agent", "market_data", "TEST", test_market_data)
    await coordinator.share_memory("data_agent", "strategy_agent", "institutional_data", "TEST", test_institutional_data)

    # Verify retrieval
    market_ns = coordinator.a2a_protocol.get_namespace("market_data")
    institutional_ns = coordinator.a2a_protocol.get_namespace("institutional_data")
    
    retrieved_market = await market_ns.retrieve_shared_memory("TEST", "strategy_agent") if market_ns else None
    retrieved_institutional = await institutional_ns.retrieve_shared_memory("TEST", "strategy_agent") if institutional_ns else None

    success = retrieved_market is not None and retrieved_institutional is not None
    logger.info(f"Data flow verification: {'SUCCESS' if success else 'FAILED'}")

    return success

if __name__ == "__main__":
    async def main():
        print("=== Shared Memory Integration Test ===")

        # Test basic functionality
        basic_test = await test_data_flow_verification()
        if not basic_test:
            print("❌ Basic shared memory test failed")
            sys.exit(1)

        # Test full integration
        integration_test = await test_shared_memory_integration()
        if not integration_test:
            print("❌ Full integration test failed")
            sys.exit(1)

        print("✅ All shared memory tests passed!")

    asyncio.run(main())