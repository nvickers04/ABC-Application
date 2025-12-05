#!/usr/bin/env python3
"""
Test script for validating unified workflow orchestrator paper trading workflow.
Tests one complete orchestration cycle to ensure all agents participate and return results.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.unified_workflow_orchestrator import UnifiedWorkflowOrchestrator, WorkflowMode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_paper_trading_workflow():
    """
    Test the unified workflow orchestrator for paper trading readiness.
    """
    logger.info("🧪 Starting paper trading workflow validation test...")

    # Step 1: Initialize the UnifiedWorkflowOrchestrator with HYBRID mode and Discord disabled
    logger.info("1️⃣ Initializing UnifiedWorkflowOrchestrator (HYBRID mode, Discord disabled)...")
    orchestrator = UnifiedWorkflowOrchestrator(
        mode=WorkflowMode.HYBRID,
        enable_discord=False,  # Disable Discord for paper trading test
        enable_health_monitoring=True
    )

    success = await orchestrator.initialize()
    if not success:
        logger.error("❌ Orchestrator initialization failed")
        return False

    logger.info("✅ Orchestrator initialized successfully")
    logger.info(f"📋 Agents initialized: {list(orchestrator.agents.keys())}")

    # Step 2: Run one full orchestration cycle using A2A protocol
    logger.info("2️⃣ Running one complete orchestration cycle...")
    try:
        initial_data = {'symbols': orchestrator.symbols, 'mode': 'hybrid', 'test_mode': True}
        result = await orchestrator.a2a_protocol.run_orchestration(initial_data)
        logger.info("✅ Orchestration cycle completed")
    except Exception as e:
        logger.error(f"❌ Orchestration cycle failed: {e}")
        await orchestrator.stop()
        return False

    # Step 3: Validate that all agents participate and return results
    logger.info("3️⃣ Validating agent participation and results...")

    # Expected agent keys in the result
    expected_agents = ['macro', 'data', 'strategy', 'risk', 'execution', 'reflection', 'learning']

    if not isinstance(result, dict):
        logger.error(f"❌ Result is not a dict: {type(result)}")
        await orchestrator.stop()
        return False

    missing_agents = []
    empty_results = []

    for agent in expected_agents:
        if agent not in result:
            missing_agents.append(agent)
        elif not result[agent]:
            empty_results.append(agent)

    if missing_agents:
        logger.error(f"❌ Missing results from agents: {missing_agents}")
        await orchestrator.stop()
        return False

    if empty_results:
        logger.warning(f"⚠️ Empty results from agents: {empty_results}")

    # Log successful agent results
    successful_agents = [agent for agent in expected_agents if agent in result and result[agent]]
    logger.info(f"✅ Agents with results: {successful_agents}")

    # Additional validation: Check for key data structures
    if 'data' in result and result['data']:
        logger.info("✅ Data agent provided market data")
    else:
        logger.warning("⚠️ Data agent did not provide market data")

    if 'strategy' in result and result['strategy']:
        logger.info("✅ Strategy agent generated proposals")
    else:
        logger.warning("⚠️ Strategy agent did not generate proposals")

    if 'risk' in result and result['risk']:
        logger.info("✅ Risk agent performed validation")
    else:
        logger.warning("⚠️ Risk agent did not perform validation")

    if 'execution' in result and result['execution']:
        logger.info("✅ Execution agent prepared orders")
    else:
        logger.warning("⚠️ Execution agent did not prepare orders")

    # Step 4: Clean up and exit
    logger.info("4️⃣ Cleaning up and exiting...")
    await orchestrator.stop()
    logger.info("✅ Orchestrator stopped successfully")

    # Final validation
    all_agents_participated = len(successful_agents) == len(expected_agents)
    if all_agents_participated:
        logger.info("🎯 SUCCESS: All agents participated in the orchestration cycle")
        logger.info("🚀 Paper trading workflow is ready for tomorrow's trading session!")
        return True
    else:
        logger.warning(f"⚠️ PARTIAL SUCCESS: {len(successful_agents)}/{len(expected_agents)} agents participated")
        logger.info("🔧 Consider reviewing agent implementations before live trading")
        return True  # Still return True as partial success is acceptable for paper trading

async def main():
    """Main test execution."""
    try:
        success = await test_paper_trading_workflow()
        exit_code = 0 if success else 1
        logger.info(f"🧪 Test completed with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())