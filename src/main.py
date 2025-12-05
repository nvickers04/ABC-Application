# Set environment variables BEFORE any imports to suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors, suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# [LABEL:COMPONENT:main_orchestrator] [LABEL:FRAMEWORK:langgraph] [LABEL:FRAMEWORK:asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Main application entry point orchestrating the 22-agent collaborative AI portfolio management system
# Dependencies: A2AProtocol, BaseAgent classes, API health monitoring, pandas, asyncio
# Related: docs/architecture.md, docs/ai-reasoning-agent-collaboration.md, src/utils/a2a_protocol.py
#
# Updated: Integrated A2A protocol for message passing between agents (replaces direct calls with send/receive for robustness/scalability up to 50 agents). Added tool calls in agents (e.g., yfinance in Data as real tool call, tf sim in Risk as stub tool call—ties to resource-mapping-and-evaluation.md for yfinance/tf-quant). Run with: python src/main.py (from root terminal)—now logs A2A sends/receives with IDs for audits.
# Purpose: Orchestrates the full macro-to-micro agent loop in the AI Portfolio Manager (Data -> Strategy -> Risk -> Execution -> Reflection -> Learning), with closed-loop batching/reflections via A2A.
# Structural Reasoning: Ties to architecture.md (sequential flows with bidirectional loops/escalations) and code-skeleton.md (LangGraph-inspired async loop with A2A edges); backs funding with traceable full-cycle logs (e.g., "Sent proposal message uuid... from data to strategy, received vetted with POP 0.72, no-traded for <5% drawdown preservation, bonus awarded for >25% ambition—enabling 15-20% profitability audits"). Async for scalability (e.g., parallel pings/sims); A2A registry/send/receive ensures reliable handoffs (e.g., DataFrame JSON payloads); tool calls in process_input for ReAct-like behaviors (e.g., Data calls yfinance_tool for real pulls, Risk calls tf_sim_tool for POP). Limits to 50 agents per A2A init. Sample data as placeholders (real IBKR in production via tools). For legacy wealth: Ensures disciplined execution (e.g., no-trade defaults vs erosion) to preserve capital while maximizing growth through experiential refinements—no forced trades, honorable path to substantial legacy—did my absolute best to make it beautiful and expandable.

# Suppress additional noisy third-party library warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow.*')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow_probability.*')
warnings.filterwarnings('ignore', category=UserWarning, module='tf_keras.*')
warnings.filterwarnings('ignore', message='.*oneDNN.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Gym.*', category=UserWarning)

import asyncio
import logging
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd  # For DataFrames in handoffs.
except ImportError as e:
    logging.warning(f"Pandas import failed (likely Windows long path issue): {e}. DataFrame functionality may be limited.")
    pd = None

from src.utils.a2a_protocol import A2AProtocol, BaseMessage  # From src/utils/a2a_protocol.py.
from src.utils.api_health_monitor import start_health_monitoring, get_api_health_summary
from src.agents.data import DataAgent
from src.agents.strategy import StrategyAgent
from src.agents.risk import RiskAgent
from src.agents.execution import ExecutionAgent
from src.agents.reflection import ReflectionAgent
from src.agents.learning import LearningAgent
from src.agents.unified_workflow_orchestrator import UnifiedWorkflowOrchestrator, WorkflowMode

# Setup centralized logging for traceability (full-cycle audits)
from src.utils.logging_config import get_logger
logger = get_logger(__name__)

async def main_continuous_workflow() -> None:
    """
    Runs the continuous AI Portfolio Manager workflow with Discord integration.
    This replaces the simple one-time StateGraph with a comprehensive workflow orchestrator.
    """
    logger.info("🚀 Starting continuous AI Portfolio Manager workflow with Discord integration")

    # Start required services (Redis, etc.)
    logger.info("🔧 Starting required services...")
    from src.utils.service_manager import start_required_services
    service_results = await start_required_services()

    # Report service startup status
    for service_name, success in service_results.items():
        if success:
            logger.info(f"✅ {service_name.title()} service started successfully")
        else:
            logger.warning(f"⚠️ {service_name.title()} service failed to start - using fallback")

    # Start API health monitoring
    logger.info("🏥 Starting API health monitoring...")
    start_health_monitoring(check_interval=300)  # Check every 5 minutes

    # Get initial health status
    health_status = get_api_health_summary()
    logger.info(f"📊 Initial API health status: {health_status['summary']}")

    # Initialize the Unified Workflow Orchestrator (paper trading ready)
    logger.info("🎯 Initializing Unified Workflow Orchestrator...")
    orchestrator = UnifiedWorkflowOrchestrator(
        mode=WorkflowMode.HYBRID,
        enable_discord=False,  # Disable Discord for paper trading
        enable_health_monitoring=True
    )

    # Initialize the orchestrator
    success = await orchestrator.initialize()
    if not success:
        logger.error("Failed to initialize orchestrator")
        return

    # Start the workflow
    logger.info("Starting paper trading workflow...")
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal, stopping orchestrator...")
        await orchestrator.stop()
    except Exception as e:
        logger.error(f"❌ Orchestrator error: {e}")
        await orchestrator.stop()
    finally:
        # Stop managed services
        logger.info("🧹 Cleaning up services...")
        from src.utils.service_manager import stop_services
        await stop_services()

        logger.info("✅ AI Portfolio Manager workflow completed")

# Entry point for tests (run python src/main.py)
if __name__ == "__main__":
    # Use continuous workflow by default
    asyncio.run(main_continuous_workflow())

class TradingSystem:
    pass
