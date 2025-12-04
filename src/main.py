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

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd  # For DataFrames in handoffs.

# Dynamic root path for imports from src/.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # Add src to path for utils imports.

from src.agents.data import DataAgent
from src.agents.execution import ExecutionAgent
from src.agents.learning import LearningAgent
from src.agents.macro import MacroAgent
from src.agents.reflection import ReflectionAgent
from src.agents.risk import RiskAgent
from src.agents.strategy import StrategyAgent
from src.agents.unified_workflow_orchestrator import UnifiedWorkflowOrchestrator, WorkflowMode
from src.utils.a2a_protocol import A2AProtocol, BaseMessage  # From src/utils/a2a_protocol.py.
from src.utils.api_health_monitor import start_health_monitoring, get_api_health_summary

# Setup centralized logging for traceability (full-cycle audits)
from src.utils.logging_config import setup_logging, get_logger, log_operation_start, log_operation_end

# Initialize centralized logging
setup_logging(level="INFO", enable_console=True, enable_file=True)
logger = get_logger(__name__)

async def main_continuous_workflow() -> None:
    """
    Runs the continuous AI Portfolio Manager workflow with Discord integration.
    This replaces the simple one-time StateGraph with a comprehensive workflow orchestrator.
    """
    log_operation_start(logger, "continuous_workflow", component="main_orchestrator")
    logger.info("Starting continuous AI Portfolio Manager workflow with Discord integration")

    # Start API health monitoring
    logger.info("Starting API health monitoring...")
    start_health_monitoring(check_interval=300)  # Check every 5 minutes

    # Get initial health status
    health_status = get_api_health_summary()
    logger.info(f"Initial API health status: {health_status['summary']}")

    # Initialize the Unified Workflow Orchestrator
    logger.info("Initializing Unified Workflow Orchestrator...")
    orchestrator = UnifiedWorkflowOrchestrator(
        mode=WorkflowMode.HYBRID,
        enable_discord=True,
        symbols=['SPY']
    )

    # Initialize and start the orchestrator
    logger.info("Starting unified workflow...")
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping orchestrator...")
        await orchestrator.stop()
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        await orchestrator.stop()
    finally:
        logger.info("AI Portfolio Manager workflow completed")# Legacy function for backward compatibility (simple one-time run)
async def main_loop() -> Dict[str, Any]:
    """
    Legacy function - runs a simple one-time StateGraph orchestration.
    Use main_continuous_workflow() for the full system.
    """
    logger.warning("Using legacy main_loop() - consider using main_continuous_workflow() for full functionality")

    # Start API health monitoring
    logger.info("Starting API health monitoring...")
    start_health_monitoring(check_interval=300)  # Check every 5 minutes

    # Get initial health status
    health_status = get_api_health_summary()
    logger.info(f"Initial API health status: {health_status['summary']}")

    # Initialize A2A with StateGraph
    a2a = A2AProtocol(max_agents=50)

    # Initialize agents
    macro_agent = MacroAgent()
    data_agent = DataAgent()
    strategy_agent = StrategyAgent()
    risk_agent = RiskAgent()
    execution_agent = ExecutionAgent()
    reflection_agent = ReflectionAgent()
    learning_agent = LearningAgent()

    # Register agents with instances
    a2a.register_agent("macro", macro_agent)
    a2a.register_agent("data", data_agent)
    a2a.register_agent("strategy", strategy_agent)
    a2a.register_agent("risk", risk_agent)
    a2a.register_agent("execution", execution_agent)
    a2a.register_agent("reflection", reflection_agent)
    a2a.register_agent("learning", learning_agent)

    # Run orchestration once
    initial_data = {'symbols': ['SPY']}
    result = await a2a.run_orchestration(initial_data)

    logger.info(f"Legacy orchestration completed: {result}")
    log_operation_end(logger, "continuous_workflow", component="main_orchestrator")
    return result

# Entry point for tests (run python src/main.py)
if __name__ == "__main__":
    # Use continuous workflow by default
    asyncio.run(main_continuous_workflow())

class TradingSystem:
    pass
