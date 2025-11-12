# src/main.py
# Updated: Integrated A2A protocol for message passing between agents (replaces direct calls with send/receive for robustness/scalability up to 50 agents). Added tool calls in agents (e.g., yfinance in Data as real tool call, tf sim in Risk as stub tool call—ties to resource-mapping-and-evaluation.md for yfinance/tf-quant). Run with: python src/main.py (from root terminal)—now logs A2A sends/receives with IDs for audits.
# Purpose: Orchestrates the full macro-to-micro agent loop in the AI Portfolio Manager (Data -> Strategy -> Risk -> Execution -> Reflection -> Learning), with closed-loop batching/reflections via A2A.
# Structural Reasoning: Ties to architecture.md (sequential flows with bidirectional loops/escalations) and code-skeleton.md (LangGraph-inspired async loop with A2A edges); backs funding with traceable full-cycle logs (e.g., "Sent proposal message uuid... from data to strategy, received vetted with POP 0.72, no-traded for <5% drawdown preservation, bonus awarded for >25% ambition—enabling 15-20% profitability audits"). Async for scalability (e.g., parallel pings/sims); A2A registry/send/receive ensures reliable handoffs (e.g., DataFrame JSON payloads); tool calls in process_input for ReAct-like behaviors (e.g., Data calls yfinance_tool for real pulls, Risk calls tf_sim_tool for POP). Limits to 50 agents per A2A init. Sample data as placeholders (real IBKR in production via tools). For legacy wealth: Ensures disciplined execution (e.g., no-trade defaults vs erosion) to preserve capital while maximizing growth through experiential refinements—no forced trades, honorable path to substantial legacy—did my absolute best to make it beautiful and expandable.

import asyncio
import logging
from typing import Dict, Any
import pandas as pd  # For DataFrames in handoffs.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Dynamic root path for imports from src/.
sys.path.insert(0, str(Path(__file__).parent))  # Add src to path for utils imports.

from utils.a2a_protocol import A2AProtocol, BaseMessage  # From src/utils/a2a_protocol.py.
from utils.api_health_monitor import start_health_monitoring, get_api_health_summary
from agents.data import DataAgent
from agents.strategy import StrategyAgent
from agents.risk import RiskAgent
from agents.execution import ExecutionAgent
from agents.reflection import ReflectionAgent
from agents.learning import LearningAgent
from agents.macro import MacroAgent

# Setup logging for traceability (full-cycle audits)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main_loop() -> Dict[str, Any]:
    """
    Runs the full agent orchestration loop using StateGraph.
    """
    logger.info("Starting main orchestration loop with StateGraph")

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

    # Run orchestration
    initial_data = {'symbols': ['SPY']}
    result = await a2a.run_orchestration(initial_data)
    
    logger.info(f"Orchestration completed: {result}")
    return result

# Entry point for tests (run python src/main.py)
if __name__ == "__main__":
    results = asyncio.run(main_loop())
    print("End-to-End Test Results with A2A:", results)