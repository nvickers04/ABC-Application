#!/usr/bin/env python3
"""
Unified Workflow Orchestrator - Paper Trading Ready
Consolidates Live Workflow, Continuous Trading, and 24/6 operations into
one bullet-proof system with multiple operating modes.

Supports paper trading without Discord integration.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

# Import core components
from src.agents.base import BaseAgent
from src.agents.data import DataAgent
from src.agents.strategy import StrategyAgent
from src.agents.risk import RiskAgent
from src.agents.execution import ExecutionAgent
from src.agents.reflection import ReflectionAgent
from src.agents.learning import LearningAgent
from src.agents.macro import MacroAgent

# Import utilities
from src.utils.a2a_protocol import A2AProtocol
from src.utils.logging_config import get_logger
from src.utils.redis_cache import RedisCacheManager
from src.utils.alert_manager import get_alert_manager

# Setup logging
logger = get_logger(__name__)

class WorkflowMode(Enum):
    """Operating modes for the unified workflow orchestrator."""
    ANALYSIS = "analysis"      # Full collaborative analysis with human intervention
    EXECUTION = "execution"    # Automated trading execution only
    HYBRID = "hybrid"         # Analysis + automated execution with oversight
    BACKTEST = "backtest"      # Historical simulation and validation

class UnifiedWorkflowOrchestrator:
    """
    Unified Workflow Orchestrator - The single source of truth for all trading operations.

    This consolidates Live Workflow, Continuous Trading, and 24/6 operations into
    one bullet-proof system with multiple operating modes.
    """

    def __init__(self,
                 mode: WorkflowMode = WorkflowMode.HYBRID,
                 enable_discord: bool = True,
                 enable_health_monitoring: bool = True,
                 symbols: Optional[List[str]] = None):
        """
        Initialize the unified workflow orchestrator.

        Args:
            mode: Operating mode (ANALYSIS, EXECUTION, HYBRID, BACKTEST)
            enable_discord: Whether to enable Discord integration
            enable_health_monitoring: Whether to enable health monitoring
            symbols: List of symbols to trade (default: ['SPY'])
        """
        self.mode = mode
        self.enable_discord = enable_discord
        self.enable_health_monitoring = enable_health_monitoring
        self.symbols = symbols or ['SPY']

        # Core components
        self.a2a_protocol = None
        self.agents = {}
        self.redis_cache = None
        self.alert_manager = None

        # State tracking
        self.initialized = False
        self.running = False

        logger.info(f"Initialized UnifiedWorkflowOrchestrator in {mode.value} mode")

    async def initialize(self) -> bool:
        """
        Initialize all components and agents.

        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("🚀 Initializing Unified Workflow Orchestrator...")

            # Initialize Redis cache (no fallbacks)
            self.redis_cache = RedisCacheManager()
            # RedisCacheManager initializes automatically in __init__
            logger.info("✅ Redis cache initialized")

            # Initialize alert manager
            self.alert_manager = get_alert_manager()
            logger.info("✅ Alert manager initialized")

            # Initialize A2A protocol
            self.a2a_protocol = A2AProtocol()
            logger.info("✅ A2A protocol initialized")

            # Initialize agents
            await self._initialize_agents()
            logger.info("✅ All agents initialized")

            # Health monitoring (if enabled)
            if self.enable_health_monitoring:
                await self._start_health_monitoring()
                logger.info("✅ Health monitoring started")

            self.initialized = True
            logger.info("🎯 Unified Workflow Orchestrator initialization complete!")
            return True

        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _initialize_agents(self):
        """Initialize all trading agents."""
        # Create agents
        self.agents['macro'] = MacroAgent(a2a_protocol=self.a2a_protocol)
        self.agents['data'] = DataAgent(a2a_protocol=self.a2a_protocol)
        self.agents['strategy'] = StrategyAgent(a2a_protocol=self.a2a_protocol)
        self.agents['risk'] = RiskAgent(a2a_protocol=self.a2a_protocol)
        self.agents['execution'] = ExecutionAgent(a2a_protocol=self.a2a_protocol)
        self.agents['reflection'] = ReflectionAgent(a2a_protocol=self.a2a_protocol)
        self.agents['learning'] = LearningAgent(a2a_protocol=self.a2a_protocol)

        # Register agents with A2A protocol
        for name, agent in self.agents.items():
            self.a2a_protocol.register_agent(name, agent)
            logger.info(f"📋 Registered agent: {name}")

        # Initialize each agent
        for name, agent in self.agents.items():
            try:
                if hasattr(agent, 'initialize'):
                    await agent.initialize()
                logger.info(f"✅ Agent {name} initialized")
            except Exception as e:
                logger.warning(f"⚠️ Agent {name} initialization failed: {e}")

    async def _start_health_monitoring(self):
        """Start health monitoring systems."""
        try:
            from src.utils.api_health_monitor import start_health_monitoring
            start_health_monitoring(check_interval=300)  # Check every 5 minutes
            logger.info("🏥 API health monitoring started")
        except Exception as e:
            logger.warning(f"⚠️ Health monitoring failed to start: {e}")

    async def start(self):
        """Start the orchestrator workflow."""
        if not self.initialized:
            logger.error("❌ Cannot start: orchestrator not initialized")
            return

        logger.info(f"🎯 Starting {self.mode.value} mode workflow...")
        self.running = True

        try:
            if self.mode == WorkflowMode.HYBRID:
                await self._run_hybrid_workflow()
            elif self.mode == WorkflowMode.ANALYSIS:
                await self._run_analysis_workflow()
            elif self.mode == WorkflowMode.EXECUTION:
                await self._run_execution_workflow()
            elif self.mode == WorkflowMode.BACKTEST:
                await self._run_backtest_workflow()

        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False

    async def _run_hybrid_workflow(self):
        """Run hybrid analysis + execution workflow using A2A orchestration."""
        logger.info("🔄 Starting hybrid workflow (analysis + execution)")

        assert self.a2a_protocol is not None, "A2A protocol not initialized"

        while self.running:
            try:
                # Use A2A protocol's built-in orchestration
                logger.info("🚀 Running complete A2A orchestration workflow")
                initial_data = {'symbols': self.symbols, 'mode': 'hybrid'}
                result = await self.a2a_protocol.run_orchestration(initial_data)

                logger.info(f"✅ Orchestration completed: {result}")

                # Wait before next cycle
                logger.info("⏱️ Waiting 15 minutes before next cycle...")
                await asyncio.sleep(900)  # 15 minutes

            except Exception as e:
                logger.error(f"❌ Hybrid workflow cycle failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error



    async def _run_analysis_workflow(self):
        """Run analysis-only workflow."""
        logger.info("🔍 Running analysis-only workflow")
        assert self.a2a_protocol is not None, "A2A protocol not initialized"
        while self.running:
            try:
                initial_data = {'symbols': self.symbols, 'mode': 'analysis'}
                result = await self.a2a_protocol.run_orchestration(initial_data)
                logger.info(f"✅ Analysis orchestration completed: {result}")
                await asyncio.sleep(900)  # 15 minutes
            except Exception as e:
                logger.error(f"❌ Analysis workflow failed: {e}")
                await asyncio.sleep(300)

    async def _run_execution_workflow(self):
        """Run execution-only workflow."""
        logger.info("💰 Running execution-only workflow")
        assert self.a2a_protocol is not None, "A2A protocol not initialized"
        while self.running:
            try:
                initial_data = {'symbols': self.symbols, 'mode': 'execution'}
                result = await self.a2a_protocol.run_orchestration(initial_data)
                logger.info(f"✅ Execution orchestration completed: {result}")
                await asyncio.sleep(300)  # 5 minutes
            except Exception as e:
                logger.error(f"❌ Execution workflow failed: {e}")
                await asyncio.sleep(300)

    async def _run_backtest_workflow(self):
        """Run backtesting workflow."""
        logger.info("📈 Running backtesting workflow")
        logger.info("Backtesting mode not yet implemented")

    async def stop(self):
        """Stop the orchestrator."""
        logger.info("🛑 Stopping Unified Workflow Orchestrator...")
        self.running = False
        logger.info("✅ Orchestrator stopped")

def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified Workflow Orchestrator")
    parser.add_argument(
        '--mode',
        choices=['analysis', 'execution', 'hybrid', 'backtest'],
        default='hybrid',
        help='Operating mode'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['SPY'],
        help='Symbols to trade/analyze'
    )
    parser.add_argument(
        '--no-discord',
        action='store_true',
        help='Disable Discord integration'
    )

    args = parser.parse_args()

    mode_map = {
        'analysis': WorkflowMode.ANALYSIS,
        'execution': WorkflowMode.EXECUTION,
        'hybrid': WorkflowMode.HYBRID,
        'backtest': WorkflowMode.BACKTEST
    }

    orchestrator = UnifiedWorkflowOrchestrator(
        mode=mode_map[args.mode],
        enable_discord=not args.no_discord,
        symbols=args.symbols
    )

    # Run initialization and start
    asyncio.run(orchestrator.initialize())
    asyncio.run(orchestrator.start())

if __name__ == "__main__":
    main()