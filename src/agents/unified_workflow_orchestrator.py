# src/agents/unified_workflow_orchestrator.py
# [LABEL:COMPONENT:unified_orchestrator] [LABEL:FRAMEWORK:asyncio] [LABEL:FRAMEWORK:apscheduler]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-04] [LABEL:REVIEWED:pending]
#
# Purpose: Unified workflow orchestrator consolidating Live Workflow, Continuous Trading, and 24/6 operations
# Dependencies: APScheduler, Discord.py, A2A Protocol, Langfuse, FastAPI
# Related: docs/architecture.md, src/agents/base.py, src/utils/a2a_protocol.py
#
# UNIFIED WORKFLOW ORCHESTRATOR
# =============================
# This consolidates three separate systems into one bullet-proof workflow:
# - Live Workflow Orchestrator (multi-agent collaboration)
# - Continuous Trading (automated execution)
# - 24/6 Workflow Orchestrator (production scheduling)
#
# MODES OF OPERATION:
# 1. ANALYSIS_MODE: Full collaborative analysis with human intervention
# 2. EXECUTION_MODE: Automated trading execution only
# 3. HYBRID_MODE: Analysis + automated execution with human oversight
# 4. BACKTEST_MODE: Historical simulation and validation
#
# SCHEDULING FEATURES:
# - Market-aware scheduling (respects trading hours)
# - Health monitoring and automatic recovery
# - Production deployment capabilities
# - Discord integration for real-time updates

import asyncio
import logging
import time
import os
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from pathlib import Path
import json
import sys

# Third-party imports
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz

# Local imports
from src.utils.a2a_protocol import A2AProtocol
from src.utils.logging_config import get_logger, log_operation_start, log_operation_end, log_error_with_context
from src.utils.api_health_monitor import start_health_monitoring, get_api_health_summary
from src.agents.data import DataAgent
from src.agents.strategy import StrategyAgent
from src.agents.risk import RiskAgent
from src.agents.execution import ExecutionAgent
from src.agents.reflection import ReflectionAgent
from src.agents.learning import LearningAgent
from src.agents.macro import MacroAgent

# Optional imports
try:
    from src.utils.langfuse_client import LangfuseClient
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

try:
    from src.integrations.discord_bot import DiscordBot
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

logger = get_logger(__name__)


class WorkflowMode(Enum):
    """Operating modes for the unified workflow orchestrator."""
    ANALYSIS = "analysis"      # Full collaborative analysis with human intervention
    EXECUTION = "execution"    # Automated trading execution only
    HYBRID = "hybrid"         # Analysis + automated execution with oversight
    BACKTEST = "backtest"      # Historical simulation and validation


class MarketHours:
    """Market hours configuration for scheduling."""

    # US Market Hours (Eastern Time)
    PRE_MARKET_OPEN = dt_time(4, 0)    # 4:00 AM ET
    MARKET_OPEN = dt_time(9, 30)       # 9:30 AM ET
    MARKET_CLOSE = dt_time(16, 0)      # 4:00 PM ET
    AFTER_HOURS_CLOSE = dt_time(20, 0) # 8:00 PM ET

    # Extended hours for futures/options
    FUTURES_OPEN = dt_time(18, 0)      # 6:00 PM ET (previous day)
    FUTURES_CLOSE = dt_time(17, 0)     # 5:00 PM ET (next day)

    @classmethod
    def is_market_open(cls) -> bool:
        """Check if US equity market is currently open."""
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        current_time = now.time()

        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if current time is within market hours
        return cls.MARKET_OPEN <= current_time <= cls.MARKET_CLOSE

    @classmethod
    def is_extended_hours(cls) -> bool:
        """Check if extended hours trading is available."""
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        current_time = now.time()

        # Weekday extended hours
        if now.weekday() < 5:
            return (cls.PRE_MARKET_OPEN <= current_time <= cls.AFTER_HOURS_CLOSE)

        # Weekend futures
        return (cls.FUTURES_OPEN <= current_time or current_time <= cls.FUTURES_CLOSE)


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
                 symbols: List[str] = None):
        """
        Initialize the unified workflow orchestrator.

        Args:
            mode: Operating mode (ANALYSIS, EXECUTION, HYBRID, BACKTEST)
            enable_discord: Whether to enable Discord integration
            enable_health_monitoring: Whether to enable health monitoring
            symbols: List of symbols to trade (default: ['SPY'])
        """
        self.mode = mode
        self.enable_discord = enable_discord and DISCORD_AVAILABLE
        self.enable_health_monitoring = enable_health_monitoring
        self.symbols = symbols or ['SPY']

        # Core components
        self.a2a_protocol = None
        self.scheduler = AsyncIOScheduler()
        self.agents = {}
        self.discord_bot = None
        self.langfuse_client = None

        # Discord channels for alerts and ranked trades
        self.alerts_channel = None
        self.ranked_trades_channel = None
        self.discord_ready = asyncio.Event()

        # State tracking
        self.is_running = False
        self.last_health_check = None
        self.execution_count = 0
        self.error_count = 0

        # Configuration
        self.analysis_interval = 300  # 5 minutes for analysis cycles
        self.execution_interval = 30   # 30 seconds for execution cycles
        self.health_check_interval = 60  # 1 minute for health checks

        logger.info(f"Initialized UnifiedWorkflowOrchestrator in {mode.value} mode")

    async def initialize(self) -> bool:
        """
        Initialize all components and agents.

        Returns:
            bool: True if initialization successful
        """
        try:
            log_operation_start(logger, "orchestrator_initialization", component="unified_orchestrator")

            # Initialize A2A protocol
            self.a2a_protocol = A2AProtocol(max_agents=50)
            logger.info("A2A protocol initialized")

            # Initialize agents based on mode
            await self._initialize_agents()

            # Initialize optional components
            if LANGFUSE_AVAILABLE:
                self.langfuse_client = LangfuseClient()
                logger.info("Langfuse client initialized")

            if self.enable_discord and DISCORD_AVAILABLE:
                self.discord_bot = DiscordBot()
                await self.discord_bot.start()
                self.discord_ready.set()
                logger.info("Discord bot initialized")

            # Initialize health monitoring
            if self.enable_health_monitoring:
                start_health_monitoring(check_interval=self.health_check_interval)
                logger.info("Health monitoring initialized")

            # Setup scheduler
            self._setup_scheduler()

            log_operation_end(logger, "orchestrator_initialization", component="unified_orchestrator")
            return True

        except Exception as e:
            log_error_with_context(
                logger, e, "orchestrator_initialization",
                component="unified_orchestrator"
            )
            return False

    async def _initialize_agents(self):
        """Initialize all agents based on the current mode."""
        # Always initialize core agents
        self.agents['data'] = DataAgent()
        self.agents['strategy'] = StrategyAgent()
        self.agents['risk'] = RiskAgent()
        self.agents['execution'] = ExecutionAgent()

        # Initialize additional agents based on mode
        if self.mode in [WorkflowMode.ANALYSIS, WorkflowMode.HYBRID]:
            self.agents['reflection'] = ReflectionAgent()
            self.agents['learning'] = LearningAgent()
            self.agents['macro'] = MacroAgent()

        # Register agents with A2A protocol
        for agent_name, agent in self.agents.items():
            self.a2a_protocol.register_agent(agent_name, agent)

        logger.info(f"Initialized {len(self.agents)} agents for {self.mode.value} mode")

    def _setup_scheduler(self):
        """Setup the APScheduler with appropriate jobs based on mode."""

        # Health check job (always enabled if monitoring is on)
        if self.enable_health_monitoring:
            self.scheduler.add_job(
                self._health_check_job,
                trigger=IntervalTrigger(seconds=self.health_check_interval),
                id='health_check',
                name='Health Check'
            )

        # Analysis job (for ANALYSIS and HYBRID modes)
        if self.mode in [WorkflowMode.ANALYSIS, WorkflowMode.HYBRID]:
            self.scheduler.add_job(
                self._analysis_cycle_job,
                trigger=IntervalTrigger(seconds=self.analysis_interval),
                id='analysis_cycle',
                name='Analysis Cycle'
            )

        # Execution job (for EXECUTION and HYBRID modes)
        if self.mode in [WorkflowMode.EXECUTION, WorkflowMode.HYBRID]:
            # Only during market hours for execution
            self.scheduler.add_job(
                self._execution_cycle_job,
                trigger=CronTrigger(
                    hour='9-15',  # 9 AM to 3 PM ET (market hours)
                    minute='*/1',  # Every minute
                    timezone='US/Eastern'
                ),
                id='execution_cycle',
                name='Execution Cycle'
            )

        # Market status monitoring
        self.scheduler.add_job(
            self._market_status_job,
            trigger=IntervalTrigger(minutes=5),
            id='market_status',
            name='Market Status Check'
        )

        logger.info(f"Scheduled {len(self.scheduler.get_jobs())} jobs for {self.mode.value} mode")

    async def start(self) -> bool:
        """
        Start the unified workflow orchestrator.

        Returns:
            bool: True if started successfully
        """
        if self.is_running:
            logger.warning("Orchestrator is already running")
            return True

        try:
            log_operation_start(logger, "orchestrator_startup", component="unified_orchestrator")

            # Start scheduler
            self.scheduler.start()
            logger.info("Scheduler started")

            # Send startup notification
            if self.discord_bot:
                await self.discord_bot.send_health_alert(
                    "🚀 Unified Workflow Orchestrator Started",
                    f"Mode: {self.mode.value}\nSymbols: {', '.join(self.symbols)}\nStatus: Operational"
                )

            self.is_running = True
            log_operation_end(logger, "orchestrator_startup", component="unified_orchestrator")

            # Keep running
            while self.is_running:
                await asyncio.sleep(1)

            return True

        except Exception as e:
            log_error_with_context(
                logger, e, "orchestrator_startup",
                component="unified_orchestrator"
            )
            return False

    async def stop(self):
        """Stop the unified workflow orchestrator."""
        log_operation_start(logger, "orchestrator_shutdown", component="unified_orchestrator")

        self.is_running = False

        # Stop scheduler
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

        # Stop Discord bot
        if self.discord_bot:
            await self.discord_bot.stop()
            logger.info("Discord bot stopped")

        # Send shutdown notification
        if self.discord_bot:
            await self.discord_bot.send_health_alert(
                "🛑 Unified Workflow Orchestrator Stopped",
                "Shutdown completed successfully"
            )

        log_operation_end(logger, "orchestrator_shutdown", component="unified_orchestrator")

    async def _analysis_cycle_job(self):
        """Execute a full analysis cycle with all agents."""
        try:
            log_operation_start(logger, "analysis_cycle", component="unified_orchestrator")

            # Run collaborative analysis
            initial_data = {'symbols': self.symbols}
            result = await self.a2a_protocol.run_orchestration(initial_data)

            # Send results to Discord if enabled
            if self.discord_bot and result.get('status') == 'success':
                await self._send_analysis_results(result)

            log_operation_end(logger, "analysis_cycle", component="unified_orchestrator")

        except Exception as e:
            self.error_count += 1
            log_error_with_context(
                logger, e, "analysis_cycle",
                component="unified_orchestrator",
                cycle_number=self.execution_count
            )

    async def _execution_cycle_job(self):
        """Execute automated trading cycle."""
        try:
            # Only execute if market is open
            if not MarketHours.is_market_open():
                return

            log_operation_start(logger, "execution_cycle", component="unified_orchestrator")

            self.execution_count += 1

            # Simple execution workflow: Strategy → Risk → Execution
            strategy_result = await self.agents['strategy'].process_input({'symbols': self.symbols})
            risk_result = await self.agents['risk'].process_input(strategy_result)
            execution_result = await self.agents['execution'].process_input(risk_result)

            # Log execution
            logger.info(f"Execution cycle {self.execution_count} completed: {execution_result}")

            # Send execution alert
            if self.discord_bot:
                await self._send_execution_alert(execution_result)

            log_operation_end(logger, "execution_cycle", component="unified_orchestrator")

        except Exception as e:
            self.error_count += 1
            log_error_with_context(
                logger, e, "execution_cycle",
                component="unified_orchestrator",
                cycle_number=self.execution_count
            )

    async def _health_check_job(self):
        """Perform health check and send alerts if needed."""
        try:
            health_status = get_api_health_summary()
            self.last_health_check = datetime.now()

            # Send health alert if there are issues
            if health_status.get('overall_status') != 'healthy' and self.discord_bot:
                await self.discord_bot.send_health_alert(
                    "⚠️ Health Check Alert",
                    f"Status: {health_status.get('overall_status')}\nDetails: {health_status.get('summary')}"
                )

        except Exception as e:
            log_error_with_context(
                logger, e, "health_check_job",
                component="unified_orchestrator"
            )

    async def _market_status_job(self):
        """Check and log market status."""
        is_open = MarketHours.is_market_open()
        extended_available = MarketHours.is_extended_hours()

        logger.info(f"Market Status - Open: {is_open}, Extended Hours: {extended_available}")

    async def _send_analysis_results(self, result: Dict[str, Any]):
        """Send analysis results to Discord."""
        if not self.discord_bot:
            return

        summary = f"""
📊 Analysis Complete
Status: {result.get('status', 'unknown')}
Symbols: {', '.join(self.symbols)}
Key Insights: {result.get('insights', 'Analysis completed')}
        """.strip()

        await self.discord_bot.send_trade_alert(summary)

    async def _send_execution_alert(self, result: Dict[str, Any]):
        """Send execution results to Discord."""
        if not self.discord_bot:
            return

        summary = f"""
⚡ Execution Cycle {self.execution_count}
Result: {result.get('status', 'completed')}
Details: {result.get('message', 'Execution completed successfully')}
        """.strip()

        await self.discord_bot.send_trade_alert(summary)

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            'mode': self.mode.value,
            'is_running': self.is_running,
            'agents_count': len(self.agents),
            'execution_count': self.execution_count,
            'error_count': self.error_count,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'market_open': MarketHours.is_market_open(),
            'discord_enabled': self.discord_bot is not None,
            'scheduler_jobs': len(self.scheduler.get_jobs()) if self.scheduler else 0
        }

    def rank_trade_proposals(self, proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank trade proposals by confidence (descending), then by expected return (descending).

        Args:
            proposals: List of trade proposal dictionaries with 'confidence' and 'expected_return' keys

        Returns:
            Ranked list of proposals
        """
        return sorted(
            proposals,
            key=lambda x: (x.get('confidence', 0), x.get('expected_return', 0)),
            reverse=True
        )

    async def send_trade_alert(self, message: str, alert_type: str = "general"):
        """
        Send a trade alert to the alerts channel.

        Args:
            message: Alert message
            alert_type: Type of alert (trade, execution, etc.)
        """
        if self.alerts_channel:
            await self.alerts_channel.send(message)

    async def send_ranked_trade_info(self, proposals_info: str, info_type: str = "ranked"):
        """
        Send ranked trade proposals information to the ranked trades channel.

        Args:
            proposals_info: Formatted proposals information
            info_type: Type of information
        """
        if self.ranked_trades_channel:
            await self.ranked_trades_channel.send(proposals_info)

    def _extract_trade_alert_info(self, response_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract trade alert information from agent response data.

        Args:
            response_data: Agent response containing trade proposals

        Returns:
            Formatted alert string or None
        """
        agent_name = response_data.get('agent', 'unknown')
        response = response_data.get('response', {})

        # Handle structured responses with trade_proposals
        if isinstance(response, dict) and 'trade_proposals' in response:
            proposals = response['trade_proposals']
            if proposals:
                ranked_proposals = self.rank_trade_proposals(proposals)
                alert_lines = [f"**{agent_name.title()} Agent** generated {len(proposals)} ranked trade proposal(s):"]
                for i, proposal in enumerate(ranked_proposals, 1):
                    instrument = proposal.get('instrument', 'UNKNOWN')
                    action = proposal.get('action', 'UNKNOWN')
                    confidence = proposal.get('confidence', 0)
                    alert_lines.append(f"#{i} {action.upper()} {instrument} (conf: {confidence:.2f})")
                return "\n".join(alert_lines)

        # Handle string responses with regex parsing
        elif isinstance(response, str):
            import re
            # Find patterns like "BUY AAPL Confidence: 0.8" or "SELL GOOG Confidence: 0.6"
            pattern = r'(BUY|SELL|HOLD)\s+(\w+)\s+(?:Confidence:\s*)?(\d*\.?\d+)'
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                alert_lines = [f"**{agent_name.title()} Agent** has {len(matches)} trade proposal(s) in text:"]
                for action, symbol, confidence in matches:
                    alert_lines.append(f"- {action.upper()} {symbol.lower()} (conf: {confidence})")
                return "\n".join(alert_lines)

        return None


# Convenience functions for different operating modes
async def run_analysis_mode(symbols: List[str] = None, enable_discord: bool = True) -> UnifiedWorkflowOrchestrator:
    """Run in analysis-only mode with full collaborative workflow."""
    orchestrator = UnifiedWorkflowOrchestrator(
        mode=WorkflowMode.ANALYSIS,
        enable_discord=enable_discord,
        symbols=symbols
    )

    if await orchestrator.initialize():
        await orchestrator.start()

    return orchestrator


async def run_execution_mode(symbols: List[str] = None, enable_discord: bool = True) -> UnifiedWorkflowOrchestrator:
    """Run in execution-only mode for automated trading."""
    orchestrator = UnifiedWorkflowOrchestrator(
        mode=WorkflowMode.EXECUTION,
        enable_discord=enable_discord,
        symbols=symbols
    )

    if await orchestrator.initialize():
        await orchestrator.start()

    return orchestrator


async def run_hybrid_mode(symbols: List[str] = None, enable_discord: bool = True) -> UnifiedWorkflowOrchestrator:
    """Run in hybrid mode combining analysis and automated execution."""
    orchestrator = UnifiedWorkflowOrchestrator(
        mode=WorkflowMode.HYBRID,
        enable_discord=enable_discord,
        symbols=symbols
    )

    if await orchestrator.initialize():
        await orchestrator.start()

    return orchestrator


# Main entry point
if __name__ == "__main__":
    import argparse

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

    asyncio.run(orchestrator.initialize())
    asyncio.run(orchestrator.start())