# [LABEL:TOOL:trading] [LABEL:TOOL:continuous] [LABEL:FRAMEWORK:asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Continuous paper trading system for 24/7 market monitoring and execution
# Dependencies: StrategyAgent, RiskAgent, ExecutionAgent, exchange calendars
# Related: tools/start_continuous_trading.bat, docs/IMPLEMENTATION/IBKR_PAPER_TRADING_DEPLOYMENT.md
#
#!/usr/bin/env python3
"""
ABC Application Continuous Paper Trading System
Runs all day during market hours, continuously monitoring and executing trades
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import signal
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.strategy import StrategyAgent
from src.agents.execution import ExecutionAgent
from src.agents.execution_tools import get_exchange_calendars_tool

# Lazy import for RiskAgent to avoid TensorFlow import issues
RiskAgent = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ContinuousTradingSystem:
    """Continuous paper trading system that runs during market hours"""

    def __init__(self):
        self.running = False
        self.strategy_agent = None
        self.risk_agent = None
        self.execution_agent = None
        self.exchange_calendars = get_exchange_calendars_tool()
        self.trading_stats = {
            'start_time': None,
            'trades_executed': 0,
            'trades_rejected': 0,
            'total_pnl': 0.0,
            'cycles_completed': 0
        }

    async def initialize_agents(self):
        """Initialize all trading agents"""
        logger.info("Initializing trading agents...")

        try:
            # Initialize agents (execution agent handles IBKR connection)
            self.execution_agent = ExecutionAgent(historical_mode=False)

            # Try to initialize RiskAgent with lazy import
            try:
                global RiskAgent
                if RiskAgent is None:
                    from src.agents.risk import RiskAgent as _RiskAgent
                    RiskAgent = _RiskAgent
                self.risk_agent = RiskAgent()
                logger.info("✅ RiskAgent initialized")
            except Exception as e:
                logger.warning(f"⚠️ RiskAgent initialization failed (likely TensorFlow import issue): {e}")
                logger.warning("Continuing without RiskAgent - all trades will be auto-approved")
                self.risk_agent = None

            self.strategy_agent = StrategyAgent()

            logger.info("✅ All agents initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
            return False

    async def check_market_hours(self):
        """Check if market is currently open"""
        try:
            market_status = self.exchange_calendars.is_market_open()
            is_open = market_status.get('market_open', False)

            if is_open:
                logger.info("MARKET: Market is OPEN - trading active")
            else:
                logger.info("MARKET: Market is CLOSED - waiting for open")

            return is_open

        except Exception as e:
            logger.warning(f"Error checking market hours: {e}")
            return False

    async def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            cycle_start = time.time()

            # Step 1: Strategy Agent generates trading ideas
            logger.info("STRATEGY: Step 1: Strategy Agent analyzing opportunities...")
            strategy_signals = await self.strategy_agent.analyze_market_opportunities()

            if not strategy_signals:
                logger.info("STATUS: No trading opportunities found this cycle")
                return

            # Step 2: Risk Agent evaluates and filters signals (if available)
            if self.risk_agent:
                logger.info("RISK: Step 2: Risk Agent evaluating signals...")
                approved_signals = []

                for signal in strategy_signals:
                    risk_assessment = await self.risk_agent.evaluate_trade_signal(signal)
                    if risk_assessment.get('approved', False):
                        approved_signals.append({
                            **signal,
                            'risk_assessment': risk_assessment
                        })
                    else:
                        logger.info(f"REJECTED: Signal rejected: {signal.get('symbol')} - {risk_assessment.get('reason', 'Unknown')}")
            else:
                logger.info("RISK: Step 2: Risk Agent not available - auto-approving all signals")
                approved_signals = strategy_signals  # Auto-approve all signals

            if not approved_signals:
                logger.info("STATUS: No signals passed risk assessment")
                return

            # Step 3: Execution Agent processes approved trades
            logger.info("EXECUTE: Step 3: Execution Agent processing trades...")
            for approved_signal in approved_signals:
                try:
                    execution_result = await self.execution_agent.process_input(approved_signal)

                    if execution_result.get('executed', False):
                        self.trading_stats['trades_executed'] += 1
                        pnl = execution_result.get('total_value', 0) * approved_signal.get('roi_estimate', 0)
                        self.trading_stats['total_pnl'] += pnl

                        logger.info(f"SUCCESS: Trade executed: {approved_signal['symbol']} x{approved_signal['quantity']} @ ${execution_result.get('price', 0):.2f}")
                    else:
                        self.trading_stats['trades_rejected'] += 1
                        reason = execution_result.get('reason', 'Unknown')
                        logger.info(f"FAILED: Trade rejected: {approved_signal['symbol']} - {reason}")

                except Exception as e:
                    logger.error(f"Error executing trade for {approved_signal.get('symbol')}: {e}")

            # Update cycle stats
            self.trading_stats['cycles_completed'] += 1
            cycle_time = time.time() - cycle_start
            logger.info(f"COMPLETE: Cycle completed in {cycle_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    async def print_status_report(self):
        """Print comprehensive status report"""
        runtime = time.time() - self.trading_stats['start_time'] if self.trading_stats['start_time'] else 0
        hours = runtime / 3600

        logger.info("=" * 60)
        logger.info("TRADING STATUS REPORT")
        logger.info("=" * 60)
        logger.info(f"Runtime: {hours:.1f} hours")
        logger.info(f"Trading Cycles: {self.trading_stats['cycles_completed']}")
        logger.info(f"Trades Executed: {self.trading_stats['trades_executed']}")
        logger.info(f"Trades Rejected: {self.trading_stats['trades_rejected']}")
        logger.info(f"Estimated P&L: ${self.trading_stats['total_pnl']:.2f}")

        if self.trading_stats['trades_executed'] > 0:
            win_rate = self.trading_stats['trades_executed'] / (self.trading_stats['trades_executed'] + self.trading_stats['trades_rejected']) * 100
            logger.info(f"Win Rate: {win_rate:.1f}%")

        logger.info("=" * 60)

    async def run_continuous_trading(self):
        """Main continuous trading loop"""
        logger.info("🚀 Starting ABC Application Continuous Paper Trading System")
        logger.info("=" * 60)

        self.running = True
        self.trading_stats['start_time'] = time.time()

        # Handle graceful shutdown
        def signal_handler(signum, frame):
            logger.info("SHUTDOWN: Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Initialize agents
            if not await self.initialize_agents():
                logger.error("INIT_FAILED: Failed to initialize agents - shutting down")
                return

            # Main trading loop
            cycle_interval = 300  # 5 minutes between cycles
            status_interval = 3600  # Status report every hour
            last_status_time = time.time()

            logger.info("START: Starting continuous trading loop...")
            logger.info(f"INTERVAL: Cycle interval: {cycle_interval} seconds")
            logger.info(f"REPORT: Status reports: Every {status_interval // 60} minutes")
            logger.info("=" * 60)

            while self.running:
                try:
                    # Check if market is open
                    market_open = await self.check_market_hours()

                    if market_open:
                        # Run trading cycle
                        await self.run_trading_cycle()
                    else:
                        logger.info("WAIT: Market closed - waiting for market open...")

                    # Print status report periodically
                    current_time = time.time()
                    if current_time - last_status_time >= status_interval:
                        await self.print_status_report()
                        last_status_time = current_time

                    # Wait for next cycle
                    await asyncio.sleep(cycle_interval)

                except Exception as e:
                    logger.error(f"LOOP_ERROR: Error in main loop: {e}")
                    await asyncio.sleep(60)  # Wait a minute before retrying

        except Exception as e:
            logger.error(f"FATAL: Fatal error in continuous trading: {e}")

        finally:
            # Final status report
            await self.print_status_report()
            logger.info("END: Continuous trading system shut down")

    async def run_daily_schedule(self):
        """Run the system according to market schedule"""
        logger.info("SCHEDULE: Starting daily trading schedule...")

        while True:
            try:
                # Check market status
                market_status = self.exchange_calendars.is_market_open()

                if market_status.get('market_open', False):
                    logger.info("OPEN: Market opened - starting continuous trading")
                    await self.run_continuous_trading()
                    break  # Exit after market closes

                elif market_status.get('next_open'):
                    next_open = market_status['next_open']
                    wait_seconds = (next_open - datetime.now(timezone.utc)).total_seconds()

                    if wait_seconds > 0:
                        logger.info(f"WAIT: Market closed - waiting {wait_seconds/3600:.1f} hours until market open")
                        await asyncio.sleep(min(wait_seconds, 3600))  # Wait up to 1 hour, then recheck
                    else:
                        # Market should be open, start trading
                        logger.info("OPEN: Market should be open - starting continuous trading")
                        await self.run_continuous_trading()
                        break

                else:
                    logger.warning("UNKNOWN: Unable to determine market schedule - waiting 1 hour")
                    await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"SCHEDULE_ERROR: Error in daily schedule: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

async def main():
    """Main entry point"""
    system = ContinuousTradingSystem()

    # Check if we should run continuously or follow market schedule
    run_continuous = os.getenv('RUN_CONTINUOUS', 'false').lower() == 'true'

    if run_continuous:
        logger.info("🔄 Running in CONTINUOUS mode (ignores market hours)")
        await system.run_continuous_trading()
    else:
        logger.info("📅 Running in SCHEDULED mode (follows market hours)")
        await system.run_daily_schedule()

if __name__ == "__main__":
    # Set environment variable to run continuously for testing
    os.environ['RUN_CONTINUOUS'] = 'true'  # Uncomment for testing

    asyncio.run(main())