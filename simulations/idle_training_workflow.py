import asyncio
import logging
import schedule
import time
import psutil
from datetime import datetime, timezone
from typing import Dict, Any

# Add project root to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.learning import LearningAgent
from simulations.historical_agent_backtesting import HistoricalBacktestingOrchestrator
from src.agents.execution_tools import get_exchange_calendars_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/idle_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IdleTrainingWorkflow:
    """
    Manages adaptive simulation-based training during idle periods.
    Runs only when markets are closed, system load is low, and no live trading is active.
    """

    def __init__(self):
        self.learning_agent = LearningAgent()
        self.calendar_tool = get_exchange_calendars_tool()
        self.config = {
            'symbols': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'],  # Configurable symbols
            'initial_capital': 100000,
            'max_cpu_usage': 50,  # % - Threshold for low load
            'max_memory_usage': 70,  # % - Threshold for low load
            'simulation_period_days': 365,  # Lookback for historical data
            'check_interval_minutes': 60  # How often to check for idle conditions
        }
        self.last_run = None

    def is_safe_to_run(self) -> bool:
        """Check if conditions are safe for running simulations (idle, low load, market closed)."""
        # Check market status
        market_status = self.calendar_tool.is_market_open()
        if market_status.get('market_open', True):
            logger.info("Market is open - skipping simulation to avoid interference.")
            return False

        # Check system load
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        if cpu_usage > self.config['max_cpu_usage'] or memory_usage > self.config['max_memory_usage']:
            logger.info(f"High system load (CPU: {cpu_usage}%, Memory: {memory_usage}%) - skipping.")
            return False

        # Check for active live sessions (simplified: check for running terminals or processes)
        # TODO: Enhance with actual check for live_workflow_orchestrator activity if needed
        active_terminals = self._check_active_terminals()
        if active_terminals:
            logger.info("Active terminals detected - possible live trading; skipping.")
            return False

        # Check time since last run (e.g., run at most once per day)
        if self.last_run and (datetime.now(timezone.utc) - self.last_run).days < 1:
            logger.info("Simulation already ran today - skipping.")
            return False

        return True

    def _check_active_terminals(self) -> bool:
        """Placeholder: Check for active PowerShell terminals indicating live trading."""
        # In a real setup, query VS Code terminals or processes
        # For now, assume no active if not detected (expand as needed)
        return False  # Replace with actual logic if available

    async def run_simulation_and_training(self):
        """Run historical simulation and feed results to LearningAgent for training."""
        try:
            logger.info("Starting idle simulation and training...")

            # Set up dates for recent historical data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=self.config['simulation_period_days'])).strftime('%Y-%m-%d')

            # Initialize and run backtesting orchestrator
            orchestrator = HistoricalBacktestingOrchestrator(
                symbols=self.config['symbols'],
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.config['initial_capital']
            )
            await orchestrator.initialize_system()
            await orchestrator.fetch_historical_data()
            sim_results = await orchestrator.run_historical_backtest()

            # Feed simulation results to LearningAgent for training
            performance_data = self._extract_performance_data(sim_results)
            training_results = self.learning_agent.train_strategy_predictor(performance_data)

            # Log and save results
            output_file = f"data/idle_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                import json
                json.dump({'simulation': sim_results, 'training': training_results}, f, indent=2, default=str)
            logger.info(f"Training completed. Results saved to {output_file}")

            self.last_run = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Error in simulation/training: {e}")

    def _extract_performance_data(self, sim_results: Dict[str, Any]) -> list[Dict[str, Any]]:
        """Extract relevant data from simulation for training."""
        # Adapt based on actual sim_results structure
        perf_metrics = sim_results.get('performance_metrics', {})
        return [{
            'total_return': perf_metrics.get('total_return', 0),
            'sharpe_ratio': perf_metrics.get('sharpe_ratio', 0),
            'max_drawdown': perf_metrics.get('max_drawdown', 0),
            # Add more features as needed
        }]

    def start_scheduler(self):
        """Start the adaptive scheduler to check and run periodically."""
        async def job():
            if self.is_safe_to_run():
                await self.run_simulation_and_training()

        schedule.every(self.config['check_interval_minutes']).minutes.do(lambda: asyncio.run(job()))

        logger.info(f"Scheduler started. Checking every {self.config['check_interval_minutes']} minutes.")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    workflow = IdleTrainingWorkflow()
    workflow.start_scheduler()
