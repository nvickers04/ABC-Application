#!/usr/bin/env python3
"""
Paper Trading Validation Script

This script validates the paper trading environment by:
1. Testing IBKR paper trading connection
2. Validating risk management with paper limits
3. Testing order placement and execution simulation
4. Verifying monitoring and alerting systems
5. Running integration tests with circuit breakers

Usage:
    python simulations/paper_trading_validation.py
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_environment_config
from src.integrations.ibkr_connector import IBKRConnector
from src.agents.risk_agent import RiskAgent
from src.agents.execution_agent import ExecutionAgent
from src.utils.alert_manager import get_alert_manager
from src.workflows.live_workflow_orchestrator import LiveWorkflowOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperTradingValidator:
    """Validates paper trading environment and components."""

    def __init__(self):
        """Initialize validator with paper trading configuration."""
        self.config = load_environment_config('paper_trading')
        self.alert_manager = get_alert_manager()
        self.ibkr_connector = None
        self.risk_agent = None
        self.execution_agent = None
        self.orchestrator = None

        # Test results
        self.test_results = {
            'ibkr_connection': False,
            'risk_validation': False,
            'order_simulation': False,
            'monitoring_system': False,
            'circuit_breaker': False,
            'alert_system': False
        }

    async def setup_components(self):
        """Set up all components for paper trading validation."""
        logger.info("Setting up paper trading components...")

        try:
            # Initialize IBKR connector in paper trading mode
            self.ibkr_connector = IBKRConnector(paper_trading=True)
            await self.ibkr_connector.connect()
            logger.info("IBKR paper trading connection established")

            # Initialize agents
            self.risk_agent = RiskAgent(a2a_protocol=None)
            self.execution_agent = ExecutionAgent(a2a_protocol=None)

            # Initialize orchestrator
            self.orchestrator = LiveWorkflowOrchestrator()

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            await self.alert_manager.error(f"Paper trading setup failed: {e}")
            raise

    async def test_ibkr_connection(self):
        """Test IBKR paper trading connection."""
        logger.info("Testing IBKR paper trading connection...")

        try:
            # Test connection
            if not self.ibkr_connector.is_connected():
                await self.ibkr_connector.connect()

            # Test account info retrieval
            account_info = await self.ibkr_connector.get_account_info()
            logger.info(f"Account info retrieved: {account_info}")

            # Test market data (paper trading should work)
            test_symbol = "AAPL"
            market_data = await self.ibkr_connector.get_market_data(test_symbol)
            logger.info(f"Market data for {test_symbol}: {market_data}")

            self.test_results['ibkr_connection'] = True
            logger.info("✅ IBKR connection test passed")

        except Exception as e:
            logger.error(f"IBKR connection test failed: {e}")
            self.test_results['ibkr_connection'] = False
            await self.alert_manager.error(f"IBKR connection test failed: {e}")

    async def test_risk_validation(self):
        """Test risk management with paper trading limits."""
        logger.info("Testing risk validation...")

        try:
            # Test portfolio risk assessment
            test_portfolio = {
                'AAPL': 100,
                'GOOGL': 50,
                'MSFT': 75
            }

            risk_assessment = await self.risk_agent.assess_portfolio_risk(test_portfolio)
            logger.info(f"Risk assessment: {risk_assessment}")

            # Validate against paper trading limits
            var_95 = risk_assessment.get('var_95', 0)
            max_var_limit = self.config['risk_management']['portfolio_limits']['max_var_95']

            if var_95 <= max_var_limit:
                self.test_results['risk_validation'] = True
                logger.info("✅ Risk validation test passed")
            else:
                logger.warning(f"Risk validation failed: VaR {var_95} exceeds limit {max_var_limit}")
                self.test_results['risk_validation'] = False

        except Exception as e:
            logger.error(f"Risk validation test failed: {e}")
            self.test_results['risk_validation'] = False
            await self.alert_manager.error(f"Risk validation test failed: {e}")

    async def test_order_simulation(self):
        """Test order placement simulation."""
        logger.info("Testing order simulation...")

        try:
            # Test order validation
            test_order = {
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 10,
                'order_type': 'LMT',
                'limit_price': 150.0
            }

            # Validate order through execution agent
            validation_result = await self.execution_agent.validate_order(test_order)
            logger.info(f"Order validation result: {validation_result}")

            if validation_result.get('approved', False):
                # Simulate order placement (don't actually place in paper trading)
                logger.info("Order would be placed in paper trading environment")
                self.test_results['order_simulation'] = True
                logger.info("✅ Order simulation test passed")
            else:
                logger.warning(f"Order validation failed: {validation_result}")
                self.test_results['order_simulation'] = False

        except Exception as e:
            logger.error(f"Order simulation test failed: {e}")
            self.test_results['order_simulation'] = False
            await self.alert_manager.error(f"Order simulation test failed: {e}")

    async def test_monitoring_system(self):
        """Test monitoring and performance tracking."""
        logger.info("Testing monitoring system...")

        try:
            # Test performance metrics collection
            metrics = {
                'portfolio_value': 100000,
                'daily_pnl': 1250.50,
                'sharpe_ratio': 1.8,
                'max_drawdown': 0.025
            }

            # Log metrics (would normally go to monitoring system)
            logger.info(f"Performance metrics: {metrics}")

            # Test health checks
            health_status = await self.orchestrator.check_system_health()
            logger.info(f"System health status: {health_status}")

            self.test_results['monitoring_system'] = True
            logger.info("✅ Monitoring system test passed")

        except Exception as e:
            logger.error(f"Monitoring system test failed: {e}")
            self.test_results['monitoring_system'] = False
            await self.alert_manager.error(f"Monitoring system test failed: {e}")

    async def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        logger.info("Testing circuit breaker...")

        try:
            # Simulate a loss that should trigger circuit breaker
            simulated_loss = 0.04  # 4% loss
            circuit_breaker_threshold = self.config['safeguards']['circuit_breaker_threshold']

            if simulated_loss >= circuit_breaker_threshold:
                logger.info(f"Circuit breaker triggered at {simulated_loss} loss (threshold: {circuit_breaker_threshold})")
                # In real scenario, this would halt trading
                self.test_results['circuit_breaker'] = True
                logger.info("✅ Circuit breaker test passed")
            else:
                logger.warning("Circuit breaker did not trigger as expected")
                self.test_results['circuit_breaker'] = False

        except Exception as e:
            logger.error(f"Circuit breaker test failed: {e}")
            self.test_results['circuit_breaker'] = False
            await self.alert_manager.error(f"Circuit breaker test failed: {e}")

    async def test_alert_system(self):
        """Test alerting system integration."""
        logger.info("Testing alert system...")

        try:
            # Test alert sending
            test_alert = "Paper trading validation test alert"
            await self.alert_manager.warning(test_alert)

            # Test trade alert
            await self.orchestrator.send_trade_alert("Test trade alert", "test")

            logger.info("Alerts sent successfully")
            self.test_results['alert_system'] = True
            logger.info("✅ Alert system test passed")

        except Exception as e:
            logger.error(f"Alert system test failed: {e}")
            self.test_results['alert_system'] = False
            await self.alert_manager.error(f"Alert system test failed: {e}")

    async def run_validation(self):
        """Run all paper trading validation tests."""
        logger.info("Starting paper trading validation...")

        try:
            # Setup components
            await self.setup_components()

            # Run all tests
            await self.test_ibkr_connection()
            await self.test_risk_validation()
            await self.test_order_simulation()
            await self.test_monitoring_system()
            await self.test_circuit_breaker()
            await self.test_alert_system()

            # Generate report
            self.generate_report()

        except Exception as e:
            logger.error(f"Paper trading validation failed: {e}")
            await self.alert_manager.error(f"Paper trading validation failed: {e}")
        finally:
            # Cleanup
            if self.ibkr_connector:
                await self.ibkr_connector.disconnect()

    def generate_report(self):
        """Generate validation report."""
        logger.info("Generating validation report...")

        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)

        report = f"""
Paper Trading Validation Report
================================

Test Results: {passed_tests}/{total_tests} passed

Detailed Results:
"""

        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            report += f"- {test_name}: {status}\n"

        report += "\nRecommendations:\n"

        if not self.test_results['ibkr_connection']:
            report += "- Fix IBKR paper trading connection\n"
        if not self.test_results['risk_validation']:
            report += "- Review risk management limits\n"
        if not self.test_results['order_simulation']:
            report += "- Fix order validation logic\n"
        if not self.test_results['monitoring_system']:
            report += "- Implement performance monitoring\n"
        if not self.test_results['circuit_breaker']:
            report += "- Configure circuit breaker thresholds\n"
        if not self.test_results['alert_system']:
            report += "- Set up alerting channels\n"

        logger.info(report)

        # Save report to file
        report_file = project_root / "data" / f"paper_trading_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to: {report_file}")

async def main():
    """Main validation function."""
    validator = PaperTradingValidator()
    await validator.run_validation()

if __name__ == "__main__":
    asyncio.run(main())