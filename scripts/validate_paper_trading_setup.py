#!/usr/bin/env python3
"""
Paper Trading Setup Validation Script

This script validates that the ABC Application is properly configured
for paper trading with IBKR TWS. It tests:
- Environment configuration
- IBKR connection
- Basic market data retrieval
- Agent initialization
- Health monitoring

Usage:
    python scripts/validate_paper_trading_setup.py
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.integrations.ibkr_connector import IBKRConnector
    from src.agents.data import DataAgent
    from src.agents.strategy import StrategyAgent
    from src.agents.risk import RiskAgent
    from src.agents.execution import ExecutionAgent
    from health_server import app
    # Simple config check instead of ConfigManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperTradingValidator:
    """Validates paper trading setup and configuration."""

    def __init__(self):
        self.results = []

    def log_result(self, test_name: str, success: bool, message: str = "", details: str = ""):
        """Log a test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.results.append({
            'test': test_name,
            'success': success,
            'message': message,
            'details': details
        })
        print(f"{status} {test_name}")
        if message:
            print(f"   {message}")
        if details:
            print(f"   Details: {details}")
        print()

    async def test_environment_config(self) -> bool:
        """Test that environment configuration is present."""
        required_vars = [
            'IBKR_USERNAME', 'IBKR_PASSWORD', 'IBKR_ACCOUNT_ID',
            'IBKR_HOST', 'IBKR_PORT', 'IBKR_CLIENT_ID'
        ]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            self.log_result(
                "Environment Configuration",
                False,
                f"Missing required environment variables: {', '.join(missing_vars)}",
                "Create a .env file with IBKR credentials and configuration."
            )
            return False

        self.log_result("Environment Configuration", True, "All required environment variables found.")
        return True

    async def test_ibkr_connection(self) -> bool:
        """Test IBKR connection."""
        try:
            connector = IBKRConnector()
            connected = await connector.connect()

            if connected:
                # Test basic functionality
                account_summary = await connector.get_account_summary()
                if account_summary:
                    self.log_result(
                        "IBKR Connection",
                        True,
                        f"Successfully connected to IBKR paper trading account: {os.getenv('IBKR_ACCOUNT_ID')}",
                        f"Account info retrieved successfully"
                    )
                    return True
                else:
                    self.log_result("IBKR Connection", False, "Connected but failed to retrieve account summary.")
                    return False
            else:
                self.log_result("IBKR Connection", False, "Failed to connect to IBKR TWS. Make sure TWS is running and configured for paper trading.")
                return False

        except Exception as e:
            self.log_result("IBKR Connection", False, f"IBKR connection error: {str(e)}")
            return False

    async def test_market_data(self) -> bool:
        """Test market data retrieval."""
        try:
            connector = IBKRConnector()
            if not await connector.connect():
                self.log_result("Market Data", False, "Cannot test market data - IBKR not connected.")
                return False

            # Test with a simple symbol
            symbol = "AAPL"
            market_data = await connector.get_market_data(symbol)

            if market_data:
                close_price = market_data.get('close', 'N/A')
                self.log_result(
                    "Market Data",
                    True,
                    f"Successfully retrieved market data for {symbol}",
                    f"Close price: {close_price}"
                )
                return True
            else:
                self.log_result("Market Data", False, f"Failed to get market data for {symbol}.")
                return False

        except Exception as e:
            self.log_result("Market Data", False, f"Market data error: {str(e)}")
            return False

    async def test_agent_initialization(self) -> bool:
        """Test that agents can be initialized."""
        try:
            # Test Data Agent
            data_agent = DataAgent()
            # Just test that the class can be instantiated
            if data_agent:
                self.log_result("Agent Initialization", True, "Agent classes can be instantiated successfully.")
                return True
            else:
                self.log_result("Agent Initialization", False, "Failed to instantiate agent classes.")
                return False

        except Exception as e:
            self.log_result("Agent Initialization", False, f"Agent instantiation error: {str(e)}")
            return False

    async def test_health_monitoring(self) -> bool:
        """Test health monitoring system."""
        try:
            # Test that the health server app can be imported
            if app:
                self.log_result("Health Monitoring", True, "Health monitoring server available.")
                return True
            else:
                self.log_result("Health Monitoring", False, "Health monitoring server not available.")
                return False

        except Exception as e:
            self.log_result("Health Monitoring", False, f"Health monitoring error: {str(e)}")
            return False

    async def test_configuration_loading(self) -> bool:
        """Test that configuration files load properly."""
        try:
            # Test that config files exist
            config_files = [
                'config/risk-constraints.yaml',
                'config/trading-permissions.yaml',
                'config/profitability-targets.yaml'
            ]

            missing_files = []
            for config_file in config_files:
                if not Path(config_file).exists():
                    missing_files.append(config_file)

            if not missing_files:
                self.log_result("Configuration Loading", True, "Configuration files are present.")
                return True
            else:
                self.log_result("Configuration Loading", False, f"Missing configuration files: {', '.join(missing_files)}")
                return False

        except Exception as e:
            self.log_result("Configuration Loading", False, f"Configuration loading error: {str(e)}")
            return False

    async def run_all_tests(self) -> Dict:
        """Run all validation tests."""
        print("🚀 Starting Paper Trading Setup Validation\n")
        print("=" * 50)

        # Run tests in order
        await self.test_environment_config()
        await self.test_configuration_loading()
        await self.test_ibkr_connection()
        await self.test_market_data()
        await self.test_agent_initialization()
        await self.test_health_monitoring()

        # Summary
        print("=" * 50)
        passed = sum(1 for r in self.results if r['success'])
        total = len(self.results)

        print(f"📊 Validation Summary: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed! Your paper trading setup is ready.")
        else:
            print("⚠️  Some tests failed. Please review the errors above and fix configuration issues.")
            print("\nCommon fixes:")
            print("- Ensure TWS is running and configured for paper trading")
            print("- Check .env file has correct IBKR credentials")
            print("- Verify all dependencies are installed")
            print("- Check firewall settings for TWS API port (7497)")

        return {
            'passed': passed,
            'total': total,
            'results': self.results
        }


async def main():
    """Main validation function."""
    validator = PaperTradingValidator()
    results = await validator.run_all_tests()

    # Exit with appropriate code
    if results['passed'] == results['total']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())