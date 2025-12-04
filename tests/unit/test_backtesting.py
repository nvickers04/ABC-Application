import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the backtesting components
from src.utils.historical_simulation_engine import (
    SimulationConfig, HistoricalSimulationEngine,
    run_historical_portfolio_simulation, run_multi_strategy_comparison
)


class TestSimulationConfig(unittest.TestCase):
    """Test SimulationConfig dataclass"""

    def test_simulation_config_creation(self):
        """Test creating a SimulationConfig instance"""
        config = SimulationConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000,
            symbols=["AAPL", "MSFT"],
            rebalance_frequency="monthly"
        )

        self.assertEqual(config.start_date, "2023-01-01")
        self.assertEqual(config.end_date, "2023-12-31")
        self.assertEqual(config.initial_capital, 100000)
        self.assertEqual(config.symbols, ["AAPL", "MSFT"])
        self.assertEqual(config.rebalance_frequency, "monthly")
        self.assertIsNone(config.weights)  # Should default to None

    def test_simulation_config_defaults(self):
        """Test SimulationConfig default values"""
        config = SimulationConfig(
            start_date="2023-01-01",
            end_date="2023-12-31"
        )

        self.assertEqual(config.initial_capital, 100000.0)
        self.assertIsNone(config.symbols)
        self.assertIsNone(config.weights)
        self.assertEqual(config.rebalance_frequency, "monthly")
        self.assertEqual(config.transaction_costs, 0.001)
        self.assertEqual(config.slippage, 0.0005)
        self.assertEqual(config.benchmark_symbol, "SPY")
        self.assertEqual(config.risk_free_rate, 0.02)


class TestHistoricalSimulationEngine(unittest.TestCase):
    """Test HistoricalSimulationEngine class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = SimulationConfig(
            start_date="2023-01-01",
            end_date="2023-01-31",
            initial_capital=100000,
            symbols=["AAPL", "MSFT"],
            weights={"AAPL": 0.6, "MSFT": 0.4},
            rebalance_frequency="monthly"
        )
        self.engine = HistoricalSimulationEngine(self.config)

    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.portfolio_value, 100000)
        self.assertEqual(self.engine.cash, 100000)
        self.assertEqual(self.engine.positions, {})
        self.assertEqual(self.engine.trades, [])
        self.assertEqual(self.engine.portfolio_history, [])
        self.assertEqual(self.engine.benchmark_history, [])

        # Check that weights were set correctly
        self.assertEqual(self.engine.config.weights, {"AAPL": 0.6, "MSFT": 0.4})

    def test_engine_initialization_equal_weights(self):
        """Test engine initialization with equal weights"""
        config = SimulationConfig(
            start_date="2023-01-01",
            end_date="2023-01-31",
            symbols=["AAPL", "MSFT", "GOOGL"]
        )
        engine = HistoricalSimulationEngine(config)

        # Should have equal weights
        expected_weights = {"AAPL": 1/3, "MSFT": 1/3, "GOOGL": 1/3}
        self.assertEqual(engine.config.weights, expected_weights)

    @patch('src.utils.historical_simulation_engine.yf.download')
    def test_fetch_historical_data_success(self, mock_download):
        """Test successful historical data fetching"""
        # Mock data for AAPL
        dates = pd.date_range('2023-01-01', periods=5)
        aapl_data = pd.DataFrame({
            'Open': [150, 152, 148, 155, 153],
            'High': [155, 157, 153, 160, 158],
            'Low': [148, 150, 145, 152, 150],
            'Close': [152, 148, 155, 153, 158],
            'Volume': [1000000, 1100000, 950000, 1200000, 1050000]
        }, index=dates)

        # Mock data for MSFT
        msft_data = pd.DataFrame({
            'Open': [250, 252, 248, 255, 253],
            'High': [255, 257, 253, 260, 258],
            'Low': [248, 250, 245, 252, 250],
            'Close': [252, 248, 255, 253, 258],
            'Volume': [800000, 900000, 750000, 1000000, 850000]
        }, index=dates)

        def mock_download_func(*args, **kwargs):
            symbol = args[0] if args else kwargs.get('tickers', [''])[0]
            if symbol == 'AAPL':
                return aapl_data
            elif symbol == 'MSFT':
                return msft_data
            else:
                return pd.DataFrame()

        mock_download.side_effect = mock_download_func

        result = self.engine.fetch_historical_data(['AAPL', 'MSFT'], '2023-01-01', '2023-01-05')

        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)
        self.assertEqual(len(result['AAPL']), 5)
        self.assertEqual(len(result['MSFT']), 5)

        # Check that Returns and Symbol columns were added
        self.assertIn('Returns', result['AAPL'].columns)
        self.assertIn('Symbol', result['AAPL'].columns)
        self.assertEqual(result['AAPL']['Symbol'].iloc[0], 'AAPL')

    @patch('src.utils.historical_simulation_engine.yf.download')
    def test_fetch_historical_data_empty(self, mock_download):
        """Test fetching data when yfinance returns empty data"""
        mock_download.return_value = pd.DataFrame()

        result = self.engine.fetch_historical_data(['INVALID'], '2023-01-01', '2023-01-05')

        self.assertEqual(len(result), 0)

    def test_should_rebalance(self):
        """Test rebalancing logic"""
        dates = [datetime(2023, 1, 1).date(), datetime(2023, 1, 2).date(),
                datetime(2023, 2, 1).date(), datetime(2023, 3, 1).date()]

        # Test monthly rebalancing
        self.assertTrue(self.engine._should_rebalance(dates[0], 0, dates))  # First day
        self.assertFalse(self.engine._should_rebalance(dates[1], 1, dates))  # Same month
        self.assertTrue(self.engine._should_rebalance(dates[2], 2, dates))  # New month

        # Test daily rebalancing
        self.engine.config.rebalance_frequency = 'daily'
        self.assertTrue(self.engine._should_rebalance(dates[1], 1, dates))

        # Test weekly rebalancing (Mondays)
        self.engine.config.rebalance_frequency = 'weekly'
        monday = datetime(2023, 1, 2).date()  # Monday
        tuesday = datetime(2023, 1, 3).date()  # Tuesday
        self.assertTrue(self.engine._should_rebalance(monday, 1, dates))
        self.assertFalse(self.engine._should_rebalance(tuesday, 2, dates))

    def test_record_portfolio_state(self):
        """Test recording portfolio state"""
        test_date = datetime(2023, 1, 1).date()
        prices = {"AAPL": 150.0, "MSFT": 250.0}

        # Set up some positions
        self.engine.positions = {"AAPL": 100, "MSFT": 50}
        self.engine.cash = 50000

        self.engine._record_portfolio_state(test_date, prices)

        self.assertEqual(len(self.engine.portfolio_history), 1)
        state = self.engine.portfolio_history[0]

        expected_value = 50000 + (100 * 150.0) + (50 * 250.0)  # 50000 + 15000 + 12500 = 77500
        self.assertEqual(state['portfolio_value'], expected_value)
        self.assertEqual(state['cash'], 50000)
        self.assertEqual(state['positions'], {"AAPL": 100, "MSFT": 50})

    @patch('src.utils.historical_simulation_engine.yf.download')
    def test_run_portfolio_simulation_success(self, mock_download):
        """Test successful portfolio simulation run"""
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=10)
        aapl_data = pd.DataFrame({
            'Open': [150] * 10,
            'High': [155] * 10,
            'Low': [145] * 10,
            'Close': [152] * 10,
            'Volume': [1000000] * 10
        }, index=dates)

        msft_data = pd.DataFrame({
            'Open': [250] * 10,
            'High': [255] * 10,
            'Low': [245] * 10,
            'Close': [252] * 10,
            'Volume': [800000] * 10
        }, index=dates)

        spy_data = pd.DataFrame({
            'Open': [400] * 10,
            'High': [405] * 10,
            'Low': [395] * 10,
            'Close': [402] * 10,
            'Volume': [5000000] * 10
        }, index=dates)

        def mock_download_func(*args, **kwargs):
            symbol = args[0] if args else kwargs.get('tickers', [''])[0]
            if symbol == 'AAPL':
                return aapl_data
            elif symbol == 'MSFT':
                return msft_data
            elif symbol == 'SPY':
                return spy_data
            else:
                return pd.DataFrame()

        mock_download.side_effect = mock_download_func

        results = self.engine.run_portfolio_simulation()

        # Check that results contain expected keys
        self.assertIn('simulation_config', results)
        self.assertIn('performance_metrics', results)
        self.assertIn('trading_statistics', results)
        self.assertIn('portfolio_history', results)
        self.assertIn('trades', results)

        # Check performance metrics
        perf = results['performance_metrics']
        self.assertIn('total_return', perf)
        self.assertIn('annualized_return', perf)
        self.assertIn('sharpe_ratio', perf)
        self.assertIn('max_drawdown', perf)

        # Check that trades were recorded
        self.assertGreater(len(results['trades']), 0)

    @patch('src.utils.historical_simulation_engine.yf.download')
    def test_run_portfolio_simulation_no_data(self, mock_download):
        """Test simulation with no available data"""
        mock_download.return_value = pd.DataFrame()

        results = self.engine.run_portfolio_simulation()

        self.assertIn('error', results)
        self.assertEqual(results['error'], 'No historical data available for simulation')


class TestRunHistoricalPortfolioSimulation(unittest.TestCase):
    """Test the convenience function for running simulations"""

    @patch('src.utils.historical_simulation_engine.HistoricalSimulationEngine')
    def test_run_historical_portfolio_simulation(self, mock_engine_class):
        """Test the convenience function"""
        mock_engine = MagicMock()
        mock_engine.run_portfolio_simulation.return_value = {"success": True}
        mock_engine_class.return_value = mock_engine

        result = run_historical_portfolio_simulation(
            symbols=["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000,
            weights={"AAPL": 0.6, "MSFT": 0.4},
            rebalance_frequency="monthly"
        )

        # Check that engine was created with correct config
        mock_engine_class.assert_called_once()
        call_args = mock_engine_class.call_args[0][0]  # Get the config argument

        self.assertEqual(call_args.start_date, "2023-01-01")
        self.assertEqual(call_args.end_date, "2023-12-31")
        self.assertEqual(call_args.initial_capital, 100000)
        self.assertEqual(call_args.symbols, ["AAPL", "MSFT"])
        self.assertEqual(call_args.weights, {"AAPL": 0.6, "MSFT": 0.4})
        self.assertEqual(call_args.rebalance_frequency, "monthly")

        # Check that simulation was run
        mock_engine.run_portfolio_simulation.assert_called_once()
        self.assertEqual(result, {"success": True})


class TestRunMultiStrategyComparison(unittest.TestCase):
    """Test multi-strategy comparison function"""

    @patch('src.utils.historical_simulation_engine.run_historical_portfolio_simulation')
    def test_run_multi_strategy_comparison(self, mock_run_simulation):
        """Test comparing multiple strategies"""
        # Mock simulation results
        mock_run_simulation.side_effect = [
            {"strategy": "equal_weight", "sharpe": 1.2},
            {"strategy": "momentum", "sharpe": 1.5}
        ]

        strategies = [
            {"name": "equal_weight", "weights": {"AAPL": 0.5, "MSFT": 0.5}},
            {"name": "momentum", "weights": {"AAPL": 0.7, "MSFT": 0.3}}
        ]

        result = run_multi_strategy_comparison(
            symbols=["AAPL", "MSFT"],
            strategies=strategies,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )

        self.assertIn("equal_weight", result)
        self.assertIn("momentum", result)
        self.assertEqual(result["equal_weight"], {"strategy": "equal_weight", "sharpe": 1.2})
        self.assertEqual(result["momentum"], {"strategy": "momentum", "sharpe": 1.5})

        # Check that run_historical_portfolio_simulation was called twice
        self.assertEqual(mock_run_simulation.call_count, 2)

    def test_run_multi_strategy_comparison_invalid_strategy(self):
        """Test multi-strategy comparison with invalid strategy config"""
        strategies = [
            {"invalid": "config"}  # Missing 'name' field
        ]

        with self.assertRaises(KeyError):
            run_multi_strategy_comparison(
                symbols=["AAPL"],
                strategies=strategies,
                start_date="2023-01-01",
                end_date="2023-12-31"
            )


if __name__ == '__main__':
    unittest.main()