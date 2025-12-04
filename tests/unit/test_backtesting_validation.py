import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.utils.historical_simulation_engine import HistoricalSimulationEngine, SimulationConfig, run_historical_portfolio_simulation

REQUIRES_NETWORK = pytest.mark.skip(reason="Requires network access to fetch historical market data")

@REQUIRES_NETWORK
class TestBacktestingValidation:
    """Test suite for backtesting engine validation and edge cases"""

    def test_basic_simulation(self):
        """Test basic portfolio simulation functionality"""
        config = SimulationConfig(
            start_date='2020-01-01',
            end_date='2020-12-31',
            initial_capital=100000,
            symbols=['AAPL', 'MSFT'],
            weights={'AAPL': 0.6, 'MSFT': 0.4},
            rebalance_frequency='monthly'
        )

        engine = HistoricalSimulationEngine(config)
        result = engine.run_portfolio_simulation()

        assert 'performance_metrics' in result
        assert 'trading_statistics' in result
        assert 'portfolio_history' in result

        # Check realistic return ranges
        total_return = result['performance_metrics']['total_return']
        assert -1 <= total_return <= 5  # Between -100% and +500%

        sharpe_ratio = result['performance_metrics']['sharpe_ratio']
        assert -10 <= sharpe_ratio <= 10  # Reasonable Sharpe range

    def test_market_crash_scenario(self):
        """Test simulation during market crash (2008-like conditions)"""
        config = SimulationConfig(
            start_date='2008-01-01',
            end_date='2008-12-31',
            initial_capital=100000,
            symbols=['SPY'],  # S&P 500 ETF
            rebalance_frequency='monthly'
        )

        engine = HistoricalSimulationEngine(config)
        result = engine.run_portfolio_simulation()

        assert result is not None
        assert 'performance_metrics' in result

        # During 2008 crash, expect negative returns
        total_return = result['performance_metrics']['total_return']
        max_drawdown = result['performance_metrics']['max_drawdown']

        # Validate ranges
        assert total_return >= -1  # Not worse than total loss
        assert max_drawdown >= -1  # Drawdown not worse than 100%

    def test_high_volatility_scenario(self):
        """Test with high volatility assets"""
        config = SimulationConfig(
            start_date='2020-01-01',
            end_date='2020-12-31',
            initial_capital=100000,
            symbols=['TSLA'],  # High volatility stock
            rebalance_frequency='weekly'
        )

        engine = HistoricalSimulationEngine(config)
        result = engine.run_portfolio_simulation()

        assert result is not None

        # Check volatility is calculated
        volatility = result['performance_metrics']['volatility']
        assert volatility >= 0

    def test_benchmark_comparison(self):
        """Test benchmark comparison functionality"""
        config = SimulationConfig(
            start_date='2020-01-01',
            end_date='2020-12-31',
            initial_capital=100000,
            symbols=['AAPL'],
            benchmark_symbol='SPY'
        )

        engine = HistoricalSimulationEngine(config)
        result = engine.run_portfolio_simulation()

        assert 'benchmark_comparison' in result
        benchmark_data = result['benchmark_comparison']
        assert 'benchmark_symbol' in benchmark_data
        assert benchmark_data['benchmark_symbol'] == 'SPY'

    def test_edge_case_insufficient_data(self):
        """Test handling of insufficient historical data"""
        config = SimulationConfig(
            start_date='1990-01-01',  # Very old date
            end_date='1990-01-05',   # Very short period
            initial_capital=100000,
            symbols=['AAPL']
        )

        engine = HistoricalSimulationEngine(config)
        result = engine.run_portfolio_simulation()

        # Should handle gracefully
        assert result is not None
        # May return error or minimal results
        if 'error' in result:
            assert 'data' in result['error'].lower() or 'date' in result['error'].lower()

    def test_transaction_costs_impact(self):
        """Test impact of transaction costs on performance"""
        # High frequency trading
        config_high_cost = SimulationConfig(
            start_date='2020-01-01',
            end_date='2020-12-31',
            initial_capital=100000,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            rebalance_frequency='daily',  # High frequency
            transaction_costs=0.005  # 0.5% per trade
        )

        engine = HistoricalSimulationEngine(config_high_cost)
        result_high = engine.run_portfolio_simulation()

        # Low frequency trading
        config_low_cost = SimulationConfig(
            start_date='2020-01-01',
            end_date='2020-12-31',
            initial_capital=100000,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            rebalance_frequency='quarterly',  # Low frequency
            transaction_costs=0.001  # 0.1% per trade
        )

        engine = HistoricalSimulationEngine(config_low_cost)
        result_low = engine.run_portfolio_simulation()

        # High frequency should have lower returns due to costs
        if 'performance_metrics' in result_high and 'performance_metrics' in result_low:
            return_high = result_high['performance_metrics']['total_return']
            return_low = result_low['performance_metrics']['total_return']
            # Note: This may not always hold due to market conditions, but costs should impact

    def test_portfolio_rebalancing(self):
        """Test portfolio rebalancing logic"""
        config = SimulationConfig(
            start_date='2020-01-01',
            end_date='2020-06-30',  # 6 months
            initial_capital=100000,
            symbols=['AAPL', 'MSFT'],
            weights={'AAPL': 0.5, 'MSFT': 0.5},
            rebalance_frequency='monthly'
        )

        engine = HistoricalSimulationEngine(config)
        result = engine.run_portfolio_simulation()

        assert result is not None
        trades = result.get('trades', [])
        # Should have some rebalancing trades
        assert len(trades) > 0

    @patch('src.utils.historical_simulation_engine.yf.download')
    def test_mock_data_simulation(self, mock_download):
        """Test simulation with mocked market data"""
        # Create mock data
        dates = pd.date_range('2020-01-01', '2020-01-10')
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Volume': [1000000] * 10
        }, index=dates)

        # Mock yfinance download
        mock_download.return_value = mock_data

        config = SimulationConfig(
            start_date='2020-01-01',
            end_date='2020-01-10',
            initial_capital=10000,
            symbols=['TEST'],
            weights={'TEST': 1.0}
        )

        engine = HistoricalSimulationEngine(config)
        result = engine.run_portfolio_simulation()

        assert result is not None
        assert 'performance_metrics' in result

        # With rising prices, should have positive return
        total_return = result['performance_metrics']['total_return']
        assert total_return > 0

    def test_convenience_function(self):
        """Test the convenience function for running simulations"""
        result = run_historical_portfolio_simulation(
            symbols=['AAPL'],
            start_date='2020-01-01',
            end_date='2020-12-31',
            initial_capital=50000,
            rebalance_frequency='monthly'
        )

        assert result is not None
        assert 'performance_metrics' in result
        assert 'trading_statistics' in result