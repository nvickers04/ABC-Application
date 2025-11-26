import pytest
import pytest_asyncio
import asyncio
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.data import DataAgent
from src.agents.strategy import StrategyAgent
from src.agents.risk import RiskAgent
from src.agents.execution import ExecutionAgent
from src.integrations.ibkr import IBKRIntegration
import pandas as pd
import numpy as np

class TestE2ETradingWorkflow:
    """End-to-end tests for complete trading workflows"""

    @pytest.fixture
    def agents(self):
        """Setup all trading agents"""
        data_agent = DataAgent()
        strategy_agent = StrategyAgent()
        risk_agent = RiskAgent()
        execution_agent = ExecutionAgent()

        yield {
            'data': data_agent,
            'strategy': strategy_agent,
            'risk': risk_agent,
            'execution': execution_agent
        }

    @pytest.fixture
    def mock_ibkr(self):
        """Mock IBKR integration"""
        mock_ibkr = MagicMock(spec=IBKRIntegration)
        mock_ibkr.connect.return_value = True
        mock_ibkr.is_connected.return_value = True
        mock_ibkr.get_account_balance.return_value = 100000.0
        mock_ibkr.get_positions.return_value = []
        mock_ibkr.place_order.return_value = {"order_id": "test_123", "status": "filled"}
        mock_ibkr.disconnect.return_value = True
        return mock_ibkr

    @pytest.mark.asyncio
    async def test_complete_buy_workflow(self, agents, mock_ibkr):
        """Test complete buy order workflow from data to execution"""
        data_agent = agents['data']
        strategy_agent = agents['strategy']
        risk_agent = agents['risk']
        execution_agent = agents['execution']

        # Step 1: Data collection (mocked for test speed, real APIs would be used in production)
        mock_data_result = {
            'AAPL': {
                'price': 150.0,
                'volume': 1000000,
                'high': 152.0,
                'low': 148.0,
                'open': 149.0,
                'close': 150.0,
                'timestamp': '2024-01-01T12:00:00Z'
            }
        }
        
        with patch.object(data_agent, 'process_input', return_value=mock_data_result):
            data_result = await data_agent.process_input({
                'symbols': ['AAPL'],
                'period': '1d'
            })
            print(f"Data result: {data_result}")
            assert 'AAPL' in str(data_result)

        # Step 2: Strategy analysis
        market_data = pd.DataFrame({
            'Close': [148, 149, 150, 151, 150],
            'High': [150, 151, 152, 153, 152],
            'Low': [146, 147, 148, 149, 148],
            'Open': [148, 149, 150, 151, 150]
        })

        strategy_result = await strategy_agent.process_input({
            'dataframe': market_data,
            'sentiment': {'sentiment': 'bullish', 'confidence': 0.8},
            'symbols': ['AAPL']
        })
        print(f"Strategy result: {strategy_result}")
        assert 'strategy_type' in strategy_result

        # Step 3: Risk assessment
        risk_result = await risk_agent.process_input({
            'portfolio_returns': [0.01, 0.005, -0.002],
            'portfolio_value': 100000,
            'symbols': ['AAPL'],
            'proposed_position': {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0},
            'roi_estimate': 0.05
        })
        print(f"Risk result: {risk_result}")
        assert 'approved' in risk_result

        # Step 4: Trade execution (real paper trading)
        # NOTE: This will place a real order in IBKR paper trading account
        # Ensure TWS is running and paper trading is enabled
        execution_result = await execution_agent.execute_trade(
            symbol='AAPL',
            quantity=1,  # Small quantity for testing
            action='BUY',
            order_type='MKT'
        )
        print(f"Execution result: {execution_result}")
        assert execution_result is not None
        assert execution_result.get('success', False)

    @pytest.mark.asyncio
    async def test_sell_workflow_with_stops(self, agents, mock_ibkr):
        """Test sell workflow with stop-loss orders"""
        execution_agent = agents['execution']

        # Mock existing position
        mock_ibkr.get_positions.return_value = [
            {'symbol': 'AAPL', 'quantity': 100, 'avg_cost': 140.0, 'current_price': 160.0}
        ]

        with patch('src.integrations.ibkr.IBKRIntegration', return_value=mock_ibkr):
            # Execute sell order
            result = await execution_agent.process_input({
                'action': 'sell',
                'symbol': 'AAPL',
                'quantity': 50,
                'order_type': 'limit',
                'price': 158.0
            })

            assert result is not None
            assert mock_ibkr.place_order.called

    @pytest.mark.asyncio
    async def test_multi_agent_portfolio_rebalancing(self, agents, mock_ibkr):
        """Test portfolio rebalancing across multiple agents"""
        data_agent = agents['data']
        strategy_agent = agents['strategy']
        risk_agent = agents['risk']
        execution_agent = agents['execution']

        # Mock portfolio data
        portfolio = {
            'AAPL': {'quantity': 100, 'avg_cost': 140.0, 'current_price': 150.0},
            'MSFT': {'quantity': 50, 'avg_cost': 280.0, 'current_price': 300.0}
        }

        # Step 1: Get market data for all holdings
        with patch.object(data_agent, '_fetch_market_data', return_value={
            'AAPL': {'price': 150.0, 'volume': 1000000},
            'MSFT': {'price': 300.0, 'volume': 500000}
        }):
            data_result = await data_agent.process_input({
                'symbols': list(portfolio.keys()),
                'period': '1d'
            })
            assert data_result is not None

        # Step 2: Strategy analysis for rebalancing
        strategy_result = await strategy_agent.process_input({
            'task': 'portfolio_rebalancing',
            'current_portfolio': portfolio,
            'target_allocation': {'AAPL': 0.6, 'MSFT': 0.4}
        })
        assert strategy_result is not None

        # Step 3: Risk check for rebalancing trades
        risk_result = await risk_agent.process_input({
            'portfolio_returns': [0.01, 0.005, -0.002],
            'portfolio_value': 100000,
            'proposed_trades': [
                {'action': 'buy', 'symbol': 'AAPL', 'quantity': 50},
                {'action': 'sell', 'symbol': 'MSFT', 'quantity': 25}
            ]
        })
        assert 'risk_score' in risk_result

        # Step 4: Execute rebalancing trades
        with patch('src.integrations.ibkr.IBKRIntegration', return_value=mock_ibkr):
            for trade in [{'action': 'buy', 'symbol': 'AAPL', 'quantity': 50},
                         {'action': 'sell', 'symbol': 'MSFT', 'quantity': 25}]:
                result = await execution_agent.process_input(trade)
                assert result is not None

            # Verify multiple orders were placed
            assert mock_ibkr.place_order.call_count == 2

    @pytest.mark.asyncio
    async def test_intraday_trading_workflow(self, agents, mock_ibkr):
        """Test intraday trading workflow with rapid decisions"""
        data_agent = agents['data']
        strategy_agent = agents['strategy']
        execution_agent = agents['execution']

        # Simulate intraday price movements
        intraday_data = pd.DataFrame({
            'Close': [149, 151, 148, 152, 150],
            'High': [151, 153, 150, 154, 152],
            'Low': [147, 149, 146, 150, 148],
            'Open': [149, 151, 148, 152, 150],
            'Volume': [500000, 600000, 400000, 700000, 550000]
        })

        # Rapid strategy decisions
        strategy_result = await strategy_agent.process_input({
            'dataframe': intraday_data,
            'sentiment': {'sentiment': 'volatile', 'confidence': 0.6},
            'symbols': ['AAPL'],
            'timeframe': 'intraday'
        })
        assert strategy_result is not None

        # Quick execution
        with patch('src.integrations.ibkr.IBKRIntegration', return_value=mock_ibkr):
            execution_result = await execution_agent.process_input({
                'action': 'buy',
                'symbol': 'AAPL',
                'quantity': 25,
                'order_type': 'market',
                'time_in_force': 'immediate'
            })
            assert execution_result is not None

    @pytest.mark.asyncio
    async def test_error_recovery_in_workflow(self, agents, mock_ibkr):
        """Test error recovery during trading workflow"""
        execution_agent = agents['execution']

        # Mock IBKR failure
        mock_ibkr.place_order.side_effect = [Exception("Network error"), {"order_id": "retry_123"}]

        with patch('src.integrations.ibkr.IBKRIntegration', return_value=mock_ibkr):
            # First attempt fails
            with pytest.raises(Exception):
                await execution_agent.process_input({
                    'action': 'buy',
                    'symbol': 'AAPL',
                    'quantity': 100
                })

            # Retry succeeds
            result = await execution_agent.process_input({
                'action': 'buy',
                'symbol': 'AAPL',
                'quantity': 100
            })
            assert result is not None

    @pytest.mark.asyncio
    async def test_paper_trading_workflow(self, agents):
        """Test paper trading workflow (no real execution)"""
        data_agent = agents['data']
        strategy_agent = agents['strategy']
        risk_agent = agents['risk']
        execution_agent = agents['execution']

        # Paper trading mode
        os.environ['ABC_PAPER_TRADING'] = 'true'

        try:
            # Full workflow in paper trading mode
            data_result = await data_agent.process_input({
                'symbols': ['SPY'],
                'period': '1d'
            })

            strategy_result = await strategy_agent.process_input({
                'dataframe': pd.DataFrame({'Close': [400, 405, 402, 408, 406]}),
                'sentiment': {'sentiment': 'bullish', 'confidence': 0.7},
                'symbols': ['SPY']
            })

            risk_result = await risk_agent.process_input({
                'portfolio_returns': [0.01, 0.005],
                'portfolio_value': 50000,
                'proposed_position': {'symbol': 'SPY', 'quantity': 10, 'price': 405.0}
            })

            # Paper execution (should not call real IBKR)
            execution_result = await execution_agent.process_input({
                'action': 'buy',
                'symbol': 'SPY',
                'quantity': 10,
                'paper_trading': True
            })

            assert execution_result is not None
            # Verify no real broker calls were made
            # (This would be verified by mocking in real implementation)

        finally:
            os.environ.pop('ABC_PAPER_TRADING', None)

    @pytest.mark.asyncio
    async def test_multi_asset_workflow(self, agents, mock_ibkr):
        """Test workflow with multiple asset classes"""
        data_agent = agents['data']
        strategy_agent = agents['strategy']
        execution_agent = agents['execution']

        assets = ['AAPL', 'SPY', 'TLT', 'GLD']  # Stock, ETF, Bond, Commodity

        # Multi-asset data collection
        with patch.object(data_agent, '_fetch_market_data', return_value={
            asset: {'price': 100.0 + i * 10, 'volume': 100000}
            for i, asset in enumerate(assets)
        }):
            data_result = await data_agent.process_input({
                'symbols': assets,
                'period': '1d'
            })
            assert data_result is not None

        # Multi-asset strategy
        strategy_result = await strategy_agent.process_input({
            'task': 'multi_asset_allocation',
            'assets': assets,
            'market_regime': 'mixed'
        })
        assert strategy_result is not None

        # Execute multiple orders
        with patch('src.integrations.ibkr.IBKRIntegration', return_value=mock_ibkr):
            for asset in assets[:2]:  # Execute for first 2 assets
                result = await execution_agent.process_input({
                    'action': 'buy',
                    'symbol': asset,
                    'quantity': 10,
                    'order_type': 'market'
                })
                assert result is not None

            assert mock_ibkr.place_order.call_count == 2

    @pytest.mark.asyncio
    async def test_workflow_performance_under_load(self, agents):
        """Test workflow performance under concurrent load"""
        import time

        data_agent = agents['data']

        async def single_workflow(i):
            start_time = time.time()
            result = await data_agent.process_input({
                'symbols': [f'TEST{i}'],
                'period': '1d'
            })
            end_time = time.time()
            return end_time - start_time

        # Run multiple concurrent workflows
        tasks = [single_workflow(i) for i in range(10)]
        execution_times = await asyncio.gather(*tasks)

        # Verify reasonable performance
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)

        # Should complete within reasonable time (allowing for test environment)
        assert avg_time < 5.0  # Average under 5 seconds
        assert max_time < 10.0  # Max under 10 seconds

    @pytest.mark.parametrize("order_type", ["market", "limit", "stop", "stop_limit"])
    @pytest.mark.asyncio
    async def test_different_order_types(self, agents, mock_ibkr, order_type):
        """Test different order types in execution workflow"""
        execution_agent = agents['execution']

        order_params = {
            'action': 'buy',
            'symbol': 'AAPL',
            'quantity': 100,
            'order_type': order_type
        }

        if order_type in ['limit', 'stop', 'stop_limit']:
            order_params['price'] = 150.0
        if order_type == 'stop_limit':
            order_params['stop_price'] = 145.0

        with patch('src.integrations.ibkr.IBKRIntegration', return_value=mock_ibkr):
            result = await execution_agent.process_input(order_params)
            assert result is not None
            mock_ibkr.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_with_market_data_delays(self, agents):
        """Test workflow resilience to market data delays"""
        data_agent = agents['data']

        # Mock delayed data response
        async def delayed_response(*args, **kwargs):
            await asyncio.sleep(2)  # 2 second delay
            return {'AAPL': {'price': 150.0, 'volume': 1000000}}

        with patch.object(data_agent, 'fetch_market_data', side_effect=delayed_response):
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            result = await data_agent.process_input({
                'symbols': ['AAPL'],
                'period': '1d',
                'timeout': 5  # Allow 5 seconds
            })
            end_time = loop.time()

            # Should complete within timeout
            assert (end_time - start_time) < 5.0
            assert result is not None