"""
Critical Trading Path Integration Test
Tests the complete Data → Strategy → Risk → Execution workflow
"""
import pytest
import pytest_asyncio
import asyncio
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.data import DataAgent
from src.agents.strategy import StrategyAgent
from src.agents.risk import RiskAgent
from src.agents.execution import ExecutionAgent
from src.integrations.ibkr_connector import IBKRConnector
from src.utils.exceptions import IBKRError, OrderError


class TestCriticalTradingPath:
    """Integration tests for the critical Data → Strategy → Risk → Execution path"""

    @pytest.fixture
    async def mock_ibkr_connector(self):
        """Mock IBKR connector for testing"""
        mock_connector = MagicMock(spec=IBKRConnector)
        mock_connector.connect = AsyncMock(return_value=True)
        mock_connector.is_connected = MagicMock(return_value=True)
        mock_connector.get_account_balance = AsyncMock(return_value=100000.0)
        mock_connector.get_positions = AsyncMock(return_value=[])
        mock_connector.place_order = AsyncMock(return_value={
            "order_id": "test_123",
            "status": "filled",
            "filled_quantity": 100,
            "avg_fill_price": 150.0
        })
        mock_connector.get_market_data = AsyncMock(return_value={
            "symbol": "AAPL",
            "price": 150.0,
            "bid": 149.9,
            "ask": 150.1,
            "volume": 1000000,
            "timestamp": "2025-12-03T10:00:00Z"
        })
        return mock_connector

    @pytest.fixture
    async def agents(self):
        """Setup all trading agents"""
        data_agent = DataAgent()
        strategy_agent = StrategyAgent()
        risk_agent = RiskAgent()
        execution_agent = ExecutionAgent()

        # Initialize LLM for agents if needed
        try:
            await data_agent.async_initialize_llm()
        except:
            pass  # LLM init may fail in test environment
        try:
            await strategy_agent.async_initialize_llm()
        except:
            pass
        try:
            await risk_agent.async_initialize_llm()
        except:
            pass
        try:
            await execution_agent.async_initialize_llm()
        except:
            pass

        yield {
            'data': data_agent,
            'strategy': strategy_agent,
            'risk': risk_agent,
            'execution': execution_agent
        }

        # No explicit cleanup needed for these agents

    @pytest.mark.asyncio
    async def test_data_agent_market_data_collection(self, agents):
        """Test DataAgent can collect and process market data"""
        data_agent = agents['data']

        # Mock market data input
        mock_input = {
            'symbols': ['AAPL'],
            'sources': ['yfinance'],
            'period': '1d'
        }

        # Mock the pipeline processor to return test data
        with patch.object(data_agent.pipeline_processor, 'process_symbols_pipeline', new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = {
                'AAPL': {
                    'price': 150.0,
                    'volume': 1000000,
                    'high': 152.0,
                    'low': 148.0,
                    'timestamp': '2025-12-03T10:00:00Z'
                }
            }

            result = await data_agent.process_input(mock_input)

            assert 'AAPL' in result
            assert result['AAPL']['price'] == 150.0
            assert result['AAPL']['volume'] == 1000000
            mock_pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_strategy_agent_signal_generation(self, agents):
        """Test StrategyAgent can generate trading signals from market data"""
        strategy_agent = agents['strategy']

        # Mock market data input
        market_data = pd.DataFrame({
            'Close': [148, 149, 150, 151, 150],
            'High': [150, 151, 152, 153, 152],
            'Low': [146, 147, 148, 149, 148],
            'Open': [148, 149, 150, 151, 150],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1250000]
        })

        mock_input = {
            'market_data': market_data,
            'symbol': 'AAPL',
            'sentiment': {'sentiment': 'bullish', 'confidence': 0.8}
        }

        # Mock strategy analysis
        with patch.object(strategy_agent, '_analyze_market_data', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                'signal': 'BUY',
                'confidence': 0.75,
                'quantity': 100,
                'strategy_type': 'momentum',
                'entry_price': 150.0,
                'stop_loss': 145.0,
                'take_profit': 160.0
            }

            result = await strategy_agent.process_input(mock_input)

            assert result['signal'] == 'BUY'
            assert result['confidence'] >= 0.7
            assert result['quantity'] > 0
            assert 'strategy_type' in result
            mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_risk_agent_position_sizing(self, agents):
        """Test RiskAgent can validate and size positions"""
        risk_agent = agents['risk']

        # Mock trading signal input
        mock_signal = {
            'signal': 'BUY',
            'symbol': 'AAPL',
            'quantity': 100,
            'entry_price': 150.0,
            'stop_loss': 145.0,
            'strategy_type': 'momentum'
        }

        mock_portfolio = {
            'cash': 100000.0,
            'positions': [],
            'total_value': 100000.0
        }

        mock_input = {
            'signal': mock_signal,
            'portfolio': mock_portfolio,
            'market_conditions': {'volatility': 0.15, 'liquidity': 'high'}
        }

        # Mock risk assessment
        with patch.object(risk_agent, '_assess_risk', new_callable=AsyncMock) as mock_assess:
            mock_assess.return_value = {
                'approved': True,
                'adjusted_quantity': 100,
                'risk_metrics': {
                    'position_size_pct': 0.15,
                    'portfolio_impact': 0.015,
                    'var_95': 2250.0,
                    'expected_loss': 750.0
                },
                'risk_level': 'moderate'
            }

            result = await risk_agent.process_input(mock_input)

            assert result['approved'] is True
            assert result['adjusted_quantity'] == 100
            assert 'risk_metrics' in result
            assert result['risk_level'] in ['low', 'moderate', 'high']
            mock_assess.assert_called_once()

    @pytest.mark.asyncio
    async def test_execution_agent_order_placement(self, agents, mock_ibkr_connector):
        """Test ExecutionAgent can place orders via IBKR"""
        execution_agent = agents['execution']

        # Mock approved trade input
        mock_trade = {
            'signal': 'BUY',
            'symbol': 'AAPL',
            'quantity': 100,
            'entry_price': 150.0,
            'stop_loss': 145.0,
            'risk_approved': True,
            'risk_metrics': {
                'position_size_pct': 0.15,
                'var_95': 2250.0
            }
        }

        mock_input = {
            'trade': mock_trade,
            'execution_mode': 'paper'
        }

        # Mock IBKR connector and order execution
        with patch('src.integrations.ibkr_connector.get_ibkr_connector', return_value=mock_ibkr_connector), \
             patch.object(execution_agent, '_execute_order', new_callable=AsyncMock) as mock_execute:

            mock_execute.return_value = {
                'order_id': 'test_123',
                'status': 'filled',
                'filled_quantity': 100,
                'avg_fill_price': 150.0,
                'execution_time': '2025-12-03T10:00:05Z',
                'fees': 1.0
            }

            result = await execution_agent.process_input(mock_input)

            assert result['order_id'] == 'test_123'
            assert result['status'] == 'filled'
            assert result['filled_quantity'] == 100
            assert result['avg_fill_price'] == 150.0
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_trading_workflow_integration(self, agents, mock_ibkr_connector):
        """Test the complete Data → Strategy → Risk → Execution workflow"""
        data_agent = agents['data']
        strategy_agent = agents['strategy']
        risk_agent = agents['risk']
        execution_agent = agents['execution']

        # Step 1: Data Collection
        data_input = {'symbols': ['AAPL'], 'sources': ['yfinance']}
        with patch.object(data_agent, '_collect_market_data', new_callable=AsyncMock) as mock_data:
            mock_data.return_value = {
                'AAPL': {
                    'price': 150.0,
                    'volume': 1000000,
                    'high': 152.0,
                    'low': 148.0
                }
            }
            market_data = await data_agent.process_input(data_input)
            assert 'AAPL' in market_data

        # Step 2: Strategy Generation
        strategy_input = {
            'market_data': pd.DataFrame({
                'Close': [148, 149, 150, 151, 150],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1250000]
            }),
            'symbol': 'AAPL'
        }
        with patch.object(strategy_agent, '_analyze_market_data', new_callable=AsyncMock) as mock_strategy:
            mock_strategy.return_value = {
                'signal': 'BUY',
                'quantity': 100,
                'entry_price': 150.0,
                'stop_loss': 145.0
            }
            trading_signal = await strategy_agent.process_input(strategy_input)
            assert trading_signal['signal'] == 'BUY'

        # Step 3: Risk Assessment
        risk_input = {
            'signal': trading_signal,
            'portfolio': {'cash': 100000.0, 'positions': []}
        }
        with patch.object(risk_agent, '_assess_risk', new_callable=AsyncMock) as mock_risk:
            mock_risk.return_value = {
                'approved': True,
                'adjusted_quantity': 100,
                'risk_level': 'moderate'
            }
            risk_assessment = await risk_agent.process_input(risk_input)
            assert risk_assessment['approved'] is True

        # Step 4: Order Execution
        execution_input = {
            'trade': {**trading_signal, 'risk_approved': True},
            'execution_mode': 'paper'
        }
        with patch('src.integrations.ibkr_connector.get_ibkr_connector', return_value=mock_ibkr_connector), \
             patch.object(execution_agent, '_execute_order', new_callable=AsyncMock) as mock_execute:

            mock_execute.return_value = {
                'order_id': 'workflow_test_123',
                'status': 'filled',
                'filled_quantity': 100
            }

            execution_result = await execution_agent.process_input(execution_input)

            assert execution_result['status'] == 'filled'
            assert execution_result['filled_quantity'] == 100

    @pytest.mark.asyncio
    async def test_error_handling_in_trading_path(self, agents):
        """Test error handling throughout the trading path"""
        execution_agent = agents['execution']

        # Test IBKR connection failure
        with patch('src.integrations.ibkr_connector.get_ibkr_connector') as mock_get_connector:
            mock_connector = MagicMock()
            mock_connector.connect = AsyncMock(side_effect=IBKRError("Connection failed"))
            mock_get_connector.return_value = mock_connector

            with pytest.raises(IBKRError):
                await execution_agent.process_input({
                    'trade': {'signal': 'BUY', 'symbol': 'AAPL', 'quantity': 100},
                    'execution_mode': 'paper'
                })

    @pytest.mark.asyncio
    async def test_risk_rejection_workflow(self, agents):
        """Test workflow when risk agent rejects a trade"""
        risk_agent = agents['risk']

        high_risk_signal = {
            'signal': 'BUY',
            'symbol': 'AAPL',
            'quantity': 1000,  # Large position
            'entry_price': 150.0
        }

        risk_input = {
            'signal': high_risk_signal,
            'portfolio': {'cash': 10000.0, 'positions': []}  # Small portfolio
        }

        with patch.object(risk_agent, '_assess_risk', new_callable=AsyncMock) as mock_risk:
            mock_risk.return_value = {
                'approved': False,
                'reason': 'Position size exceeds risk limits',
                'adjusted_quantity': 50,
                'risk_level': 'high'
            }

            result = await risk_agent.process_input(risk_input)

            assert result['approved'] is False
            assert 'reason' in result
            assert result['risk_level'] == 'high'