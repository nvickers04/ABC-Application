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

            # DataAgent now returns cross-verification results, not raw data
            assert 'cross_verification' in result
            assert isinstance(result['cross_verification'], dict)
            # Check that the pipeline was called
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
            'dataframe': market_data,
            'symbols': ['AAPL'],
            'sentiment': {'sentiment': 'bullish', 'confidence': 0.8}
        }

        # Mock the analyzers to return test data
        with patch.object(strategy_agent.options_analyzer, 'process_input', new_callable=AsyncMock) as mock_options, \
             patch.object(strategy_agent.flow_analyzer, 'process_input', new_callable=AsyncMock) as mock_flow, \
             patch.object(strategy_agent.ai_analyzer, 'process_input', new_callable=AsyncMock) as mock_ai, \
             patch.object(strategy_agent.multi_instrument_analyzer, 'process_input', new_callable=AsyncMock) as mock_multi:

            mock_options.return_value = {'options': {'strategy_type': 'long_call', 'roi_estimate': 0.25}}
            mock_flow.return_value = {'flow': {'strategy_type': 'momentum', 'roi_estimate': 0.20}}
            mock_ai.return_value = {'ai': {'strategy_type': 'ml_prediction', 'roi_estimate': 0.30}}
            mock_multi.return_value = {'multi_instrument': {'strategy_type': 'pairs_trade', 'roi_estimate': 0.15}}

            result = await strategy_agent.process_input(mock_input)

            assert 'strategy_type' in result
            assert 'roi_estimate' in result
            assert isinstance(result['roi_estimate'], (int, float))
            assert 'symbol' in result
            assert result['symbol'] == 'AAPL'

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
            'roi_estimate': 0.25,
            'symbol': 'AAPL',
            'quantity': 100,
            'setup': 'momentum'
        }

        # Mock the methods that _process_input calls
        with patch.object(risk_agent, '_get_vix_volatility', new_callable=AsyncMock) as mock_vix, \
             patch.object(risk_agent, '_run_stochastics') as mock_stochastics, \
             patch.object(risk_agent, '_vet_proposal', new_callable=AsyncMock) as mock_vet, \
             patch.object(risk_agent, '_adjust_post_batch') as mock_adjust, \
             patch.object(risk_agent, '_generate_yaml_diffs') as mock_diffs:

            mock_vix.return_value = 0.15
            mock_stochastics.return_value = {
                'pop': 0.72,
                'var_95': 2250.0,
                'cvar_95': 3000.0,
                'max_drawdown_sim': 0.08,
                'sharpe_ratio_sim': 1.5
            }
            mock_vet.return_value = {
                'approved': True,
                'rationale': 'Risk parameters within acceptable limits'
            }
            mock_adjust.return_value = {}
            mock_diffs.return_value = {}

            result = await risk_agent.process_input(mock_input)

            assert result['approved'] is True
            assert 'simulated_pop' in result
            assert 'yaml_diffs' in result
            assert 'var_95' in result
            assert result['var_95'] == 2250.0

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

        # Mock IBKR TWS status check
        with patch.object(execution_agent, '_check_ibkr_tws_status', new_callable=AsyncMock) as mock_tws:

            mock_tws.return_value = {'connected': True, 'version': '10.19'}

            result = await execution_agent.process_input(mock_input)

            assert result['success'] is True
            assert result['processed'] is True
            assert 'tws_status' in result
            assert result['tws_status']['connected'] is True
            mock_tws.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_trading_workflow_integration(self, agents, mock_ibkr_connector):
        """Test the complete Data → Strategy → Risk → Execution workflow"""
        data_agent = agents['data']
        strategy_agent = agents['strategy']
        risk_agent = agents['risk']
        execution_agent = agents['execution']

        # Step 1: Data Collection
        data_input = {'symbols': ['AAPL'], 'sources': ['yfinance']}
        with patch.object(data_agent.pipeline_processor, 'process_symbols_pipeline', new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = {
                'AAPL': {
                    'price': 150.0,
                    'volume': 1000000,
                    'high': 152.0,
                    'low': 148.0
                }
            }
            market_data = await data_agent.process_input(data_input)
            assert 'cross_verification' in market_data

        # Step 2: Strategy Generation
        strategy_input = {
            'dataframe': pd.DataFrame({
                'Close': [148, 149, 150, 151, 150],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1250000]
            }),
            'symbols': ['AAPL']
        }
        with patch.object(strategy_agent.options_analyzer, 'process_input', new_callable=AsyncMock) as mock_options, \
             patch.object(strategy_agent.flow_analyzer, 'process_input', new_callable=AsyncMock) as mock_flow, \
             patch.object(strategy_agent.ai_analyzer, 'process_input', new_callable=AsyncMock) as mock_ai, \
             patch.object(strategy_agent.multi_instrument_analyzer, 'process_input', new_callable=AsyncMock) as mock_multi:

            mock_options.return_value = {'options': {'strategy_type': 'long_call', 'roi_estimate': 0.25}}
            mock_flow.return_value = {'flow': {'strategy_type': 'momentum', 'roi_estimate': 0.20}}
            mock_ai.return_value = {'ai': {'strategy_type': 'ml_prediction', 'roi_estimate': 0.30}}
            mock_multi.return_value = {'multi_instrument': {'strategy_type': 'pairs_trade', 'roi_estimate': 0.15}}

            trading_signal = await strategy_agent.process_input(strategy_input)
            assert 'strategy_type' in trading_signal
            assert 'roi_estimate' in trading_signal

        # Step 3: Risk Assessment
        risk_input = {
            'roi_estimate': trading_signal.get('roi_estimate', 0.25),
            'symbol': 'AAPL',
            'quantity': 100
        }
        with patch.object(risk_agent, '_get_vix_volatility', new_callable=AsyncMock) as mock_vix, \
             patch.object(risk_agent, '_run_stochastics') as mock_stochastics, \
             patch.object(risk_agent, '_vet_proposal', new_callable=AsyncMock) as mock_vet:

            mock_vix.return_value = 0.15
            mock_stochastics.return_value = {
                'pop': 0.72,
                'var_95': 2250.0,
                'cvar_95': 3000.0,
                'max_drawdown_sim': 0.08,
                'sharpe_ratio_sim': 1.5
            }
            mock_vet.return_value = {
                'approved': True,
                'rationale': 'Risk parameters within acceptable limits'
            }

            risk_assessment = await risk_agent.process_input(risk_input)
            assert risk_assessment['approved'] is True

        # Step 4: Order Execution
        execution_input = {
            'type': 'equity_order',
            'symbol': 'AAPL',
            'quantity': 100
        }
        with patch.object(execution_agent, '_check_ibkr_tws_status', new_callable=AsyncMock) as mock_tws:

            mock_tws.return_value = {'connected': True, 'version': '10.19'}

            execution_result = await execution_agent.process_input(execution_input)

            assert execution_result['success'] is True
            assert execution_result['processed'] is True

    @pytest.mark.asyncio
    async def test_error_handling_in_trading_path(self, agents):
        """Test error handling throughout the trading path"""
        execution_agent = agents['execution']

        # Test IBKR TWS connection failure
        with patch.object(execution_agent, '_check_ibkr_tws_status', new_callable=AsyncMock) as mock_tws:

            mock_tws.return_value = {'connected': False, 'error': 'TWS not running'}

            result = await execution_agent.process_input({
                'type': 'equity_order',
                'symbol': 'AAPL',
                'quantity': 100
            })

            assert result['success'] is False
            assert 'error' in result
            assert result['error'] == 'IBKR TWS not connected'

    @pytest.mark.asyncio
    async def test_risk_rejection_workflow(self, agents):
        """Test workflow when risk agent rejects a trade"""
        risk_agent = agents['risk']

        high_risk_proposal = {
            'roi_estimate': 0.25,
            'symbol': 'AAPL',
            'quantity': 1000,  # Large position
            'position_value': 150000.0  # Large position value
        }

        with patch.object(risk_agent, '_get_vix_volatility', new_callable=AsyncMock) as mock_vix, \
             patch.object(risk_agent, '_run_stochastics') as mock_stochastics, \
             patch.object(risk_agent, '_vet_proposal', new_callable=AsyncMock) as mock_vet:

            mock_vix.return_value = 0.15
            mock_stochastics.return_value = {
                'pop': 0.72,
                'var_95': 2250.0,
                'cvar_95': 3000.0,
                'max_drawdown_sim': 0.08,
                'sharpe_ratio_sim': 1.5
            }
            mock_vet.return_value = {
                'approved': False,
                'rationale': 'Position size exceeds risk limits - POP below threshold'
            }

            result = await risk_agent.process_input(high_risk_proposal)

            assert result['approved'] is False
            assert 'rationale' in result
            assert 'simulated_pop' in result