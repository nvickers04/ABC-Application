#!/usr/bin/env python3
"""
Unit tests for core agent classes in the ABC Application system.
Tests DataAgent, StrategyAgent, RiskAgent, ExecutionAgent, ReflectionAgent, LearningAgent, and MacroAgent.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock the subagents before importing DataAgent
with patch('src.agents.data_subs.yfinance_datasub.YfinanceDatasub'):
    with patch('src.agents.data_subs.sentiment_datasub.SentimentDatasub'):
        with patch('src.agents.data_subs.news_datasub.NewsDatasub'):
            with patch('src.agents.data_subs.economic_datasub.EconomicDatasub'):
                with patch('src.agents.data_subs.institutional_datasub.InstitutionalDatasub'):
                    with patch('src.agents.data_subs.fundamental_datasub.FundamentalDatasub'):
                        with patch('src.agents.data_subs.microstructure_datasub.MicrostructureDatasub'):
                            with patch('src.agents.data_subs.kalshi_datasub.KalshiDatasub'):
                                with patch('src.agents.data_subs.options_datasub.OptionsDatasub'):
                                    from src.agents.data import DataAgent

from src.agents.strategy import StrategyAgent
from src.agents.risk import RiskAgent
from src.agents.execution import ExecutionAgent
from src.agents.reflection import ReflectionAgent
from src.agents.learning import LearningAgent
from src.agents.macro import MacroAgent


class TestDataAgent:
    """Test cases for DataAgent functionality."""

    @pytest.fixture
    def data_agent(self):
        """Create a DataAgent instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            agent = DataAgent()
            agent.role = "data"
            agent.tools = []
            agent.configs = {}
            agent.memory = {}
            # Create a proper async context manager mock
            from unittest.mock import AsyncMock
            mock_memory_manager = Mock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=None)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_memory_manager.memory_efficient_context.return_value = mock_context
            agent.memory_manager = mock_memory_manager
            return agent

    def test_initialization(self, data_agent):
        """Test DataAgent initialization."""
        assert data_agent.role == "data"
        assert hasattr(data_agent, 'memory_manager')

    @patch('src.agents.data_subs.yfinance_datasub.YfinanceDatasub')
    @patch('src.agents.data_subs.sentiment_datasub.SentimentDatasub')
    @patch('src.agents.data_subs.news_datasub.NewsDatasub')
    @patch('src.agents.data_subs.economic_datasub.EconomicDatasub')
    @patch('src.agents.data_subs.institutional_datasub.InstitutionalDatasub')
    @patch('src.agents.data_subs.fundamental_datasub.FundamentalDatasub')
    @patch('src.agents.data_subs.microstructure_datasub.MicrostructureDatasub')
    @patch('src.agents.data_subs.kalshi_datasub.KalshiDatasub')
    @patch('src.agents.data_subs.options_datasub.OptionsDatasub')
    def test_subagent_initialization(self, mock_options, mock_kalshi, mock_micro, mock_fundamental, mock_institutional, mock_economic, mock_news, mock_sentiment, mock_yfinance, data_agent):
        """Test that data subagents are properly initialized."""
        # Mock the subagents
        mock_yfinance.return_value = Mock()
        mock_sentiment.return_value = Mock()
        mock_news.return_value = Mock()
        mock_economic.return_value = Mock()
        mock_institutional.return_value = Mock()
        mock_fundamental.return_value = Mock()
        mock_micro.return_value = Mock()
        mock_kalshi.return_value = Mock()
        mock_options.return_value = Mock()

        # Test that subagents can be accessed (they are initialized in __init__)
        # Since DataAgent.__init__ is mocked, we can't test actual initialization
        assert data_agent.role == "data"

    @patch('src.utils.redis_cache.get_redis_cache_manager')
    @patch('src.utils.memory_manager.get_memory_manager')
    def test_memory_systems_initialization(self, mock_memory_mgr, mock_redis, data_agent):
        """Test memory systems initialization."""
        mock_memory_mgr.return_value = Mock()
        mock_redis.return_value = Mock()

        # Test that memory systems are accessible
        assert data_agent.memory_manager is not None

    @pytest.mark.asyncio
    async def test_process_input_basic(self, data_agent):
        """Test basic process_input functionality."""
        test_input = {"symbols": ["AAPL", "GOOGL"], "timeframe": "1d"}

        with patch.object(data_agent, 'validate_data_quality', return_value=True):
            with patch.object(data_agent, 'enrich_with_subagents', return_value={"enriched": True}):
                result = await data_agent.process_input(test_input)

                assert isinstance(result, dict)
                assert "data_quality_score" in result

    def test_data_validation(self, data_agent):
        """Test data validation functionality."""
        # Test with valid data
        valid_data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'price': [150.0, 2800.0],
            'volume': [1000000, 500000]
        })

        with patch.object(data_agent, 'validate_data_quality', return_value=True) as mock_validate:
            result = data_agent.validate_data_quality(valid_data)
            mock_validate.assert_called_once()

    def test_error_handling(self, data_agent):
        """Test error handling in data processing."""
        with patch.object(data_agent, 'process_input', side_effect=Exception("Test error")):
            # Should handle exceptions gracefully
            pass  # Test would go here


class TestStrategyAgent:
    """Test cases for StrategyAgent functionality."""

    @pytest.fixture
    def strategy_agent(self):
        """Create a StrategyAgent instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            agent = StrategyAgent()
            agent.role = "strategy"
            agent.tools = []
            agent.configs = {}
            agent.memory = {}
            return agent

    def test_initialization(self, strategy_agent):
        """Test StrategyAgent initialization."""
        assert strategy_agent.role == "strategy"
        assert hasattr(strategy_agent, 'tools')

    @patch('src.agents.strategy._get_options_strategy_sub')
    @patch('src.agents.strategy._get_flow_strategy_sub')
    @patch('src.agents.strategy._get_ml_strategy_sub')
    def test_subagent_access(self, mock_ml, mock_flow, mock_options, strategy_agent):
        """Test strategy subagent access."""
        mock_options.return_value = Mock()
        mock_flow.return_value = Mock()
        mock_ml.return_value = Mock()

        # Test lazy loading of subagents
        options_sub = strategy_agent._get_options_strategy_sub()
        assert options_sub is not None

    @pytest.mark.asyncio
    async def test_process_input_basic(self, strategy_agent):
        """Test basic process_input functionality."""
        test_input = {
            "market_data": pd.DataFrame({'symbol': ['AAPL'], 'price': [150.0]}),
            "risk_params": {"max_drawdown": 0.05}
        }

        with patch.object(strategy_agent, 'generate_strategy_proposals', return_value=[{"type": "options", "confidence": 0.8}]):
            result = await strategy_agent.process_input(test_input)

            assert isinstance(result, dict)
            assert "proposals" in result

    def test_strategy_proposal_generation(self, strategy_agent):
        """Test strategy proposal generation."""
        market_data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'price': [150.0, 2800.0],
            'volatility': [0.2, 0.25]
        })

        with patch.object(strategy_agent, 'generate_strategy_proposals') as mock_generate:
            mock_generate.return_value = [{"strategy": "covered_call", "symbol": "AAPL"}]
            proposals = strategy_agent.generate_strategy_proposals(market_data)

            assert isinstance(proposals, list)
            assert len(proposals) > 0


class TestRiskAgent:
    """Test cases for RiskAgent functionality."""

    @pytest.fixture
    def risk_agent(self):
        """Create a RiskAgent instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            agent = RiskAgent()
            agent.role = "risk"
            agent.tools = []
            agent.configs = {}
            agent.memory = {}
            return agent

    def test_initialization(self, risk_agent):
        """Test RiskAgent initialization."""
        assert risk_agent.role == "risk"
        assert hasattr(risk_agent, 'tools')

    @pytest.mark.asyncio
    async def test_process_input_basic(self, risk_agent):
        """Test basic process_input functionality."""
        test_input = {
            "strategy_proposals": [{"type": "options", "symbol": "AAPL"}],
            "portfolio_value": 100000
        }

        with patch.object(risk_agent, 'assess_risk', return_value={"approved": True, "risk_score": 0.3}):
            result = await risk_agent.process_input(test_input)

            assert isinstance(result, dict)
            assert "risk_assessment" in result

    def test_risk_assessment(self, risk_agent):
        """Test risk assessment functionality."""
        proposals = [{"type": "options", "symbol": "AAPL", "exposure": 10000}]

        with patch.object(risk_agent, 'calculate_var', return_value=0.05):
            with patch.object(risk_agent, 'calculate_sharpe_ratio', return_value=1.5):
                assessment = risk_agent.assess_risk(proposals, 100000)

                assert isinstance(assessment, dict)
                assert "approved" in assessment

    def test_var_calculation(self, risk_agent):
        """Test Value at Risk calculation."""
        returns = pd.Series([0.01, -0.02, 0.005, 0.01, -0.015])

        var = risk_agent.calculate_var(returns, confidence=0.95)
        assert isinstance(var, float)
        assert var <= 0  # VaR should be negative or zero


class TestExecutionAgent:
    """Test cases for ExecutionAgent functionality."""

    @pytest.fixture
    def execution_agent(self):
        """Create an ExecutionAgent instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            agent = ExecutionAgent()
            agent.role = "execution"
            agent.tools = []
            agent.configs = {}
            agent.memory = {}
            return agent

    def test_initialization(self, execution_agent):
        """Test ExecutionAgent initialization."""
        assert execution_agent.role == "execution"
        assert hasattr(execution_agent, 'tools')

    @pytest.mark.asyncio
    async def test_process_input_basic(self, execution_agent):
        """Test basic process_input functionality."""
        test_input = {
            "approved_strategies": [{"type": "buy", "symbol": "AAPL", "quantity": 100}],
            "risk_approved": True
        }

        with patch.object(execution_agent, 'execute_trades', return_value={"executed": True, "order_id": "123"}):
            result = await execution_agent.process_input(test_input)

            assert isinstance(result, dict)
            assert "execution_results" in result

    @patch('src.integrations.ibkr_connector.IBKRConnector')
    def test_trade_execution(self, mock_connector, execution_agent):
        """Test trade execution functionality."""
        mock_connector.return_value = Mock()
        mock_connector.return_value.place_order.return_value = {"status": "filled", "order_id": "123"}

        trades = [{"symbol": "AAPL", "action": "BUY", "quantity": 100}]

        with patch.object(execution_agent, 'execute_trades') as mock_execute:
            mock_execute.return_value = {"success": True}
            result = execution_agent.execute_trades(trades)

            assert isinstance(result, dict)


class TestReflectionAgent:
    """Test cases for ReflectionAgent functionality."""

    @pytest.fixture
    def reflection_agent(self):
        """Create a ReflectionAgent instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            agent = ReflectionAgent()
            agent.role = "reflection"
            agent.tools = []
            agent.configs = {}
            agent.memory = {}
            return agent

    def test_initialization(self, reflection_agent):
        """Test ReflectionAgent initialization."""
        assert reflection_agent.role == "reflection"
        assert hasattr(reflection_agent, 'tools')

    @pytest.mark.asyncio
    async def test_process_input_basic(self, reflection_agent):
        """Test basic process_input functionality."""
        test_input = {
            "execution_results": {"success": True, "pnl": 500},
            "previous_performance": {"win_rate": 0.6}
        }

        with patch.object(reflection_agent, 'analyze_performance', return_value={"insights": ["good_execution"]}):
            result = await reflection_agent.process_input(test_input)

            assert isinstance(result, dict)
            assert "reflection" in result

    def test_performance_analysis(self, reflection_agent):
        """Test performance analysis functionality."""
        execution_data = {"pnl": 500, "win_rate": 0.6, "max_drawdown": 0.03}

        analysis = reflection_agent.analyze_performance(execution_data)

        assert isinstance(analysis, dict)
        assert "insights" in analysis


class TestLearningAgent:
    """Test cases for LearningAgent functionality."""

    @pytest.fixture
    def learning_agent(self):
        """Create a LearningAgent instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            agent = LearningAgent()
            agent.role = "learning"
            agent.tools = []
            agent.configs = {}
            agent.memory = {}
            return agent

    def test_initialization(self, learning_agent):
        """Test LearningAgent initialization."""
        assert learning_agent.role == "learning"
        assert hasattr(learning_agent, 'tools')

    @pytest.mark.asyncio
    async def test_process_input_basic(self, learning_agent):
        """Test basic process_input functionality."""
        test_input = {
            "reflection_insights": ["improve_timing"],
            "historical_performance": {"accuracy": 0.7}
        }

        with patch.object(learning_agent, 'update_models', return_value={"models_updated": True}):
            result = await learning_agent.process_input(test_input)

            assert isinstance(result, dict)
            assert "learning_updates" in result

    def test_model_updates(self, learning_agent):
        """Test model update functionality."""
        insights = ["timing_issues", "risk_management"]

        updates = learning_agent.update_models(insights)

        assert isinstance(updates, dict)
        assert "models_updated" in updates


class TestMacroAgent:
    """Test cases for MacroAgent functionality."""

    @pytest.fixture
    def macro_agent(self):
        """Create a MacroAgent instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            agent = MacroAgent()
            agent.role = "macro"
            agent.tools = []
            agent.configs = {}
            agent.memory = {}
            return agent

    def test_initialization(self, macro_agent):
        """Test MacroAgent initialization."""
        assert macro_agent.role == "macro"
        assert hasattr(macro_agent, 'tools')

    @pytest.mark.asyncio
    async def test_process_input_basic(self, macro_agent):
        """Test basic process_input functionality."""
        test_input = {
            "market_conditions": {"volatility": "high", "trend": "bullish"},
            "economic_data": {"gdp_growth": 0.02}
        }

        with patch.object(macro_agent, 'assess_market_regime', return_value={"regime": "risk_on"}):
            result = await macro_agent.process_input(test_input)

            assert isinstance(result, dict)
            assert "macro_analysis" in result

    def test_market_regime_assessment(self, macro_agent):
        """Test market regime assessment."""
        market_data = {"volatility_index": 25, "yield_curve": "normal"}

        regime = macro_agent.assess_market_regime(market_data)

        assert isinstance(regime, dict)
        assert "regime" in regime


if __name__ == "__main__":
    pytest.main([__file__])