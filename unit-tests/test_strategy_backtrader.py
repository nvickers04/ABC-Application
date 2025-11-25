import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.agents.strategy import StrategyAgent
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

class TestStrategyBacktrader:
    """Test suite for StrategyAgent backtrader integration"""

    @pytest.fixture
    def sample_data(self):
        """Sample market data for testing"""
        return {
            'dataframe': pd.DataFrame({
                'Close': [100, 105, 102, 108, 106, 110, 115, 112, 118, 120],
                'High': [102, 107, 104, 110, 108, 112, 117, 114, 120, 122],
                'Low': [98, 103, 100, 106, 104, 108, 113, 110, 116, 118],
                'Open': [100, 105, 102, 108, 106, 110, 115, 112, 118, 120]
            }),
            'sentiment': {'sentiment': 'bullish', 'confidence': 0.8},
            'symbols': ['SPY']
        }

    @pytest.mark.asyncio
    async def test_strategy_agent_basic(self, sample_data):
        """Test basic strategy agent functionality"""
        agent = StrategyAgent()

        result = await agent.process_input(sample_data)

        assert 'strategy_type' in result
        assert 'validation_confidence' in result
        assert isinstance(result['validation_confidence'], (int, float))

    @pytest.mark.asyncio
    async def test_backtrader_validation(self, sample_data):
        """Test backtrader validation integration"""
        agent = StrategyAgent()

        result = await agent.process_input(sample_data)

        # Check if backtrader validation was attempted
        backtrader_result = result.get('backtrader_validation', {})
        assert isinstance(backtrader_result, dict)

    @pytest.mark.asyncio
    async def test_edge_case_market_crash(self):
        """Test strategy performance during market crash scenario"""
        crash_data = {
            'dataframe': pd.DataFrame({
                'Close': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],  # Severe decline
                'High': [102, 92, 82, 72, 62, 52, 42, 32, 22, 12],
                'Low': [98, 88, 78, 68, 58, 48, 38, 28, 18, 8],
                'Open': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
            }),
            'sentiment': {'sentiment': 'bearish', 'confidence': 0.9},
            'symbols': ['SPY']
        }

        agent = StrategyAgent()
        result = await agent.process_input(crash_data)

        assert result is not None
        # Should handle crash scenario without errors
        assert 'strategy_type' in result

    @pytest.mark.asyncio
    async def test_edge_case_high_volatility(self):
        """Test strategy with high volatility data"""
        volatile_data = {
            'dataframe': pd.DataFrame({
                'Close': [100, 120, 80, 140, 60, 160, 40, 180, 20, 200],  # Extreme swings
                'High': [110, 130, 90, 150, 70, 170, 50, 190, 30, 210],
                'Low': [90, 110, 70, 130, 50, 150, 30, 170, 10, 190],
                'Open': [100, 120, 80, 140, 60, 160, 40, 180, 20, 200]
            }),
            'sentiment': {'sentiment': 'neutral', 'confidence': 0.5},
            'symbols': ['SPY']
        }

        agent = StrategyAgent()
        result = await agent.process_input(volatile_data)

        assert result is not None
        assert 'validation_confidence' in result

    @pytest.mark.asyncio
    async def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_data = {
            'dataframe': pd.DataFrame(),
            'sentiment': {'sentiment': 'neutral', 'confidence': 0.5},
            'symbols': ['SPY']
        }

        agent = StrategyAgent()

        result = await agent.process_input(empty_data)
        assert result is not None  # Handles empty dataframe gracefully

    @pytest.mark.asyncio
    async def test_invalid_sentiment(self):
        """Test with invalid sentiment data"""
        invalid_data = {
            'dataframe': pd.DataFrame({'Close': [100, 105, 102]}),
            'sentiment': {'sentiment': 'invalid', 'confidence': 2.0},  # Invalid confidence > 1
            'symbols': ['SPY']
        }

        agent = StrategyAgent()
        result = await agent.process_input(invalid_data)

        # Should handle gracefully
        assert result is not None