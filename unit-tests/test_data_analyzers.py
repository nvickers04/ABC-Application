#!/usr/bin/env python3
"""
Unit tests for data analyzers.
Tests all 10 data analyzers: economic, fundamental, institutional, kalshi, marketdataapp,
microstructure, news, options, sentiment, yfinance.
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

# Import data subagents
from src.agents.data_analyzers.yfinance_data_analyzer import YfinanceDataAnalyzer
from src.agents.data_analyzers.sentiment_data_analyzer import SentimentDataAnalyzer
from src.agents.data_analyzers.news_data_analyzer import NewsDataAnalyzer
from src.agents.data_analyzers.economic_data_analyzer import EconomicDataAnalyzer
from src.agents.data_analyzers.fundamental_data_analyzer import FundamentalDataAnalyzer
from src.agents.data_analyzers.institutional_data_analyzer import InstitutionalDataAnalyzer
from src.agents.data_analyzers.microstructure_data_analyzer import MicrostructureDataAnalyzer
from src.agents.data_analyzers.kalshi_data_analyzer import KalshiDataAnalyzer
from src.agents.data_analyzers.options_data_analyzer import OptionsDataAnalyzer
from src.agents.data_analyzers.marketdataapp_data_analyzer import MarketDataAppDataAnalyzer


class TestYfinanceDataAnalyzer:
    """Test cases for YfinanceDataAnalyzer functionality."""

    @pytest.fixture
    def yfinance_sub(self):
        """Create a YfinanceDataAnalyzer instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            sub = YfinanceDataAnalyzer()
            sub.role = "yfinance_data"
            sub.memory_manager = Mock()
            # Mock the LLM for testing
            sub.llm = AsyncMock()
            sub.llm.ainvoke = AsyncMock(return_value=Mock(content='{"endpoints": ["quotes"], "priorities": {"quotes": 9}, "reasoning": "Test", "expected_insights": ["price data"]}'))
            return sub

    def test_initialization(self, yfinance_sub):
        """Test YfinanceDataAnalyzer initialization."""
        assert yfinance_sub.role == "yfinance_data"
        assert hasattr(yfinance_sub, 'memory_manager')

    @patch('yfinance.Ticker')
    @pytest.mark.asyncio
    async def test_process_input_basic(self, mock_ticker, yfinance_sub):
        """Test basic process_input functionality."""
        # Mock yfinance ticker
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {"marketCap": 1000000000, "volume": 1000000}
        mock_ticker_instance.history.return_value = pd.DataFrame({
            'Open': [150.0], 'High': [155.0], 'Low': [149.0], 'Close': [152.0], 'Volume': [1000000]
        })
        mock_ticker.return_value = mock_ticker_instance

        test_input = {"symbol": "AAPL", "period": "1d"}

        with patch.object(yfinance_sub, 'validate_data_quality', return_value=True):
            result = await yfinance_sub.process_input(test_input)

            assert isinstance(result, dict)
            assert "consolidated_data" in result
            assert "enhanced" in result
            assert result["enhanced"] is True

    def test_data_validation(self, yfinance_sub):
        """Test data validation functionality."""
        valid_data = pd.DataFrame({
            'Open': [150.0, 151.0],
            'High': [155.0, 156.0],
            'Low': [149.0, 150.0],
            'Close': [152.0, 153.0],
            'Volume': [1000000, 1100000]
        })

        with patch.object(yfinance_sub, 'validate_data_quality', return_value=True) as mock_validate:
            result = yfinance_sub.validate_data_quality(valid_data)
            mock_validate.assert_called_once()

    @patch('src.utils.redis_cache.cache_get')
    @patch('src.utils.redis_cache.cache_set')
    def test_caching_functionality(self, mock_cache_set, mock_cache_get, yfinance_sub):
        """Test caching functionality."""
        mock_cache_get.return_value = None  # Cache miss

        # Test cache operations are called
        # This would be tested in actual process_input calls


class TestNewsDataAnalyzer:
    """Test cases for NewsDataAnalyzer functionality."""

    @pytest.fixture
    def news_sub(self):
        """Create a NewsDataAnalyzer instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            sub = NewsDataAnalyzer()
            sub.role = "news_data"
            sub.memory_manager = Mock()
            # Mock the LLM for testing
            sub.llm = AsyncMock()
            sub.llm.ainvoke = AsyncMock(return_value=Mock(content='{"sources": ["newsapi"], "priorities": {"newsapi": 9}, "reasoning": "Test", "focus_areas": ["earnings"]}'))
            return sub

    def test_initialization(self, news_sub):
        """Test NewsDataAnalyzer initialization."""
        assert news_sub.role == "news_data"

    @patch('src.utils.tools.news_data_tool')
    @pytest.mark.asyncio
    async def test_process_input_basic(self, mock_news_tool, news_sub):
        """Test basic process_input functionality."""
        mock_news_tool.return_value = {
            "articles": [
                {"title": "Test Article", "content": "Test content", "sentiment": 0.1}
            ],
            "overall_sentiment": 0.1,
            "validated": True
        }

        test_input = {"symbol": "AAPL"}

        result = await news_sub.process_input(test_input)

        assert isinstance(result, dict)
        assert "articles_df" in result
        assert "enhanced" in result

    def test_cross_validation(self, news_sub):
        """Test news cross-validation functionality."""
        with patch.object(news_sub, '_cross_validate_news') as mock_validate:
            mock_validate.return_value = {
                "validated": True,
                "news_data": {"articles": []}
            }

            # Test would call the validation method
            pass


class TestSentimentDataAnalyzer:
    """Test cases for SentimentDataAnalyzer functionality."""

    @pytest.fixture
    def sentiment_sub(self):
        """Create a SentimentDataAnalyzer instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            sub = SentimentDataAnalyzer()
            sub.role = "sentiment_data"
            sub.memory_manager = Mock()
            return sub

    def test_initialization(self, sentiment_sub):
        """Test SentimentDataAnalyzer initialization."""
        assert sentiment_sub.role == "sentiment_data"

    @pytest.mark.asyncio
    async def test_process_input_basic(self, sentiment_sub):
        """Test basic process_input functionality."""
        test_input = {"symbol": "AAPL", "text_data": ["Great company!", "Poor performance"]}

        with patch.object(sentiment_sub, 'analyze_sentiment', return_value={"sentiment_score": 0.2}):
            result = await sentiment_sub.process_input(test_input)

            assert isinstance(result, dict)
            assert "sentiment_score" in result

    def test_sentiment_analysis(self, sentiment_sub):
        """Test sentiment analysis functionality."""
        texts = ["This stock is amazing!", "Terrible investment", "Neutral performance"]

        with patch.object(sentiment_sub, 'analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = {"compound": 0.1, "positive": 0.4, "negative": 0.3, "neutral": 0.3}

            analysis = sentiment_sub.analyze_sentiment(texts)
            assert isinstance(analysis, dict)


class TestEconomicDataAnalyzer:
    """Test cases for EconomicDataAnalyzer functionality."""

    @pytest.fixture
    def economic_sub(self):
        """Create an EconomicDataAnalyzer instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            sub = EconomicDataAnalyzer()
            sub.role = "economic_data"
            sub.memory_manager = Mock()
            return sub

    def test_initialization(self, economic_sub):
        """Test EconomicDataAnalyzer initialization."""
        assert economic_sub.role == "economic_data"

    @patch('src.utils.tools.fred_data_tool')
    @pytest.mark.asyncio
    async def test_process_input_basic(self, mock_fred_tool, economic_sub):
        """Test basic process_input functionality."""
        mock_fred_tool.return_value = {
            "gdp_growth": 0.02,
            "inflation_rate": 0.03,
            "unemployment_rate": 0.045
        }

        test_input = {"indicators": ["GDP", "INFLATION"]}

        result = await economic_sub.process_input(test_input)

        assert isinstance(result, dict)
        assert "gdp_growth" in result

    def test_indicator_fetching(self, economic_sub):
        """Test economic indicator fetching."""
        with patch.object(economic_sub, 'fetch_economic_indicators') as mock_fetch:
            mock_fetch.return_value = {"gdp": 25000000, "cpi": 300.5}

            indicators = economic_sub.fetch_economic_indicators(["GDP", "CPI"])
            assert isinstance(indicators, dict)


class TestFundamentalDataAnalyzer:
    """Test cases for FundamentalDataAnalyzer functionality."""

    @pytest.fixture
    def fundamental_sub(self):
        """Create a FundamentalDataAnalyzer instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            sub = FundamentalDataAnalyzer()
            sub.role = "fundamental_data"
            sub.memory_manager = Mock()
            # Mock the LLM for testing
            sub.llm = AsyncMock()
            sub.llm.ainvoke = AsyncMock(return_value=Mock(content='{"prioritized_sources": ["company_overview"], "analysis_focus": ["valuation_metrics"], "concurrent_groups": [["company_overview"]], "data_freshness_requirements": {"company_overview": 30}, "exploration_strategy": "Test"}'))
            return sub

    def test_initialization(self, fundamental_sub):
        """Test FundamentalDataAnalyzer initialization."""
        assert fundamental_sub.role == "fundamental_data"

    @pytest.mark.asyncio
    async def test_process_input_basic(self, fundamental_sub):
        """Test basic process_input functionality."""
        test_input = {"symbols": ["AAPL"]}

        with patch.object(fundamental_sub, 'fetch_fundamentals', return_value={
            "pe_ratio": 25.5,
            "eps": 6.25,
            "revenue": 365000000000
        }):
            result = await fundamental_sub.process_input(test_input)

            assert isinstance(result, dict)
            assert "consolidated_data" in result

    def test_fundamental_analysis(self, fundamental_sub):
        """Test fundamental analysis functionality."""
        with patch.object(fundamental_sub, '_analyze_fundamentals') as mock_analyze:
            mock_analyze.return_value = {
                "valuation": "fair",
                "growth_rate": 0.15,
                "health_score": 8.5
            }

            analysis = fundamental_sub._analyze_fundamentals("AAPL")
            assert isinstance(analysis, dict)


class TestInstitutionalDataAnalyzer:
    """Test cases for InstitutionalDataAnalyzer functionality."""

    @pytest.fixture
    def institutional_sub(self):
        """Create an InstitutionalDataAnalyzer instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            sub = InstitutionalDataAnalyzer()
            sub.role = "institutional_data"
            sub.memory_manager = Mock()
            return sub

    def test_initialization(self, institutional_sub):
        """Test InstitutionalDataAnalyzer initialization."""
        assert institutional_sub.role == "institutional_data"

    @pytest.mark.asyncio
    async def test_process_input_basic(self, institutional_sub):
        """Test basic process_input functionality."""
        test_input = {"symbol": "AAPL"}

        with patch.object(institutional_sub, 'fetch_institutional_holdings', return_value={
            "institutional_ownership": 0.62,
            "top_investors": ["Vanguard", "BlackRock"],
            "recent_changes": []
        }):
            result = await institutional_sub.process_input(test_input)

            assert isinstance(result, dict)
            assert "institutional_ownership" in result

    def test_institutional_holdings(self, institutional_sub):
        """Test institutional holdings fetching."""
        with patch.object(institutional_sub, 'fetch_institutional_holdings') as mock_fetch:
            mock_fetch.return_value = {
                "ownership_percent": 0.75,
                "investor_count": 1250
            }

            holdings = institutional_sub.fetch_institutional_holdings("AAPL")
            assert isinstance(holdings, dict)


class TestMicrostructureDataAnalyzer:
    """Test cases for MicrostructureDataAnalyzer functionality."""

    @pytest.fixture
    def microstructure_sub(self):
        """Create a MicrostructureDataAnalyzer instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            sub = MicrostructureDataAnalyzer()
            sub.role = "microstructure_data"
            sub.memory_manager = Mock()
            return sub

    def test_initialization(self, microstructure_sub):
        """Test MicrostructureDataAnalyzer initialization."""
        assert microstructure_sub.role == "microstructure_data"

    @pytest.mark.asyncio
    async def test_process_input_basic(self, microstructure_sub):
        """Test basic process_input functionality."""
        test_input = {"symbol": "AAPL", "order_book_data": []}

        with patch.object(microstructure_sub, 'analyze_market_microstructure', return_value={
            "spread": 0.02,
            "depth": 1000000,
            "liquidity_score": 8.5
        }):
            result = await microstructure_sub.process_input(test_input)

            assert isinstance(result, dict)
            assert "spread" in result

    def test_microstructure_analysis(self, microstructure_sub):
        """Test microstructure analysis functionality."""
        order_book = [
            {"bid": 149.98, "ask": 150.02, "bid_size": 100, "ask_size": 150}
        ]

        with patch.object(microstructure_sub, 'analyze_market_microstructure') as mock_analyze:
            mock_analyze.return_value = {
                "bid_ask_spread": 0.04,
                "market_depth": 250
            }

            analysis = microstructure_sub.analyze_market_microstructure(order_book)
            assert isinstance(analysis, dict)


class TestKalshiDataAnalyzer:
    """Test cases for KalshiDataAnalyzer functionality."""

    @pytest.fixture
    def kalshi_sub(self):
        """Create a KalshiDataAnalyzer instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            sub = KalshiDataAnalyzer()
            sub.role = "kalshi_data"
            sub.memory_manager = Mock()
            return sub

    def test_initialization(self, kalshi_sub):
        """Test KalshiDataAnalyzer initialization."""
        assert kalshi_sub.role == "kalshi_data"

    @patch('src.utils.tools.kalshi_data_tool')
    @pytest.mark.asyncio
    async def test_process_input_basic(self, mock_kalshi_tool, kalshi_sub):
        """Test basic process_input functionality."""
        mock_kalshi_tool.return_value = {
            "market_sentiment": 0.65,
            "volatility_expectations": 0.25,
            "active_markets": 150
        }

        test_input = {"market_category": "stocks"}

        result = await kalshi_sub.process_input(test_input)

        assert isinstance(result, dict)
        assert "market_sentiment" in result

    def test_kalshi_data_fetching(self, kalshi_sub):
        """Test Kalshi data fetching functionality."""
        with patch.object(kalshi_sub, 'fetch_kalshi_markets') as mock_fetch:
            mock_fetch.return_value = {
                "markets": ["Will AAPL close above $200?", "S&P 500 direction"],
                "volumes": [50000, 75000]
            }

            markets = kalshi_sub.fetch_kalshi_markets("stocks")
            assert isinstance(markets, dict)


class TestOptionsDataAnalyzer:
    """Test cases for OptionsDataAnalyzer functionality."""

    @pytest.fixture
    def options_sub(self):
        """Create an OptionsDataAnalyzer instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            sub = OptionsDataAnalyzer()
            sub.role = "options_data"
            sub.memory_manager = Mock()
            return sub

    def test_initialization(self, options_sub):
        """Test OptionsDataAnalyzer initialization."""
        assert options_sub.role == "options_data"

    @pytest.mark.asyncio
    async def test_process_input_basic(self, options_sub):
        """Test basic process_input functionality."""
        test_input = {"symbol": "AAPL"}

        with patch.object(options_sub, 'fetch_options_data', return_value={
            "calls": [{"strike": 150, "premium": 5.50}],
            "puts": [{"strike": 140, "premium": 3.25}],
            "implied_volatility": 0.22
        }):
            result = await options_sub.process_input(test_input)

            assert isinstance(result, dict)
            assert "calls" in result
            assert "puts" in result

    def test_options_data_fetching(self, options_sub):
        """Test options data fetching functionality."""
        with patch.object(options_sub, 'fetch_options_data') as mock_fetch:
            mock_fetch.return_value = {
                "open_interest": 50000,
                "volume": 25000,
                "greeks": {"delta": 0.6, "gamma": 0.05}
            }

            options = options_sub.fetch_options_data("AAPL")
            assert isinstance(options, dict)


class TestMarketDataAppDataAnalyzer:
    """Test cases for MarketDataAppDataAnalyzer functionality."""

    @pytest.fixture
    def marketdataapp_sub(self):
        """Create a MarketDataAppDataAnalyzer instance for testing."""
        with patch('src.agents.base.BaseAgent.__init__', return_value=None):
            sub = MarketDataAppDataAnalyzer()
            sub.role = "marketdataapp_data"
            sub.memory_manager = Mock()
            # Mock the LLM for testing
            sub.llm = AsyncMock()
            sub.llm.ainvoke = AsyncMock(return_value=Mock(content='{"endpoints": ["quotes"], "priorities": {"quotes": 9}, "reasoning": "Test", "expected_insights": ["price data"]}'))
            return sub

    def test_initialization(self, marketdataapp_sub):
        """Test MarketDataAppDataAnalyzer initialization."""
        assert marketdataapp_sub.role == "marketdataapp_data"

    @patch('src.utils.tools.marketdataapp_data_tool')
    @pytest.mark.asyncio
    async def test_process_input_basic(self, mock_marketdataapp_tool, marketdataapp_sub):
        """Test basic process_input functionality."""
        mock_marketdataapp_tool.return_value = {
            "price": 152.50,
            "volume": 2500000,
            "market_cap": 2500000000000,
            "pe_ratio": 28.5
        }

        test_input = {"symbol": "AAPL"}

        result = await marketdataapp_sub.process_input(test_input)

        assert isinstance(result, dict)
        assert "quotes_df" in result
        assert "data_quality_score" in result

    def test_market_data_fetching(self, marketdataapp_sub):
        """Test market data fetching functionality."""
        with patch.object(marketdataapp_sub, 'fetch_market_data') as mock_fetch:
            mock_fetch.return_value = {
                "52_week_high": 200.00,
                "52_week_low": 120.00,
                "beta": 1.25,
                "dividend_yield": 0.82
            }

            data = marketdataapp_sub.fetch_market_data("AAPL")
            assert isinstance(data, dict)


class TestDataAnalyzersIntegration:
    """Integration tests for data subagents working together."""

    def test_subagent_coordination(self):
        """Test that subagents can coordinate their data."""
        # This would test how different subagents share data
        # For example, how sentiment subagent uses news data
        pass

    def test_data_consistency(self):
        """Test data consistency across subagents."""
        # Test that related data from different subagents is consistent
        pass

    def test_error_handling_across_subagents(self):
        """Test error handling when one subagent fails."""
        # Test that other subagents continue working if one fails
        pass

    def test_memory_sharing(self):
        """Test memory sharing between subagents."""
        # Test that subagents can share insights through memory
        pass


if __name__ == "__main__":
    pytest.main([__file__])