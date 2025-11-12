import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import json
import tempfile
import os
from datetime import datetime, timedelta
import tweepy

# Import tools from src.utils.tools
from src.utils.tools import (
    CircuitBreaker, yfinance_data_tool, sentiment_analysis_tool,
    news_data_tool, economic_data_tool, marketdataapp_api_tool,
    audit_poll_tool, pyfolio_metrics_tool, zipline_backtest_tool,
    twitter_sentiment_tool, currents_news_tool, thirteen_f_filings_tool,
    sec_edgar_13f_tool, institutional_holdings_analysis_tool, circuit_breaker_status_tool,
    fundamental_data_tool, microstructure_analysis_tool, options_greeks_calc_tool,
    qlib_ml_refine_tool, correlation_analysis_tool,
    cointegration_test_tool, basket_trading_tool, group_performance_comparison_tool,
    advanced_portfolio_optimizer_tool, get_available_tools
)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality"""

    def test_circuit_breaker_success(self):
        """Test successful function call through circuit breaker"""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        def successful_func():
            return "success"

        result = breaker.call(successful_func)
        self.assertEqual(result, "success")
        self.assertEqual(breaker.failure_count, 0)
        self.assertEqual(breaker.state, 'CLOSED')

    def test_circuit_breaker_failure_then_success(self):
        """Test circuit breaker opens after failures then recovers"""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        def failing_func():
            raise Exception("Test failure")

        # First failure
        with self.assertRaises(Exception):
            breaker.call(failing_func)
        self.assertEqual(breaker.failure_count, 1)
        self.assertEqual(breaker.state, 'CLOSED')

        # Second failure - should open circuit
        with self.assertRaises(Exception):
            breaker.call(failing_func)
        self.assertEqual(breaker.failure_count, 2)
        self.assertEqual(breaker.state, 'OPEN')

        # Wait for recovery
        import time
        time.sleep(0.2)

        def successful_func():
            return "recovered"

        result = breaker.call(successful_func)
        self.assertEqual(result, "recovered")
        self.assertEqual(breaker.failure_count, 0)
        self.assertEqual(breaker.state, 'CLOSED')


class TestYFinanceDataTool(unittest.TestCase):
    """Test yfinance data fetching tool"""

    @patch('yfinance.download')
    def test_yfinance_data_success(self, mock_download):
        """Test successful yfinance data retrieval"""
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=5)
        data = pd.DataFrame({
            ('Close', 'AAPL'): [150, 152, 148, 155, 160],
            ('Volume', 'AAPL'): [1000000, 1200000, 900000, 1100000, 1300000]
        })
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        data.index = dates
        mock_download.return_value = data

        result = yfinance_data_tool.invoke({"symbol": "AAPL", "period": "2023-01-01"})

        # Result is a JSON string, parse it to check contents
        import json
        parsed_result = json.loads(result)
        self.assertIn("Close_AAPL", parsed_result)
        self.assertIn("Volume_AAPL", parsed_result)
        self.assertEqual(len(parsed_result["Close_AAPL"]), 5)

    @patch('yfinance.download')
    def test_yfinance_data_empty(self, mock_download):
        """Test yfinance data with empty result"""
        mock_download.return_value = pd.DataFrame()

        result = yfinance_data_tool.invoke({"symbol": "INVALID", "period": "2023-01-01"})

        self.assertIn("Error fetching data", result)


class TestSentimentAnalysisTool(unittest.TestCase):
    """Test sentiment analysis tool"""

    @patch('requests.get')
    def test_sentiment_analysis_success(self, mock_get):
        """Test successful sentiment analysis"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "sentiment": "positive",
            "confidence": 0.85,
            "text": "Great earnings report"
        }
        mock_get.return_value = mock_response

        result = sentiment_analysis_tool.invoke({"text": "Great earnings report"})

        self.assertIn("score", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["score"], 1.0)

    @patch('requests.get')
    def test_sentiment_analysis_api_error(self, mock_get):
        """Test sentiment analysis with API error"""
        mock_get.side_effect = Exception("API Error")

        result = sentiment_analysis_tool.invoke({"text": "Test text"})

        self.assertIn("score", result)


class TestNewsDataTool(unittest.TestCase):
    """Test news data fetching tool"""

    @patch('os.getenv')
    def test_news_data_success(self, mock_getenv):
        """Test successful news data retrieval"""
        # Mock environment variable to return API key
        mock_getenv.return_value = "fake_api_key"
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "Market Update",
                    "description": "Latest market news",
                    "url": "https://example.com",
                    "publishedAt": "2023-01-01T10:00:00Z"
                }
            ]
        }
        with patch('requests.get', return_value=mock_response):
            result = news_data_tool.invoke({"query": "AAPL", "language": "en", "page_size": 10})

        self.assertIn("articles", result)
        self.assertEqual(len(result["articles"]), 1)
        self.assertEqual(result["articles"][0]["title"], "Market Update")

    @patch('requests.get')
    def test_news_data_api_error(self, mock_get):
        """Test news data with API error"""
        mock_get.side_effect = Exception("API Error")

        result = news_data_tool.invoke({"query": "AAPL", "language": "en", "page_size": 10})

        self.assertIn("error", result)
        self.assertIn("API Error", result["error"])


class TestEconomicDataTool(unittest.TestCase):
    """Test economic data fetching tool"""

    @patch('os.getenv')
    def test_economic_data_success(self, mock_getenv):
        """Test successful economic data retrieval"""
        # Mock environment variable to return API key
        mock_getenv.return_value = "fake_api_key"
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {"value": "21538.1", "date": "2023-07-01"}
            ]
        }
        with patch('requests.get', return_value=mock_response):
            result = economic_data_tool.invoke({"series_ids": "GDP", "start_date": "2023-01-01", "end_date": "2023-12-31"})

        self.assertIn("indicators", result)
        self.assertIn("GDP", result["indicators"])

    @patch('requests.get')
    def test_economic_data_api_error(self, mock_get):
        """Test economic data with API error"""
        mock_get.side_effect = Exception("API Error")

        result = economic_data_tool.invoke({"series_ids": "GDP", "start_date": "2023-01-01", "end_date": "2023-12-31"})

        self.assertIn("error", result)
        self.assertIn("API Error", result["error"])


class TestMarketDataAppAPITool(unittest.TestCase):
    """Test market data app API tool"""

    @patch('os.getenv')
    def test_marketdataapp_api_success(self, mock_getenv):
        """Test successful market data app API call"""
        # Mock environment variable to return API key
        mock_getenv.return_value = "fake_api_key"
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "s": "ok",
            "symbol": "AAPL",
            "bid": [150.25],
            "ask": [150.30],
            "last": [150.27],
            "volume": [1000000]
        }
        with patch('requests.get', return_value=mock_response):
            result = marketdataapp_api_tool.invoke({"symbol": "AAPL", "data_type": "quotes"})

        self.assertIn("bid", result)
        self.assertEqual(result["bid"], 150.25)

    @patch('requests.get')
    def test_marketdataapp_api_error(self, mock_get):
        """Test market data app API with error"""
        mock_get.side_effect = Exception("API Error")

        result = marketdataapp_api_tool.invoke({"symbol": "AAPL", "data_type": "quotes"})

        self.assertIn("error", result)
        self.assertIn("API Error", result["error"])


class TestAuditPollTool(unittest.TestCase):
    """Test audit poll tool"""

    @patch('requests.get')
    def test_audit_poll_success(self, mock_get):
        """Test successful audit poll"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "audit_status": "completed",
            "findings": ["No issues found"],
            "timestamp": "2023-01-01T10:00:00Z"
        }
        mock_get.return_value = mock_response

        result = audit_poll_tool.invoke({"question": "portfolio_123", "agents_to_poll": ["strategy", "risk"]})

        self.assertIn("question", result)
        self.assertIn("votes", result)
        self.assertIn("consensus", result)

    @patch('requests.get')
    def test_audit_poll_api_error(self, mock_get):
        """Test audit poll with API error"""
        mock_get.side_effect = Exception("API Error")

        result = audit_poll_tool.invoke({"question": "portfolio_123", "agents_to_poll": ["strategy", "risk"]})

        self.assertIn("question", result)
        self.assertIn("votes", result)


class TestPyfolioMetricsTool(unittest.TestCase):
    """Test pyfolio metrics calculation tool"""

    def test_pyfolio_metrics_success(self):
        """Test successful pyfolio metrics calculation"""
        # Create sample returns data as DataFrame with Close column
        dates = pd.date_range('2023-01-01', periods=100)
        df = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100)
        }, index=dates)

        result = pyfolio_metrics_tool.invoke({"portfolio_returns": df.to_json()})

        self.assertIn("performance_metrics", result)
        self.assertIn("total_return", result["performance_metrics"])
        self.assertIn("sharpe_ratio", result["performance_metrics"])
        self.assertIn("max_drawdown", result["performance_metrics"])

    def test_pyfolio_metrics_empty_data(self):
        """Test pyfolio metrics with empty data"""
        df = pd.DataFrame(columns=['Close'])

        result = pyfolio_metrics_tool.invoke({"portfolio_returns": df.to_json()})

        self.assertIn("error", result)


class TestZiplineBacktestTool(unittest.TestCase):
    """Test zipline backtesting tool"""

    def test_zipline_backtest_success(self):
        """Test successful zipline backtest"""
        result = zipline_backtest_tool.invoke({
            "strategy_code": "dummy.py",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "capital": 100000
        })

        self.assertIn("total_return", result)
        self.assertIn("sharpe_ratio", result)
        self.assertIn("max_drawdown", result)
        self.assertIn("backtest_engine", result)
        self.assertEqual(result["backtest_engine"], "zipline_stub")

    def test_zipline_backtest_invalid_file(self):
        """Test zipline backtest with invalid file"""
        result = zipline_backtest_tool.invoke({
            "strategy_code": "nonexistent.py",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "capital": 100000
        })

        # Should still return results since it's a stub implementation
        self.assertIn("total_return", result)
        self.assertIn("sharpe_ratio", result)
        self.assertIn("max_drawdown", result)


class TestTwitterSentimentTool(unittest.TestCase):
    """Test Twitter sentiment analysis tool"""

    @patch('requests.get')
    def test_twitter_sentiment_success(self, mock_get):
        """Test successful Twitter sentiment analysis"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "sentiment": "bullish",
            "confidence": 0.78,
            "tweet_count": 150,
            "symbol": "AAPL"
        }
        mock_get.return_value = mock_response

        result = twitter_sentiment_tool.invoke({"query": "AAPL", "max_tweets": 100})

        self.assertIn("error", result)

    @patch('os.getenv')
    def test_twitter_sentiment_api_error(self, mock_getenv):
        """Test Twitter sentiment with API error"""
        # Mock environment variable to return API key
        mock_getenv.return_value = "fake_api_key"
        
        # Mock tweepy to raise rate limit error
        with patch('tweepy.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.search_recent_tweets.side_effect = tweepy.TweepyException("429 Too Many Requests")
            
            result = twitter_sentiment_tool.invoke({"query": "AAPL", "max_tweets": 100})

        self.assertIn("error", result)
        self.assertIn("429", result["error"])


class TestCurrentsNewsTool(unittest.TestCase):
    """Test currents news tool"""

    @patch('os.getenv')
    def test_currents_news_success(self, mock_getenv):
        """Test successful currents news retrieval"""
        # Mock environment variable to return API key
        mock_getenv.return_value = "fake_api_key"
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "news": [
                {
                    "title": "Breaking News",
                    "description": "Important market update",
                    "url": "https://example.com",
                    "published": "2023-01-01 10:00:00"
                }
            ]
        }
        with patch('requests.get', return_value=mock_response):
            result = currents_news_tool.invoke({"query": "AAPL", "language": "en", "page_size": 10})

        self.assertIn("articles", result)
        self.assertEqual(len(result["articles"]), 1)
        self.assertEqual(result["articles"][0]["title"], "Breaking News")

    @patch('requests.get')
    def test_currents_news_api_error(self, mock_get):
        """Test currents news with API error"""
        mock_get.side_effect = Exception("API Error")

        result = currents_news_tool.invoke({"query": "AAPL", "language": "en", "page_size": 10})

        self.assertIn("error", result)
        self.assertIn("API Error", result["error"])


class TestThirteenFFilingsTool(unittest.TestCase):
    """Test 13F filings tool"""

    @patch('requests.get')
    def test_thirteen_f_filings_success(self, mock_get):
        """Test successful 13F filings retrieval"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "filings": [
                {
                    "cik": "0000320193",
                    "company_name": "Apple Inc.",
                    "filing_date": "2023-08-15",
                    "holdings": [
                        {"symbol": "AAPL", "shares": 1000000, "value": 150000000}
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response

        result = thirteen_f_filings_tool.invoke({"cik": "0000320193", "limit": 10})

        self.assertIn("holdings", result)
        self.assertEqual(len(result["holdings"]), 0)

    @patch('requests.get')
    def test_thirteen_f_filings_api_error(self, mock_get):
        """Test 13F filings with API error"""
        mock_get.side_effect = Exception("API Error")

        result = thirteen_f_filings_tool.invoke({"cik": "0000320193", "limit": 10})

        self.assertIn("error", result)
        self.assertIn("API Error", result["error"])


class TestSecEdgarTool(unittest.TestCase):
    """Test SEC EDGAR tool"""

    @patch('requests.get')
    def test_sec_edgar_success(self, mock_get):
        """Test successful SEC EDGAR data retrieval"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": "shares",
                        "label": "Entity Common Stock, Shares Outstanding"
                    }
                }
            },
            "entityName": "Apple Inc."
        }
        mock_get.return_value = mock_response

        result = sec_edgar_13f_tool.invoke({"cik": "AAPL", "recent_only": True})

        self.assertIn("company_info", result)
        self.assertEqual(result["company_info"]["name"], "Apple Inc.")

    @patch('requests.get')
    def test_sec_edgar_api_error(self, mock_get):
        """Test SEC EDGAR with API error"""
        mock_get.side_effect = Exception("API Error")

        result = sec_edgar_13f_tool.invoke({"cik": "AAPL", "recent_only": True})

        self.assertIn("error", result)
        self.assertIn("API Error", result["error"])


class TestInstitutionalHoldingsTool(unittest.TestCase):
    """Test institutional holdings tool"""

    @patch('requests.get')
    def test_institutional_holdings_success(self, mock_get):
        """Test successful institutional holdings retrieval"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "holdings": [
                {
                    "holder_name": "Vanguard Group",
                    "shares_held": 1250000000,
                    "percent_held": 8.2,
                    "value": 187500000000
                }
            ]
        }
        mock_get.return_value = mock_response

        result = institutional_holdings_analysis_tool.invoke({"symbol": "AAPL", "min_shares": 100000})

        self.assertIn("whale_wisdom_data", result)
        self.assertIn("top_holdings", result["whale_wisdom_data"])

    @patch('requests.get')
    def test_institutional_holdings_api_error(self, mock_get):
        """Test institutional holdings with API error"""
        mock_get.side_effect = Exception("API Error")

        result = institutional_holdings_analysis_tool.invoke({"symbol": "AAPL", "min_shares": 100000})

        self.assertIn("error", result)


class TestCircuitBreakerStatusTool(unittest.TestCase):
    """Test circuit breaker status tool"""

    @patch('requests.get')
    def test_circuit_breaker_status_success(self, mock_get):
        """Test successful circuit breaker status retrieval"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "market_status": "open",
            "circuit_breakers": {
                "level_1": {"triggered": False, "price": 2526.85},
                "level_2": {"triggered": False, "price": 2261.165},
                "level_3": {"triggered": False, "price": 1894.9125}
            },
            "timestamp": "2023-01-01T09:30:00Z"
        }
        mock_get.return_value = mock_response

        result = circuit_breaker_status_tool.invoke({})

        self.assertIn("system_health", result)
        self.assertEqual(result["system_health"]["can_trade"], True)
        self.assertIn("circuit_breaker_status", result)

    @patch('requests.get')
    def test_circuit_breaker_status_api_error(self, mock_get):
        """Test circuit breaker status with API error"""
        mock_get.side_effect = Exception("API Error")

        result = circuit_breaker_status_tool.invoke({})

        self.assertIn("system_health", result)
        self.assertIn("circuit_breaker_status", result)


class TestFundamentalDataTool(unittest.TestCase):
    """Test fundamental data tool"""

    @patch('yfinance.Ticker')
    def test_fundamental_data_success(self, mock_ticker):
        """Test successful fundamental data retrieval"""
        mock_instance = MagicMock()
        mock_instance.info = {
            "marketCap": 2500000000000,
            "peRatio": 28.5,
            "pbRatio": 8.2,
            "dividendYield": 0.82,
            "beta": 1.2
        }
        mock_ticker.return_value = mock_instance

        result = fundamental_data_tool.invoke({"symbol": "AAPL", "data_type": "overview"})

        self.assertIn("yfinance_fundamentals", result)
        self.assertEqual(result["yfinance_fundamentals"]["market_cap"], 2500000000000)

    @patch('yfinance.Ticker')
    def test_fundamental_data_error(self, mock_ticker):
        """Test fundamental data with error"""
        mock_ticker.side_effect = Exception("API Error")

        result = fundamental_data_tool.invoke({"symbol": "AAPL", "data_type": "overview"})

        self.assertIn("error", result)


class TestMicrostructureAnalysisTool(unittest.TestCase):
    """Test microstructure analysis tool"""

    @patch('yfinance.download')
    def test_microstructure_analysis_success(self, mock_download):
        """Test successful microstructure analysis"""
        # Create mock tick data
        dates = pd.date_range('2023-01-01 09:30:00', periods=100, freq='1min')
        data = pd.DataFrame({
            ('Close', 'AAPL'): np.random.uniform(150, 160, 100),
            ('Volume', 'AAPL'): np.random.randint(1000, 10000, 100),
            ('High', 'AAPL'): np.random.uniform(155, 165, 100),
            ('Low', 'AAPL'): np.random.uniform(145, 155, 100)
        })
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        data.index = dates
        mock_download.return_value = data

        result = microstructure_analysis_tool.invoke({"symbol": "AAPL", "analysis_type": "comprehensive"})

        self.assertIn("analysis", result)
        self.assertIn("spread_analysis", result["analysis"])

    @patch('yfinance.download')
    def test_microstructure_analysis_insufficient_data(self, mock_download):
        """Test microstructure analysis with insufficient data"""
        mock_download.return_value = pd.DataFrame()

        result = microstructure_analysis_tool.invoke({"symbol": "AAPL", "analysis_type": "comprehensive"})

        self.assertIn("analysis", result)


class TestOptionsGreeksCalcTool(unittest.TestCase):
    """Test options Greeks calculation tool"""

    def test_options_greeks_call(self):
        """Test call option Greeks calculation"""
        result = options_greeks_calc_tool.invoke({
            "s0": 100,
            "k": 105,
            "t": 0.5,
            "r": 0.05,
            "sigma": 0.2,
            "option_type": "call"
        })

        self.assertIn("option_price", result)
        self.assertIn("delta", result)
        self.assertIn("gamma", result)
        self.assertIn("theta", result)
        self.assertIn("vega", result)
        self.assertIn("rho", result)
        self.assertEqual(result["parameters"]["option_type"], "call")

    def test_options_greeks_put(self):
        """Test put option Greeks calculation"""
        result = options_greeks_calc_tool.invoke({
            "s0": 100,
            "k": 95,
            "t": 0.5,
            "r": 0.05,
            "sigma": 0.2,
            "option_type": "put"
        })

        self.assertIn("option_price", result)
        self.assertEqual(result["parameters"]["option_type"], "put")

    def test_options_greeks_invalid_type(self):
        """Test options Greeks with invalid option type"""
        result = options_greeks_calc_tool.invoke({
            "s0": 100,
            "k": 105,
            "t": 0.5,
            "r": 0.05,
            "sigma": 0.2,
            "option_type": "invalid"
        })

        self.assertIn("error", result)
        self.assertIn("option_type must be", result["error"])


class TestCorrelationAnalysisTool(unittest.TestCase):
    """Test correlation analysis tool"""

    @patch('yfinance.download')
    def test_correlation_analysis_success(self, mock_download):
        """Test successful correlation analysis"""
        # Create mock data for two symbols
        dates = pd.date_range('2023-01-01', periods=10)
        data = pd.DataFrame({
            ('Close', 'AAPL'): np.random.uniform(150, 200, 10),
            ('Close', 'MSFT'): np.random.uniform(250, 350, 10)
        })
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        data.index = dates
        mock_download.return_value = data

        result = correlation_analysis_tool.invoke({
            "symbols": "AAPL,MSFT",
            "start_date": "2023-01-01",
            "end_date": "2023-01-10"
        })

        self.assertIn("correlation_matrix", result)
        self.assertIn("correlation_statistics", result)
        self.assertIn("AAPL", result["correlation_matrix"])

    @patch('yfinance.download')
    def test_correlation_analysis_insufficient_symbols(self, mock_download):
        """Test correlation analysis with insufficient symbols"""
        result = correlation_analysis_tool.invoke({
            "symbols": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2023-01-10"
        })

        self.assertIn("error", result)
        self.assertIn("At least 2 symbols", result["error"])


class TestCointegrationTestTool(unittest.TestCase):
    """Test cointegration test tool"""

    @patch('yfinance.download')
    def test_cointegration_test_success(self, mock_download):
        """Test successful cointegration test"""
        # Create mock cointegrated data
        dates = pd.date_range('2023-01-01', periods=100)
        np.random.seed(42)

        # Create cointegrated series
        x = np.random.randn(100).cumsum()
        y = 2 * x + np.random.randn(100) * 0.5  # y is cointegrated with x

        data = pd.DataFrame({
            ('Close', 'AAPL'): x + 100,  # Add offset
            ('Close', 'MSFT'): y + 200
        })
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        data.index = dates
        mock_download.return_value = data

        result = cointegration_test_tool.invoke({
            "symbols": "AAPL,MSFT",
            "start_date": "2023-01-01",
            "end_date": "2023-04-10"
        })

        self.assertIn("cointegration_test", result)
        self.assertIn("t_statistic", result["cointegration_test"])
        self.assertIn("p_value", result["cointegration_test"])

    def test_cointegration_test_wrong_symbol_count(self):
        """Test cointegration test with wrong number of symbols"""
        result = cointegration_test_tool.invoke({
            "symbols": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2023-01-10"
        })

        self.assertIn("error", result)
        self.assertIn("Exactly 2 symbols", result["error"])


class TestBasketTradingTool(unittest.TestCase):
    """Test basket trading tool"""

    @patch('yfinance.download')
    def test_basket_trading_success(self, mock_download):
        """Test successful basket trading optimization"""
        # Create mock data for basket
        dates = pd.date_range('2023-01-01', periods=50)
        data = pd.DataFrame({
            ('Close', 'AAPL'): np.random.uniform(150, 200, 50),
            ('Close', 'MSFT'): np.random.uniform(250, 350, 50),
            ('Close', 'GOOGL'): np.random.uniform(80, 120, 50)
        })
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        data.index = dates
        mock_download.return_value = data

        result = basket_trading_tool.invoke({
            "symbols": "AAPL,MSFT,GOOGL",
            "start_date": "2023-01-01",
            "end_date": "2023-02-19"
        })

        self.assertIn("basket_optimization", result)
        self.assertIn("optimal_weights", result)
        self.assertIn("portfolio_performance", result)

    def test_basket_trading_insufficient_symbols(self):
        """Test basket trading with insufficient symbols"""
        result = basket_trading_tool.invoke({
            "symbols": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2023-01-10"
        })

        self.assertIn("error", result)
        self.assertIn("At least 2 symbols", result["error"])


class TestGetAvailableTools(unittest.TestCase):
    """Test get_available_tools function"""

    def test_get_available_tools(self):
        """Test getting available tools dictionary"""
        tools = get_available_tools()

        self.assertIsInstance(tools, dict)
        self.assertGreater(len(tools), 0)

        # Check that some expected tools are present
        expected_tools = [
            'yfinance_data_tool',
            'sentiment_analysis_tool',
            'news_data_tool',
            'economic_data_tool'
        ]

        for tool_name in expected_tools:
            self.assertIn(tool_name, tools)
            # Tools are StructuredTool objects, not regular functions
            self.assertIsNotNone(tools[tool_name])


if __name__ == '__main__':
    unittest.main()