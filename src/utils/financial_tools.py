#!/usr/bin/env python3
"""
Financial data collection and analysis tools.
Provides tools for stock data, sentiment analysis, risk calculation, and strategy proposals.
"""

import os
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import requests

from .validation import circuit_breaker, DataValidator
from .constants import DEFAULT_API_TIMEOUT, ERROR_NO_DATA_FOUND
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class YFinanceDataInput(BaseModel):
    symbol: str = Field(description="Stock ticker symbol")
    period: str = Field(default="2y", description="Time period for data (e.g., '1y', '2y', '5y')")


class YFinanceDataTool(BaseTool):
    name: str = "yfinance_data_tool"
    description: str = "Fetch historical stock data using yfinance."
    args_schema: type = YFinanceDataInput

    def _run(self, symbol: str, period: str = "2y") -> Dict[str, Any]:
        """
        Fetch historical stock data using yfinance.
        Args:
            symbol: Stock ticker symbol
            period: Time period for data (e.g., '1y', '2y', '5y')
        Returns:
            dict: Stock data or error message
        """
        try:
            import yfinance as yf

            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                return {"error": "Invalid stock symbol provided"}

            # Fetch data
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)

            if data.empty:
                return {"error": f"{ERROR_NO_DATA_FOUND} for symbol: {symbol}"}

            # Format response
            latest = data.iloc[-1] if not data.empty else None
            if latest is not None:
                response = {
                    "stock": symbol.upper(),
                    "period": period,
                    "latest_price": round(float(latest['Close']), 2),
                    "volume": int(latest['Volume']),
                    "market_cap": stock.info.get('marketCap', 'N/A')
                }
                return response
            else:
                return {"error": ERROR_NO_DATA_FOUND}

        except Exception as e:
            logger.error(f"Error fetching yfinance data: {e}")
            return {"error": str(e)}


yfinance_data_tool = YFinanceDataTool()


class SentimentAnalysisInput(BaseModel):
    text: str = Field(description="Text to analyze for sentiment")


class SentimentAnalysisTool(BaseTool):
    name: str = "sentiment_analysis_tool"
    description: str = "Analyze sentiment of text using multiple methods."
    args_schema: type = SentimentAnalysisInput

    def _run(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using multiple methods.
        Args:
            text: Text to analyze
        Returns:
            dict: Sentiment analysis results
        """
        try:
            # Validate input
            text = DataValidator.sanitize_text_input(text)
            if not text:
                return {"error": "No valid text provided for sentiment analysis"}

            # Primary: xAI Grok API
            grok_key = os.getenv('GROK_API_KEY')
            if grok_key:
                try:
                    import requests

                    headers = {
                        'Authorization': f'Bearer {grok_key}',
                        'Content-Type': 'application/json'
                    }

                    payload = {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Analyze the sentiment of this text and provide a score from -1 (very negative) to 1 (very positive), plus a brief explanation: {text[:1000]}"
                            }
                        ],
                        "model": "grok-beta",
                        "temperature": 0.1
                    }

                    response = requests.post(
                        'https://api.x.ai/v1/chat/completions',
                        headers=headers,
                        json=payload,
                        timeout=DEFAULT_API_TIMEOUT
                    )

                    if response.status_code == 200:
                        data = response.json()
                        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

                        # Parse sentiment score
                        import re
                        score_match = re.search(r'-?\d+\.?\d*', content)
                        if score_match:
                            score = float(score_match.group())
                            score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
                            return {
                                "sentiment_score": score,
                                "confidence": 0.9,
                                "method": "xAI Grok",
                                "explanation": content[:200],
                                "text_length": len(text)
                            }

                except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.error(f"xAI API call failed: {e}")

        except Exception as e:
            logger.error(f"Sentiment analysis completely failed: {str(e)}")
            return {"error": f"Sentiment analysis failed: {str(e)}"}

        # Fallback: Rule-based sentiment analysis
        try:
            from textblob import TextBlob

            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            return {
                "sentiment_score": polarity,
                "confidence": 0.6,
                "method": "TextBlob (fallback)",
                "explanation": f"Rule-based analysis: polarity = {polarity:.3f}",
                "text_length": len(text)
            }

        except Exception as e:
            return {
                "error": f"All sentiment analysis methods failed: {str(e)}",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "method": "failed",
                "text_length": len(text)
            }


sentiment_analysis_tool = SentimentAnalysisTool()


class RiskCalculationInput(BaseModel):
    data: str = Field(description="Financial data as CSV string with Close column")


class RiskCalculationTool(BaseTool):
    name: str = "risk_calculation_tool"
    description: str = "Calculate risk metrics from financial data."
    args_schema: type = RiskCalculationInput

    def _run(self, data: str) -> Dict[str, Any]:
        """
        Calculate risk metrics from financial data.
        Args:
            data: Financial data as string
        Returns:
            dict: Risk metrics
        """
        try:
            # Parse data (assuming CSV format)
            import io
            df = pd.read_csv(io.StringIO(data))

            if df.empty:
                return {"error": "No data provided for risk calculation"}

            # Calculate basic risk metrics
            if 'Close' in df.columns:
                returns = df['Close'].pct_change().dropna()

                if len(returns) > 0:
                    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                    sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
                    max_drawdown = (df['Close'] / df['Close'].cummax() - 1).min()

                    return {
                        "volatility": volatility,
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                        "total_return": (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) if len(df) > 1 else 0,
                        "data_points": len(df)
                    }

            return {"error": "Invalid data format - missing Close column"}

        except Exception as e:
            return {"error": f"Risk calculation failed: {str(e)}"}


risk_calculation_tool = RiskCalculationTool()


class StrategyProposalInput(BaseModel):
    data: str = Field(description="Financial data as string")
    sentiment: Dict[str, Any] = Field(description="Sentiment analysis results dictionary")


class StrategyProposalTool(BaseTool):
    name: str = "strategy_proposal_tool"
    description: str = "Propose trading strategies based on data and sentiment."
    args_schema: type = StrategyProposalInput

    def _run(self, data: str, sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose trading strategies based on data and sentiment.
        Args:
            data: Financial data
            sentiment: Sentiment analysis results
        Returns:
            dict: Strategy recommendations
        """
        try:
            # Parse sentiment
            sentiment_score = sentiment.get('sentiment_score', 0)
            confidence = sentiment.get('confidence', 0)

            # Basic strategy logic based on sentiment
            if confidence > 0.7:
                if sentiment_score > 0.2:
                    strategy = "bullish"
                    action = "Consider long positions"
                elif sentiment_score < -0.2:
                    strategy = "bearish"
                    action = "Consider short positions or hedging"
                else:
                    strategy = "neutral"
                    action = "Hold current positions"
            else:
                strategy = "uncertain"
                action = "Wait for clearer signals"

            return {
                "strategy": strategy,
                "action": action,
                "sentiment_score": sentiment_score,
                "confidence": confidence,
                "risk_level": "high" if abs(sentiment_score) > 0.5 else "medium" if abs(sentiment_score) > 0.2 else "low"
            }

        except Exception as e:
            return {"error": f"Strategy proposal failed: {str(e)}"}


strategy_proposal_tool = StrategyProposalTool()


class FundamentalAnalysisInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")


class FundamentalAnalysisTool(BaseTool):
    name: str = "fundamental_analysis_tool"
    description: str = "Perform fundamental analysis on a stock."
    args_schema: type = FundamentalAnalysisInput

    def _run(self, ticker: str) -> Dict[str, Any]:
        """
        Perform fundamental analysis on a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with analysis results
        """
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "pe_ratio": info.get("forwardPE", "N/A"),
                "eps": info.get("forwardEps", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "revenue_growth": info.get("revenueGrowth", "N/A")
            }
        except Exception as e:
            return {"error": str(e)}


fundamental_analysis_tool = FundamentalAnalysisTool()


class OptionsGreeksCalcInput(BaseModel):
    s0: float = Field(description="Spot price")
    k: float = Field(description="Strike price")
    t: float = Field(description="Time to expiration in years")
    r: float = Field(description="Risk-free rate")
    sigma: float = Field(description="Volatility")
    option_type: str = Field(description="Option type: 'call' or 'put'")


class OptionsGreeksCalcTool(BaseTool):
    name: str = "options_greeks_calc_tool"
    description: str = "Calculate options Greeks."
    args_schema: type = OptionsGreeksCalcInput

    def _run(self, s0: float, k: float, t: float, r: float, sigma: float, option_type: str) -> Dict[str, Any]:
        """
        Calculate options Greeks.

        Args:
            s0: Spot price
            k: Strike price
            t: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Dict with Greeks (delta, gamma, theta, vega, rho)
        """
        try:
            from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho
            from py_vollib.black_scholes import black_scholes
            flag = 'c' if option_type == 'call' else 'p'
            option_price = black_scholes(flag, s0, k, t, r, sigma)
            greeks = {
                "option_price": option_price,
                "delta": delta(flag, s0, k, t, r, sigma),
                "gamma": gamma(flag, s0, k, t, r, sigma),
                "theta": theta(flag, s0, k, t, r, sigma),
                "vega": vega(flag, s0, k, t, r, sigma),
                "rho": rho(flag, s0, k, t, r, sigma),
                "parameters": {
                    "s0": s0,
                    "k": k,
                    "t": t,
                    "r": r,
                    "sigma": sigma,
                    "option_type": option_type
                }
            }
            return greeks
        except Exception as e:
            return {"error": str(e)}


options_greeks_calc_tool = OptionsGreeksCalcTool()


class FlowAlphaCalcInput(BaseModel):
    flow_data: Dict[str, Any] = Field(description="Dictionary with buy_volume, sell_volume, price")


class FlowAlphaCalcTool(BaseTool):
    name: str = "flow_alpha_calc_tool"
    description: str = "Calculate flow alpha from order flow data."
    args_schema: type = FlowAlphaCalcInput

    def _run(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate flow alpha from order flow data.

        Args:
            flow_data: Dict with buy_volume, sell_volume, price

        Returns:
            Dict with flow alpha metrics
        """
        try:
            net_flow = flow_data['buy_volume'] - flow_data['sell_volume']
            alpha = net_flow / (flow_data['buy_volume'] + flow_data['sell_volume']) if (flow_data['buy_volume'] + flow_data['sell_volume']) > 0 else 0
            return {"net_flow": net_flow, "flow_alpha": alpha}
        except Exception as e:
            return {"error": str(e)}


flow_alpha_calc_tool = FlowAlphaCalcTool()


class CorrelationAnalysisInput(BaseModel):
    symbols: str = Field(description="Comma-separated list of stock tickers")
    start_date: str = Field(description="Start date for analysis")
    end_date: str = Field(description="End date for analysis")


class CorrelationAnalysisTool(BaseTool):
    name: str = "correlation_analysis_tool"
    description: str = "Perform correlation analysis between assets."
    args_schema: type = CorrelationAnalysisInput

    def _run(self, symbols: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Perform correlation analysis between assets.

        Args:
            symbols: Comma-separated list of stock tickers
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dict with correlation matrix
        """
        try:
            import yfinance as yf
            tickers = [s.strip() for s in symbols.split(',')]
            if len(tickers) < 2:
                return {"error": "At least 2 symbols required for correlation analysis"}

            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            corr = data.pct_change().corr()
            return {
                "correlation_matrix": corr.to_dict(),
                "correlation_statistics": {
                    "symbols": tickers,
                    "period": f"{start_date} to {end_date}"
                }
            }
        except Exception as e:
            return {"error": str(e)}


correlation_analysis_tool = CorrelationAnalysisTool()


class CointegrationTestInput(BaseModel):
    symbols: str = Field(description="Comma-separated list of exactly 2 stock tickers")
    start_date: str = Field(description="Start date for analysis")
    end_date: str = Field(description="End date for analysis")


class CointegrationTestTool(BaseTool):
    name: str = "cointegration_test_tool"
    description: str = "Perform cointegration test between two assets."
    args_schema: type = CointegrationTestInput

    def _run(self, symbols: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Perform cointegration test between two assets.

        Args:
            symbols: Comma-separated list of exactly 2 stock tickers
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dict with test results
        """
        try:
            import yfinance as yf
            from statsmodels.tsa.stattools import coint
            tickers = [s.strip() for s in symbols.split(',')]
            if len(tickers) != 2:
                return {"error": "Exactly 2 symbols required for cointegration test"}

            data1 = yf.download(tickers[0], start=start_date, end=end_date)['Close']
            data2 = yf.download(tickers[1], start=start_date, end=end_date)['Close']
            score, pvalue, _ = coint(data1, data2)
            return {
                "cointegration_test": {
                    "t_statistic": score,
                    "p_value": pvalue,
                    "cointegrated": pvalue < 0.05
                },
                "symbols": tickers
            }
        except Exception as e:
            return {"error": str(e)}


cointegration_test_tool = CointegrationTestTool()


class BasketTradingInput(BaseModel):
    symbols: str = Field(description="Comma-separated list of tickers")
    start_date: str = Field(description="Start date for analysis (YYYY-MM-DD)")
    end_date: str = Field(description="End date for analysis (YYYY-MM-DD)")


class BasketTradingTool(BaseTool):
    name: str = "basket_trading_tool"
    description: str = "Create a trading basket with equal weights."
    args_schema: type = BasketTradingInput

    def _run(self, symbols: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Create a trading basket with equal weights.

        Args:
            symbols: Comma-separated list of tickers
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dict with basket details
        """
        try:
            basket = [s.strip() for s in symbols.split(',')]
            if len(basket) < 2:
                return {"error": "At least 2 symbols required for basket trading"}

            weights = [1.0 / len(basket)] * len(basket)  # Equal weights
            return {
                "basket_optimization": {
                    "symbols": basket,
                    "optimal_weights": dict(zip(basket, weights))
                },
                "portfolio_performance": {
                    "expected_return": 0.0,  # Placeholder
                    "volatility": 0.0,  # Placeholder
                    "sharpe_ratio": 0.0  # Placeholder
                },
                "start_date": start_date,
                "end_date": end_date
            }
        except Exception as e:
            return {"error": str(e)}


basket_trading_tool = BasketTradingTool()


class AdvancedPortfolioOptimizerInput(BaseModel):
    assets: List[str] = Field(description="List of asset tickers")
    constraints: Dict[str, Any] = Field(description="Optimization constraints dictionary")


class AdvancedPortfolioOptimizerTool(BaseTool):
    name: str = "advanced_portfolio_optimizer_tool"
    description: str = "Perform advanced portfolio optimization."
    args_schema: type = AdvancedPortfolioOptimizerInput

    def _run(self, assets: List[str], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform advanced portfolio optimization.

        Args:
            assets: List of assets
            constraints: Optimization constraints

        Returns:
            Dict with optimal weights
        """
        try:
            from pypfopt import EfficientFrontier, risk_models, expected_returns
            import yfinance as yf
            data = yf.download(assets)['Close']
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
            return {"weights": weights}
        except Exception as e:
            return {"error": str(e)}


advanced_portfolio_optimizer_tool = AdvancedPortfolioOptimizerTool()

# end of file