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

logger = logging.getLogger(__name__)


@circuit_breaker("yfinance")
def yfinance_data_tool(symbol: str, period: str = "2y") -> str:
    """
    Fetch historical stock data using yfinance.
    Args:
        symbol: Stock ticker symbol
        period: Time period for data (e.g., '1y', '2y', '5y')
    Returns:
        str: Formatted stock data or error message
    """
    try:
        import yfinance as yf

        # Validate inputs
        if not symbol or not isinstance(symbol, str):
            return "Error: Invalid stock symbol provided"

        # Fetch data
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)

        if data.empty:
            return f"No data found for symbol: {symbol}"

        # Format response
        latest = data.iloc[-1] if not data.empty else None
        if latest is not None:
            response = f"""
Stock: {symbol.upper()}
Period: {period}
Latest Price: ${latest['Close']:.2f}
Volume: {latest['Volume']:,.0f}
Market Cap: {stock.info.get('marketCap', 'N/A')}

Recent Performance:
{data.tail(5).to_string()}
"""
            return response.strip()
        else:
            return f"Unable to retrieve data for {symbol}"

    except (requests.RequestException, ValueError, KeyError, AttributeError, pd.errors.EmptyDataError) as e:
        logger.warning(f"yfinance failed for {symbol}: {e}")
        return f"Error fetching data for {symbol}: {str(e)}"


@circuit_breaker("sentiment_analysis")
def sentiment_analysis_tool(text: str) -> Dict[str, Any]:
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
                    timeout=30
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


def risk_calculation_tool(data: str) -> Dict[str, Any]:
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


def strategy_proposal_tool(data: str, sentiment: Dict[str, Any]) -> Dict[str, Any]:
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


@circuit_breaker("fundamental_analysis")
def fundamental_analysis_tool(ticker: str) -> Dict[str, Any]:
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


@circuit_breaker("options_greeks_calc")
def options_greeks_calc_tool(option_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate options Greeks.
    
    Args:
        option_data: Dict with option parameters (strike, spot, time, vol, rate)
        
    Returns:
        Dict with Greeks (delta, gamma, theta, vega, rho)
    """
    try:
        from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho
        flag = 'c' if option_data.get('type', 'call') == 'call' else 'p'
        greeks = {
            "delta": delta(flag, option_data['spot'], option_data['strike'], option_data['time'], option_data['rate'], option_data['vol']),
            "gamma": gamma(flag, option_data['spot'], option_data['strike'], option_data['time'], option_data['rate'], option_data['vol']),
            "theta": theta(flag, option_data['spot'], option_data['strike'], option_data['time'], option_data['rate'], option_data['vol']),
            "vega": vega(flag, option_data['spot'], option_data['strike'], option_data['time'], option_data['rate'], option_data['vol']),
            "rho": rho(flag, option_data['spot'], option_data['strike'], option_data['time'], option_data['rate'], option_data['vol'])
        }
        return greeks
    except Exception as e:
        return {"error": str(e)}


@circuit_breaker("flow_alpha_calc")
def flow_alpha_calc_tool(flow_data: Dict[str, Any]) -> Dict[str, Any]:
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


@circuit_breaker("correlation_analysis")
def correlation_analysis_tool(tickers: List[str], period: str = "1y") -> Dict[str, Any]:
    """
    Perform correlation analysis between assets.
    
    Args:
        tickers: List of stock tickers
        period: Time period
        
    Returns:
        Dict with correlation matrix
    """
    try:
        import yfinance as yf
        data = yf.download(tickers, period=period)['Close']
        corr = data.pct_change().corr()
        return {"correlation_matrix": corr.to_dict()}
    except Exception as e:
        return {"error": str(e)}


@circuit_breaker("cointegration_test")
def cointegration_test_tool(ticker1: str, ticker2: str, period: str = "1y") -> Dict[str, Any]:
    """
    Perform cointegration test between two assets.
    
    Args:
        ticker1: First ticker
        ticker2: Second ticker
        period: Time period
        
    Returns:
        Dict with test results
    """
    try:
        import yfinance as yf
        from statsmodels.tsa.stattools import coint
        data1 = yf.download(ticker1, period=period)['Close']
        data2 = yf.download(ticker2, period=period)['Close']
        score, pvalue, _ = coint(data1, data2)
        return {"cointegrated": pvalue < 0.05, "pvalue": pvalue, "score": score}
    except Exception as e:
        return {"error": str(e)}


@circuit_breaker("basket_trading")
def basket_trading_tool(basket: List[str], weights: List[float]) -> Dict[str, Any]:
    """
    Create a trading basket with weights.
    
    Args:
        basket: List of tickers
        weights: List of weights (must sum to 1)
        
    Returns:
        Dict with basket details
    """
    try:
        if sum(weights) != 1:
            raise ValueError("Weights must sum to 1")
        return {"basket": dict(zip(basket, weights))}
    except Exception as e:
        return {"error": str(e)}


@circuit_breaker("advanced_portfolio_optimizer")
def advanced_portfolio_optimizer_tool(assets: List[str], constraints: Dict[str, Any]) -> Dict[str, Any]:
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

# end of file