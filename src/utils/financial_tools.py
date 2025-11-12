#!/usr/bin/env python3
"""
Financial data collection and analysis tools.
Provides tools for stock data, sentiment analysis, risk calculation, and strategy proposals.
"""

import os
import logging
from typing import Dict, Any, Optional
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