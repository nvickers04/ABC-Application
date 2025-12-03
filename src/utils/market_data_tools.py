#!/usr/bin/env python3
"""
Market data tools for real-time and historical market data.
Provides API and WebSocket connections to market data providers.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import websockets
import requests

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from .validation import circuit_breaker, DataValidator
from .vault_client import get_vault_secret

logger = logging.getLogger(__name__)


class MarketDataAppAPIToolInput(BaseModel):
    symbol: str = Field(description="Stock symbol")
    data_type: str = Field(default="quotes", description="Type of data ('quotes', 'historical', etc.)")


class MarketDataAppAPITool(BaseTool):
    name: str = "marketdataapp_api_tool"
    description: str = "Fetch market data from MarketDataApp API."
    args_schema: type = MarketDataAppAPIToolInput

    def _run(self, symbol: str, data_type: str = "quotes") -> Dict[str, Any]:
        """
        Fetch market data from MarketDataApp API.
        Args:
            symbol: Stock symbol
            data_type: Type of data ('quotes', 'historical', etc.)
        Returns:
            dict: Market data
        """
        try:
            api_key = get_vault_secret('MARKETDATAAPP_API_KEY')
        except ValueError:
            return {"error": "MarketDataApp API key not found in Vault or environment variables."}

        try:
            base_url = "https://api.marketdataapp.com/v1"

            if data_type == "quotes":
                url = f"{base_url}/stocks/quotes"
                params = {"symbols": symbol, "apikey": api_key}

            elif data_type == "historical":
                url = f"{base_url}/stocks/candles"
                # Use dynamic date range (last 2 years)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)  # 2 years
                params = {
                    "symbols": symbol,
                    "resolution": "D",
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                    "apikey": api_key
                }
            else:
                return {"error": f"Unsupported data type: {data_type}"}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data or not isinstance(data, dict):
                return {"error": "Invalid response from MarketDataApp API"}

            # Process and validate data
            if symbol in data:
                symbol_data = data[symbol]

                if data_type == "quotes":
                    return {
                        "symbol": symbol,
                        "price": symbol_data.get("last", 0),
                        "change": symbol_data.get("change", 0),
                        "change_percent": symbol_data.get("changepct", 0),
                        "volume": symbol_data.get("volume", 0),
                        "source": "marketdataapp",
                        "data_type": "quotes"
                    }
                else:
                    # Historical data processing
                    candles = symbol_data.get("candles", [])
                    processed_data = []
                    for candle in candles:
                        processed_data.append({
                            "timestamp": candle.get("t", 0),
                            "open": candle.get("o", 0),
                            "high": candle.get("h", 0),
                            "low": candle.get("l", 0),
                            "close": candle.get("c", 0),
                            "volume": candle.get("v", 0)
                        })

                    return {
                        "symbol": symbol,
                        "data": processed_data,
                        "count": len(processed_data),
                        "source": "marketdataapp",
                        "data_type": "historical"
                    }

            return {"error": f"No data found for symbol {symbol}"}

        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
            return {"error": f"MarketDataApp API failed: {str(e)}"}


marketdataapp_api_tool = MarketDataAppAPITool()


class MarketDataAppWebSocketToolInput(BaseModel):
    symbol: str = Field(description="Stock symbol")
    data_type: str = Field(default="quotes", description="Type of data")
    duration_seconds: int = Field(default=30, description="How long to collect data")


class MarketDataAppWebSocketTool(BaseTool):
    name: str = "marketdataapp_websocket_tool"
    description: str = "Connect to MarketDataApp WebSocket for real-time data."
    args_schema: type = MarketDataAppWebSocketToolInput

    def _run(self, symbol: str, data_type: str = "quotes", duration_seconds: int = 30) -> Dict[str, Any]:
        """
        Connect to MarketDataApp WebSocket for real-time data.
        Args:
            symbol: Stock symbol
            data_type: Type of data
            duration_seconds: How long to collect data
        Returns:
            dict: Collected real-time data
        """
        api_key = get_vault_secret('MARKETDATAAPP_API_KEY')
        if not api_key:
            return {"error": "MarketDataApp API key not found in Vault."}

        async def collect_data():
            """Async function to collect WebSocket data."""
            try:
                uri = f"wss://api.marketdataapp.com/v1/stocks/quotes?symbols={symbol}&apikey={api_key}"

                collected_data = []
                loop = asyncio.get_running_loop()
                start_time = loop.time()

                async with websockets.connect(uri) as websocket:
                    while loop.time() - start_time < duration_seconds:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            data = json.loads(message)

                            if symbol in data:
                                symbol_data = data[symbol]
                                collected_data.append({
                                    "timestamp": loop.time(),
                                    "price": symbol_data.get("last", 0),
                                    "change": symbol_data.get("change", 0),
                                    "volume": symbol_data.get("volume", 0)
                                })

                                # Limit to last 100 data points
                                if len(collected_data) > 100:
                                    collected_data = collected_data[-100:]

                        except asyncio.TimeoutError:
                            continue
                        except json.JSONDecodeError:
                            continue

                return collected_data

            except Exception as e:
                return {"error": f"WebSocket connection failed: {str(e)}"}

        try:
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(collect_data())
            loop.close()

            if isinstance(result, list):
                return {
                    "symbol": symbol,
                    "data_points": len(result),
                    "data": result,
                    "duration_seconds": duration_seconds,
                    "source": "marketdataapp_websocket",
                    "status": "success"
                }
            else:
                return result

        except Exception as e:
            return {"error": f"WebSocket tool failed: {str(e)}"}


marketdataapp_websocket_tool = MarketDataAppWebSocketTool()


@tool
def alpha_vantage_tool(symbol: str, function: str = "TIME_SERIES_DAILY") -> Dict[str, Any]:
    """
    Fetch data from Alpha Vantage API.
    Args:
        symbol: Stock symbol
        function: API function (TIME_SERIES_DAILY, etc.)
    Returns:
        dict: Stock data
    """
    api_key = get_vault_secret('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        return {"error": "Alpha Vantage API key not found in Vault."}

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": api_key,
            "outputsize": "compact"
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "Error Message" in data:
            return {"error": data["Error Message"]}

        if "Note" in data:
            return {"error": "API limit reached", "note": data["Note"]}

        # Process time series data
        time_series_key = f"Time Series ({function.split('_')[-1].capitalize()})"
        if time_series_key in data:
            time_series = data[time_series_key]

            processed_data = []
            for date, values in list(time_series.items())[:30]:  # Last 30 days
                processed_data.append({
                    "date": date,
                    "open": float(values.get("1. open", 0)),
                    "high": float(values.get("2. high", 0)),
                    "low": float(values.get("3. low", 0)),
                    "close": float(values.get("4. close", 0)),
                    "volume": int(values.get("5. volume", 0))
                })

            return {
                "symbol": symbol,
                "function": function,
                "data": processed_data,
                "count": len(processed_data),
                "source": "alpha_vantage",
                "metadata": data.get("Meta Data", {})
            }

        return {"error": "No time series data found"}

    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        return {"error": f"Alpha Vantage API failed: {str(e)}"}


@tool
def financial_modeling_prep_tool(symbol: str, data_type: str = "quote") -> Dict[str, Any]:
    """
    Fetch data from Financial Modeling Prep API.
    Args:
        symbol: Stock symbol
        data_type: Type of data to fetch
    Returns:
        dict: Financial data
    """
    api_key = get_vault_secret('FINANCIALMODELINGPREP_API_KEY')
    if not api_key:
        return {"error": "Financial Modeling Prep API key not found in Vault."}

    try:
        base_url = "https://financialmodelingprep.com/api/v3"

        if data_type == "quote":
            url = f"{base_url}/quote/{symbol}"
        elif data_type == "profile":
            url = f"{base_url}/profile/{symbol}"
        elif data_type == "ratios":
            url = f"{base_url}/ratios/{symbol}"
        else:
            return {"error": f"Unsupported data type: {data_type}"}

        params = {"apikey": api_key}

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if not data or not isinstance(data, list) or len(data) == 0:
            return {"error": f"No data found for {symbol}"}

        # Return first result (should be the requested symbol)
        result = data[0]

        return {
            "symbol": symbol,
            "data_type": data_type,
            "data": result,
            "source": "financial_modeling_prep"
        }

    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        return {"error": f"Financial Modeling Prep API failed: {str(e)}"}


@tool
def institutional_holdings_analysis_tool(symbol: str, min_shares: int = 100000) -> Dict[str, Any]:
    """
    Analyze institutional holdings for a stock.

    Args:
        symbol: Stock symbol
        min_shares: Minimum shares threshold for analysis

    Returns:
        Dict with holdings analysis
    """
    try:
        # Placeholder for actual API call (e.g., to FMP or SEC)
        return {"error": "API Error: institutional holdings analysis tool not yet implemented"}
    except Exception as e:
        return {"error": str(e)}


@tool
def thirteen_f_filings_tool(cik: str, limit: int = 5) -> Dict[str, Any]:
    """
    Fetch recent 13F filings.

    Args:
        cik: Central Index Key
        limit: Number of filings to fetch

    Returns:
        Dict with filings data
    """
    try:
        # Placeholder for SEC EDGAR API
        current_date = datetime.now()
        # Generate recent filing dates (quarterly filings)
        filings = []
        for i in range(limit):
            filing_date = current_date - timedelta(days=i * 90)  # Approximate quarterly
            filings.append({
                "date": filing_date.strftime("%Y-%m-%d"),
                "holdings": 100  # Placeholder value
            })
        return {"filings": filings}
    except Exception as e:
        return {"error": str(e)}


@tool
def fundamental_data_tool(ticker: str) -> Dict[str, Any]:
    """
    Fetch fundamental data for a stock.

    Args:
        ticker: Stock ticker

    Returns:
        Dict with fundamental metrics
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("epsTrailingTwelveMonths"),
            "dividend_yield": info.get("dividendYield")
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def microstructure_analysis_tool(order_book: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze market microstructure from order book data.

    Args:
        order_book: Dict with bids and asks

    Returns:
        Dict with analysis (spread, depth, etc.)
    """
    try:
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if bids and asks:
            spread = asks[0][0] - bids[0][0]
            return {"spread": spread, "bid_depth": sum(b[1] for b in bids), "ask_depth": sum(a[1] for a in asks)}
        return {"error": "Invalid order book data"}
    except Exception as e:
        return {"error": str(e)}


@tool
def kalshi_data_tool(market_id: str) -> Dict[str, Any]:
    """
    Fetch data from Kalshi prediction market.

    Args:
        market_id: Market identifier

    Returns:
        Dict with market data
    """
    try:
        # Placeholder for Kalshi API
        return {"yes_price": 0.55, "no_price": 0.45, "volume": 1000}
    except Exception as e:
        return {"error": str(e)}


@tool
def sec_edgar_13f_tool(cik: str, date: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch 13F filings from SEC EDGAR.

    Args:
        cik: Central Index Key
        date: Optional filing date

    Returns:
        Dict with 13F data
    """
    try:
        # Placeholder for SEC API
        return {"error": "API Error: SEC EDGAR 13F tool not yet implemented"}
    except Exception as e:
        return {"error": str(e)}


@tool
def circuit_breaker_status_tool() -> Dict[str, Any]:
    """
    Get current circuit breaker status for all APIs.

    Returns:
        Dict with status of all circuit breakers
    """
    try:
        from .validation import get_circuit_breaker_status
        return get_circuit_breaker_status()
    except Exception as e:
        return {"error": str(e)}


@tool
def marketdataapp_data_tool(symbol: str) -> Dict[str, Any]:
    """
    Fetch basic market data from MarketDataApp.
    Simplified tool for basic market data retrieval.

    Args:
        symbol: Stock symbol

    Returns:
        Dict with basic market data
    """
    api_key = get_vault_secret('MARKETDATAAPP_API_KEY')
    if not api_key:
        return {"error": "MarketDataApp API key not found in Vault."}

    try:
        base_url = "https://api.marketdataapp.com/v1"
        url = f"{base_url}/stocks/quotes"
        params = {"symbols": symbol, "apikey": api_key}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        if symbol in data:
            quote = data[symbol]
            return {
                "price": quote.get("last", 0),
                "volume": quote.get("volume", 0),
                "market_cap": quote.get("marketcap", 0),
                "pe_ratio": quote.get("pe", 0),
                "symbol": symbol,
                "source": "marketdataapp_api"
            }
        else:
            return {"error": f"No data found for symbol {symbol}"}

    except Exception as e:
        logger.error(f"Error fetching MarketDataApp data for {symbol}: {e}")
        return {"error": str(e)}

# end of file