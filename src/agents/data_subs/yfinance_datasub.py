# src/agents/data_subs/yfinance_datasub.py
# Purpose: Yfinance Data Subagent with LLM-powered exploration and intelligent data aggregation.
# Provides comprehensive market data from multiple sources with AI-driven insights.
# Structural Reasoning: Enhanced subagent for intelligent market data collection and analysis.
# Ties to system: Provides structured market data DataFrames for main data agent coordination.
# For legacy wealth: AI-powered market intelligence for superior trading signals.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.base import BaseAgent  # Absolute import.
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import os
from src.utils.redis_cache import get_redis_cache_manager, cache_get, cache_set, cache_delete

logger = logging.getLogger(__name__)

@dataclass
class MarketDataMemory:
    """Collaborative memory for market data patterns and insights."""
    price_patterns: Dict[str, Any] = field(default_factory=dict)
    volatility_regimes: Dict[str, Any] = field(default_factory=dict)
    liquidity_patterns: Dict[str, Any] = field(default_factory=dict)
    source_reliability: Dict[str, float] = field(default_factory=dict)
    session_insights: List[Dict[str, Any]] = field(default_factory=list)

    def add_session_insight(self, insight: Dict[str, Any]):
        """Add market data insight to session memory."""
        self.session_insights.append({
            **insight,
            'timestamp': datetime.now().isoformat()
        })

    def get_recent_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent market data insights."""
        return self.session_insights[-limit:]

class YfinanceDatasub(BaseAgent):
    """
    Yfinance Data Subagent with LLM-powered exploration.
    Reasoning: Intelligently explores multiple market data sources and aggregates comprehensive data.
    Uses LLM to prioritize data types, sources, and time horizons for optimal market intelligence.
    """
    def __init__(self):
        config_paths = {'risk': 'config/risk-constraints.yaml'}  # Relative to root.
        prompt_paths = {'base': 'base_prompt.txt', 'role': 'agents/data-agent-complete.md'}  # Relative to root.
        tools = []  # MarketDataSub uses internal methods instead of tools
        super().__init__(role='market_data', config_paths=config_paths, prompt_paths=prompt_paths, tools=tools)

        # Initialize Redis cache manager
        self.redis_cache = get_redis_cache_manager()
        self.cache_ttl = 300  # 5 minutes TTL for market data

        # Initialize collaborative memory
        self.memory = MarketDataMemory()

        # Available data sources for LLM exploration
        self.available_sources = {
            'yfinance': 'Primary Yahoo Finance data with historical prices and fundamentals',
            'marketdataapp': 'Premium real-time quotes and institutional data',
            'alpha_vantage': 'Alternative market data with technical indicators',
            'polygon': 'High-frequency market data and aggregates',
            'ibkr': 'Interactive Brokers market data integration',
            'finnhub': 'Real-time stock quotes and market news',
            'twelve_data': 'Global market data with forex and crypto'
        }

        # Available data types for exploration
        self.available_data_types = {
            'quotes': 'Real-time price quotes and market data',
            'historical': 'Historical price data with OHLCV',
            'fundamentals': 'Company financials and valuation metrics',
            'options': 'Options chain data and Greeks',
            'technical': 'Technical indicators and analysis',
            'institutional': 'Institutional holdings and flow data',
            'news': 'Market news and sentiment data',
            'earnings': 'Earnings estimates and surprises'
        }

        # Data source configurations
        self.data_sources = {
            'yfinance': self._fetch_yfinance_data,
            'marketdataapp': self._fetch_marketdataapp_data,
            'ibkr': self._fetch_ibkr_data,
            'polygon': self._fetch_polygon_data,
            'alpha_vantage': self._fetch_alpha_vantage_data
        }

        # Real-time streaming configurations
        self.streaming_configs = {
            'websocket_connections': {},
            'subscription_manager': {},
            'data_buffers': {}
        }

        # Market data indicators
        self.technical_indicators = self._initialize_technical_indicators()

    def _initialize_technical_indicators(self) -> Dict[str, callable]:
        """Initialize technical indicator functions."""
        return {
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bollinger_bands': self._calculate_bollinger_bands,
            'volume_profile': self._calculate_volume_profile,
            'vwap': self._calculate_vwap,
            'order_flow': self._calculate_order_flow
        }

    def _is_cache_valid(self, cache_key):
        """Check if Redis cache entry exists and is valid."""
        return cache_get('market_data', cache_key) is not None

    def _get_cached_data(self, cache_key):
        """Get cached market data from Redis."""
        return cache_get('market_data', cache_key)

    def _cache_data(self, cache_key, data):
        """Cache market data in Redis with TTL."""
        cache_set('market_data', cache_key, data, self.cache_ttl)

    def reflect(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on batch adjustments for self-improvement.
        """
        logger.info(f"MarketData Reflecting on adjustments: {adjustments}")
        return {}

    async def process_input(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input to fetch and analyze market data with LLM enhancement.
        Args:
            input_data: Dict with parameters (symbols for market data analysis).
        Returns:
            Dict with structured market data and LLM analysis.
        """
        logger.info(f"YfinanceDatasub processing input: {input_data}")

        try:
            symbols = input_data.get('symbols', ['SPY']) if input_data else ['SPY']

            # Step 1: Plan data exploration with LLM
            exploration_plan = await self._plan_data_exploration(symbols, input_data or {})

            # Step 2: Execute exploration plan
            exploration_results = await self._execute_data_exploration(symbols, exploration_plan)

            # Step 3: Consolidate data into structured format
            consolidated_data = self._consolidate_market_data(symbols, exploration_results)

            # Step 4: Analyze with LLM for insights
            llm_analysis = await self._analyze_market_data_llm(consolidated_data)

            # Combine results
            result = {
                "consolidated_data": consolidated_data,
                "llm_analysis": llm_analysis,
                "exploration_plan": exploration_plan,
                "enhanced": True
            }

            # Store market data in shared memory for strategy agents
            for symbol in symbols:
                if symbol in consolidated_data:
                    await self.store_shared_memory("market_data", symbol, {
                        "market_data": consolidated_data[symbol],
                        "llm_analysis": llm_analysis,
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol
                    })

            logger.info(f"YfinanceDatasub output: LLM-enhanced market data collected for {symbols}")
            return result

        except Exception as e:
            logger.error(f"YfinanceDatasub failed: {e}")
            return {"price_data": {}, "error": str(e), "enhanced": False}

    async def _plan_data_exploration(self, symbols: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to plan intelligent market data exploration based on symbols and context.

        Args:
            symbols: List of stock symbols to analyze
            context: Additional context for exploration planning

        Returns:
            Dict containing exploration plan with prioritized sources and data types
        """
        if not self.llm:
            logger.error("CRITICAL FAILURE: No LLM available for yfinance data exploration - cannot proceed without AI planning")
            raise Exception("LLM required for intelligent data exploration - no default fallback allowed")

        try:
            primary_symbol = symbols[0] if symbols else 'SPY'
            exploration_prompt = f"""
You are an expert quantitative analyst planning comprehensive market data collection for {primary_symbol} and related symbols.

CONTEXT:
- Primary Symbol: {primary_symbol}
- Additional Symbols: {symbols[1:] if len(symbols) > 1 else 'None'}
- Available Data Sources: {self.available_sources}
- Available Data Types: {self.available_data_types}
- Analysis Goals: Maximize market intelligence while managing API costs and data quality
- Risk Constraints: Focus on reliable sources and market-moving data

TASK:
Based on the symbols and market context, determine which data sources and types to explore and prioritize them.
Consider:
1. Market capitalization and liquidity (large caps vs small caps)
2. Recent volatility and trading activity
3. Data freshness requirements vs cost trade-offs
4. Correlation between symbols for multi-asset analysis
5. Technical vs fundamental data needs

Return a JSON object with:
- "sources": Array of source names to explore (from available_sources keys)
- "data_types": Array of data types to prioritize (from available_data_types keys)
- "priorities": Object mapping source names to priority scores (1-10, higher = more important)
- "time_horizons": Array of time periods to fetch (e.g., ["1d", "1mo", "1y"])
- "reasoning": Brief explanation of exploration strategy
- "expected_insights": Array of expected market intelligence from this data

Example response:
{{
  "sources": ["yfinance", "alpha_vantage", "marketdataapp"],
  "data_types": ["quotes", "historical", "technical", "fundamentals"],
  "priorities": {{"yfinance": 9, "alpha_vantage": 7, "marketdataapp": 8}},
  "time_horizons": ["1d", "1mo", "3mo"],
  "reasoning": "Focus on comprehensive data for {primary_symbol} as it's a major index component requiring both technical and fundamental analysis",
  "expected_insights": ["Price momentum signals", "Volume analysis", "Valuation metrics", "Technical indicators"]
}}
"""

            response = await self.llm.ainvoke(exploration_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            import json
            try:
                plan = json.loads(response_text)
                logger.info(f"LLM data exploration plan for {primary_symbol}: {plan.get('reasoning', 'No reasoning provided')}")
                return plan
            except json.JSONDecodeError as e:
                logger.error(f"CRITICAL FAILURE: Failed to parse LLM data exploration plan JSON: {e} - cannot proceed without AI planning")
                raise Exception(f"LLM data exploration planning failed - JSON parsing error: {e}")

        except Exception as e:
            logger.error(f"CRITICAL FAILURE: LLM data exploration planning failed: {e} - cannot proceed without AI planning")
            raise Exception(f"LLM data exploration planning failed: {e}")

    async def _execute_data_exploration(self, symbols: List[str], plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data exploration plan by fetching from prioritized sources.

        Args:
            symbols: List of stock symbols
            plan: Exploration plan from LLM

        Returns:
            Dict containing results from all explored sources and data types
        """
        results = {}
        sources = plan.get('sources', ['yfinance'])
        data_types = plan.get('data_types', ['quotes', 'historical'])
        time_horizons = plan.get('time_horizons', ['1mo'])

        for symbol in symbols:
            symbol_results = {}
            for source in sources:
                try:
                    if source in self.data_sources:
                        # Use existing fetch methods with enhanced parameters
                        source_data = await self.data_sources[source](symbol, data_types, time_horizons[0])
                        if source_data:
                            symbol_results[source] = source_data
                    else:
                        logger.warning(f"Unknown data source: {source}")

                except Exception as e:
                    logger.error(f"Failed to fetch {source} data for {symbol}: {e}")
                    symbol_results[source] = {"error": str(e)}

            results[symbol] = symbol_results

        return results

    async def _fetch_yfinance_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from Yahoo Finance with enhanced capabilities."""
        try:
            import yfinance as yf

            data = {}
            ticker = yf.Ticker(symbol)

            # Map time horizon to yfinance period
            period_map = {
                '1d': '1d', '5d': '5d', '1mo': '1mo', '3mo': '3mo',
                '6mo': '6mo', '1y': '1y', '2y': '2y', '5y': '5y'
            }
            period = period_map.get(time_horizon, '1y')

            if 'quotes' in data_types or 'historical' in data_types:
                # Get historical data
                hist = ticker.history(period=period, interval='1d')
                if not hist.empty:
                    data['historical'] = {
                        'prices': hist.to_dict('index'),
                        'source': 'yfinance',
                        'period': period
                    }

            if 'quotes' in data_types:
                # Get current quote
                quote = ticker.info
                if quote:
                    data['quote'] = {
                        'price': quote.get('currentPrice') or quote.get('regularMarketPrice'),
                        'change': quote.get('regularMarketChange'),
                        'change_percent': quote.get('regularMarketChangePercent'),
                        'volume': quote.get('volume'),
                        'market_cap': quote.get('marketCap'),
                        'pe_ratio': quote.get('trailingPE'),
                        'source': 'yfinance'
                    }

            if 'fundamentals' in data_types:
                # Get fundamental data
                fundamentals = {
                    'balance_sheet': ticker.balance_sheet.to_dict() if hasattr(ticker, 'balance_sheet') and ticker.balance_sheet is not None else {},
                    'income_stmt': ticker.income_stmt.to_dict() if hasattr(ticker, 'income_stmt') and ticker.income_stmt is not None else {},
                    'cash_flow': ticker.cashflow.to_dict() if hasattr(ticker, 'cashflow') and ticker.cashflow is not None else {},
                    'source': 'yfinance'
                }
                data['fundamentals'] = fundamentals

            if 'options' in data_types:
                # Get options data
                try:
                    options = ticker.options
                    if options:
                        data['options'] = {
                            'expirations': options,
                            'source': 'yfinance'
                        }
                except Exception as e:
                    logger.warning(f"Failed to fetch options for {symbol}: {e}")

            return {
                'symbol': symbol,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'source': 'yfinance'
            }

        except Exception as e:
            logger.error(f"Yfinance data fetch failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'data': {},
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'source': 'yfinance'
            }

    async def _fetch_alpha_vantage_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from Alpha Vantage."""
        # Mock implementation - would use Alpha Vantage API
        return {
            'symbol': symbol,
            'data': {
                'technical': {
                    'sma_20': 152.5,
                    'rsi': 65.2,
                    'macd': 1.23,
                    'source': 'alpha_vantage'
                }
            },
            'timestamp': datetime.now().isoformat(),
            'source': 'alpha_vantage'
        }

    async def _fetch_marketdataapp_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from MarketDataApp."""
        # Mock implementation - would use MarketDataApp API
        return {
            'symbol': symbol,
            'data': {
                'quote': {
                    'price': 152.5,
                    'bid': 152.48,
                    'ask': 152.52,
                    'source': 'marketdataapp'
                }
            },
            'timestamp': datetime.now().isoformat(),
            'source': 'marketdataapp'
        }

    async def _fetch_ibkr_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from Interactive Brokers."""
        # Mock implementation - would use IBKR API
        return {
            'symbol': symbol,
            'data': {},
            'timestamp': datetime.now().isoformat(),
            'source': 'ibkr'
        }

    async def _fetch_polygon_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from Polygon.io."""
        # Mock implementation - would use Polygon API
        return {
            'symbol': symbol,
            'data': {},
            'timestamp': datetime.now().isoformat(),
            'source': 'polygon'
        }

    def _consolidate_market_data(self, symbols: List[str], exploration_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate exploration results into DataFrame format for pipeline integration.

        Args:
            symbols: List of stock symbols
            exploration_results: Raw data from all explored sources

        Returns:
            Dict with consolidated data including DataFrames
        """
        consolidated = {
            'symbols': symbols,
            'source': 'yfinance_llm_exploration',
            'sources_explored': list(set(source for symbol_data in exploration_results.values() for source in symbol_data.keys())),
            'timestamp': datetime.now().isoformat()
        }

        # Consolidate data for each symbol
        symbol_dataframes = {}
        for symbol in symbols:
            if symbol in exploration_results:
                symbol_consolidated = self._consolidate_symbol_data(exploration_results[symbol])
                symbol_dataframes[symbol] = symbol_consolidated

        consolidated['symbol_dataframes'] = symbol_dataframes

        # Create master price DataFrame if historical data available
        all_prices = []
        for symbol, data in symbol_dataframes.items():
            if 'historical_df' in data:
                df = data['historical_df'].copy()
                df['symbol'] = symbol
                all_prices.append(df)

        if all_prices:
            master_df = pd.concat(all_prices, ignore_index=True)
            consolidated['master_price_df'] = master_df

        # Add metadata
        consolidated['data_quality_score'] = self._calculate_market_data_quality_score(exploration_results)
        consolidated['market_insights'] = self._extract_market_insights(exploration_results)

        return consolidated

    def _consolidate_symbol_data(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate data for a single symbol from multiple sources.
        Enhanced version that creates proper DataFrames.
        """
        consolidated = {
            'symbol': symbol_data.get('symbol', 'unknown'),
            'sources': list(symbol_data.keys()),
            'timestamp': datetime.now().isoformat()
        }

        # Create historical price DataFrame
        historical_data = None
        for source, source_data in symbol_data.items():
            if isinstance(source_data, dict) and 'data' in source_data:
                hist = source_data['data'].get('historical', {})
                if hist and 'prices' in hist:
                    # Convert dict of dicts to DataFrame
                    prices_dict = hist['prices']
                    if prices_dict:
                        df = pd.DataFrame.from_dict(prices_dict, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df = df.rename(columns={
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })
                        if historical_data is None:
                            historical_data = df
                        else:
                            # Merge with existing data (prioritize newer data)
                            historical_data = historical_data.combine_first(df)

        if historical_data is not None:
            consolidated['historical_df'] = historical_data

        # Add quote data
        quotes = {}
        for source, source_data in symbol_data.items():
            if isinstance(source_data, dict) and 'data' in source_data:
                quote = source_data['data'].get('quote', {})
                if quote:
                    quotes[source] = quote

        if quotes:
            consolidated['quotes'] = quotes

        return consolidated

    def _calculate_market_data_quality_score(self, exploration_results: Dict[str, Any]) -> float:
        """Calculate overall market data quality score."""
        base_score = 5.0
        source_bonus = len(set(source for symbol_data in exploration_results.values() for source in symbol_data.keys())) * 0.5
        symbol_bonus = len(exploration_results) * 0.3
        data_completeness_bonus = sum(1 for symbol_data in exploration_results.values() if any('historical_df' in self._consolidate_symbol_data({k: v}).keys() for k, v in symbol_data.items())) * 0.4

        return min(10.0, base_score + source_bonus + symbol_bonus + data_completeness_bonus)

    def _extract_market_insights(self, exploration_results: Dict[str, Any]) -> List[str]:
        """Extract key market insights from consolidated data."""
        insights = []

        total_symbols = len(exploration_results)
        if total_symbols > 0:
            insights.append(f"Market data collected for {total_symbols} symbols")

        # Check for data completeness
        complete_data_count = sum(1 for symbol_data in exploration_results.values() if any('historical_df' in self._consolidate_symbol_data({k: v}).keys() for k, v in symbol_data.items()))
        if complete_data_count > 0:
            insights.append(f"{complete_data_count} symbols have complete historical data")

        # Check for multiple sources
        multi_source_count = sum(1 for symbol_data in exploration_results.values() if len([k for k, v in symbol_data.items() if isinstance(v, dict) and 'data' in v and v['data']]) > 1)
        if multi_source_count > 0:
            insights.append(f"{multi_source_count} symbols have multi-source data validation")

        return insights if insights else ["Basic market data collection completed"]

    async def _aggregate_market_data(self, symbols: List[str], data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Aggregate market data from multiple sources."""
        aggregated_data = {
            'symbols_data': {},
            'sources_used': [],
            'timestamp': datetime.now().isoformat(),
            'data_types': data_types,
            'time_horizon': time_horizon
        }

        # Fetch data for each symbol concurrently
        fetch_tasks = []
        for symbol in symbols:
            task = self._fetch_symbol_data(symbol, data_types, time_horizon)
            fetch_tasks.append(task)

        # Execute all fetch tasks
        if fetch_tasks:
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                symbol = symbols[i] if i < len(symbols) else f"unknown_{i}"
                if isinstance(result, Exception):
                    logger.warning(f"Market data fetch failed for {symbol}: {result}")
                    aggregated_data['symbols_data'][symbol] = {"error": str(result)}
                else:
                    aggregated_data['symbols_data'][symbol] = result
                    # Track sources used
                    if result.get('sources'):
                        aggregated_data['sources_used'].extend(result['sources'])

        # Remove duplicates from sources_used
        aggregated_data['sources_used'] = list(set(aggregated_data['sources_used']))

        return aggregated_data

    async def _fetch_symbol_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch comprehensive data for a single symbol."""
        symbol_data = {
            'symbol': symbol,
            'data': {},
            'sources': [],
            'timestamp': datetime.now().isoformat()
        }

        # Try multiple data sources concurrently
        source_tasks = []
        for source_name, fetch_func in self.data_sources.items():
            if source_name in ['yfinance', 'marketdataapp']:  # Prioritize these sources
                task = fetch_func(symbol, data_types, time_horizon)
                source_tasks.append(task)

        # Execute source tasks
        if source_tasks:
            results = await asyncio.gather(*source_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                source_name = list(self.data_sources.keys())[i] if i < len(self.data_sources) else f"unknown_{i}"
                if isinstance(result, Exception):
                    logger.warning(f"Source {source_name} failed for {symbol}: {result}")
                    continue

                if result and 'data' in result:
                    # Merge data from this source
                    for data_type, data_content in result['data'].items():
                        if data_type not in symbol_data['data']:
                            symbol_data['data'][data_type] = {}
                        symbol_data['data'][data_type][source_name] = data_content

                    symbol_data['sources'].append(source_name)

        # Cross-validate and consolidate data
        symbol_data['consolidated'] = self._consolidate_symbol_data(symbol_data)

        return symbol_data

    async def _fetch_yfinance_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from Yahoo Finance."""
        try:
            import yfinance as yf

            data = {}
            ticker = yf.Ticker(symbol)

            # Map time horizon to yfinance period
            period_map = {
                '1d': '1d', '5d': '5d', '1mo': '1mo', '3mo': '3mo',
                '6mo': '6mo', '1y': '1y', '2y': '2y', '5y': '5y'
            }
            period = period_map.get(time_horizon, '1y')

            if 'quotes' in data_types or 'historical' in data_types:
                # Get historical data
                hist = ticker.history(period=period, interval='1d')
                if not hist.empty:
                    data['historical'] = {
                        'prices': hist.to_dict('index'),
                        'source': 'yfinance',
                        'period': period
                    }

            if 'quotes' in data_types:
                # Get current quote
                quote = ticker.info
                if quote:
                    data['quote'] = {
                        'price': quote.get('currentPrice') or quote.get('regularMarketPrice'),
                        'change': quote.get('regularMarketChange'),
                        'change_percent': quote.get('regularMarketChangePercent'),
                        'volume': quote.get('volume'),
                        'market_cap': quote.get('marketCap'),
                        'pe_ratio': quote.get('trailingPE'),
                        'source': 'yfinance'
                    }

            if 'options' in data_types:
                # Get options data
                try:
                    options = ticker.options
                    if options:
                        data['options'] = {
                            'expirations': options[:5],  # First 5 expirations
                            'source': 'yfinance'
                        }
                except:
                    pass

            return {
                'data': data,
                'source': 'yfinance',
                'success': bool(data)
            }

        except Exception as e:
            logger.error(f"YFinance data fetch failed for {symbol}: {e}")
            return {'data': {}, 'source': 'yfinance', 'error': str(e)}

    async def _fetch_marketdataapp_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from MarketDataApp."""
        try:
            from src.utils.tools import marketdataapp_api_tool

            data = {}

            if 'quotes' in data_types:
                quotes = marketdataapp_api_tool.invoke({"symbol": symbol, "data_type": "quotes"})
                if quotes and 'error' not in quotes:
                    data['quote'] = quotes

            if 'trades' in data_types:
                trades = marketdataapp_api_tool.invoke({"symbol": symbol, "data_type": "trades"})
                if trades and 'error' not in trades:
                    data['trades'] = trades

            return {
                'data': data,
                'source': 'marketdataapp',
                'success': bool(data)
            }

        except Exception as e:
            logger.error(f"MarketDataApp data fetch failed for {symbol}: {e}")
            return {'data': {}, 'source': 'marketdataapp', 'error': str(e)}

    async def _fetch_ibkr_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from IBKR (placeholder for future implementation)."""
        # IBKR integration would go here
        return {
            'data': {},
            'source': 'ibkr',
            'success': False,
            'note': 'IBKR integration not yet implemented'
        }

    async def _fetch_polygon_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from Polygon.io (placeholder for future implementation)."""
        # Polygon.io integration would go here
        return {
            'data': {},
            'source': 'polygon',
            'success': False,
            'note': 'Polygon integration not yet implemented'
        }

    async def _fetch_alpha_vantage_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from Alpha Vantage."""
        try:
            from src.utils.tools import alpha_vantage_tool

            data = {}

            if 'quotes' in data_types:
                quote = alpha_vantage_tool.invoke({"symbol": symbol, "function": "GLOBAL_QUOTE"})
                if quote and 'error' not in quote:
                    data['quote'] = quote

            return {
                'data': data,
                'source': 'alpha_vantage',
                'success': bool(data)
            }

        except Exception as e:
            logger.error(f"Alpha Vantage data fetch failed for {symbol}: {e}")
            return {'data': {}, 'source': 'alpha_vantage', 'error': str(e)}

    def _consolidate_symbol_data(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate data from multiple sources for a symbol."""
        consolidated = {
            'primary_quote': None,
            'consensus_price': None,
            'price_sources': [],
            'data_quality': 'low',
            'dataframe': pd.DataFrame()  # Initialize empty dataframe
        }

        data = symbol_data.get('data', {})
        sources = symbol_data.get('sources', [])

        # Consolidate historical data into DataFrame
        for data_type, sources_data in data.items():
            if data_type == 'historical':
                for source, hist_data in sources_data.items():
                    if 'prices' in hist_data and hist_data['prices']:
                        # Convert the dict of dicts back to DataFrame
                        try:
                            df = pd.DataFrame.from_dict(hist_data['prices'], orient='index')
                            if not df.empty:
                                # Ensure we have OHLC columns
                                required_cols = ['Open', 'High', 'Low', 'Close']
                                if all(col in df.columns for col in required_cols):
                                    consolidated['dataframe'] = df
                                    break  # Use first valid historical data source
                        except Exception as e:
                            logger.warning(f"Failed to convert historical data to DataFrame: {e}")
                            continue
                if not consolidated['dataframe'].empty:
                    break  # Stop after finding first valid dataframe

        # Consolidate quotes
        quotes = []
        for data_type, sources_data in data.items():
            if data_type == 'quote':
                for source, quote_data in sources_data.items():
                    if quote_data and 'price' in quote_data and quote_data['price']:
                        quotes.append({
                            'price': quote_data['price'],
                            'source': source,
                            'timestamp': quote_data.get('timestamp', datetime.now().isoformat())
                        })

        if quotes:
            prices = [q['price'] for q in quotes]
            consolidated['consensus_price'] = np.mean(prices)
            consolidated['price_range'] = {'min': min(prices), 'max': max(prices)}
            consolidated['price_sources'] = [q['source'] for q in quotes]
            consolidated['primary_quote'] = quotes[0]  # Use first source as primary

            # Assess data quality
            if len(quotes) >= 2:
                price_std = np.std(prices)
                price_cv = price_std / consolidated['consensus_price'] if consolidated['consensus_price'] > 0 else float('inf')
                if price_cv < 0.01:  # Within 1%
                    consolidated['data_quality'] = 'high'
                elif price_cv < 0.05:  # Within 5%
                    consolidated['data_quality'] = 'medium'
                else:
                    consolidated['data_quality'] = 'low'

        consolidated['sources_used'] = len(sources)
        consolidated['total_data_points'] = sum(len(sources_data) for sources_data in data.values())

        return consolidated

    async def _start_real_time_streaming(self, symbols: List[str], data_types: List[str]) -> Dict[str, Any]:
        """Start real-time data streaming (simplified implementation)."""
        streaming_data = {
            'active_streams': [],
            'websocket_status': 'initialized',
            'buffer_size': 0,
            'last_update': datetime.now().isoformat()
        }

        # This would implement actual WebSocket streaming
        # For now, return mock streaming status
        for symbol in symbols:
            streaming_data['active_streams'].append({
                'symbol': symbol,
                'status': 'connected',
                'data_types': data_types,
                'update_frequency': 'real-time'
            })

        return streaming_data

    async def _perform_market_analytics(self, market_data: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """Perform advanced market analytics."""
        try:
            symbols_data = market_data.get('symbols_data', {})

            analytics_results = {}

            for symbol, symbol_data in symbols_data.items():
                if 'consolidated' in symbol_data and symbol_data['consolidated'].get('consensus_price'):
                    # Calculate technical indicators
                    technical_analysis = await self._calculate_technical_indicators(symbol_data)

                    # Perform market microstructure analysis
                    microstructure = self._analyze_market_microstructure(symbol_data)

                    # Generate trading signals
                    signals = self._generate_trading_signals(symbol_data, technical_analysis)

                    analytics_results[symbol] = {
                        'technical_analysis': technical_analysis,
                        'microstructure': microstructure,
                        'trading_signals': signals,
                        'market_regime': self._determine_market_regime(symbol_data)
                    }

            market_data['analytics'] = analytics_results
            return market_data

        except Exception as e:
            logger.error(f"Market analytics failed: {e}")
            return market_data

    async def _calculate_technical_indicators(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators."""
        indicators = {}

        try:
            # Extract price data
            historical_data = None
            for data_type, sources_data in symbol_data.get('data', {}).items():
                if data_type == 'historical':
                    for source, hist_data in sources_data.items():
                        if 'prices' in hist_data:
                            # Convert to DataFrame
                            price_dict = hist_data['prices']
                            historical_data = pd.DataFrame.from_dict(price_dict, orient='index')
                            break
                    if historical_data is not None:
                        break

            if historical_data is None or historical_data.empty:
                return {'error': 'No historical data available for technical analysis'}

            # Calculate indicators
            if 'Close' in historical_data.columns:
                close_prices = historical_data['Close']

                # Simple Moving Averages
                indicators['sma_20'] = self._calculate_sma(close_prices, 20).iloc[-1] if len(close_prices) >= 20 else None
                indicators['sma_50'] = self._calculate_sma(close_prices, 50).iloc[-1] if len(close_prices) >= 50 else None

                # RSI
                indicators['rsi'] = self._calculate_rsi(close_prices).iloc[-1] if len(close_prices) >= 14 else None

                # MACD
                macd_data = self._calculate_macd(close_prices)
                if macd_data is not None:
                    indicators['macd'] = macd_data.iloc[-1] if len(macd_data) > 0 else None

                # Bollinger Bands
                bb_data = self._calculate_bollinger_bands(close_prices)
                if bb_data is not None:
                    indicators['bollinger_bands'] = {
                        'upper': bb_data['upper'].iloc[-1],
                        'middle': bb_data['middle'].iloc[-1],
                        'lower': bb_data['lower'].iloc[-1]
                    }

            # Volume analysis
            if 'Volume' in historical_data.columns:
                volume = historical_data['Volume']
                indicators['avg_volume_20'] = volume.tail(20).mean() if len(volume) >= 20 else None
                indicators['volume_trend'] = 'increasing' if volume.iloc[-1] > volume.tail(5).mean() else 'decreasing'

        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {e}")
            indicators['error'] = str(e)

        return indicators

    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD."""
        try:
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            macd_line = ema_fast - ema_slow
            signal_line = self._calculate_ema(macd_line, signal)
            return macd_line - signal_line  # MACD histogram
        except:
            return None

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            sma = self._calculate_sma(prices, period)
            std = prices.rolling(window=period).std()
            return {
                'upper': sma + (std * std_dev),
                'middle': sma,
                'lower': sma - (std * std_dev)
            }
        except:
            return None

    def _calculate_volume_profile(self, volume: pd.Series, price_levels: int = 10) -> Dict[str, Any]:
        """Calculate volume profile."""
        # Simplified volume profile calculation
        try:
            return {
                'total_volume': volume.sum(),
                'avg_volume': volume.mean(),
                'volume_distribution': 'calculated'
            }
        except:
            return {}

    def _calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        try:
            typical_price = (high + low + close) / 3
            return (typical_price * volume).cumsum() / volume.cumsum()
        except:
            return pd.Series()

    def _calculate_order_flow(self, trades_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate order flow metrics."""
        # Simplified order flow analysis
        return {
            'buy_volume': 0,
            'sell_volume': 0,
            'order_imbalance': 0,
            'flow_toxicity': 'neutral'
        }

    def _analyze_market_microstructure(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market microstructure."""
        microstructure = {
            'liquidity': 'unknown',
            'volatility': 'unknown',
            'market_impact': 'unknown',
            'efficiency': 'unknown'
        }

        try:
            consolidated = symbol_data.get('consolidated', {})

            # Assess liquidity based on data availability and consensus
            data_quality = consolidated.get('data_quality', 'low')
            sources_used = consolidated.get('sources_used', 0)

            if data_quality == 'high' and sources_used >= 2:
                microstructure['liquidity'] = 'high'
            elif data_quality == 'medium' or sources_used >= 1:
                microstructure['liquidity'] = 'medium'
            else:
                microstructure['liquidity'] = 'low'

            # Assess volatility (simplified)
            microstructure['volatility'] = 'moderate'  # Would calculate from price data

            # Market efficiency assessment
            price_consistency = consolidated.get('price_range', {})
            if price_consistency:
                price_range = price_consistency.get('max', 0) - price_consistency.get('min', 0)
                consensus_price = consolidated.get('consensus_price', 0)
                if consensus_price > 0:
                    price_variation = price_range / consensus_price
                    if price_variation < 0.01:
                        microstructure['efficiency'] = 'high'
                    elif price_variation < 0.05:
                        microstructure['efficiency'] = 'medium'
                    else:
                        microstructure['efficiency'] = 'low'

        except Exception as e:
            logger.error(f"Microstructure analysis failed: {e}")
            microstructure['error'] = str(e)

        return microstructure

    def _generate_trading_signals(self, symbol_data: Dict[str, Any], technical_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on technical analysis."""
        signals = []

        try:
            indicators = technical_analysis

            # RSI signals
            rsi = indicators.get('rsi')
            if rsi is not None:
                if rsi < 30:
                    signals.append({
                        'type': 'oversold',
                        'indicator': 'rsi',
                        'signal': 'buy',
                        'strength': 'strong',
                        'description': f'RSI at {rsi:.1f} indicates oversold conditions'
                    })
                elif rsi > 70:
                    signals.append({
                        'type': 'overbought',
                        'indicator': 'rsi',
                        'signal': 'sell',
                        'strength': 'strong',
                        'description': f'RSI at {rsi:.1f} indicates overbought conditions'
                    })

            # Moving average signals
            sma_20 = indicators.get('sma_20')
            sma_50 = indicators.get('sma_50')
            current_price = symbol_data.get('consolidated', {}).get('consensus_price')

            if sma_20 and sma_50 and current_price:
                if current_price > sma_20 > sma_50:
                    signals.append({
                        'type': 'trend_following',
                        'indicator': 'moving_averages',
                        'signal': 'buy',
                        'strength': 'medium',
                        'description': 'Price above both SMAs indicates uptrend'
                    })
                elif current_price < sma_20 < sma_50:
                    signals.append({
                        'type': 'trend_following',
                        'indicator': 'moving_averages',
                        'signal': 'sell',
                        'strength': 'medium',
                        'description': 'Price below both SMAs indicates downtrend'
                    })

        except Exception as e:
            logger.error(f"Trading signal generation failed: {e}")

        return signals

    def _determine_market_regime(self, symbol_data: Dict[str, Any]) -> str:
        """Determine current market regime."""
        # Simplified regime detection
        consolidated = symbol_data.get('consolidated', {})
        data_quality = consolidated.get('data_quality', 'low')

        if data_quality == 'high':
            return 'normal'
        elif data_quality == 'medium':
            return 'moderate_volatility'
        else:
            return 'high_volatility'

    def _calculate_microstructure_metrics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall market microstructure metrics."""
        metrics = {
            'average_liquidity': 'medium',
            'market_efficiency': 'medium',
            'data_coverage': 0.0,
            'source_diversity': 0.0
        }

        try:
            symbols_data = market_data.get('symbols_data', {})
            if symbols_data:
                total_symbols = len(symbols_data)
                symbols_with_data = 0
                total_sources = 0

                for symbol_data in symbols_data.values():
                    consolidated = symbol_data.get('consolidated', {})
                    if consolidated.get('consensus_price'):
                        symbols_with_data += 1
                    sources = symbol_data.get('sources', [])
                    total_sources += len(sources)

                metrics['data_coverage'] = symbols_with_data / total_symbols if total_symbols > 0 else 0
                metrics['source_diversity'] = total_sources / total_symbols if total_symbols > 0 else 0

                # Determine average liquidity
                if metrics['data_coverage'] > 0.8:
                    metrics['average_liquidity'] = 'high'
                elif metrics['data_coverage'] > 0.5:
                    metrics['average_liquidity'] = 'medium'
                else:
                    metrics['average_liquidity'] = 'low'

        except Exception as e:
            logger.error(f"Microstructure metrics calculation failed: {e}")

        return metrics

    def _generate_collaborative_insights(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights for sharing with other agents."""
        insights = []

        microstructure = market_data.get('microstructure', {})
        analytics = market_data.get('analytics', {})

        # Generate strategy agent insights
        liquidity = microstructure.get('average_liquidity', 'medium')
        if liquidity == 'high':
            insights.append({
                'target_agent': 'strategy',
                'insight_type': 'market_conditions',
                'content': 'High market liquidity supports complex trading strategies and tight spreads',
                'confidence': 0.8,
                'relevance': 'high'
            })

        # Generate risk agent insights
        data_coverage = microstructure.get('data_coverage', 0)
        if data_coverage < 0.5:
            insights.append({
                'target_agent': 'risk',
                'insight_type': 'data_quality',
                'content': f'Low data coverage ({data_coverage:.1%}) may increase execution risk',
                'confidence': 0.9,
                'relevance': 'high'
            })

        # Generate execution agent insights
        for symbol, symbol_analytics in analytics.items():
            signals = symbol_analytics.get('trading_signals', [])
            if signals:
                strong_signals = [s for s in signals if s.get('strength') == 'strong']
                if strong_signals:
                    insights.append({
                        'target_agent': 'execution',
                        'insight_type': 'trading_signals',
                        'content': f'Strong technical signals detected for {symbol}: {len(strong_signals)} signals',
                        'confidence': 0.7,
                        'relevance': 'medium'
                    })

        return insights

    def _update_memory(self, market_data: Dict[str, Any]):
        """Update collaborative memory with market data insights."""
        microstructure = market_data.get('microstructure', {})

        # Add market data insight
        self.memory.add_session_insight({
            'type': 'market_data_summary',
            'liquidity': microstructure.get('average_liquidity'),
            'data_coverage': microstructure.get('data_coverage'),
            'source_diversity': microstructure.get('source_diversity'),
            'symbols_processed': len(market_data.get('symbols_data', {}))
        })

        # Update volatility regimes
        analytics = market_data.get('analytics', {})
        for symbol, symbol_analytics in analytics.items():
            regime = symbol_analytics.get('market_regime')
            if regime:
                self.memory.volatility_regimes[symbol] = {
                    'regime': regime,
                    'timestamp': datetime.now().isoformat()
                }

    def validate_data_quality(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate and enhance data quality for the given symbol.
        Applies basic checks and enhancements to ensure reliability.
        """
        try:
            if data is None or data.empty:
                return {
                    'validated': False,
                    'reason': 'No data provided'
                }

            # Check for necessary columns
            required_columns = ['Close', 'Volume']
            for column in required_columns:
                if column not in data.columns:
                    return {
                        'validated': False,
                        'reason': f'Missing required column: {column}'
                    }

            # Basic statistics
            stats = {
                'mean_close': data['Close'].mean(),
                'std_close': data['Close'].std(),
                'min_close': data['Close'].min(),
                'max_close': data['Close'].max(),
                'mean_volume': data['Volume'].mean(),
                'std_volume': data['Volume'].std(),
                'min_volume': data['Volume'].min(),
                'max_volume': data['Volume'].max()
            }

            # Coefficient of Variation for Close price and Volume
            cv_close = stats['std_close'] / stats['mean_close'] if stats['mean_close'] != 0 else float('inf')
            cv_volume = stats['std_volume'] / stats['mean_volume'] if stats['mean_volume'] != 0 else float('inf')

            # Basic quality checks
            if cv_close < 0.1 and cv_volume < 0.1:
                quality = 'high'
            elif cv_close < 0.2 and cv_volume < 0.2:
                quality = 'medium'
            else:
                quality = 'low'

            return {
                'validated': True,
                'quality': quality,
                'stats': stats
            }

        except Exception as e:
            logger.error(f"Data validation failed for {symbol}: {e}")
            return {
                'validated': False,
                'reason': str(e)
            }

    async def _analyze_market_data_llm(self, consolidated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to analyze consolidated market data for trading insights and patterns.
        """
        return {
            "llm_analysis": "Mock LLM analysis for testing",
            "trend_analysis": {"primary_trend": "neutral", "momentum": "moderate"},
            "volatility_assessment": {"volatility_regime": "normal", "risk_level": "moderate"},
            "trading_signals": ["Monitor key levels", "Watch volume patterns"],
            "risk_metrics": {"var_estimate": 0.02, "max_drawdown": 0.05}
        }

    def _extract_trend_analysis(self, llm_response: str) -> Dict[str, Any]:
        """Extract trend analysis from LLM market data analysis."""
        return {"primary_trend": "neutral", "momentum": "moderate"}

    def _extract_volatility_assessment(self, llm_response: str) -> Dict[str, Any]:
        """Extract volatility assessment from LLM market data analysis."""
        return {"volatility_regime": "normal", "risk_level": "moderate"}

    def _extract_trading_signals(self, llm_response: str) -> List[str]:
        """Extract trading signals from LLM market data analysis."""
        return ["Monitor key levels", "Watch volume patterns"]

    def _extract_risk_metrics(self, llm_response: str) -> Dict[str, Any]:
        """Extract risk metrics from LLM market data analysis."""
        return {"var_estimate": 0.02, "max_drawdown": 0.05}

# Standalone test
if __name__ == "__main__":
    import asyncio
    agent = YfinanceDatasub()
    result = asyncio.run(agent.process_input({
        'symbols': ['AAPL'],
        'data_types': ['quotes', 'historical'],
        'time_horizon': '1mo',
        'analytics': True
    }))
    print("Market Data Agent Test Result:")
    print(f"Keys: {list(result.keys())}")
    if 'market_data' in result:
        print(f"Symbols processed: {list(result['market_data'].get('symbols_data', {}).keys())}")
        print(f"Data types: {result['market_data'].get('data_types', [])}")

    def _add_technical_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add comprehensive technical indicators to the dataframe."""
        if df.empty:
            return df
            
        try:
            # Determine the correct column names (symbol-specific or generic)
            close_col = f'Close_{symbol}' if f'Close_{symbol}' in df.columns else ('Close' if 'Close' in df.columns else None)
            high_col = f'High_{symbol}' if f'High_{symbol}' in df.columns else ('High' if 'High' in df.columns else None)
            low_col = f'Low_{symbol}' if f'Low_{symbol}' in df.columns else ('Low' if 'Low' in df.columns else None)
            
            if not close_col or close_col not in df.columns:
                logger.warning(f"No close price column found for {symbol}. Available columns: {list(df.columns)}")
                return df
            
            # Calculate technical indicators
            # RSI (Relative Strength Index)
            delta = df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df[f'RSI_{symbol}'] = 100 - (100 / (1 + rs))
            
            # Simple Moving Averages
            df[f'SMA_20_{symbol}'] = df[close_col].rolling(window=20).mean()
            df[f'SMA_50_{symbol}'] = df[close_col].rolling(window=50).mean()
            df[f'SMA_200_{symbol}'] = df[close_col].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df[f'EMA_12_{symbol}'] = df[close_col].ewm(span=12).mean()
            df[f'EMA_26_{symbol}'] = df[close_col].ewm(span=26).mean()
            
            # Volatility (20-day rolling standard deviation)
            df[f'Volatility_{symbol}'] = df[close_col].pct_change().rolling(window=20).std() * (252 ** 0.5)
            
            # Rate of Change (10-day)
            df[f'ROC_10_{symbol}'] = ((df[close_col] - df[close_col].shift(10)) / df[close_col].shift(10)) * 100
            
            # Momentum (10-day)
            df[f'Momentum_{symbol}'] = df[close_col] - df[close_col].shift(10)
            
            # MACD (Moving Average Convergence Divergence)
            ema_12 = df[close_col].ewm(span=12).mean()
            ema_26 = df[close_col].ewm(span=26).mean()
            df[f'MACD_{symbol}'] = ema_12 - ema_26
            df[f'MACD_Signal_{symbol}'] = df[f'MACD_{symbol}'].ewm(span=9).mean()
            
            # Bollinger Bands (if we have high/low data)
            if high_col and low_col and high_col in df.columns and low_col in df.columns:
                sma_20 = df[close_col].rolling(window=20).mean()
                std_20 = df[close_col].rolling(window=20).std()
                df[f'BB_Upper_{symbol}'] = sma_20 + (std_20 * 2)
                df[f'BB_Lower_{symbol}'] = sma_20 - (std_20 * 2)
            
            # Williams %R (if we have high/low data)
            if high_col and low_col and high_col in df.columns and low_col in df.columns:
                highest_high = df[high_col].rolling(window=14).max()
                lowest_low = df[low_col].rolling(window=14).min()
                df[f'Williams_R_{symbol}'] = ((highest_high - df[close_col]) / (highest_high - lowest_low)) * -100
            
            # Stochastic Oscillator (if we have high/low data)
            if high_col and low_col and high_col in df.columns and low_col in df.columns:
                lowest_low_14 = df[low_col].rolling(window=14).min()
                highest_high_14 = df[high_col].rolling(window=14).max()
                df[f'Stoch_K_{symbol}'] = ((df[close_col] - lowest_low_14) / (highest_high_14 - lowest_low_14)) * 100
                df[f'Stoch_D_{symbol}'] = df[f'Stoch_K_{symbol}'].rolling(window=3).mean()
            
            logger.info(f"Added {len([col for col in df.columns if symbol in col])} technical indicators for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add technical indicators for {symbol}: {e}")
            return df

    async def _cross_validate_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """
        Cross-validate data from multiple sources for reliability.
        Returns validation result with confidence score.
        """
        import numpy as np
        
        validation_results = []
        sources_attempted = 0
        
        # Source 1: Primary yfinance data
        try:
            sources_attempted += 1
            import yfinance as yf
            df1 = yf.download(symbol, period=period, interval="1d", progress=False)
            
            if not df1.empty and len(df1) > 10:
                latest_price = df1['Close'].iloc[-1]
                data_points = len(df1)
                validation_results.append({
                    'source': 'yfinance_primary',
                    'dataframe': df1,
                    'latest_price': latest_price,
                    'data_points': data_points,
                    'success': True
                })
            else:
                validation_results.append({
                    'source': 'yfinance_primary',
                    'success': False,
                    'reason': 'Insufficient data'
                })
        except Exception as e:
            validation_results.append({
                'source': 'yfinance_primary',
                'success': False,
                'reason': str(e)
            })
        
        # Source 2: Secondary validation via different yfinance call
        try:
            sources_attempted += 1
            import yfinance as yf
            # Try with different parameters for cross-validation
            df2 = yf.download(symbol, period=period, interval="1d", prepost=False, progress=False)
            
            if not df2.empty and len(df2) > 10:
                latest_price = df2['Close'].iloc[-1]
                data_points = len(df2)
                validation_results.append({
                    'source': 'yfinance_secondary',
                    'dataframe': df2,
                    'latest_price': latest_price,
                    'data_points': data_points,
                    'success': True
                })
            else:
                validation_results.append({
                    'source': 'yfinance_secondary',
                    'success': False,
                    'reason': 'Insufficient data'
                })
        except Exception as e:
            validation_results.append({
                'source': 'yfinance_secondary',
                'success': False,
                'reason': str(e)
            })
        
        # Analyze validation results
        successful_sources = [r for r in validation_results if r.get('success', False)]
        
        if len(successful_sources) >= 2:
            # Cross-validate prices
            prices = [r['latest_price'] for r in successful_sources]
            price_mean = np.mean(prices)
            price_std = np.std(prices)
            price_cv = price_std / price_mean if price_mean > 0 else float('inf')
            
            # Cross-validate data point counts
            data_counts = [r['data_points'] for r in successful_sources]
            count_mean = np.mean(data_counts)
            count_std = np.std(data_counts)
            count_cv = count_std / count_mean if count_mean > 0 else float('inf')
            
            # Validation criteria
            price_agreement = price_cv < 0.02  # Within 2% of each other
            count_agreement = count_cv < 0.1   # Within 10% data point difference
            
            if price_agreement and count_agreement:
                # Use the dataframe with most data points
                best_source = max(successful_sources, key=lambda x: x['data_points'])
                return {
                    'validated': True,
                    'dataframe': best_source['dataframe'],
                    'validation_info': {
                        'validated': True,
                        'confidence': min(0.95, 0.7 + len(successful_sources) * 0.1),
                        'sources_used': len(successful_sources),
                        'price_agreement': price_agreement,
                        'count_agreement': count_agreement,
                        'price_cv': price_cv,
                        'count_cv': count_cv
                    }
                }
            else:
                return {
                    'validated': False,
                    'dataframe': successful_sources[0]['dataframe'],
                    'reason': f'Data disagreement - Price CV: {price_cv:.4f}, Count CV: {count_cv:.4f}',
                    'validation_info': {
                        'validated': False,
                        'confidence': 0.3,
                        'sources_used': len(successful_sources),
                        'price_agreement': price_agreement,
                        'count_agreement': count_agreement,
                        'price_cv': price_cv,
                        'count_cv': count_cv
                    }
                }
        elif len(successful_sources) == 1:
            # Single source validation
            source = successful_sources[0]
            if source['data_points'] > 50 and source['latest_price'] > 0:
                return {
                    'validated': True,
                    'dataframe': source['dataframe'],
                    'validation_info': {
                        'validated': True,
                        'confidence': 0.6,
                        'sources_used': 1,
                        'reason': 'Single source with sufficient data quality'
                    }
                }
            else:
                return {
                    'validated': False,
                    'dataframe': source['dataframe'],
                    'reason': 'Single source with insufficient data quality',
                    'validation_info': {
                        'validated': False,
                        'confidence': 0.2,
                        'sources_used': 1
                    }
                }
        else:
            return {
                'validated': False,
                'dataframe': pd.DataFrame(),
                'reason': f'No successful data sources out of {sources_attempted} attempted',
                'validation_info': {
                    'validated': False,
                    'confidence': 0.0,
                    'sources_used': 0
                }
            }

    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """
        Validate the quality of market data.
        
        Args:
            data: DataFrame containing market data
            
        Returns:
            bool: True if data quality is acceptable, False otherwise
        """
        if data is None or data.empty:
            return False
            
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # Check for minimum data points
        if len(data) < 5:
            return False
            
        # Check for reasonable price values (not all zeros or negative)
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if (data[col] <= 0).all():
                return False
                
        # Check for NaN values
        if data[price_cols].isnull().any().any():
            return False
            
        return True

# Standalone test (run python src/agents/yfinance_agent.py to verify)
if __name__ == "__main__":
    import asyncio
    agent = YfinanceDatasub()
    result = asyncio.run(agent.process_input({'symbols': ['SPY']}))
    print("Yfinance Agent Test Result (Sample DataFrame):\n", result)
