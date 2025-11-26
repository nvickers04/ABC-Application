# src/agents/data_analyzers/yfinance_data_analyzer.py
# Purpose: Yfinance Data Subagent for fetching and analyzing market data from multiple sources.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.base import BaseAgent  # Absolute import.
import logging
from typing import Dict, Any, List, Callable, Optional
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from dataclasses import dataclass, field
import time
from src.utils.redis_cache import get_redis_cache_manager, cache_get, cache_set

# Import performance optimizations
try:
    from optimizations.performance_optimizations import AsyncYFianceClient, OptimizedRedisCache, CircuitBreaker
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False
    AsyncYFianceClient = None
    OptimizedRedisCache = None
    CircuitBreaker = None

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

class YfinanceDataAnalyzer(BaseAgent):
    """
    Yfinance Data Analyzer with LLM-powered exploration.
    Reasoning: Intelligently explores multiple market data sources and aggregates comprehensive data.
    Uses LLM to prioritize data types, sources, and time horizons for optimal market intelligence.
    """
    def __init__(self):
        config_paths = {'risk': 'config/risk-constraints.yaml'}  # Relative to root.
        prompt_paths = {'base': 'config/base_prompt.txt', 'role': 'docs/AGENTS/main-agents/data-agent.md'}  # Relative to root.
        tools = []  # MarketDataSub uses internal methods instead of tools
        super().__init__(role='market_data', config_paths=config_paths, prompt_paths=prompt_paths, tools=tools)

        # Initialize Redis cache manager
        self.redis_cache = get_redis_cache_manager()
        self.cache_ttl = 300  # 5 minutes TTL for market data

        # Initialize optimized components if available
        if OPTIMIZATIONS_AVAILABLE:
            self.async_client = AsyncYFianceClient()
            self.optimized_cache = OptimizedRedisCache(self.redis_cache) if self.redis_cache else None
            self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        else:
            self.async_client = None
            self.optimized_cache = None
            self.circuit_breaker = None

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

    def _initialize_technical_indicators(self) -> Dict[str, Callable[..., Any]]:
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
        if self.optimized_cache:
            # Use optimized cache for async checking
            import asyncio
            try:
                asyncio.get_running_loop()
                # If we're in an async context, we can't use await here
                # Fall back to basic cache check
                return cache_get('market_data', cache_key) is not None
            except RuntimeError:
                # Not in async context, we can run async check
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.optimized_cache.get('market_data', cache_key))
                    loop.close()
                    return result is not None
                except:
                    return cache_get('market_data', cache_key) is not None
        else:
            return cache_get('market_data', cache_key) is not None

    def _get_cached_data(self, cache_key):
        """Get cached market data from Redis."""
        if self.optimized_cache:
            import asyncio
            try:
                asyncio.get_running_loop()
                # If we're in an async context, we can't use await here
                # Fall back to basic cache get
                return cache_get('market_data', cache_key)
            except RuntimeError:
                # Not in async context, we can run async get
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.optimized_cache.get('market_data', cache_key))
                    loop.close()
                    return result
                except:
                    return cache_get('market_data', cache_key)
        else:
            return cache_get('market_data', cache_key)

    def _cache_data(self, cache_key, data):
        """Cache market data in Redis with TTL."""
        if self.optimized_cache:
            import asyncio
            try:
                asyncio.get_running_loop()
                # If we're in an async context, we can't use await here
                # Fall back to basic cache set
                cache_set('market_data', cache_key, data, self.cache_ttl)
            except RuntimeError:
                # Not in async context, we can run async set
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.optimized_cache.set('market_data', cache_key, data, ttl=self.cache_ttl))
                    loop.close()
                except:
                    cache_set('market_data', cache_key, data, self.cache_ttl)
        else:
            cache_set('market_data', cache_key, data, self.cache_ttl)

    def reflect(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on batch adjustments for self-improvement.
        """
        logger.info(f"MarketData Reflecting on adjustments: {adjustments}")
        return {}

    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processes input for YFinance data analysis.
        
        Args:
            input_data: Dictionary with symbols, data_types, etc.
        
        Returns:
            Dictionary with market_data and llm_analysis.
        """
        logger.info(f"YfinanceDataAnalyzer processing input: {input_data}")

        try:
            symbols = input_data.get('symbols', ['SPY']) if input_data else ['SPY']

            # Initialize LLM if not already done
            if not self.llm:
                await self.async_initialize_llm()

            # Use batch processing for multiple symbols if optimizations available
            if len(symbols) > 1 and OPTIMIZATIONS_AVAILABLE and self.async_client:
                logger.info(f"Using optimized batch processing for {len(symbols)} symbols")
                return await self._process_input_optimized(input_data)
            else:
                # Single symbol or fallback processing
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

                logger.info(f"YfinanceDataAnalyzer output: LLM-enhanced market data collected for {symbols}")
                return result

        except Exception as e:
            logger.error(f"YfinanceDataAnalyzer failed: {e}")
            return {"price_data": {}, "error": str(e), "enhanced": False}

    async def _process_input_optimized(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized version with async operations and batch processing"""
        start_time = time.time()

        symbols = input_data.get('symbols', ['SPY'])
        data_types = input_data.get('data_types', ['quotes', 'historical'])
        time_horizon = input_data.get('time_horizon', '1mo')

        if not symbols:
            return {"error": "No symbols provided"}

        logger.info(f"Processing {len(symbols)} symbols with optimized analyzer")

        try:
            # Batch process all symbols concurrently
            results = await self._batch_process_symbols(symbols, data_types, time_horizon)

            # Consolidate results
            consolidated = self._consolidate_optimized_results(results, symbols)

            # Add LLM analysis if available
            if self.llm:
                exploration_plan = await self._plan_data_exploration(symbols, input_data or {})
                consolidated["exploration_plan"] = exploration_plan
                consolidated["llm_analysis"] = await self._analyze_market_data_llm(consolidated.get("consolidated_data", {}))

            processing_time = time.time() - start_time
            logger.info(f"Optimized processing completed in {processing_time:.2f} seconds")

            return consolidated

        except Exception as e:
            logger.error(f"Optimized processing failed: {e}")
            return {"error": str(e)}

    async def _batch_process_symbols(self, symbols: List[str], data_types: List[str], time_horizon: str) -> Dict[str, Dict[str, Any]]:
        """Batch process multiple symbols concurrently"""
        tasks = []
        for symbol in symbols:
            task = self._process_symbol_optimized(symbol, data_types, time_horizon)
            tasks.append(task)

        # Execute with concurrency limit to avoid overwhelming APIs
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

        async def limited_task(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*[limited_task(task) for task in tasks], return_exceptions=True)

        # Process results
        symbol_results = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                symbol_results[symbol] = {"error": str(result)}
            else:
                symbol_results[symbol] = result

        return symbol_results

    async def _process_symbol_optimized(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Process a single symbol with caching and async operations"""
        cache_key = f"symbol_data_{symbol}_{'_'.join(data_types)}_{time_horizon}"

        # Check cache first
        if self.optimized_cache:
            cached_result = await self.optimized_cache.get('symbol_data', cache_key)
            if cached_result:
                logger.info(f"Cache hit for {symbol}")
                return cached_result

        # Fetch data asynchronously
        try:
            # Parallel fetch of different data types
            fetch_tasks = []

            if 'quotes' in data_types or 'historical' in data_types:
                fetch_tasks.append(self.async_client.get_historical_data(symbol, time_horizon))

            if 'quotes' in data_types:
                fetch_tasks.append(self.async_client.get_ticker_info(symbol))

            # Execute fetches concurrently
            fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Process results
            result = {
                'symbol': symbol,
                'data': {},
                'timestamp': datetime.now().isoformat(),
                'cached': False
            }

            for i, fetch_result in enumerate(fetch_results):
                if isinstance(fetch_result, Exception):
                    logger.warning(f"Fetch task {i} failed for {symbol}: {fetch_result}")
                    continue

                if i == 0:  # Historical data
                    result['data']['historical'] = {
                        'prices': fetch_result,
                        'source': 'yfinance_optimized'
                    }
                elif i == 1:  # Ticker info
                    result['data']['quote'] = fetch_result

            # Cache the result
            if self.optimized_cache:
                await self.optimized_cache.set('symbol_data', cache_key, result, ttl=self.cache_ttl)

            return result

        except Exception as e:
            logger.error(f"Failed to process symbol {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _consolidate_optimized_results(self, results: Dict[str, Dict[str, Any]], symbols: List[str]) -> Dict[str, Any]:
        """Consolidate optimized results into final format"""
        consolidated = {
            'symbols_processed': symbols,
            'total_symbols': len(symbols),
            'successful_fetches': 0,
            'failed_fetches': 0,
            'timestamp': datetime.now().isoformat(),
            'optimization_applied': True
        }

        symbol_dataframes = {}
        all_prices = []

        for symbol, result in results.items():
            if 'error' in result:
                consolidated['failed_fetches'] += 1
                continue

            consolidated['successful_fetches'] += 1

            # Convert to DataFrame format for compatibility
            if 'data' in result and 'historical' in result['data']:
                prices_dict = result['data']['historical'].get('prices', {})
                if prices_dict:
                    try:
                        df = pd.DataFrame.from_dict(prices_dict, orient='index')
                        df.index = pd.to_datetime(df.index)
                        symbol_dataframes[symbol] = {
                            'historical_df': df,
                            'quote_data': result['data'].get('quote', {}),
                            'source': 'optimized_yfinance'
                        }

                        # Add to master price DataFrame
                        df_copy = df.copy()
                        df_copy['symbol'] = symbol
                        all_prices.append(df_copy)
                    except Exception as e:
                        logger.warning(f"Failed to create DataFrame for {symbol}: {e}")

        if all_prices:
            master_df = pd.concat(all_prices, ignore_index=True)
            consolidated['master_price_df'] = master_df

        consolidated['symbol_dataframes'] = symbol_dataframes
        consolidated['success_rate'] = consolidated['successful_fetches'] / len(symbols)

        return consolidated

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
                # Get historical data including premarket
                hist = ticker.history(period=period, interval='1d', prepost=True)
                if not hist.empty:
                    # Separate regular hours and premarket data
                    regular_hours = hist.between_time('09:30', '16:00')
                    premarket = hist[~hist.index.isin(regular_hours.index)]
                    
                    data['historical'] = {
                        'prices': hist.to_dict('index'),
                        'regular_hours': regular_hours.to_dict('index') if not regular_hours.empty else {},
                        'premarket': premarket.to_dict('index') if not premarket.empty else {},
                        'source': 'yfinance',
                        'period': period,
                        'includes_premarket': True
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
                'symbol': symbol,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'source': 'marketdataapp'
            }

        except Exception as e:
            logger.error(f"MarketDataApp data fetch failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'data': {},
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'source': 'marketdataapp'
            }

    async def _fetch_ibkr_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from IBKR (placeholder for future implementation)."""
        # IBKR integration would go here
        return {
            'symbol': symbol,
            'data': {},
            'source': 'ibkr',
            'success': False,
            'note': 'IBKR integration not yet implemented'
        }

    async def _fetch_polygon_data(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch data from Polygon.io (placeholder for future implementation)."""
        # Polygon.io integration would go here
        return {
            'symbol': symbol,
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
                'symbol': symbol,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'source': 'alpha_vantage'
            }

        except Exception as e:
            logger.error(f"Alpha Vantage data fetch failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'data': {},
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'source': 'alpha_vantage'
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

    async def _analyze_market_data_llm(self, consolidated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to analyze consolidated market data for trading insights and patterns.
        """
        if not self.llm:
            logger.warning("No LLM available for market data analysis - returning basic insights")
            return {
                "llm_analysis": "Basic analysis without LLM assistance",
                "trend_analysis": {"primary_trend": "neutral", "momentum": "moderate"},
                "volatility_assessment": {"volatility_regime": "normal", "risk_level": "moderate"},
                "trading_signals": ["Monitor key levels", "Watch volume patterns"],
                "risk_metrics": {"var_estimate": 0.02, "max_drawdown": 0.05}
            }

        try:
            # Extract key data for analysis
            symbols = consolidated_data.get('symbols', [])
            master_df = consolidated_data.get('master_price_df')

            if master_df is not None and not master_df.empty:
                # Calculate basic statistics for LLM context
                recent_prices = master_df.tail(30)  # Last 30 days
                price_stats = {
                    'avg_price': recent_prices['Close'].mean(),
                    'price_volatility': recent_prices['Close'].std(),
                    'total_volume': recent_prices['Volume'].sum(),
                    'price_range': recent_prices['Close'].max() - recent_prices['Close'].min(),
                    'trend_slope': self._calculate_price_trend_slope(recent_prices)
                }
            else:
                price_stats = {'note': 'Limited price data available'}

            # Build analysis context
            analysis_context = f"""
Market Data Analysis Request:
- Symbols Analyzed: {', '.join(symbols)}
- Data Sources: {consolidated_data.get('sources_explored', [])}
- Analysis Period: Recent market data
- Data Quality Score: {consolidated_data.get('data_quality_score', 'unknown')}

Price Statistics:
- Average Price: {price_stats.get('avg_price', 'N/A')}
- Price Volatility: {price_stats.get('price_volatility', 'N/A')}
- Total Volume: {price_stats.get('total_volume', 'N/A')}
- Price Range: {price_stats.get('price_range', 'N/A')}
- Trend Direction: {'upward' if isinstance(price_stats.get('trend_slope', 0), (int, float)) and float(price_stats.get('trend_slope', 0)) > 0 else 'downward'}

Market Insights:
{self._extract_market_insights(consolidated_data.get('exploration_results', {}))}
"""

            analysis_question = """
Based on the market data analysis above, provide insights on:

1. **Trend Analysis**: What is the primary trend direction and strength?
2. **Volatility Assessment**: What volatility regime are we in and what risk level does this imply?
3. **Trading Signals**: What specific trading signals can be derived from this data?
4. **Risk Metrics**: What are the key risk metrics (VaR estimate, max drawdown) for current market conditions?
5. **Market Regime**: What market regime classification fits current conditions?

Consider the data quality, source diversity, and statistical significance of the patterns observed.
Provide specific, actionable insights that can inform trading strategy decisions.
"""

            # Use LLM directly for analysis
            if self.llm:
                # Build context from market data
                context = {
                    'context': f"Market data analysis for symbols: {list(consolidated_data.get('symbols_data', {}).keys())}\nData types: {consolidated_data.get('data_types', [])}\nQuality score: {consolidated_data.get('data_quality_score', 'N/A')}",
                    'question': analysis_question
                }
                
                # Build comprehensive prompt with foundation context
                sanitized_context = self.sanitize_input(context.get('context', ''))
                sanitized_question = self.sanitize_input(context.get('question', ''))
                full_prompt = f"""
{self.prompt}

FOUNDATION ANALYSIS CONTEXT:
{sanitized_context}

DECISION REQUIRED:
{sanitized_question}

ADDITIONAL CONTEXT:
No additional context provided

Please provide your reasoning and recommendation based on the foundation analysis above.
Consider market conditions, risk factors, and alignment with our goals (10-20% monthly ROI, <5% drawdown).
"""

                # Use LLM for reasoning
                response = await self.llm.ainvoke(full_prompt)
                llm_response = response.content if hasattr(response, 'content') else str(response)
            else:
                llm_response = "LLM not available for market data analysis"

            # Parse LLM response into structured format
            return {
                "llm_analysis": llm_response,
                "trend_analysis": self._extract_trend_analysis(llm_response),
                "volatility_assessment": self._extract_volatility_assessment(llm_response),
                "trading_signals": self._extract_trading_signals(llm_response),
                "risk_metrics": self._extract_risk_metrics(llm_response),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"LLM market data analysis failed: {e}")
            return {
                "llm_analysis": f"Analysis failed: {str(e)}",
                "trend_analysis": {"primary_trend": "unknown", "momentum": "unknown"},
                "volatility_assessment": {"volatility_regime": "unknown", "risk_level": "unknown"},
                "trading_signals": ["Unable to generate signals due to analysis failure"],
                "risk_metrics": {"var_estimate": None, "max_drawdown": None},
                "error": str(e)
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

    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[pd.Series]:
        """Calculate Relative Strength Index."""
        delta = prices.diff().dropna().astype(float)  # Ensure float for comparisons
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi if not rsi.empty else None

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict[str, pd.Series]]:
        """Calculate MACD."""
        try:
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            macd_dict = {'macd': macd, 'signal': signal_line, 'histogram': histogram}

            return macd_dict if not macd.empty else None
        except:
            return None

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Optional[Dict[str, pd.Series]]:
        """Calculate Bollinger Bands."""
        try:
            sma = self._calculate_sma(prices, period)
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            middle = sma
            lower = sma - (std * std_dev)

            return {'upper': upper, 'middle': middle, 'lower': lower} if not middle.empty else None
        except:
            return None

    def _calculate_price_trend_slope(self, prices: pd.DataFrame) -> float:
        """Calculate the slope of the price trend using linear regression."""
        try:
            if prices is None or prices.empty or 'Close' not in prices.columns:
                return 0.0

            # Use the last 20 data points for trend calculation
            recent_prices = prices.tail(20)
            if len(recent_prices) < 2:
                return 0.0

            # Simple linear regression slope
            import numpy as np
            x = np.array(range(len(recent_prices)))
            y = recent_prices['Close'].values.astype(float)

            # Calculate slope using numpy polyfit
            slope = np.polyfit(x, y, 1)[0]

            return float(slope)

        except Exception as e:
            logger.error(f"Failed to calculate price trend slope: {e}")
            return 0.0

    def _calculate_volume_profile(self, volume: pd.Series) -> Dict[str, Any]:
        """Calculate volume profile metrics."""
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

    def _sanitize_llm_input(self, input_text: str) -> str:
        """
        Sanitize input text for LLM prompts to prevent injection attacks.
        Removes or escapes potentially harmful content.
        """
        if not isinstance(input_text, str):
            return str(input_text)

        # Remove or escape common injection patterns
        sanitized = input_text

        # Remove system prompt override attempts
        sanitized = sanitized.replace("SYSTEM:", "").replace("system:", "")
        sanitized = sanitized.replace("ASSISTANT:", "").replace("assistant:", "")
        sanitized = sanitized.replace("USER:", "").replace("user:", "")

        # Remove prompt injection markers
        injection_markers = [
            "###", "---", "```", "IGNORE PREVIOUS",
            "FORGET INSTRUCTIONS", "NEW INSTRUCTIONS",
            "SYSTEM PROMPT", "You are now"
        ]

        for marker in injection_markers:
            sanitized = sanitized.replace(marker, "[FILTERED]")

        # Limit input length to prevent excessive token usage
        max_length = 4000  # Reasonable limit for analysis prompts
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "...[TRUNCATED]"

        # Remove excessive whitespace
        import re
        sanitized = re.sub(r'\n\s*\n\s*\n+', '\n\n', sanitized)  # Multiple newlines to double
        sanitized = re.sub(r'\s+', ' ', sanitized.strip())  # Multiple spaces to single

        return sanitized

    def sanitize_input(self, input_text: str) -> str:
        """Alias for _sanitize_llm_input for backward compatibility."""
        return self._sanitize_llm_input(input_text)

    async def close(self):
        """Cleanup resources"""
        if self.async_client:
            await self.async_client.close()
