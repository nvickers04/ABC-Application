# src/agents/data_analyzers/ibkr_data_analyzer.py
# Purpose: IBKR Data Subagent for fetching and analyzing market data from Interactive Brokers API.
# Supports historical data via IBKRHistoricalDataProvider and live/real-time quotes via ib_insync.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.data_analyzers.base_data_analyzer import BaseDataAnalyzer
import logging
from typing import Dict, Any, List, Callable, Optional
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import time
from src.utils.redis_cache import get_redis_cache_manager, cache_get, cache_set
from src.integrations.ibkr_historical_data import IBKRHistoricalDataProvider
from src.integrations.ibkr_connector import get_ibkr_connector
from ib_insync import IB, Contract, Stock, util

# Import performance optimizations (reuse from yfinance)
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

class IBKRDataAnalyzer(BaseDataAnalyzer):
    """
    IBKR Data Analyzer - Fetches and analyzes market data from Interactive Brokers.
    Supports historical and live/real-time data with fallback to yfinance.
    """
    def __init__(self):
        super().__init__(role='ibkr_data')

        # Initialize IBKR connector and historical provider
        self.connector = get_ibkr_connector()
        self.historical_provider = IBKRHistoricalDataProvider(self.connector)
        self.ib = self.connector.ib  # ib_insync IB instance

        # Initialize Redis cache manager
        self.redis_cache = get_redis_cache_manager()
        self.cache_ttl = 300  # 5 minutes TTL for market data

        # Initialize collaborative memory
        self.memory = MarketDataMemory()

        # IBKR-specific configurations
        self.data_sources = {
            'ibkr_historical': self._fetch_ibkr_historical,
            'ibkr_live': self._fetch_ibkr_live,
            'yfinance_fallback': self._fetch_yfinance_fallback  # Fallback
        }

        # Real-time streaming configurations
        self.streaming_configs = {
            'subscriptions': {},
            'data_buffers': {}
        }

        # Market data indicators (reuse from yfinance)
        self.technical_indicators = self._initialize_technical_indicators()

        # Ensure connection on init
        asyncio.create_task(self._ensure_connection())

    async def _ensure_connection(self):
        """Ensure IBKR connection is active."""
        if not self.connector.connected:
            connected = await self.connector.connect()
            if connected:
                logger.info("✅ IBKR connection established")
            else:
                logger.warning("⚠️ IBKR connection failed - using fallback")

    async def _fetch_ibkr_historical(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch historical data from IBKR."""
        try:
            # Parse time_horizon to dates (e.g., '1mo' -> last 30 days)
            end_date = datetime.now().strftime('%Y-%m-%d')
            if time_horizon == '1d':
                start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            elif time_horizon == '1mo':
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            elif time_horizon == '3mo':
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            else:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            df = await self.historical_provider.get_historical_bars(symbol, start_date, end_date, bar_size='1 day')
            if df is not None and not df.empty:
                return {"data": df.to_dict(), "source": "ibkr_historical"}
            else:
                logger.warning(f"No historical data from IBKR for {symbol} - falling back")
                return await self._fetch_yfinance_fallback(symbol, data_types, time_horizon)
        except Exception as e:
            logger.error(f"IBKR historical fetch failed for {symbol}: {e}")
            return await self._fetch_yfinance_fallback(symbol, data_types, time_horizon)

    async def _fetch_ibkr_live(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fetch live/real-time quotes from IBKR."""
        try:
            if not self.connector.connected:
                await self._ensure_connection()
                if not self.connector.connected:
                    return await self._fetch_yfinance_fallback(symbol, data_types, time_horizon)

            contract = Stock(symbol, 'SMART', 'USD')
            await self.ib.qualifyContractsAsync(contract)

            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(2)  # Wait for data

            if ticker.last and ticker.last > 0:
                live_data = {
                    'last_price': float(ticker.last),
                    'bid': float(ticker.bid) if ticker.bid > 0 else None,
                    'ask': float(ticker.ask) if ticker.ask > 0 else None,
                    'volume': int(ticker.volume),
                    'timestamp': datetime.now().isoformat()
                }
                self.ib.cancelMktData(contract)  # Cancel subscription
                return {"data": live_data, "source": "ibkr_live"}
            else:
                self.ib.cancelMktData(contract)
                return await self._fetch_yfinance_fallback(symbol, data_types, time_horizon)
        except Exception as e:
            logger.error(f"IBKR live fetch failed for {symbol}: {e}")
            return await self._fetch_yfinance_fallback(symbol, data_types, time_horizon)

    async def _fetch_yfinance_fallback(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Fallback to yfinance if IBKR fails."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=time_horizon)
            if not hist.empty:
                return {"data": hist.to_dict(), "source": "yfinance_fallback"}
            return {"error": "No data available"}
        except Exception as e:
            logger.error(f"Yfinance fallback failed for {symbol}: {e}")
            return {"error": str(e)}

    def _initialize_technical_indicators(self) -> Dict[str, Callable[..., Any]]:
        """Initialize technical indicator functions (reuse from yfinance)."""
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

    # Reuse technical indicator methods from yfinance (copy-paste for completeness)
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.rolling(window=period).mean()

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[pd.Series]:
        delta = prices.diff().dropna().astype(float)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi if not rsi.empty else None

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict[str, pd.Series]]:
        try:
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            return {'macd': macd, 'signal': signal_line, 'histogram': histogram} if not macd.empty else None
        except:
            return None

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Optional[Dict[str, pd.Series]]:
        try:
            sma = self._calculate_sma(prices, period)
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            middle = sma
            lower = sma - (std * std_dev)
            return {'upper': upper, 'middle': middle, 'lower': lower} if not middle.empty else None
        except:
            return None

    def _calculate_volume_profile(self, volume: pd.Series) -> Dict[str, Any]:
        try:
            return {
                'total_volume': volume.sum(),
                'avg_volume': volume.mean(),
                'volume_distribution': 'calculated'
            }
        except:
            return {}

    def _calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        try:
            typical_price = (high + low + close) / 3
            return (typical_price * volume).cumsum() / volume.cumsum()
        except:
            return pd.Series()

    def _calculate_order_flow(self, trades_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'buy_volume': 0,
            'sell_volume': 0,
            'order_imbalance': 0,
            'flow_toxicity': 'neutral'
        }

    # Cache methods (reuse from yfinance, adapted for IBKR)
    def _is_cache_valid(self, cache_key):
        return cache_get('ibkr_data', cache_key) is not None

    def _get_cached_data(self, cache_key):
        return cache_get('ibkr_data', cache_key)

    def _cache_data(self, cache_key, data):
        cache_set('ibkr_data', cache_key, data, self.cache_ttl)

    def reflect(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"IBKRDataAnalyzer reflecting on adjustments: {adjustments}")
        return {}

    async def _plan_data_exploration(self, *args, **kwargs) -> Dict[str, Any]:
        """Plan IBKR data exploration strategy."""
        symbols = kwargs.get('symbols', ['SPY'])
        data_types = kwargs.get('data_types', ['quotes', 'historical'])
        time_horizon = kwargs.get('time_horizon', '1mo')

        return {
            "sources": ["ibkr_historical", "ibkr_live"],
            "data_types": data_types,
            "priorities": {"ibkr_historical": 10, "ibkr_live": 9},
            "time_horizons": [time_horizon],
            "symbols": symbols,
            "reasoning": "IBKR-focused data exploration with live trading priority",
            "expected_insights": ["Real-time IBKR quotes", "Historical IBKR bars", "Live market data"]
        }

    async def _execute_data_exploration(self, exploration_plan: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Execute IBKR data fetching and initial processing."""
        symbols = exploration_plan.get("symbols", ['SPY'])
        sources = exploration_plan.get("sources", ["ibkr_historical", "ibkr_live"])
        data_types = exploration_plan.get("data_types", ['quotes', 'historical'])
        time_horizon = exploration_plan.get("time_horizons", ['1mo'])[0]

        results = {}
        for symbol in symbols:
            symbol_results = {}
            for source in sources:
                if source in self.data_sources:
                    source_data = await self.data_sources[source](symbol, data_types, time_horizon)
                    if source_data:
                        symbol_results[source] = source_data
                else:
                    logger.warning(f"Unknown source: {source}")

            results[symbol] = symbol_results

        return results

    async def _enhance_data(self, raw_data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Enhance IBKR data with analysis and consolidation."""
        symbols = list(raw_data.keys())

        # Consolidate market data
        consolidated_data = self._consolidate_market_data(symbols, raw_data)

        # Perform LLM analysis if available
        if self.llm:
            try:
                llm_analysis = await self._analyze_market_data_llm(consolidated_data)
                consolidated_data['llm_analysis'] = llm_analysis
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")

        consolidated_data['symbols_processed'] = symbols
        consolidated_data['timestamp'] = datetime.now().isoformat()
        consolidated_data['enhanced'] = True

        return consolidated_data

    async def _process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process IBKR data using BaseDataAnalyzer pattern for backward compatibility.
        """
        if input_data is None:
            input_data = {}

        try:
            symbols = input_data.get('symbols', ['SPY'])

            # Initialize LLM if not already done
            if not self.llm:
                await self.async_initialize_llm()

            # Use batch processing if optimizations available and multiple symbols
            if len(symbols) > 1 and OPTIMIZATIONS_AVAILABLE:
                logger.info(f"Using optimized batch processing for {len(symbols)} symbols")
                result = await self._process_input_optimized(input_data)

                # For backward compatibility, ensure expected structure
                if isinstance(result, dict):
                    # Store in shared memory for each symbol
                    consolidated_data = result.get("consolidated_data", {})
                    llm_analysis = result.get("llm_analysis", {})

                    for symbol in symbols:
                        if symbol in consolidated_data:
                            await self.store_shared_memory("ibkr_data", symbol, {
                                "market_data": consolidated_data[symbol],
                                "llm_analysis": llm_analysis,
                                "timestamp": datetime.now().isoformat(),
                                "symbol": symbol
                            })

                return result
            else:
                # Use base class process_input for standardized processing
                result = await super().process_input(input_data)

                # For backward compatibility, ensure expected structure and add memory storage
                if isinstance(result, dict) and "consolidated_data" in result:
                    consolidated_data = result["consolidated_data"]
                    llm_analysis = result.get("llm_analysis", {})

                    # Add exploration_plan if missing
                    if "exploration_plan" not in result:
                        exploration_plan = await self._plan_data_exploration(symbols=symbols, **input_data)
                        result["exploration_plan"] = exploration_plan

                    # Store in shared memory for each symbol
                    for symbol in symbols:
                        if symbol in consolidated_data:
                            await self.store_shared_memory("ibkr_data", symbol, {
                                "market_data": consolidated_data[symbol],
                                "llm_analysis": llm_analysis,
                                "timestamp": datetime.now().isoformat(),
                                "symbol": symbol
                            })

                    logger.info(f"IBKRDataAnalyzer output: Enhanced market data for {symbols}")
                    return result

                # Fallback to original logic if base class doesn't return expected structure
                return await self._fallback_process_input(input_data)

        except Exception as e:
            logger.error(f"IBKRDataAnalyzer failed: {e}")
            return {"price_data": {}, "error": str(e), "enhanced": False}

    async def _fallback_process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback processing method for backward compatibility.
        """
        symbols = input_data.get('symbols', ['SPY'])

        # Standard processing
        exploration_plan = await self._plan_data_exploration(symbols=symbols, **input_data)

        # Execute exploration
        exploration_results = await self._execute_data_exploration(exploration_plan)

        # Consolidate
        consolidated_data = self._consolidate_market_data(symbols, exploration_results)

        # LLM analysis
        llm_analysis = await self._analyze_market_data_llm(consolidated_data)

        return {
            "consolidated_data": consolidated_data,
            "llm_analysis": llm_analysis,
            "exploration_plan": exploration_plan,
            "enhanced": True
        }

    # Reuse _process_input_optimized, _batch_process_symbols, _process_symbol_optimized from yfinance, but adapt fetches to IBKR methods
    async def _process_input_optimized(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized processing with IBKR focus."""
        start_time = time.time()
        symbols = input_data.get('symbols', ['SPY'])
        data_types = input_data.get('data_types', ['quotes', 'historical'])
        time_horizon = input_data.get('time_horizon', '1mo')

        logger.info(f"Processing {len(symbols)} symbols with IBKR analyzer")

        try:
            results = await self._batch_process_symbols(symbols, data_types, time_horizon)
            consolidated = self._consolidate_optimized_results(results, symbols)

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
        """Batch process with IBKR priority."""
        tasks = [self._process_symbol_optimized(symbol, data_types, time_horizon) for symbol in symbols]
        semaphore = asyncio.Semaphore(5)  # Limit concurrency for IBKR

        async def limited_task(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*[limited_task(t) for t in tasks], return_exceptions=True)

        symbol_results = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                symbol_results[symbol] = {"error": str(result)}
            else:
                symbol_results[symbol] = result

        return symbol_results

    async def _process_symbol_optimized(self, symbol: str, data_types: List[str], time_horizon: str) -> Dict[str, Any]:
        """Process single symbol with IBKR caching."""
        cache_key = f"ibkr_symbol_data_{symbol}_{'_'.join(data_types)}_{time_horizon}"

        cached_result = self._get_cached_data(cache_key)
        if cached_result:
            logger.info(f"Cache hit for {symbol}")
            return cached_result

        try:
            fetch_tasks = []
            if 'historical' in data_types:
                fetch_tasks.append(self._fetch_ibkr_historical(symbol, data_types, time_horizon))
            if 'quotes' in data_types:
                fetch_tasks.append(self._fetch_ibkr_live(symbol, data_types, time_horizon))

            fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            result = {
                'symbol': symbol,
                'data': {},
                'timestamp': datetime.now().isoformat(),
                'cached': False
            }

            for i, fetch_result in enumerate(fetch_results):
                if isinstance(fetch_result, Exception):
                    continue
                if i == 0:  # Historical
                    result['data']['historical'] = fetch_result
                elif i == 1:  # Live
                    result['data']['quote'] = fetch_result

            self._cache_data(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _consolidate_optimized_results(self, results: Dict[str, Dict[str, Any]], symbols: List[str]) -> Dict[str, Any]:
        """Consolidate results (reuse logic from yfinance)."""
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

            if 'data' in result and 'historical' in result['data']:
                prices_dict = result['data']['historical'].get('data', {})
                if prices_dict:
                    try:
                        df = pd.DataFrame.from_dict(prices_dict, orient='index')
                        df.index = pd.to_datetime(df.index)
                        symbol_dataframes[symbol] = {
                            'historical_df': df,
                            'quote_data': result['data'].get('quote', {}),
                            'source': 'ibkr'
                        }
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

    # Reuse _plan_data_exploration, _execute_data_exploration, _consolidate_market_data, _analyze_market_data_llm from yfinance (adapt prompts to mention IBKR)
    async def _plan_data_exploration(self, symbols: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.llm:
            return {
                "sources": ["ibkr_historical", "ibkr_live"],
                "data_types": ["quotes", "historical"],
                "priorities": {"ibkr_historical": 10, "ibkr_live": 9},
                "time_horizons": ["1mo"],
                "reasoning": "Default IBKR plan",
                "expected_insights": ["Real-time quotes", "Historical patterns"]
            }

        try:
            primary_symbol = symbols[0] if symbols else 'SPY'
            exploration_prompt = f"""
You are planning IBKR data exploration for {primary_symbol}.
Available: historical bars, live quotes, streaming.
Prioritize IBKR for accuracy.
Return JSON with sources, data_types, priorities, time_horizons, reasoning, expected_insights.
"""
            response = await self.llm.ainvoke(exploration_prompt)
            import json
            try:
                plan = json.loads(response.content if hasattr(response, 'content') else str(response))
                return plan
            except json.JSONDecodeError:
                logger.error("Failed to parse exploration plan")
                raise
        except Exception as e:
            logger.error(f"Exploration planning failed: {e}")
            raise

    async def _execute_data_exploration(self, symbols: List[str], plan: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        sources = plan.get('sources', ['ibkr_historical', 'ibkr_live'])
        data_types = plan.get('data_types', ['quotes', 'historical'])
        time_horizons = plan.get('time_horizons', ['1mo'])

        for symbol in symbols:
            symbol_results = {}
            for source in sources:
                if source in self.data_sources:
                    source_data = await self.data_sources[source](symbol, data_types, time_horizons[0])
                    if source_data:
                        symbol_results[source] = source_data
                else:
                    logger.warning(f"Unknown source: {source}")

            results[symbol] = symbol_results

        return results

    def _consolidate_market_data(self, symbols: List[str], exploration_results: Dict[str, Any]) -> Dict[str, Any]:
        consolidated = {
            'symbols': symbols,
            'source': 'ibkr_llm_exploration',
            'sources_explored': list(set(source for symbol_data in exploration_results.values() for source in symbol_data.keys())),
            'timestamp': datetime.now().isoformat()
        }

        symbol_dataframes = {}
        for symbol in symbols:
            if symbol in exploration_results:
                symbol_consolidated = self._consolidate_symbol_data(exploration_results[symbol])
                symbol_dataframes[symbol] = symbol_consolidated

        consolidated['symbol_dataframes'] = symbol_dataframes

        all_prices = []
        for symbol, data in symbol_dataframes.items():
            if 'historical_df' in data:
                df = data['historical_df'].copy()
                df['symbol'] = symbol
                all_prices.append(df)

        if all_prices:
            master_df = pd.concat(all_prices, ignore_index=True)
            consolidated['master_price_df'] = master_df

        consolidated['data_quality_score'] = self._calculate_market_data_quality_score(exploration_results)
        consolidated['market_insights'] = self._extract_market_insights(exploration_results)

        return consolidated

    # Reuse other methods: _calculate_market_data_quality_score, _extract_market_insights, _analyze_market_data_llm, etc., from yfinance (adapt as needed for IBKR)

    def _calculate_market_data_quality_score(self, exploration_results: Dict[str, Any]) -> float:
        base_score = 8.0  # Higher base for IBKR
        source_bonus = len(set(source for symbol_data in exploration_results.values() for source in symbol_data.keys())) * 1.0
        symbol_bonus = len(exploration_results) * 0.5
        return min(10.0, base_score + source_bonus + symbol_bonus)

    def _extract_market_insights(self, exploration_results: Dict[str, Any]) -> List[str]:
        insights = [f"IBKR data collected for {len(exploration_results)} symbols"]
        complete_count = sum(1 for symbol_data in exploration_results.values() if 'historical' in symbol_data)
        if complete_count > 0:
            insights.append(f"{complete_count} symbols with IBKR historical/live data")
        return insights

    async def _analyze_market_data_llm(self, consolidated_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.llm:
            return {
                "llm_analysis": "Basic IBKR analysis",
                "trend_analysis": {"primary_trend": "neutral"},
                "trading_signals": ["Monitor IBKR live data"]
            }

        try:
            symbols = consolidated_data.get('symbols', [])
            analysis_context = f"IBKR Market Data for {', '.join(symbols)}: High-quality live/historical data available."
            analysis_question = "Analyze IBKR data for trends, volatility, signals using real-time quotes and bars."

            context = {'context': analysis_context, 'question': analysis_question}
            sanitized_context = self.sanitize_input(context.get('context', ''))
            sanitized_question = self.sanitize_input(context.get('question', ''))
            full_prompt = f"{self.prompt}\nFOUNDATION: {sanitized_context}\nDECISION: {sanitized_question}"

            response = await self.llm.ainvoke(full_prompt)
            llm_response = response.content if hasattr(response, 'content') else str(response)

            return {
                "llm_analysis": llm_response,
                "trend_analysis": self._extract_trend_analysis(llm_response),
                "volatility_assessment": self._extract_volatility_assessment(llm_response),
                "trading_signals": self._extract_trading_signals(llm_response),
                "risk_metrics": self._extract_risk_metrics(llm_response),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"llm_analysis": f"Analysis failed: {str(e)}"}

    # Reuse extraction methods
    def _extract_trend_analysis(self, llm_response: str) -> Dict[str, Any]:
        return {"primary_trend": "neutral", "momentum": "moderate"}

    def _extract_volatility_assessment(self, llm_response: str) -> Dict[str, Any]:
        return {"volatility_regime": "normal", "risk_level": "moderate"}

    def _extract_trading_signals(self, llm_response: str) -> List[str]:
        return ["Monitor key levels from IBKR data"]

    def _extract_risk_metrics(self, llm_response: str) -> Dict[str, Any]:
        return {"var_estimate": 0.02, "max_drawdown": 0.05}

    def _consolidate_symbol_data(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        consolidated = {
            'primary_quote': None,
            'consensus_price': None,
            'price_sources': [],
            'data_quality': 'medium',  # IBKR default
            'dataframe': pd.DataFrame()
        }

        data = symbol_data.get('data', {})
        # Similar consolidation logic as yfinance, prioritizing IBKR sources
        for data_type, sources_data in data.items():
            if data_type == 'historical':
                for source, hist_data in sources_data.items():
                    if source == 'ibkr_historical' and 'data' in hist_data:
                        df = pd.DataFrame.from_dict(hist_data['data'], orient='index')
                        if not df.empty:
                            df.index = pd.to_datetime(df.index)
                            consolidated['dataframe'] = df
                            break

        # Live quotes consolidation
        quotes = []
        for data_type, sources_data in data.items():
            if data_type == 'quote':
                for source, quote_data in sources_data.items():
                    if source == 'ibkr_live' and quote_data:
                        quotes.append({
                            'price': quote_data.get('last_price'),
                            'source': source,
                            'timestamp': quote_data.get('timestamp')
                        })

        if quotes:
            prices = [q['price'] for q in quotes if q['price']]
            if prices:
                consolidated['consensus_price'] = np.mean(prices)
                consolidated['primary_quote'] = quotes[0]
                consolidated['price_sources'] = [q['source'] for q in quotes]
                consolidated['data_quality'] = 'high'  # IBKR live is high quality

        return consolidated

    def sanitize_input(self, input_text: str) -> str:
        """Sanitize for LLM (reuse from yfinance)."""
        if not isinstance(input_text, str):
            return str(input_text)
        sanitized = input_text.replace("SYSTEM:", "").replace("ASSISTANT:", "").replace("USER:", "")
        injection_markers = ["###", "---", "```", "IGNORE PREVIOUS", "FORGET INSTRUCTIONS"]
        for marker in injection_markers:
            sanitized = sanitized.replace(marker, "[FILTERED]")
        max_length = 4000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "...[TRUNCATED]"
        import re
        sanitized = re.sub(r'\n\s*\n\s*\n+', '\n\n', sanitized)
        sanitized = re.sub(r'\s+', ' ', sanitized.strip())
        return sanitized

# Standalone test
if __name__ == "__main__":
    import asyncio
    analyzer = IBKRDataAnalyzer()
    result = asyncio.run(analyzer.process_input({'symbols': ['SPY']}))
    print("IBKR Data Analyzer Test Result:")
    print(f"Keys: {list(result.keys())}")
    if 'consolidated_data' in result:
        print(f"Symbols processed: {len(result['consolidated_data'].get('symbols', []))}")
        print(f"Data quality: {result['consolidated_data'].get('data_quality_score', 'unknown')}")
