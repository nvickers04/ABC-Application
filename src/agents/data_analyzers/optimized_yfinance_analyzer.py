
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.agents.data_analyzers.yfinance_data_analyzer import YfinanceDataAnalyzer
from optimizations.performance_optimizations import AsyncYFianceClient, OptimizedRedisCache, CircuitBreaker

logger = logging.getLogger(__name__)

class OptimizedYfinanceDataAnalyzer(YfinanceDataAnalyzer):
    """Optimized Yfinance Data Analyzer with async operations and caching"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize optimized components
        self.async_client = AsyncYFianceClient()
        self.cache = OptimizedRedisCache(self.redis_cache) if self.redis_cache else None
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

    async def process_input_optimized(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimized version with async operations and caching"""
        start_time = time.time()

        if not input_data:
            return {"error": "No input data provided"}

        symbols = input_data.get('symbols', [])
        data_types = input_data.get('data_types', ['quotes', 'historical'])
        time_horizon = input_data.get('time_horizon', '1mo')

        if not symbols:
            return {"error": "No symbols provided"}

        logger.info(f"Processing {len(symbols)} symbols with optimized analyzer")

        try:
            # Batch process all symbols asynchronously
            results = await self._batch_process_symbols(symbols, data_types, time_horizon)

            # Consolidate results
            consolidated = self._consolidate_optimized_results(results, symbols)

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
        if self.cache:
            cached_result = await self.cache.get('symbol_data', cache_key)
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
            if self.cache:
                await self.cache.set('symbol_data', cache_key, result, ttl=300)  # 5 minutes

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
                        import pandas as pd
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
            import pandas as pd
            consolidated['master_price_df'] = pd.concat(all_prices, ignore_index=True)

        consolidated['symbol_dataframes'] = symbol_dataframes
        consolidated['success_rate'] = consolidated['successful_fetches'] / len(symbols)

        return consolidated

    async def close(self):
        """Cleanup resources"""
        await self.async_client.close()
