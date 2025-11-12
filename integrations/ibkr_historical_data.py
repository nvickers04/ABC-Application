# integrations/ibkr_historical_data.py
# Purpose: IBKR Historical Data Provider for backtesting and simulation
# Provides access to IBKR's extensive historical market data for comprehensive simulations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import os
import time

from ib_insync import IB, Contract, Stock, util
# Local imports
from .ibkr_connector import get_ibkr_connector

logger = logging.getLogger(__name__)

class IBKRHistoricalDataProvider:
    """
    IBKR Historical Data Provider for backtesting and simulation.
    Provides access to IBKR's comprehensive historical market data.
    """

    def __init__(self, connector=None):
        """
        Initialize IBKR Historical Data Provider

        Args:
            connector: Optional IBKRConnector instance. If None, creates new one.
        """
        self.connector = connector or get_ibkr_connector()
        self.data_cache = {}  # Cache for downloaded data
        self.cache_dir = Path("data/ibkr_historical_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def get_historical_bars(self, symbol: str, start_date: str, end_date: str,
                                 bar_size: str = '1 day', use_rth: bool = True,
                                 max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Get historical bars from IBKR for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'SPY', 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bar_size: Bar size ('1 secs', '5 secs', '15 secs', '30 secs', '1 min', '2 mins',
                       '3 mins', '5 mins', '15 mins', '30 mins', '1 hour', '1 day')
            use_rth: Use regular trading hours only
            max_retries: Maximum number of retry attempts

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_{bar_size}_{use_rth}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        # Check cache first
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded {symbol} data from cache ({len(df)} bars)")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")

        # Ensure connection
        if not self.connector.connected:
            connected = await self.connector.connect()
            if not connected:
                logger.error(f"Failed to connect to IBKR for {symbol} historical data")
                return None

        # Create contract
        contract = Stock(symbol, 'SMART', 'USD')

        # Calculate duration string for IBKR
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        days_diff = (end_dt - start_dt).days

        # IBKR duration strings
        if days_diff <= 1:
            duration = '1 D'
        elif days_diff <= 7:
            duration = '1 W'
        elif days_diff <= 30:
            duration = '1 M'
        elif days_diff <= 365:
            duration = f'{days_diff} D'
        else:
            duration = f'{days_diff // 365 + 1} Y'

        logger.info(f"Requesting {symbol} historical data: {start_date} to {end_date} ({duration}, {bar_size})")

        for attempt in range(max_retries):
            try:
                # Request historical data
                bars = await self.connector._run_in_executor(
                    self.connector.ib.reqHistoricalData,
                    contract,
                    endDateTime='',  # Empty string = current time
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow='TRADES',
                    useRTH=use_rth,
                    formatDate=1,
                    keepUpToDate=False,
                    chartOptions=[]
                )

                if not bars:
                    logger.warning(f"No historical data received for {symbol} (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return None

                # Convert to DataFrame
                data = []
                for bar in bars:
                    data.append({
                        'timestamp': bar.date,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    })

                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')

                # Filter date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]

                # Add returns column
                df['returns'] = df['close'].pct_change()

                # Add symbol column
                df['symbol'] = symbol

                logger.info(f"Successfully retrieved {len(df)} bars for {symbol}")

                # Cache the data
                try:
                    df.to_parquet(cache_file)
                    logger.debug(f"Cached {symbol} data to {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to cache {symbol} data: {e}")

                return df

            except Exception as e:
                logger.error(f"Error getting historical data for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None

        return None

    async def get_multiple_symbols_data(self, symbols: List[str], start_date: str, end_date: str,
                                      bar_size: str = '1 day', max_concurrent: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols concurrently.

        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bar_size: Bar size setting
            max_concurrent: Maximum concurrent requests

        Returns:
            Dict mapping symbols to DataFrames
        """
        async def fetch_symbol(symbol):
            return symbol, await self.get_historical_bars(symbol, start_date, end_date, bar_size)

        # Process in batches to avoid overwhelming IBKR
        results = {}
        for i in range(0, len(symbols), max_concurrent):
            batch = symbols[i:i + max_concurrent]
            logger.info(f"Processing batch {i//max_concurrent + 1}: {batch}")

            tasks = [fetch_symbol(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    continue

                symbol, data = result
                if data is not None:
                    results[symbol] = data
                else:
                    logger.warning(f"No data retrieved for {symbol}")

            # Small delay between batches
            if i + max_concurrent < len(symbols):
                await asyncio.sleep(1)

        logger.info(f"Completed data retrieval for {len(results)}/{len(symbols)} symbols")
        return results

    async def get_portfolio_historical_data(self, symbols: List[str], weights: Dict[str, float],
                                          start_date: str, end_date: str, bar_size: str = '1 day') -> Dict[str, Any]:
        """
        Get historical data for a portfolio and calculate portfolio-level metrics.

        Args:
            symbols: List of symbols in portfolio
            weights: Dict mapping symbols to weights (should sum to 1.0)
            start_date: Start date
            end_date: End date
            bar_size: Bar size

        Returns:
            Dict with individual symbol data and portfolio metrics
        """
        # Get data for all symbols
        symbol_data = await self.get_multiple_symbols_data(symbols, start_date, end_date, bar_size)

        if not symbol_data:
            return {'error': 'No historical data retrieved for portfolio'}

        # Find common dates across all symbols
        common_dates = None
        for symbol, df in symbol_data.items():
            dates = set(df.index.date)
            common_dates = dates if common_dates is None else common_dates.intersection(dates)

        if not common_dates:
            return {'error': 'No common trading dates found across portfolio symbols'}

        common_dates = sorted(list(common_dates))

        # Calculate portfolio returns
        portfolio_returns = []
        portfolio_values = []

        for date in common_dates:
            daily_return = 0.0
            portfolio_value = 0.0

            for symbol in symbols:
                if symbol in symbol_data:
                    df = symbol_data[symbol]
                    date_data = df[df.index.date == date]
                    if not date_data.empty:
                        weight = weights.get(symbol, 0.0)
                        symbol_return = date_data.iloc[0]['returns']
                        if not pd.isna(symbol_return):
                            daily_return += weight * symbol_return
                        portfolio_value += weight * date_data.iloc[0]['close']

            if not pd.isna(daily_return):
                portfolio_returns.append({'date': date, 'return': daily_return, 'value': portfolio_value})

        return {
            'symbol_data': symbol_data,
            'portfolio_returns': portfolio_returns,
            'common_dates': common_dates,
            'symbols_retrieved': list(symbol_data.keys()),
            'total_symbols_requested': len(symbols)
        }

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached historical data.

        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            # Clear specific symbol cache
            for cache_file in self.cache_dir.glob(f"*{symbol}*.parquet"):
                try:
                    cache_file.unlink()
                    logger.info(f"Cleared cache for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to clear cache for {symbol}: {e}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.parquet"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clear cache file {cache_file}: {e}")
            logger.info("Cleared all historical data cache")

    async def get_available_data_range(self, symbol: str) -> Optional[Tuple[str, str]]:
        """
        Get the available data range for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (start_date, end_date) or None if failed
        """
        # Try to get a small sample to determine date range
        sample_data = await self.get_historical_bars(symbol, '2020-01-01', '2020-01-05', '1 day')
        if sample_data is not None and not sample_data.empty:
            # IBKR typically has data going back several years
            # This is a rough estimate - in practice you'd need to check IBKR's data availability
            return ('2000-01-01', datetime.now().strftime('%Y-%m-%d'))
        return None

# Convenience functions
async def get_ibkr_historical_data(symbol: str, start_date: str, end_date: str,
                                  bar_size: str = '1 day') -> Optional[pd.DataFrame]:
    """
    Convenience function to get historical data from IBKR.

    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        bar_size: Bar size

    Returns:
        DataFrame with historical data or None
    """
    provider = IBKRHistoricalDataProvider()
    return await provider.get_historical_bars(symbol, start_date, end_date, bar_size)

async def get_portfolio_historical_data(symbols: List[str], weights: Dict[str, float],
                                       start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Convenience function to get portfolio historical data from IBKR.

    Args:
        symbols: List of symbols
        weights: Portfolio weights
        start_date: Start date
        end_date: End date

    Returns:
        Dict with portfolio data
    """
    provider = IBKRHistoricalDataProvider()
    return await provider.get_portfolio_historical_data(symbols, weights, start_date, end_date)

# Test function
async def test_ibkr_historical_data():
    """Test IBKR historical data functionality"""
    print("🧪 Testing IBKR Historical Data Provider")
    print("=" * 50)

    provider = IBKRHistoricalDataProvider()

    # Test single symbol
    print("\n📊 Testing single symbol (SPY)...")
    spy_data = await provider.get_historical_bars('SPY', '2024-01-01', '2024-01-31', '1 day')
    if spy_data is not None:
        print(f"✅ Retrieved {len(spy_data)} bars for SPY")
        print(f"   Date range: {spy_data.index.min()} to {spy_data.index.max()}")
        print(f"   Sample close: ${spy_data['close'].iloc[-1]:.2f}")
    else:
        print("❌ Failed to retrieve SPY data")

    # Test multiple symbols
    print("\n📊 Testing multiple symbols...")
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    multi_data = await provider.get_multiple_symbols_data(symbols, '2024-01-01', '2024-01-31', '1 day')
    print(f"✅ Retrieved data for {len(multi_data)}/{len(symbols)} symbols")

    # Test portfolio data
    print("\n📊 Testing portfolio data...")
    weights = {'AAPL': 0.4, 'MSFT': 0.4, 'GOOGL': 0.2}
    portfolio_data = await provider.get_portfolio_historical_data(symbols, weights, '2024-01-01', '2024-01-31')
    if 'portfolio_returns' in portfolio_data:
        print(f"✅ Portfolio data retrieved with {len(portfolio_data['portfolio_returns'])} return periods")
    else:
        print("❌ Failed to retrieve portfolio data")

    print("\n✅ IBKR Historical Data Provider test completed")

if __name__ == "__main__":
    asyncio.run(test_ibkr_historical_data())