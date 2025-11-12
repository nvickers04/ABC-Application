# src/utils/historical_simulation_engine.py
# Purpose: Comprehensive historical simulation engine for portfolio backtesting and analysis
# Provides multi-asset, multi-strategy simulation capabilities with realistic trading conditions

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
import matplotlib.pyplot as plt
import io
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration for historical simulation"""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    symbols: List[str] = None
    weights: Dict[str, float] = None
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'
    transaction_costs: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    benchmark_symbol: str = 'SPY'
    risk_free_rate: float = 0.02

class HistoricalSimulationEngine:
    """
    Comprehensive engine for running historical portfolio simulations.
    Supports multi-asset portfolios, rebalancing, transaction costs, and detailed analytics.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions = {}  # symbol -> shares
        self.trades = []
        self.portfolio_history = []
        self.benchmark_history = []

        # Set default equal weights if not provided
        if config.weights is None and config.symbols:
            self.config.weights = {symbol: 1.0 / len(config.symbols) for symbol in config.symbols}

    def fetch_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for multiple symbols concurrently.

        Args:
            symbols: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dict mapping symbols to price DataFrames
        """
        def fetch_symbol_data(symbol):
            try:
                logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)

                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    return symbol, None

                # Handle MultiIndex columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    logger.info(f"MultiIndex columns for {symbol}: {data.columns.tolist()}")
                    # For single symbol download, yfinance creates MultiIndex with (price_type, symbol)
                    # We want the price_type level (Close, High, Low, Open, Volume)
                    try:
                        data.columns = data.columns.get_level_values(0)
                        logger.info(f"After flattening level 0: {data.columns.tolist()}")
                    except Exception as e:
                        logger.error(f"Failed to flatten columns for {symbol}: {e}")
                        return symbol, None

                # Ensure we have the required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_cols):
                    logger.warning(f"Missing required columns for {symbol}: {list(data.columns)}")
                    return symbol, None

                # Add returns column - ensure it's a Series operation
                if isinstance(data['Close'], pd.Series):
                    data = data.copy()  # Make a copy to avoid SettingWithCopyWarning
                    data['Returns'] = data['Close'].pct_change()
                    data['Symbol'] = symbol
                else:
                    logger.error(f"Unexpected data structure for {symbol} Close column: {type(data['Close'])}")
                    return symbol, None

                return symbol, data

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return symbol, None

        # Fetch data sequentially to avoid yfinance batching issues
        data_dict = {}
        for symbol in symbols:
            result = fetch_symbol_data(symbol)
            symbol_name, data = result
            if data is not None:
                data_dict[symbol_name] = data

        logger.info(f"Successfully fetched data for {len(data_dict)} out of {len(symbols)} symbols")
        return data_dict

    def run_portfolio_simulation(self) -> Dict[str, Any]:
        """
        Run complete portfolio simulation with rebalancing and transaction costs.

        Returns:
            Comprehensive simulation results
        """
        try:
            # Fetch historical data
            symbols = self.config.symbols + [self.config.benchmark_symbol] if self.config.symbols else [self.config.benchmark_symbol]
            historical_data = self.fetch_historical_data(symbols, self.config.start_date, self.config.end_date)

            if not historical_data:
                return {'error': 'No historical data available for simulation'}

            # Get common trading dates
            common_dates = None
            for symbol, data in historical_data.items():
                dates = set(data.index.date)
                common_dates = dates if common_dates is None else common_dates.intersection(dates)

            if not common_dates:
                return {'error': 'No common trading dates found'}

            common_dates = sorted(list(common_dates))

            # Initialize portfolio
            self._initialize_portfolio(historical_data, common_dates[0])

            # Run simulation day by day
            for i, current_date in enumerate(common_dates):
                self._process_trading_day(historical_data, current_date, i, common_dates)

            # Calculate final results
            results = self._calculate_simulation_results(historical_data, common_dates)

            return results

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {'error': str(e)}

    def _initialize_portfolio(self, historical_data: Dict[str, pd.DataFrame], start_date: datetime.date):
        """Initialize portfolio positions at the start of simulation"""
        if not self.config.symbols:
            return

        # Find the first valid trading day
        start_prices = {}
        for symbol in self.config.symbols:
            if symbol in historical_data:
                data = historical_data[symbol]
                # Find the first valid price on or after start_date
                valid_dates = data.index[data.index.date >= start_date]
                if not valid_dates.empty:
                    first_date = valid_dates[0]
                    start_price = data.loc[first_date, 'Open']
                    start_prices[symbol] = start_price
                    logger.info(f"Start price for {symbol} on {first_date.date()}: ${start_price:.2f}")

        if not start_prices:
            logger.warning("No valid start prices found for portfolio initialization")
            return

        if not start_prices:
            logger.warning("No valid start prices found for portfolio initialization")
            return

        # Allocate capital according to weights
        total_allocated = 0
        for symbol, weight in self.config.weights.items():
            if symbol in start_prices:
                price = start_prices[symbol]
                allocation = self.portfolio_value * weight
                shares = allocation / price
                self.positions[symbol] = shares
                total_allocated += allocation

                logger.info(f"Initial position: {symbol} - {shares:.2f} shares @ ${price:.2f} = ${allocation:.2f}")

                # Record initial trade
                self.trades.append({
                    'date': start_date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'value': allocation,
                    'commission': allocation * self.config.transaction_costs
                })

        self.cash = self.portfolio_value - total_allocated
        logger.info(f"Initial cash after allocation: ${self.cash:.2f}, Total allocated: ${total_allocated:.2f}")

        # Record initial portfolio state
        self._record_portfolio_state(start_date, start_prices)

    def _process_trading_day(self, historical_data: Dict[str, pd.DataFrame],
                           current_date: datetime.date, day_index: int, all_dates: List[datetime.date]):
        """Process a single trading day"""
        # Check if rebalancing is needed
        if self._should_rebalance(current_date, day_index, all_dates):
            self._rebalance_portfolio(historical_data, current_date)

        # Update portfolio value for the day
        self._update_portfolio_value(historical_data, current_date)

    def _should_rebalance(self, current_date: datetime.date, day_index: int, all_dates: List[datetime.date]) -> bool:
        """Determine if portfolio should be rebalanced on this date"""
        if self.config.rebalance_frequency == 'daily':
            return True
        elif self.config.rebalance_frequency == 'weekly':
            # Rebalance on Mondays
            return current_date.weekday() == 0
        elif self.config.rebalance_frequency == 'monthly':
            # Rebalance on first trading day of month
            if day_index == 0:
                return True
            prev_date = all_dates[day_index - 1]
            return current_date.month != prev_date.month
        elif self.config.rebalance_frequency == 'quarterly':
            # Rebalance on first day of quarter
            if day_index == 0:
                return True
            prev_date = all_dates[day_index - 1]
            return (current_date.month - 1) // 3 != (prev_date.month - 1) // 3
        elif self.config.rebalance_frequency == 'none':
            # Never rebalance
            return False

        return False

    def _rebalance_portfolio(self, historical_data: Dict[str, pd.DataFrame], current_date: datetime.date):
        """Rebalance portfolio to target weights"""
        if not self.config.symbols:
            return

        # Get current prices
        current_prices = {}
        for symbol in self.config.symbols:
            if symbol in historical_data:
                data = historical_data[symbol]
                date_data = data[data.index.date == current_date]
                if not date_data.empty:
                    current_prices[symbol] = date_data.iloc[0]['Open']

        if not current_prices:
            return

        # Calculate current portfolio value
        current_value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
                current_value += shares * current_prices[symbol]

        # Skip rebalancing if portfolio value is negative or zero
        if current_value <= 0:
            logger.info(f"Skipping rebalancing on {current_date}: portfolio value ${current_value:,.0f} is negative or zero")
            return

        logger.info(f"Rebalancing on {current_date}: current_value=${current_value:,.0f}, cash=${self.cash:,.0f}")

        # Calculate target positions
        target_positions = {}
        for symbol, weight in self.config.weights.items():
            if symbol in current_prices:
                target_value = current_value * weight
                target_shares = target_value / current_prices[symbol]
                target_positions[symbol] = target_shares
                logger.info(f"  {symbol}: {self.positions.get(symbol, 0):.2f} -> {target_shares:.2f} shares")

        # Execute rebalancing trades
        for symbol in self.config.symbols:
            if symbol in current_prices:
                current_shares = self.positions.get(symbol, 0)
                target_shares = target_positions.get(symbol, 0)
                shares_to_trade = target_shares - current_shares

                if abs(shares_to_trade) > 0.01:  # Minimum trade size
                    price = current_prices[symbol]
                    # Apply slippage: increase price for buying, decrease for selling
                    if shares_to_trade > 0:  # Buying
                        effective_price = price * (1 + self.config.slippage)
                    else:  # Selling
                        effective_price = price * (1 - self.config.slippage)

                    value = abs(shares_to_trade) * effective_price
                    commission = value * self.config.transaction_costs

                    # Correct cash calculation: subtract cost for buying, add proceeds for selling
                    self.cash -= shares_to_trade * effective_price + commission

                    action = 'BUY' if shares_to_trade > 0 else 'SELL'

                    logger.info(f"  Trading {symbol}: {action} {abs(shares_to_trade):.2f} shares @ ${effective_price:.2f} (slippage adjusted)")

                    self.trades.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': action,
                        'shares': abs(shares_to_trade),
                        'price': effective_price,
                        'value': value,
                        'commission': commission
                    })

                    self.positions[symbol] = target_shares

        logger.info(f"After rebalancing: cash=${self.cash:,.0f}")

    def _update_portfolio_value(self, historical_data: Dict[str, pd.DataFrame], current_date: datetime.date):
        """Update portfolio value for the current date"""
        # Get closing prices for the day
        closing_prices = {}
        for symbol in self.config.symbols or []:
            if symbol in historical_data:
                data = historical_data[symbol]
                date_data = data[data.index.date == current_date]
                if not date_data.empty:
                    closing_prices[symbol] = date_data.iloc[0]['Close']

        # Calculate portfolio value
        portfolio_value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in closing_prices:
                portfolio_value += shares * closing_prices[symbol]

        # Record portfolio state
        self._record_portfolio_state(current_date, closing_prices)

    def _record_portfolio_state(self, date: datetime.date, prices: Dict[str, float]):
        """Record the current portfolio state"""
        portfolio_value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in prices:
                portfolio_value += shares * prices.get(symbol, 0)

        self.portfolio_history.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy()
        })

    def _calculate_simulation_results(self, historical_data: Dict[str, pd.DataFrame],
                                    trading_dates: List[datetime.date]) -> Dict[str, Any]:
        """Calculate comprehensive simulation results and analytics"""

        if not self.portfolio_history:
            return {'error': 'No portfolio history available'}

        # Convert to DataFrame for analysis
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.set_index('date')

        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1

        # Calculate benchmark returns
        benchmark_data = historical_data.get(self.config.benchmark_symbol)
        if benchmark_data is not None:
            # Filter benchmark data to match trading dates
            trading_date_set = set(trading_dates)
            mask = [d in trading_date_set for d in benchmark_data.index.date]
            benchmark_returns = benchmark_data['Returns'].loc[mask]
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
        else:
            benchmark_cumulative = pd.Series([0] * len(portfolio_df), index=portfolio_df.index)

        # Calculate performance metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.config.initial_capital) - 1
        if total_return > -1:  # Only calculate if not a total loss
            annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        else:
            annualized_return = -1.0  # Total loss

        # Risk metrics
        volatility = portfolio_df['returns'].std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative = (1 + portfolio_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calculate benchmark comparison
        benchmark_total_return = benchmark_cumulative.iloc[-1] if not benchmark_cumulative.empty else 0
        excess_return = total_return - benchmark_total_return

        # Trading statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade['action'] == 'SELL')  # Simplified
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Total commissions paid
        total_commissions = sum(trade['commission'] for trade in self.trades)

        return {
            'simulation_config': {
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'initial_capital': self.config.initial_capital,
                'symbols': self.config.symbols,
                'weights': self.config.weights,
                'rebalance_frequency': self.config.rebalance_frequency
            },
            'performance_metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'benchmark_total_return': benchmark_total_return,
                'excess_return': excess_return
            },
            'trading_statistics': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_commissions': total_commissions,
                'final_portfolio_value': portfolio_df['portfolio_value'].iloc[-1],
                'final_cash': portfolio_df['cash'].iloc[-1]
            },
            'portfolio_history': portfolio_df.reset_index().to_dict('records'),
            'trades': self.trades,
            'benchmark_comparison': {
                'benchmark_symbol': self.config.benchmark_symbol,
                'benchmark_cumulative_returns': benchmark_cumulative.tolist() if not benchmark_cumulative.empty else []
            }
        }

def run_historical_portfolio_simulation(symbols: List[str],
                                       start_date: str,
                                       end_date: str,
                                       initial_capital: float = 100000,
                                       weights: Dict[str, float] = None,
                                       rebalance_frequency: str = 'monthly') -> Dict[str, Any]:
    """
    Convenience function to run a historical portfolio simulation.

    Args:
        symbols: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting portfolio value
        weights: Portfolio weights (equal weight if None)
        rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly', 'quarterly')

    Returns:
        Comprehensive simulation results
    """
    config = SimulationConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        symbols=symbols,
        weights=weights,
        rebalance_frequency=rebalance_frequency
    )

    engine = HistoricalSimulationEngine(config)
    return engine.run_portfolio_simulation()

def run_multi_strategy_comparison(symbols: List[str],
                                 strategies: List[Dict[str, Any]],
                                 start_date: str,
                                 end_date: str,
                                 initial_capital: float = 100000) -> Dict[str, Any]:
    """
    Run multiple strategy simulations for comparison.

    Args:
        symbols: List of ticker symbols
        strategies: List of strategy configurations
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting portfolio value

    Returns:
        Comparison of strategy performance
    """
    results = {}

    for strategy in strategies:
        strategy_name = strategy.get('name', 'unnamed_strategy')
        weights = strategy.get('weights', None)
        rebalance_freq = strategy.get('rebalance_frequency', 'monthly')

        logger.info(f"Running simulation for strategy: {strategy_name}")

        result = run_historical_portfolio_simulation(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            weights=weights,
            rebalance_frequency=rebalance_freq
        )

        results[strategy_name] = result

    return results