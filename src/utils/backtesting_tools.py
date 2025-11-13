#!/usr/bin/env python3
"""
Backtesting and performance analysis tools.
Provides tools for strategy backtesting and portfolio analysis.
"""

import os
import logging
from typing import Dict, Any, Optional, List
import pandas as pd

from .validation import circuit_breaker, DataValidator

logger = logging.getLogger(__name__)


@circuit_breaker("pyfolio_analysis")
def pyfolio_metrics_tool(data: str, benchmark_symbol: str = "SPY") -> Dict[str, Any]:
    """
    Calculate portfolio performance metrics using pyfolio.
    Args:
        data: Portfolio returns data as CSV string
        benchmark_symbol: Benchmark symbol for comparison
    Returns:
        dict: Performance metrics
    """
    try:
        import io

        # Parse portfolio data
        df = pd.read_csv(io.StringIO(data))

        if df.empty or 'returns' not in df.columns:
            return {"error": "Invalid portfolio data format. Expected CSV with 'returns' column."}

        # Ensure we have a datetime index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Assume daily returns if no date column
            df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')

        returns = df['returns']

        # Calculate basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = returns.mean() * 252  # Assuming daily returns
        volatility = returns.std() * (252 ** 0.5)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = cumulative / running_max - 1
        max_drawdown = drawdown.min()

        # Benchmark comparison (simplified)
        try:
            import yfinance as yf
            benchmark_data = yf.Ticker(benchmark_symbol).history(period="2y")
            if not benchmark_data.empty:
                benchmark_returns = benchmark_data['Close'].pct_change().dropna()
                benchmark_total_return = (1 + benchmark_returns).prod() - 1
                alpha = total_return - benchmark_total_return
            else:
                alpha = None
        except Exception:
            alpha = None

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "alpha_vs_benchmark": alpha,
            "benchmark_symbol": benchmark_symbol,
            "data_points": len(returns),
            "source": "pyfolio_metrics"
        }

    except Exception as e:
        return {"error": f"Pyfolio metrics calculation failed: {str(e)}"}


@circuit_breaker("zipline_backtest")
def zipline_backtest_tool(strategy_code: str, start_date: str, end_date: str, capital: float = 100000, symbol: str = "SPY") -> Dict[str, Any]:
    """
    Run a backtest using Zipline (if available) or simulate basic backtesting.
    Args:
        strategy_code: Trading strategy code
        start_date: Backtest start date
        end_date: Backtest end date
        capital: Starting capital
        symbol: Primary symbol to trade
    Returns:
        dict: Backtest results
    """
    try:
        # Validate inputs
        strategy_code = DataValidator.sanitize_text_input(strategy_code)
        if not strategy_code:
            return {"error": "No strategy code provided"}

        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        if start >= end:
            return {"error": "Start date must be before end date"}

        # For now, simulate a basic backtest since Zipline setup is complex
        # In a real implementation, this would execute the strategy_code

        # Get historical data
        try:
            import yfinance as yf
            data = yf.Ticker(symbol).history(start=start, end=end)

            if data.empty:
                return {"error": f"No historical data found for {symbol}"}

            # Simulate basic buy-and-hold strategy
            initial_price = data['Close'].iloc[0]
            final_price = data['Close'].iloc[-1]
            total_return = (final_price / initial_price - 1) * 100

            # Calculate some basic metrics
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)  # Annualized
            max_drawdown = ((data['Close'] / data['Close'].cummax()) - 1).min()

            return {
                "strategy": "buy_and_hold_simulation",
                "symbol": symbol,
                "start_date": str(start.date()),
                "end_date": str(end.date()),
                "initial_capital": capital,
                "final_value": capital * (1 + total_return / 100),
                "total_return_percent": total_return,
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "data_points": len(data),
                "note": "This is a simplified simulation. Full Zipline integration would be needed for complex strategies.",
                "source": "zipline_backtest_simulation"
            }

        except Exception as e:
            return {"error": f"Data fetch failed: {str(e)}"}

    except Exception as e:
        return {"error": f"Zipline backtest failed: {str(e)}"}


def backtrader_strategy_tool(strategy_config: Dict[str, Any], data: str) -> Dict[str, Any]:
    """
    Run a strategy using Backtrader framework.
    Args:
        strategy_config: Strategy configuration
        data: Historical data as CSV
    Returns:
        dict: Backtest results
    """
    try:
        import io
        import backtrader as bt

        # Parse data
        df = pd.read_csv(io.StringIO(data))

        if df.empty:
            return {"error": "No data provided"}

        # Convert to Backtrader format
        df['datetime'] = pd.to_datetime(df.get('date', df.index))
        df.set_index('datetime', inplace=True)

        # Create data feed
        data_feed = bt.feeds.PandasData(dataname=df)

        # Create cerebro
        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        # Run backtest (simplified - no custom strategy)
        results = cerebro.run()

        if results:
            result = results[0]

            return {
                "sharpe_ratio": result.analyzers.sharpe.get_analysis().get('sharperatio', 0),
                "max_drawdown": result.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0),
                "total_return": result.analyzers.returns.get_analysis().get('rtot', 0),
                "source": "backtrader"
            }

        return {"error": "Backtest execution failed"}

    except Exception as e:
        return {"error": f"Backtrader strategy failed: {str(e)}"}


def risk_analytics_tool(portfolio_data: str, risk_model: str = "historical") -> Dict[str, Any]:
    """
    Perform risk analytics on portfolio data.
    Args:
        portfolio_data: Portfolio data as CSV
        risk_model: Risk model to use
    Returns:
        dict: Risk analytics results
    """
    try:
        import io
        from scipy import stats

        df = pd.read_csv(io.StringIO(portfolio_data))

        if df.empty or 'returns' not in df.columns:
            return {"error": "Invalid portfolio data"}

        returns = df['returns'].dropna()

        # Basic risk metrics
        mean_return = returns.mean()
        volatility = returns.std()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Value at Risk (VaR)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)

        # Expected Shortfall (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = cumulative / running_max - 1
        max_drawdown = drawdown.min()

        return {
            "mean_return": mean_return,
            "volatility": volatility,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "max_drawdown": max_drawdown,
            "risk_model": risk_model,
            "data_points": len(returns),
            "source": "risk_analytics"
        }

    except Exception as e:
        return {"error": f"Risk analytics failed: {str(e)}"}


def tf_quant_monte_carlo_tool(returns: List[float], simulations: int = 1000, periods: int = 252) -> Dict[str, Any]:
    """
    Perform Monte Carlo simulation using TensorFlow for quantitative projections.
    
    Args:
        returns: Historical returns data
        simulations: Number of simulation paths
        periods: Number of periods to simulate
        
    Returns:
        Dict with simulation results (mean, std, var95)
    """
    try:
        import tensorflow as tf
        import numpy as np
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        sims = tf.random.normal([simulations, periods], mean, std)
        paths = tf.cumprod(1 + sims, axis=1)
        
        final_values = paths[:, -1]
        mean_final = tf.reduce_mean(final_values).numpy()
        std_final = tf.math.reduce_std(final_values).numpy()
        var_95 = np.percentile(final_values.numpy(), 5)
        
        return {
            "mean_final": mean_final,
            "std_final": std_final,
            "var_95": var_95,
            "simulations": simulations
        }
    except Exception as e:
        return {"error": str(e)}

# end of file