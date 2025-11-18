#!/usr/bin/env python3
"""
Trading Strategy Simulation Framework
Test strategies with historical data before live deployment
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.nautilus_ibkr_bridge import get_nautilus_ibkr_bridge

class TradingSimulator:
    """Simulates trading strategies using historical market data"""

    def __init__(self):
        self.bridge = None
        self.portfolio = {
            'cash': 100000.0,  # Starting capital
            'positions': {},    # Symbol -> {'shares': int, 'avg_price': float}
            'trades': [],       # List of trade records
            'performance': []   # Daily portfolio values
        }

    async def initialize(self):
        """Initialize the simulator with bridge"""
        self.bridge = get_nautilus_ibkr_bridge()
        success = await self.bridge.initialize()
        return success

    async def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical market data for simulation"""
        try:
            # Use yfinance directly for historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')

            return df
        except Exception as e:
            print(f"❌ Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()

    def execute_simulated_trade(self, symbol: str, action: str, quantity: int,
                               price: float, timestamp: datetime) -> bool:
        """
        Execute a simulated trade
        action: 'BUY' or 'SELL'
        """
        try:
            cost = quantity * price
            commission = cost * 0.0005  # 0.05% commission

            if action.upper() == 'BUY':
                total_cost = cost + commission
                if total_cost > self.portfolio['cash']:
                    print(f"❌ Insufficient funds for {action} {quantity} {symbol}")
                    return False

                # Update cash
                self.portfolio['cash'] -= total_cost

                # Update position
                if symbol not in self.portfolio['positions']:
                    self.portfolio['positions'][symbol] = {'shares': 0, 'avg_price': 0}

                current_shares = self.portfolio['positions'][symbol]['shares']
                current_avg = self.portfolio['positions'][symbol]['avg_price']

                # Calculate new average price
                total_value = (current_shares * current_avg) + cost
                new_shares = current_shares + quantity
                new_avg = total_value / new_shares if new_shares > 0 else 0

                self.portfolio['positions'][symbol]['shares'] = new_shares
                self.portfolio['positions'][symbol]['avg_price'] = new_avg

            elif action.upper() == 'SELL':
                if symbol not in self.portfolio['positions'] or \
                   self.portfolio['positions'][symbol]['shares'] < quantity:
                    print(f"❌ Insufficient shares for {action} {quantity} {symbol}")
                    return False

                # Update cash
                proceeds = cost - commission
                self.portfolio['cash'] += proceeds

                # Update position
                self.portfolio['positions'][symbol]['shares'] -= quantity

                # Remove position if zero
                if self.portfolio['positions'][symbol]['shares'] == 0:
                    del self.portfolio['positions'][symbol]

            # Record trade
            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action.upper(),
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'total_value': cost
            }
            self.portfolio['trades'].append(trade)

            print(f"✅ {action} {quantity} {symbol} @ ${price:.2f} (Commission: ${commission:.2f})")
            return True

        except Exception as e:
            print(f"❌ Error executing trade: {e}")
            return False

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        cash = self.portfolio['cash']
        positions_value = 0

        for symbol, position in self.portfolio['positions'].items():
            if symbol in current_prices:
                positions_value += position['shares'] * current_prices[symbol]

        return cash + positions_value

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        return {
            'cash': self.portfolio['cash'],
            'positions': self.portfolio['positions'].copy(),
            'total_trades': len(self.portfolio['trades']),
            'total_value': self.get_portfolio_value({})  # Will be updated with current prices
        }

    async def run_simple_strategy(self, symbol: str = 'SPY', days: int = 30):
        """
        Example strategy: Buy on dips, sell on rallies
        This is just a demo - replace with your actual strategy
        """
        print(f"🚀 Running Simple Strategy Simulation for {symbol}")
        print("=" * 50)

        # Get historical data
        data = await self.get_historical_data(symbol, days)
        if data.empty:
            print("❌ No historical data available")
            return

        print(f"📊 Loaded {len(data)} days of {symbol} data")

        # Simple moving average crossover strategy
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()

        position = 0  # 0 = no position, 1 = long
        initial_value = self.portfolio['cash']

        for date, row in data.iterrows():
            current_price = row['Close']

            # Strategy logic
            if pd.notna(row['SMA_10']) and pd.notna(row['SMA_20']):
                if row['SMA_10'] > row['SMA_20'] and position == 0:
                    # Buy signal
                    shares = int(self.portfolio['cash'] * 0.1 / current_price)  # 10% of cash
                    if shares > 0:
                        self.execute_simulated_trade(symbol, 'BUY', shares, current_price, date.to_pydatetime())
                        position = 1

                elif row['SMA_10'] < row['SMA_20'] and position == 1:
                    # Sell signal
                    if symbol in self.portfolio['positions']:
                        shares = self.portfolio['positions'][symbol]['shares']
                        if shares > 0:
                            self.execute_simulated_trade(symbol, 'SELL', shares, current_price, date.to_pydatetime())
                            position = 0

            # Record daily portfolio value
            current_prices = {symbol: current_price}
            portfolio_value = self.get_portfolio_value(current_prices)
            self.portfolio['performance'].append({
                'date': date.to_pydatetime(),
                'value': portfolio_value
            })

        # Final results
        final_value = self.get_portfolio_value({symbol: data['Close'].iloc[-1]})
        profit_loss = final_value - initial_value
        profit_pct = (profit_loss / initial_value) * 100

        print("\n📈 Simulation Results:")
        print(f"Initial Capital: ${initial_value:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"P&L: ${profit_loss:,.2f} ({profit_pct:.2f}%)")
        print(f"Total Trades: {len(self.portfolio['trades'])}")
        print(f"Remaining Cash: ${self.portfolio['cash']:,.2f}")

        if self.portfolio['positions']:
            print("Open Positions:")
            for sym, pos in self.portfolio['positions'].items():
                print(f"  {sym}: {pos['shares']} shares @ ${pos['avg_price']:.2f}")

    def reset_portfolio(self):
        """Reset portfolio for new simulation iteration"""
        self.portfolio = {
            'cash': 100000.0,
            'positions': {},
            'trades': [],
            'performance': []
        }

    async def run_multiple_iterations(self, symbol: str = 'SPY', iterations: int = 250,
                                    min_days: int = 60, max_days: int = 120):
        """
        Run multiple simulation iterations with random time periods
        """
        print(f"🎯 Running {iterations} Strategy Iterations for {symbol}")
        print("=" * 60)

        # Get extended historical data for random sampling
        extended_data = await self.get_historical_data(symbol, days=500)  # 500 days of data
        if extended_data.empty or len(extended_data) < max_days:
            print("❌ Insufficient historical data available")
            return

        print(f"📊 Loaded {len(extended_data)} days of historical {symbol} data")

        # Storage for iteration results
        results = []

        for i in range(iterations):
            if (i + 1) % 50 == 0:
                print(f"📈 Completed {i + 1}/{iterations} iterations...")

            # Reset portfolio for each iteration
            self.reset_portfolio()

            # Randomly select a time period
            import random
            period_length = random.randint(min_days, max_days)
            max_start = len(extended_data) - period_length
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + period_length

            # Extract data for this iteration
            iteration_data = extended_data.iloc[start_idx:end_idx].copy()

            # Run strategy on this data subset
            iteration_result = await self._run_strategy_iteration(symbol, iteration_data, i + 1)
            results.append(iteration_result)

        # Analyze results
        self._analyze_iteration_results(results, symbol)

    async def _run_strategy_iteration(self, symbol: str, data: pd.DataFrame, iteration_num: int) -> Dict:
        """Run a single strategy iteration and return results"""
        initial_value = self.portfolio['cash']

        # Calculate moving averages
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()

        position = 0  # 0 = no position, 1 = long

        for date, row in data.iterrows():
            current_price = row['Close']

            # Strategy logic
            if pd.notna(row['SMA_10']) and pd.notna(row['SMA_20']):
                if row['SMA_10'] > row['SMA_20'] and position == 0:
                    # Buy signal
                    shares = int(self.portfolio['cash'] * 0.1 / current_price)  # 10% of cash
                    if shares > 0:
                        self.execute_simulated_trade(symbol, 'BUY', shares, current_price, date.to_pydatetime())
                        position = 1

                elif row['SMA_10'] < row['SMA_20'] and position == 1:
                    # Sell signal
                    if symbol in self.portfolio['positions']:
                        shares = self.portfolio['positions'][symbol]['shares']
                        if shares > 0:
                            self.execute_simulated_trade(symbol, 'SELL', shares, current_price, date.to_pydatetime())
                            position = 0

            # Record daily portfolio value
            current_prices = {symbol: current_price}
            portfolio_value = self.get_portfolio_value(current_prices)
            self.portfolio['performance'].append({
                'date': date.to_pydatetime(),
                'value': portfolio_value
            })

        # Calculate final results
        final_value = self.get_portfolio_value({symbol: data['Close'].iloc[-1]})
        profit_loss = final_value - initial_value
        profit_pct = (profit_loss / initial_value) * 100

        # Calculate drawdown
        peak_value = initial_value
        max_drawdown = 0
        for perf in self.portfolio['performance']:
            peak_value = max(peak_value, perf['value'])
            drawdown = (peak_value - perf['value']) / peak_value * 100
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'iteration': iteration_num,
            'initial_value': initial_value,
            'final_value': final_value,
            'profit_loss': profit_loss,
            'profit_pct': profit_pct,
            'total_trades': len(self.portfolio['trades']),
            'max_drawdown': max_drawdown,
            'win': profit_pct > 0,
            'period_days': len(data)
        }

    def _analyze_iteration_results(self, results: List[Dict], symbol: str):
        """Analyze and display comprehensive statistics from all iterations"""
        import statistics

        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE STRATEGY ANALYSIS")
        print("=" * 80)

        # Extract key metrics
        profits = [r['profit_pct'] for r in results]
        wins = [r for r in results if r['win']]
        losses = [r for r in results if not r['win']]

        # Basic statistics
        print("\n🎯 OVERALL PERFORMANCE:")
        print(f"Total Iterations: {len(results)}")
        print(f"Win Rate: {len(wins)}/{len(results)} ({len(wins)/len(results)*100:.1f}%)")
        print(f"Average Return: {statistics.mean(profits):.2f}%")
        print(f"Median Return: {statistics.median(profits):.2f}%")
        print(f"Best Return: {max(profits):.2f}%")
        print(f"Worst Return: {min(profits):.2f}%")
        print(f"Standard Deviation: {statistics.stdev(profits):.2f}%")

        # Profit/Loss Analysis
        print("\n💰 PROFIT/LOSS BREAKDOWN:")
        if wins:
            print(f"Average Win: {statistics.mean([w['profit_pct'] for w in wins]):.2f}%")
            print(f"Largest Win: {max([w['profit_pct'] for w in wins]):.2f}%")
        if losses:
            print(f"Average Loss: {statistics.mean([l['profit_pct'] for l in losses]):.2f}%")
            print(f"Largest Loss: {min([l['profit_pct'] for l in losses]):.2f}%")

        # Risk Metrics
        drawdowns = [r['max_drawdown'] for r in results]
        print("\n⚠️  RISK METRICS:")
        print(f"Average Max Drawdown: {statistics.mean(drawdowns):.2f}%")
        print(f"Worst Drawdown: {max(drawdowns):.2f}%")

        # Sharpe-like ratio (assuming 0% risk-free rate)
        if statistics.stdev(profits) > 0:
            sharpe_ratio = statistics.mean(profits) / statistics.stdev(profits)
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Trade frequency
        trades_per_iteration = [r['total_trades'] for r in results]
        print("\n📈 TRADING ACTIVITY:")
        print(f"Average Trades per Iteration: {statistics.mean(trades_per_iteration):.1f}")
        print(f"Total Trades Across All Iterations: {sum(trades_per_iteration)}")

        # Distribution analysis
        print("\n📊 RETURN DISTRIBUTION:")
        # Create histogram-like output
        bins = [-20, -10, -5, 0, 5, 10, 20, 50]
        print("Return Range | Frequency")
        print("-" * 25)
        for i in range(len(bins) - 1):
            count = sum(1 for p in profits if bins[i] <= p < bins[i + 1])
            range_str = f"{bins[i]}% to {bins[i+1]}%"
            print(f"{range_str:<15} | {count}")

        # Final assessment
        avg_return = statistics.mean(profits)
        win_rate = len(wins) / len(results)
        avg_drawdown = statistics.mean(drawdowns)

        print("\n🎖️  STRATEGY ASSESSMENT:")
        if avg_return > 2 and win_rate > 0.5 and avg_drawdown < 15:
            print("🟢 STRONG: Good returns, solid win rate, manageable drawdown")
        elif avg_return > 0 and win_rate > 0.4:
            print("🟡 MODERATE: Positive returns but needs refinement")
        else:
            print("🔴 WEAK: Strategy needs significant improvement")

        print("\n💡 RECOMMENDATIONS:")
        if win_rate < 0.4:
            print("- Consider adjusting entry/exit signals")
        if avg_drawdown > 20:
            print("- High drawdown: Consider position sizing or stop losses")
        if statistics.stdev(profits) > abs(avg_return) * 2:
            print("- High volatility: Strategy may need stabilization")
        if avg_return > 5 and win_rate > 0.6:
            print("- Promising strategy! Consider live testing with small position sizes")

async def main():
    """Run the trading simulation with multiple iterations"""
    simulator = TradingSimulator()

    print("🎯 Advanced Trading Strategy Simulator")
    print("=" * 45)

    # Initialize
    success = await simulator.initialize()
    if not success:
        print("❌ Failed to initialize simulator")
        return

    # Run multiple iterations for robust analysis
    await simulator.run_multiple_iterations('SPY', iterations=250, min_days=60, max_days=120)

    print("\n✅ Multi-iteration analysis completed!")
    print("💡 Use these results to refine your trading strategy!")

if __name__ == "__main__":
    asyncio.run(main())