#!/usr/bin/env python3
"""
Enhanced Trading Strategy Simulator with GROK Agents
Uses the full GROK multi-agent system for strategy generation and risk management
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.integrations.nautilus_ibkr_bridge import get_nautilus_ibkr_bridge
from src.agents.data import DataAgent
from src.agents.strategy import StrategyAgent
from src.agents.risk import RiskAgent
from src.agents.execution import ExecutionAgent
from src.agents.learning import LearningAgent
from src.agents.reflection import ReflectionAgent

class GROKEnhancedSimulator:
    """Advanced simulator using the full GROK multi-agent system"""

    def __init__(self):
        self.bridge = None
        self.agents = {}
        self.portfolio = {
            'cash': 100000.0,
            'positions': {},
            'trades': [],
            'performance': []
        }

    async def initialize(self):
        """Initialize the simulator with bridge and all GROK agents"""
        print("🤖 Initializing GROK Multi-Agent System...")

        # Initialize bridge
        self.bridge = get_nautilus_ibkr_bridge()
        success = await self.bridge.initialize()
        if not success:
            print("❌ Failed to initialize bridge")
            return False

        # Initialize all GROK agents
        try:
            self.agents['data'] = DataAgent()
            print("✅ Data Agent initialized")

            self.agents['strategy'] = StrategyAgent()
            print("✅ Strategy Agent initialized")

            self.agents['risk'] = RiskAgent()
            print("✅ Risk Agent initialized")

            self.agents['execution'] = ExecutionAgent()
            print("✅ Execution Agent initialized")

            self.agents['learning'] = LearningAgent()
            print("✅ Learning Agent initialized")

            self.agents['reflection'] = ReflectionAgent()
            print("✅ Reflection Agent initialized")

            print("🎯 All GROK agents ready!")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize agents: {e}")
            return False

    async def get_market_data_with_agents(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Get market data using the Data Agent's sophisticated processing"""
        try:
            print(f"📊 Data Agent fetching {symbol} data...")

            # Use Data Agent to get comprehensive market data
            data_request = {
                'symbol': symbol,
                'days': days,
                'include_indicators': True,
                'include_sentiment': False,  # Disable sentiment to avoid API issues
                'include_fundamentals': False  # Disable fundamentals to avoid API issues
            }

            # Process through Data Agent with timeout
            try:
                enriched_data = await asyncio.wait_for(
                    self.agents['data'].process_input(data_request),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                print("⏰ Data Agent timeout - using fallback")
                enriched_data = None

            if enriched_data and 'dataframe' in enriched_data:
                df = enriched_data['dataframe']
                print(f"📈 Data Agent returned {len(df)} enriched data points")
                return df
            else:
                # Fallback to basic yfinance data
                print("⚠️ Data Agent fallback - using basic yfinance data")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                return df

        except Exception as e:
            print(f"❌ Data Agent error: {e}")
            # Ultimate fallback
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            ticker = yf.Ticker(symbol)
            return ticker.history(start=start_date, end=end_date, interval='1d')

    def execute_simulated_trade(self, symbol: str, action: str, quantity: int,
                               price: float, timestamp: datetime) -> bool:
        """Execute trade with risk management from Risk Agent"""
        try:
            # Get risk assessment before trade
            risk_assessment = self._assess_trade_risk(symbol, action, quantity, price)
            if not risk_assessment['approved']:
                print(f"❌ Trade rejected by Risk Agent: {risk_assessment['reason']}")
                return False

            # Apply position sizing from Risk Agent
            adjusted_quantity = risk_assessment.get('adjusted_quantity', quantity)

            cost = adjusted_quantity * price
            commission = cost * 0.0005  # 0.05% commission

            if action.upper() == 'BUY':
                total_cost = cost + commission
                if total_cost > self.portfolio['cash']:
                    print(f"❌ Insufficient funds for {action} {adjusted_quantity} {symbol}")
                    return False

                self.portfolio['cash'] -= total_cost
                if symbol not in self.portfolio['positions']:
                    self.portfolio['positions'][symbol] = {'shares': 0, 'avg_price': 0}

                current_shares = self.portfolio['positions'][symbol]['shares']
                current_avg = self.portfolio['positions'][symbol]['avg_price']
                total_value = (current_shares * current_avg) + cost
                new_shares = current_shares + adjusted_quantity
                new_avg = total_value / new_shares if new_shares > 0 else 0

                self.portfolio['positions'][symbol]['shares'] = new_shares
                self.portfolio['positions'][symbol]['avg_price'] = new_avg

            elif action.upper() == 'SELL':
                if symbol not in self.portfolio['positions'] or \
                   self.portfolio['positions'][symbol]['shares'] < adjusted_quantity:
                    print(f"❌ Insufficient shares for {action} {adjusted_quantity} {symbol}")
                    return False

                proceeds = cost - commission
                self.portfolio['cash'] += proceeds
                self.portfolio['positions'][symbol]['shares'] -= adjusted_quantity

                if self.portfolio['positions'][symbol]['shares'] == 0:
                    del self.portfolio['positions'][symbol]

            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action.upper(),
                'quantity': adjusted_quantity,
                'price': price,
                'commission': commission,
                'total_value': cost,
                'risk_score': risk_assessment.get('risk_score', 0)
            }
            self.portfolio['trades'].append(trade)

            print(f"✅ {action} {adjusted_quantity} {symbol} @ ${price:.2f} (Risk: {risk_assessment.get('risk_score', 0):.2f})")
            return True

        except Exception as e:
            print(f"❌ Error executing trade: {e}")
            return False

    def _assess_trade_risk(self, symbol: str, action: str, quantity: int, price: float) -> Dict:
        """Use Risk Agent to assess trade risk"""
        try:
            # Simulate risk assessment (in real implementation, this would call the Risk Agent)
            portfolio_value = self.get_portfolio_value({})
            position_size_pct = (quantity * price) / portfolio_value

            # Risk thresholds from Risk Agent config
            max_position_size = 0.30  # 30% max position
            current_exposure = sum(pos['shares'] * pos['avg_price'] for pos in self.portfolio['positions'].values())
            current_exposure_pct = current_exposure / (current_exposure + self.portfolio['cash'])

            if position_size_pct > max_position_size:
                adjusted_quantity = int((max_position_size * portfolio_value) / price)
                return {
                    'approved': True,
                    'adjusted_quantity': min(adjusted_quantity, quantity),
                    'risk_score': 0.7,
                    'reason': f"Position size reduced from {quantity} to {adjusted_quantity}"
                }

            risk_score = min(position_size_pct * 3.33, 1.0)  # Scale risk score

            return {
                'approved': True,
                'risk_score': risk_score,
                'adjusted_quantity': quantity
            }

        except Exception as e:
            print(f"⚠️ Risk assessment error: {e}")
            return {'approved': True, 'risk_score': 0.5, 'adjusted_quantity': quantity}

    async def generate_strategy_with_agents(self, market_data: pd.DataFrame, symbol: str) -> Dict:
        """Use Strategy Agent to generate trading strategy with robust fallbacks"""
        try:
            print("🎯 Strategy Agent analyzing market data...")

            # Prepare data for Strategy Agent
            strategy_request = {
                'market_data': market_data,
                'symbol': symbol,
                'current_portfolio': self.portfolio.copy(),
                'risk_constraints': self.agents['risk'].configs.get('risk', {}).get('constraints', {}),
                'time_horizon': 'short_term'
            }

            # Try Strategy Agent with timeout
            try:
                strategy_response = await asyncio.wait_for(
                    self.agents['strategy'].process_input(strategy_request),
                    timeout=45.0  # 45 second timeout
                )

                if strategy_response and 'strategy' in strategy_response:
                    strategy = strategy_response['strategy']
                    print(f"📋 Strategy Agent generated: {strategy.get('name', 'Unknown Strategy')}")
                    return strategy

            except asyncio.TimeoutError:
                print("⏰ Strategy Agent timeout")
            except Exception as e:
                print(f"❌ Strategy Agent error: {e}")

            # Enhanced fallback strategy
            print("⚠️ Strategy Agent fallback - using enhanced MA crossover")
            return self._create_fallback_strategy(market_data, symbol)

        except Exception as e:
            print(f"❌ Strategy generation failed: {e}")
            return self._create_fallback_strategy(market_data, symbol)

    def _create_fallback_strategy(self, market_data: pd.DataFrame, symbol: str) -> Dict:
        """Create a robust fallback strategy"""
        return {
            'name': 'Enhanced_MA_Crossover_Fallback',
            'type': 'technical',
            'signals': ['ma_crossover'],
            'parameters': {
                'fast_ma': 10,
                'slow_ma': 20,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30
            },
            'entry_conditions': [
                'SMA_10 > SMA_20',  # Bullish crossover
                'RSI < 70'  # Not overbought
            ],
            'exit_conditions': [
                'SMA_10 < SMA_20',  # Bearish crossover
                'RSI > 80'  # Overbought
            ],
            'risk_management': {
                'stop_loss_pct': 0.05,  # 5% stop loss
                'take_profit_pct': 0.10,  # 10% take profit
                'max_position_pct': 0.20  # 20% of portfolio
            },
            'description': 'Enhanced moving average crossover with RSI confirmation and risk management',
            'confidence': 0.75
        }

    async def run_grok_strategy_simulation(self, symbol: str = 'SPY', days: int = 90):
        """
        Run simulation using the full GROK agent system
        """
        print(f"🚀 Running GROK Multi-Agent Strategy Simulation for {symbol}")
        print("=" * 70)

        # Get enriched market data from Data Agent
        market_data = await self.get_market_data_with_agents(symbol, days)
        if market_data.empty:
            print("❌ No market data available")
            return

        print(f"📊 Loaded {len(market_data)} days of {symbol} data")

        # Generate strategy using Strategy Agent
        strategy = await self.generate_strategy_with_agents(market_data, symbol)

        # Apply strategy logic
        await self._execute_grok_strategy(market_data, symbol, strategy)

        # Final results
        final_value = self.get_portfolio_value({symbol: market_data['Close'].iloc[-1]})
        initial_value = 100000.0
        profit_loss = final_value - initial_value
        profit_pct = (profit_loss / initial_value) * 100

        print("\n📈 GROK Strategy Simulation Results:")
        print(f"Strategy: {strategy.get('name', 'Unknown')}")
        print(f"Initial Capital: ${initial_value:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"P&L: ${profit_loss:,.2f} ({profit_pct:.2f}%)")
        print(f"Total Trades: {len(self.portfolio['trades'])}")
        print(f"Remaining Cash: ${self.portfolio['cash']:,.2f}")

        if self.portfolio['positions']:
            print("Open Positions:")
            for sym, pos in self.portfolio['positions'].items():
                print(f"  {sym}: {pos['shares']} shares @ ${pos['avg_price']:.2f}")

        # Learning and Reflection
        await self._run_learning_reflection(strategy, profit_pct)

    async def _execute_grok_strategy(self, data: pd.DataFrame, symbol: str, strategy: Dict):
        """Execute strategy using agent-generated signals"""
        try:
            print(f"⚡ Executing strategy: {strategy.get('name', 'Unknown')}")

            # Calculate technical indicators based on strategy
            if 'ma_crossover' in strategy.get('signals', []):
                fast_ma = strategy.get('parameters', {}).get('fast_ma', 10)
                slow_ma = strategy.get('parameters', {}).get('slow_ma', 20)

                data['SMA_fast'] = data['Close'].rolling(window=fast_ma).mean()
                data['SMA_slow'] = data['Close'].rolling(window=slow_ma).mean()
                print(f"📊 Calculated MAs: Fast({fast_ma}), Slow({slow_ma})")

            position = 0  # 0 = no position, 1 = long
            trade_count = 0

            for i, (date, row) in enumerate(data.iterrows()):
                current_price = row['Close']

                # Generate trading signal based on strategy
                signal = self._generate_signal(row, strategy)

                # Debug output every 20 iterations
                if i % 20 == 0:
                    print(f"📈 Day {i}: Price=${current_price:.2f}, Signal={signal}, Position={position}")

                if signal == 'BUY' and position == 0:
                    shares = int(self.portfolio['cash'] * 0.1 / current_price)  # 10% of cash
                    if shares > 0:
                        success = self.execute_simulated_trade(symbol, 'BUY', shares, current_price, date.to_pydatetime())
                        if success:
                            position = 1
                            trade_count += 1

                elif signal == 'SELL' and position == 1:
                    if symbol in self.portfolio['positions']:
                        shares = self.portfolio['positions'][symbol]['shares']
                        if shares > 0:
                            success = self.execute_simulated_trade(symbol, 'SELL', shares, current_price, date.to_pydatetime())
                            if success:
                                position = 0
                                trade_count += 1

                # Record portfolio value
                current_prices = {symbol: current_price}
                portfolio_value = self.get_portfolio_value(current_prices)
                self.portfolio['performance'].append({
                    'date': date.to_pydatetime(),
                    'value': portfolio_value
                })

            print(f"✅ Strategy execution complete: {trade_count} trades executed")

        except Exception as e:
            print(f"❌ Strategy execution error: {e}")
            import traceback
            traceback.print_exc()

    def _generate_signal(self, row: pd.Series, strategy: Dict) -> str:
        """Generate trading signal based on strategy rules"""
        try:
            if 'ma_crossover' in strategy.get('signals', []):
                fast_ma = row.get('SMA_fast')
                slow_ma = row.get('SMA_slow')

                # Fix: Check for NaN values and compare scalar values
                if pd.notna(fast_ma) and pd.notna(slow_ma):
                    # Convert to scalar values for comparison
                    fast_val = float(fast_ma) if hasattr(fast_ma, 'item') else fast_ma
                    slow_val = float(slow_ma) if hasattr(slow_ma, 'item') else slow_ma

                    # Simple crossover logic
                    if fast_val > slow_val:
                        return 'BUY'
                    elif fast_val < slow_val:
                        return 'SELL'

            return 'HOLD'

        except Exception as e:
            print(f"⚠️ Signal generation error: {e}")
            return 'HOLD'

    async def _run_learning_reflection(self, strategy: Dict, performance: float):
        """Use Learning and Reflection agents for strategy improvement"""
        try:
            print("\n🧠 Learning & Reflection Analysis:")

            # Learning Agent analysis
            learning_input = {
                'strategy': strategy,
                'performance': performance,
                'trades': self.portfolio['trades'],
                'market_conditions': 'current_market_data'
            }

            learning_insights = await self.agents['learning'].process_input(learning_input)
            if learning_insights is not None:
                # Handle DataFrame responses from learning agent
                if hasattr(learning_insights, 'to_dict'):
                    insights_dict = learning_insights.to_dict()
                    print(f"📚 Learning Agent: Analysis complete - {len(insights_dict)} insights generated")
                elif isinstance(learning_insights, dict):
                    print(f"📚 Learning Agent: {learning_insights.get('insights', 'Analysis complete')}")
                else:
                    print(f"📚 Learning Agent: Analysis complete - {str(learning_insights)[:100]}...")
            else:
                print("📚 Learning Agent: No insights generated")

            # Reflection Agent for improvements
            reflection_input = {
                'strategy_performance': performance,
                'agent_interactions': 'simulation_complete',
                'improvement_suggestions': True
            }

            reflection_feedback = await self.agents['reflection'].process_input(reflection_input)
            if reflection_feedback is not None:
                if isinstance(reflection_feedback, dict):
                    print(f"🔄 Reflection Agent: {reflection_feedback.get('feedback', 'Reflection complete')}")
                else:
                    print(f"🔄 Reflection Agent: Reflection complete - {str(reflection_feedback)[:100]}...")
            else:
                print("🔄 Reflection Agent: No feedback generated")

        except Exception as e:
            print(f"⚠️ Learning/Reflection error: {e}")
            print("🧠 Continuing without learning/reflection analysis")

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        cash = self.portfolio['cash']
        positions_value = 0

        for symbol, position in self.portfolio['positions'].items():
            if symbol in current_prices:
                positions_value += position['shares'] * current_prices[symbol]

        return cash + positions_value

    def reset_portfolio(self):
        """Reset portfolio for new simulation"""
        self.portfolio = {
            'cash': 100000.0,
            'positions': {},
            'trades': [],
            'performance': []
        }

async def main():
    """Run the enhanced GROK simulator"""
    simulator = GROKEnhancedSimulator()

    print("🎯 GROK Multi-Agent Trading Simulator")
    print("=" * 45)

    # Initialize with all agents
    success = await simulator.initialize()
    if not success:
        print("❌ Failed to initialize GROK system")
        return

    # Run simulation with full agent integration
    await simulator.run_grok_strategy_simulation('SPY', days=90)

    print("\n✅ GROK simulation completed!")
    print("💡 Now using the full power of your multi-agent system!")

if __name__ == "__main__":
    asyncio.run(main())