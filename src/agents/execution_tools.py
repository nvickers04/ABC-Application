# src/agents/execution_tools.py
# Purpose: Paper trading tools for the Execution Agent
# Provides IBKR integration tools for order execution, market validation, and position monitoring
# Uses the IBKR connector for actual trading operations

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import exchange_calendars as ecals
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.tools import microstructure_analysis_tool
from src.integrations.ibkr_connector import get_ibkr_connector
from src.integrations.nautilus_ibkr_bridge import get_nautilus_ibkr_bridge
from src.utils.alert_manager import get_alert_manager

logger = logging.getLogger(__name__)
alert_manager = get_alert_manager()

class IBKRExecuteTool:
    """
    Tool for executing orders via IBKR paper trading using IBKRConnector
    """

    def __init__(self):
        self.connector = get_ibkr_connector()
        self.name = "ibkr_execute_tool"
        self.description = "Execute orders through IBKR paper trading account"

    async def execute(self, symbol: str, quantity: int, action: str = 'BUY',
                     order_type: str = 'MKT', price: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute an order via IBKR

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            action: 'BUY' or 'SELL'
            order_type: 'MKT' or 'LMT'
            price: Limit price for LMT orders

        Returns:
            Dict with execution results
        """
        try:
            logger.info(f"Executing {action} {quantity} {symbol} via NautilusIBKRBridge")

            result = await self.connector.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type=order_type,
                action=action,
                price=price
            )

            if 'error' in result:
                logger.error(f"Order execution failed: {result['error']}")
                return {
                    'success': False,
                    'error': result['error'],
                    'symbol': symbol,
                    'quantity': quantity,
                    'action': action
                }
            else:
                logger.info(f"Order executed successfully: {result}")
                return {
                    'success': True,
                    'order_id': result.get('order_id'),
                    'symbol': symbol,
                    'quantity': quantity,
                    'action': action,
                    'status': result.get('status'),
                    'filled_quantity': result.get('filled_quantity', 0),
                    'avg_fill_price': result.get('avg_fill_price'),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

        except Exception as e:
            logger.error(f"Error in IBKR execute tool: {e}")
            alert_manager.error(
                f"IBKR order execution failed for {symbol}",
                {"symbol": symbol, "quantity": quantity, "action": action, "order_type": order_type, "error": str(e)},
                "execution_agent"
            )
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'quantity': quantity,
                'action': action
            }

class ExchangeCalendarsTool:
    """
    Tool for validating market hours and trading schedules
    """

    def __init__(self):
        self.calendar = ecals.get_calendar('NYSE')
        self.name = "exchange_calendars_tool"
        self.description = "Validate market hours and trading schedules"

    def is_market_open(self) -> Dict[str, Any]:
        """
        Check if the market is currently open

        Returns:
            Dict with market status
        """
        try:
            now = datetime.now(timezone.utc)
            # Convert to pandas Timestamp for exchange_calendars
            now_ts = pd.Timestamp(now)
            # Use proper exchange calendar to check if market is open
            is_open = self.calendar.is_open_at_time(now_ts)

            # Get next market open/close times
            next_open = self.calendar.next_open(now_ts)
            next_close = self.calendar.next_close(now_ts)

            return {
                'market_open': is_open,
                'current_time': now.isoformat(),
                'next_open': next_open.isoformat() if next_open else None,
                'next_close': next_close.isoformat() if next_close else None,
                'timezone': 'UTC'
            }

        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return {
                'market_open': False,
                'error': str(e),
                'current_time': datetime.now(timezone.utc).isoformat()
            }

    def get_trading_schedule(self, date: str = None) -> Dict[str, Any]:
        """
        Get trading schedule for a specific date

        Args:
            date: Date in YYYY-MM-DD format (default: today)

        Returns:
            Dict with schedule information
        """
        try:
            if date:
                target_date = datetime.fromisoformat(date).date()
            else:
                target_date = datetime.now(timezone.utc).date()

            schedule = self.calendar.schedule.loc[target_date]

            if not schedule.empty:
                market_open = schedule.iloc[0]['market_open']
                market_close = schedule.iloc[0]['market_close']

                return {
                    'date': target_date.isoformat(),
                    'market_open': market_open.isoformat(),
                    'market_close': market_close.isoformat(),
                    'is_trading_day': True
                }
            else:
                return {
                    'date': target_date.isoformat(),
                    'is_trading_day': False,
                    'reason': 'Holiday or weekend'
                }

        except Exception as e:
            logger.error(f"Error getting trading schedule: {e}")
            return {
                'error': str(e),
                'date': date or datetime.now(timezone.utc).date().isoformat()
            }

class ScalingPingTool:
    """
    Tool for monitoring positions and triggering scaling decisions
    """

    def __init__(self):
        self.bridge = get_nautilus_ibkr_bridge()
        self.name = "scaling_ping_tool"
        self.description = "Monitor positions and assess scaling opportunities with nautilus enhancements"

    async def ping_position(self, symbol: str) -> Dict[str, Any]:
        """
        Ping a position for scaling assessment

        Args:
            symbol: Stock symbol to monitor

        Returns:
            Dict with position status and scaling recommendations
        """
        try:
            # Get current positions using bridge
            positions = await self.connector.get_positions()

            # Find position for symbol
            position_data = None
            for pos in positions:
                if pos['symbol'] == symbol:
                    position_data = pos
                    break

            if not position_data:
                return {
                    'symbol': symbol,
                    'has_position': False,
                    'recommendation': 'no_position',
                    'message': f'No position found for {symbol}'
                }

            # Get current market data using bridge
            market_data = await self.bridge.get_market_data(symbol)

            # Calculate P&L and scaling recommendations
            current_price = market_data['close'] if market_data else position_data['avg_cost']
            position_size = position_data['position']
            avg_cost = position_data['avg_cost']

            unrealized_pnl = position_data.get('unrealized_pnl', 0)
            realized_pnl = position_data.get('realized_pnl', 0)

            # Simple scaling logic (can be enhanced)
            pnl_pct = unrealized_pnl / (abs(position_size) * avg_cost) if position_size != 0 else 0

            if pnl_pct > 0.10:  # 10% profit
                recommendation = 'scale_out'
                message = f"Profit threshold reached ({pnl_pct:.1%}), consider taking profits"
            elif pnl_pct < -0.05:  # 5% loss
                recommendation = 'scale_out'
                message = f"Stop loss triggered ({pnl_pct:.1%}), consider cutting losses"
            else:
                recommendation = 'hold'
                message = f"Position within normal range ({pnl_pct:.1%})"

            return {
                'symbol': symbol,
                'has_position': True,
                'position_size': position_size,
                'avg_cost': avg_cost,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'pnl_percentage': pnl_pct,
                'recommendation': recommendation,
                'message': message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error in scaling ping for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'recommendation': 'error'
            }

class NautilusPaperSimTool:
    """
    Tool for paper trading simulation using nautilus_trader concepts
    """

    def __init__(self):
        self.name = "nautilus_paper_sim_tool"
        self.description = "Paper trading simulation with realistic slippage and fees"

    def simulate_order(self, symbol: str, quantity: int, action: str = 'BUY',
                      order_type: str = 'MKT', price: Optional[float] = None,
                      base_price: float = 100.0) -> Dict[str, Any]:
        """
        Simulate an order execution with realistic slippage based on microstructure analysis

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            action: 'BUY' or 'SELL'
            order_type: 'MKT' or 'LMT'
            price: Limit price
            base_price: Base price for simulation

        Returns:
            Dict with simulated execution results
        """
        try:
            import numpy as np

            # Get microstructure analysis for sophisticated slippage modeling
            microstructure = microstructure_analysis_tool.invoke({
                "symbol": symbol,
                "analysis_type": "comprehensive"
            })

            # Extract slippage parameters from microstructure analysis
            optimal_slippage = 0.0005  # Default 0.05%
            slippage_multiplier = 1.0

            if "analysis" in microstructure:
                analysis = microstructure["analysis"]

                # Use calculated optimal slippage if available
                if "slippage_model" in analysis:
                    optimal_slippage = analysis["slippage_model"].get("optimal_slippage", 0.0005)
                    slippage_multiplier = analysis["slippage_model"].get("multiplier", 1.0)

                # Adjust for market conditions
                market_condition = analysis.get("market_condition", "neutral")
                if market_condition == "favorable":
                    slippage_multiplier *= 0.8  # Reduce slippage in favorable conditions
                elif market_condition == "challenging":
                    slippage_multiplier *= 1.3  # Increase slippage in challenging conditions

                # Adjust for order flow momentum
                if "order_flow" in analysis:
                    momentum = analysis["order_flow"].get("momentum", "neutral")
                    if (action == "BUY" and momentum == "bullish") or (action == "SELL" and momentum == "bearish"):
                        slippage_multiplier *= 0.9  # Better slippage when trading with momentum
                    elif (action == "BUY" and momentum == "bearish") or (action == "SELL" and momentum == "bullish"):
                        slippage_multiplier *= 1.2  # Worse slippage when trading against momentum

            # Apply final slippage calculation
            base_slippage = optimal_slippage * slippage_multiplier

            # Add some randomness but keep it realistic
            slippage_noise = np.random.uniform(0.5, 1.5)  # 50% to 150% of calculated slippage
            final_slippage = base_slippage * slippage_noise

            # Ensure minimum and maximum bounds
            final_slippage = max(0.0001, min(final_slippage, 0.01))  # 0.01% to 1.0%

            slippage_direction = 1 if action == 'BUY' else -1

            if order_type == 'MKT':
                fill_price = base_price * (1 + final_slippage * slippage_direction)
            elif order_type == 'LMT' and price:
                # For limit orders, slippage is smaller but still present
                limit_slippage = final_slippage * 0.3  # 30% of market slippage
                fill_price = price * (1 + limit_slippage * slippage_direction)
            else:
                fill_price = base_price

            # Calculate fees (IBKR paper trading has no commissions, but simulate small fees)
            commission = 0.0  # Paper trading

            # Simulate partial fills based on market conditions
            fill_probability = 0.98  # Default 98% chance of full fill

            # Reduce fill probability in challenging market conditions
            if "analysis" in microstructure and microstructure["analysis"].get("market_condition") == "challenging":
                fill_probability = 0.85  # 85% chance in challenging conditions

            # Reduce fill probability for large orders
            if quantity > 10000:  # Large order
                fill_probability *= 0.8

            if np.random.random() > fill_probability:
                filled_quantity = int(quantity * np.random.uniform(0.3, 0.9))
            else:
                filled_quantity = quantity

            total_value = filled_quantity * fill_price
            total_cost = total_value + commission

            # Calculate slippage impact
            expected_price = base_price if order_type == 'MKT' else (price or base_price)
            slippage_impact = (fill_price - expected_price) / expected_price * 100

            return {
                'symbol': symbol,
                'action': action,
                'order_type': order_type,
                'requested_quantity': quantity,
                'filled_quantity': filled_quantity,
                'fill_price': round(fill_price, 2),
                'expected_price': round(expected_price, 2),
                'total_value': round(total_value, 2),
                'commission': commission,
                'total_cost': round(total_cost, 2),
                'slippage_pct': round(final_slippage * 100, 3),
                'slippage_impact_pct': round(slippage_impact, 3),
                'microstructure_used': "analysis" in microstructure,
                'market_condition': microstructure.get("analysis", {}).get("market_condition", "unknown"),
                'simulated': True,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error in enhanced paper trading simulation: {e}")
            # Fallback to basic simulation
            import numpy as np
            slippage_pct = np.random.uniform(0.0005, 0.005)
            slippage_direction = 1 if action == 'BUY' else -1

            fill_price = base_price * (1 + slippage_pct * slippage_direction)
            filled_quantity = quantity

            return {
                'symbol': symbol,
                'action': action,
                'order_type': order_type,
                'requested_quantity': quantity,
                'filled_quantity': filled_quantity,
                'fill_price': round(fill_price, 2),
                'total_value': round(filled_quantity * fill_price, 2),
                'commission': 0.0,
                'total_cost': round(filled_quantity * fill_price, 2),
                'slippage_pct': round(slippage_pct * 100, 3),
                'microstructure_used': False,
                'fallback': True,
                'error': str(e),
                'simulated': True,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

# Global tool instances
_ibkr_execute_tool = None
_exchange_calendars_tool = None
_scaling_ping_tool = None
_nautilus_paper_sim_tool = None

def get_ibkr_execute_tool() -> IBKRExecuteTool:
    """Get singleton IBKR execute tool instance"""
    global _ibkr_execute_tool
    if _ibkr_execute_tool is None:
        _ibkr_execute_tool = IBKRExecuteTool()
    return _ibkr_execute_tool

def get_exchange_calendars_tool() -> ExchangeCalendarsTool:
    """Get singleton exchange calendars tool instance"""
    global _exchange_calendars_tool
    if _exchange_calendars_tool is None:
        _exchange_calendars_tool = ExchangeCalendarsTool()
    return _exchange_calendars_tool

def get_scaling_ping_tool() -> ScalingPingTool:
    """Get singleton scaling ping tool instance"""
    global _scaling_ping_tool
    if _scaling_ping_tool is None:
        _scaling_ping_tool = ScalingPingTool()
    return _scaling_ping_tool

def get_nautilus_paper_sim_tool() -> NautilusPaperSimTool:
    """Get singleton nautilus paper sim tool instance"""
    global _nautilus_paper_sim_tool
    if _nautilus_paper_sim_tool is None:
        _nautilus_paper_sim_tool = NautilusPaperSimTool()
    return _nautilus_paper_sim_tool

# Test functions
async def test_ibkr_tools():
    """Test the IBKR tools"""
    print("Testing IBKR Paper Trading Tools...")
    print("=" * 50)

    # Test exchange calendars
    calendars_tool = get_exchange_calendars_tool()
    market_status = calendars_tool.is_market_open()
    print(f"Market Open: {market_status['market_open']}")

    # Test paper trading simulation
    sim_tool = get_nautilus_paper_sim_tool()
    sim_result = sim_tool.simulate_order('SPY', 100, 'BUY', 'MKT', base_price=450.0)
    print(f"Simulated SPY Buy: {sim_result['filled_quantity']} @ ${sim_result['fill_price']}")

    print("✅ Tools initialized successfully")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ibkr_tools())

# ===== ADVANCED MARKET IMPACT AND PYRAMIDING TOOLS =====

class MarketImpactTool:
    """
    Advanced market impact calculation tool using multi-factor analysis
    """

    def __init__(self):
        self.name = "calculate_market_impact"
        self.description = "Calculate expected market impact using sophisticated multi-factor models"

    async def execute(self, symbol: str, trade_size: float, market_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive market impact for a trade.

        Args:
            symbol: Trading symbol
            trade_size: Size of the trade
            market_conditions: Current market conditions

        Returns:
            Dict with impact analysis
        """
        try:
            # Volume-based impact
            volume_impact = await self._calculate_volume_based_impact(symbol, trade_size)

            # Liquidity depth impact
            liquidity_impact = await self._calculate_liquidity_depth_impact(symbol, trade_size)

            # Volatility-adjusted impact
            volatility_impact = await self._calculate_volatility_adjusted_impact(symbol, trade_size)

            # Time-based impact
            time_impact = await self._calculate_time_based_impact(symbol, trade_size)

            # Composite impact calculation
            total_impact_pct = (
                volume_impact['impact_pct'] * 0.4 +
                liquidity_impact['impact_pct'] * 0.3 +
                volatility_impact['impact_pct'] * 0.2 +
                time_impact['impact_pct'] * 0.1
            )

            # Confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(total_impact_pct)

            return {
                'symbol': symbol,
                'trade_size': trade_size,
                'total_impact_pct': total_impact_pct,
                'expected_slippage': total_impact_pct * 0.5,  # Rough estimate
                'volume_based_impact': volume_impact,
                'liquidity_depth_impact': liquidity_impact,
                'volatility_adjusted_impact': volatility_impact,
                'time_based_impact': time_impact,
                'confidence_intervals': confidence_intervals,
                'market_conditions': market_conditions or {},
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating market impact for {symbol}: {e}")
            return {'error': str(e)}

    async def _calculate_volume_based_impact(self, symbol: str, trade_size: float) -> Dict[str, Any]:
        """Calculate volume-based market impact"""
        try:
            # Placeholder - would use real volume data
            avg_daily_volume = 1000000  # Placeholder
            trade_size_pct = trade_size / avg_daily_volume

            # Kyle's lambda model approximation
            if trade_size_pct < 0.001:
                impact_pct = trade_size_pct * 0.1
            elif trade_size_pct < 0.01:
                impact_pct = trade_size_pct * 0.2
            else:
                impact_pct = trade_size_pct * 0.5

            return {
                'impact_pct': impact_pct,
                'trade_size_pct_of_volume': trade_size_pct,
                'model_used': 'kyle_lambda_approximation'
            }

        except Exception as e:
            logger.error(f"Error calculating volume impact: {e}")
            return {'impact_pct': 0.001, 'error': str(e)}

    async def _calculate_liquidity_depth_impact(self, symbol: str, trade_size: float) -> Dict[str, Any]:
        """Calculate liquidity depth impact"""
        try:
            # Placeholder - would use order book data
            bid_ask_spread = 0.05  # 5 cents spread
            order_book_depth = 10000  # Shares available

            depth_ratio = trade_size / order_book_depth
            impact_pct = min(depth_ratio * 0.02, 0.005)  # Cap at 0.5%

            return {
                'impact_pct': impact_pct,
                'bid_ask_spread': bid_ask_spread,
                'order_book_depth': order_book_depth,
                'depth_utilization_pct': depth_ratio
            }

        except Exception as e:
            logger.error(f"Error calculating liquidity impact: {e}")
            return {'impact_pct': 0.001, 'error': str(e)}

    async def _calculate_volatility_adjusted_impact(self, symbol: str, trade_size: float) -> Dict[str, Any]:
        """Calculate volatility-adjusted impact"""
        try:
            # Placeholder - would use volatility data
            current_volatility = 0.25  # 25% annualized
            avg_volatility = 0.20  # 20% average

            vol_multiplier = current_volatility / avg_volatility
            base_impact = 0.002  # Base 0.2% impact
            adjusted_impact = base_impact * vol_multiplier

            return {
                'impact_pct': adjusted_impact,
                'current_volatility': current_volatility,
                'avg_volatility': avg_volatility,
                'volatility_multiplier': vol_multiplier
            }

        except Exception as e:
            logger.error(f"Error calculating volatility impact: {e}")
            return {'impact_pct': 0.002, 'error': str(e)}

    async def _calculate_time_based_impact(self, symbol: str, trade_size: float) -> Dict[str, Any]:
        """Calculate time-based impact"""
        try:
            # Placeholder - time of day impact
            current_hour = datetime.now().hour

            if 9 <= current_hour <= 10:  # Market open
                time_multiplier = 1.5
            elif 15 <= current_hour <= 16:  # Market close
                time_multiplier = 1.3
            else:
                time_multiplier = 1.0

            base_impact = 0.001
            time_impact = base_impact * time_multiplier

            return {
                'impact_pct': time_impact,
                'time_multiplier': time_multiplier,
                'current_hour': current_hour
            }

        except Exception as e:
            logger.error(f"Error calculating time impact: {e}")
            return {'impact_pct': 0.001, 'error': str(e)}

    def _calculate_confidence_intervals(self, base_impact: float) -> Dict[str, Any]:
        """Calculate confidence intervals for impact estimate"""
        try:
            # Simple confidence intervals
            return {
                '95_confidence_lower': base_impact * 0.7,
                '95_confidence_upper': base_impact * 1.3,
                '99_confidence_lower': base_impact * 0.5,
                '99_confidence_upper': base_impact * 1.5
            }

        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {}


class PyramidingOptimizationTool:
    """
    Tool for optimizing pyramiding strategy parameters
    """

    def __init__(self):
        self.name = "optimize_pyramiding_strategy"
        self.description = "Determine optimal pyramiding parameters based on market conditions"

    async def execute(self, symbol: str, trend_strength: float, volatility: float,
                     current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize pyramiding strategy parameters.

        Args:
            symbol: Trading symbol
            trend_strength: Strength of current trend (0-1)
            volatility: Current volatility level
            current_price: Current market price

        Returns:
            Dict with optimized pyramiding parameters
        """
        try:
            # Base parameters
            base_layers = 4
            base_layer_multiplier = 0.5
            base_price_increment = 0.02

            # Adjust for trend strength
            if trend_strength > 0.8:  # Strong trend
                layers = min(base_layers + 1, 5)
                layer_multiplier = base_layer_multiplier * 1.2
                price_increment = base_price_increment * 0.8
            elif trend_strength > 0.6:  # Moderate trend
                layers = base_layers
                layer_multiplier = base_layer_multiplier
                price_increment = base_price_increment
            else:  # Weak trend
                layers = max(base_layers - 1, 2)
                layer_multiplier = base_layer_multiplier * 0.8
                price_increment = base_price_increment * 1.2

            # Adjust for volatility
            vol_adjustment = min(volatility * 2, 1.5)  # Cap at 1.5x

            # Risk parameters
            stop_loss_pct = 0.04 * vol_adjustment  # Base 4%, adjusted for vol
            take_profit_pct = 0.12 / vol_adjustment  # Base 12%, adjusted for vol

            # Timing parameters
            inter_layer_delay = int(60 * vol_adjustment)  # Base 60 min, adjusted
            max_execution_window = 7  # days

            return {
                'symbol': symbol,
                'max_layers': layers,
                'layer_size_multiplier': layer_multiplier,
                'price_increment_pct': price_increment,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'inter_layer_delay_minutes': inter_layer_delay,
                'max_execution_window_days': max_execution_window,
                'volatility_multiplier': vol_adjustment,
                'trend_strength': trend_strength,
                'volatility': volatility,
                'optimization_timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error optimizing pyramiding strategy for {symbol}: {e}")
            return {'error': str(e)}


class PyramidingExecutionTool:
    """
    Tool for executing pyramiding layers
    """

    def __init__(self):
        self.name = "execute_pyramiding_layer"
        self.description = "Execute a specific layer in the pyramiding strategy"

    async def execute(self, symbol: str, layer_number: int, position_size: float,
                     trigger_price: float, execution_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a pyramiding layer.

        Args:
            symbol: Trading symbol
            layer_number: Which layer to execute
            position_size: Size of the position to add
            trigger_price: Price that triggered this layer
            execution_params: Execution parameters

        Returns:
            Dict with execution results
        """
        try:
            # Get execution tool
            execute_tool = get_ibkr_execute_tool()

            # Execute the order
            execution_result = await execute_tool.execute(
                symbol=symbol,
                quantity=int(position_size),
                action='BUY',  # Assuming long positions for pyramiding
                order_type=execution_params.get('order_type', 'MKT') if execution_params else 'MKT',
                price=execution_params.get('price') if execution_params else None
            )

            # Add layer-specific metadata
            execution_result.update({
                'layer_number': layer_number,
                'pyramiding_layer': True,
                'trigger_price': trigger_price,
                'execution_params': execution_params or {},
                'layer_execution_timestamp': datetime.now(timezone.utc).isoformat()
            })

            logger.info(f"Pyramiding layer {layer_number} executed for {symbol}: {execution_result}")

            return execution_result

        except Exception as e:
            logger.error(f"Error executing pyramiding layer {layer_number} for {symbol}: {e}")
            return {'error': str(e), 'layer_number': layer_number}


class TrailingStopManagementTool:
    """
    Tool for managing trailing stops on pyramiding positions
    """

    def __init__(self):
        self.name = "manage_trailing_stops"
        self.description = "Dynamically adjust trailing stops for pyramiding positions"

    async def execute(self, symbol: str, current_price: float, average_entry: float,
                     trailing_pct: float = 0.03) -> Dict[str, Any]:
        """
        Manage trailing stops for a position.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            average_entry: Average entry price
            trailing_pct: Trailing percentage

        Returns:
            Dict with trailing stop updates
        """
        try:
            # Calculate current profit/loss
            pnl_pct = (current_price - average_entry) / average_entry

            # Calculate trailing stop price
            trailing_stop_price = current_price * (1 - trailing_pct)

            # Check if position should be stopped out
            should_stop_out = current_price <= trailing_stop_price

            return {
                'symbol': symbol,
                'current_price': current_price,
                'average_entry': average_entry,
                'pnl_pct': pnl_pct,
                'trailing_stop_price': trailing_stop_price,
                'trailing_pct': trailing_pct,
                'should_stop_out': should_stop_out,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error managing trailing stops for {symbol}: {e}")
            return {'error': str(e)}


# ===== TOOL FACTORY FUNCTIONS =====

def get_market_impact_tool() -> MarketImpactTool:
    """Factory function for market impact tool"""
    return MarketImpactTool()

def get_pyramiding_optimization_tool() -> PyramidingOptimizationTool:
    """Factory function for pyramiding optimization tool"""
    return PyramidingOptimizationTool()

def get_pyramiding_execution_tool() -> PyramidingExecutionTool:
    """Factory function for pyramiding execution tool"""
    return PyramidingExecutionTool()

def get_trailing_stop_tool() -> TrailingStopManagementTool:
    """Factory function for trailing stop management tool"""
    return TrailingStopManagementTool()