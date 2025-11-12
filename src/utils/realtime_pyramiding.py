# src/utils/realtime_pyramiding.py
# Purpose: Implements real-time pyramiding triggers for live position monitoring and automatic scaling.
# Provides live market monitoring with automatic pyramiding execution based on predefined triggers.
# Structural Reasoning: Extends the static pyramiding engine with real-time capabilities for live trading.
# For legacy wealth: Enables automated position scaling during live market hours for optimal capital deployment.

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import numpy as np

from .pyramiding import PyramidingEngine

logger = logging.getLogger(__name__)

@dataclass
class PositionState:
    """Represents the current state of a position."""
    symbol: str
    entry_price: float
    current_price: float
    quantity: int
    entry_time: datetime
    last_update: datetime
    unrealized_pnl: float
    realized_pnl: float = 0.0
    pyramiding_tiers_executed: int = 0
    last_tier_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_levels: List[float] = None

    def __post_init__(self):
        if self.take_profit_levels is None:
            self.take_profit_levels = []

@dataclass
class PyramidingTrigger:
    """Represents a pyramiding trigger condition."""
    trigger_type: str  # 'price_target', 'pnl_threshold', 'time_based', 'volatility_drop'
    threshold_value: float
    action: str  # 'add_position', 'take_profit', 'stop_loss', 'scale_out'
    quantity_multiplier: float = 1.0
    cooldown_minutes: int = 5

class RealTimePyramidingMonitor:
    """
    Real-time pyramiding monitor for live position management.
    Integrated with execution agent for automated order execution.
    """

    def __init__(self, pyramiding_engine: PyramidingEngine, execution_agent=None, a2a_protocol=None):
        self.pyramiding_engine = pyramiding_engine
        self.execution_agent = execution_agent
        self.a2a_protocol = a2a_protocol
        self.positions: Dict[str, PositionState] = {}
        self.active_triggers: Dict[str, List[PyramidingTrigger]] = {}
        self.monitoring_active = False
        self.market_hours = {
            'start': time(9, 30),  # 9:30 AM ET
            'end': time(16, 0)     # 4:00 PM ET
        }
        self.last_trigger_times: Dict[str, datetime] = {}

    async def start_monitoring(self, market_data_callback: Callable):
        """
        Start real-time monitoring of positions.

        Args:
            market_data_callback: Function that provides live market data
        """
        self.monitoring_active = True
        logger.info("Real-time pyramiding monitoring started")

        while self.monitoring_active:
            try:
                # Get live market data
                market_data = await market_data_callback()

                # Update all positions with latest prices
                await self._update_positions(market_data)

                # Check all pyramiding triggers
                await self._check_triggers()

                # Brief pause to prevent excessive CPU usage
                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in real-time monitoring: {e}")
                await asyncio.sleep(5)  # Wait longer on errors

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        logger.info("Real-time pyramiding monitoring stopped")

    def add_position(self, symbol: str, entry_price: float, quantity: int,
                    pyramiding_plan: Dict[str, Any]):
        """
        Add a new position to monitor with pyramiding plan.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Initial quantity
            pyramiding_plan: Pyramiding plan from PyramidingEngine
        """
        position = PositionState(
            symbol=symbol,
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            last_update=datetime.now(),
            unrealized_pnl=0.0,
            last_tier_price=entry_price
        )

        self.positions[symbol] = position
        self._setup_triggers(symbol, pyramiding_plan)

        logger.info(f"Added position {symbol} to real-time monitoring with {len(self.active_triggers.get(symbol, []))} triggers")

    def remove_position(self, symbol: str):
        """Remove a position from monitoring."""
        if symbol in self.positions:
            del self.positions[symbol]
        if symbol in self.active_triggers:
            del self.active_triggers[symbol]
        if symbol in self.last_trigger_times:
            del self.last_trigger_times[symbol]
        logger.info(f"Removed position {symbol} from monitoring")

    def _setup_triggers(self, symbol: str, pyramiding_plan: Dict[str, Any]):
        """
        Set up pyramiding triggers based on the pyramiding plan.
        """
        triggers = []

        # Price target triggers for pyramiding
        price_triggers = pyramiding_plan.get('price_triggers', [])
        scaling_factors = pyramiding_plan.get('scaling_factors', [])

        for i, (trigger_price, scale_factor) in enumerate(zip(price_triggers, scaling_factors[1:])):  # Skip first (base position)
            trigger = PyramidingTrigger(
                trigger_type='price_target',
                threshold_value=trigger_price,
                action='add_position',
                quantity_multiplier=scale_factor,
                cooldown_minutes=5
            )
            triggers.append(trigger)

        # P&L threshold triggers
        pnl_thresholds = [0.05, 0.10, 0.15, 0.25]  # 5%, 10%, 15%, 25% profit levels
        for threshold in pnl_thresholds:
            trigger = PyramidingTrigger(
                trigger_type='pnl_threshold',
                threshold_value=threshold,
                action='add_position',
                quantity_multiplier=1.2,  # 20% additional position
                cooldown_minutes=10
            )
            triggers.append(trigger)

        # Stop loss triggers
        stops = pyramiding_plan.get('stops', {})
        if 'initial_stop' in stops:
            trigger = PyramidingTrigger(
                trigger_type='stop_loss',
                threshold_value=stops['initial_stop'],
                action='stop_loss',
                cooldown_minutes=0  # Immediate action
            )
            triggers.append(trigger)

        # Take profit triggers
        take_profit_levels = self.pyramiding_engine.calculate_take_profit_levels(
            self.positions[symbol].entry_price,
            self.positions[symbol].current_price,
            pyramiding_plan.get('tiers', 3)
        )

        for i, tp_level in enumerate(take_profit_levels):
            scale_out_pct = 0.25 * (i + 1)  # Scale out 25%, 50%, 75% at each level
            trigger = PyramidingTrigger(
                trigger_type='take_profit',
                threshold_value=tp_level,
                action='scale_out',
                quantity_multiplier=scale_out_pct,
                cooldown_minutes=15
            )
            triggers.append(trigger)

        self.active_triggers[symbol] = triggers

    async def _update_positions(self, market_data: Dict[str, Any]):
        """
        Update all positions with latest market data.
        """
        for symbol, position in self.positions.items():
            if symbol in market_data:
                price_data = market_data[symbol]
                current_price = price_data.get('price', position.current_price)

                # Update position state
                position.current_price = current_price
                position.last_update = datetime.now()

                # Calculate unrealized P&L
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity

    async def _check_triggers(self):
        """
        Check all active triggers and execute actions if conditions are met.
        """
        for symbol, position in self.positions.items():
            if symbol not in self.active_triggers:
                continue

            triggers = self.active_triggers[symbol]

            for trigger in triggers:
                if await self._evaluate_trigger(trigger, position):
                    await self._execute_trigger_action(trigger, position)
                    break  # Only execute one trigger per check cycle

    async def _evaluate_trigger(self, trigger: PyramidingTrigger, position: PositionState) -> bool:
        """
        Evaluate if a trigger condition is met.
        """
        # Check cooldown
        if trigger.trigger_type in self.last_trigger_times:
            last_trigger = self.last_trigger_times[trigger.trigger_type]
            cooldown_end = last_trigger + timedelta(minutes=trigger.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return False

        # Check market hours (only trade during market hours)
        now = datetime.now().time()
        if not (self.market_hours['start'] <= now <= self.market_hours['end']):
            return False

        # Evaluate trigger conditions
        if trigger.trigger_type == 'price_target':
            return position.current_price >= trigger.threshold_value

        elif trigger.trigger_type == 'pnl_threshold':
            pnl_pct = position.unrealized_pnl / (position.entry_price * position.quantity)
            return pnl_pct >= trigger.threshold_value

        elif trigger.trigger_type == 'stop_loss':
            return position.current_price <= trigger.threshold_value

        elif trigger.trigger_type == 'take_profit':
            return position.current_price >= trigger.threshold_value

        elif trigger.trigger_type == 'time_based':
            # Time-based triggers (e.g., add position after holding for X minutes)
            holding_time = (datetime.now() - position.entry_time).total_seconds() / 60
            return holding_time >= trigger.threshold_value

        elif trigger.trigger_type == 'volatility_drop':
            # Trigger when volatility drops below threshold
            # This would require volatility calculation from market data
            return False  # Placeholder

        return False

    async def _execute_trigger_action(self, trigger: PyramidingTrigger, position: PositionState):
        """
        Execute the trigger action via execution agent.
        """
        logger.info(f"Executing trigger action: {trigger.action} for {position.symbol}")

        try:
            if trigger.action == 'add_position':
                await self._execute_pyramiding_order(position, trigger.quantity_multiplier)

            elif trigger.action == 'take_profit':
                await self._execute_take_profit_order(position, 1.0)  # Full exit

            elif trigger.action == 'scale_out':
                await self._execute_take_profit_order(position, trigger.quantity_multiplier)

            elif trigger.action == 'stop_loss':
                await self._execute_stop_loss_order(position)

            # Update last trigger time
            self.last_trigger_times[trigger.trigger_type] = datetime.now()

        except Exception as e:
            logger.error(f"CRITICAL FAILURE: Error executing trigger action {trigger.action} for {position.symbol}: {e} - cannot fallback to simulation")
            raise Exception(f"Trigger action execution failed for {position.symbol}: {e} - no simulation fallback allowed")

    async def _add_to_position(self, position: PositionState, multiplier: float):
        """
        Add to an existing position (pyramiding).
        """
        # Calculate additional quantity
        base_quantity = abs(position.quantity)  # Original position size
        additional_quantity = int(base_quantity * (multiplier - 1.0))

        # Update position
        position.quantity += additional_quantity
        position.pyramiding_tiers_executed += 1
        position.last_tier_price = position.current_price

        # Recalculate P&L
        position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity

        logger.info(f"Pyramided position {position.symbol}: added {additional_quantity} shares, "
                   f"new total: {position.quantity}, tier: {position.pyramiding_tiers_executed}")

        # In a real implementation, this would send actual orders to the broker
        # For now, just log the action
        print(f"🚀 PYRAMIDING TRIGGER: Added {additional_quantity} shares to {position.symbol} "
              f"at ${position.current_price:.2f}")

    async def _take_profit(self, position: PositionState, scale_out_pct: float):
        """
        Take partial or full profit.
        """
        exit_quantity = int(position.quantity * scale_out_pct)
        exit_value = exit_quantity * position.current_price

        # Update position
        position.quantity -= exit_quantity
        position.realized_pnl += (position.current_price - position.entry_price) * exit_quantity

        logger.info(f"Took profit on {position.symbol}: sold {exit_quantity} shares at ${position.current_price:.2f}, "
                   f"realized P&L: ${position.realized_pnl:.2f}")

        print(f"💰 TAKE PROFIT: Sold {exit_quantity} shares of {position.symbol} "
              f"at ${position.current_price:.2f}, realized P&L: ${position.realized_pnl:.2f}")

        # Remove position if fully closed
        if position.quantity == 0:
            self.remove_position(position.symbol)

    async def _stop_loss(self, position: PositionState):
        """
        Execute stop loss.
        """
        exit_value = position.quantity * position.current_price
        loss = (position.entry_price - position.current_price) * abs(position.quantity)

        logger.warning(f"Stop loss triggered for {position.symbol}: closed position at ${position.current_price:.2f}, "
                      f"loss: ${loss:.2f}")

        print(f"🛑 STOP LOSS: Closed {position.symbol} position at ${position.current_price:.2f}, "
              f"loss: ${loss:.2f}")

        # Remove position
        self.remove_position(position.symbol)

    async def _execute_pyramiding_order(self, position: PositionState, multiplier: float):
        """
        Execute pyramiding order via execution agent.
        """
        # Calculate additional quantity
        base_quantity = abs(position.quantity)
        additional_quantity = int(base_quantity * (multiplier - 1.0))

        if additional_quantity <= 0:
            return

        order_details = {
            'symbol': position.symbol,
            'quantity': additional_quantity,
            'action': 'BUY',
            'order_type': 'MKT',
            'pyramiding_trigger': True,
            'trigger_price': position.current_price,
            'strategy_id': f"realtime_pyramid_{position.symbol}_{datetime.now().timestamp()}"
        }

        # Try execution agent first
        if self.execution_agent:
            try:
                result = await self.execution_agent.execute_order(order_details)
                if result.get('success'):
                    # Update position on successful execution
                    position.quantity += additional_quantity
                    position.pyramiding_tiers_executed += 1
                    position.last_tier_price = position.current_price
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity

                    logger.info(f"✅ Pyramiding order executed: {additional_quantity} shares of {position.symbol}")
                    return
            except Exception as e:
                logger.warning(f"Execution agent failed for pyramiding: {e}")

        # Try A2A protocol
        if self.a2a_protocol:
            try:
                message = {
                    'type': 'execute_order',
                    'payload': order_details,
                    'from_agent': 'strategy',
                    'to_agent': 'execution'
                }
                await self.a2a_protocol.send_message(message)
                logger.info(f"📤 Sent pyramiding order via A2A: {additional_quantity} shares of {position.symbol}")
                return
            except Exception as e:
                logger.error(f"CRITICAL FAILURE: A2A protocol failed for pyramiding: {e} - cannot fallback to simulation")
                raise Exception(f"Pyramiding execution failed: {e} - no simulation fallback allowed")

    async def _execute_take_profit_order(self, position: PositionState, scale_out_pct: float):
        """
        Execute take profit order via execution agent.
        """
        exit_quantity = int(position.quantity * scale_out_pct)
        if exit_quantity <= 0:
            return

        order_details = {
            'symbol': position.symbol,
            'quantity': exit_quantity,
            'action': 'SELL',
            'order_type': 'MKT',
            'take_profit_trigger': True,
            'trigger_price': position.current_price,
            'strategy_id': f"realtime_tp_{position.symbol}_{datetime.now().timestamp()}"
        }

        # Try execution agent first
        if self.execution_agent:
            try:
                result = await self.execution_agent.execute_order(order_details)
                if result.get('success'):
                    # Update position on successful execution
                    position.quantity -= exit_quantity
                    position.realized_pnl += (position.current_price - position.entry_price) * exit_quantity

                    logger.info(f"✅ Take profit executed: {exit_quantity} shares of {position.symbol}")
                    if position.quantity == 0:
                        self.remove_position(position.symbol)
                    return
            except Exception as e:
                logger.warning(f"Execution agent failed for take profit: {e}")

        # Try A2A protocol
        if self.a2a_protocol:
            try:
                message = {
                    'type': 'execute_order',
                    'payload': order_details,
                    'from_agent': 'strategy',
                    'to_agent': 'execution'
                }
                await self.a2a_protocol.send_message(message)
                logger.info(f"📤 Sent take profit order via A2A: {exit_quantity} shares of {position.symbol}")
                return
            except Exception as e:
                logger.warning(f"A2A protocol failed for take profit: {e}")

        # Fallback to simulation
        await self._simulate_take_profit_order(position, exit_quantity)

    async def _execute_stop_loss_order(self, position: PositionState):
        """
        Execute stop loss order via execution agent.
        """
        exit_quantity = position.quantity

        order_details = {
            'symbol': position.symbol,
            'quantity': abs(exit_quantity),
            'action': 'SELL',
            'order_type': 'MKT',
            'stop_loss_trigger': True,
            'trigger_price': position.current_price,
            'strategy_id': f"realtime_sl_{position.symbol}_{datetime.now().timestamp()}"
        }

        # Try execution agent first
        if self.execution_agent:
            try:
                result = await self.execution_agent.execute_order(order_details)
                if result.get('success'):
                    logger.warning(f"✅ Stop loss executed: closed {position.symbol} position")
                    self.remove_position(position.symbol)
                    return
            except Exception as e:
                logger.error(f"Execution agent failed for stop loss: {e}")

        # Try A2A protocol
        if self.a2a_protocol:
            try:
                message = {
                    'type': 'execute_order',
                    'payload': order_details,
                    'from_agent': 'strategy',
                    'to_agent': 'execution'
                }
                await self.a2a_protocol.send_message(message)
                logger.warning(f"📤 Sent stop loss order via A2A: closed {position.symbol} position")
                return
            except Exception as e:
                logger.error(f"A2A protocol failed for stop loss: {e}")

        # Fallback to simulation
        await self._simulate_stop_loss_order(position)

    async def _simulate_trigger_action(self, trigger: PyramidingTrigger, position: PositionState):
        """
        Fallback simulation when execution agents are unavailable.
        """
        if trigger.action == 'add_position':
            await self._simulate_pyramiding_order(position, trigger.quantity_multiplier)
        elif trigger.action in ['take_profit', 'scale_out']:
            await self._simulate_take_profit_order(position, trigger.quantity_multiplier)
        elif trigger.action == 'stop_loss':
            await self._simulate_stop_loss_order(position)

    async def _simulate_pyramiding_order(self, position: PositionState, multiplier: float):
        """
        Simulate pyramiding order execution.
        """
        base_quantity = abs(position.quantity)
        additional_quantity = int(base_quantity * (multiplier - 1.0))

        position.quantity += additional_quantity
        position.pyramiding_tiers_executed += 1
        position.last_tier_price = position.current_price
        position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity

        logger.info(f"🎯 SIMULATED: Added {additional_quantity} shares to {position.symbol} at ${position.current_price:.2f}")
        print(f"🎯 SIMULATED PYRAMIDING: Added {additional_quantity} shares to {position.symbol} at ${position.current_price:.2f}")

    async def _simulate_take_profit_order(self, position: PositionState, scale_out_pct: float):
        """
        Simulate take profit order execution.
        """
        exit_quantity = int(position.quantity * scale_out_pct)
        position.quantity -= exit_quantity
        position.realized_pnl += (position.current_price - position.entry_price) * exit_quantity

        logger.info(f"🎯 SIMULATED: Took profit on {exit_quantity} shares of {position.symbol} at ${position.current_price:.2f}")
        print(f"🎯 SIMULATED TAKE PROFIT: Sold {exit_quantity} shares of {position.symbol} at ${position.current_price:.2f}")

        if position.quantity == 0:
            self.remove_position(position.symbol)

    async def _simulate_stop_loss_order(self, position: PositionState):
        """
        Simulate stop loss order execution.
        """
        logger.warning(f"🎯 SIMULATED: Stop loss triggered for {position.symbol} at ${position.current_price:.2f}")
        print(f"🎯 SIMULATED STOP LOSS: Closed {position.symbol} position at ${position.current_price:.2f}")
        self.remove_position(position.symbol)

    def get_position_status(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a position.
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        pnl_pct = position.unrealized_pnl / (position.entry_price * abs(position.quantity))

        return {
            'symbol': symbol,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'quantity': position.quantity,
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'pnl_percentage': pnl_pct,
            'pyramiding_tiers': position.pyramiding_tiers_executed,
            'holding_time_minutes': (datetime.now() - position.entry_time).total_seconds() / 60,
            'active_triggers': len(self.active_triggers.get(symbol, []))
        }

    def get_all_positions_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all monitored positions.
        """
        return {symbol: self.get_position_status(symbol) for symbol in self.positions.keys()}