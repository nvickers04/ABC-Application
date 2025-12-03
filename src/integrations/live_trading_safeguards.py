# integrations/live_trading_safeguards.py
"""
Live Trading Safeguards and Risk Management System

Provides comprehensive risk management for live trading operations including:
- Pre-trade risk checks
- Position size limits
- Daily loss limits
- Circuit breakers
- Emergency stop functionality
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingState(Enum):
    """Current trading system state"""
    NORMAL = "normal"
    CAUTION = "caution"
    RESTRICTED = "restricted"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size_pct: float = 0.05  # 5% of portfolio per position
    max_daily_loss_pct: float = 0.02  # 2% daily loss limit
    max_total_exposure_pct: float = 0.30  # 30% total exposure
    max_single_stock_exposure_pct: float = 0.10  # 10% in single stock
    max_orders_per_hour: int = 10
    max_orders_per_day: int = 50
    require_pre_trade_approval: bool = True
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold_pct: float = 0.05  # 5% loss triggers circuit breaker


@dataclass
class TradingSession:
    """Trading session tracking"""
    session_id: str
    start_time: datetime
    orders_placed: int = 0
    orders_filled: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    last_update: datetime = None

    def __post_init__(self):
        if self.last_update is None:
            self.last_update = self.start_time


class LiveTradingSafeguards:
    """
    Comprehensive risk management system for live trading
    """

    def __init__(self, config_path: str = "config/risk-constraints.yaml"):
        self.config_path = config_path
        self.risk_limits = RiskLimits()
        self.trading_state = TradingState.NORMAL
        self.current_session = None
        self.daily_stats = {}
        self.order_history = []
        self.circuit_breaker_triggered = False

        # Load configuration
        self._load_config()

        # Initialize session
        self._start_new_session()

    def _load_config(self):
        """Load risk configuration from YAML"""
        try:
            import yaml
            config_file = Path(__file__).parent.parent.parent / self.config_path
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                # Update risk limits from config
                risk_config = config.get('live_trading_safeguards', {})
                for key, value in risk_config.items():
                    if hasattr(self.risk_limits, key):
                        setattr(self.risk_limits, key, value)

                logger.info("Live trading safeguards configuration loaded")
            else:
                logger.error(f"CRITICAL FAILURE: Risk config file not found: {config_file} - cannot proceed with defaults")
                raise Exception(f"Risk configuration file {config_file} not found - no fallback defaults allowed")

        except Exception as e:
            logger.error(f"CRITICAL FAILURE: Error loading risk configuration: {e} - cannot proceed with defaults")
            raise Exception(f"Risk configuration loading failed: {e} - no fallback defaults allowed")

    def _start_new_session(self):
        """Start a new trading session"""
        session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.current_session = TradingSession(
            session_id=session_id,
            start_time=datetime.now(timezone.utc)
        )
        logger.info(f"Started new trading session: {session_id}")

    async def pre_trade_risk_check(self, symbol: str, quantity: int, price: float,
                                  order_type: str, account_info: Dict[str, Any],
                                  positions: List[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Perform comprehensive pre-trade risk assessment

        Args:
            symbol: Stock symbol
            quantity: Order quantity
            price: Order price
            order_type: Type of order
            account_info: Current account information
            positions: Current positions

        Returns:
            Tuple of (approved: bool, reason: str, risk_analysis: dict)
        """
        risk_analysis = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'order_value': quantity * price,
            'checks_passed': [],
            'checks_failed': [],
            'warnings': [],
            'risk_level': RiskLevel.LOW.value
        }

        # Check if emergency stop is active
        if self.trading_state == TradingState.EMERGENCY_STOP:
            return False, "Emergency stop is active - all trading suspended", risk_analysis

        # Check circuit breaker
        if self.circuit_breaker_triggered:
            return False, "Circuit breaker triggered - trading suspended", risk_analysis

        # Get portfolio value
        portfolio_value = account_info.get('TotalCashValue', account_info.get('cash_balance', 100000))

        # 1. Position size limit check
        position_size_pct = (quantity * price) / portfolio_value
        if position_size_pct > self.risk_limits.max_position_size_pct:
            risk_analysis['checks_failed'].append('position_size_limit')
            risk_analysis['risk_level'] = RiskLevel.HIGH.value
            return False, f"Position size {position_size_pct:.1%} exceeds limit {self.risk_limits.max_position_size_pct:.1%}", risk_analysis

        risk_analysis['checks_passed'].append('position_size_limit')

        # 2. Single stock exposure check
        current_exposure = sum(p.get('market_value', 0) for p in positions if p.get('symbol') == symbol)
        new_exposure = current_exposure + (quantity * price)
        exposure_pct = new_exposure / portfolio_value

        if exposure_pct > self.risk_limits.max_single_stock_exposure_pct:
            risk_analysis['checks_failed'].append('single_stock_exposure')
            risk_analysis['risk_level'] = RiskLevel.HIGH.value
            return False, f"Single stock exposure {exposure_pct:.1%} exceeds limit {self.risk_limits.max_single_stock_exposure_pct:.1%}", risk_analysis

        risk_analysis['checks_passed'].append('single_stock_exposure')

        # 3. Total exposure check
        total_exposure = sum(p.get('market_value', 0) for p in positions) + (quantity * price)
        total_exposure_pct = total_exposure / portfolio_value

        if total_exposure_pct > self.risk_limits.max_total_exposure_pct:
            risk_analysis['checks_failed'].append('total_exposure_limit')
            risk_analysis['risk_level'] = RiskLevel.MEDIUM.value
            return False, f"Total exposure {total_exposure_pct:.1%} exceeds limit {self.risk_limits.max_total_exposure_pct:.1%}", risk_analysis

        risk_analysis['checks_passed'].append('total_exposure_limit')

        # 4. Daily loss limit check
        daily_pnl_pct = self.current_session.daily_pnl / portfolio_value
        if daily_pnl_pct < -self.risk_limits.max_daily_loss_pct:
            risk_analysis['checks_failed'].append('daily_loss_limit')
            risk_analysis['risk_level'] = RiskLevel.CRITICAL.value
            return False, f"Daily loss {daily_pnl_pct:.1%} exceeds limit {-self.risk_limits.max_daily_loss_pct:.1%}", risk_analysis

        risk_analysis['checks_passed'].append('daily_loss_limit')

        # 5. Order frequency limits
        hourly_orders = len([o for o in self.order_history
                           if (datetime.now(timezone.utc) - o['timestamp']).seconds < 3600])
        if hourly_orders >= self.risk_limits.max_orders_per_hour:
            risk_analysis['warnings'].append('hourly_order_limit_approaching')
            if self.trading_state == TradingState.CAUTION:
                return False, f"Hourly order limit ({self.risk_limits.max_orders_per_hour}) exceeded", risk_analysis

        daily_orders = len([o for o in self.order_history
                          if (datetime.now(timezone.utc) - o['timestamp']).days == 0])
        if daily_orders >= self.risk_limits.max_orders_per_day:
            risk_analysis['checks_failed'].append('daily_order_limit')
            risk_analysis['risk_level'] = RiskLevel.HIGH.value
            return False, f"Daily order limit ({self.risk_limits.max_orders_per_day}) exceeded", risk_analysis

        risk_analysis['checks_passed'].append('order_frequency_limits')

        # 6. Market hours check (simplified - should integrate with exchange calendar)
        current_hour = datetime.now().hour
        if not (9 <= current_hour <= 16):  # Basic market hours check
            risk_analysis['warnings'].append('outside_market_hours')

        # 7. Volatility check (placeholder - would need market data)
        # This would check recent volatility and reject orders in high volatility periods

        # Update risk level based on warnings
        if risk_analysis['warnings']:
            risk_analysis['risk_level'] = RiskLevel.MEDIUM.value

        # All checks passed
        risk_analysis['checks_passed'].extend(['market_hours_check', 'volatility_check'])
        return True, "All risk checks passed", risk_analysis

    async def update_trading_state(self, account_info: Dict[str, Any], positions: List[Dict[str, Any]]):
        """
        Update the current trading state based on market conditions and performance
        """
        portfolio_value = account_info.get('TotalCashValue', account_info.get('cash_balance', 100000))

        # Calculate current P&L
        total_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
        daily_pnl_pct = total_pnl / portfolio_value if portfolio_value > 0 else 0

        # Update session
        self.current_session.total_pnl = total_pnl
        self.current_session.daily_pnl = total_pnl
        self.current_session.last_update = datetime.now(timezone.utc)

        # Check circuit breaker
        if self.risk_limits.enable_circuit_breaker and daily_pnl_pct < -self.risk_limits.circuit_breaker_threshold_pct:
            self.circuit_breaker_triggered = True
            self.trading_state = TradingState.EMERGENCY_STOP
            logger.critical(f"Circuit breaker triggered: Daily loss {daily_pnl_pct:.1%} exceeds threshold {self.risk_limits.circuit_breaker_threshold_pct:.1%}")
            return

        # Update trading state based on performance
        if daily_pnl_pct < -0.01:  # -1% loss
            self.trading_state = TradingState.CAUTION
        elif daily_pnl_pct < -0.03:  # -3% loss
            self.trading_state = TradingState.RESTRICTED
        else:
            self.trading_state = TradingState.NORMAL

        # Reset circuit breaker if P&L recovers
        if self.circuit_breaker_triggered and daily_pnl_pct > -self.risk_limits.circuit_breaker_threshold_pct * 0.5:
            self.circuit_breaker_triggered = False
            self.trading_state = TradingState.CAUTION
            logger.info("Circuit breaker reset - P&L recovered")

    def record_order(self, order_info: Dict[str, Any]):
        """Record an order for risk tracking"""
        order_record = {
            'timestamp': datetime.now(timezone.utc),
            'order_info': order_info,
            'session_id': self.current_session.session_id
        }
        self.order_history.append(order_record)
        self.current_session.orders_placed += 1

        # Keep only recent orders (last 24 hours)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self.order_history = [o for o in self.order_history if o['timestamp'] > cutoff_time]

    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Trigger emergency stop"""
        self.trading_state = TradingState.EMERGENCY_STOP
        logger.critical(f"Emergency stop triggered: {reason}")

    def reset_emergency_stop(self):
        """Reset emergency stop (use with caution)"""
        if self.trading_state == TradingState.EMERGENCY_STOP:
            self.trading_state = TradingState.CAUTION
            logger.warning("Emergency stop reset - proceeding with caution")

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        return {
            'trading_state': self.trading_state.value,
            'circuit_breaker_triggered': self.circuit_breaker_triggered,
            'current_session': {
                'session_id': self.current_session.session_id,
                'orders_placed': self.current_session.orders_placed,
                'total_pnl': self.current_session.total_pnl,
                'daily_pnl': self.current_session.daily_pnl,
                'start_time': self.current_session.start_time.isoformat()
            },
            'risk_limits': {
                'max_position_size_pct': self.risk_limits.max_position_size_pct,
                'max_daily_loss_pct': self.risk_limits.max_daily_loss_pct,
                'max_total_exposure_pct': self.risk_limits.max_total_exposure_pct,
                'max_single_stock_exposure_pct': self.risk_limits.max_single_stock_exposure_pct,
                'max_orders_per_hour': self.risk_limits.max_orders_per_hour,
                'max_orders_per_day': self.risk_limits.max_orders_per_day
            },
            'recent_orders': len(self.order_history),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def validate_market_conditions(self) -> Tuple[bool, str]:
        """
        Validate current market conditions for safe trading

        Returns:
            Tuple of (safe_to_trade: bool, reason: str)
        """
        # Check trading state
        if self.trading_state == TradingState.EMERGENCY_STOP:
            return False, "Emergency stop is active"

        if self.trading_state == TradingState.RESTRICTED:
            return False, "Trading is restricted due to risk conditions"

        # Check market hours (simplified)
        current_time = datetime.now()
        if current_time.weekday() >= 5:  # Weekend
            return False, "Market is closed (weekend)"

        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

        if not (market_open <= current_time <= market_close):
            return False, "Market is closed (outside trading hours)"

        # Could add more checks here:
        # - Circuit breaker status
        # - High volatility periods
        # - News events
        # - Liquidity conditions

        return True, "Market conditions are safe for trading"


# Global safeguards instance
_safeguards_instance: Optional[LiveTradingSafeguards] = None


def get_live_trading_safeguards() -> LiveTradingSafeguards:
    """Get singleton LiveTradingSafeguards instance"""
    global _safeguards_instance
    if _safeguards_instance is None:
        _safeguards_instance = LiveTradingSafeguards()
    return _safeguards_instance


# Convenience functions
async def check_pre_trade_risk(symbol: str, quantity: int, price: float, order_type: str,
                              account_info: Dict[str, Any], positions: List[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
    """Convenience function for pre-trade risk checking"""
    safeguards = get_live_trading_safeguards()
    return await safeguards.pre_trade_risk_check(symbol, quantity, price, order_type, account_info, positions)


async def validate_trading_conditions() -> Tuple[bool, str]:
    """Convenience function for market condition validation"""
    safeguards = get_live_trading_safeguards()
    return await safeguards.validate_market_conditions()


def get_risk_status() -> Dict[str, Any]:
    """Convenience function to get current risk status"""
    safeguards = get_live_trading_safeguards()
    return safeguards.get_risk_status()


def emergency_stop(reason: str = "Manual emergency stop"):
    """Convenience function to trigger emergency stop"""
    safeguards = get_live_trading_safeguards()
    safeguards.emergency_stop(reason)