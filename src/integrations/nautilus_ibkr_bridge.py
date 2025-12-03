# integrations/nautilus_ibkr_bridge.py
"""
NautilusTrader IBKR Bridge Adapter

Provides a unified interface that can work with both:
1. Current ib_insync implementation (backward compatibility)
2. Full nautilus_trader IBKR adapter (advanced features)

This bridge allows gradual migration to nautilus_trader while maintaining
all existing functionality.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

# Current IBKR connector
from .ibkr_connector import IBKRConnector
from .live_trading_safeguards import (
    get_live_trading_safeguards,
    check_pre_trade_risk,
    validate_trading_conditions
)
from src.utils.alert_manager import get_alert_manager

logger = logging.getLogger(__name__)
alert_manager = get_alert_manager()


class BridgeMode(Enum):
    """Bridge operation modes"""
    IB_INSYNC_ONLY = "ib_insync_only"  # Use only current ib_insync
    NAUTILUS_ENHANCED = "nautilus_enhanced"  # Use nautilus with ib_insync fallback
    NAUTILUS_FULL = "nautilus_full"  # Use full nautilus (future)


# Nautilus imports
try:
    from nautilus_trader.core.nautilus_pyo3 import (
        AccountBalance, AccountId, AccountState, AccountType,
        Position, Order, OrderSide, OrderType, OrderStatus,
        Venue, InstrumentId, Symbol, Currency
    )
    # Import risk management components
    from nautilus_trader.risk.engine import RiskEngine
    from nautilus_trader.risk.sizing import PositionSizer

    # Try to import IBKR adapter - may fail due to ibapi compatibility
    try:
        from nautilus_trader.adapters.interactive_brokers.execution import InteractiveBrokersExecutionClient
        NAUTILUS_IBKR_AVAILABLE = True
        logger.info("Nautilus Trader IBKR adapter available")
    except ImportError as e:
        logger.warning(f"Nautilus Trader IBKR adapter not available due to dependency issues: {e}")
        NAUTILUS_IBKR_AVAILABLE = False
        InteractiveBrokersExecutionClient = None

    NAUTILUS_AVAILABLE = True
    NAUTILUS_RISK_AVAILABLE = True
except ImportError as e:
    NAUTILUS_AVAILABLE = False
    NAUTILUS_IBKR_AVAILABLE = False
    NAUTILUS_RISK_AVAILABLE = False
    InteractiveBrokersExecutionClient = None
    logger.warning(f"nautilus_trader core not available: {e}, running in compatibility mode")


@dataclass
class BridgeConfig:
    """Configuration for the Nautilus IBKR Bridge"""
    mode: BridgeMode = BridgeMode.NAUTILUS_ENHANCED
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    client_id: int = 1
    account_id: Optional[str] = None
    enable_paper_trading: bool = True
    enable_risk_management: bool = True  # Enable risk management by default
    enable_position_sizing: bool = True  # Enable position sizing by default


class NautilusIBKRBridge:
    """
    Bridge adapter for NautilusTrader IBKR integration

    Provides unified interface for trading operations using either:
    - Current ib_insync implementation (backward compatible)
    - Enhanced nautilus_trader features (when available)
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        self.ibkr_connector = IBKRConnector()
        self.nautilus_client = None
        self.risk_engine = None
        self.position_sizer = None
        self._initialized = False

        # Initialize nautilus components if available
        if NAUTILUS_AVAILABLE and self.config.mode != BridgeMode.IB_INSYNC_ONLY:
            self._init_nautilus_components()

    def _init_nautilus_components(self):
        """Initialize nautilus-specific components"""
        try:
            # Initialize risk management components
            if NAUTILUS_RISK_AVAILABLE:
                # Note: RiskEngine and PositionSizer require more complex setup with
                # TraderId, MessageBus, Portfolio, etc. For now, we'll implement
                # simplified risk management using the core concepts
                logger.info("Nautilus risk management components available")

            # Create nautilus account ID
            if self.config.account_id:
                account_id = AccountId(self.config.account_id)
            else:
                account_id = AccountId("DU1234567")  # Default paper account

            # Initialize nautilus IBKR client (when IBKR adapter compatibility is resolved)
            if NAUTILUS_IBKR_AVAILABLE:
                # self.nautilus_client = InteractiveBrokersExecutionClient(
                #     account_id=account_id,
                #     host=self.config.ibkr_host,
                #     port=self.config.ibkr_port,
                #     client_id=self.config.client_id
                # )
                logger.info("Nautilus IBKR client available but not initialized due to compatibility issues")
            else:
                logger.info("Nautilus IBKR client not available - using enhanced risk management only")

            logger.info("Nautilus components initialized with risk management capabilities")

        except Exception as e:
            logger.warning(f"Failed to initialize nautilus components: {e}")
            self.config.mode = BridgeMode.IB_INSYNC_ONLY

    async def initialize(self) -> bool:
        """Initialize the bridge and connect to IBKR"""
        try:
            # Always initialize ib_insync connector for backward compatibility
            await self.ibkr_connector.connect()
            logger.info("IBKR connector initialized")

            # Initialize nautilus components if in enhanced mode
            if self.config.mode == BridgeMode.NAUTILUS_ENHANCED and NAUTILUS_AVAILABLE:
                # Future: Initialize nautilus client
                logger.info("Nautilus enhanced mode ready")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize bridge: {e}")
            return False

    async def disconnect(self):
        """Disconnect from IBKR"""
        try:
            await self.ibkr_connector.disconnect()
            if self.nautilus_client:
                # Future: disconnect nautilus client
                pass
            logger.info("Bridge disconnected")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    # Market Data Methods
    async def get_market_data(self, symbol: str, bar_size: str = '1 min',
                             duration: str = '1 D') -> Optional[Dict[str, Any]]:
        """Get market data for a symbol with specified parameters"""
        try:
            # Use IBKR connector with full parameters
            return await self.ibkr_connector.get_market_data(symbol, bar_size, duration)
        except Exception as e:
            logger.error(f"IBKR market data failed for {symbol}: {e}")
            await alert_manager.error(
                e,
                {"symbol": symbol, "bar_size": bar_size, "duration": duration},
                "ibkr_integration"
            )
            return None



    # Account Methods
    async def get_account_summary(self) -> Optional[Dict[str, Any]]:
        """Get account summary information"""
        return await self.ibkr_connector.get_account_summary()

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        positions = await self.ibkr_connector.get_positions()

        # Enhance with nautilus position analysis if available
        if self.config.mode in [BridgeMode.NAUTILUS_ENHANCED, BridgeMode.NAUTILUS_FULL] and NAUTILUS_RISK_AVAILABLE:
            enhanced_positions = []
            for pos in positions:
                # Convert to nautilus Position object for advanced analysis
                try:
                    nautilus_pos = self._convert_to_nautilus_position(pos)
                    enhanced_pos = {
                        **pos,
                        'nautilus_position': nautilus_pos,
                        'enhanced_analysis': True
                    }
                    enhanced_positions.append(enhanced_pos)
                except Exception as e:
                    logger.warning(f"Failed to enhance position {pos.get('symbol')}: {e}")
                    enhanced_positions.append(pos)
            return enhanced_positions

        return positions

    def _convert_to_nautilus_position(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance position data with nautilus-style analysis"""
        try:
            symbol = position_data.get('symbol', '')
            quantity = position_data.get('position', 0)
            avg_cost = position_data.get('avg_cost', 0)
            unrealized_pnl = position_data.get('unrealized_pnl', 0)

            # Calculate position metrics
            position_value = abs(quantity) * avg_cost
            pnl_percentage = (unrealized_pnl / position_value * 100) if position_value > 0 else 0

            # Risk metrics
            position_risk = {
                'position_value': position_value,
                'pnl_percentage': pnl_percentage,
                'is_long': quantity > 0,
                'is_short': quantity < 0,
                'exposure': position_value,
                'risk_weight': min(position_value / 100000, 1.0),  # Simplified risk weight
                'nautilus_enhanced': True
            }

            return position_risk

        except Exception as e:
            logger.warning(f"Failed to enhance position data: {e}")
            return {
                'symbol': position_data.get('symbol'),
                'nautilus_enhanced': False,
                'error': str(e)
            }

    # Order Methods
    async def place_order(self, symbol: str, quantity: int, order_type: str = "MKT",
                         action: str = "BUY", price: Optional[float] = None) -> Dict[str, Any]:
        """Place an order with enhanced nautilus features and risk management"""

        # Validate market conditions
        market_safe, market_reason = await validate_trading_conditions()
        if not market_safe:
            return {
                'success': False,
                'error': f'Market conditions not safe: {market_reason}',
                'market_check': False
            }

        # Get current account and position info for risk checks
        account_info = await self.get_account_summary()
        if not account_info or 'error' in account_info:
            error_msg = account_info.get('error', 'Unknown account error') if account_info else 'Cannot get account summary'
            return {
                'success': False,
                'error': f'Cannot get account info: {error_msg}',
                'account_check': False
            }

        positions = await self.get_positions()

        # Get current price for risk calculations
        current_price = price
        if not current_price:
            market_data = await self.get_market_data(symbol)
            if market_data and 'close' in market_data:
                current_price = market_data['close']
            else:
                return {
                    'success': False,
                    'error': 'Cannot determine current price for risk calculation',
                    'price_check': False
                }

        # Perform pre-trade risk check
        risk_approved, risk_reason, risk_analysis = await check_pre_trade_risk(
            symbol=symbol,
            quantity=quantity,
            price=current_price,
            order_type=order_type,
            account_info=account_info,
            positions=positions
        )

        if not risk_approved:
            return {
                'success': False,
                'error': f'Risk check failed: {risk_reason}',
                'risk_analysis': risk_analysis
            }

        # Use nautilus risk management if enabled
        if self.config.enable_risk_management and NAUTILUS_RISK_AVAILABLE:
            logger.info(f"Applying Nautilus risk management for {symbol} {action} {quantity}")
            risk_check = await self._check_nautilus_risk(symbol, quantity, action)
            if not risk_check['approved']:
                return {
                    'success': False,
                    'error': f"Nautilus risk check failed: {risk_check['reason']}",
                    'nautilus_risk_check': risk_check
                }

        # Use nautilus position sizing if enabled
        if self.config.enable_position_sizing and NAUTILUS_RISK_AVAILABLE:
            logger.info(f"Applying Nautilus position sizing for {symbol}")
            sized_quantity = await self._calculate_nautilus_position_size(symbol, quantity)
            if sized_quantity != quantity:
                logger.info(f"Position size adjusted from {quantity} to {sized_quantity} for risk management")
                quantity = sized_quantity

        # Place order through ib_insync connector
        try:
            order_result = await self.ibkr_connector.place_order(symbol, quantity, order_type, action, price)
        except Exception as e:
            logger.error(f"IBKR order placement failed for {symbol}: {e}")
            await alert_manager.error(
                e,
                {"symbol": symbol, "quantity": quantity, "action": action},
                "ibkr_integration"
            )
            return {
                'success': False,
                'error': f'Order placement failed: {str(e)}',
                'risk_analysis': risk_analysis
            }

        # Record order in safeguards system if successful
        if order_result.get('success', False) or 'order_id' in order_result:
            safeguards = get_live_trading_safeguards()
            safeguards.record_order(order_result)

        # Update trading state after order
        await safeguards.update_trading_state(account_info, positions)

        # Add risk analysis to result
        order_result['risk_analysis'] = risk_analysis
        order_result['market_check'] = True

        return order_result

    async def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """Cancel an order with safeguards"""
        safeguards = get_live_trading_safeguards()

        # Check if emergency stop is active
        if safeguards.trading_state.value == 'emergency_stop':
            return {
                'success': False,
                'error': 'Emergency stop active - order cancellation blocked',
                'order_id': order_id
            }

        try:
            return await self.ibkr_connector.cancel_order(order_id)
        except Exception as e:
            logger.error(f"IBKR order cancellation failed for order {order_id}: {e}")
            await alert_manager.error(
                e,
                {"order_id": order_id},
                "ibkr_integration"
            )
            return {
                'success': False,
                'error': f'Order cancellation failed: {str(e)}',
                'order_id': order_id
            }

    async def modify_order(self, order_id: int, quantity: Optional[int] = None,
                          price: Optional[float] = None) -> Dict[str, Any]:
        """Modify an order with safeguards"""
        safeguards = get_live_trading_safeguards()

        # Check if emergency stop is active
        if safeguards.trading_state.value == 'emergency_stop':
            return {
                'success': False,
                'error': 'Emergency stop active - order modification blocked',
                'order_id': order_id
            }

        try:
            return await self.ibkr_connector.modify_order(order_id, quantity, price)
        except Exception as e:
            logger.error(f"IBKR order modification failed for order {order_id}: {e}")
            await alert_manager.error(
                e,
                {"order_id": order_id, "quantity": quantity, "price": price},
                "ibkr_integration"
            )
            return {
                'success': False,
                'error': f'Order modification failed: {str(e)}',
                'order_id': order_id
            }

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get open orders"""
        return await self.ibkr_connector.get_open_orders()

    async def get_order_status(self, order_id: int) -> Optional[Dict[str, Any]]:
        """Get order status"""
        return await self.ibkr_connector.get_order_status(order_id)

    async def place_bracket_order(self, symbol: str, quantity: int, entry_price: float,
                                stop_loss_pct: float = 0.05, take_profit_pct: float = 0.10) -> Dict[str, Any]:
        """Place bracket order with risk checks"""
        # Perform risk check for the entry order
        account_info = await self.get_account_summary()
        positions = await self.get_positions()

        risk_approved, risk_reason, risk_analysis = await check_pre_trade_risk(
            symbol=symbol,
            quantity=quantity,
            price=entry_price,
            order_type='LMT',
            account_info=account_info,
            positions=positions
        )

        if not risk_approved:
            return {
                'success': False,
                'error': f'Risk check failed: {risk_reason}',
                'risk_analysis': risk_analysis
            }

        result = await self.ibkr_connector.place_bracket_order(symbol, quantity, entry_price,
                                                             stop_loss_pct, take_profit_pct)

        # Record order if successful
        if result.get('success', False):
            safeguards = get_live_trading_safeguards()
            safeguards.record_order(result)

        return result

    async def get_portfolio_pnl(self) -> Dict[str, Any]:
        """Get portfolio P&L with risk status"""
        pnl_data = await self.ibkr_connector.get_portfolio_pnl()

        # Add risk status
        safeguards = get_live_trading_safeguards()
        risk_status = safeguards.get_risk_status()

        pnl_data['risk_status'] = risk_status
        return pnl_data

    async def _check_nautilus_risk(self, symbol: str, quantity: int, action: str) -> Dict[str, Any]:
        """Perform nautilus-style risk analysis"""
        try:
            account_summary = await self.get_account_summary()
            if not account_summary:
                return {'approved': False, 'reason': 'Cannot get account summary'}

            positions = await self.get_positions()
            market_data = await self.get_market_data(symbol)

            # Get current price for calculations
            current_price = market_data.get('close', 0) if market_data else 0
            if not current_price:
                return {'approved': False, 'reason': 'Cannot get current market price'}

            # Calculate position values and exposure
            total_exposure = 0
            symbol_exposure = 0
            for position in positions:
                pos_size = abs(position.get('position', 0))
                avg_cost = position.get('avg_cost', 0)
                pos_value = pos_size * avg_cost
                total_exposure += pos_value

                if position.get('symbol') == symbol:
                    symbol_exposure = pos_value

            # Nautilus-style risk limits
            account_value = account_summary.get('TotalCashValue', 100000)
            max_portfolio_risk = account_value * 0.02  # 2% max portfolio risk per trade
            max_symbol_concentration = account_value * 0.1  # 10% max concentration per symbol

            # Calculate proposed trade value
            trade_value = abs(quantity) * current_price

            # Check portfolio risk limit
            if trade_value > max_portfolio_risk:
                return {
                    'approved': False,
                    'reason': f'Trade value ${trade_value:.2f} exceeds max portfolio risk ${max_portfolio_risk:.2f} (2%)'
                }

            # Check symbol concentration limit
            new_symbol_exposure = symbol_exposure + trade_value
            if new_symbol_exposure > max_symbol_concentration:
                return {
                    'approved': False,
                    'reason': f'New exposure ${new_symbol_exposure:.2f} exceeds max symbol concentration ${max_symbol_concentration:.2f} (10%)'
                }

            # Check diversification (max 20% of portfolio in any single position)
            max_single_position = account_value * 0.2
            if new_symbol_exposure > max_single_position:
                return {
                    'approved': False,
                    'reason': f'Position would exceed 20% of portfolio value'
                }

            # Additional Nautilus-style checks
            risk_checks = {
                'portfolio_risk_limit': trade_value <= max_portfolio_risk,
                'concentration_limit': new_symbol_exposure <= max_symbol_concentration,
                'diversification_limit': new_symbol_exposure <= max_single_position,
                'account_value_check': account_value > 10000  # Minimum account size
            }

            failed_checks = [k for k, v in risk_checks.items() if not v]
            if failed_checks:
                return {
                    'approved': False,
                    'reason': f'Failed risk checks: {", ".join(failed_checks)}',
                    'details': risk_checks
                }

            return {
                'approved': True,
                'reason': 'All Nautilus risk checks passed',
                'details': {
                    'trade_value': trade_value,
                    'max_portfolio_risk': max_portfolio_risk,
                    'new_symbol_exposure': new_symbol_exposure,
                    'max_symbol_concentration': max_symbol_concentration
                }
            }

        except Exception as e:
            logger.warning(f"Nautilus risk check failed: {e}")
            return {'approved': False, 'reason': f'Risk check error: {str(e)}'}

    async def _calculate_nautilus_position_size(self, symbol: str, requested_quantity: int) -> int:
        """Calculate position size using nautilus-inspired methods"""
        try:
            account_summary = await self.get_account_summary()
            if not account_summary:
                logger.warning("Cannot get account summary for position sizing")
                return requested_quantity

            market_data = await self.get_market_data(symbol)
            if not market_data or 'close' not in market_data:
                logger.warning(f"Cannot get market data for {symbol}")
                return requested_quantity

            current_price = market_data['close']
            account_value = account_summary.get('TotalCashValue', 100000)

            # Get volatility estimate (simplified - in real Nautilus this would use proper vol calculation)
            volatility = self._estimate_volatility(symbol, market_data)

            # Nautilus-style position sizing based on risk management principles
            # Use fixed percentage of account with volatility adjustment
            base_risk_percentage = 0.005  # 0.5% of account per position

            # Adjust for volatility (higher vol = smaller position)
            if volatility > 0.02:  # 2% daily volatility threshold
                risk_multiplier = max(0.5, 1.0 - (volatility - 0.02) * 10)  # Reduce size for high vol
            else:
                risk_multiplier = 1.0

            adjusted_risk_percentage = base_risk_percentage * risk_multiplier

            # Calculate position size
            max_position_value = account_value * adjusted_risk_percentage
            max_quantity = int(max_position_value / current_price)

            # Apply minimum and maximum bounds
            min_quantity = 1
            max_quantity = min(max_quantity, 1000)  # Cap at 1000 shares/contracts

            # Use the smaller of requested quantity and calculated max
            final_quantity = min(abs(requested_quantity), max_quantity)
            final_quantity = max(final_quantity, min_quantity)

            # Preserve sign for buy/sell
            if requested_quantity < 0:
                final_quantity = -final_quantity

            logger.info(f"Nautilus position sizing: {symbol} requested={requested_quantity}, "
                       f"calculated={final_quantity}, price=${current_price:.2f}, "
                       f"volatility={volatility:.4f}, risk_mult={risk_multiplier:.2f}")

            return final_quantity

        except Exception as e:
            logger.warning(f"Nautilus position sizing failed: {e}")
            return requested_quantity

    def _estimate_volatility(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Estimate volatility for position sizing (simplified)"""
        # In a real implementation, this would calculate proper historical volatility
        # For now, use a simple range-based estimate
        try:
            high = market_data.get('high', 0)
            low = market_data.get('low', 0)
            close = market_data.get('close', 0)

            if high > 0 and low > 0 and close > 0:
                # Daily range as % of price
                daily_range_pct = (high - low) / close
                # Estimate annualized volatility (simplified)
                # Assuming 252 trading days, scale daily range
                annualized_vol = daily_range_pct * (252 ** 0.5)
                return min(annualized_vol, 1.0)  # Cap at 100% vol
            else:
                return 0.02  # Default 2% daily volatility
        except:
            return 0.02

    # Utility Methods
    def is_nautilus_available(self) -> bool:
        """Check if nautilus components are available"""
        return NAUTILUS_AVAILABLE and self.nautilus_client is not None

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status"""
        return {
            'initialized': self._initialized,
            'mode': self.config.mode.value,
            'nautilus_available': NAUTILUS_AVAILABLE,
            'nautilus_ibkr_available': NAUTILUS_IBKR_AVAILABLE,
            'nautilus_active': self.is_nautilus_available(),
            'ibkr_connected': self.ibkr_connector.connected,
            'risk_management_enabled': self.config.enable_risk_management,
            'position_sizing_enabled': self.config.enable_position_sizing
        }


# Global bridge instance
_bridge_instance: Optional[NautilusIBKRBridge] = None


def get_nautilus_ibkr_bridge(config: Optional[BridgeConfig] = None) -> NautilusIBKRBridge:
    """Get singleton NautilusIBKRBridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = NautilusIBKRBridge(config)
    return _bridge_instance


# Convenience functions for backward compatibility
async def initialize_bridge(mode: str = "ib_insync_only") -> bool:
    """Initialize the bridge with specified mode"""
    config = BridgeConfig(mode=BridgeMode(mode))
    bridge = get_nautilus_ibkr_bridge(config)
    return await bridge.initialize()


async def get_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Get market data (backward compatible)"""
    bridge = get_nautilus_ibkr_bridge()
    return await bridge.get_market_data(symbol)


async def place_order(symbol: str, quantity: int, order_type: str = "MKT",
                     action: str = "BUY", price: Optional[float] = None) -> Dict[str, Any]:
    """Place order (backward compatible)"""
    bridge = get_nautilus_ibkr_bridge()
    return await bridge.place_order(symbol, quantity, order_type, action, price)