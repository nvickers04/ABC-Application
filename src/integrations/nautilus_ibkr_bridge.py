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

logger = logging.getLogger(__name__)


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
except ImportError as e:
    NAUTILUS_AVAILABLE = False
    NAUTILUS_IBKR_AVAILABLE = False
    InteractiveBrokersExecutionClient = None
    logger.warning(f"nautilus_trader core not available: {e}, running in compatibility mode")


@dataclass
class BridgeConfig:
    """Configuration for the IBKR bridge"""
    mode: BridgeMode = BridgeMode.IB_INSYNC_ONLY
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

    def __init__(self, config: BridgeConfig = None):
        self.config = config or BridgeConfig()
        self.ibkr_connector = IBKRConnector()
        self.nautilus_client = None
        self._initialized = False

        # Initialize nautilus components if available
        if NAUTILUS_AVAILABLE and self.config.mode != BridgeMode.IB_INSYNC_ONLY:
            self._init_nautilus_components()

    def _init_nautilus_components(self):
        """Initialize nautilus-specific components"""
        try:
            # Create nautilus account ID
            if self.config.account_id:
                account_id = AccountId(self.config.account_id)
            else:
                account_id = AccountId("DU1234567")  # Default paper account

            # Initialize nautilus IBKR client (when fully implemented)
            # self.nautilus_client = InteractiveBrokersLiveExecutionClient(
            #     account_id=account_id,
            #     host=self.config.ibkr_host,
            #     port=self.config.ibkr_port,
            #     client_id=self.config.client_id
            # )

            logger.info("Nautilus components initialized (simulation mode)")

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
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for a symbol"""
        try:
            # Use IBKR only
            return await self.ibkr_connector.get_market_data(symbol)
        except Exception as e:
            logger.error(f"IBKR market data failed for {symbol}: {e}")
            return None



    # Account Methods
    async def get_account_summary(self) -> Optional[Dict[str, Any]]:
        """Get account summary information"""
        return await self.ibkr_connector.get_account_summary()

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        positions = await self.ibkr_connector.get_positions()

        # Enhance with nautilus position objects if available
        if self.config.mode == BridgeMode.NAUTILUS_ENHANCED and NAUTILUS_AVAILABLE:
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

    def _convert_to_nautilus_position(self, position_data: Dict[str, Any]) -> Any:
        """Convert position data to nautilus Position object"""
        # This would create actual nautilus Position objects
        # For now, return a placeholder
        return {
            'symbol': position_data.get('symbol'),
            'quantity': position_data.get('position'),
            'avg_price': position_data.get('avg_cost'),
            'unrealized_pnl': position_data.get('unrealized_pnl', 0),
            'nautilus_enhanced': True
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
        if 'error' in account_info:
            return {
                'success': False,
                'error': f'Cannot get account info: {account_info["error"]}',
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
        if self.config.enable_risk_management and NAUTILUS_AVAILABLE:
            risk_check = await self._check_nautilus_risk(symbol, quantity, action)
            if not risk_check['approved']:
                return {
                    'success': False,
                    'error': f"Nautilus risk check failed: {risk_check['reason']}",
                    'nautilus_risk_check': risk_check
                }

        # Use nautilus position sizing if enabled
        if self.config.enable_position_sizing and NAUTILUS_AVAILABLE:
            sized_quantity = await self._calculate_nautilus_position_size(symbol, quantity)
            quantity = sized_quantity

        # Place order through ib_insync connector
        order_result = await self.ibkr_connector.place_order(symbol, quantity, order_type, action, price)

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

        return await self.ibkr_connector.cancel_order(order_id)

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

        return await self.ibkr_connector.modify_order(order_id, quantity, price)

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
        # Placeholder for nautilus risk management
        # Would integrate with nautilus RiskEngine

        # Basic risk checks for now
        account_summary = await self.get_account_summary()
        if not account_summary:
            return {'approved': False, 'reason': 'Cannot get account summary'}

        # Check position limits (simplified)
        positions = await self.get_positions()
        current_exposure = sum(abs(p.get('position', 0)) * p.get('avg_cost', 0) for p in positions)

        max_exposure = account_summary.get('TotalCashValue', 100000) * 0.5  # 50% max exposure

        if current_exposure > max_exposure:
            return {
                'approved': False,
                'reason': f'Exposure limit exceeded: ${current_exposure:.2f} > ${max_exposure:.2f}'
            }

        return {'approved': True, 'reason': 'Risk check passed'}

    async def _calculate_nautilus_position_size(self, symbol: str, requested_quantity: int) -> int:
        """Calculate position size using nautilus methods"""
        # Placeholder for nautilus position sizing
        # Would use nautilus PositionSizer

        # For now, return requested quantity
        # Future: Implement Kelly criterion, volatility-based sizing, etc.
        return requested_quantity

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


def get_nautilus_ibkr_bridge(config: BridgeConfig = None) -> NautilusIBKRBridge:
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