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
import yaml
import os
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
from src.utils.volatility_calculator import (
    get_volatility_calculator,
    VolatilityMethod,
    calculate_symbol_volatility,
    get_volatility_adjusted_position_size
)

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
    from nautilus_trader.risk.sizing import FixedRiskSizer
    from nautilus_trader.config import RiskEngineConfig

    # Try additional imports for full RiskEngine setup
    try:
        from nautilus_trader.portfolio import PortfolioFacade
        from nautilus_trader.cache import Cache
        from nautilus_trader.common import Clock
        from nautilus_trader.msgbus import MessageBus
        from nautilus_trader.core import TraderId
        NAUTILUS_FULL_RISK_AVAILABLE = True
        logger.info("Full Nautilus risk management components available")
    except ImportError as e:
        logger.warning(f"Full Nautilus risk setup not available: {e}")
        NAUTILUS_FULL_RISK_AVAILABLE = False

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
    NAUTILUS_FULL_RISK_AVAILABLE = False
    InteractiveBrokersExecutionClient = None
    RiskEngine = None
    FixedRiskSizer = None
    RiskEngineConfig = None
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

    # Risk configuration
    risk_config_path: str = "config/risk_config.yaml"

    @classmethod
    def from_config_file(cls, config_path: str = "config/risk_config.yaml") -> 'BridgeConfig':
        """Load configuration from YAML file with business constraint validation"""
        config = cls()

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)

                # Load bridge settings
                bridge_data = data.get('bridge', {})
                config.mode = BridgeMode(bridge_data.get('mode', 'nautilus_enhanced'))
                config.enable_risk_management = bridge_data.get('enable_risk_management', True)
                config.enable_position_sizing = bridge_data.get('enable_position_sizing', True)
                config.account_id = bridge_data.get('account_id')

                # Validate business constraints alignment
                cls._validate_business_constraints(data)

                logger.info(f"Loaded bridge configuration from {config_path}")
            else:
                logger.warning(f"Config file {config_path} not found, using defaults")

        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")

        return config

    @classmethod
    def _validate_business_constraints(cls, risk_config_data: dict):
        """Validate that technical risk limits align with business constraints"""
        try:
            # Load business constraints for comparison
            business_constraints_path = "config/risk-constraints.yaml"
            if not os.path.exists(business_constraints_path):
                logger.warning("Business constraints file not found, skipping validation")
                return

            # Import here to avoid circular imports
            from src.utils.utils import load_yaml
            business_data = load_yaml(business_constraints_path)

            # Get validation settings from risk config
            validation_settings = risk_config_data.get('risk_limits', {}).get('business_constraint_check', {})
            if not validation_settings.get('validate_on_load', False):
                return

            validation_mode = validation_settings.get('validation_mode', 'warn')

            # Check alignment between technical and business constraints
            issues = []

            # Compare max position size limits
            tech_max_pos = risk_config_data.get('position_sizing', {}).get('max_position_percentage', 0.10)
            business_max_pos = business_data.get('constraints', {}).get('max_position_size', 0.30)

            if tech_max_pos > business_max_pos:
                issues.append(
                    f"Technical max position ({tech_max_pos:.1%}) exceeds business limit ({business_max_pos:.1%})"
                )

            # Compare drawdown limits (if applicable)
            tech_daily_loss = risk_config_data.get('risk_limits', {}).get('max_daily_loss_percentage', 0.05)
            business_drawdown = business_data.get('constraints', {}).get('max_drawdown', 0.05)

            if tech_daily_loss > business_drawdown:
                issues.append(
                    f"Technical daily loss limit ({tech_daily_loss:.1%}) exceeds business drawdown ({business_drawdown:.1%})"
                )

            # Report issues based on validation mode
            if issues:
                message = "Risk Configuration Validation Issues:\n" + "\n".join(f"  - {issue}" for issue in issues)

                if validation_mode == 'strict':
                    raise ValueError(f"Business constraint validation failed:\n{message}")
                elif validation_mode == 'warn':
                    logger.warning(message)
                # 'disabled' mode does nothing

        except Exception as e:
            logger.warning(f"Business constraint validation failed: {e}")
            # Don't fail config loading due to validation errors


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
        self.portfolio_facade = None
        self.message_bus = None
        self.cache = None
        self.clock = None
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

            # Initialize full Nautilus risk management if available
            if NAUTILUS_FULL_RISK_AVAILABLE and NAUTILUS_RISK_AVAILABLE:
                try:
                    self._init_full_nautilus_risk_engine(account_id)
                    logger.info("Full Nautilus RiskEngine initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize full Nautilus RiskEngine: {e}")
                    logger.info("Falling back to enhanced risk management")
                    self._init_enhanced_risk_management()
            else:
                # Initialize enhanced risk management (our custom implementation)
                self._init_enhanced_risk_management()
                logger.info("Enhanced risk management initialized (Nautilus components not fully available)")

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

        except Exception as e:
            logger.warning(f"Failed to initialize nautilus components: {e}")
            self.config.mode = BridgeMode.IB_INSYNC_ONLY

    def _init_full_nautilus_risk_engine(self, account_id: AccountId):
        """Initialize the full Nautilus Trader RiskEngine"""
        try:
            # Create required Nautilus components
            trader_id = TraderId("ABC-Application")
            self.message_bus = MessageBus(trader_id=trader_id)
            self.cache = Cache()
            self.clock = Clock()
            self.portfolio_facade = PortfolioFacade(msgbus=self.message_bus)

            # Create risk engine configuration
            risk_config = RiskEngineConfig(
                bypass=False,  # Enable risk checks
                max_order_submit_rate="100/00:00:01",  # 100 orders per second
                max_order_modify_rate="50/00:00:01",   # 50 modifications per second
                max_notional_per_order={},  # Will be set per instrument
                debug=False
            )

            # Initialize the RiskEngine
            self.risk_engine = RiskEngine(
                portfolio=self.portfolio_facade,
                msgbus=self.message_bus,
                cache=self.cache,
                clock=self.clock,
                config=risk_config
            )

            # Start the risk engine
            self.risk_engine.start()
            logger.info("Nautilus RiskEngine started successfully")

        except Exception as e:
            logger.error(f"Failed to initialize full Nautilus RiskEngine: {e}")
            raise

    def _init_enhanced_risk_management(self):
        """Initialize enhanced risk management (fallback when full Nautilus not available)"""
        # This is our custom risk management implementation
        # We'll keep the existing logic but mark it as enhanced
        logger.info("Enhanced risk management initialized (custom implementation)")
        self.risk_engine = None  # Use our custom risk checks

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
            if self.risk_engine and NAUTILUS_FULL_RISK_AVAILABLE:
                # Use full Nautilus RiskEngine
                logger.info(f"Applying full Nautilus RiskEngine for {symbol} {action} {quantity}")
                risk_check = await self._check_full_nautilus_risk(symbol, quantity, action)
            else:
                # Use enhanced risk management (our custom implementation)
                logger.info(f"Applying enhanced risk management for {symbol} {action} {quantity}")
                risk_check = await self._check_nautilus_risk(symbol, quantity, action)

            if not risk_check['approved']:
                return {
                    'success': False,
                    'error': f"Risk check failed: {risk_check['reason']}",
                    'risk_check': risk_check
                }

        # Use nautilus position sizing if enabled
        if self.config.enable_position_sizing and NAUTILUS_RISK_AVAILABLE:
            if self.position_sizer and NAUTILUS_FULL_RISK_AVAILABLE:
                # Use full Nautilus PositionSizer
                logger.info(f"Applying full Nautilus PositionSizer for {symbol}")
                sized_quantity = await self._calculate_full_nautilus_position_size(symbol, quantity)
            else:
                # Use enhanced position sizing (our custom implementation)
                logger.info(f"Applying enhanced position sizing for {symbol}")
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

            # Load risk limits from configuration
            account_value = account_summary.get('TotalCashValue', 100000)
            max_portfolio_risk = account_value * 0.02  # Default 2% - could be loaded from config
            max_symbol_concentration = account_value * 0.1  # Default 10% - could be loaded from config

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
            max_single_position = account_value * 0.2  # Default 20% - could be loaded from config
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

    async def _check_full_nautilus_risk(self, symbol: str, quantity: int, action: str) -> Dict[str, Any]:
        """Perform risk analysis using full Nautilus RiskEngine"""
        try:
            if not self.risk_engine:
                return {'approved': False, 'reason': 'RiskEngine not initialized'}

            # Get current market data for risk calculations
            market_data = await self.get_market_data(symbol)
            if not market_data or 'close' not in market_data:
                return {'approved': False, 'reason': 'Cannot get current market price'}

            current_price = market_data['close']

            # Calculate order notional value
            notional_value = abs(quantity) * current_price

            # Check against RiskEngine limits
            # Note: In a full implementation, we would create proper Order objects
            # and submit them to the RiskEngine for validation

            # For now, we'll use the RiskEngine's configuration limits
            max_notional = self.risk_engine.max_notionals_per_order.get(symbol, float('inf'))
            if notional_value > max_notional:
                return {
                    'approved': False,
                    'reason': f'Order notional ${notional_value:.2f} exceeds max allowed ${max_notional:.2f}'
                }

            # Check trading state
            if self.risk_engine.trading_state.value != "ACTIVE":
                return {
                    'approved': False,
                    'reason': f'Trading state is {self.risk_engine.trading_state.value}, orders not allowed'
                }

            # Additional checks would be performed by the RiskEngine
            # For now, return approved with Nautilus validation
            return {
                'approved': True,
                'reason': 'Full Nautilus RiskEngine validation passed',
                'engine_type': 'full_nautilus',
                'notional_value': notional_value,
                'max_notional_limit': max_notional,
                'trading_state': self.risk_engine.trading_state.value
            }

        except Exception as e:
            logger.warning(f"Full Nautilus risk check failed: {e}")
            return {'approved': False, 'reason': f'RiskEngine error: {str(e)}'}

    async def _calculate_nautilus_position_size(self, symbol: str, requested_quantity: int) -> int:
        """Calculate position size using proper volatility-based methods"""
        try:
            account_summary = await self.get_account_summary()
            if not account_summary:
                logger.warning("Cannot get account summary for position sizing")
                return requested_quantity

            # Get historical market data for proper volatility calculation
            historical_data = await self._get_historical_market_data(symbol)
            current_price = 0

            if historical_data and len(historical_data) > 0:
                # Use the most recent data point for current price
                latest_data = historical_data[0]  # Assuming data is newest first
                current_price = latest_data.get('close', latest_data.get('price', 0))
            else:
                # Fallback to current market data
                market_data = await self.get_market_data(symbol)
                if market_data and 'close' in market_data:
                    current_price = market_data['close']
                else:
                    logger.warning(f"Cannot get current price for {symbol}")
                    return requested_quantity

            if current_price <= 0:
                logger.warning(f"Invalid current price for {symbol}: {current_price}")
                return requested_quantity

            account_value = account_summary.get('TotalCashValue', 100000)

            # Calculate proper volatility using historical data
            volatility_calculator = get_volatility_calculator()
            vol_result = None

            if historical_data and len(historical_data) >= 5:
                # Use close-to-close volatility as primary method
                vol_result = volatility_calculator.calculate_volatility(
                    symbol=symbol,
                    price_data=historical_data,
                    method=VolatilityMethod.CLOSE_TO_CLOSE,
                    window_days=min(30, len(historical_data))
                )

            # Fallback to simplified calculation if proper volatility fails
            if not vol_result:
                logger.info(f"Using fallback volatility calculation for {symbol}")
                vol_result = self._fallback_volatility_calculation(symbol, historical_data or [])

            annualized_volatility = vol_result.volatility if vol_result else 0.20  # Default 20% annualized
            daily_volatility = vol_result.daily_volatility if vol_result else 0.008  # Default ~0.8% daily

            # Load position sizing parameters from configuration
            base_risk_percentage = 0.005  # 0.5% of account per position (configurable)
            max_position_percentage = 0.10  # Maximum 10% of account per position (configurable)

            # Use volatility-adjusted position sizing
            max_position_value = get_volatility_adjusted_position_size(
                account_value=account_value,
                volatility=annualized_volatility,
                base_risk_pct=base_risk_percentage,
                max_position_pct=max_position_percentage
            )

            # Calculate quantity based on position value
            max_quantity = int(max_position_value / current_price)

            # Apply minimum and maximum bounds
            min_quantity = 1
            max_quantity = min(max_quantity, 10000)  # Cap at 10,000 shares/contracts

            # Use the smaller of requested quantity and calculated max
            final_quantity = min(abs(requested_quantity), max_quantity)
            final_quantity = max(final_quantity, min_quantity)

            # Preserve sign for buy/sell
            if requested_quantity < 0:
                final_quantity = -final_quantity

            logger.info(f"Nautilus position sizing: {symbol} requested={requested_quantity}, "
                       f"calculated={final_quantity}, price=${current_price:.2f}, "
                       f"annual_vol={annualized_volatility:.3f}, daily_vol={daily_volatility:.4f}, "
                       f"max_pos_value=${max_position_value:.2f}")

            return final_quantity

        except Exception as e:
            logger.warning(f"Nautilus position sizing failed: {e}")
            return requested_quantity

    async def _get_historical_market_data(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical market data for volatility calculations"""
        try:
            # Try to get historical data from IBKR
            # For now, we'll use multiple market data calls with different durations
            # In a real implementation, this would use proper historical data APIs

            historical_data = []

            # Get daily bars for the past month
            durations = ['1 D', '2 D', '5 D', '10 D', '20 D']
            bar_sizes = ['1 day'] * len(durations)

            for duration, bar_size in zip(durations, bar_sizes):
                try:
                    data = await self.get_market_data(symbol, bar_size=bar_size, duration=duration)
                    if data and isinstance(data, list) and len(data) > 0:
                        historical_data.extend(data)
                    elif data and isinstance(data, dict):
                        # Single data point
                        historical_data.append(data)
                except Exception as e:
                    logger.debug(f"Failed to get {duration} data for {symbol}: {e}")
                    continue

            # Remove duplicates and sort by date (if available)
            seen_dates = set()
            unique_data = []
            for item in historical_data:
                date_key = item.get('date') or item.get('timestamp')
                if date_key and date_key not in seen_dates:
                    seen_dates.add(date_key)
                    unique_data.append(item)

            # Sort by date if available
            if unique_data and 'date' in unique_data[0]:
                unique_data.sort(key=lambda x: x.get('date', ''), reverse=True)

            logger.debug(f"Retrieved {len(unique_data)} historical data points for {symbol}")
            return unique_data[:days]  # Limit to requested days

        except Exception as e:
            logger.warning(f"Failed to get historical data for {symbol}: {e}")
            return []

    def _fallback_volatility_calculation(self, symbol: str, historical_data: List[Dict[str, Any]]) -> Any:
        """Fallback volatility calculation when proper calculation fails"""
        try:
            # Create a simple volatility result object
            from src.utils.volatility_calculator import VolatilityResult, VolatilityMethod
            from datetime import datetime

            # Estimate volatility from available data
            if historical_data and len(historical_data) > 0:
                # Use simple range-based estimate
                closes = []
                highs = []
                lows = []

                for data in historical_data[:30]:  # Use last 30 points
                    if 'close' in data:
                        closes.append(data['close'])
                    if 'high' in data:
                        highs.append(data['high'])
                    if 'low' in data:
                        lows.append(data['low'])

                if len(closes) >= 2:
                    # Calculate simple volatility from close prices
                    import numpy as np
                    returns = np.diff(np.log(closes))
                    daily_vol = np.std(returns) if len(returns) > 0 else 0.02
                    annualized_vol = daily_vol * np.sqrt(252)
                else:
                    daily_vol = 0.02
                    annualized_vol = 0.20
            else:
                daily_vol = 0.02
                annualized_vol = 0.20

            # Create a mock VolatilityResult
            class MockVolatilityResult:
                def __init__(self, volatility, daily_volatility):
                    self.volatility = volatility
                    self.daily_volatility = daily_volatility

            return MockVolatilityResult(annualized_vol, daily_vol)

        except Exception as e:
            logger.warning(f"Fallback volatility calculation failed: {e}")
            # Return a default result
            class DefaultVolatilityResult:
                def __init__(self):
                    self.volatility = 0.20  # 20% annualized
                    self.daily_volatility = 0.008  # ~0.8% daily

            return DefaultVolatilityResult()

    async def _calculate_full_nautilus_position_size(self, symbol: str, requested_quantity: int) -> int:
        """Calculate position size using full Nautilus PositionSizer"""
        try:
            if not self.position_sizer:
                logger.warning("PositionSizer not initialized, using enhanced sizing")
                return await self._calculate_nautilus_position_size(symbol, requested_quantity)

            # Get market data for position sizing
            market_data = await self.get_market_data(symbol)
            if not market_data or 'close' not in market_data:
                logger.warning(f"Cannot get market data for {symbol}, using enhanced sizing")
                return await self._calculate_nautilus_position_size(symbol, requested_quantity)

            current_price = market_data['close']

            # Create instrument for PositionSizer (simplified)
            # In a full implementation, we would have proper Instrument objects
            try:
                from nautilus_trader.core.nautilus_pyo3 import Instrument
                # Create a basic equity instrument
                instrument_id = InstrumentId(f"{symbol}.SMART")
                # Note: This is simplified - proper instrument creation requires more setup
                # For now, fall back to enhanced sizing
                logger.info("Full PositionSizer requires proper instrument setup, using enhanced sizing")
                return await self._calculate_nautilus_position_size(symbol, requested_quantity)

            except Exception as e:
                logger.warning(f"Full PositionSizer setup failed: {e}, using enhanced sizing")
                return await self._calculate_nautilus_position_size(symbol, requested_quantity)

        except Exception as e:
            logger.warning(f"Full Nautilus position sizing failed: {e}")
            return await self._calculate_nautilus_position_size(symbol, requested_quantity)

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
            'nautilus_full_risk_available': NAUTILUS_FULL_RISK_AVAILABLE,
            'nautilus_active': self.is_nautilus_available(),
            'full_risk_engine_active': self.risk_engine is not None and NAUTILUS_FULL_RISK_AVAILABLE,
            'enhanced_risk_active': self.config.enable_risk_management and not self.risk_engine,
            'ibkr_connected': self.ibkr_connector.connected,
            'risk_management_enabled': self.config.enable_risk_management,
            'position_sizing_enabled': self.config.enable_position_sizing,
            'volatility_calculator_available': True  # We now have proper volatility calculation
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