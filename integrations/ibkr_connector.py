# integrations/ibkr_connector.py
# Purpose: IBKR Paper Trading Connector for the AI Portfolio Manager
# Provides paper trading functionality using ib_insync and nautilus_trader
# Handles connection, account management, order execution, and position monitoring

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import asyncio
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import time

# IBKR imports
from ib_insync import IB, Contract, Order, util
from ib_insync.contract import Stock, Option
import exchange_calendars as ecals

# Local imports
# from src.utils.config import load_yaml  # Not needed for IBKR connector
from src.utils.config import get_api_key
import os

logger = logging.getLogger(__name__)

class IBKRConnector:
    """
    IBKR Paper Trading Connector
    Handles connection to IBKR paper trading account and provides trading functionality
    Thread-safe singleton implementation.
    """

    _instance: Optional['IBKRConnector'] = None
    _lock: Lock = Lock()

    def __new__(cls, config_path: str = 'config/ibkr_config.ini') -> 'IBKRConnector':
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern for thread safety
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = 'config/ibkr_config.ini'):
        """
        Initialize IBKR connector with paper trading configuration
        Note: This may be called multiple times due to singleton pattern,
        but initialization should only happen once.
        """
        # Check if already initialized (singleton pattern)
        if hasattr(self, '_initialized'):
            return

        self.config = self._load_config(config_path)

        # Load IBKR credentials from environment variables
        self.username = os.getenv('IBKR_USERNAME')
        self.password = os.getenv('IBKR_PASSWORD')
        self.account_id_env = os.getenv('IBKR_ACCOUNT_ID')

        self.ib = IB()
        self.connected = False
        self.account_id = None
        self.calendar = ecals.get_calendar('NYSE')  # For market hours validation

        # Connection settings from environment or config
        self.host = os.getenv('IBKR_HOST', self.config.get('paper_host', '127.0.0.1'))
        self.port = int(os.getenv('IBKR_PORT', self.config.get('paper_port', '7497')))  # Paper trading port
        self.client_id = int(os.getenv('IBKR_CLIENT_ID', self.config.get('client_id', '2')))

        # Account settings
        self.account_currency = self.config.get('account_currency', 'USD')

        # Thread pool for blocking IBKR operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ibkr")

        # Validate credentials
        if not self.username or not self.password:
            logger.warning("IBKR credentials not found in environment variables. Set IBKR_USERNAME and IBKR_PASSWORD.")

        # Mark as initialized
        self._initialized = True

        logger.info("IBKR Paper Trading Connector initialized")

    def __del__(self):
        """Cleanup thread pool executor on destruction"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    def _run_in_executor(self, func, *args, timeout: float = 30.0, **kwargs):
        """
        Safely run a blocking function in the thread pool executor.
        Handles cases where there's no running event loop and prevents deadlocks.

        Args:
            func: The blocking function to run
            *args: Positional arguments for the function
            timeout: Timeout in seconds (default 30)
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call

        Raises:
            asyncio.TimeoutError: If the operation times out
            Exception: If the operation fails
        """
        async def _run_with_timeout():
            try:
                # Try to get the running loop first (preferred for async contexts)
                loop = asyncio.get_running_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(self._executor, func, *args, **kwargs),
                    timeout=timeout
                )
            except RuntimeError:
                # No running loop, try to get existing loop
                try:
                    loop = asyncio.get_event_loop()
                    return await asyncio.wait_for(
                        loop.run_in_executor(self._executor, func, *args, **kwargs),
                        timeout=timeout
                    )
                except RuntimeError:
                    # No event loop at all, this shouldn't happen in normal async code
                    logger.error("No event loop available for IBKR operation")
                    raise RuntimeError("IBKR operations must be called from within an async context")

        return _run_with_timeout()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load IBKR configuration from file"""
        try:
            config_file = Path(__file__).parent.parent / config_path
            if config_file.exists():
                with open(config_file, 'r') as f:
                    # Parse simple key=value format
                    config = {}
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if '=' in line:
                                key, value = line.split('=', 1)
                                config[key.strip()] = value.strip()
                    return config
            else:
                logger.error(f"CRITICAL FAILURE: Config file {config_path} not found - cannot proceed with defaults")
                raise Exception(f"IBKR config file {config_path} not found - no fallback defaults allowed")
        except Exception as e:
            logger.error(f"CRITICAL FAILURE: Error loading IBKR config: {e} - cannot proceed with defaults")
            raise Exception(f"IBKR config loading failed: {e} - no fallback defaults allowed")

    async def connect(self) -> bool:
        """
        Connect to IBKR paper trading account with improved asyncio handling.

        Returns:
            bool: True if connection successful
        """
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                if self.connected:
                    logger.info("Already connected to IBKR")
                    return True

                if not self.username or not self.password:
                    logger.error("IBKR credentials not available. Please set IBKR_USERNAME and IBKR_PASSWORD in .env file.")
                    return False

                # Use threading with a new event loop for IBKR operations
                import threading
                import asyncio as asyncio_module

                connection_result = {'connected': False, 'error': None}

                def connect_in_thread():
                    # Create new event loop for this thread
                    loop = asyncio_module.new_event_loop()
                    asyncio_module.set_event_loop(loop)

                    try:
                        # Create IB instance
                        temp_ib = IB()
                        logger.info(f"Attempting to connect to IBKR at {self.host}:{self.port} with client ID {self.client_id}")

                        # Use connectAsync directly in our event loop
                        connect_task = temp_ib.connectAsync(self.host, self.port, self.client_id, timeout=10)
                        loop.run_until_complete(connect_task)

                        # Wait for connection to be fully established
                        timeout = 15  # Increased timeout
                        start_time = time.time()
                        logger.info("Waiting for connection to be established...")
                        while not temp_ib.isConnected() and (time.time() - start_time) < timeout:
                            time.sleep(0.5)  # Longer sleep interval
                            logger.debug(f"Still waiting for connection... {(time.time() - start_time):.1f}s elapsed")

                        if temp_ib.isConnected():
                            logger.info("Connection established, attempting to get managed accounts...")
                            # Successfully connected, transfer to main instance
                            self.ib = temp_ib
                            connection_result['connected'] = True
                            self.account_id = self.ib.managedAccounts()[0] if self.ib.managedAccounts() else self.account_id_env
                            logger.info(f"Successfully connected to IBKR Paper Trading. Account: {self.account_id}")
                        else:
                            logger.warning("Connection timeout - disconnecting...")
                            temp_ib.disconnect()
                            connection_result['error'] = 'Connection timeout'

                    except Exception as e:
                        connection_result['error'] = str(e)
                        logger.error(f"Connection error: {e}")
                        logger.error(f"Error type: {type(e)}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                    finally:
                        loop.close()

                # Run connection in separate thread with its own event loop
                connect_thread = threading.Thread(target=connect_in_thread)
                connect_thread.start()
                connect_thread.join(timeout=15)  # 15 second timeout

                if connection_result['connected']:
                    self.connected = True
                    return True
                else:
                    error_msg = connection_result.get('error', 'Unknown error')
                    logger.warning(f"Connection attempt {attempt + 1} failed: {error_msg}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff

            except Exception as e:
                logger.error(f"Error in connection attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5

        logger.error("Failed to connect to IBKR Paper Trading after all retries")
        logger.info("Make sure:")
        logger.info("1. IBKR Trader Workstation (TWS) or Gateway is running")
        logger.info("2. Paper trading account is enabled")
        logger.info("3. API connections are enabled in TWS/Gateway")
        logger.info("4. Correct host/port settings (127.0.0.1:7497 for paper trading)")
        return False

    async def _wait_for_connection(self) -> None:
        """
        Wait for IBKR connection to be fully established.
        """
        # Give IBKR a moment to establish the connection
        await asyncio.sleep(1.0)
        # Additional connection validation could be added here

    async def disconnect(self):
        """Disconnect from IBKR"""
        try:
            if self.connected:
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from IBKR")
        except Exception as e:
            logger.error(f"Error disconnecting from IBKR: {e}")

    async def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary information with error recovery.

        Returns:
            Dict with account details including cash, positions, etc.
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return {'error': 'Not connected to IBKR'}

            # Get real account summary data using proper async handling
            account_values = await self._run_in_executor(self.ib.accountValues)

            account_info = {
                'account_id': self.account_id,
                'currency': self.account_currency,
                'positions': [],  # Will get from positions call
                'total_positions': 0,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'real_data': True  # Flag to indicate this is real data
            }

            # Parse account values
            for value in account_values:
                tag = value.tag
                val = value.value

                if tag == 'NetLiquidation':
                    account_info['NetLiquidation'] = float(val)
                    account_info['total_value'] = float(val)
                elif tag == 'TotalCashValue':
                    account_info['TotalCashValue'] = float(val)
                    account_info['cash_balance'] = float(val)
                elif tag == 'BuyingPower':
                    account_info['BuyingPower'] = float(val)
                    account_info['buying_power'] = float(val)
                elif tag == 'AvailableFunds':
                    account_info['AvailableFunds'] = float(val)
                    account_info['available_funds'] = float(val)
                elif tag == 'EquityWithLoanValue':
                    account_info['EquityWithLoanValue'] = float(val)
                elif tag == 'InitMarginReq':
                    account_info['InitMarginReq'] = float(val)
                elif tag == 'MaintMarginReq':
                    account_info['MaintMarginReq'] = float(val)
                elif tag == 'DayTradesRemaining':
                    account_info['DayTradesRemaining'] = int(float(val))
                    account_info['day_trades_remaining'] = int(float(val))

            # Set defaults if not found
            if 'cash_balance' not in account_info:
                account_info['cash_balance'] = 0.0
                account_info['TotalCashValue'] = 0.0
            if 'total_value' not in account_info:
                account_info['total_value'] = account_info['cash_balance']
                account_info['NetLiquidation'] = account_info['cash_balance']

            logger.info(f"Account summary retrieved: ${account_info['cash_balance']:,.2f} cash, "
                       f"${account_info['total_value']:,.2f} total value")
            return account_info

        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            # Try to reconnect on error
            self.connected = False
            return {'error': str(e)}

    async def place_order(self, symbol: str, quantity: int, order_type: str = 'MKT',
                         action: str = 'BUY', price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an order on IBKR paper trading with improved asyncio handling.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            order_type: Order type ('MKT', 'LMT', etc.)
            action: 'BUY' or 'SELL'
            price: Limit price (for LMT orders)

        Returns:
            Dict with order details and status
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return {'error': 'Not connected to IBKR'}

            # Check if market is open
            if not self._is_market_open():
                return {'error': 'Market is closed', 'market_open': False}

            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')

            # Create order
            order = Order()
            order.action = action
            order.totalQuantity = abs(quantity)
            order.orderType = order_type

            if order_type == 'LMT' and price:
                order.lmtPrice = price

            # Place order using synchronous IBKR method
            trade = self.ib.placeOrder(contract, order)

            # Wait a bit for order to process
            await asyncio.sleep(1)

            order_result = {
                'success': True,
                'order_id': trade.order.orderId,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type,
                'status': trade.orderStatus.status if hasattr(trade, 'orderStatus') else 'Submitted',
                'filled_quantity': trade.orderStatus.filled if hasattr(trade, 'orderStatus') else 0,
                'remaining_quantity': trade.orderStatus.remaining if hasattr(trade, 'orderStatus') else quantity,
                'avg_fill_price': trade.orderStatus.avgFillPrice if hasattr(trade, 'orderStatus') else 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            logger.info(f"Order placed: {action} {quantity} {symbol} at {order_type}")
            return order_result

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            # Try to reconnect on error
            self.connected = False
            return {'error': str(e)}

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions

        Returns:
            List of position dictionaries
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return []

            # Get real positions data using proper async handling
            positions = await self._run_in_executor(self.ib.positions)

            position_list = []
            for position in positions:
                if position.position != 0:  # Only include non-zero positions
                    position_dict = {
                        'account': position.account,
                        'symbol': position.contract.symbol,
                        'secType': position.contract.secType,
                        'currency': position.contract.currency,
                        'position': position.position,
                        'avgCost': position.avgCost,
                        'marketPrice': getattr(position, 'marketPrice', 0),
                        'marketValue': getattr(position, 'marketValue', 0),
                        'unrealizedPNL': getattr(position, 'unrealizedPNL', 0),
                        'realizedPNL': getattr(position, 'realizedPNL', 0),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    position_list.append(position_dict)

            logger.info(f"Retrieved {len(position_list)} positions from IBKR")
            return position_list

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def _is_market_open(self) -> bool:
        """
        Check if the market is currently open

        Returns:
            bool: True if market is open
        """
        try:
            now = datetime.now(timezone.utc)
            return self.calendar.is_open_on_timestamp(now)
        except Exception as e:
            logger.warning(f"Error checking market hours: {e}")
            # Default to assuming market is open if calendar check fails
            return True

    async def get_market_data(self, symbol: str, bar_size: str = '1 min',
                             duration: str = '1 D') -> Optional[Dict[str, Any]]:
        """
        Get market data for a symbol

        Args:
            symbol: Stock symbol
            bar_size: Bar size ('1 min', '5 mins', etc.)
            duration: Duration ('1 D', '1 W', etc.)

        Returns:
            Dict with market data or None if error
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return None

            contract = Stock(symbol, 'SMART', 'USD')

            # Request historical data
            bars = await self._run_in_executor(
                self.ib.reqHistoricalData,
                contract, '', duration, bar_size, 'TRADES', 0, 1, False, []
            )

            if bars:
                latest_bar = bars[-1]
                return {
                    'symbol': symbol,
                    'timestamp': latest_bar.date,
                    'open': latest_bar.open,
                    'high': latest_bar.high,
                    'low': latest_bar.low,
                    'close': latest_bar.close,
                    'volume': latest_bar.volume
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    async def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """
        Cancel an open order

        Args:
            order_id: IBKR order ID to cancel

        Returns:
            Dict with cancellation status
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return {'error': 'Not connected to IBKR'}

            # Cancel order with timeout
            await self._run_in_executor(self.ib.cancelOrder, order_id, timeout=10.0)

            return {
                'success': True,
                'order_id': order_id,
                'action': 'cancelled',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {'error': str(e), 'order_id': order_id}

    async def modify_order(self, order_id: int, quantity: Optional[int] = None,
                          price: Optional[float] = None) -> Dict[str, Any]:
        """
        Modify an existing order

        Args:
            order_id: IBKR order ID to modify
            quantity: New quantity (optional)
            price: New price for limit orders (optional)

        Returns:
            Dict with modification status
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return {'error': 'Not connected to IBKR'}

            # Get existing order
            order = None
            for o in self.ib.openOrders():
                if o.orderId == order_id:
                    order = o
                    break

            if not order:
                return {'error': f'Order {order_id} not found', 'order_id': order_id}

            # Modify order parameters
            modified = False
            if quantity is not None:
                order.totalQuantity = abs(quantity)
                modified = True
            if price is not None and order.orderType == 'LMT':
                order.lmtPrice = price
                modified = True

            if modified:
                # Place modified order
                await self._run_in_executor(self.ib.placeOrder, order.contract, order, timeout=10.0)

                return {
                    'success': True,
                    'order_id': order_id,
                    'action': 'modified',
                    'new_quantity': quantity,
                    'new_price': price,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            else:
                return {'error': 'No modifications specified', 'order_id': order_id}

        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            return {'error': str(e), 'order_id': order_id}

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders

        Returns:
            List of open order dictionaries
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return []

            orders = await self._run_in_executor(self.ib.openOrders)

            open_orders = []
            for order in orders:
                open_orders.append({
                    'order_id': order.orderId,
                    'symbol': order.contract.symbol,
                    'action': order.action,
                    'quantity': order.totalQuantity,
                    'order_type': order.orderType,
                    'status': order.orderStatus.status if hasattr(order, 'orderStatus') else 'Unknown',
                    'filled_quantity': order.orderStatus.filled if hasattr(order, 'orderStatus') else 0,
                    'remaining_quantity': order.orderStatus.remaining if hasattr(order, 'orderStatus') else order.totalQuantity,
                    'avg_fill_price': order.orderStatus.avgFillPrice if hasattr(order, 'orderStatus') else 0,
                    'lmt_price': getattr(order, 'lmtPrice', None),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

            return open_orders

        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    async def get_order_status(self, order_id: int) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific order

        Args:
            order_id: IBKR order ID

        Returns:
            Dict with order status or None if not found
        """
        open_orders = await self.get_open_orders()
        for order in open_orders:
            if order['order_id'] == order_id:
                return order

        # Check if order was filled and moved to executions
        try:
            executions = await self._run_in_executor(self.ib.executions)
            for exec in executions:
                if exec.orderId == order_id:
                    return {
                        'order_id': order_id,
                        'symbol': exec.contract.symbol,
                        'action': exec.side,
                        'quantity': exec.shares,
                        'price': exec.price,
                        'avg_price': exec.avgPrice,
                        'status': 'Filled',
                        'execution_time': exec.time,
                        'commission': exec.commission
                    }
        except Exception as e:
            logger.warning(f"Error checking executions for order {order_id}: {e}")

        return None

    async def place_bracket_order(self, symbol: str, quantity: int, entry_price: float,
                                stop_loss_pct: float = 0.05, take_profit_pct: float = 0.10) -> Dict[str, Any]:
        """
        Place a bracket order (entry + stop loss + take profit)

        Args:
            symbol: Stock symbol
            quantity: Order quantity
            entry_price: Entry price
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage

        Returns:
            Dict with bracket order details
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return {'error': 'Not connected to IBKR'}

            contract = Stock(symbol, 'SMART', 'USD')

            # Calculate stop and target prices
            stop_price = entry_price * (1 - stop_loss_pct)
            target_price = entry_price * (1 + take_profit_pct)

            # Create bracket order
            parent_order = Order()
            parent_order.action = 'BUY'
            parent_order.totalQuantity = quantity
            parent_order.orderType = 'LMT'
            parent_order.lmtPrice = entry_price
            parent_order.transmit = False  # Don't transmit yet

            # Stop loss order
            stop_order = Order()
            stop_order.action = 'SELL'
            stop_order.totalQuantity = quantity
            stop_order.orderType = 'STP'
            stop_order.auxPrice = stop_price
            stop_order.parentId = 0  # Will be set after parent order
            stop_order.transmit = False

            # Take profit order
            target_order = Order()
            target_order.action = 'SELL'
            target_order.totalQuantity = quantity
            target_order.orderType = 'LMT'
            target_order.lmtPrice = target_price
            target_order.parentId = 0  # Will be set after parent order
            target_order.transmit = True  # Transmit the bracket

            # Place orders
            parent_trade = await self._run_in_executor(self.ib.placeOrder, contract, parent_order)
            parent_id = parent_trade.order.orderId

            # Set parent IDs and place child orders
            stop_order.parentId = parent_id
            target_order.parentId = parent_id

            await self._run_in_executor(self.ib.placeOrder, contract, stop_order)
            await self._run_in_executor(self.ib.placeOrder, contract, target_order)

            return {
                'success': True,
                'bracket_order_id': parent_id,
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': entry_price,
                'stop_loss_price': stop_price,
                'take_profit_price': target_price,
                'orders': ['entry', 'stop_loss', 'take_profit'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error placing bracket order for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    async def get_portfolio_pnl(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio P&L information

        Returns:
            Dict with portfolio P&L details
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return {'error': 'Not connected to IBKR'}

            # Get account summary for total values
            summary = await self._run_in_executor(self.ib.reqAccountSummary)

            # Get positions
            positions = await self.get_positions()

            # Calculate P&L
            total_unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
            total_realized_pnl = sum(p.get('realized_pnl', 0) for p in positions)

            # Get cash balance
            cash_balance = 0.0
            total_value = 0.0
            for item in summary:
                if item.tag == 'TotalCashBalance':
                    cash_balance = float(item.value)
                elif item.tag == 'NetLiquidation':
                    total_value = float(item.value)

            return {
                'cash_balance': cash_balance,
                'total_portfolio_value': total_value,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': total_realized_pnl,
                'total_pnl': total_unrealized_pnl + total_realized_pnl,
                'positions_count': len(positions),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting portfolio P&L: {e}")
            return {'error': str(e)}

    async def get_news_bulletins(self, all_messages: bool = True) -> List[Dict[str, Any]]:
        """
        Get news bulletins from IBKR.
        Requires market data subscription for full access.

        Args:
            all_messages: Whether to get all messages or just new ones

        Returns:
            List of news bulletin dictionaries
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return []

            # Request news bulletins
            bulletins = await self._run_in_executor(self.ib.reqNewsBulletins, all_messages)

            news_bulletins = []
            for bulletin in bulletins:
                news_bulletins.append({
                    'id': bulletin.id,
                    'message': bulletin.message,
                    'exchange': bulletin.exchange,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'type': 'bulletin'
                })

            logger.info(f"Retrieved {len(news_bulletins)} news bulletins from IBKR")
            return news_bulletins

        except Exception as e:
            logger.error(f"Error getting news bulletins: {e}")
            return []

    async def get_historical_news(self, contract_id: int, provider_codes: str = "",
                                 start_date: str = "", end_date: str = "",
                                 total_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get historical news for a specific contract.
        Requires market data subscription.

        Args:
            contract_id: IBKR contract ID
            provider_codes: News provider codes (comma-separated)
            start_date: Start date (YYYYMMDD format)
            end_date: End date (YYYYMMDD format)
            total_results: Maximum number of results

        Returns:
            List of historical news items
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return []

            # Request historical news
            news_items = await self._run_in_executor(
                self.ib.reqHistoricalNews,
                contract_id, provider_codes, start_date, end_date, total_results
            )

            historical_news = []
            for news_item in news_items:
                historical_news.append({
                    'time': news_item.time,
                    'provider_code': news_item.providerCode,
                    'article_id': news_item.articleId,
                    'headline': news_item.headline,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'type': 'historical_news'
                })

            logger.info(f"Retrieved {len(historical_news)} historical news items from IBKR")
            return historical_news

        except Exception as e:
            logger.error(f"Error getting historical news: {e}")
            return []

    async def get_news_article(self, provider_code: str, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific news article content.
        Requires market data subscription.

        Args:
            provider_code: News provider code
            article_id: Article ID

        Returns:
            News article content or None if not found
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return None

            # Request news article
            article = await self._run_in_executor(
                self.ib.reqNewsArticle,
                provider_code, article_id
            )

            if article:
                return {
                    'article_type': article.articleType,
                    'article_text': article.articleText,
                    'provider_code': provider_code,
                    'article_id': article_id,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'type': 'news_article'
                }

            return None

        except Exception as e:
            logger.error(f"Error getting news article {article_id}: {e}")
            return None

    def setup_news_event_handlers(self):
        """
        Set up event handlers for real-time news feeds.
        Call this after connecting to enable real-time news.
        """
        try:
            # Set up news bulletin event handler
            @self.ib.newsBulletinsEvent
            def on_news_bulletin(bulletin):
                logger.info(f"NEWS BULLETIN: {bulletin.message} (Exchange: {bulletin.exchange})")
                # In production, would forward to strategy agents for analysis

            # Set up news article event handler
            @self.ib.newsArticleEvent
            def on_news_article(article_id, article_type, extra_data):
                logger.info(f"NEWS ARTICLE: {article_id} - {article_type}")
                # In production, would fetch and analyze article content

            logger.info("IBKR news event handlers configured")

        except Exception as e:
            logger.error(f"Error setting up news event handlers: {e}")

    async def get_account_permissions(self) -> Dict[str, Any]:
        """
        Get detailed account permissions and trading restrictions from IBKR.

        Returns:
            Dict with account type, trading permissions, and restrictions
        """
        try:
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return {'error': 'Not connected to IBKR'}

            # Get account summary for detailed account information
            summary = await self._run_in_executor(self.ib.reqAccountSummary)

            # Parse account features and permissions
            account_features = {}
            trading_permissions = {
                'equities': {'enabled': False, 'exchanges': [], 'restrictions': []},
                'options': {'enabled': False, 'types': [], 'restrictions': []},
                'futures': {'enabled': False, 'exchanges': [], 'restrictions': []},
                'forex': {'enabled': False, 'pairs': [], 'restrictions': []},
                'crypto': {'enabled': False, 'exchanges': [], 'restrictions': []}
            }

            # Parse account summary tags for permissions
            for item in summary:
                tag = item.tag
                value = item.value

                # Account type identification
                if tag == 'AccountType':
                    account_features['account_type'] = value
                elif tag == 'AccountCode':
                    account_features['account_code'] = value

                # Trading permissions
                elif tag == 'Cushion':
                    account_features['cushion'] = float(value)  # Buying power cushion
                elif tag == 'LookAheadNextChange':
                    account_features['next_reset'] = value  # Pattern day trading reset

                # Margin information
                elif tag == 'AvailableFunds':
                    account_features['available_funds'] = float(value)
                elif tag == 'BuyingPower':
                    account_features['buying_power'] = float(value)
                elif tag == 'EquityWithLoanValue':
                    account_features['equity_with_loan'] = float(value)

                # Pattern day trading status
                elif tag == 'DayTradesRemaining':
                    account_features['day_trades_remaining'] = int(float(value))

            # Determine account type and permissions based on IBKR data
            account_type = self._determine_account_type(account_features)

            # Get trading permissions based on account type and features
            permissions = self._get_trading_permissions(account_type, account_features)

            return {
                'account_id': self.account_id,
                'account_type': account_type,
                'account_features': account_features,
                'trading_permissions': permissions,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting account permissions: {e}")
            return {'error': str(e)}

    def _determine_account_type(self, account_features: Dict[str, Any]) -> str:
        """
        Determine account type from IBKR account features.

        Args:
            account_features: Account features from IBKR

        Returns:
            Account type string
        """
        account_type = account_features.get('account_type', '')
        account_code = account_features.get('account_code', '')

        # Check for paper trading account
        if self.account_id.startswith('D') and len(self.account_id) == 9:
            return 'paper_trading'

        # Map IBKR account types to our categories
        type_mapping = {
            'INDIVIDUAL': 'individual_margin',
            'CASH': 'individual_cash',
            'IRA': 'ira',
            'TRUST': 'individual_margin',
            'LLC': 'individual_margin',
            'PARTNERSHIP': 'institutional',
            'CORPORATION': 'institutional',
            'INSTITUTIONAL': 'institutional'
        }

        # Try to determine from account type
        if account_type in type_mapping:
            return type_mapping[account_type]

        # Fallback based on account code patterns
        if 'IRA' in account_code.upper():
            return 'ira'
        elif any(term in account_code.upper() for term in ['LLC', 'INC', 'CORP', 'LTD']):
            return 'institutional'

        # Default to individual cash account
        return 'individual_cash'

    def _get_trading_permissions(self, account_type: str, account_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get trading permissions based on account type and features.

        Args:
            account_type: Determined account type
            account_features: Account features from IBKR

        Returns:
            Dict with detailed trading permissions
        """
        # Base permissions by account type
        base_permissions = {
            'paper_trading': {
                'equities': {'enabled': True, 'exchanges': ['NASDAQ', 'NYSE', 'AMEX', 'ARCA']},
                'options': {'enabled': True, 'types': ['CALL', 'PUT', 'SPREAD', 'STRADDLE', 'STRANGLE']},
                'futures': {'enabled': True, 'exchanges': ['CME', 'CBOT', 'NYMEX', 'COMEX']},
                'forex': {'enabled': True, 'pairs': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD']},
                'crypto': {'enabled': False},
                'margin': {'enabled': True, 'leverage_max': 4.0},
                'short_selling': {'enabled': True}
            },
            'individual_cash': {
                'equities': {'enabled': True, 'exchanges': ['NASDAQ', 'NYSE', 'AMEX', 'ARCA']},
                'options': {'enabled': True, 'types': ['CALL', 'PUT']},
                'futures': {'enabled': False},
                'forex': {'enabled': False},
                'crypto': {'enabled': False},
                'margin': {'enabled': False},
                'short_selling': {'enabled': False}
            },
            'individual_margin': {
                'equities': {'enabled': True, 'exchanges': ['NASDAQ', 'NYSE', 'AMEX', 'ARCA']},
                'options': {'enabled': True, 'types': ['CALL', 'PUT', 'SPREAD', 'STRADDLE', 'STRANGLE', 'IRON_CONDOR']},
                'futures': {'enabled': False},
                'forex': {'enabled': True, 'pairs': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD']},
                'crypto': {'enabled': False},
                'margin': {'enabled': True, 'leverage_max': 4.0},
                'short_selling': {'enabled': True}
            },
            'ira': {
                'equities': {'enabled': True, 'exchanges': ['NASDAQ', 'NYSE', 'AMEX', 'ARCA']},
                'options': {'enabled': False},
                'futures': {'enabled': False},
                'forex': {'enabled': False},
                'crypto': {'enabled': False},
                'margin': {'enabled': False},
                'short_selling': {'enabled': False}
            },
            'institutional': {
                'equities': {'enabled': True, 'exchanges': ['NASDAQ', 'NYSE', 'AMEX', 'ARCA', 'OTC']},
                'options': {'enabled': True, 'types': ['ALL']},
                'futures': {'enabled': True, 'exchanges': ['CME', 'CBOT', 'NYMEX', 'COMEX', 'ICE']},
                'forex': {'enabled': True, 'pairs': ['ALL']},
                'crypto': {'enabled': True},
                'margin': {'enabled': True, 'leverage_max': 10.0},
                'short_selling': {'enabled': True}
            }
        }

        # Get base permissions for account type
        permissions = base_permissions.get(account_type, base_permissions['individual_cash']).copy()

        # Apply dynamic restrictions based on account features
        day_trades_remaining = account_features.get('day_trades_remaining', -1)
        if day_trades_remaining == 0:
            permissions['pattern_day_trading_restricted'] = True
            permissions['equities']['restrictions'].append('Pattern day trading limit reached')

        # Check buying power for margin accounts
        buying_power = account_features.get('buying_power', 0)
        if buying_power <= 0 and permissions.get('margin', {}).get('enabled'):
            permissions['margin']['restricted'] = True
            permissions['margin']['restrictions'] = ['Insufficient buying power']

        return permissions

    async def get_trading_permissions_config(self) -> Dict[str, Any]:
        """
        Get combined trading permissions from YAML config and IBKR account data.

        Returns:
            Dict with complete trading permissions configuration
        """
        try:
            import yaml
            import os

            # Load static configuration
            config_path = 'config/trading-permissions.yaml'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    static_config = yaml.safe_load(f)
            else:
                logger.warning(f"Trading permissions config not found: {config_path}")
                static_config = {}

            # Get dynamic account permissions
            dynamic_permissions = await self.get_account_permissions()
            if 'error' in dynamic_permissions:
                logger.warning(f"Could not get dynamic permissions: {dynamic_permissions['error']}")
                dynamic_permissions = {}

            # Combine static and dynamic permissions
            account_type = dynamic_permissions.get('account_type', 'individual_cash')
            static_account_config = static_config.get('account_types', {}).get(account_type, {})

            # Merge permissions (dynamic overrides static where available)
            combined_permissions = static_account_config.get('permissions', {}).copy()

            if 'trading_permissions' in dynamic_permissions:
                # Update with dynamic permissions
                dynamic_trading = dynamic_permissions['trading_permissions']
                for asset_class, permissions in dynamic_trading.items():
                    if asset_class in combined_permissions:
                        combined_permissions[asset_class].update(permissions)
                    else:
                        combined_permissions[asset_class] = permissions

            # Add account features and metadata
            result = {
                'account_id': self.account_id,
                'account_type': account_type,
                'static_config': static_account_config,
                'dynamic_permissions': dynamic_permissions,
                'combined_permissions': combined_permissions,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            logger.info(f"Trading permissions loaded for account {self.account_id} (type: {account_type})")
            return result

        except Exception as e:
            logger.error(f"Error loading trading permissions config: {e}")
            return {'error': str(e)}

    async def can_trade_instrument(self, symbol: str, instrument_type: str = 'equity',
                                  exchange: str = None) -> Dict[str, Any]:
        """
        Check if the account can trade a specific instrument.

        Args:
            symbol: Trading symbol
            instrument_type: Type of instrument ('equity', 'option', 'future', 'forex', 'crypto')
            exchange: Exchange where instrument trades

        Returns:
            Dict with tradability status and restrictions
        """
        try:
            permissions = await self.get_trading_permissions_config()
            if 'error' in permissions:
                return {'can_trade': False, 'error': permissions['error']}

            combined_permissions = permissions.get('combined_permissions', {})

            # Check if instrument type is enabled
            if instrument_type not in combined_permissions:
                return {
                    'can_trade': False,
                    'reason': f'Instrument type {instrument_type} not supported',
                    'instrument_type': instrument_type
                }

            type_permissions = combined_permissions[instrument_type]

            if not type_permissions.get('enabled', False):
                return {
                    'can_trade': False,
                    'reason': f'{instrument_type} trading not enabled for this account',
                    'instrument_type': instrument_type,
                    'restrictions': type_permissions.get('restrictions', [])
                }

            # Check exchange permissions
            if exchange and 'exchanges' in type_permissions:
                allowed_exchanges = type_permissions['exchanges']
                if allowed_exchanges and exchange not in allowed_exchanges and 'ALL' not in allowed_exchanges:
                    return {
                        'can_trade': False,
                        'reason': f'Exchange {exchange} not allowed for {instrument_type}',
                        'instrument_type': instrument_type,
                        'exchange': exchange,
                        'allowed_exchanges': allowed_exchanges
                    }

            # Check for any restrictions
            restrictions = type_permissions.get('restrictions', [])
            if restrictions:
                return {
                    'can_trade': True,
                    'restricted': True,
                    'restrictions': restrictions,
                    'instrument_type': instrument_type,
                    'exchange': exchange
                }

            return {
                'can_trade': True,
                'instrument_type': instrument_type,
                'exchange': exchange
            }

        except Exception as e:
            logger.error(f"Error checking tradability for {symbol}: {e}")
            return {'can_trade': False, 'error': str(e)}

# Global connector instance - now thread-safe
def get_ibkr_connector(config_path: str = 'config/ibkr_config.ini') -> IBKRConnector:
    """Get singleton IBKR connector instance (thread-safe)"""
    return IBKRConnector(config_path)

# Test function
async def test_connection():
    """Test IBKR paper trading connection"""
    connector = get_ibkr_connector()

    print("Testing IBKR Paper Trading Connection...")
    print("=" * 50)

    # Connect
    connected = await connector.connect()
    if not connected:
        print("❌ Failed to connect to IBKR Paper Trading")
        print("Make sure:")
        print("1. IBKR Trader Workstation (TWS) is running")
        print("2. Paper trading account is enabled")
        print("3. API connections are enabled in TWS")
        return

    print("✅ Connected to IBKR Paper Trading")

    # Get account summary
    account = await connector.get_account_summary()
    if 'error' in account:
        print(f"❌ Error getting account summary: {account['error']}")
    else:
        print(f"📊 Account: {account['account_id']}")
        print(f"💰 Cash Balance: ${account['cash_balance']:,.2f}")
        print(f"📈 Positions: {account['total_positions']}")

    # Test market data
    market_data = await connector.get_market_data('SPY')
    if market_data:
        print(f"📈 SPY Latest: ${market_data['close']:.2f}")
    else:
        print("⚠️  Could not get market data")

    # Disconnect
    await connector.disconnect()
    print("✅ Disconnected from IBKR")

if __name__ == "__main__":
    asyncio.run(test_connection())
