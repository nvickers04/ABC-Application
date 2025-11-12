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
                        connect_task = temp_ib.connectAsync(self.host, self.port, self.client_id)
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

            # Use threading with new event loop for IBKR operations
            import threading
            import asyncio as asyncio_module

            account_result = {'summary': None, 'positions': None, 'error': None}

            def get_account_in_thread():
                loop = asyncio_module.new_event_loop()
                asyncio_module.set_event_loop(loop)

                try:
                    # Get account summary and positions synchronously
                    summary = loop.run_until_complete(self.ib.reqAccountSummaryAsync())
                    positions = loop.run_until_complete(self.ib.reqPositionsAsync())

                    account_result['summary'] = summary
                    account_result['positions'] = positions

                except Exception as e:
                    account_result['error'] = str(e)
                finally:
                    loop.close()

            # Run account retrieval in thread
            account_thread = threading.Thread(target=get_account_in_thread)
            account_thread.start()
            account_thread.join(timeout=10)  # 10 second timeout

            if account_result['error']:
                logger.error(f"Error in account thread: {account_result['error']}")
                # Try to reconnect on error
                self.connected = False
                return {'error': account_result['error']}

            summary = account_result['summary']
            positions = account_result['positions']

            # Get cash balance
            cash_balance = 0.0
            for item in summary:
                if item.tag == 'TotalCashBalance':
                    cash_balance = float(item.value)

            account_info = {
                'account_id': self.account_id,
                'cash_balance': cash_balance,
                'currency': self.account_currency,
                'positions': [
                    {
                        'symbol': pos.contract.symbol,
                        'position': pos.position,
                        'avg_cost': pos.avgCost,
                        'market_value': pos.marketValue
                    } for pos in positions
                ],
                'total_positions': len(positions),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            logger.info(f"Account summary retrieved: ${cash_balance:.2f} cash, {len(positions)} positions")
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

            # Place order using threading with new event loop
            import threading
            import asyncio as asyncio_module
            import time

            order_result = {'trade': None, 'error': None}

            def place_order_in_thread():
                loop = asyncio_module.new_event_loop()
                asyncio_module.set_event_loop(loop)

                try:
                    trade = loop.run_until_complete(self.ib.placeOrderAsync(contract, order))
                    order_result['trade'] = trade
                except Exception as e:
                    order_result['error'] = str(e)
                finally:
                    loop.close()

            # Run order placement in thread
            order_thread = threading.Thread(target=place_order_in_thread)
            order_thread.start()
            order_thread.join(timeout=10)  # 10 second timeout

            if order_result['error']:
                logger.error(f"Error in order thread: {order_result['error']}")
                return {'error': order_result['error'], 'symbol': symbol, 'quantity': quantity, 'action': action}

            trade = order_result['trade']

            # Wait a bit for order to process
            time.sleep(1)

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

            # Use threading with new event loop for IBKR operations
            import threading
            import asyncio as asyncio_module

            positions_result = {'positions': None, 'error': None}

            def get_positions_in_thread():
                loop = asyncio_module.new_event_loop()
                asyncio_module.set_event_loop(loop)

                try:
                    positions = loop.run_until_complete(self.ib.reqPositionsAsync())
                    positions_result['positions'] = positions
                except Exception as e:
                    positions_result['error'] = str(e)
                finally:
                    loop.close()

            # Run positions retrieval in thread
            positions_thread = threading.Thread(target=get_positions_in_thread)
            positions_thread.start()
            positions_thread.join(timeout=10)  # 10 second timeout

            if positions_result['error']:
                logger.error(f"Error in positions thread: {positions_result['error']}")
                return []

            positions = positions_result['positions']

            position_list = []
            for pos in positions:
                position_list.append({
                    'symbol': pos.contract.symbol,
                    'position': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_value': pos.marketValue,
                    'unrealized_pnl': pos.unrealizedPNL,
                    'realized_pnl': pos.realizedPNL
                })

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
