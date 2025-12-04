# [LABEL:TEST:fixtures] [LABEL:FRAMEWORK:ibkr] [LABEL:INTEGRATION:fixtures]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-04] [LABEL:REVIEWED:pending]
#
# Purpose: IBKR-specific test fixtures with proper setup and cleanup
# Dependencies: pytest, pytest-asyncio, ib_insync
# Related: tests/integration/test_ibkr_*.py, src/integrations/ibkr_connector.py

import pytest
import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_ibkr_connector():
    """Mock IBKR connector for unit testing."""
    mock_connector = MagicMock()

    # Mock connection methods
    mock_connector.connect = AsyncMock(return_value=True)
    mock_connector.disconnect = AsyncMock(return_value=True)
    mock_connector.is_connected = MagicMock(return_value=True)

    # Mock trading methods
    mock_connector.place_order = AsyncMock(return_value={'order_id': '12345', 'status': 'submitted'})
    mock_connector.cancel_order = AsyncMock(return_value=True)
    mock_connector.get_order_status = AsyncMock(return_value={'status': 'filled'})

    # Mock market data methods
    mock_connector.get_market_data = AsyncMock(return_value={
        'symbol': 'AAPL',
        'bid': 150.00,
        'ask': 150.10,
        'last': 150.05,
        'volume': 1000000,
        'timestamp': '2025-12-04T10:30:00Z'
    })

    # Mock account methods
    mock_connector.get_account_summary = AsyncMock(return_value={
        'cash': 50000.00,
        'total_value': 75000.00,
        'buying_power': 100000.00,
        'maintenance_margin': 25000.00
    })

    mock_connector.get_positions = AsyncMock(return_value=[
        {'symbol': 'AAPL', 'quantity': 100, 'avg_cost': 145.00, 'current_price': 150.00},
        {'symbol': 'GOOGL', 'quantity': 50, 'avg_cost': 130.00, 'current_price': 135.00}
    ])

    return mock_connector

@pytest.fixture
def mock_ib_insync():
    """Mock ib_insync components for testing."""
    # Mock IB instance
    mock_ib = MagicMock()
    mock_ib.connect = MagicMock(return_value=True)
    mock_ib.disconnect = MagicMock(return_value=True)
    mock_ib.isConnected = MagicMock(return_value=True)

    # Mock contract
    mock_contract = MagicMock()
    mock_contract.symbol = 'AAPL'
    mock_contract.secType = 'STK'
    mock_contract.exchange = 'SMART'

    # Mock order
    mock_order = MagicMock()
    mock_order.orderId = 12345
    mock_order.status = 'Submitted'

    # Mock trade
    mock_trade = MagicMock()
    mock_trade.order = mock_order
    mock_trade.contract = mock_contract

    return {
        'ib': mock_ib,
        'contract': mock_contract,
        'order': mock_order,
        'trade': mock_trade
    }

@pytest.fixture
async def ibkr_test_connector():
    """Real IBKR connector fixture with proper cleanup for integration tests."""
    from src.integrations.ibkr_connector import IBKRConnector

    connector = None
    try:
        connector = IBKRConnector()
        logger.info("IBKR test connector initialized")

        # Optional: attempt connection if TWS is available
        try:
            connected = await asyncio.wait_for(connector.connect(), timeout=5.0)
            if connected:
                logger.info("IBKR test connector connected")
            else:
                logger.warning("IBKR test connector not connected (TWS may not be running)")
        except asyncio.TimeoutError:
            logger.warning("IBKR connection timeout - continuing with disconnected state")
        except Exception as e:
            logger.warning(f"IBKR connection failed: {e} - continuing with mock state")

        yield connector

    finally:
        # Cleanup
        if connector:
            try:
                await connector.disconnect()
                logger.info("IBKR test connector disconnected")
            except Exception as e:
                logger.warning(f"IBKR connector cleanup failed: {e}")

@pytest.fixture
def sample_ibkr_order():
    """Sample IBKR order for testing."""
    return {
        'symbol': 'AAPL',
        'quantity': 100,
        'action': 'BUY',
        'order_type': 'LMT',
        'limit_price': 150.00,
        'time_in_force': 'DAY',
        'account': 'DU1234567'  # Paper trading account
    }

@pytest.fixture
def sample_ibkr_contract():
    """Sample IBKR contract for testing."""
    return {
        'symbol': 'AAPL',
        'secType': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'conId': 265598  # AAPL contract ID
    }

@pytest.fixture
def mock_tws_connection():
    """Mock TWS connection for testing."""
    import socket
    from unittest.mock import patch

    # Mock successful connection
    mock_socket = MagicMock()
    mock_socket.connect = MagicMock(return_value=None)
    mock_socket.close = MagicMock(return_value=None)

    with patch('socket.create_connection', return_value=mock_socket) as mock_create:
        yield mock_create

@pytest.fixture
async def ibkr_test_environment(ibkr_test_connector, mock_tws_connection):
    """Complete IBKR test environment with all components."""
    env = {
        'connector': ibkr_test_connector,
        'tws_available': True,
        'paper_trading': True,
        'test_account': 'DU1234567'
    }

    logger.info("IBKR test environment setup complete")
    yield env

    # Environment cleanup
    logger.info("IBKR test environment cleanup complete")

@pytest.fixture
def ibkr_error_scenarios():
    """Common IBKR error scenarios for testing."""
    return {
        'connection_timeout': {
            'error': 'TimeoutError',
            'message': 'IBKR connection timeout',
            'retry_count': 3
        },
        'invalid_contract': {
            'error': 'ContractError',
            'message': 'Invalid contract specification',
            'contract': {'symbol': 'INVALID', 'secType': 'STK'}
        },
        'insufficient_funds': {
            'error': 'InsufficientFundsError',
            'message': 'Account has insufficient funds',
            'required': 15000.00,
            'available': 5000.00
        },
        'market_closed': {
            'error': 'MarketClosedError',
            'message': 'Market is closed',
            'next_open': '2025-12-05T09:30:00Z'
        },
        'circuit_breaker': {
            'error': 'CircuitBreakerError',
            'message': 'Trading halted due to circuit breaker',
            'resume_time': '2025-12-04T10:45:00Z'
        }
    }

@pytest.fixture
async def ibkr_cleanup_manager():
    """Manager for IBKR test cleanup operations."""
    cleanup_tasks = []

    def add_cleanup_task(task):
        """Add a cleanup task to be executed."""
        cleanup_tasks.append(task)

    async def execute_cleanup():
        """Execute all cleanup tasks."""
        logger.info(f"Executing {len(cleanup_tasks)} IBKR cleanup tasks")
        for task in reversed(cleanup_tasks):  # Reverse order for proper cleanup
            try:
                if asyncio.iscoroutinefunction(task):
                    await task()
                else:
                    task()
            except Exception as e:
                logger.warning(f"Cleanup task failed: {e}")

    yield {
        'add_task': add_cleanup_task,
        'cleanup': execute_cleanup
    }

    # Automatic cleanup on fixture teardown
    await execute_cleanup()