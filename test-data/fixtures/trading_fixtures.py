# [LABEL:TEST:data] [LABEL:FRAMEWORK:pytest] [LABEL:DATA:fixtures]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-02] [LABEL:REVIEWED:pending]
#
# Purpose: Standardized trading data fixtures for consistent testing
# Dependencies: pytest, pydantic
# Related: test-data/fixtures/, unit-tests/conftest.py

"""
Standardized test fixtures for trading system components.
Provides consistent, realistic test data across all test suites.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import MagicMock

# Sample trading symbols and market data
SAMPLE_SYMBOLS = {
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'exchange': 'NASDAQ'},
    'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'exchange': 'NASDAQ'},
    'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'exchange': 'NASDAQ'},
    'TSLA': {'name': 'Tesla Inc.', 'sector': 'Consumer Discretionary', 'exchange': 'NASDAQ'},
    'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'exchange': 'NASDAQ'},
}

# Sample market data
SAMPLE_MARKET_DATA = {
    'AAPL': {
        'price': Decimal('150.25'),
        'volume': 45238900,
        'bid': Decimal('150.20'),
        'ask': Decimal('150.30'),
        'last_update': datetime.now(),
        'volatility': Decimal('0.25'),
        'beta': Decimal('1.2'),
    },
    'GOOGL': {
        'price': Decimal('2800.50'),
        'volume': 1234567,
        'bid': Decimal('2800.00'),
        'ask': Decimal('2801.00'),
        'last_update': datetime.now(),
        'volatility': Decimal('0.30'),
        'beta': Decimal('1.1'),
    },
    'MSFT': {
        'price': Decimal('380.75'),
        'volume': 23456789,
        'bid': Decimal('380.70'),
        'ask': Decimal('380.80'),
        'last_update': datetime.now(),
        'volatility': Decimal('0.22'),
        'beta': Decimal('0.9'),
    },
}

# Sample trade orders
SAMPLE_ORDERS = [
    {
        'symbol': 'AAPL',
        'quantity': 100,
        'action': 'BUY',
        'order_type': 'MKT',
        'price': None,
        'account_id': 'TEST123',
        'order_id': 'ORD_001',
        'timestamp': datetime.now(),
    },
    {
        'symbol': 'GOOGL',
        'quantity': 50,
        'action': 'SELL',
        'order_type': 'LMT',
        'price': Decimal('2850.00'),
        'account_id': 'TEST123',
        'order_id': 'ORD_002',
        'timestamp': datetime.now() - timedelta(minutes=5),
    },
]

# Sample portfolio data
SAMPLE_PORTFOLIO = {
    'account_id': 'TEST123',
    'cash_balance': Decimal('100000.00'),
    'total_value': Decimal('150000.00'),
    'positions': [
        {
            'symbol': 'AAPL',
            'quantity': 200,
            'avg_cost': Decimal('145.50'),
            'current_price': Decimal('150.25'),
            'unrealized_pnl': Decimal('950.00'),
        },
        {
            'symbol': 'MSFT',
            'quantity': 150,
            'avg_cost': Decimal('375.00'),
            'current_price': Decimal('380.75'),
            'unrealized_pnl': Decimal('862.50'),
        },
    ],
}

# Sample risk parameters
SAMPLE_RISK_PARAMS = {
    'max_position_size': Decimal('0.10'),  # 10% of portfolio
    'max_daily_loss': Decimal('0.05'),    # 5% daily loss limit
    'max_drawdown': Decimal('0.15'),      # 15% max drawdown
    'volatility_target': Decimal('0.20'), # 20% annualized volatility target
    'sharpe_target': Decimal('1.5'),      # Minimum Sharpe ratio
}

@pytest.fixture
def sample_symbols():
    """Fixture providing sample trading symbols."""
    return SAMPLE_SYMBOLS.copy()

@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data."""
    return SAMPLE_MARKET_DATA.copy()

@pytest.fixture
def sample_orders():
    """Fixture providing sample trade orders."""
    return [order.copy() for order in SAMPLE_ORDERS]

@pytest.fixture
def sample_portfolio():
    """Fixture providing sample portfolio data."""
    return SAMPLE_PORTFOLIO.copy()

@pytest.fixture
def sample_risk_params():
    """Fixture providing sample risk parameters."""
    return SAMPLE_RISK_PARAMS.copy()

@pytest.fixture
def mock_ibkr_connector():
    """Mock IBKR connector for testing."""
    connector = MagicMock()
    connector.connect.return_value = True
    connector.is_connected.return_value = True
    connector.get_account_balance.return_value = SAMPLE_PORTFOLIO['cash_balance']
    connector.get_positions.return_value = SAMPLE_PORTFOLIO['positions']
    connector.place_order.return_value = {
        'success': True,
        'order_id': 'MOCK_ORDER_123',
        'status': 'SUBMITTED'
    }
    return connector

@pytest.fixture
def mock_tigerbeetle_client():
    """Mock TigerBeetle client for testing."""
    client = MagicMock()
    client.submit.return_value = [{'id': 12345, 'code': 0}]
    client.query_accounts.return_value = []
    client.query_transfers.return_value = []
    return client

@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    redis_client = MagicMock()
    redis_client.get.return_value = None
    redis_client.set.return_value = True
    redis_client.exists.return_value = False
    return redis_client

@pytest.fixture
def trading_execution_agent(mock_ibkr_connector, mock_tigerbeetle_client, mock_redis_client):
    """Fixture providing a configured trading execution agent."""
    from src.agents.execution import ExecutionAgent

    agent = ExecutionAgent(
        agent_id="test_execution_agent",
        ibkr_connector=mock_ibkr_connector,
        tb_client=mock_tigerbeetle_client,
        redis_client=mock_redis_client
    )
    return agent

@pytest.fixture
def sample_trade_scenario():
    """Comprehensive trade scenario for integration testing."""
    return {
        'initial_portfolio': SAMPLE_PORTFOLIO,
        'target_trades': [
            {
                'symbol': 'AAPL',
                'quantity': 50,
                'action': 'BUY',
                'expected_outcome': 'success',
            },
            {
                'symbol': 'GOOGL',
                'quantity': 25,
                'action': 'SELL',
                'expected_outcome': 'success',
            }
        ],
        'expected_final_portfolio': {
            'cash_balance': Decimal('85000.00'),  # After buying AAPL and selling GOOGL
            'positions': [
                {
                    'symbol': 'AAPL',
                    'quantity': 250,  # 200 + 50
                    'avg_cost': Decimal('146.70'),  # Weighted average
                },
                {
                    'symbol': 'MSFT',
                    'quantity': 150,
                    'avg_cost': Decimal('375.00'),
                },
            ],
        }
    }