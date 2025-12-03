# [LABEL:TEST:data] [LABEL:FRAMEWORK:fixtures] [LABEL:DATA:mocks]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-02] [LABEL:REVIEWED:pending]
#
# Purpose: Standardized mock agent fixtures for testing
# Dependencies: pytest, unittest.mock
# Related: tests/fixtures/*.py, src/agents/*.py

import pytest
from unittest.mock import MagicMock, AsyncMock
from decimal import Decimal

@pytest.fixture
def mock_execution_agent():
    """Fixture providing a mock execution agent."""
    agent = MagicMock()
    agent.execute_trade = AsyncMock(return_value={
        'success': True,
        'order_id': 'TEST_12345',
        'status': 'FILLED',
        'filled_quantity': 100,
        'avg_fill_price': Decimal('150.50'),
        'symbol': 'AAPL',
        'action': 'BUY'
    })
    agent.get_positions = AsyncMock(return_value=[])
    agent.get_account_balance = AsyncMock(return_value=Decimal('10000.00'))
    agent.cancel_order = AsyncMock(return_value=True)
    return agent

@pytest.fixture
def mock_data_analyzer():
    """Fixture providing a mock data analyzer."""
    analyzer = MagicMock()
    analyzer.analyze_market_data = AsyncMock(return_value={
        'trend': 'bullish',
        'strength': 0.75,
        'volatility': 0.25,
        'momentum': 0.60,
        'support_levels': [145.00, 148.50, 152.00],
        'resistance_levels': [155.00, 158.50, 162.00],
    })
    analyzer.get_fundamental_data = AsyncMock(return_value={
        'pe_ratio': 25.5,
        'eps': 7.50,
        'revenue_growth': 0.15,
        'profit_margin': 0.22,
        'debt_to_equity': 0.45,
    })
    return analyzer

@pytest.fixture
def mock_risk_manager():
    """Fixture providing a mock risk manager."""
    manager = MagicMock()
    manager.assess_risk = AsyncMock(return_value={
        'risk_score': 0.3,
        'max_position_size': 500,
        'acceptable': True,
        'warnings': [],
    })
    manager.check_portfolio_risk = AsyncMock(return_value={
        'total_risk': 0.05,
        'var_95': Decimal('250.00'),
        'max_drawdown': 0.08,
        'within_limits': True,
    })
    return manager

@pytest.fixture
def mock_memory_agent():
    """Fixture providing a mock memory agent."""
    agent = MagicMock()
    agent.store_memory = AsyncMock(return_value=True)
    agent.retrieve_memory = AsyncMock(return_value={
        'similar_trades': [
            {'symbol': 'AAPL', 'action': 'BUY', 'profit': 150.00, 'date': '2024-01-01'},
            {'symbol': 'GOOGL', 'action': 'SELL', 'profit': -50.00, 'date': '2024-01-02'},
        ],
        'market_conditions': {
            'volatility': 'moderate',
            'trend': 'upward',
            'liquidity': 'good',
        }
    })
    agent.get_patterns = AsyncMock(return_value=[
        {'pattern': 'breakout', 'confidence': 0.85, 'direction': 'bullish'},
        {'pattern': 'volume_surge', 'confidence': 0.70, 'direction': 'bullish'},
    ])
    return agent

@pytest.fixture
def mock_alert_manager():
    """Fixture providing a mock alert manager."""
    manager = MagicMock()
    manager.send_alert = AsyncMock(return_value=True)
    manager.error = AsyncMock(return_value=None)
    manager.warning = AsyncMock(return_value=None)
    manager.info = AsyncMock(return_value=None)
    return manager

@pytest.fixture
def mock_discord_bot():
    """Fixture providing a mock Discord bot."""
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=True)
    bot.get_channel = MagicMock(return_value=MagicMock())
    bot.is_ready = MagicMock(return_value=True)
    return bot

@pytest.fixture
def mock_redis_client():
    """Fixture providing a mock Redis client."""
    client = MagicMock()
    client.get = MagicMock(return_value=b'{"test": "data"}')
    client.set = MagicMock(return_value=True)
    client.delete = MagicMock(return_value=1)
    client.exists = MagicMock(return_value=1)
    client.expire = MagicMock(return_value=True)
    return client

@pytest.fixture
def mock_ibkr_connector():
    """Fixture providing a mock IBKR connector."""
    connector = MagicMock()
    connector.connect = AsyncMock(return_value=True)
    connector.disconnect = AsyncMock(return_value=True)
    connector.place_order = AsyncMock(return_value={
        'order_id': 'IBKR_12345',
        'status': 'SUBMITTED',
        'symbol': 'AAPL',
        'quantity': 100,
        'action': 'BUY',
    })
    connector.cancel_order = AsyncMock(return_value=True)
    connector.get_positions = AsyncMock(return_value=[])
    connector.get_account_balance = AsyncMock(return_value=Decimal('50000.00'))
    connector.is_connected = MagicMock(return_value=True)
    return connector

@pytest.fixture
def mock_tigerbeetle_client():
    """Fixture providing a mock TigerBeetle client."""
    client = MagicMock()
    client.create_accounts = MagicMock(return_value=[])
    client.create_transfers = MagicMock(return_value=[])
    client.lookup_accounts = MagicMock(return_value=[])
    client.lookup_transfers = MagicMock(return_value=[])
    return client

@pytest.fixture
def mock_strategy_analyzer():
    """Fixture providing a mock strategy analyzer."""
    analyzer = MagicMock()
    analyzer.analyze_strategy = AsyncMock(return_value={
        'strategy_type': 'momentum',
        'confidence': 0.80,
        'expected_return': 0.12,
        'risk_level': 'moderate',
        'timeframe': '1d',
        'signals': [
            {'type': 'entry', 'strength': 0.85, 'price': 152.50},
            {'type': 'stop_loss', 'strength': 0.90, 'price': 148.00},
        ]
    })
    return analyzer

@pytest.fixture
def mock_orchestrator():
    """Fixture providing a mock orchestrator."""
    orchestrator = MagicMock()
    orchestrator.process_workflow = AsyncMock(return_value={
        'status': 'completed',
        'result': 'success',
        'execution_time': 2.5,
        'steps_completed': 5,
    })
    orchestrator.get_status = AsyncMock(return_value='idle')
    orchestrator.cancel_workflow = AsyncMock(return_value=True)
    return orchestrator