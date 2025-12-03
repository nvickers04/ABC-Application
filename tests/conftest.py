# [LABEL:TEST:config] [LABEL:FRAMEWORK:pytest] [LABEL:CONFIG:fixtures]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-02] [LABEL:REVIEWED:pending]
#
# Purpose: Centralized pytest configuration and fixtures for all test suites
# Dependencies: pytest, pytest-asyncio
# Related: tests/fixtures/*.py, unit-tests/conftest.py, integration-tests/conftest.py

import pytest
import asyncio
import sys
import os
from pathlib import Path

# Add src and tests to path for all tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'tests'))

# Import all fixture modules
from tests.fixtures.market_data import *
from tests.fixtures.trading_config import *
from tests.fixtures.mock_agents import *

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio

# Global test configuration
@pytest.fixture(scope="session", autouse=True)
def configure_test_env():
    """Configure test environment variables."""
    os.environ.setdefault('TESTING', 'true')
    os.environ.setdefault('LOG_LEVEL', 'WARNING')
    os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/1')  # Test database
    yield
    # Cleanup after all tests
    pass

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir(tmp_path):
    """Fixture providing a temporary directory for tests."""
    return tmp_path

@pytest.fixture
def sample_trade_request():
    """Fixture providing a sample trade request."""
    return {
        'symbol': 'AAPL',
        'quantity': 100,
        'action': 'BUY',
        'order_type': 'market',
        'time_in_force': 'day',
    }

@pytest.fixture
def sample_portfolio():
    """Fixture providing a sample portfolio."""
    return {
        'cash': 50000.00,
        'positions': {
            'AAPL': {'quantity': 100, 'avg_cost': 150.50, 'current_price': 152.00},
            'GOOGL': {'quantity': 50, 'avg_cost': 135.00, 'current_price': 137.50},
        },
        'total_value': 66750.00,
    }

@pytest.fixture
def sample_api_response():
    """Fixture providing a sample API response."""
    return {
        'status': 'success',
        'data': {
            'symbol': 'AAPL',
            'price': 152.50,
            'volume': 1000000,
            'timestamp': '2024-01-01T10:30:00Z',
        },
        'metadata': {
            'source': 'test_api',
            'cached': False,
        }
    }

@pytest.fixture
def mock_external_api(mocker):
    """Fixture to mock external API calls comprehensively."""
    # Mock requests
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'status': 'success', 'data': {}}
    mock_response.text = '{"status": "success", "data": {}}'

    mocker.patch('requests.get', return_value=mock_response)
    mocker.patch('requests.post', return_value=mock_response)
    mocker.patch('requests.put', return_value=mock_response)
    mocker.patch('requests.delete', return_value=mock_response)

    return mock_response

@pytest.fixture
def mock_database(mocker):
    """Fixture to mock database operations."""
    mock_db = mocker.MagicMock()
    mock_db.connect.return_value = True
    mock_db.disconnect.return_value = True
    mock_db.execute.return_value = []
    mock_db.fetchone.return_value = None
    mock_db.fetchall.return_value = []
    mock_db.commit.return_value = None
    mock_db.rollback.return_value = None

    return mock_db

@pytest.fixture
def mock_file_system(mocker, tmp_path):
    """Fixture to mock file system operations."""
    # Create a temporary directory structure
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    # Create some test files
    (test_dir / "config.yaml").write_text("test: config")
    (test_dir / "data.json").write_text('{"test": "data"}')

    # Mock pathlib operations to use temp directory
    mocker.patch('pathlib.Path.cwd', return_value=test_dir)

    return test_dir

@pytest.fixture
def performance_timer():
    """Fixture providing a simple performance timer."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0

    return Timer()

# Error simulation fixtures
@pytest.fixture
def simulate_network_error(mocker):
    """Fixture to simulate network errors."""
    mocker.patch('requests.get', side_effect=ConnectionError("Network error"))
    mocker.patch('requests.post', side_effect=ConnectionError("Network error"))

@pytest.fixture
def simulate_api_rate_limit(mocker):
    """Fixture to simulate API rate limiting."""
    mock_response = mocker.MagicMock()
    mock_response.status_code = 429
    mock_response.json.return_value = {'error': 'Rate limit exceeded'}
    mocker.patch('requests.get', return_value=mock_response)
    mocker.patch('requests.post', return_value=mock_response)

@pytest.fixture
def simulate_database_error(mocker):
    """Fixture to simulate database errors."""
    mocker.patch('sqlite3.connect', side_effect=Exception("Database connection failed"))

# Test data generators
@pytest.fixture
def generate_random_trades():
    """Fixture providing a function to generate random trades."""
    import random
    from decimal import Decimal

    def _generate_trades(count=10, symbols=None):
        if symbols is None:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

        trades = []
        for _ in range(count):
            trade = {
                'symbol': random.choice(symbols),
                'quantity': random.randint(1, 1000),
                'price': Decimal(str(round(random.uniform(50, 500), 2))),
                'action': random.choice(['BUY', 'SELL']),
                'timestamp': '2024-01-01T10:30:00Z',
            }
            trades.append(trade)

        return trades

    return _generate_trades

@pytest.fixture
def generate_market_data_series():
    """Fixture providing a function to generate market data series."""
    import random
    from datetime import datetime, timedelta

    def _generate_series(symbol='AAPL', days=30, base_price=150.0):
        data = []
        current_price = base_price

        for i in range(days):
            # Random walk with slight upward trend
            change = random.uniform(-0.05, 0.06)  # -5% to +6%
            current_price *= (1 + change)

            # Ensure price stays reasonable
            current_price = max(50, min(1000, current_price))

            data_point = {
                'symbol': symbol,
                'date': (datetime.now() - timedelta(days=days-i)).strftime('%Y-%m-%d'),
                'open': round(current_price * random.uniform(0.98, 1.02), 2),
                'high': round(current_price * random.uniform(1.00, 1.05), 2),
                'low': round(current_price * random.uniform(0.95, 1.00), 2),
                'close': round(current_price, 2),
                'volume': random.randint(100000, 10000000),
            }
            data.append(data_point)

        return data

    return _generate_series