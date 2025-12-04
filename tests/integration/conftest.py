import sys
import os
import pytest

# Add project root to sys.path for imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from unittest.mock import MagicMock

@pytest.fixture
def mock_redis(mocker):
    """Fixture to mock Redis connections."""
    mocker.patch('redis.Redis', autospec=True)
    return mocker

@pytest.fixture
def mock_api_calls(mocker):
    """Fixture to mock external API calls."""
    mocker.patch('requests.get', return_value=MagicMock(status_code=200, json=lambda: {}))
    mocker.patch('requests.post', return_value=MagicMock(status_code=200, json=lambda: {}))
    return mocker

@pytest.fixture(scope="session")
def check_tws():
    """Fixture to check if TWS is running for IBKR tests."""
    import socket
    try:
        s = socket.create_connection(("localhost", 7497), timeout=1)
        s.close()
        return True
    except:
        pytest.skip("TWS not running - skipping IBKR test")
