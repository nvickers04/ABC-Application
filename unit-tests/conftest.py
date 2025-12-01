# [LABEL:TEST:config] [LABEL:FRAMEWORK:pytest] [LABEL:FRAMEWORK:pytest_asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Pytest configuration and fixtures for unit test suite
# Dependencies: pytest, pytest-asyncio
# Related: unit-tests/*.py, pytest.ini
#
import pytest
import asyncio
import sys
import os

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure pytest-asyncio
# pytestmark = pytest.mark.asyncio  # Removed to avoid forcing all tests async

from unittest.mock import MagicMock

@pytest.fixture
def mock_redis(mocker):
    """Fixture to mock Redis connections."""
    mocker.patch('redis.Redis', autospec=True)
    return mocker

@pytest.fixture
def mock_api_calls(monkeypatch):
    """Fixture to mock external API calls."""
    from unittest.mock import MagicMock
    mock_get = MagicMock(status_code=200, json=lambda: {})
    mock_post = MagicMock(status_code=200, json=lambda: {})
    monkeypatch.setattr('requests.get', mock_get)
    monkeypatch.setattr('requests.post', mock_post)
    return {'get': mock_get, 'post': mock_post}

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
