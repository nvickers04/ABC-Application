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
from unittest.mock import AsyncMock, MagicMock
import aiohttp

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

# Additional fixtures from root conftest.py
@pytest.fixture
def data_agent():
    """Mock data agent for testing."""
    agent = AsyncMock()
    agent.process_data = AsyncMock(return_value={"status": "success", "data": {"articles_df": []}})
    agent.process_input = AsyncMock(return_value={"status": "success"})
    # Add mock subagents
    agent.yfinance_sub = AsyncMock()
    agent.yfinance_sub.process_input = AsyncMock(return_value={"data": "mocked"})
    agent.sentiment_sub = AsyncMock()
    agent.sentiment_sub.process_input = AsyncMock(return_value={"sentiment": "positive"})
    agent.news_sub = AsyncMock()
    agent.news_sub.process_input = AsyncMock(return_value={"news": []})
    agent.economic_sub = AsyncMock()
    agent.economic_sub.process_input = AsyncMock(return_value={"gdp": 0.02})
    agent.institutional_sub = AsyncMock()
    agent.institutional_sub.process_input = AsyncMock(return_value={"institutional": {}})
    agent.fundamental_sub = AsyncMock()
    agent.fundamental_sub.process_input = AsyncMock(return_value={"fundamentals": {}})
    agent.microstructure_sub = AsyncMock()
    agent.microstructure_sub.process_input = AsyncMock(return_value={"microstructure": {}})
    agent.kalshi_sub = AsyncMock()
    agent.kalshi_sub.process_input = AsyncMock(return_value={"kalshi": {}})
    agent.marketdataapp_sub = AsyncMock()
    agent.marketdataapp_sub.process_input = AsyncMock(return_value={"marketdataapp": {}})
    yield agent

@pytest.fixture
async def system_setup():
    """Mock system setup for integration tests."""
    setup = AsyncMock()
    setup.initialize = AsyncMock(return_value=True)
    setup.cleanup = AsyncMock(return_value=True)
    yield setup

# Mock aiohttp session to prevent unclosed session warnings
@pytest.fixture
async def mock_session():
    session = AsyncMock(spec=aiohttp.ClientSession)
    session.get = AsyncMock()
    session.post = AsyncMock()
    session.close = AsyncMock()
    yield session
