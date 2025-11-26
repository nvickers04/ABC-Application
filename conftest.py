import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import aiohttp

# Async fixtures for common test dependencies
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