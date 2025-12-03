# integration-tests/test_nautilus_bridge.py
"""
Test script for NautilusIBKRBridge integration
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.integrations.nautilus_ibkr_bridge import (
    NautilusIBKRBridge,
    BridgeConfig,
    BridgeMode,
    get_nautilus_ibkr_bridge,
    initialize_bridge,
    get_market_data,
    place_order
)

logger = logging.getLogger(__name__)


@pytest.fixture
def bridge_config():
    """Fixture for bridge configuration"""
    return BridgeConfig(mode=BridgeMode.IB_INSYNC_ONLY)


@pytest.fixture
def mock_ibkr_connector():
    """Mock IBKR connector"""
    mock_connector = MagicMock()
    mock_connector.connect = AsyncMock(return_value=True)
    mock_connector.disconnect = AsyncMock(return_value=True)
    mock_connector.get_market_data = AsyncMock(return_value={"price": 100.0, "volume": 1000})
    mock_connector.get_account_summary = AsyncMock(return_value={"balance": 10000})
    mock_connector.get_positions = AsyncMock(return_value=[])
    return mock_connector


@pytest.mark.asyncio
async def test_bridge_instantiation(bridge_config):
    """Test basic bridge instantiation"""
    bridge = NautilusIBKRBridge(bridge_config)
    assert bridge is not None
    assert bridge.config == bridge_config
    assert not bridge._initialized


@pytest.mark.asyncio
async def test_bridge_status(bridge_config):
    """Test bridge status retrieval"""
    bridge = NautilusIBKRBridge(bridge_config)
    status = bridge.get_bridge_status()

    assert isinstance(status, dict)
    assert "mode" in status
    assert "nautilus_available" in status
    assert "nautilus_active" in status
    assert status["mode"] == bridge_config.mode.value


@pytest.mark.asyncio
@patch('src.integrations.nautilus_ibkr_bridge.IBKRConnector')
async def test_bridge_initialization(mock_connector_class, bridge_config, mock_ibkr_connector):
    """Test bridge initialization"""
    mock_connector_class.return_value = mock_ibkr_connector

    bridge = NautilusIBKRBridge(bridge_config)
    success = await bridge.initialize()

    assert success is True
    assert bridge._initialized is True
    mock_ibkr_connector.connect.assert_called_once()


@pytest.mark.asyncio
@patch('src.integrations.nautilus_ibkr_bridge.IBKRConnector')
async def test_get_market_data(mock_connector_class, bridge_config, mock_ibkr_connector):
    """Test market data retrieval"""
    mock_connector_class.return_value = mock_ibkr_connector

    bridge = NautilusIBKRBridge(bridge_config)
    await bridge.initialize()

    result = await bridge.get_market_data("SPY")

    assert result == {"price": 100.0, "volume": 1000}
    mock_ibkr_connector.get_market_data.assert_called_once_with("SPY", "1 min", "1 D")


@pytest.mark.asyncio
async def test_singleton_bridge():
    """Test singleton bridge pattern"""
    # Clear any existing instance
    import src.integrations.nautilus_ibkr_bridge as bridge_module
    bridge_module._bridge_instance = None

    bridge1 = get_nautilus_ibkr_bridge()
    bridge2 = get_nautilus_ibkr_bridge()

    assert bridge1 is bridge2
    assert isinstance(bridge1, NautilusIBKRBridge)


@pytest.mark.asyncio
@patch('src.integrations.nautilus_ibkr_bridge.get_nautilus_ibkr_bridge')
async def test_initialize_bridge_convenience(mock_get_bridge):
    """Test initialize_bridge convenience function"""
    mock_bridge = MagicMock()
    mock_bridge.initialize = AsyncMock(return_value=True)
    mock_get_bridge.return_value = mock_bridge

    result = await initialize_bridge("ib_insync_only")

    assert result is True
    mock_get_bridge.assert_called_once()
    mock_bridge.initialize.assert_called_once()


@pytest.mark.asyncio
@patch('src.integrations.nautilus_ibkr_bridge.get_nautilus_ibkr_bridge')
async def test_get_market_data_convenience(mock_get_bridge):
    """Test get_market_data convenience function"""
    mock_bridge = MagicMock()
    mock_bridge.get_market_data = AsyncMock(return_value={"price": 150.0})
    mock_get_bridge.return_value = mock_bridge

    result = await get_market_data("AAPL")

    assert result == {"price": 150.0}
    mock_bridge.get_market_data.assert_called_once_with("AAPL")


@pytest.mark.asyncio
async def test_bridge_modes():
    """Test different bridge modes"""
    modes = [BridgeMode.IB_INSYNC_ONLY, BridgeMode.NAUTILUS_ENHANCED]

    for mode in modes:
        config = BridgeConfig(mode=mode)
        bridge = NautilusIBKRBridge(config)

        status = bridge.get_bridge_status()
        assert status["mode"] == mode.value
        assert isinstance(status["nautilus_available"], bool)
        assert isinstance(status["nautilus_active"], bool)


@pytest.mark.asyncio
@patch('src.integrations.nautilus_ibkr_bridge.check_pre_trade_risk')
@patch('src.integrations.nautilus_ibkr_bridge.validate_trading_conditions')
@patch('src.integrations.nautilus_ibkr_bridge.IBKRConnector')
async def test_place_order(mock_connector_class, mock_validate_trading, mock_check_risk, mock_ibkr_connector):
    """Test order placement"""
    mock_connector_class.return_value = mock_ibkr_connector
    mock_ibkr_connector.place_order = AsyncMock(return_value={"order_id": 123, "status": "submitted"})
    mock_ibkr_connector.get_market_data = AsyncMock(return_value={
        'symbol': 'SPY',
        'close': 450.0,
        'open': 445.0,
        'high': 452.0,
        'low': 444.0,
        'volume': 1000
    })  # Mock market data for risk calc
    mock_ibkr_connector.get_account_summary = AsyncMock(return_value={'balance': 10000})
    mock_ibkr_connector.get_positions = AsyncMock(return_value=[])
    mock_check_risk.return_value = (True, "Risk check passed", {"score": 0.1})
    mock_validate_trading.return_value = (True, "Market conditions OK")  # Returns tuple (bool, str)

    bridge = NautilusIBKRBridge(BridgeConfig())
    await bridge.initialize()

    result = await bridge.place_order(
        symbol="SPY",
        quantity=100,
        order_type="MKT",
        action="BUY"
    )

    assert result == {"order_id": 123, "status": "submitted"}
    mock_ibkr_connector.place_order.assert_called_once()
    mock_check_risk.assert_called_once()
    mock_validate_trading.assert_called_once()


@pytest.mark.asyncio
@patch('src.integrations.nautilus_ibkr_bridge.get_nautilus_ibkr_bridge')
async def test_place_order_convenience(mock_get_bridge):
    """Test place_order convenience function"""
    mock_bridge = MagicMock()
    mock_bridge.place_order = AsyncMock(return_value={"order_id": 456, "status": "filled"})
    mock_get_bridge.return_value = mock_bridge

    result = await place_order("AAPL", 50, "MKT", "BUY")

    assert result == {"order_id": 456, "status": "filled"}
    mock_bridge.place_order.assert_called_once_with("AAPL", 50, "MKT", "BUY", None)


@pytest.mark.asyncio
@patch('src.integrations.nautilus_ibkr_bridge.IBKRConnector')
async def test_get_account_summary(mock_connector_class, bridge_config, mock_ibkr_connector):
    """Test account summary retrieval"""
    mock_connector_class.return_value = mock_ibkr_connector

    bridge = NautilusIBKRBridge(bridge_config)
    await bridge.initialize()

    result = await bridge.get_account_summary()

    assert result == {"balance": 10000}
    mock_ibkr_connector.get_account_summary.assert_called_once()


@pytest.mark.asyncio
@patch('src.integrations.nautilus_ibkr_bridge.IBKRConnector')
async def test_get_positions(mock_connector_class, bridge_config, mock_ibkr_connector):
    """Test positions retrieval"""
    mock_connector_class.return_value = mock_ibkr_connector

    bridge = NautilusIBKRBridge(bridge_config)
    await bridge.initialize()

    result = await bridge.get_positions()

    assert result == []
    mock_ibkr_connector.get_positions.assert_called_once()