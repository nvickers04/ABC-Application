#!/usr/bin/env python3
"""
IBKR Fixtures Test
Tests the new IBKR test fixtures and cleanup procedures
"""

import pytest
import asyncio
import logging
from unittest.mock import patch

logger = logging.getLogger(__name__)

class TestIBKRFixtures:
    """Test IBKR fixtures functionality"""

    @pytest.mark.asyncio
    async def test_mock_ibkr_connector(self, mock_ibkr_connector):
        """Test mock IBKR connector fixture"""
        # Test connection
        connected = await mock_ibkr_connector.connect()
        assert connected is True

        # Test market data
        data = await mock_ibkr_connector.get_market_data('AAPL')
        assert data['symbol'] == 'AAPL'
        assert 'bid' in data
        assert 'ask' in data

        # Test account summary
        account = await mock_ibkr_connector.get_account_summary()
        assert 'cash' in account
        assert account['cash'] == 50000.00

        # Test positions
        positions = await mock_ibkr_connector.get_positions()
        assert len(positions) == 2
        assert positions[0]['symbol'] == 'AAPL'

    def test_mock_ib_insync(self, mock_ib_insync):
        """Test mock ib_insync fixture"""
        ib = mock_ib_insync['ib']
        contract = mock_ib_insync['contract']
        order = mock_ib_insync['order']

        # Test IB connection
        assert ib.connect() is True
        assert ib.isConnected() is True

        # Test contract
        assert contract.symbol == 'AAPL'
        assert contract.secType == 'STK'

        # Test order
        assert order.orderId == 12345
        assert order.status == 'Submitted'

    @pytest.mark.asyncio
    async def test_ibkr_test_connector(self, ibkr_test_connector):
        """Test real IBKR connector fixture with cleanup"""
        # Test that connector is initialized
        assert ibkr_test_connector is not None

        # Test connection status (may be disconnected if TWS not running)
        status = ibkr_test_connector.get_connection_status()
        assert isinstance(status, dict)
        assert 'connected' in status

        # Fixture should handle cleanup automatically

    def test_sample_ibkr_order(self, sample_ibkr_order):
        """Test sample IBKR order fixture"""
        order = sample_ibkr_order

        assert order['symbol'] == 'AAPL'
        assert order['quantity'] == 100
        assert order['action'] == 'BUY'
        assert order['order_type'] == 'LMT'
        assert order['limit_price'] == 150.00

    def test_sample_ibkr_contract(self, sample_ibkr_contract):
        """Test sample IBKR contract fixture"""
        contract = sample_ibkr_contract

        assert contract['symbol'] == 'AAPL'
        assert contract['secType'] == 'STK'
        assert contract['exchange'] == 'SMART'

    def test_mock_tws_connection(self, mock_tws_connection):
        """Test mock TWS connection fixture"""
        # The fixture mocks socket.create_connection
        # This test verifies the mock is working
        import socket
        sock = socket.create_connection(("localhost", 7497), timeout=1)
        assert sock is not None

    @pytest.mark.asyncio
    async def test_ibkr_test_environment(self, ibkr_test_environment):
        """Test complete IBKR test environment"""
        env = ibkr_test_environment

        assert 'connector' in env
        assert 'tws_available' in env
        assert 'paper_trading' in env
        assert env['paper_trading'] is True
        assert env['test_account'] == 'DU1234567'

    def test_ibkr_error_scenarios(self, ibkr_error_scenarios):
        """Test IBKR error scenarios fixture"""
        scenarios = ibkr_error_scenarios

        assert 'connection_timeout' in scenarios
        assert 'invalid_contract' in scenarios
        assert 'insufficient_funds' in scenarios
        assert 'market_closed' in scenarios
        assert 'circuit_breaker' in scenarios

        # Test specific scenario structure
        timeout_scenario = scenarios['connection_timeout']
        assert 'error' in timeout_scenario
        assert 'message' in timeout_scenario
        assert timeout_scenario['retry_count'] == 3

    @pytest.mark.asyncio
    async def test_ibkr_cleanup_manager(self, ibkr_cleanup_manager):
        """Test IBKR cleanup manager fixture"""
        manager = ibkr_cleanup_manager

        # Test adding cleanup tasks
        cleanup_called = []

        async def async_cleanup():
            cleanup_called.append('async')

        def sync_cleanup():
            cleanup_called.append('sync')

        manager['add_task'](async_cleanup)
        manager['add_task'](sync_cleanup)

        # Execute cleanup
        await manager['cleanup']()

        # Verify cleanup was called
        assert 'async' in cleanup_called
        assert 'sync' in cleanup_called

    @pytest.mark.asyncio
    async def test_fixture_integration(self, ibkr_test_environment, mock_ibkr_connector, ibkr_cleanup_manager):
        """Test integration of multiple IBKR fixtures"""
        env = ibkr_test_environment
        connector = mock_ibkr_connector
        cleanup = ibkr_cleanup_manager

        # Add a test cleanup task
        def test_cleanup():
            logger.info("Test cleanup executed")

        cleanup['add_task'](test_cleanup)

        # Test environment has connector
        assert env['connector'] is not None

        # Mock connector works
        connected = await connector.connect()
        assert connected is True

        # Cleanup will be handled automatically by fixtures

if __name__ == "__main__":
    pytest.main([__file__, "-v"])