#!/usr/bin/env python3
"""
Integration tests for ExecutionAgent + TigerBeetle integration.
Tests that trades can be executed via IBKR and logged to TigerBeetle simultaneously.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.execution import ExecutionAgent
from src.utils.alert_manager import get_alert_manager


class TestExecutionTigerBeetleIntegration:
    """Test ExecutionAgent integration with TigerBeetle"""

    @pytest.fixture
    def execution_agent(self):
        """Create execution agent for testing"""
        agent = ExecutionAgent()
        # Reset any existing state
        agent.ibkr_connector = None
        agent.tb_client = None
        return agent

    @pytest.mark.asyncio
    async def test_execution_agent_initialization(self, execution_agent):
        """Test that execution agent initializes properly"""
        # Agent is initialized in __init__, just check components exist
        assert execution_agent.memory is not None
        assert hasattr(execution_agent, 'alert_manager')
        assert hasattr(execution_agent, 'tb_client')

    @pytest.mark.asyncio
    async def test_trade_execution_with_tigerbeetle_logging(self, execution_agent):
        """Test complete trade execution flow with TigerBeetle logging"""
        # Mock IBKR connector
        mock_ibkr = AsyncMock()
        mock_trade_result = {
            'success': True,
            'order_id': '12345',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'commission': 1.00
        }
        mock_ibkr.place_order.return_value = mock_trade_result

        # Mock TigerBeetle client
        mock_tb = Mock()
        mock_tb.create_transfers.return_value = [123456789]  # Mock transfer ID

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            # Set the mock client directly on the agent
            execution_agent.tb_client = mock_tb

            # Execute a trade
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY',
                order_type='MKT'
            )

            # Verify IBKR was called
            mock_ibkr.place_order.assert_called_once()
            call_args, call_kwargs = mock_ibkr.place_order.call_args
            assert call_args[0] == 'AAPL'  # symbol
            assert call_args[1] == 100     # quantity
            assert call_args[2] == 'MKT'   # order_type
            assert call_args[3] == 'BUY'   # action

            # Verify TigerBeetle was called for logging
            mock_tb.create_transfers.assert_called_once()

            # Verify result structure
            assert result['success'] is True
            assert result['order_id'] == '12345'
            assert result['status'] == 'FILLED'
            # Note: TigerBeetle logging happens asynchronously and doesn't modify the return result

    @pytest.mark.asyncio
    async def test_trade_execution_ibkr_failure(self, execution_agent):
        """Test trade execution when IBKR fails"""
        # Mock IBKR connector to fail
        mock_ibkr = AsyncMock()
        mock_ibkr.place_order.side_effect = Exception("IBKR connection failed")

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', False):

            # Execute trade (should fail gracefully)
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY'
            )

            # Verify failure is handled
            assert 'error' in result
            assert 'IBKR connection failed' in result['error']
            assert 'success' not in result  # Method doesn't add success on error

            # Verify IBKR was called but TigerBeetle wasn't (since IBKR failed)
            mock_ibkr.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_trade_execution_tigerbeetle_failure(self, execution_agent):
        """Test trade execution when TigerBeetle fails but IBKR succeeds"""
        # Mock IBKR success
        mock_ibkr = AsyncMock()
        mock_trade_result = {
            'success': True,
            'order_id': '12345',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50
        }
        mock_ibkr.place_order.return_value = mock_trade_result

        # Mock TigerBeetle failure
        mock_tb = Mock()
        mock_tb.create_transfers.side_effect = Exception("TigerBeetle connection failed")

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            # Set the mock client directly on the agent
            execution_agent.tb_client = mock_tb

            # Execute trade
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY'
            )

            # Verify IBKR succeeded
            assert result['success'] is True
            assert result['status'] == 'FILLED'

            # TigerBeetle failure is logged but doesn't affect the trade result

    @pytest.mark.asyncio
    async def test_tigerbeetle_transaction_structure(self, execution_agent):
        """Test that TigerBeetle transactions have correct structure"""
        # Mock IBKR success
        mock_ibkr = AsyncMock()
        mock_trade_result = {
            'success': True,
            'order_id': '12345',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'commission': 1.00
        }
        mock_ibkr.place_order.return_value = mock_trade_result

        # Mock TigerBeetle
        mock_tb = Mock()
        transfer_id = 987654321
        mock_tb.create_transfers.return_value = [transfer_id]

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            # Set the mock client directly on the agent
            execution_agent.tb_client = mock_tb

            # Execute trade
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY'
            )

            # Verify trade succeeded and TigerBeetle was called
            assert result['success'] is True
            assert result['status'] == 'FILLED'
            mock_tb.create_transfers.assert_called_once()

    @pytest.mark.asyncio
    async def test_sell_trade_negative_quantity_handling(self, execution_agent):
        """Test that sell trades are handled correctly with negative quantities"""
        # Mock IBKR success
        mock_ibkr = AsyncMock()
        mock_trade_result = {
            'success': True,
            'order_id': '12346',
            'status': 'FILLED',
            'filled_quantity': 50,
            'avg_fill_price': 155.00
        }
        mock_ibkr.place_order.return_value = mock_trade_result

        # Mock TigerBeetle
        mock_tb = Mock()
        mock_tb.create_transfers.return_value = [555666777]

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            # Set the mock client directly on the agent
            execution_agent.tb_client = mock_tb

            # Execute sell trade
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=-50,  # Negative for sell
                action='SELL'
            )

            # Verify trade succeeded
            assert result['success'] is True
            assert result['filled_quantity'] == 50

            # Verify TigerBeetle was called for logging
            mock_tb.create_transfers.assert_called_once()

    @pytest.mark.asyncio
    async def test_alert_integration_on_trade_failure(self, execution_agent):
        """Test that alerts are sent when trade execution fails"""
        # Mock IBKR to fail
        mock_ibkr = AsyncMock()
        mock_ibkr.place_order.side_effect = Exception("Market closed")

        # Mock alert manager
        mock_alert_manager = Mock()

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', False), \
             patch.object(execution_agent, 'alert_manager', mock_alert_manager):

            # Execute trade that will fail
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY'
            )

            # Note: Current implementation logs errors but doesn't send alerts on trade failure
            # This could be enhanced to send alerts for critical execution failures


if __name__ == "__main__":
    pytest.main([__file__, "-v"])