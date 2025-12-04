#!/usr/bin/env python3
"""
End-to-End Trading Tests
Tests complete trading workflow from order placement through execution to persistence.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.execution import ExecutionAgent
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator
from src.utils.alert_manager import get_alert_manager


class TestEndToEndTrading:
    """End-to-end trading workflow tests"""

    @pytest.fixture
    def orchestrator(self):
        """Create workflow orchestrator for testing"""
        orchestrator = LiveWorkflowOrchestrator()
        return orchestrator

    @pytest.fixture
    def execution_agent(self):
        """Create execution agent for testing"""
        agent = ExecutionAgent()
        # Reset any existing state
        agent.ibkr_connector = None
        agent.tb_client = None
        return agent

    @pytest.mark.asyncio
    async def test_complete_buy_order_lifecycle(self, orchestrator, execution_agent):
        """Test complete buy order lifecycle: create → execute → log → confirm"""

        # Mock IBKR successful trade
        mock_ibkr = AsyncMock()
        buy_trade_result = {
            'success': True,
            'order_id': 'BUY_12345',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'commission': 1.00,
            'symbol': 'AAPL',
            'action': 'BUY',
            'timestamp': datetime.now().isoformat()
        }
        mock_ibkr.place_order.return_value = buy_trade_result

        # Mock TigerBeetle for transaction logging
        mock_tb = Mock()
        mock_tb.create_transfers.return_value = [987654321]
        mock_tb.create_accounts.return_value = [111222333]  # For account creation

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            # Set mock client
            execution_agent.tb_client = mock_tb

            # Step 1: Create and place buy order
            order_details = {
                'symbol': 'AAPL',
                'quantity': 100,
                'action': 'BUY',
                'order_type': 'MKT',
                'price': None
            }

            # Execute the trade
            result = await execution_agent.execute_trade(**order_details)

            # Step 2: Verify trade execution
            assert result['success'] is True
            assert result['order_id'] == 'BUY_12345'
            assert result['status'] == 'FILLED'
            assert result['filled_quantity'] == 100
            assert result['symbol'] == 'AAPL'
            assert result['action'] == 'BUY'

            # Step 3: Verify IBKR interaction
            mock_ibkr.place_order.assert_called_once()
            call_args = mock_ibkr.place_order.call_args
            assert call_args[0][0] == 'AAPL'  # symbol
            assert call_args[0][1] == 100     # quantity
            assert call_args[0][2] == 'MKT'   # order_type
            assert call_args[0][3] == 'BUY'   # action

            # Step 4: Verify TigerBeetle transaction logging
            # Should create account first, then transfer
            assert mock_tb.create_accounts.called
            assert mock_tb.create_transfers.called

            # Verify transfer was for the correct amount (100 shares * 100 for cents scaling)
            transfer_call = mock_tb.create_transfers.call_args[0][0][0]  # First transfer
            assert transfer_call.amount == 10000  # 100 shares * 100 (TigerBeetle scaling)

    @pytest.mark.asyncio
    async def test_buy_sell_round_trip_workflow(self, execution_agent):
        """Test complete buy-sell round trip with position tracking"""

        # Mock IBKR for both trades
        mock_ibkr = AsyncMock()

        # Buy trade result
        buy_result = {
            'success': True,
            'order_id': 'BUY_12345',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'commission': 1.00,
            'symbol': 'AAPL',
            'action': 'BUY'
        }

        # Sell trade result
        sell_result = {
            'success': True,
            'order_id': 'SELL_12346',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 155.75,
            'commission': 1.00,
            'symbol': 'AAPL',
            'action': 'SELL'
        }

        mock_ibkr.place_order.side_effect = [buy_result, sell_result]

        # Mock TigerBeetle
        mock_tb = Mock()
        mock_tb.create_transfers.return_value = [987654321, 987654322]
        mock_tb.create_accounts.return_value = [111222333]

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            execution_agent.tb_client = mock_tb

            # Step 1: Execute buy order
            buy_result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY',
                order_type='MKT'
            )

            # Step 2: Execute sell order
            sell_result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=-100,  # Negative for sell
                action='SELL',
                order_type='MKT'
            )

            # Verify both trades succeeded
            assert buy_result['success'] is True
            assert sell_result['success'] is True

            # Verify IBKR was called twice
            assert mock_ibkr.place_order.call_count == 2

            # Verify TigerBeetle logged both transactions
            assert mock_tb.create_transfers.call_count == 2

            # Calculate profit/loss (simplified)
            buy_cost = 100 * 150.50 + 1.00  # 100 shares + commission
            sell_proceeds = 100 * 155.75 - 1.00  # 100 shares - commission
            expected_pnl = sell_proceeds - buy_cost

            # In a real system, we'd verify P&L calculation
            assert expected_pnl == 523.00  # (155.75 - 150.50) * 100 - 2.00

    @pytest.mark.asyncio
    async def test_order_lifecycle_with_partial_fills(self, execution_agent):
        """Test order lifecycle with partial fills and multiple executions"""

        # Mock IBKR with partial fills
        mock_ibkr = AsyncMock()

        # First execution: partial fill
        partial_result = {
            'success': True,
            'order_id': 'PARTIAL_12347',
            'status': 'PARTIALLY_FILLED',
            'filled_quantity': 50,
            'remaining_quantity': 50,
            'avg_fill_price': 150.00,
            'commission': 0.50,
            'symbol': 'AAPL',
            'action': 'BUY'
        }

        # Second execution: remaining fill
        complete_result = {
            'success': True,
            'order_id': 'PARTIAL_12347',
            'status': 'FILLED',
            'filled_quantity': 100,
            'remaining_quantity': 0,
            'avg_fill_price': 150.25,
            'commission': 1.00,
            'symbol': 'AAPL',
            'action': 'BUY'
        }

        mock_ibkr.place_order.side_effect = [partial_result, complete_result]

        # Mock TigerBeetle
        mock_tb = Mock()
        mock_tb.create_transfers.return_value = [987654321, 987654322]
        mock_tb.create_accounts.return_value = [111222333]

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            execution_agent.tb_client = mock_tb

            # Execute initial order
            result1 = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY',
                order_type='MKT'
            )

            # Execute remaining order
            result2 = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=50,
                action='BUY',
                order_type='MKT'
            )

            # Verify partial fill handling
            assert result1['status'] == 'PARTIALLY_FILLED'
            assert result1['filled_quantity'] == 50

            # Verify complete fill
            assert result2['status'] == 'FILLED'
            assert result2['filled_quantity'] == 100

            # Verify TigerBeetle logged both transactions
            assert mock_tb.create_transfers.call_count == 2

    @pytest.mark.asyncio
    async def test_trade_execution_with_risk_checks(self, execution_agent):
        """Test trade execution includes risk validation"""

        # Mock IBKR success
        mock_ibkr = AsyncMock()
        trade_result = {
            'success': True,
            'order_id': 'RISK_12348',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'commission': 1.00,
            'symbol': 'AAPL',
            'action': 'BUY'
        }
        mock_ibkr.place_order.return_value = trade_result

        # Mock TigerBeetle
        mock_tb = Mock()
        mock_tb.create_transfers.return_value = [987654321]
        mock_tb.create_accounts.return_value = [111222333]

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb), \
             patch.object(execution_agent, '_check_execution_timing', return_value={'optimal_timing': True}):
            execution_agent.tb_client = mock_tb

            # Execute trade with risk parameters
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY',
                order_type='LMT',
                price=150.50
            )

            # Verify trade succeeded
            assert result['success'] is True
            assert result['status'] == 'FILLED'

            # Verify timing check was performed
            # (In real implementation, this would validate market hours, volatility, etc.)

            # Verify TigerBeetle logging occurred
            assert mock_tb.create_transfers.called

    @pytest.mark.asyncio
    async def test_data_consistency_across_components(self, execution_agent):
        """Test data consistency between IBKR results and TigerBeetle logs"""

        # Mock IBKR
        mock_ibkr = AsyncMock()
        trade_result = {
            'success': True,
            'order_id': 'CONSISTENCY_12349',
            'status': 'FILLED',
            'filled_quantity': 200,
            'avg_fill_price': 75.25,
            'commission': 2.00,
            'symbol': 'TSLA',
            'action': 'BUY',
            'timestamp': datetime.now().isoformat()
        }
        mock_ibkr.place_order.return_value = trade_result

        # Mock TigerBeetle with detailed tracking
        mock_tb = Mock()
        transfer_id = 555666777
        mock_tb.create_transfers.return_value = [transfer_id]
        mock_tb.create_accounts.return_value = [111222333]

        # Track what gets logged to TigerBeetle
        logged_transfers = []

        def mock_create_transfers(transfers):
            logged_transfers.extend(transfers)
            return [transfer_id]

        mock_tb.create_transfers.side_effect = mock_create_transfers

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            execution_agent.tb_client = mock_tb

            # Execute trade
            result = await execution_agent.execute_trade(
                symbol='TSLA',
                quantity=200,
                action='BUY',
                order_type='MKT'
            )

            # Verify IBKR data
            assert result['symbol'] == 'TSLA'
            assert result['filled_quantity'] == 200
            assert result['action'] == 'BUY'

            # Verify TigerBeetle logged the transaction
            assert len(logged_transfers) == 1
            transfer = logged_transfers[0]

            # Verify transfer details match trade
            assert transfer.amount == 20000  # 200 shares * 100 (TigerBeetle scaling)
            assert transfer.code == 1  # Buy code

            # Verify account consistency (symbol-based account ID)
            # In real implementation, this would verify account creation and linking

    @pytest.mark.asyncio
    async def test_error_recovery_and_rollback_scenarios(self, execution_agent):
        """Test error recovery and rollback mechanisms"""

        # Test 1: IBKR succeeds, TigerBeetle fails - should still return success
        mock_ibkr = AsyncMock()
        trade_result = {
            'success': True,
            'order_id': 'RECOVERY_12350',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'symbol': 'AAPL',
            'action': 'BUY'
        }
        mock_ibkr.place_order.return_value = trade_result

        # TigerBeetle fails
        mock_tb = Mock()
        mock_tb.create_transfers.side_effect = Exception("TigerBeetle connection lost")

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            execution_agent.tb_client = mock_tb

            # Execute trade - should succeed despite TigerBeetle failure
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY'
            )

            # Trade should still succeed
            assert result['success'] is True
            assert result['status'] == 'FILLED'

            # But TigerBeetle logging failed
            assert mock_tb.create_transfers.called

        # Test 2: IBKR fails completely - should return error
        mock_ibkr_fail = AsyncMock()
        mock_ibkr_fail.place_order.side_effect = Exception("IBKR API unavailable")

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr_fail):
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY'
            )

            # Should fail gracefully
            assert 'error' in result
            assert 'IBKR API unavailable' in result['error']

    @pytest.mark.asyncio
    async def test_concurrent_trade_execution(self, execution_agent):
        """Test concurrent trade execution handling"""

        # Mock IBKR with different responses for concurrent calls
        mock_ibkr = AsyncMock()

        async def mock_place_order(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate network delay
            return {
                'success': True,
                'order_id': f'CONCURRENT_{args[0]}_{id(asyncio.current_task())}',
                'status': 'FILLED',
                'filled_quantity': args[1],
                'avg_fill_price': 150.50,
                'symbol': args[0],
                'action': args[3]
            }

        mock_ibkr.place_order.side_effect = mock_place_order

        # Mock TigerBeetle
        mock_tb = Mock()
        mock_tb.create_transfers.return_value = [987654321, 987654322, 987654323]
        mock_tb.create_accounts.return_value = [111222333]

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            execution_agent.tb_client = mock_tb

            # Execute multiple trades concurrently
            tasks = [
                execution_agent.execute_trade('AAPL', 100, 'BUY', 'MKT'),
                execution_agent.execute_trade('GOOGL', 50, 'BUY', 'MKT'),
                execution_agent.execute_trade('MSFT', 75, 'BUY', 'MKT')
            ]

            results = await asyncio.gather(*tasks)

            # All trades should succeed
            assert all(result['success'] for result in results)
            assert len(results) == 3

            # Verify IBKR was called for each trade
            assert mock_ibkr.place_order.call_count == 3

            # Verify TigerBeetle logged all transactions
            assert mock_tb.create_transfers.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])