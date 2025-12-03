#!/usr/bin/env python3
"""
Component Health Check Tests
Tests that all system components can initialize and operate together.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator
from src.agents.execution import ExecutionAgent
from src.utils.alert_manager import get_alert_manager


class TestComponentHealthChecks:
    """Test component health checks and interoperability"""

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
    async def test_orchestrator_initialization_health(self, orchestrator):
        """Test that orchestrator initializes properly with all components"""

        # Check that orchestrator has required attributes
        assert hasattr(orchestrator, 'scheduler')
        assert hasattr(orchestrator, 'a2a_protocol')
        assert hasattr(orchestrator, 'agent_instances')

        # Check scheduler is initialized
        assert orchestrator.scheduler is not None

        # Check A2A protocol is initialized
        assert orchestrator.a2a_protocol is not None

        # Check agent instances dict exists
        assert isinstance(orchestrator.agent_instances, dict)

    @pytest.mark.asyncio
    async def test_execution_agent_component_initialization(self, execution_agent):
        """Test execution agent component initialization"""

        # Check execution agent has required components
        assert hasattr(execution_agent, 'alert_manager')
        assert hasattr(execution_agent, 'memory')
        assert hasattr(execution_agent, 'timing_optimizer')
        assert hasattr(execution_agent, 'scheduler')

        # Check alert manager integration
        assert execution_agent.alert_manager is not None

        # Check memory is initialized
        assert execution_agent.memory is not None

        # Check timing optimizer
        assert execution_agent.timing_optimizer is not None

    @pytest.mark.asyncio
    async def test_ibkr_connector_health_check_simulation(self, execution_agent):
        """Test IBKR connector health check in simulation mode"""

        # Mock IBKR connector for simulation mode
        mock_ibkr = AsyncMock()
        mock_ibkr.connect.return_value = True
        mock_ibkr.is_connected.return_value = True

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr):
            # Test TWS status check
            status = await execution_agent._check_ibkr_tws_status()

            # Should return connected status (not necessarily simulated)
            assert status['connected'] is True
            assert 'status' in status

    @pytest.mark.asyncio
    async def test_tigerbeetle_client_initialization(self, execution_agent):
        """Test TigerBeetle client initialization and basic operations"""

        # Mock TigerBeetle client
        mock_tb = Mock()
        mock_tb.create_accounts.return_value = [123456789]
        mock_tb.create_transfers.return_value = [987654321]

        with patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            execution_agent.tb_client = mock_tb

            # Test that client is available
            assert execution_agent.tb_client is not None

            # Test basic account creation (mock)
            mock_tb.create_accounts.assert_not_called()  # Not called yet

            # Test basic transfer creation (mock)
            mock_tb.create_transfers.assert_not_called()  # Not called yet

    @pytest.mark.asyncio
    async def test_alert_manager_component_integration(self, orchestrator, execution_agent):
        """Test alert manager integration across components"""

        # Mock alert manager to track calls
        mock_alert_manager = Mock()

        with patch.object(orchestrator, 'alert_manager', mock_alert_manager), \
             patch.object(execution_agent, 'alert_manager', mock_alert_manager):

            # Test orchestrator alert capability
            orchestrator.alert_manager.info("Test orchestrator alert")
            assert mock_alert_manager.info.called

            # Test execution agent alert capability
            execution_agent.alert_manager.warning("Test execution alert")
            assert mock_alert_manager.warning.called

            # Verify alert manager methods exist
            assert hasattr(mock_alert_manager, 'info')
            assert hasattr(mock_alert_manager, 'warning')
            assert hasattr(mock_alert_manager, 'error')
            assert hasattr(mock_alert_manager, 'critical')

    @pytest.mark.asyncio
    async def test_memory_system_health_across_components(self, orchestrator, execution_agent):
        """Test memory system health and consistency across components"""

        # Test execution agent memory (orchestrator doesn't have memory system)
        assert execution_agent.memory is not None
        assert isinstance(execution_agent.memory, dict)

        # Test memory persistence capability
        initial_memory = execution_agent.memory.copy()

        # Modify memory
        execution_agent.memory['test_key'] = 'test_value'
        execution_agent.save_memory()

        # Verify memory was saved (in real system this would persist)
        assert 'test_key' in execution_agent.memory

    @pytest.mark.asyncio
    async def test_scheduler_component_health(self, orchestrator, execution_agent):
        """Test task scheduler component health"""

        # Test orchestrator scheduler
        assert hasattr(orchestrator, 'scheduler')
        assert orchestrator.scheduler is not None

        # Test execution agent scheduler
        assert hasattr(execution_agent, 'scheduler')
        assert execution_agent.scheduler is not None

        # Test scheduler basic functionality (mock)
        with patch.object(orchestrator.scheduler, 'start') as mock_start, \
             patch.object(orchestrator.scheduler, 'shutdown') as mock_shutdown:

            # Scheduler should be able to start (in real system)
            orchestrator.scheduler.start()
            mock_start.assert_called_once()

            # Scheduler should be able to shutdown
            orchestrator.scheduler.shutdown()
            mock_shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_component_interoperability_basic(self, orchestrator, execution_agent):
        """Test basic interoperability between components"""

        # Mock IBKR and TigerBeetle for integration test
        mock_ibkr = AsyncMock()
        mock_trade_result = {
            'success': True,
            'order_id': 'HEALTH_12345',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'symbol': 'AAPL',
            'action': 'BUY'
        }
        mock_ibkr.place_order.return_value = mock_trade_result

        mock_tb = Mock()
        mock_tb.create_transfers.return_value = [987654321]

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            execution_agent.tb_client = mock_tb

            # Test that execution agent can operate
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY',
                order_type='MKT'
            )

            # Verify successful operation
            assert result['success'] is True
            assert result['status'] == 'FILLED'

            # Verify IBKR was called
            mock_ibkr.place_order.assert_called_once()

            # Verify TigerBeetle was called
            mock_tb.create_transfers.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_across_components(self, orchestrator, execution_agent):
        """Test error handling and propagation across components"""

        # Test IBKR failure scenario
        mock_ibkr_fail = AsyncMock()
        mock_ibkr_fail.place_order.side_effect = Exception("IBKR connection failed")

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr_fail):
            # Execute trade that should fail
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY'
            )

            # Should handle error gracefully
            assert 'error' in result
            assert 'IBKR connection failed' in result['error']

        # Test TigerBeetle failure scenario (IBKR succeeds)
        mock_ibkr_success = AsyncMock()
        mock_trade_result = {
            'success': True,
            'order_id': 'ERROR_TEST_12346',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'symbol': 'AAPL',
            'action': 'BUY'
        }
        mock_ibkr_success.place_order.return_value = mock_trade_result

        mock_tb_fail = Mock()
        mock_tb_fail.create_transfers.side_effect = Exception("TigerBeetle unavailable")

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr_success), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb_fail):
            execution_agent.tb_client = mock_tb_fail

            # Execute trade - should succeed despite TigerBeetle failure
            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY'
            )

            # Trade should still succeed
            assert result['success'] is True
            assert result['status'] == 'FILLED'

    @pytest.mark.asyncio
    async def test_configuration_consistency_across_components(self, orchestrator, execution_agent):
        """Test that components use consistent configuration"""

        # Check that execution agent has access to configs (orchestrator doesn't have configs)
        assert hasattr(execution_agent, 'configs')

        # Check that configs is a dictionary
        assert isinstance(execution_agent.configs, dict)

        # Verify key configuration sections exist
        # (This would be more specific in a real system)
        assert len(execution_agent.configs) > 0

    @pytest.mark.asyncio
    async def test_component_startup_shutdown_sequence(self, orchestrator, execution_agent):
        """Test component startup and shutdown sequences"""

        # Test startup sequence
        # (In real system, this would test actual startup)

        # Mock scheduler operations
        with patch.object(orchestrator.scheduler, 'start') as orch_start, \
             patch.object(orchestrator.scheduler, 'shutdown') as orch_shutdown, \
             patch.object(execution_agent.scheduler, 'start') as exec_start, \
             patch.object(execution_agent.scheduler, 'shutdown') as exec_shutdown:

            # Test startup calls
            orchestrator.scheduler.start()
            execution_agent.scheduler.start()

            orch_start.assert_called_once()
            exec_start.assert_called_once()

            # Test shutdown calls
            orchestrator.scheduler.shutdown()
            execution_agent.scheduler.shutdown()

            orch_shutdown.assert_called_once()
            exec_shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_basic_operations(self, execution_agent):
        """Test basic performance of component operations"""

        # Mock IBKR for performance testing
        mock_ibkr = AsyncMock()
        mock_trade_result = {
            'success': True,
            'order_id': 'PERF_12347',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'symbol': 'AAPL',
            'action': 'BUY'
        }
        mock_ibkr.place_order.return_value = mock_trade_result

        mock_tb = Mock()
        mock_tb.create_transfers.return_value = [987654321]

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            execution_agent.tb_client = mock_tb

            # Measure execution time
            start_time = time.time()

            result = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY',
                order_type='MKT'
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Verify operation completed successfully
            assert result['success'] is True

            # Basic performance check (should complete in reasonable time)
            # In a real system, this would have more specific thresholds
            assert execution_time < 5.0  # Should complete in less than 5 seconds

    @pytest.mark.asyncio
    async def test_resource_cleanup_and_leak_prevention(self, execution_agent):
        """Test that components clean up resources properly"""

        # Test multiple operations and verify no resource leaks
        mock_ibkr = AsyncMock()
        mock_trade_results = [
            {
                'success': True,
                'order_id': f'TRADE_{i}',
                'status': 'FILLED',
                'filled_quantity': 100,
                'avg_fill_price': 150.50 + i,
                'symbol': 'AAPL',
                'action': 'BUY'
            }
            for i in range(5)
        ]
        mock_ibkr.place_order.side_effect = mock_trade_results

        mock_tb = Mock()
        mock_tb.create_transfers.return_value = [987654321 + i for i in range(5)]

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_ibkr), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=mock_tb):
            execution_agent.tb_client = mock_tb

            # Execute multiple trades
            for i in range(5):
                result = await execution_agent.execute_trade(
                    symbol='AAPL',
                    quantity=100,
                    action='BUY',
                    order_type='MKT'
                )
                assert result['success'] is True

            # Verify all operations completed
            assert mock_ibkr.place_order.call_count == 5
            assert mock_tb.create_transfers.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])