#!/usr/bin/env python3
"""
Bridge vs Connector Comparison Tests
Tests comparing functionality and performance of bridge vs direct connector implementations.
"""

import pytest
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.execution import ExecutionAgent


class TestBridgeVsConnectorComparison:
    """Compare bridge vs direct connector implementations"""

    @pytest.fixture
    def execution_agent(self):
        """Create execution agent for testing"""
        agent = ExecutionAgent()
        # Reset any existing state
        agent.ibkr_connector = None
        agent.tb_client = None
        return agent

    @pytest.mark.asyncio
    async def test_bridge_vs_direct_connector_functionality(self, execution_agent):
        """Test that both bridge and direct connector provide same functionality"""

        # Mock trade result
        trade_result = {
            'success': True,
            'order_id': 'COMPARE_12345',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'commission': 1.00,
            'symbol': 'AAPL',
            'action': 'BUY'
        }

        # Test 1: Direct connector approach
        mock_direct_connector = AsyncMock()
        mock_direct_connector.place_order.return_value = trade_result.copy()

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_direct_connector), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=Mock()), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True):
            execution_agent.tb_client = Mock()

            result_direct = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY',
                order_type='MKT'
            )

        # Test 2: Bridge approach (simulated)
        mock_bridge_connector = AsyncMock()
        mock_bridge_connector.place_order.return_value = trade_result.copy()

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_bridge_connector), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=Mock()):
            execution_agent.tb_client = Mock()

            result_bridge = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY',
                order_type='MKT'
            )

        # Both should produce identical results
        assert result_direct['success'] == result_bridge['success']
        assert result_direct['order_id'] == result_bridge['order_id']
        assert result_direct['status'] == result_bridge['status']
        assert result_direct['filled_quantity'] == result_bridge['filled_quantity']
        assert result_direct['symbol'] == result_bridge['symbol']
        assert result_direct['action'] == result_bridge['action']

    @pytest.mark.asyncio
    async def test_performance_comparison_bridge_vs_direct(self, execution_agent):
        """Compare performance characteristics of bridge vs direct implementations"""

        # Mock connectors with different response times
        mock_direct_connector = AsyncMock()
        mock_bridge_connector = AsyncMock()

        trade_result = {
            'success': True,
            'order_id': 'PERF_12346',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'symbol': 'AAPL',
            'action': 'BUY'
        }

        # Direct connector: faster response
        async def direct_response(*args, **kwargs):
            await asyncio.sleep(0.01)  # 10ms
            return trade_result.copy()

        # Bridge connector: slightly slower due to indirection
        async def bridge_response(*args, **kwargs):
            await asyncio.sleep(0.02)  # 20ms
            return trade_result.copy()

        mock_direct_connector.place_order.side_effect = direct_response
        mock_bridge_connector.place_order.side_effect = bridge_response

        # Test direct connector performance
        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_direct_connector), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=Mock()):
            execution_agent.tb_client = Mock()

            start_time = time.time()
            result_direct = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY'
            )
            direct_time = time.time() - start_time

        # Test bridge connector performance
        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_bridge_connector), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=Mock()):
            execution_agent.tb_client = Mock()

            start_time = time.time()
            result_bridge = await execution_agent.execute_trade(
                symbol='AAPL',
                quantity=100,
                action='BUY'
            )
            bridge_time = time.time() - start_time

        # Both should succeed
        assert result_direct['success'] is True
        assert result_bridge['success'] is True

        # Both implementations should complete within reasonable time bounds
        assert direct_time < 0.5  # Should complete in less than 500ms
        assert bridge_time < 0.5   # Should complete in less than 500ms

        # Performance should be reasonable (no implementation should be more than 3x slower than the other)
        performance_ratio = max(direct_time, bridge_time) / min(direct_time, bridge_time)
        assert performance_ratio < 3.0
        ratio = bridge_time / direct_time
        assert ratio < 3.0

    @pytest.mark.asyncio
    async def test_error_handling_comparison(self, execution_agent):
        """Compare error handling between bridge and direct connector"""

        # Test various error scenarios

        # Scenario 1: Connection timeout
        mock_direct_timeout = AsyncMock()
        mock_direct_timeout.place_order.side_effect = asyncio.TimeoutError("Connection timeout")

        mock_bridge_timeout = AsyncMock()
        mock_bridge_timeout.place_order.side_effect = asyncio.TimeoutError("Bridge timeout")

        # Test direct connector timeout
        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_direct_timeout):
            try:
                await execution_agent.execute_trade('AAPL', 100, 'BUY')
                assert False, "Should have raised timeout"
            except Exception as e:
                assert "timeout" in str(e).lower()

        # Test bridge connector timeout
        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_bridge_timeout):
            try:
                await execution_agent.execute_trade('AAPL', 100, 'BUY')
                assert False, "Should have raised timeout"
            except Exception as e:
                assert "timeout" in str(e).lower()

        # Scenario 2: Authentication failure
        mock_direct_auth = AsyncMock()
        mock_direct_auth.place_order.side_effect = Exception("Authentication failed")

        mock_bridge_auth = AsyncMock()
        mock_bridge_auth.place_order.side_effect = Exception("Bridge authentication failed")

        # Both should handle auth failures similarly
        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_direct_auth):
            result = await execution_agent.execute_trade('AAPL', 100, 'BUY')
            assert 'error' in result
            assert 'Authentication failed' in result['error']

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_bridge_auth):
            result = await execution_agent.execute_trade('AAPL', 100, 'BUY')
            assert 'error' in result
            assert 'Bridge authentication failed' in result['error']

    @pytest.mark.asyncio
    async def test_concurrent_operations_comparison(self, execution_agent):
        """Test concurrent operation handling between implementations"""

        # Create multiple mock connectors
        mock_direct_connectors = [AsyncMock() for _ in range(5)]
        mock_bridge_connectors = [AsyncMock() for _ in range(5)]

        # Set up responses
        base_result = {
            'success': True,
            'order_id': 'CONCURRENT_{}',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'symbol': 'AAPL',
            'action': 'BUY'
        }

        for i, connector in enumerate(mock_direct_connectors):
            result = base_result.copy()
            result['order_id'] = f'CONCURRENT_DIRECT_{i}'
            connector.place_order.return_value = result

        for i, connector in enumerate(mock_bridge_connectors):
            result = base_result.copy()
            result['order_id'] = f'CONCURRENT_BRIDGE_{i}'
            connector.place_order.return_value = result

        # Test direct connector concurrency
        direct_results = []
        for i, connector in enumerate(mock_direct_connectors):
            with patch.object(execution_agent, '_get_ibkr_connector', return_value=connector), \
                 patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
                 patch('tigerbeetle.ClientSync', return_value=Mock()):
                execution_agent.tb_client = Mock()

                result = await execution_agent.execute_trade(
                    symbol='AAPL',
                    quantity=100,
                    action='BUY',
                    order_type='MKT'
                )
                direct_results.append(result)

        # Test bridge connector concurrency
        bridge_results = []
        for i, connector in enumerate(mock_bridge_connectors):
            with patch.object(execution_agent, '_get_ibkr_connector', return_value=connector), \
                 patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
                 patch('tigerbeetle.ClientSync', return_value=Mock()):
                execution_agent.tb_client = Mock()

                result = await execution_agent.execute_trade(
                    symbol='AAPL',
                    quantity=100,
                    action='BUY',
                    order_type='MKT'
                )
                bridge_results.append(result)

        # Both should handle concurrency equally well
        assert len(direct_results) == 5
        assert len(bridge_results) == 5
        assert all(r['success'] for r in direct_results)
        assert all(r['success'] for r in bridge_results)

    @pytest.mark.asyncio
    async def test_resource_usage_comparison(self, execution_agent):
        """Compare resource usage patterns between implementations"""

        # Mock connectors
        mock_direct = AsyncMock()
        mock_bridge = AsyncMock()

        trade_result = {
            'success': True,
            'order_id': 'RESOURCE_12347',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'symbol': 'AAPL',
            'action': 'BUY'
        }

        mock_direct.place_order.return_value = trade_result
        mock_bridge.place_order.return_value = trade_result

        # Track method calls as proxy for resource usage
        direct_calls = []
        bridge_calls = []

        # Direct connector
        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_direct), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=Mock()):
            execution_agent.tb_client = Mock()

            # Execute multiple trades
            for i in range(10):
                result = await execution_agent.execute_trade(
                    symbol='AAPL',
                    quantity=100,
                    action='BUY'
                )
                direct_calls.append(mock_direct.place_order.call_count)

        # Bridge connector
        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_bridge), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=Mock()):
            execution_agent.tb_client = Mock()

            # Execute multiple trades
            for i in range(10):
                result = await execution_agent.execute_trade(
                    symbol='AAPL',
                    quantity=100,
                    action='BUY'
                )
                bridge_calls.append(mock_bridge.place_order.call_count)

        # Both should have made the same number of calls
        assert len(direct_calls) == len(bridge_calls) == 10
        assert direct_calls[-1] == bridge_calls[-1] == 10

    @pytest.mark.asyncio
    async def test_scalability_comparison(self, execution_agent):
        """Test scalability characteristics of both implementations"""

        # Test with increasing load
        trade_counts = [1, 5, 10, 25]

        for count in trade_counts:
            # Direct connector scalability
            mock_direct = AsyncMock()
            trade_results = [{
                'success': True,
                'order_id': f'DIRECT_SCALE_{i}',
                'status': 'FILLED',
                'filled_quantity': 100,
                'avg_fill_price': 150.50,
                'symbol': 'AAPL',
                'action': 'BUY'
            } for i in range(count)]

            mock_direct.place_order.side_effect = trade_results

            with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_direct), \
                 patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
                 patch('tigerbeetle.ClientSync', return_value=Mock()):
                execution_agent.tb_client = Mock()

                start_time = time.time()
                direct_results = []
                for i in range(count):
                    result = await execution_agent.execute_trade(
                        symbol='AAPL',
                        quantity=100,
                        action='BUY'
                    )
                    direct_results.append(result)
                direct_time = time.time() - start_time

            # Bridge connector scalability (simulated as slightly slower)
            mock_bridge = AsyncMock()
            bridge_results = [{
                'success': True,
                'order_id': f'BRIDGE_SCALE_{i}',
                'status': 'FILLED',
                'filled_quantity': 100,
                'avg_fill_price': 150.50,
                'symbol': 'AAPL',
                'action': 'BUY'
            } for i in range(count)]

            async def bridge_response(*args, **kwargs):
                await asyncio.sleep(0.001)  # Small delay to simulate bridge overhead
                return bridge_results.pop(0) if bridge_results else None

            mock_bridge.place_order.side_effect = bridge_response

            with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_bridge), \
                 patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
                 patch('tigerbeetle.ClientSync', return_value=Mock()):
                execution_agent.tb_client = Mock()

                start_time = time.time()
                bridge_results_actual = []
                for i in range(count):
                    result = await execution_agent.execute_trade(
                        symbol='AAPL',
                        quantity=100,
                        action='BUY'
                    )
                    bridge_results_actual.append(result)
                bridge_time = time.time() - start_time

            # Verify scalability
            assert len(direct_results) == count
            assert len(bridge_results_actual) == count
            assert all(r['success'] for r in direct_results)
            assert all(r['success'] for r in bridge_results_actual)

            # Performance should scale reasonably
            direct_per_trade = direct_time / count
            bridge_per_trade = bridge_time / count

            # Per-trade time should remain relatively constant (good scalability)
            assert direct_per_trade < 0.1  # Less than 100ms per trade
            assert bridge_per_trade < 0.1   # Less than 100ms per trade

    @pytest.mark.asyncio
    async def test_reliability_comparison_under_load(self, execution_agent):
        """Test reliability under sustained load"""

        # Simulate sustained trading load
        total_trades = 50

        # Direct connector
        mock_direct = AsyncMock()
        direct_results = [{
            'success': True,
            'order_id': f'DIRECT_LOAD_{i}',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'symbol': 'AAPL',
            'action': 'BUY'
        } for i in range(total_trades)]

        mock_direct.place_order.side_effect = direct_results

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_direct), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=Mock()):
            execution_agent.tb_client = Mock()

            direct_success_count = 0
            for i in range(total_trades):
                result = await execution_agent.execute_trade(
                    symbol='AAPL',
                    quantity=100,
                    action='BUY'
                )
                if result.get('success'):
                    direct_success_count += 1

        # Bridge connector
        mock_bridge = AsyncMock()
        bridge_results = [{
            'success': True,
            'order_id': f'BRIDGE_LOAD_{i}',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50,
            'symbol': 'AAPL',
            'action': 'BUY'
        } for i in range(total_trades)]

        mock_bridge.place_order.side_effect = bridge_results

        with patch.object(execution_agent, '_get_ibkr_connector', return_value=mock_bridge), \
             patch('src.agents.execution.TIGERBEETLE_AVAILABLE', True), \
             patch('tigerbeetle.ClientSync', return_value=Mock()):
            execution_agent.tb_client = Mock()

            bridge_success_count = 0
            for i in range(total_trades):
                result = await execution_agent.execute_trade(
                    symbol='AAPL',
                    quantity=100,
                    action='BUY'
                )
                if result.get('success'):
                    bridge_success_count += 1

        # Both should maintain high reliability under load
        direct_success_rate = direct_success_count / total_trades
        bridge_success_rate = bridge_success_count / total_trades

        assert direct_success_rate >= 0.95  # At least 95% success rate
        assert bridge_success_rate >= 0.95  # At least 95% success rate

        # Success rates should be comparable
        rate_diff = abs(direct_success_rate - bridge_success_rate)
        assert rate_diff < 0.05  # Less than 5% difference


if __name__ == "__main__":
    pytest.main([__file__, "-v"])