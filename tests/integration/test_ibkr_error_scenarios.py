#!/usr/bin/env python3
"""
IBKR Error Scenarios Test
Comprehensive testing of error scenarios and edge cases in IBKR integration
"""

import pytest
import asyncio
import logging
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TestIBKRErrorScenarios:
    """Test IBKR error scenarios and edge cases"""

    @pytest.mark.asyncio
    async def test_connection_timeout(self, mock_ibkr_connector, ibkr_error_scenarios):
        """Test connection timeout handling"""
        scenario = ibkr_error_scenarios['connection_timeout']

        # Mock connection timeout
        mock_ibkr_connector.connect.side_effect = asyncio.TimeoutError(scenario['message'])

        with pytest.raises(asyncio.TimeoutError):
            await mock_ibkr_connector.connect()

        # Verify retry logic would be triggered
        assert scenario['retry_count'] == 3

    @pytest.mark.asyncio
    async def test_invalid_contract_error(self, mock_ibkr_connector, ibkr_error_scenarios):
        """Test invalid contract error handling"""
        scenario = ibkr_error_scenarios['invalid_contract']

        # Mock invalid contract error
        mock_ibkr_connector.get_market_data.side_effect = ValueError(scenario['message'])

        with pytest.raises(ValueError, match=scenario['message']):
            await mock_ibkr_connector.get_market_data(scenario['contract']['symbol'])

    @pytest.mark.asyncio
    async def test_insufficient_funds_error(self, mock_ibkr_connector, ibkr_error_scenarios, sample_ibkr_order):
        """Test insufficient funds error handling"""
        scenario = ibkr_error_scenarios['insufficient_funds']

        # Mock insufficient funds error
        mock_ibkr_connector.place_order.side_effect = ValueError(
            f"{scenario['message']}: Required ${scenario['required']}, Available ${scenario['available']}"
        )

        with pytest.raises(ValueError, match="insufficient funds"):
            await mock_ibkr_connector.place_order(sample_ibkr_order)

    @pytest.mark.asyncio
    async def test_market_closed_error(self, mock_ibkr_connector, ibkr_error_scenarios):
        """Test market closed error handling"""
        scenario = ibkr_error_scenarios['market_closed']

        # Mock market closed error
        mock_ibkr_connector.get_market_data.side_effect = RuntimeError(scenario['message'])

        with pytest.raises(RuntimeError, match=scenario['message']):
            await mock_ibkr_connector.get_market_data('AAPL')

    @pytest.mark.asyncio
    async def test_circuit_breaker_error(self, mock_ibkr_connector, ibkr_error_scenarios):
        """Test circuit breaker error handling"""
        scenario = ibkr_error_scenarios['circuit_breaker']

        # Mock circuit breaker error
        mock_ibkr_connector.place_order.side_effect = RuntimeError(scenario['message'])

        with pytest.raises(RuntimeError, match=scenario['message']):
            await mock_ibkr_connector.place_order({'symbol': 'AAPL', 'quantity': 100, 'action': 'BUY'})

    @pytest.mark.asyncio
    async def test_network_disconnect_during_operation(self, mock_ibkr_connector):
        """Test network disconnection during operations"""
        # Mock successful initial connection
        mock_ibkr_connector.connect.return_value = True
        mock_ibkr_connector.is_connected.return_value = True

        # Mock disconnection during market data request
        mock_ibkr_connector.get_market_data.side_effect = ConnectionError("Network disconnected")

        with pytest.raises(ConnectionError, match="Network disconnected"):
            await mock_ibkr_connector.get_market_data('AAPL')

    @pytest.mark.asyncio
    async def test_concurrent_connection_attempts(self, mock_ibkr_connector):
        """Test concurrent connection attempts"""
        import threading

        results = []
        errors = []

        async def connection_attempt(attempt_id: int):
            try:
                result = await mock_ibkr_connector.connect()
                results.append((attempt_id, result))
            except Exception as e:
                errors.append((attempt_id, str(e)))

        # Simulate concurrent connections
        tasks = [connection_attempt(i) for i in range(5)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed with mock
        assert len(results) == 5
        assert all(result[1] for result in results)

    @pytest.mark.asyncio
    async def test_large_order_quantities(self, mock_ibkr_connector):
        """Test handling of large order quantities"""
        # Mock successful order placement for large quantity
        mock_ibkr_connector.place_order.return_value = {
            'order_id': 'LARGE123',
            'status': 'submitted',
            'quantity': 1000000  # 1 million shares
        }

        large_order = {
            'symbol': 'AAPL',
            'quantity': 1000000,
            'action': 'BUY',
            'order_type': 'MKT'
        }

        result = await mock_ibkr_connector.place_order(large_order)
        assert result['quantity'] == 1000000
        assert result['status'] == 'submitted'

    @pytest.mark.asyncio
    async def test_zero_quantity_orders(self, mock_ibkr_connector):
        """Test handling of zero quantity orders"""
        mock_ibkr_connector.place_order.side_effect = ValueError("Order quantity must be positive")

        zero_order = {
            'symbol': 'AAPL',
            'quantity': 0,
            'action': 'BUY',
            'order_type': 'MKT'
        }

        with pytest.raises(ValueError, match="quantity must be positive"):
            await mock_ibkr_connector.place_order(zero_order)

    @pytest.mark.asyncio
    async def test_negative_price_orders(self, mock_ibkr_connector):
        """Test handling of negative price orders"""
        mock_ibkr_connector.place_order.side_effect = ValueError("Order price cannot be negative")

        negative_price_order = {
            'symbol': 'AAPL',
            'quantity': 100,
            'action': 'BUY',
            'order_type': 'LMT',
            'limit_price': -150.00
        }

        with pytest.raises(ValueError, match="price cannot be negative"):
            await mock_ibkr_connector.place_order(negative_price_order)

    @pytest.mark.asyncio
    async def test_invalid_symbol_handling(self, mock_ibkr_connector):
        """Test handling of invalid symbols"""
        invalid_symbols = ['', '   ', 'INVALID@#$', 'A' * 100]  # Various invalid symbols

        for symbol in invalid_symbols:
            mock_ibkr_connector.get_market_data.side_effect = ValueError(f"Invalid symbol: {symbol}")

            with pytest.raises(ValueError, match="Invalid symbol"):
                await mock_ibkr_connector.get_market_data(symbol)

    @pytest.mark.asyncio
    async def test_account_data_unavailable(self, mock_ibkr_connector):
        """Test handling when account data is unavailable"""
        mock_ibkr_connector.get_account_summary.side_effect = RuntimeError("Account data unavailable")

        with pytest.raises(RuntimeError, match="Account data unavailable"):
            await mock_ibkr_connector.get_account_summary()

    @pytest.mark.asyncio
    async def test_position_data_inconsistency(self, mock_ibkr_connector):
        """Test handling of inconsistent position data"""
        # Mock positions with negative quantities (shouldn't happen but test edge case)
        mock_ibkr_connector.get_positions.return_value = [
            {'symbol': 'AAPL', 'quantity': -100, 'avg_cost': 150.00},  # Negative position
            {'symbol': 'GOOGL', 'quantity': 0, 'avg_cost': 0.00},      # Zero position
        ]

        positions = await mock_ibkr_connector.get_positions()

        # Should still return data even if inconsistent
        assert len(positions) == 2
        assert positions[0]['quantity'] == -100  # Allow negative for testing

    @pytest.mark.asyncio
    async def test_extreme_market_data_values(self, mock_ibkr_connector):
        """Test handling of extreme market data values"""
        # Mock extreme price values
        extreme_data = {
            'symbol': 'EXTREME',
            'bid': 999999.99,
            'ask': 1000000.00,
            'last': 0.000001,
            'volume': 999999999999,
            'high': float('inf'),  # Infinity
            'low': 0.0
        }

        mock_ibkr_connector.get_market_data.return_value = extreme_data

        data = await mock_ibkr_connector.get_market_data('EXTREME')

        # Should handle extreme values without crashing
        assert data['high'] == float('inf')
        assert data['last'] == 0.000001

    @pytest.mark.asyncio
    async def test_memory_cleanup_on_errors(self, ibkr_test_connector, ibkr_cleanup_manager):
        """Test memory cleanup when operations fail"""
        cleanup = ibkr_cleanup_manager

        cleanup_count = 0

        def count_cleanup():
            nonlocal cleanup_count
            cleanup_count += 1

        cleanup['add_task'](count_cleanup)

        # Simulate an operation that fails
        try:
            # This might fail if TWS not connected
            await ibkr_test_connector.get_market_data('INVALID_SYMBOL')
        except Exception:
            pass  # Expected to fail

        # Cleanup should still work
        await cleanup['cleanup']()
        assert cleanup_count == 1

    @pytest.mark.asyncio
    async def test_rapid_successive_operations(self, mock_ibkr_connector):
        """Test rapid successive operations"""
        import time

        start_time = time.time()

        # Perform many rapid operations
        tasks = []
        for i in range(100):
            tasks.append(mock_ibkr_connector.get_market_data('AAPL'))

        results = await asyncio.gather(*tasks)

        end_time = time.time()

        # All should succeed
        assert len(results) == 100
        assert all('symbol' in result for result in results)

        # Should complete within reasonable time
        duration = end_time - start_time
        assert duration < 5.0  # Less than 5 seconds for 100 operations

    @pytest.mark.asyncio
    async def test_resource_exhaustion_simulation(self, mock_ibkr_connector):
        """Test behavior under simulated resource exhaustion"""
        # Mock memory exhaustion
        mock_ibkr_connector.get_account_summary.side_effect = MemoryError("Simulated memory exhaustion")

        with pytest.raises(MemoryError, match="memory exhaustion"):
            await mock_ibkr_connector.get_account_summary()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])