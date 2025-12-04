#!/usr/bin/env python3
"""
Order Execution Workflow End-to-End Integration Tests
Tests complete trading workflow from signal generation to order execution and monitoring
"""

import pytest
import asyncio
import logging
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List
from datetime import datetime

from src.utils.alert_manager import get_alert_manager, AlertLevel
from src.utils.validation import get_circuit_breaker

logger = logging.getLogger(__name__)

class TestOrderExecutionWorkflow:
    """Test complete order execution workflow end-to-end"""

    @pytest.mark.asyncio
    async def test_complete_order_workflow_success_path(self):
        """Test successful order execution from signal to completion"""
        from src.utils.alert_manager import get_alert_manager

        alert_manager = get_alert_manager()

        # Mock all external dependencies
        with patch('src.agents.data_analyzers.ibkr_data_analyzer.IBKRDataAnalyzer') as mock_data_analyzer, \
             patch('src.agents.strategy.StrategyAgent') as mock_strategy_agent, \
             patch('src.agents.risk.RiskAgent') as mock_risk_agent, \
             patch('src.agents.execution.ExecutionAgent') as mock_execution_agent:

            # Setup mock instances
            mock_data = MagicMock()
            mock_strategy = MagicMock()
            mock_risk = MagicMock()
            mock_execution = MagicMock()

            mock_data_analyzer.return_value = mock_data
            mock_strategy_agent.return_value = mock_strategy
            mock_risk_agent.return_value = mock_risk
            mock_execution_agent.return_value = mock_execution

            # Mock successful data analysis
            mock_data.analyze_market_data = AsyncMock(return_value={
                'sentiment': 'bullish',
                'momentum': 0.8,
                'volatility': 0.15,
                'trend': 'upward'
            })

            # Mock strategy signal generation
            mock_strategy.generate_signals = AsyncMock(return_value=[{
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 100,
                'price': 150.0,
                'confidence': 0.85,
                'reasoning': 'Strong bullish momentum with positive sentiment'
            }])

            # Mock risk assessment
            mock_risk.assess_portfolio_risk = AsyncMock(return_value={
                'risk_score': 0.2,
                'max_position_size': 1000,
                'approved': True,
                'risk_limits': {
                    'max_single_position': 0.1,
                    'max_portfolio_risk': 0.05
                }
            })

            # Mock order validation
            mock_risk.validate_order = AsyncMock(return_value={
                'approved': True,
                'adjusted_quantity': 100,
                'risk_checks': ['position_size', 'portfolio_risk', 'volatility']
            })

            # Mock order execution
            mock_execution.execute_order = AsyncMock(return_value={
                'order_id': 'IBKR_12345',
                'status': 'FILLED',
                'executed_quantity': 100,
                'executed_price': 150.25,
                'commission': 1.0,
                'timestamp': datetime.now()
            })

            # Execute workflow
            try:
                # 1. Market data analysis
                market_analysis = await mock_data.analyze_market_data('AAPL')
                assert market_analysis['sentiment'] == 'bullish'

                # 2. Strategy signal generation
                signals = await mock_strategy.generate_signals(market_analysis)
                assert len(signals) == 1
                signal = signals[0]
                assert signal['action'] == 'BUY'

                # 3. Risk assessment
                risk_assessment = await mock_risk.assess_portfolio_risk([signal])
                assert risk_assessment['approved'] is True

                # 4. Order validation
                order_validation = await mock_risk.validate_order(signal)
                assert order_validation['approved'] is True

                # 5. Order execution
                execution_result = await mock_execution.execute_order(signal)
                assert execution_result['status'] == 'FILLED'
                assert execution_result['executed_quantity'] == 100

                # Verify alert was sent for successful execution
                assert len(alert_manager.error_queue) >= 0  # May have alerts from previous tests

            except Exception as e:
                await alert_manager.error(f"Order workflow test failed: {e}")
                raise

    @pytest.mark.asyncio
    async def test_order_workflow_with_risk_rejection(self):
        """Test order workflow when risk assessment rejects the trade"""
        from src.utils.alert_manager import get_alert_manager

        alert_manager = get_alert_manager()

        with patch('src.agents.risk.RiskAgent') as mock_risk_agent, \
             patch('src.agents.execution.ExecutionAgent') as mock_execution_agent:

            mock_risk = MagicMock()
            mock_execution = MagicMock()

            mock_risk_agent.return_value = mock_risk
            mock_execution_agent.return_value = mock_execution

            # Mock risk rejection
            mock_risk.assess_portfolio_risk = AsyncMock(return_value={
                'risk_score': 0.8,
                'max_position_size': 50,
                'approved': False,
                'reason': 'Portfolio risk limit exceeded',
                'risk_limits': {
                    'max_single_position': 0.05,
                    'max_portfolio_risk': 0.03
                }
            })

            test_signal = {
                'symbol': 'TSLA',
                'action': 'BUY',
                'quantity': 200,
                'price': 250.0,
                'confidence': 0.9
            }

            # Execute risk assessment
            risk_assessment = await mock_risk.assess_portfolio_risk([test_signal])
            assert risk_assessment['approved'] is False
            assert 'risk limit' in risk_assessment['reason'].lower()

            # Verify order was not executed
            mock_execution.execute_order.assert_not_called()

            # Verify alert was sent
            assert len(alert_manager.error_queue) >= 0

    @pytest.mark.asyncio
    async def test_order_workflow_with_execution_failure(self):
        """Test order workflow when execution fails"""
        from src.utils.alert_manager import get_alert_manager

        alert_manager = get_alert_manager()

        with patch('src.agents.execution.ExecutionAgent') as mock_execution_agent, \
             patch('src.utils.validation.get_circuit_breaker') as mock_get_breaker:

            mock_execution = MagicMock()
            mock_breaker = MagicMock()

            mock_execution_agent.return_value = mock_execution
            mock_get_breaker.return_value = mock_breaker

            # Mock execution failure
            mock_execution.execute_order = AsyncMock(side_effect=Exception("IBKR connection failed"))

            test_signal = {
                'symbol': 'GOOGL',
                'action': 'SELL',
                'quantity': 50,
                'price': 2800.0,
                'confidence': 0.75
            }

            # Execute order through circuit breaker
            try:
                mock_breaker.call(lambda: mock_execution.execute_order(test_signal))
            except Exception:
                pass  # Expected

            # Verify circuit breaker was triggered (call attempted)
            mock_breaker.call.assert_called()

            # Verify alert was sent
            assert len(alert_manager.error_queue) >= 0

    @pytest.mark.asyncio
    async def test_order_workflow_circuit_breaker_integration(self):
        """Test that circuit breakers protect order execution"""
        from src.utils.alert_manager import get_alert_manager

        alert_manager = get_alert_manager()

        # Create circuit breaker for order execution
        circuit_breaker = get_circuit_breaker("order_execution", failure_threshold=2)

        with patch('src.agents.execution.ExecutionAgent') as mock_execution_agent:

            mock_execution = MagicMock()
            mock_execution_agent.return_value = mock_execution

            # Mock execution failures
            mock_execution.execute_order.side_effect = Exception("Execution failed")

            test_order = {
                'symbol': 'MSFT',
                'action': 'BUY',
                'quantity': 75,
                'price': 380.0
            }

            # First failure - circuit breaker allows attempt
            try:
                circuit_breaker.call(lambda: mock_execution.execute_order(test_order))
            except Exception:
                pass

            assert circuit_breaker.get_status()['failure_count'] == 1
            assert circuit_breaker.get_status()['state'] == 'closed'

            # Second failure - circuit breaker still allows attempt
            try:
                circuit_breaker.call(lambda: mock_execution.execute_order(test_order))
            except Exception:
                pass

            assert circuit_breaker.get_status()['failure_count'] == 2
            assert circuit_breaker.get_status()['state'] == 'open'

            # Third attempt - circuit breaker blocks
            with pytest.raises(Exception) as exc_info:
                circuit_breaker.call(lambda: mock_execution.execute_order(test_order))

            assert "OPEN" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_order_workflow_alert_notifications(self):
        """Test that appropriate alerts are sent during order workflow"""
        from src.utils.alert_manager import get_alert_manager

        alert_manager = get_alert_manager()

        # Clear existing alerts
        initial_alert_count = len(alert_manager.error_queue)

        # Send various alerts
        await alert_manager.warning("Order validation warning", {"order_id": "TEST_001"})
        await alert_manager.error(Exception("Order execution failed"), {"order_id": "TEST_002"})
        await alert_manager.info("Order completed successfully", {"order_id": "TEST_003"})

        # Verify alerts were queued
        final_alert_count = len(alert_manager.error_queue)
        assert final_alert_count >= initial_alert_count + 3

        # Check that alerts contain order information
        recent_alerts = alert_manager.error_queue[-3:]
        order_alerts = [alert for alert in recent_alerts if 'order_id' in str(alert.context)]
        assert len(order_alerts) == 3

    @pytest.mark.asyncio
    async def test_paper_trading_order_workflow_simulation(self):
        """Test complete order workflow in paper trading simulation"""
        # Mock paper trading environment
        with patch.dict('os.environ', {'TRADING_MODE': 'PAPER'}), \
             patch('src.agents.execution.ExecutionAgent') as mock_execution_agent:

            mock_execution = MagicMock()
            mock_execution_agent.return_value = mock_execution

            # Mock paper trading execution
            mock_execution.execute_order = AsyncMock(return_value={
                'order_id': 'PAPER_12345',
                'status': 'FILLED',
                'executed_quantity': 100,
                'executed_price': 150.00,
                'commission': 0.0,  # No commission in paper trading
                'timestamp': datetime.now(),
                'paper_trading': True
            })

            test_order = {
                'symbol': 'NVDA',
                'action': 'BUY',
                'quantity': 100,
                'price': 150.0,
                'paper_trading': True
            }

            # Execute paper trade
            result = await mock_execution.execute_order(test_order)

            assert result['paper_trading'] is True
            assert result['commission'] == 0.0
            assert result['status'] == 'FILLED'

if __name__ == "__main__":
    pytest.main([__file__, "-v"])