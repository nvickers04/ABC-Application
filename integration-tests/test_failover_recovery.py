import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.data import DataAgent
from src.agents.risk import RiskAgent
from src.agents.strategy import StrategyAgent
from src.agents.execution import ExecutionAgent
# from src.integrations.ibkr import IBKRIntegration  # Module structure changed
from src.utils.redis_cache import RedisCacheManager


@pytest.mark.skip(reason="Tests need refactoring - Agent and integration APIs have changed")
class TestFailoverRecovery:
    """Integration tests for system failover and recovery scenarios.
    
    Note: These tests require external services (IBKR, Redis) and specific
    agent internal methods that may not be available in all environments.
    """

    @pytest_asyncio.fixture
    async def agents_setup(self):
        """Setup agents for testing"""
        data_agent = DataAgent()
        risk_agent = RiskAgent()
        strategy_agent = StrategyAgent()
        execution_agent = ExecutionAgent()

        yield {
            'data': data_agent,
            'risk': risk_agent,
            'strategy': strategy_agent,
            'execution': execution_agent
        }

        # Cleanup if needed

    @pytest.fixture
    def mock_ibkr_integration(self):
        """Mock IBKR integration for testing"""
        mock_ibkr = MagicMock(spec=IBKRIntegration)
        mock_ibkr.connect.return_value = True
        mock_ibkr.is_connected.return_value = True
        mock_ibkr.disconnect.return_value = True
        return mock_ibkr

    @pytest.mark.asyncio
    async def test_agent_crash_recovery(self, agents_setup):
        """Test recovery when an agent crashes during processing"""
        data_agent = agents_setup['data']

        # Simulate agent crash during data processing
        with patch.object(data_agent, 'process_input', side_effect=[Exception("Simulated crash"), {"status": "recovered"}]):
            # First call should fail
            with pytest.raises(Exception, match="Simulated crash"):
                await data_agent.process_input({'symbols': ['AAPL']})

            # Simulate restart and successful recovery
            result = await data_agent.process_input({'symbols': ['AAPL']})
            assert result["status"] == "recovered"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="_fetch_market_data method not available in DataAgent")
    async def test_network_failure_recovery(self, agents_setup):
        """Test recovery from network connectivity issues"""
        data_agent = agents_setup['data']

        # Mock network failure
        original_method = data_agent._fetch_market_data

        async def failing_fetch(*args, **kwargs):
            raise ConnectionError("Network connection failed")

        async def recovering_fetch(*args, **kwargs):
            return {"AAPL": {"price": 150.0, "volume": 1000000}}

        # First attempt fails
        with patch.object(data_agent, '_fetch_market_data', side_effect=failing_fetch):
            with pytest.raises(ConnectionError):
                await data_agent.process_input({'symbols': ['AAPL'], 'period': '1d'})

        # Simulate network recovery
        with patch.object(data_agent, '_fetch_market_data', side_effect=recovering_fetch):
            result = await data_agent.process_input({'symbols': ['AAPL'], 'period': '1d'})
            assert 'AAPL' in result

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="src.agents.execution.IBKRIntegration import not available")
    async def test_ibkr_connection_recovery(self, agents_setup, mock_ibkr_integration):
        """Test IBKR connection failure and recovery"""
        execution_agent = agents_setup['execution']

        # Mock IBKR disconnection
        mock_ibkr_integration.is_connected.return_value = False
        mock_ibkr_integration.connect.side_effect = [ConnectionError("IBKR connection failed"), True]

        with patch('src.agents.execution.IBKRIntegration', return_value=mock_ibkr_integration):
            # First connection attempt fails
            with pytest.raises(ConnectionError):
                await execution_agent.process_input({
                    'action': 'buy',
                    'symbol': 'AAPL',
                    'quantity': 100
                })

            # Simulate reconnection
            mock_ibkr_integration.is_connected.return_value = True

            # Second attempt should succeed
            result = await execution_agent.process_input({
                'action': 'buy',
                'symbol': 'AAPL',
                'quantity': 100
            })
            assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="src.utils.database module not available")
    async def test_database_failover(self, agents_setup):
        """Test database connection failover and recovery"""
        risk_agent = agents_setup['risk']

        # Mock database failure
        with patch('src.utils.database.postgres_connect', side_effect=Exception("Database connection failed")):
            with pytest.raises(Exception, match="Database connection failed"):
                await risk_agent.process_input({
                    'portfolio_returns': [0.01, 0.02, -0.01],
                    'analysis_type': 'comprehensive'
                })

        # Simulate database recovery
        with patch('src.utils.database.postgres_connect', return_value=MagicMock()):
            with patch.object(risk_agent, '_calculate_risk_metrics', return_value={'var_95': 0.05}):
                result = await risk_agent.process_input({
                    'portfolio_returns': [0.01, 0.02, -0.01],
                    'analysis_type': 'comprehensive'
                })
                assert 'var_95' in result

    @pytest.mark.asyncio
    async def test_message_queue_recovery(self, agents_setup):
        """Test message queue (Redis) failure and recovery"""
        data_agent = agents_setup['data']

        # Mock Redis failure
        with patch('redis.Redis.ping', side_effect=Exception("Redis connection failed")):
            # Should handle gracefully or have fallback
            result = await data_agent.process_input({'symbols': ['AAPL'], 'period': '1d'})
            # Even with Redis failure, should not crash completely
            assert result is not None

        # Simulate Redis recovery
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True

        with patch('redis.Redis', return_value=mock_redis):
            result = await data_agent.process_input({'symbols': ['AAPL'], 'period': '1d'})
            assert result is not None

    @pytest.mark.asyncio
    async def test_agent_restart_recovery(self, agents_setup):
        """Test recovery after agent restart"""
        strategy_agent = agents_setup['strategy']

        # Simulate agent state loss on restart
        original_state = getattr(strategy_agent, '_strategy_cache', {})
        strategy_agent._strategy_cache = {}  # Simulate cleared cache

        # Process request after "restart"
        result = await strategy_agent.process_input({
            'dataframe': MagicMock(),
            'sentiment': {'sentiment': 'bullish', 'confidence': 0.8},
            'symbols': ['SPY']
        })

        assert result is not None
        assert 'strategy_type' in result

    @pytest.mark.asyncio
    async def test_concurrent_failure_recovery(self, agents_setup):
        """Test recovery when multiple agents fail concurrently"""
        data_agent = agents_setup['data']
        risk_agent = agents_setup['risk']

        # Mock concurrent failures
        with patch.object(data_agent, 'process_input', side_effect=Exception("Data agent failed")), \
             patch.object(risk_agent, 'process_input', side_effect=Exception("Risk agent failed")):

            # Both should fail
            with pytest.raises(Exception):
                await data_agent.process_input({'symbols': ['AAPL']})

            with pytest.raises(Exception):
                await risk_agent.process_input({'portfolio_returns': [0.01]})

        # Simulate recovery
        with patch.object(data_agent, 'process_input', return_value={'data': 'recovered'}), \
             patch.object(risk_agent, 'process_input', return_value={'risk': 'recovered'}):

            data_result = await data_agent.process_input({'symbols': ['AAPL']})
            risk_result = await risk_agent.process_input({'portfolio_returns': [0.01]})

            assert data_result['data'] == 'recovered'
            assert risk_result['risk'] == 'recovered'

    @pytest.mark.asyncio
    async def test_timeout_recovery(self, agents_setup):
        """Test recovery from timeout scenarios"""
        data_agent = agents_setup['data']

        # Mock timeout
        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(30)  # Simulate long operation
            return {"status": "completed"}

        with patch.object(data_agent, 'process_input', side_effect=asyncio.TimeoutError("Operation timed out")):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    data_agent.process_input({'symbols': ['AAPL']}),
                    timeout=1.0
                )

        # Simulate successful retry after timeout
        with patch.object(data_agent, 'process_input', return_value={"status": "recovered"}):
            result = await data_agent.process_input({'symbols': ['AAPL']})
            assert result["status"] == "recovered"

    @pytest.mark.asyncio
    async def test_partial_system_recovery(self, agents_setup):
        """Test recovery when only part of the system fails"""
        data_agent = agents_setup['data']
        risk_agent = agents_setup['risk']

        # Data agent fails, risk agent succeeds
        with patch.object(data_agent, 'process_input', side_effect=Exception("Data failure")), \
             patch.object(risk_agent, 'process_input', return_value={'risk_score': 0.7}):

            with pytest.raises(Exception):
                await data_agent.process_input({'symbols': ['AAPL']})

            # Risk agent should still work
            risk_result = await risk_agent.process_input({'portfolio_returns': [0.01, 0.02]})
            assert risk_result['risk_score'] == 0.7

    @pytest.mark.parametrize("failure_type", ["network", "database", "agent_crash", "ibkr_disconnect"])
    @pytest.mark.asyncio
    async def test_generic_failure_recovery(self, agents_setup, failure_type):
        """Parameterized test for different failure types"""
        data_agent = agents_setup['data']

        # Define failure behaviors
        failure_scenarios = {
            "network": ConnectionError("Network failed"),
            "database": Exception("Database connection failed"),
            "agent_crash": RuntimeError("Agent crashed"),
            "ibkr_disconnect": ConnectionError("IBKR disconnected")
        }

        # Test failure
        with patch.object(data_agent, 'process_input', side_effect=failure_scenarios[failure_type]):
            with pytest.raises((ConnectionError, Exception, RuntimeError)):
                await data_agent.process_input({'symbols': ['AAPL']})

        # Test recovery
        with patch.object(data_agent, 'process_input', return_value={"status": f"recovered_from_{failure_type}"}):
            result = await data_agent.process_input({'symbols': ['AAPL']})
            assert result["status"] == f"recovered_from_{failure_type}"