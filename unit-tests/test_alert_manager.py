#!/usr/bin/env python3
"""
Unit tests for AlertManager and related components.
Tests alert creation, queue management, Discord integration, and resilience patterns.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Add src to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.alert_manager import (
    AlertManager, Alert, AlertLevel, get_alert_manager,
    ConnectionError, AuthenticationError, RateLimitError, DataQualityError, ValidationGateError,
    ValidationGate, CircuitBreaker, retry_with_backoff, graceful_degradation
)


class TestAlertManager:
    """Test AlertManager core functionality"""

    def setup_method(self):
        """Reset alert manager before each test"""
        manager = get_alert_manager()
        manager.clear_alerts()

    def test_alert_creation(self):
        """Test creating alerts with different levels"""
        alert = Alert(
            level=AlertLevel.ERROR,
            component="TestComponent",
            message="Test message",
            context={"key": "value"}
        )

        assert alert.level == AlertLevel.ERROR
        assert alert.component == "TestComponent"
        assert alert.message == "Test message"
        assert alert.context == {"key": "value"}
        assert isinstance(alert.timestamp, datetime)
        assert alert.error_id is None

    @pytest.mark.asyncio
    async def test_alert_manager_send_alert(self):
        """Test sending alerts to alert manager"""
        manager = get_alert_manager()

        # Send alert
        await manager.send_alert(
            level=AlertLevel.WARNING,
            component="TestComponent",
            message="Test alert",
            context={"test": True}
        )

        # Check alert was added to queue
        assert len(manager.error_queue) == 1
        alert = manager.error_queue[0]
        assert alert.level == AlertLevel.WARNING
        assert alert.component == "TestComponent"
        assert alert.message == "Test alert"
        assert alert.context == {"test": True}

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality"""
        manager = get_alert_manager()

        # Add some test alerts
        await manager.send_alert(AlertLevel.CRITICAL, "Test", "Critical alert")
        await manager.send_alert(AlertLevel.ERROR, "Test", "Error alert")
        await manager.send_alert(AlertLevel.WARNING, "Test", "Warning alert")

        health = manager.check_health()

        assert health['alert_queue_size'] == 3
        assert health['recent_critical_alerts'] == 1
        assert health['recent_error_alerts'] == 2
        assert not health['orchestrator_connected']  # No orchestrator set
        assert not health['discord_enabled']  # No orchestrator set

    @pytest.mark.asyncio
    async def test_get_recent_alerts(self):
        """Test retrieving recent alerts"""
        manager = get_alert_manager()

        # Add alerts
        await manager.send_alert(AlertLevel.INFO, "Test1", "Message 1")
        await manager.send_alert(AlertLevel.ERROR, "Test2", "Message 2")
        await manager.send_alert(AlertLevel.WARNING, "Test3", "Message 3")

        recent = manager.get_recent_alerts(2)
        assert len(recent) == 2
        assert recent[0].message == "Message 2"  # Second most recent
        assert recent[1].message == "Message 3"  # Most recent

    @pytest.mark.asyncio
    async def test_clear_alerts(self):
        """Test clearing alert queue"""
        manager = get_alert_manager()

        await manager.send_alert(AlertLevel.ERROR, "Test", "Test alert")
        assert len(manager.error_queue) == 1

        manager.clear_alerts()
        assert len(manager.error_queue) == 0


class TestExceptionClasses:
    """Test custom exception classes"""

    def test_connection_error(self):
        """Test ConnectionError exception"""
        with pytest.raises(ConnectionError):
            raise ConnectionError("Connection failed")

    def test_authentication_error(self):
        """Test AuthenticationError exception"""
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("Auth failed")

    def test_rate_limit_error(self):
        """Test RateLimitError exception"""
        with pytest.raises(RateLimitError):
            raise RateLimitError("Rate limited")

    def test_data_quality_error(self):
        """Test DataQualityError exception"""
        with pytest.raises(DataQualityError):
            raise DataQualityError("Bad data")

    def test_validation_gate_error(self):
        """Test ValidationGateError exception"""
        with pytest.raises(ValidationGateError):
            raise ValidationGateError("Validation failed")


class TestValidationGate:
    """Test ValidationGate functionality"""

    def test_validation_gate_creation(self):
        """Test creating a validation gate"""
        gate = ValidationGate("TestGate")
        assert gate.name == "TestGate"
        assert gate.validation_results == []

    def test_validation_success(self):
        """Test successful validation"""
        gate = ValidationGate("TestGate")

        def check_true():
            return True

        def check_false():
            return False

        result = gate.validate({"check1": check_true, "check2": check_false})

        assert not result['all_passed']
        assert result['results']['check1']['passed'] is True
        assert result['results']['check2']['passed'] is False
        assert len(gate.validation_results) == 1

    def test_validation_enforce_success(self):
        """Test enforce method with successful validation"""
        gate = ValidationGate("TestGate")

        def check_pass():
            return True

        # Should not raise
        result = gate.enforce({"check": check_pass}, raise_on_failure=True)
        assert result is True

    def test_validation_enforce_failure(self):
        """Test enforce method with failed validation"""
        # Mock alert manager to avoid async issues
        mock_manager = Mock()
        gate = ValidationGate("TestGate", alert_manager=mock_manager)

        def check_fail():
            return False

        # Should raise ValidationGateError
        with pytest.raises(ValidationGateError):
            gate.enforce({"check": check_fail}, raise_on_failure=True)

    def test_get_recent_results(self):
        """Test getting recent validation results"""
        gate = ValidationGate("TestGate")

        def check():
            return True

        gate.validate({"check": check})
        gate.validate({"check": check})

        results = gate.get_recent_results(1)
        assert len(results) == 1


class TestCircuitBreaker:
    """Test CircuitBreaker functionality"""

    def test_circuit_breaker_creation(self):
        """Test creating a circuit breaker"""
        cb = CircuitBreaker("TestBreaker")
        assert cb.name == "TestBreaker"
        assert cb.state == 'CLOSED'
        assert cb.failure_count == 0
        assert cb.last_failure_time is None

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls"""
        cb = CircuitBreaker("TestBreaker")

        async def success_func():
            return "success"

        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == 'CLOSED'
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self):
        """Test circuit breaker with failed calls"""
        cb = CircuitBreaker("TestBreaker", failure_threshold=2)

        async def fail_func():
            raise ConnectionError("Failed")

        # First failure
        with pytest.raises(ConnectionError):
            await cb.call(fail_func)
        assert cb.state == 'CLOSED'
        assert cb.failure_count == 1

        # Second failure - should open circuit
        with pytest.raises(ConnectionError):
            await cb.call(fail_func)
        assert cb.state == 'OPEN'
        assert cb.failure_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self):
        """Test circuit breaker blocks calls when open"""
        cb = CircuitBreaker("TestBreaker", failure_threshold=1)

        async def fail_func():
            raise ConnectionError("Failed")

        # Open the circuit
        with pytest.raises(ConnectionError):
            await cb.call(fail_func)
        assert cb.state == 'OPEN'

        # Next call should be blocked
        async def success_func():
            return "success"

        with pytest.raises(ConnectionError):
            await cb.call(success_func)

    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset after timeout"""
        # Mock alert manager to avoid async issues
        mock_manager = Mock()
        cb = CircuitBreaker("TestBreaker", failure_threshold=1, recovery_timeout=1, alert_manager=mock_manager)

        # Simulate failure and open circuit
        cb._on_failure()
        assert cb.state == 'OPEN'

        # Wait for recovery timeout
        import time
        time.sleep(1.1)  # Wait longer than the 1 second timeout

        # Should allow attempt now
        assert cb._should_attempt_reset() is True

    def test_get_status(self):
        """Test getting circuit breaker status"""
        cb = CircuitBreaker("TestBreaker")
        status = cb.get_status()

        assert status['name'] == 'TestBreaker'
        assert status['state'] == 'CLOSED'
        assert status['failure_count'] == 0
        assert status['last_failure'] is None


class TestRetryMechanism:
    """Test retry_with_backoff functionality"""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test retry with immediate success"""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_with_backoff(success_func, max_retries=3)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_eventual_success(self):
        """Test retry with eventual success"""
        call_count = 0

        async def eventual_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await retry_with_backoff(eventual_success, max_retries=5)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test retry exhaustion"""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            await retry_with_backoff(always_fail, max_retries=2)

        assert call_count == 3  # Initial + 2 retries


class TestGracefulDegradation:
    """Test graceful degradation decorator"""

    @pytest.mark.asyncio
    async def test_graceful_degradation_success(self):
        """Test graceful degradation with success"""
        @graceful_degradation
        async def success_func():
            return "success"

        result = await success_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_graceful_degradation_failure(self):
        """Test graceful degradation with failure"""
        @graceful_degradation
        async def fail_func():
            raise ConnectionError("Failed")

        result = await fail_func()
        assert result is None  # Should return None on failure


if __name__ == "__main__":
    pytest.main([__file__])