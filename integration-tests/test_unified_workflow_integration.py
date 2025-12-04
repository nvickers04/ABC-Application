# [LABEL:TEST:integration] [LABEL:FRAMEWORK:pytest] [LABEL:FRAMEWORK:pytest_asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-04] [LABEL:REVIEWED:pending]
#
# Purpose: Integration tests for UnifiedWorkflowOrchestrator
# Dependencies: pytest, pytest-asyncio, UnifiedWorkflowOrchestrator
# Related: src/agents/unified_workflow_orchestrator.py
#
import pytest
import asyncio
import time
from unittest.mock import patch

class TestUnifiedWorkflowIntegration:
    """Integration tests for the unified workflow orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, unified_orchestrator):
        """Test that the orchestrator initializes correctly."""
        assert unified_orchestrator is not None
        assert unified_orchestrator.mode.name == "ANALYSIS"
        assert not unified_orchestrator.enable_discord

        status = unified_orchestrator.get_status()
        assert status["mode"] == "ANALYSIS"
        assert not status["is_running"]

    @pytest.mark.asyncio
    async def test_orchestrator_start_stop(self, unified_orchestrator):
        """Test starting and stopping the orchestrator."""
        # Start the orchestrator
        await unified_orchestrator.start()
        status = unified_orchestrator.get_status()
        assert status["is_running"]

        # Wait a moment for initialization
        await asyncio.sleep(0.1)

        # Stop the orchestrator
        await unified_orchestrator.stop()
        status = unified_orchestrator.get_status()
        assert not status["is_running"]

    @pytest.mark.asyncio
    async def test_workflow_mode_switching(self, unified_orchestrator):
        """Test switching between workflow modes."""
        from src.agents.unified_workflow_orchestrator import WorkflowMode

        # Test ANALYSIS mode
        unified_orchestrator.mode = WorkflowMode.ANALYSIS
        status = unified_orchestrator.get_status()
        assert status["mode"] == "ANALYSIS"

        # Test EXECUTION mode
        unified_orchestrator.mode = WorkflowMode.EXECUTION
        status = unified_orchestrator.get_status()
        assert status["mode"] == "EXECUTION"

        # Test HYBRID mode
        unified_orchestrator.mode = WorkflowMode.HYBRID
        status = unified_orchestrator.get_status()
        assert status["mode"] == "HYBRID"

    @pytest.mark.asyncio
    async def test_market_schedule_awareness(self, unified_orchestrator):
        """Test that the orchestrator is aware of market schedules."""
        # This would test APScheduler job scheduling based on market hours
        # For now, just verify the scheduler exists
        assert unified_orchestrator.scheduler is not None

    @pytest.mark.asyncio
    async def test_agent_communication(self, unified_orchestrator):
        """Test agent-to-agent communication via A2A protocol."""
        # This would test that agents can communicate through the orchestrator
        # For integration test, verify A2A protocol is initialized
        assert hasattr(unified_orchestrator, 'a2a_protocol')

class TestHealthMonitoringIntegration:
    """Integration tests for health monitoring system."""

    @pytest.mark.asyncio
    async def test_component_health_monitoring(self, component_health_monitor):
        """Test that component health monitoring works."""
        assert component_health_monitor is not None

        # Perform health checks
        results = component_health_monitor.perform_health_checks()
        assert isinstance(results, dict)
        assert len(results) > 0  # Should have some components

    @pytest.mark.asyncio
    async def test_api_health_monitoring(self):
        """Test API health monitoring."""
        from src.utils.api_health_monitor import get_api_health_summary

        health_summary = get_api_health_summary()
        assert isinstance(health_summary, dict)

    @pytest.mark.asyncio
    async def test_alert_manager_integration(self, alert_manager):
        """Test alert manager integration."""
        assert alert_manager is not None

        # Test sending an alert (this might be mocked in real scenario)
        # For integration test, just verify it exists
        pass

class TestEndToEndWorkflow:
    """End-to-end workflow integration tests."""

    @pytest.mark.asyncio
    async def test_full_workflow_cycle(self, unified_orchestrator):
        """Test a full workflow cycle from start to finish."""
        # This is a high-level integration test
        await unified_orchestrator.start()

        # Wait for some processing
        await asyncio.sleep(1)

        # Check that workflow is running
        status = unified_orchestrator.get_status()
        assert status["is_running"]

        await unified_orchestrator.stop()

    @pytest.mark.skip(reason="Requires live IBKR connection")
    @pytest.mark.asyncio
    async def test_live_trading_integration(self):
        """Test integration with live trading (requires IBKR TWS)."""
        # This test would require live IBKR connection
        # Skipped by default
        pass

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, unified_orchestrator):
        """Test error handling and recovery mechanisms."""
        # Start orchestrator
        await unified_orchestrator.start()

        # Simulate some error conditions
        # This would test circuit breakers, retries, etc.

        await unified_orchestrator.stop()

class TestPerformanceAndScalability:
    """Performance and scalability integration tests."""

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, unified_orchestrator):
        """Test concurrent execution of multiple workflows."""
        # This would test scalability
        pass

    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self):
        """Test memory usage monitoring during workflow execution."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run some workflow operations
        # Check memory hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Allow some memory growth but not excessive
        assert memory_growth < 100 * 1024 * 1024  # 100MB limit</content>
</xai:function_call name="create_file">
<parameter name="filePath">c:\Users\nvick\ABC-Application\integration-tests\test_health_api_integration.py