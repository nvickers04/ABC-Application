#!/usr/bin/env python3
"""
Integration tests for AlertManager workflows and Discord integration.
Tests end-to-end alert processing, Discord notifications, and command handling.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

# Add src to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.alert_manager import AlertManager, Alert, AlertLevel, get_alert_manager
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator


class TestAlertDiscordIntegration:
    """Test AlertManager Discord integration"""

    def setup_method(self):
        """Reset alert manager before each test"""
        global alert_manager
        alert_manager = AlertManager()

    @pytest.mark.asyncio
    async def test_discord_alert_sending(self):
        """Test sending alerts to Discord"""
        manager = get_alert_manager()

        # Mock Discord components
        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock()

        # Set up orchestrator mock
        mock_orchestrator = Mock()
        mock_orchestrator.alerts_channel = mock_channel
        manager.orchestrator = mock_orchestrator

        # Send alert
        manager.send_alert(
            level=AlertLevel.ERROR,
            component="TestComponent",
            message="Test Discord alert",
            context={"test": True}
        )

        # Give async operations time to complete
        await asyncio.sleep(0.1)

        # Verify Discord message was sent
        mock_channel.send.assert_called_once()
        call_args = mock_channel.send.call_args
        embed = call_args[1]['embed']

        assert "Test Discord alert" in embed.description
        assert embed.color == 0xFF0000  # Red for ERROR
        assert len(embed.fields) >= 2  # Component and Time fields

    @pytest.mark.asyncio
    async def test_discord_alert_with_context(self):
        """Test sending alerts with context to Discord"""
        manager = get_alert_manager()

        # Mock Discord components
        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock()

        # Set up orchestrator mock
        mock_orchestrator = Mock()
        mock_orchestrator.alerts_channel = mock_channel
        manager.orchestrator = mock_orchestrator

        # Send alert with context
        manager.send_alert(
            level=AlertLevel.WARNING,
            component="TestComponent",
            message="Alert with context",
            context={"key1": "value1", "key2": "value2"}
        )

        await asyncio.sleep(0.1)

        # Verify context was included
        call_args = mock_channel.send.call_args
        embed = call_args[1]['embed']

        # Should have context field
        context_field = None
        for field in embed.fields:
            if field.name == 'Context':
                context_field = field
                break

        assert context_field is not None
        assert 'key1: value1' in context_field.value
        assert 'key2: value2' in context_field.value


class TestAlertCommandIntegration:
    """Test Discord command integration for alerts"""

    @pytest.mark.asyncio
    async def test_alert_test_command(self):
        """Test !alert_test command"""
        # Mock orchestrator and message
        orchestrator = Mock()
        message = Mock()
        message.channel = AsyncMock()
        message.author = Mock()
        message.author.display_name = "TestUser"

        # Import and call the handler
        from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator

        # Create instance and mock alert manager
        with patch('src.agents.live_workflow_orchestrator.get_alert_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            # Create orchestrator instance (minimal)
            orch = LiveWorkflowOrchestrator.__new__(LiveWorkflowOrchestrator)

            # Call command handler
            await orch.handle_alert_test_command(message)

            # Verify alert was sent
            mock_manager.send_alert.assert_called_once_with(
                level="INFO",
                component="DiscordCommand",
                message="Test alert triggered via !alert_test command",
                context={"user": str(message.author), "channel": str(message.channel)}
            )

            # Verify response sent
            message.channel.send.assert_called_once_with("✅ **Alert test sent!** Check for notification in alerts channel.")

    @pytest.mark.asyncio
    async def test_check_health_now_command(self):
        """Test !check_health_now command"""
        # Mock components
        orchestrator = Mock()
        message = Mock()
        message.channel = AsyncMock()

        with patch('src.agents.live_workflow_orchestrator.get_alert_manager') as mock_get_manager, \
             patch('src.agents.live_workflow_orchestrator.discord') as mock_discord:

            mock_manager = Mock()
            mock_manager.check_health.return_value = {
                'alert_queue_size': 5,
                'recent_critical_alerts': 1,
                'recent_error_alerts': 2,
                'orchestrator_connected': True,
                'discord_enabled': True
            }
            mock_get_manager.return_value = mock_manager

            mock_embed = Mock()
            mock_discord.Embed.return_value = mock_embed

            # Create orchestrator instance
            orch = LiveWorkflowOrchestrator.__new__(LiveWorkflowOrchestrator)

            # Call command handler
            await orch.handle_check_health_now_command(message)

            # Verify embed was created and sent
            mock_discord.Embed.assert_called_once_with(
                title="🔍 Health Check Results",
                color=0x00FF00,  # Green for connected
                timestamp=pytest.any()
            )

            message.channel.send.assert_called_once_with(embed=mock_embed)

    @pytest.mark.asyncio
    async def test_alert_history_command(self):
        """Test !alert_history command"""
        message = Mock()
        message.channel = AsyncMock()

        with patch('src.agents.live_workflow_orchestrator.get_alert_manager') as mock_get_manager, \
             patch('src.agents.live_workflow_orchestrator.discord') as mock_discord:

            mock_manager = Mock()
            # Create mock alerts
            mock_alert1 = Mock()
            mock_alert1.timestamp = datetime.now()
            mock_alert1.level.value = "ERROR"
            mock_alert1.component = "TestComponent"
            mock_alert1.message = "Test message"

            mock_manager.get_recent_alerts.return_value = [mock_alert1]
            mock_get_manager.return_value = mock_manager

            mock_embed = Mock()
            mock_discord.Embed.return_value = mock_embed

            # Create orchestrator instance
            orch = LiveWorkflowOrchestrator.__new__(LiveWorkflowOrchestrator)

            # Call command handler
            await orch.handle_alert_history_command(message)

            # Verify embed was created and sent
            mock_discord.Embed.assert_called_once_with(
                title="📋 Recent Alert History",
                color=0xFFA500,
                timestamp=pytest.any()
            )

            message.channel.send.assert_called_once_with(embed=mock_embed)

    @pytest.mark.asyncio
    async def test_alert_stats_command(self):
        """Test !alert_stats command"""
        message = Mock()
        message.channel = AsyncMock()

        with patch('src.agents.live_workflow_orchestrator.get_alert_manager') as mock_get_manager, \
             patch('src.agents.live_workflow_orchestrator.discord') as mock_discord:

            mock_manager = Mock()
            # Create mock alerts
            mock_alerts = []
            for i in range(3):
                mock_alert = Mock()
                mock_alert.level = f"LEVEL_{i}"
                mock_alert.component = f"Component{i}"
                mock_alerts.append(mock_alert)

            mock_manager.error_queue = mock_alerts
            mock_get_manager.return_value = mock_manager

            mock_embed = Mock()
            mock_discord.Embed.return_value = mock_embed

            # Create orchestrator instance
            orch = LiveWorkflowOrchestrator.__new__(LiveWorkflowOrchestrator)

            # Call command handler
            await orch.handle_alert_stats_command(message)

            # Verify embed was created and sent
            mock_discord.Embed.assert_called_once_with(
                title="📊 Alert Statistics",
                color=0x3498DB,
                timestamp=pytest.any()
            )

            message.channel.send.assert_called_once_with(embed=mock_embed)


class TestAlertWorkflowIntegration:
    """Test complete alert workflows"""

    @pytest.mark.asyncio
    async def test_error_alert_workflow(self):
        """Test complete error alert workflow"""
        manager = get_alert_manager()

        # Mock Discord
        mock_channel = AsyncMock()
        mock_orchestrator = Mock()
        mock_orchestrator.alerts_channel = mock_channel
        manager.orchestrator = mock_orchestrator

        # Trigger error alert
        manager.send_alert(
            level=AlertLevel.CRITICAL,
            component="IBKR",
            message="Connection failed",
            context={"endpoint": "api.ibkr.com"}
        )

        await asyncio.sleep(0.1)

        # Verify Discord notification sent
        assert mock_channel.send.called
        call_args = mock_channel.send.call_args
        embed = call_args[1]['embed']

        assert "Connection failed" in embed.description
        assert embed.color == 0xFF0000  # Red for CRITICAL

    @pytest.mark.asyncio
    async def test_validation_gate_with_alerts(self):
        """Test validation gate integration with alerts"""
        from src.utils.alert_manager import ValidationGate

        mock_manager = Mock()
        gate = ValidationGate("TestGate", alert_manager=mock_manager)

        def failing_check():
            return False

        # Should raise and send alert
        with pytest.raises(Exception):  # ValidationGateError
            gate.enforce({"check": failing_check})

        # Verify alert was sent
        mock_manager.send_alert.assert_called_once()
        call_args = mock_manager.send_alert.call_args
        assert "Validation gate 'TestGate' failed" in call_args[1]['message']

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_alerts(self):
        """Test circuit breaker integration with alerts"""
        from src.utils.alert_manager import CircuitBreaker

        mock_manager = Mock()
        cb = CircuitBreaker("TestBreaker", failure_threshold=1, alert_manager=mock_manager)

        async def failing_func():
            raise ConnectionError("Failed")

        # Trigger failure that opens circuit
        with pytest.raises(ConnectionError):
            await cb.call(failing_func)

        # Verify alert was sent
        mock_manager.send_alert.assert_called_once()
        call_args = mock_manager.send_alert.call_args
        assert "Circuit breaker opened" in call_args[1]['message']


if __name__ == "__main__":
    pytest.main([__file__])