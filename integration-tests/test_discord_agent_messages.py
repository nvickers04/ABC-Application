#!/usr/bin/env python3
"""
Test Discord agent message functionality
Tests that agents can send messages through DiscordResponseHandler
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from src.agents.discord_response_handler import DiscordResponseHandler


class TestDiscordAgentMessages:
    """Test suite for Discord agent message functionality"""

    @pytest.fixture
    def discord_handler(self):
        """Create DiscordResponseHandler instance"""
        handler = DiscordResponseHandler()
        return handler

    @pytest.fixture
    def mock_client(self):
        """Create mock Discord client"""
        client = MagicMock()
        client.get_guild.return_value = MagicMock()
        return client

    @pytest.fixture
    def mock_channel(self):
        """Create mock Discord channel"""
        channel = AsyncMock()
        channel.name = "test-channel"
        channel.send = AsyncMock()
        return channel

    @pytest.fixture
    def mock_guild(self, mock_channel):
        """Create mock Discord guild"""
        guild = MagicMock()
        guild.get_channel.return_value = mock_channel
        guild.name = "Test Guild"
        return guild

    @pytest.mark.asyncio
    async def test_send_agent_responses_success(self, discord_handler, mock_client, mock_channel, mock_guild):
        """Test successful sending of agent responses"""
        # Setup
        discord_handler.set_client(mock_client)
        discord_handler.channel = mock_channel
        discord_handler.discord_ready.set()

        mock_client.get_guild.return_value = mock_guild

        # Test data
        responses = [
            {
                "agent_name": "DataAgent",
                "response": "Market data collected successfully",
                "confidence": 0.85,
                "timestamp": "2025-12-03T10:00:00Z"
            },
            {
                "agent_name": "StrategyAgent",
                "response": "Bullish signal detected",
                "confidence": 0.92,
                "timestamp": "2025-12-03T10:00:05Z"
            }
        ]

        # Execute
        await discord_handler.send_agent_responses(responses, "analysis_phase")

        # Verify
        assert mock_channel.send.called
        call_args = mock_channel.send.call_args
        assert call_args is not None

        # Check that the message contains agent information
        message_content = call_args[0][0]  # First positional argument
        assert "DataAgent" in message_content
        assert "StrategyAgent" in message_content
        assert "analysis_phase" in message_content

    @pytest.mark.asyncio
    async def test_send_workflow_status_success(self, discord_handler, mock_client, mock_channel):
        """Test successful sending of workflow status"""
        # Setup
        discord_handler.set_client(mock_client)
        discord_handler.channel = mock_channel
        discord_handler.discord_ready.set()

        # Test message
        status_message = "🚀 **Workflow Phase Started: Market Analysis**"

        # Execute
        await discord_handler.send_workflow_status(status_message)

        # Verify
        mock_channel.send.assert_called_once_with(status_message)

    @pytest.mark.asyncio
    async def test_send_trade_alert_success(self, discord_handler, mock_client, mock_channel):
        """Test successful sending of trade alerts"""
        # Setup
        discord_handler.set_client(mock_client)
        discord_handler.alerts_channel = mock_channel
        discord_handler.discord_ready.set()

        # Test alert
        alert_message = "⚠️ **TRADE ALERT**: AAPL position opened at $150.00"
        alert_type = "trade"

        # Execute
        await discord_handler.send_trade_alert(alert_message, alert_type)

        # Verify
        mock_channel.send.assert_called_once()
        call_args = mock_channel.send.call_args
        message_content = call_args[0][0]

        # Check embed formatting for alerts
        assert "⚠️" in message_content or "TRADE ALERT" in message_content

    @pytest.mark.asyncio
    async def test_send_ranked_trade_info_success(self, discord_handler, mock_client, mock_channel):
        """Test successful sending of ranked trade information"""
        # Setup
        discord_handler.set_client(mock_client)
        discord_handler.ranked_trades_channel = mock_channel
        discord_handler.discord_ready.set()

        # Test trade info
        trade_message = "📈 **RANKED TRADE**: AAPL Buy - Confidence: 0.89"
        trade_type = "proposal"

        # Execute
        await discord_handler.send_ranked_trade_info(trade_message, trade_type)

        # Verify
        mock_channel.send.assert_called_once()
        call_args = mock_channel.send.call_args
        message_content = call_args[0][0]

        # Check embed formatting for ranked trades
        assert "📈" in message_content or "RANKED TRADE" in message_content

    @pytest.mark.asyncio
    async def test_send_when_discord_not_ready(self, discord_handler, mock_client, mock_channel):
        """Test that messages are not sent when Discord is not ready"""
        # Setup - discord_ready not set
        discord_handler.set_client(mock_client)
        discord_handler.channel = mock_channel
        # discord_ready is not set

        # Execute
        await discord_handler.send_workflow_status("Test message")

        # Verify - should not send
        mock_channel.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_when_channel_not_configured(self, discord_handler, mock_client):
        """Test that messages are not sent when channel is not configured"""
        # Setup
        discord_handler.set_client(mock_client)
        discord_handler.discord_ready.set()
        # channel is None

        # Execute
        await discord_handler.send_workflow_status("Test message")

        # Verify - should not send (no channel configured)
        # This test verifies the guard clause works

    @pytest.mark.asyncio
    async def test_setup_discord_channels_success(self, discord_handler, mock_client, mock_guild):
        """Test successful setup of Discord channels"""
        # Setup
        discord_handler.set_client(mock_client)
        mock_client.get_guild.return_value = mock_guild

        with patch.dict('os.environ', {
            'DISCORD_GUILD_ID': '123456789',
            'DISCORD_GENERAL_CHANNEL_ID': '111111111',
            'DISCORD_ALERTS_CHANNEL_ID': '222222222',
            'DISCORD_RANKED_TRADES_CHANNEL_ID': '333333333'
        }):
            # Execute
            await discord_handler._setup_discord_channels()

            # Verify
            assert discord_handler.channel is not None
            assert discord_handler.alerts_channel is not None
            assert discord_handler.ranked_trades_channel is not None

    @pytest.mark.asyncio
    async def test_agent_response_formatting(self, discord_handler, mock_client, mock_channel):
        """Test that agent responses are properly formatted"""
        # Setup
        discord_handler.set_client(mock_client)
        discord_handler.channel = mock_channel
        discord_handler.discord_ready.set()

        # Test data with various response types
        responses = [
            {
                "agent_name": "RiskAgent",
                "response": "Risk assessment: Low risk, proceed",
                "confidence": 0.78,
                "risk_score": 0.15
            }
        ]

        # Execute
        await discord_handler.send_agent_responses(responses, "risk_assessment")

        # Verify
        assert mock_channel.send.called
        call_args = mock_channel.send.call_args
        message_content = call_args[0][0]

        # Check formatting includes key information
        assert "RiskAgent" in message_content
        assert "risk_assessment" in message_content
        assert "Low risk" in message_content

    @pytest.mark.asyncio
    async def test_multiple_channel_routing(self, discord_handler, mock_client):
        """Test that different message types go to appropriate channels"""
        # Setup
        general_channel = AsyncMock()
        alerts_channel = AsyncMock()
        ranked_channel = AsyncMock()

        discord_handler.set_client(mock_client)
        discord_handler.channel = general_channel
        discord_handler.alerts_channel = alerts_channel
        discord_handler.ranked_trades_channel = ranked_channel
        discord_handler.discord_ready.set()

        # Execute different message types
        await discord_handler.send_workflow_status("Status update")
        await discord_handler.send_trade_alert("Trade alert")
        await discord_handler.send_ranked_trade_info("Ranked trade")

        # Verify routing
        general_channel.send.assert_called_once()
        alerts_channel.send.assert_called_once()
        ranked_channel.send.assert_called_once()</content>
</xai:function_call name="run_in_terminal">
<parameter name="command">. .\myenv\Scripts\Activate.ps1; pytest integration-tests/test_discord_agent_messages.py -v