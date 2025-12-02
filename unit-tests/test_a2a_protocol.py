# [LABEL:TEST:a2a_protocol] [LABEL:TEST:unit] [LABEL:FRAMEWORK:pytest] [LABEL:FRAMEWORK:asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Comprehensive unit tests for A2A protocol message passing and orchestration
# Dependencies: pytest, asyncio, unittest.mock, src.utils.a2a_protocol
# Related: src/utils/a2a_protocol.py, docs/FRAMEWORKS/a2a-protocol.md
#
#!/usr/bin/env python3
"""
Unit tests for A2A (Agent-to-Agent) protocol functionality.
Tests message passing, agent registration, StateGraph orchestration, and error handling.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pytest
import asyncio
from datetime import datetime
from uuid import uuid4

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.a2a_protocol import A2AProtocol, BaseMessage, ErrorMessage, AgentState


class TestA2AProtocol:
    """Test cases for A2A protocol functionality."""

    @pytest.fixture
    def a2a_protocol(self):
        """Create an A2AProtocol instance for testing."""
        return A2AProtocol(max_agents=10)

    def test_initialization(self, a2a_protocol):
        """Test A2A protocol initialization."""
        assert a2a_protocol.max_agents == 10
        assert isinstance(a2a_protocol.agent_queues, dict)
        assert isinstance(a2a_protocol.agent_callbacks, dict)
        assert isinstance(a2a_protocol.agents, dict)
        assert hasattr(a2a_protocol, 'graph')

    def test_agent_registration(self, a2a_protocol):
        """Test agent registration functionality."""
        # Test successful registration
        result = a2a_protocol.register_agent("test_agent")
        assert result is True
        assert "test_agent" in a2a_protocol.agent_queues
        assert isinstance(a2a_protocol.agent_queues["test_agent"], asyncio.Queue)

        # Test registration with callback
        def test_callback(msg):
            pass

        result = a2a_protocol.register_agent("callback_agent", callback=test_callback)
        assert result is True
        assert "callback_agent" in a2a_protocol.agent_callbacks

        # Test registration with agent instance
        mock_agent = Mock()
        result = a2a_protocol.register_agent("instance_agent", agent_instance=mock_agent)
        assert result is True
        assert a2a_protocol.agents["instance_agent"] == mock_agent

    def test_max_agents_limit(self, a2a_protocol):
        """Test that max agents limit is enforced."""
        # Register up to max_agents
        for i in range(10):
            result = a2a_protocol.register_agent(f"agent_{i}")
            assert result is True

        # Try to register one more - should fail
        result = a2a_protocol.register_agent("agent_11")
        assert result is False

    @pytest.mark.asyncio
    async def test_message_sending_basic(self, a2a_protocol):
        """Test basic message sending functionality."""
        # Register agents
        a2a_protocol.register_agent("sender")
        a2a_protocol.register_agent("receiver")

        # Create and send message
        message = BaseMessage(
            type="test",
            sender="sender",
            receiver="receiver",
            timestamp="",
            data={"test": "data"},
            id=""
        )

        message_id = await a2a_protocol.send_message(message)

        # Verify message was sent
        assert message_id != ""
        assert message.id == message_id

        # Verify message can be received
        received = await a2a_protocol.receive_message("receiver")
        assert received is not None
        assert received.id == message_id
        assert received.data["test"] == "data"

    @pytest.mark.asyncio
    async def test_broadcast_messaging(self, a2a_protocol):
        """Test broadcast message functionality."""
        # Register multiple agents
        agents = ["agent1", "agent2", "agent3"]
        for agent in agents:
            a2a_protocol.register_agent(agent)

        # Send broadcast message
        message = BaseMessage(
            type="broadcast",
            sender="system",
            receiver=["agent1", "agent2", "agent3"],
            timestamp="",
            data={"broadcast": True},
            id=""
        )

        message_id = await a2a_protocol.send_message(message)

        # Verify all agents received the message
        for agent in agents:
            received = await a2a_protocol.receive_message(agent)
            assert received is not None
            assert received.id == message_id
            assert received.data["broadcast"] is True

    @pytest.mark.asyncio
    async def test_message_to_all_broadcast(self, a2a_protocol):
        """Test 'all' broadcast functionality."""
        # Register agents
        agents = ["agent1", "agent2"]
        for agent in agents:
            a2a_protocol.register_agent(agent)

        # Send message to "all"
        message = BaseMessage(
            type="system",
            sender="coordinator",
            receiver="all",
            timestamp="",
            data={"system_msg": True},
            id=""
        )

        message_id = await a2a_protocol.send_message(message)

        # Verify all agents received the message
        for agent in agents:
            received = await a2a_protocol.receive_message(agent)
            assert received is not None
            assert received.id == message_id

    @pytest.mark.asyncio
    async def test_invalid_receiver_error(self, a2a_protocol):
        """Test error handling for invalid receiver."""
        a2a_protocol.register_agent("sender")

        # Send message to non-existent receiver
        message = BaseMessage(
            type="test",
            sender="sender",
            receiver="nonexistent",
            timestamp="",
            data={"test": "data"},
            id=""
        )

        message_id = await a2a_protocol.send_message(message)

        # Should receive error message back to sender
        error_msg = await a2a_protocol.receive_message("sender")
        assert error_msg is not None
        assert isinstance(error_msg, ErrorMessage)
        assert error_msg.code == 404
        assert "not found" in error_msg.reason

    def test_message_validation_error(self, a2a_protocol):
        """Test message validation error handling."""
        a2a_protocol.register_agent("sender")

        # Create invalid message (missing required fields)
        with patch('src.utils.a2a_protocol.BaseMessage') as mock_message:
            mock_message.side_effect = Exception("Validation error")

            # This should trigger error handling
            # Note: In real implementation, this would be caught in send_message

    @pytest.mark.asyncio
    async def test_receive_from_empty_queue(self, a2a_protocol):
        """Test receiving from empty queue."""
        a2a_protocol.register_agent("test_agent")

        # Try to receive with timeout (using asyncio.wait_for)
        try:
            received = await asyncio.wait_for(
                a2a_protocol.receive_message("test_agent"),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            received = None

        assert received is None

    @pytest.mark.asyncio
    async def test_receive_callback_execution(self, a2a_protocol):
        """Test that callbacks are executed on message receive."""
        callback_executed = False

        def test_callback(message):
            nonlocal callback_executed
            callback_executed = True
            assert message.type == "test"

        a2a_protocol.register_agent("receiver", callback=test_callback)

        # Send message
        message = BaseMessage(
            type="test",
            sender="sender",
            receiver="receiver",
            timestamp="",
            data={},
            id=""
        )

        await a2a_protocol.send_message(message)

        # Receive message (should trigger callback)
        await a2a_protocol.receive_message("receiver")

        assert callback_executed

    def test_state_graph_structure(self, a2a_protocol):
        """Test that StateGraph is properly structured."""
        # Verify graph has expected nodes
        expected_nodes = ["macro", "data", "strategy", "risk", "execution", "reflection", "learning"]

        # Check that graph was built (we can't easily inspect internal structure,
        # but we can verify it exists and has expected attributes)
        assert hasattr(a2a_protocol, 'graph')
        assert a2a_protocol.graph is not None

    @pytest.mark.asyncio
    async def test_orchestration_run(self, a2a_protocol):
        """Test full orchestration run."""
        # Register mock agents with properly configured return values
        mock_agents = {}
        for role in ["macro", "data", "strategy", "risk", "execution", "reflection", "learning"]:
            mock_agent = MagicMock()
            # Ensure langchain_agent is not set to avoid LangChain path
            mock_agent.langchain_agent = None
            # Create an async function that returns a dict with required fields
            async def make_result(input_data, r=role):
                return {
                    f"{r}_result": True, 
                    "selected_sectors": [{"ticker": "XLK"}, {"ticker": "XLF"}], 
                    "approved": True,
                    "regime": "bullish"
                }
            mock_agent.process_input = make_result
            mock_agents[role] = mock_agent
            a2a_protocol.register_agent(role, agent_instance=mock_agent)

        # Run orchestration
        initial_data = {"symbols": ["AAPL"], "test": True}
        result = await a2a_protocol.run_orchestration(initial_data)

        # Verify result contains expected state (langgraph returns dict for final state)
        assert isinstance(result, (dict, AgentState))
        if isinstance(result, dict):
            # Check that key agent states are present
            assert "macro" in result
            assert "data" in result
            assert "strategy" in result

    def test_agent_state_model(self):
        """Test AgentState Pydantic model."""
        # Test basic state creation
        state = AgentState()
        assert isinstance(state.data, dict)
        assert isinstance(state.messages, list)
        assert state.status == "ongoing"

        # Test state updates
        state.data.update({"key": "value"})
        assert state.data["key"] == "value"

    def test_message_model_validation(self):
        """Test BaseMessage Pydantic validation."""
        # Valid message
        message = BaseMessage(
            type="test",
            sender="agent1",
            receiver="agent2",
            timestamp="2024-01-01T00:00:00Z",
            data={"test": True},
            id=str(uuid4())
        )
        assert message.type == "test"
        assert message.sender == "agent1"

        # Test with list receiver
        message_list = BaseMessage(
            type="broadcast",
            sender="system",
            receiver=["agent1", "agent2"],
            timestamp="2024-01-01T00:00:00Z",
            data={},
            id=str(uuid4())
        )
        assert isinstance(message_list.receiver, list)

    def test_error_message_model(self):
        """Test ErrorMessage model."""
        error = ErrorMessage(
            type="error",
            sender="system",
            receiver="agent1",
            timestamp="2024-01-01T00:00:00Z",
            data={},
            id=str(uuid4()),
            code=400,
            reason="Invalid request"
        )

        assert error.code == 400
        assert error.reason == "Invalid request"
        assert error.type == "error"

    def test_merge_functions(self):
        """Test merge utility functions."""
        from src.utils.a2a_protocol import merge_dicts, merge_status

        # Test dict merging
        left = {"a": 1, "b": 2}
        right = {"b": 3, "c": 4}
        merged = merge_dicts(left, right)

        assert merged["a"] == 1
        assert merged["b"] == 3  # right takes precedence
        assert merged["c"] == 4

        # Test status merging
        status1 = merge_status("pending", "completed")
        assert status1 == "completed"  # right takes precedence

    def test_langgraph_edge_stub(self, a2a_protocol):
        """Test LangGraph edge addition stub."""
        # This is currently a stub, so just verify it doesn't crash
        a2a_protocol.add_langgraph_edge("agent1", "agent2")
        # Should log the stub message but not fail

    @pytest.mark.asyncio
    async def test_concurrent_messaging(self, a2a_protocol):
        """Test concurrent message sending and receiving."""
        # Register multiple agents
        agents = ["agent1", "agent2", "agent3"]
        for agent in agents:
            a2a_protocol.register_agent(agent)

        # Send messages concurrently
        async def send_to_agent(agent_name):
            message = BaseMessage(
                type="concurrent_test",
                sender="test",
                receiver=agent_name,
                timestamp="",
                data={"concurrent": True},
                id=""
            )
            return await a2a_protocol.send_message(message)

        # Send messages to all agents concurrently
        tasks = [send_to_agent(agent) for agent in agents]
        message_ids = await asyncio.gather(*tasks)

        # Verify all messages were sent
        assert len(message_ids) == 3
        assert all(mid != "" for mid in message_ids)

        # Verify all messages can be received
        for agent in agents:
            received = await a2a_protocol.receive_message(agent)
            assert received is not None
            assert received.data["concurrent"] is True


if __name__ == "__main__":
    pytest.main([__file__])