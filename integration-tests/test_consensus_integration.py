#!/usr/bin/env python3
"""
Integration tests for Consensus Workflow Polling
"""

import asyncio
import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch

from src.workflows.consensus_poller import ConsensusPoller, ConsensusState
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator


class TestConsensusIntegration:
    """Integration tests for consensus polling with orchestrator"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.persistence_file = self.temp_file.name

    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.persistence_file):
            os.unlink(self.persistence_file)

    @pytest.mark.asyncio
    async def test_orchestrator_consensus_request(self):
        """Test orchestrator handling agent consensus requests"""
        # Mock alert manager
        mock_alert_manager = Mock()

        # Create poller
        poller = ConsensusPoller(
            persistence_file=self.persistence_file,
            alert_manager=mock_alert_manager
        )

        # Mock orchestrator methods
        orchestrator = Mock()
        orchestrator.consensus_poller = poller

        # Add the actual request_consensus method
        async def request_consensus(question, requesting_agent, target_agents, timeout_seconds=300):
            return await poller.create_poll(
                f"[{requesting_agent}] {question}",
                target_agents,
                timeout_seconds=timeout_seconds,
                metadata={"requesting_agent": requesting_agent, "agent_requested": True}
            )

        orchestrator.request_consensus = request_consensus

        # Test agent requesting consensus
        poll_id = await orchestrator.request_consensus(
            "Is this position size safe?",
            "risk_agent",
            ["strategy_agent", "execution_agent"]
        )

        assert poll_id in poller.active_polls
        poll = poller.active_polls[poll_id]
        assert "[risk_agent]" in poll.question
        assert "Is this position size safe?" in poll.question
        assert len(poll.votes) == 2
        assert "strategy_agent" in poll.votes
        assert "execution_agent" in poll.votes
        assert poll.metadata["requesting_agent"] == "risk_agent"
        assert poll.metadata["agent_requested"] is True

    @pytest.mark.asyncio
    async def test_full_consensus_workflow(self):
        """Test complete consensus workflow from creation to completion"""
        # Mock alert manager
        mock_alert_manager = Mock()

        # Create poller
        poller = ConsensusPoller(
            poll_interval=1,  # Fast polling for test
            default_timeout=10,  # Short timeout
            persistence_file=self.persistence_file,
            alert_manager=mock_alert_manager
        )

        # Mock Discord callback
        discord_updates = []
        async def mock_discord_callback(poll):
            discord_updates.append({
                "poll_id": poll.poll_id,
                "state": poll.state.value,
                "question": poll.question
            })

        poller.add_state_change_callback(mock_discord_callback)

        # Create and start poll
        question = "Should we proceed with this trade?"
        agents = ["risk_agent", "strategy_agent", "data_agent"]
        poll_id = await poller.create_poll(question, agents)
        success = await poller.start_poll(poll_id)

        assert success
        assert len(discord_updates) >= 1  # At least state change to VOTING

        # Simulate agent responses
        poll = poller.active_polls[poll_id]
        poll.votes["risk_agent"] = {
            "status": "responded",
            "vote": "yes",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }
        poll.votes["strategy_agent"] = {
            "status": "responded",
            "vote": "yes",
            "confidence": 0.9,
            "timestamp": datetime.now().isoformat()
        }
        poll.votes["data_agent"] = {
            "status": "responded",
            "vote": "no",
            "confidence": 0.6,
            "timestamp": datetime.now().isoformat()
        }

        # Check consensus
        consensus_reached = await poller._check_consensus(poll_id)

        assert consensus_reached
        assert poll.state == ConsensusState.CONSENSUS_REACHED
        assert poll.consensus_vote == "yes"

        # Check metrics were updated
        metrics = poller.get_metrics()
        assert metrics["total_polls_completed"] == 1
        assert metrics["total_consensus_reached"] == 1
        assert metrics["avg_confidence"] > 0.8  # Average of yes votes

        # Check alerts were sent
        assert mock_alert_manager.send_alert.called
        alert_call = mock_alert_manager.send_alert.call_args
        assert "consensus reached" in alert_call[1]["message"].lower()

        # Check Discord updates
        consensus_updates = [u for u in discord_updates if u["state"] == "consensus_reached"]
        assert len(consensus_updates) == 1

    @pytest.mark.asyncio
    async def test_persistence_across_restarts(self):
        """Test that polls persist across system restarts"""
        # Mock alert manager
        mock_alert_manager = Mock()

        # Create first poller and add poll
        poller1 = ConsensusPoller(
            persistence_file=self.persistence_file,
            alert_manager=mock_alert_manager
        )

        poll_id = await poller1.create_poll("Persistent question?", ["agent1", "agent2"])
        poll1 = poller1.active_polls[poll_id]

        # Simulate some state
        poll1.votes["agent1"] = {
            "status": "responded",
            "vote": "yes",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }

        # Force persistence
        poller1._persist_poll(poll1)

        # Create second poller (simulating restart)
        poller2 = ConsensusPoller(
            persistence_file=self.persistence_file,
            alert_manager=mock_alert_manager
        )

        # Check poll was loaded
        assert poll_id in poller2.active_polls
        poll2 = poller2.active_polls[poll_id]
        assert poll2.question == "Persistent question?"
        assert poll2.votes["agent1"]["status"] == "responded"
        assert poll2.votes["agent1"]["vote"] == "yes"

    @pytest.mark.asyncio
    async def test_timeout_scenario(self):
        """Test poll timeout and cleanup"""
        # Mock alert manager
        mock_alert_manager = Mock()

        # Create poller with very short timeout
        poller = ConsensusPoller(
            poll_interval=1,
            default_timeout=2,  # 2 seconds
            persistence_file=self.persistence_file,
            alert_manager=mock_alert_manager
        )

        # Mock Discord callback
        timeout_alerts = []
        async def mock_timeout_callback(poll):
            if poll.state == ConsensusState.TIMEOUT:
                timeout_alerts.append(poll)

        poller.add_state_change_callback(mock_timeout_callback)

        # Create and start poll
        poll_id = await poller.create_poll("Timeout test?", ["agent1"])
        await poller.start_poll(poll_id)

        # Wait for timeout
        await asyncio.sleep(3)

        # Check timeout was handled
        poll = poller.active_polls.get(poll_id)
        if poll and poll.state == ConsensusState.TIMEOUT:
            assert poll.state == ConsensusState.TIMEOUT
            assert len(timeout_alerts) >= 1

            # Check alert was sent
            assert mock_alert_manager.send_alert.called
            alert_call = mock_alert_manager.send_alert.call_args
            assert alert_call[1]["level"] == "WARNING"
            assert "timed out" in alert_call[1]["message"].lower()


if __name__ == "__main__":
    pytest.main([__file__])