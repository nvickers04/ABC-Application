#!/usr/bin/env python3
"""
Comprehensive test suite for collaborative memory sessions functionality.
Tests cover unit tests, integration tests, edge cases, and concurrency.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import pytest

# Setup logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestCollaborativeSessions:
    """Comprehensive test suite for collaborative memory sessions."""

    def setup_method(self):
        """Setup before each test."""
        from src.utils.shared_memory import get_multi_agent_coordinator
        self.coordinator = get_multi_agent_coordinator()
        # Don't start the coordinator in tests to avoid background tasks

    def teardown_method(self):
        """Cleanup after each test."""
        # Clear any active sessions
        self.coordinator.active_sessions.clear()

    async def test_session_creation_and_basic_properties(self):
        """Test collaborative session creation and basic properties."""
        # Test session creation
        session_id = await self.coordinator.create_collaborative_session(
            creator_agent="strategy_agent",
            topic="Test Portfolio Strategy",
            max_participants=3,
            session_timeout=1800
        )

        assert session_id is not None
        assert "session_strategy_agent_" in session_id

        # Test session exists
        session = self.coordinator.active_sessions.get(session_id)
        assert session is not None
        assert session.creator_agent == "strategy_agent"
        assert session.topic == "Test Portfolio Strategy"
        assert session.max_participants == 3
        assert session.session_timeout == 1800
        assert session.status == "active"

    async def test_agent_join_leave_session(self):
        """Test agents joining and leaving collaborative sessions."""
        # Create session
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Test Session"
        )

        # Test joining
        success = await self.coordinator.join_collaborative_session(
            session_id, "data_agent", {"expertise": "market_data"}
        )
        assert success is True

        session = self.coordinator.active_sessions[session_id]
        assert "data_agent" in session.participants
        assert session.participants["data_agent"]["context"] == {"expertise": "market_data"}

        # Test joining already joined agent
        success = await self.coordinator.join_collaborative_session(
            session_id, "data_agent"
        )
        assert success is True  # Should not fail

        # Test leaving
        success = await self.coordinator.leave_collaborative_session(
            session_id, "data_agent"
        )
        assert success is True
        assert "data_agent" not in session.participants

        # Test leaving non-participant
        success = await self.coordinator.leave_collaborative_session(
            session_id, "nonexistent_agent"
        )
        assert success is False

    async def test_max_participants_limit(self):
        """Test that session respects max participants limit."""
        # Create session with max 2 participants
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Limited Session", max_participants=2
        )

        # Join first agent
        success1 = await self.coordinator.join_collaborative_session(
            session_id, "agent1"
        )
        assert success1 is True

        # Join second agent
        success2 = await self.coordinator.join_collaborative_session(
            session_id, "agent2"
        )
        assert success2 is True

        # Try to join third agent (should fail)
        success3 = await self.coordinator.join_collaborative_session(
            session_id, "agent3"
        )
        assert success3 is False

        session = self.coordinator.active_sessions[session_id]
        assert len(session.participants) == 2

    async def test_insight_contribution_and_validation(self):
        """Test contributing and validating insights in sessions."""
        # Create and join session
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Insight Test"
        )
        await self.coordinator.join_collaborative_session(session_id, "data_agent")
        await self.coordinator.join_collaborative_session(session_id, "risk_agent")

        # Contribute insight
        insight = {
            "type": "market_analysis",
            "content": "SPY showing bullish signals",
            "confidence": 0.85,
            "evidence": "Technical indicators",
            "market_data": {"rsi": 65, "macd": "bullish"}
        }

        success = await self.coordinator.contribute_to_session(
            session_id, "data_agent", insight
        )
        assert success is True

        session = self.coordinator.active_sessions[session_id]
        assert len(session.session_data["insights"]) == 1
        assert session.session_data["insights"][0]["agent"] == "data_agent"
        assert session.session_data["insights"][0]["insight"] == insight

        # Validate insight
        validation = {
            "agreement": True,
            "confidence_boost": 0.1,
            "reasoning": "Corroborates with volume analysis",
            "additional_evidence": "Volume confirms uptrend"
        }

        success = await self.coordinator.validate_session_insight(
            session_id, "risk_agent", 0, validation
        )
        assert success is True

        # Check validation was recorded
        assert len(session.session_data["insights"][0]["validated_by"]) == 1
        assert session.session_data["insights"][0]["validated_by"][0]["validator"] == "risk_agent"
        assert session.session_data["insights"][0]["validated_by"][0]["validation"] == validation

    @pytest.mark.asyncio
    async def test_shared_context_management(self):
        """Test updating and retrieving shared session context."""
        # Create and join session
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Context Test"
        )
        await self.coordinator.join_collaborative_session(session_id, "data_agent")

        # Update context
        context_updates = {
            "market_regime": "bullish_trend",
            "volatility": 0.25,
            "sector_focus": ["technology", "healthcare"],
            "risk_tolerance": 0.15
        }

        for key, value in context_updates.items():
            success = await self.coordinator.update_session_context(
                session_id, "strategy_agent", key, value
            )
            assert success is True

        # Retrieve context
        context = await self.coordinator.get_session_context(session_id)
        assert context["market_regime"]["data"] == "bullish_trend"
        assert context["volatility"]["data"] == 0.25
        assert context["sector_focus"]["data"] == ["technology", "healthcare"]
        assert context["risk_tolerance"]["data"] == 0.15

        # Check metadata
        assert context["market_regime"]["agent"] == "strategy_agent"
        assert "timestamp" in context["market_regime"]

    @pytest.mark.asyncio
    async def test_decision_recording(self):
        """Test recording collaborative decisions."""
        # Create and join session
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Decision Test"
        )
        await self.coordinator.join_collaborative_session(session_id, "data_agent")
        await self.coordinator.join_collaborative_session(session_id, "risk_agent")
        await self.coordinator.join_collaborative_session(session_id, "execution_agent")

        # Record decision
        decision = {
            "conclusion": "Execute portfolio rebalancing to 60/40 stock/bond",
            "rationale": "Collaborative analysis shows bullish market with controlled risk",
            "confidence": 0.88,
            "trade_details": {
                "SPY": {"action": "buy", "percentage": 0.4, "limit_price": 450.00},
                "TLT": {"action": "buy", "percentage": 0.3, "limit_price": 95.00},
                "existing_positions": {"action": "hold", "percentage": 0.3}
            },
            "execution_timeline": "Next trading day 9:30 AM ET",
            "risk_checks": ["VaR < 5%", "Diversification maintained", "Liquidity sufficient"]
        }

        success = await self.coordinator.record_session_decision(
            session_id, "strategy_agent", decision
        )
        assert success is True

        session = self.coordinator.active_sessions[session_id]
        assert len(session.session_data["decisions"]) == 1
        recorded_decision = session.session_data["decisions"][0]
        assert recorded_decision["agent"] == "strategy_agent"
        assert recorded_decision["decision"] == decision
        assert recorded_decision["participants"] == ["strategy_agent", "data_agent", "risk_agent", "execution_agent"]

    @pytest.mark.asyncio
    async def test_session_summary_and_insights(self):
        """Test retrieving session summaries and insights."""
        # Create session and add activity
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Summary Test", max_participants=4
        )

        agents = ["data_agent", "risk_agent", "execution_agent"]
        for agent in agents:
            await self.coordinator.join_collaborative_session(session_id, agent)

        # Add insights
        insights = [
            {"type": "market", "content": "Bullish SPY", "confidence": 0.8},
            {"type": "risk", "content": "VaR acceptable", "confidence": 0.9},
            {"type": "execution", "content": "Use limits", "confidence": 0.7}
        ]

        for i, insight in enumerate(insights):
            await self.coordinator.contribute_to_session(session_id, agents[i], insight)

        # Get summary
        summary = await self.coordinator.get_session_summary(session_id)
        assert summary["session_id"] == session_id
        assert summary["topic"] == "Summary Test"
        assert summary["creator"] == "strategy_agent"
        assert summary["participant_count"] == 4  # creator + 3 others
        assert summary["insights_count"] >= 3
        assert summary["decisions_count"] == 0
        assert summary["status"] == "active"

        # Get all insights
        all_insights = await self.coordinator.get_session_insights(session_id)
        assert len(all_insights) == 3

        # Get insights by agent
        data_insights = await self.coordinator.get_session_insights(session_id, "data_agent")
        assert len(data_insights) == 1
        assert data_insights[0]["agent"] == "data_agent"

    @pytest.mark.asyncio
    async def test_session_archiving(self):
        """Test archiving collaborative sessions."""
        # Create and populate session
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Archive Test"
        )
        await self.coordinator.join_collaborative_session(session_id, "data_agent")

        # Add some content
        await self.coordinator.contribute_to_session(
            session_id, "data_agent", {"type": "test", "content": "test insight"}
        )

        # Archive session
        success = await self.coordinator.archive_session(session_id)
        assert success is True

        session = self.coordinator.active_sessions.get(session_id)
        assert session.status == "archived"

        # Session should be removed from active sessions
        assert session_id not in self.coordinator.active_sessions

    @pytest.mark.asyncio
    async def test_session_timeout_handling(self):
        """Test session timeout and expiration handling."""
        # Create session with short timeout
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Timeout Test", session_timeout=1  # 1 second
        )

        session = self.coordinator.active_sessions[session_id]
        assert not session.is_expired()

        # Manually set last activity to past
        session.last_activity = (datetime.now() - timedelta(seconds=2)).isoformat()

        # Should be expired now
        assert session.is_expired()

    @pytest.mark.asyncio
    async def test_error_handling_invalid_sessions(self):
        """Test error handling for invalid session operations."""
        # Test operations on non-existent session
        success = await self.coordinator.join_collaborative_session(
            "nonexistent_session", "test_agent"
        )
        assert success is False

        success = await self.coordinator.contribute_to_session(
            "nonexistent_session", "test_agent", {"test": "insight"}
        )
        assert success is False

        summary = await self.coordinator.get_session_summary("nonexistent_session")
        assert summary is None

    @pytest.mark.asyncio
    async def test_error_handling_unauthorized_operations(self):
        """Test error handling for unauthorized session operations."""
        # Create session
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Auth Test"
        )

        # Try operations without being a participant
        success = await self.coordinator.contribute_to_session(
            session_id, "unauthorized_agent", {"test": "insight"}
        )
        assert success is False

        success = await self.coordinator.update_session_context(
            session_id, "unauthorized_agent", "test_key", "test_value"
        )
        assert success is False

        # Try to archive session as non-creator
        await self.coordinator.join_collaborative_session(session_id, "other_agent")
        success = await self.coordinator.archive_session(session_id)
        assert success is False  # Should fail as other_agent is not creator

    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self):
        """Test concurrent operations on the same session."""
        # Create session
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Concurrency Test"
        )

        # Simulate concurrent joins
        join_tasks = []
        for i in range(3):
            task = self.coordinator.join_collaborative_session(
                session_id, f"agent_{i}"
            )
            join_tasks.append(task)

        results = await asyncio.gather(*join_tasks)
        assert all(results)  # All joins should succeed

        session = self.coordinator.active_sessions[session_id]
        assert len(session.participants) == 4  # creator + 3 agents

    @pytest.mark.asyncio
    async def test_memory_persistence_integration(self):
        """Test that session data persists correctly."""
        # This test assumes memory persistence is working
        # Create session and add data
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Persistence Test"
        )

        await self.coordinator.join_collaborative_session(session_id, "data_agent")

        insight = {"type": "test", "content": "persistence test", "confidence": 0.9}
        await self.coordinator.contribute_to_session(session_id, "data_agent", insight)

        # Get session and check data is stored
        session = self.coordinator.active_sessions[session_id]
        assert len(session.session_data["insights"]) == 1

        # Note: Full persistence testing would require mocking the memory manager
        # and checking that data is actually written to storage

    def test_list_active_sessions(self):
        """Test listing active collaborative sessions."""
        # This is a synchronous test for the list functionality
        initial_count = len(self.coordinator.list_active_sessions())

        # Create a couple sessions
        async def create_sessions():
            session1 = await self.coordinator.create_collaborative_session(
                "strategy_agent", "List Test 1"
            )
            session2 = await self.coordinator.create_collaborative_session(
                "data_agent", "List Test 2"
            )
            return session1, session2

        session1, session2 = asyncio.run(create_sessions())

        sessions = self.coordinator.list_active_sessions()
        assert len(sessions) == initial_count + 2

        # Check session data
        session_ids = [s["session_id"] for s in sessions]
        assert session1 in session_ids
        assert session2 in session_ids

    @pytest.mark.asyncio
    async def test_session_data_integrity(self):
        """Test that session data maintains integrity across operations."""
        # Create session
        session_id = await self.coordinator.create_collaborative_session(
            "strategy_agent", "Integrity Test"
        )

        # Add multiple agents and insights
        agents = ["data_agent", "risk_agent", "execution_agent", "reflection_agent"]
        for agent in agents:
            await self.coordinator.join_collaborative_session(session_id, agent)

            insight = {
                "type": f"{agent}_analysis",
                "content": f"Analysis from {agent}",
                "confidence": 0.8 + len(agent) * 0.02,  # Varying confidence
                "agent": agent
            }
            await self.coordinator.contribute_to_session(session_id, agent, insight)

        # Validate all data
        session = self.coordinator.active_sessions[session_id]
        assert len(session.participants) == 5  # creator + 4 agents
        assert len(session.session_data["insights"]) == 4

        # Check each insight has correct agent
        for i, insight_data in enumerate(session.session_data["insights"]):
            assert insight_data["agent"] == agents[i]
            assert insight_data["insight"]["agent"] == agents[i]

        # Test summary accuracy
        summary = await self.coordinator.get_session_summary(session_id)
        assert summary["participant_count"] == 5
        assert summary["insights_count"] == 4
        assert summary["decisions_count"] == 0

class TestBaseAgentCollaborativeIntegration:
    """Test integration between BaseAgent and collaborative sessions."""

    def setup_method(self):
        """Setup test environment."""
        from src.agents.base import BaseAgent
        from src.utils.config import load_yaml
        import tempfile
        import os

        # Create temporary config files
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

        with open(self.config_path, 'w') as f:
            f.write("""
risk:
  constraints:
    variance_sd_threshold: 1.0
    max_drawdown: 0.05
""")

        # Create agent
        self.agent = BaseAgent(
            role="test_agent",
            config_paths={"risk": self.config_path},
            prompt_paths={"base": "base_prompt.txt", "role": "agents/test_agent_prompt.md"}
        )

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_agent_session_creation(self):
        """Test BaseAgent can create collaborative sessions."""
        session_id = await self.agent.create_collaborative_session(
            topic="Agent Integration Test",
            max_participants=3
        )

        assert session_id is not None
        assert "session_test_agent_" in session_id

    @pytest.mark.asyncio
    async def test_agent_session_participation(self):
        """Test BaseAgent can participate in sessions."""
        # Create session
        session_id = await self.agent.create_collaborative_session("Participation Test")

        # Join session
        success = await self.agent.join_collaborative_session(session_id)
        assert success is True

        # Contribute insight
        insight = {
            "type": "test_analysis",
            "content": "Test insight from agent",
            "confidence": 0.85
        }
        success = await self.agent.contribute_session_insight(session_id, insight)
        assert success is True

        # Get insights
        insights = await self.agent.get_session_insights(session_id)
        assert len(insights) == 1
        assert insights[0]["agent"] == "test_agent"

    @pytest.mark.asyncio
    async def test_agent_session_management(self):
        """Test BaseAgent session management methods."""
        # Create session
        session_id = await self.agent.create_collaborative_session("Management Test")

        # Get summary
        summary = await self.agent.get_session_summary(session_id)
        assert summary is not None
        assert summary["creator"] == "test_agent"

        # List sessions
        sessions = self.agent.list_my_sessions()
        assert len(sessions) >= 1

        # Archive session
        success = await self.agent.archive_session(session_id)
        assert success is True

if __name__ == "__main__":
    # Run basic tests manually
    import asyncio

    async def run_manual_tests():
        test_instance = TestCollaborativeSessions()

        print("Running manual collaborative sessions tests...")

        # Run a few key tests
        tests = [
            "test_session_creation_and_basic_properties",
            "test_agent_join_leave_session",
            "test_max_participants_limit",
            "test_insight_contribution_and_validation"
        ]

        for test_name in tests:
            try:
                test_instance.setup_method()
                method = getattr(test_instance, test_name)
                await method()
                print(f"✅ {test_name} passed")
            except Exception as e:
                print(f"❌ {test_name} failed: {e}")
            finally:
                test_instance.teardown_method()

        print("Manual test run complete.")

    asyncio.run(run_manual_tests())