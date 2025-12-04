# [LABEL:COMPONENT:consensus_poller] [LABEL:FRAMEWORK:asyncio] [LABEL:FRAMEWORK:pydantic]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-03] [LABEL:REVIEWED:pending]
#
# Purpose: Consensus polling system for agent-to-agent decision making with Discord visibility
# Dependencies: asyncio, pydantic, typing, datetime, json, redis (optional)
# Related: src/utils/a2a_protocol.py, src/integrations/discord/, docs/workflows.md
#
# Purpose: Implements a polling-based consensus mechanism for agent collaboration with configurable
# timeouts, state tracking, and Discord integration for visibility and interaction.
#
# States: pending, voting, consensus_reached, timeout, failed

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from src.utils.logging_config import log_error_with_context
# Simplified without pydantic for compatibility

logger = logging.getLogger(__name__)

class ConsensusState(Enum):
    """States for consensus polling process"""
    PENDING = "pending"           # Poll created but not started
    VOTING = "voting"            # Actively polling agents
    CONSENSUS_REACHED = "consensus_reached"  # Consensus achieved
    TIMEOUT = "timeout"           # Poll timed out without consensus
    FAILED = "failed"            # Poll failed due to errors

class ConsensusResult:
    """Result of a consensus poll"""
    def __init__(
        self,
        poll_id: str,
        question: str,
        state: ConsensusState,
        consensus_vote: Optional[str] = None,
        consensus_confidence: float = 0.0,
        supporting_agents: Optional[List[str]] = None,
        total_votes: int = 0,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        timeout_at: Optional[datetime] = None,
        votes: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.poll_id = poll_id
        self.question = question
        self.state = state
        self.consensus_vote = consensus_vote
        self.consensus_confidence = consensus_confidence
        self.supporting_agents = supporting_agents or []
        self.total_votes = total_votes
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.timeout_at = timeout_at
        self.votes = votes or {}
        self.metadata = metadata or {}

    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "poll_id": self.poll_id,
            "question": self.question,
            "state": self.state.value if isinstance(self.state, ConsensusState) else self.state,
            "consensus_vote": self.consensus_vote,
            "consensus_confidence": self.consensus_confidence,
            "supporting_agents": self.supporting_agents,
            "total_votes": self.total_votes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "timeout_at": self.timeout_at.isoformat() if self.timeout_at else None,
            "votes": self.votes,
            "metadata": self.metadata
        }

class ConsensusPoller:
    """Polls agents for consensus with configurable timeout and Discord visibility"""

    def __init__(
        self,
        poll_interval: int = 30,  # seconds
        default_timeout: int = 300,  # 5 minutes
        min_confidence: float = 0.6,
        persistence_file: str = "data/consensus_polls.json",
        redis_client = None,
        alert_manager = None
    ):
        self.poll_interval = poll_interval
        self.default_timeout = default_timeout
        self.min_confidence = min_confidence
        self.persistence_file = persistence_file
        self.redis_client = redis_client
        self.alert_manager = alert_manager

        # Active polls: poll_id -> ConsensusResult
        self.active_polls: Dict[str, ConsensusResult] = {}

        # Callbacks for state changes
        self.state_change_callbacks: List[Callable] = []

        # Metrics tracking
        self.metrics = {
            "total_polls_created": 0,
            "total_polls_completed": 0,
            "total_consensus_reached": 0,
            "total_timeouts": 0,
            "total_failures": 0,
            "avg_response_time": 0.0,
            "avg_confidence": 0.0,
            "poll_durations": [],  # Keep last 100 for rolling average
            "response_times": []   # Keep last 100 for rolling average
        }

        # Load persisted polls on startup
        self._load_persisted_polls()

    def add_state_change_callback(self, callback: Callable):
        """Add callback for state changes"""
        self.state_change_callbacks.append(callback)

    def _notify_state_change(self, poll: ConsensusResult):
        """Notify all callbacks of state change"""
        for callback in self.state_change_callbacks:
            try:
                asyncio.create_task(callback(poll))
            except Exception as e:
                log_error_with_context(
                    logger, e, "state_change_callback",
                    component="consensus_poller",
                    extra_context={"poll_id": poll.poll_id}
                )

        # Update metrics and send alerts based on state change
        self._update_metrics(poll)
        self._send_state_alert(poll)

    def _update_metrics(self, poll: ConsensusResult):
        """Update metrics based on poll state changes"""
        try:
            if poll.state in [ConsensusState.CONSENSUS_REACHED, ConsensusState.TIMEOUT, ConsensusState.FAILED]:
                self.metrics["total_polls_completed"] += 1

                if poll.state == ConsensusState.CONSENSUS_REACHED:
                    self.metrics["total_consensus_reached"] += 1
                    self.metrics["avg_confidence"] = (
                        (self.metrics["avg_confidence"] * (self.metrics["total_consensus_reached"] - 1) + poll.consensus_confidence)
                        / self.metrics["total_consensus_reached"]
                    )
                elif poll.state == ConsensusState.TIMEOUT:
                    self.metrics["total_timeouts"] += 1
                elif poll.state == ConsensusState.FAILED:
                    self.metrics["total_failures"] += 1

                # Track poll duration
                if poll.created_at and poll.updated_at:
                    duration = (poll.updated_at - poll.created_at).total_seconds()
                    self.metrics["poll_durations"].append(duration)
                    if len(self.metrics["poll_durations"]) > 100:
                        self.metrics["poll_durations"].pop(0)

                    # Update average duration
                    self.metrics["avg_response_time"] = sum(self.metrics["poll_durations"]) / len(self.metrics["poll_durations"])

        except Exception as e:
            log_error_with_context(
                logger, e, "update_metrics",
                component="consensus_poller",
                extra_context={"poll_id": poll.poll_id}
            )

    def _send_state_alert(self, poll: ConsensusResult):
        """Send alerts for important poll state changes"""
        if not self.alert_manager:
            return

        try:
            if poll.state == ConsensusState.CONSENSUS_REACHED:
                self.alert_manager.send_alert(
                    level="INFO",
                    component="ConsensusPoller",
                    message=f"Consensus reached: {poll.question[:50]}... ({poll.consensus_confidence:.1%} confidence)",
                    metadata={"poll_id": poll.poll_id, "consensus_vote": poll.consensus_vote}
                )
            elif poll.state == ConsensusState.TIMEOUT:
                self.alert_manager.send_alert(
                    level="WARNING",
                    component="ConsensusPoller",
                    message=f"Consensus poll timed out: {poll.question[:50]}...",
                    metadata={"poll_id": poll.poll_id}
                )
            elif poll.state == ConsensusState.FAILED:
                self.alert_manager.send_alert(
                    level="ERROR",
                    component="ConsensusPoller",
                    message=f"Consensus poll failed: {poll.question[:50]}...",
                    metadata={"poll_id": poll.poll_id, "error": poll.metadata.get("error", "Unknown")}
                )
        except Exception as e:
            logger.error(f"Error sending alert for poll {poll.poll_id}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()

    def _generate_poll_id(self) -> str:
        """Generate unique poll ID"""
        return f"consensus_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(datetime.now()) % 1000}"

    async def create_poll(
        self,
        question: str,
        agents_to_poll: List[str],
        timeout_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new consensus poll"""
        poll_id = self._generate_poll_id()
        timeout_at = datetime.now() + timedelta(seconds=timeout_seconds or self.default_timeout)

        poll = ConsensusResult(
            poll_id=poll_id,
            question=question,
            state=ConsensusState.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            timeout_at=timeout_at,
            metadata=metadata or {},
            votes={agent: {"status": "pending"} for agent in agents_to_poll}
        )

        self.active_polls[poll_id] = poll
        self._persist_poll(poll)

        logger.info(f"Created consensus poll {poll_id} for question: {question}")
        self._notify_state_change(poll)

        return poll_id

    async def start_poll(self, poll_id: str) -> bool:
        """Start polling for the given poll ID"""
        if poll_id not in self.active_polls:
            logger.error(f"Poll {poll_id} not found")
            return False

        poll = self.active_polls[poll_id]
        if poll.state != ConsensusState.PENDING:
            logger.warning(f"Poll {poll_id} is not in pending state")
            return False

        poll.state = ConsensusState.VOTING
        poll.updated_at = datetime.now()
        self._persist_poll(poll)
        self._notify_state_change(poll)

        # Start the polling task
        asyncio.create_task(self._poll_agents(poll_id))

        logger.info(f"Started polling for consensus poll {poll_id}")
        return True

    async def _poll_agents(self, poll_id: str):
        """Poll agents until consensus or timeout"""
        poll = self.active_polls.get(poll_id)
        if not poll:
            return

        last_status_update = datetime.now()
        status_update_interval = 60  # Send status updates every minute

        try:
            while poll.state == ConsensusState.VOTING:
                current_time = datetime.now()

                # Check for timeout
                if poll.timeout_at and current_time >= poll.timeout_at:
                    await self._handle_timeout(poll_id)
                    break

                # Poll agents (this would integrate with A2A protocol)
                await self._collect_votes(poll_id)

                # Check for consensus
                if await self._check_consensus(poll_id):
                    break

                # Send periodic status update
                if (current_time - last_status_update).total_seconds() >= status_update_interval:
                    await self._send_status_update(poll_id)
                    last_status_update = current_time

                # Wait before next poll
                await asyncio.sleep(self.poll_interval)

        except Exception as e:
            log_error_with_context(
                logger, e, "poll_agents",
                component="consensus_poller",
                extra_context={"poll_id": poll_id}
            )
            await self._handle_failure(poll_id, str(e))

    async def _send_status_update(self, poll_id: str):
        """Send periodic status update for ongoing poll"""
        poll = self.active_polls.get(poll_id)
        if not poll:
            return

        # Create a status update event (similar to state change but for progress)
        status_update = ConsensusResult(
            poll_id=poll.poll_id,
            question=poll.question,
            state=poll.state,
            consensus_vote=poll.consensus_vote,
            consensus_confidence=poll.consensus_confidence,
            supporting_agents=poll.supporting_agents.copy(),
            total_votes=poll.total_votes,
            created_at=poll.created_at,
            updated_at=poll.updated_at,
            timeout_at=poll.timeout_at,
            votes=poll.votes.copy(),
            metadata={**poll.metadata, "status_update": True}
        )

        # Notify callbacks with status update
        for callback in self.state_change_callbacks:
            try:
                await callback(status_update)
            except Exception as e:
                logger.error(f"Error in status update callback: {e}")

    async def _collect_votes(self, poll_id: str):
        """Collect votes from agents (placeholder for A2A integration)"""
        poll = self.active_polls.get(poll_id)
        if not poll:
            return

        # TODO: Integrate with A2A protocol to actually poll agents
        # For now, simulate some votes for testing
        for agent in poll.votes.keys():
            if poll.votes[agent]["status"] == "pending":
                # Simulate agent response
                import random
                if random.random() > 0.3:  # 70% chance of response
                    poll.votes[agent] = {
                        "status": "responded",
                        "vote": random.choice(["yes", "no", "abstain"]),
                        "confidence": random.uniform(0.5, 0.9),
                        "timestamp": datetime.now().isoformat()
                    }

        poll.updated_at = datetime.now()
        self._persist_poll(poll)

    async def _check_consensus(self, poll_id: str) -> bool:
        """Check if consensus has been reached"""
        poll = self.active_polls.get(poll_id)
        if not poll:
            return False

        # Count votes
        vote_counts = {}
        total_confidence = 0
        total_votes = 0

        for agent, vote_data in poll.votes.items():
            if vote_data.get("status") == "responded":
                vote = vote_data.get("vote")
                confidence = vote_data.get("confidence", 0)

                if vote not in vote_counts:
                    vote_counts[vote] = {"count": 0, "confidence": 0, "agents": []}

                vote_counts[vote]["count"] += 1
                vote_counts[vote]["confidence"] += confidence
                vote_counts[vote]["agents"].append(agent)

                total_confidence += confidence
                total_votes += 1

        poll.total_votes = total_votes

        # Check for consensus (majority with sufficient confidence)
        if vote_counts:
            # Find the majority vote
            majority_vote = max(vote_counts.keys(), key=lambda x: vote_counts[x]["count"])
            majority_count = vote_counts[majority_vote]["count"]
            majority_confidence = vote_counts[majority_vote]["confidence"] / majority_count

            # Consensus requires: majority > 50% and confidence >= min_confidence
            majority_ratio = majority_count / total_votes
            if majority_ratio > 0.5 and majority_confidence >= self.min_confidence:
                poll.state = ConsensusState.CONSENSUS_REACHED
                poll.consensus_vote = majority_vote
                poll.consensus_confidence = majority_confidence
                poll.supporting_agents = vote_counts[majority_vote]["agents"]
                poll.updated_at = datetime.now()

                self._persist_poll(poll)
                self._notify_state_change(poll)

                logger.info(f"Consensus reached for poll {poll_id}: {majority_vote} ({majority_confidence:.2f} confidence)")
                return True

        return False

    async def _handle_timeout(self, poll_id: str):
        """Handle poll timeout"""
        poll = self.active_polls.get(poll_id)
        if not poll:
            return

        poll.state = ConsensusState.TIMEOUT
        poll.updated_at = datetime.now()
        self._persist_poll(poll)
        self._notify_state_change(poll)

        logger.info(f"Poll {poll_id} timed out without consensus")

    async def _handle_failure(self, poll_id: str, error: str):
        """Handle poll failure"""
        poll = self.active_polls.get(poll_id)
        if not poll:
            return

        poll.state = ConsensusState.FAILED
        poll.metadata["error"] = error
        poll.updated_at = datetime.now()
        self._persist_poll(poll)
        self._notify_state_change(poll)

        logger.error(f"Poll {poll_id} failed: {error}")

    def get_poll_status(self, poll_id: str) -> Optional[ConsensusResult]:
        """Get current status of a poll"""
        return self.active_polls.get(poll_id)

    def get_active_polls(self) -> List[ConsensusResult]:
        """Get all active polls"""
        return [poll for poll in self.active_polls.values()
                if poll.state in [ConsensusState.PENDING, ConsensusState.VOTING]]

    def get_completed_polls(self, limit: int = 10) -> List[ConsensusResult]:
        """Get recently completed polls"""
        completed = [poll for poll in self.active_polls.values()
                    if poll.state in [ConsensusState.CONSENSUS_REACHED, ConsensusState.TIMEOUT, ConsensusState.FAILED]]

        # Sort by updated_at descending
        completed.sort(key=lambda x: x.updated_at, reverse=True)
        return completed[:limit]

    def _persist_poll(self, poll: ConsensusResult):
        """Persist poll to storage"""
        try:
            # Convert to dict for JSON serialization
            poll_dict = poll.model_dump()

            # Convert enums and datetimes
            poll_dict["state"] = poll.state.value
            poll_dict["created_at"] = poll.created_at.isoformat()
            poll_dict["updated_at"] = poll.updated_at.isoformat()
            if poll.timeout_at:
                poll_dict["timeout_at"] = poll.timeout_at.isoformat()

            # Save to file (in production, use Redis)
            try:
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {}

            data[poll.poll_id] = poll_dict

            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error persisting poll {poll.poll_id}: {e}")

    def _load_persisted_polls(self):
        """Load persisted polls on startup"""
        try:
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)

            for poll_id, poll_dict in data.items():
                # Convert back to ConsensusResult
                poll_dict["state"] = ConsensusState(poll_dict["state"])
                poll_dict["created_at"] = datetime.fromisoformat(poll_dict["created_at"])
                poll_dict["updated_at"] = datetime.fromisoformat(poll_dict["updated_at"])
                if poll_dict.get("timeout_at"):
                    poll_dict["timeout_at"] = datetime.fromisoformat(poll_dict["timeout_at"])

                poll = ConsensusResult(**poll_dict)
                self.active_polls[poll_id] = poll

            logger.info(f"Loaded {len(self.active_polls)} persisted polls")

        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No persisted polls found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading persisted polls: {e}")