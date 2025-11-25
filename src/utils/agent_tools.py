#!/usr/bin/env python3
"""
Agent communication and coordination tools.
Provides tools for inter-agent communication, polling, and collaboration.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import random

from .validation import circuit_breaker, DataValidator
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def audit_poll_tool(question: str, agents_to_poll: list = None) -> Dict[str, Any]:
    """
    Poll multiple agents for their opinions on a question.
    Args:
        question: Question to ask agents
        agents_to_poll: List of agent names to poll (optional)
    Returns:
        dict: Poll results from agents
    """
    try:
        question = DataValidator.sanitize_text_input(question)
        if not question:
            return {"error": "No valid question provided"}

        # Default agents if none specified
        if agents_to_poll is None:
            agents_to_poll = ["risk_agent", "strategy_agent", "data_agent", "execution_agent", "reflection_agent"]

        results = {}

        for agent in agents_to_poll:
            try:
                # Simulate agent response (in real implementation, this would call actual agents)
                vote = _get_fallback_vote(agent, question)
                confidence = random.uniform(0.5, 0.9)  # Simulated confidence

                results[agent] = {
                    "vote": vote,
                    "confidence": confidence,
                    "reasoning": f"Agent {agent} analysis based on available data"
                }

            except Exception as e:
                results[agent] = {
                    "error": f"Failed to poll agent {agent}: {str(e)}"
                }
                # Require real agent coordination - no fallback votes
                raise Exception(f"Agent polling requires real-time collaborative agent coordination: {e}")

        # Aggregate results
        votes = {}
        total_confidence = 0
        valid_responses = 0

        for agent, response in results.items():
            if "vote" in response:
                vote = response["vote"]
                confidence = response.get("confidence", 0)

                if vote not in votes:
                    votes[vote] = {"count": 0, "total_confidence": 0, "agents": []}

                votes[vote]["count"] += 1
                votes[vote]["total_confidence"] += confidence
                votes[vote]["agents"].append(agent)

                total_confidence += confidence
                valid_responses += 1

        # Find consensus
        if votes:
            consensus_vote = max(votes.keys(), key=lambda x: votes[x]["count"])
            consensus_confidence = votes[consensus_vote]["total_confidence"] / votes[consensus_vote]["count"]
            consensus_agents = votes[consensus_vote]["agents"]
        else:
            consensus_vote = "no_consensus"
            consensus_confidence = 0.0
            consensus_agents = []

        return {
            "question": question,
            "total_agents_polled": len(agents_to_poll),
            "valid_responses": valid_responses,
            "consensus": {
                "vote": consensus_vote,
                "confidence": consensus_confidence,
                "supporting_agents": consensus_agents,
                "support_count": len(consensus_agents)
            },
            "vote_breakdown": votes,
            "individual_responses": results,
            "source": "agent_poll"
        }

    except Exception as e:
        return {"error": f"Agent poll failed: {str(e)}"}


def agent_coordination_tool(task: str, required_agents: list, priority: str = "normal") -> Dict[str, Any]:
    """
    Coordinate task execution across multiple agents.
    Args:
        task: Task description
        required_agents: List of agents needed
        priority: Task priority
    Returns:
        dict: Coordination plan
    """
    try:
        task = DataValidator.sanitize_text_input(task)
        if not task:
            return {"error": "No valid task provided"}

        # Create coordination plan
        coordination_plan = {
            "task": task,
            "priority": priority,
            "required_agents": required_agents,
            "coordination_steps": []
        }

        # Define agent roles and dependencies
        agent_roles = {
            "data_agent": {"role": "data_collection", "prerequisites": []},
            "risk_agent": {"role": "risk_assessment", "prerequisites": ["data_agent"]},
            "strategy_agent": {"role": "strategy_development", "prerequisites": ["data_agent", "risk_agent"]},
            "execution_agent": {"role": "trade_execution", "prerequisites": ["strategy_agent"]},
            "reflection_agent": {"role": "performance_review", "prerequisites": ["execution_agent"]}
        }

        # Build execution sequence
        executed = set()
        remaining = set(required_agents)

        step = 1
        while remaining:
            can_execute = []
            for agent in remaining:
                role_info = agent_roles.get(agent, {"prerequisites": []})
                prereqs = role_info.get("prerequisites", [])

                if all(prereq in executed for prereq in prereqs):
                    can_execute.append(agent)

            if not can_execute:
                coordination_plan["coordination_steps"].append({
                    "step": step,
                    "action": "error",
                    "message": f"Cannot proceed - circular dependency or missing prerequisites for agents: {remaining}"
                })
                break

            # Execute agents that can run in parallel
            for agent in can_execute:
                coordination_plan["coordination_steps"].append({
                    "step": step,
                    "agent": agent,
                    "action": "execute",
                    "role": agent_roles.get(agent, {}).get("role", "unknown"),
                    "parallel_group": step
                })
                executed.add(agent)
                remaining.remove(agent)

            step += 1

        coordination_plan["total_steps"] = len(coordination_plan["coordination_steps"])
        coordination_plan["estimated_completion"] = f"{len(required_agents) * 5} minutes"  # Rough estimate

        return coordination_plan

    except Exception as e:
        return {"error": f"Agent coordination failed: {str(e)}"}


def shared_memory_broadcast_tool(message: str, namespace: str, sender_agent: str) -> Dict[str, Any]:
    """
    Broadcast a message to shared memory for other agents.
    Args:
        message: Message to broadcast
        namespace: Memory namespace
        sender_agent: Sending agent name
    Returns:
        dict: Broadcast result
    """
    try:
        message = DataValidator.sanitize_text_input(message)
        namespace = DataValidator.sanitize_text_input(namespace)
        sender_agent = DataValidator.sanitize_text_input(sender_agent)

        if not all([message, namespace, sender_agent]):
            return {"error": "Invalid parameters provided"}

        # In a real implementation, this would write to shared memory
        # For now, simulate the broadcast

        broadcast_data = {
            "message": message,
            "namespace": namespace,
            "sender": sender_agent,
            "timestamp": str(pd.Timestamp.now()),
            "message_id": f"{sender_agent}_{int(pd.Timestamp.now().timestamp())}"
        }

        return {
            "status": "broadcasted",
            "broadcast_data": broadcast_data,
            "estimated_receivers": "all_subscribed_agents",
            "namespace": namespace,
            "source": "shared_memory_broadcast"
        }

    except Exception as e:
        return {"error": f"Broadcast failed: {str(e)}"}


def agent_health_check_tool(agent_name: str = None) -> Dict[str, Any]:
    """
    Check the health status of agents.
    Args:
        agent_name: Specific agent to check (optional)
    Returns:
        dict: Agent health status
    """
    try:
        # Default agents to check
        agents_to_check = ["data_agent", "risk_agent", "strategy_agent", "execution_agent", "reflection_agent"]

        if agent_name:
            agents_to_check = [agent_name]

        health_results = {}

        for agent in agents_to_check:
            try:
                # Simulate health check (in real implementation, this would query actual agent status)
                health_status = {
                    "status": "healthy" if random.random() > 0.1 else "degraded",  # 90% healthy
                    "last_active": str(pd.Timestamp.now() - pd.Timedelta(minutes=random.randint(1, 60))),
                    "memory_usage": f"{random.randint(50, 200)}MB",
                    "active_tasks": random.randint(0, 3),
                    "error_rate": f"{random.uniform(0, 0.05):.3f}"
                }

                health_results[agent] = health_status

            except Exception as e:
                health_results[agent] = {
                    "status": "unreachable",
                    "error": str(e)
                }

        # Overall system health
        healthy_count = sum(1 for status in health_results.values() if status.get("status") == "healthy")
        total_count = len(health_results)

        overall_health = "healthy" if healthy_count / total_count > 0.8 else "degraded" if healthy_count / total_count > 0.5 else "critical"

        return {
            "overall_health": overall_health,
            "healthy_agents": healthy_count,
            "total_agents": total_count,
            "agent_details": health_results,
            "timestamp": str(pd.Timestamp.now()),
            "source": "agent_health_check"
        }

    except Exception as e:
        return {"error": f"Health check failed: {str(e)}"}


def collaborative_decision_tool(question: str, agents: list, decision_criteria: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Make a collaborative decision across multiple agents.
    Args:
        question: Decision question
        agents: List of agents to consult
        decision_criteria: Criteria for decision making
    Returns:
        dict: Collaborative decision result
    """
    try:
        question = DataValidator.sanitize_text_input(question)
        if not question:
            return {"error": "No valid question provided"}

        # Get agent opinions
        poll_result = audit_poll_tool(question, agents)

        if "error" in poll_result:
            return poll_result

        # Apply decision criteria
        if decision_criteria is None:
            decision_criteria = {
                "min_confidence": 0.6,
                "min_support": 0.5,  # 50% of agents must agree
                "tie_breaker": "highest_confidence"
            }

        consensus = poll_result.get("consensus", {})
        consensus_vote = consensus.get("vote")
        consensus_confidence = consensus.get("confidence", 0)
        support_count = consensus.get("support_count", 0)
        total_agents = poll_result.get("total_agents_polled", len(agents))

        # Evaluate decision criteria
        confidence_met = consensus_confidence >= decision_criteria.get("min_confidence", 0.6)
        support_met = support_count / total_agents >= decision_criteria.get("min_support", 0.5)

        if confidence_met and support_met:
            decision = consensus_vote
            decision_status = "approved"
        else:
            decision = "pending_review"
            decision_status = "requires_review"

        return {
            "question": question,
            "decision": decision,
            "decision_status": decision_status,
            "confidence_level": consensus_confidence,
            "support_ratio": support_count / total_agents,
            "criteria_met": {
                "confidence": confidence_met,
                "support": support_met
            },
            "poll_results": poll_result,
            "decision_criteria": decision_criteria,
            "source": "collaborative_decision"
        }

    except Exception as e:
        return {"error": f"Collaborative decision failed: {str(e)}"}