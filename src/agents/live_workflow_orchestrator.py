#!/usr/bin/env python3
"""
Live Workflow Orchestrator - Real-time Interactive Iterative Reasoning
Watches Discord and orchestrates the collaborative reasoning workflow automatically,
while allowing human intervention and questions during the process.

Enhanced to support both Discord-based operation and direct agent method calls
for improved reliability, testing, and integration with BaseAgent architecture.

Human Input Features:
- Human messages/interventions are only processed at the beginning of each workflow iteration
- Mid-iteration inputs are queued for the next iteration with acknowledgment
- `!share_news` command allows sharing news links for Data Agent processing
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import discord
import os
import time
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, cast
from urllib.parse import urlparse
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz  # For timezone handling

# Suppress TensorFlow deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow_probability.*')

# Set up logging for human input features
logger = logging.getLogger(__name__)

# Import AlertManager for error handling
from src.utils.alert_manager import get_alert_manager

# Import BaseAgent and agent classes for direct integration
from src.agents.base import BaseAgent

# Import agent scope definitions for intelligent command routing
from src.agents.agent_scopes import get_agent_scope_definitions

# Try to import agents with alert handling
alert_manager = get_alert_manager()
try:
    from src.agents.macro import MacroAgent
    from src.agents.data import DataAgent
    from src.agents.strategy import StrategyAgent
    from src.agents.risk import RiskAgent
    from src.agents.reflection import ReflectionAgent
    from src.agents.execution import ExecutionAgent
    from src.agents.learning import LearningAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    alert_manager.critical(e, {"component": "agent_imports"}, "orchestrator")
    # This will raise HealthAlertError and abort startup
from src.utils.a2a_protocol import A2AProtocol
from src.utils.vault_client import get_vault_secret

load_dotenv()

class LiveWorkflowOrchestrator:
    """
    Orchestrates the iterative reasoning workflow in real-time on Discord,
    allowing human participation and intervention during the process.
    
    Enhanced with direct BaseAgent integration for improved reliability and testing.
    """

    def __init__(self):
        # Discord components
        self.client = None
        self.channel = None  # General channel for summaries
        self.alerts_channel = None  # Dedicated channel for trade alerts
        self.ranked_trades_channel = None  # Dedicated channel for ranked trade proposals
        self.commands_channel = None  # Dedicated channel for command documentation
        self.agent_channels = {}  # Map agent types to their channels
        
        # Discord readiness synchronization
        self.discord_ready = asyncio.Event()
        
        # Direct agent integration
        self.a2a_protocol = A2AProtocol(max_agents=50)
        self.agent_instances: Dict[str, BaseAgent] = {}
        self.collaborative_session_id: Optional[str] = None
        
        # Workflow state
        self.workflow_active = False
        self.current_phase = "waiting"
        self.phase_commands = {}
        self.responses_collected = []
        self.human_interventions = []
        self.workflow_log = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_log = []
        self.monitoring_responses = []
        self.scheduler = AsyncIOScheduler(timezone='America/New_York')  # Use ET timezone
        
        # Human input queue for iteration-start processing
        # Messages received mid-iteration are queued here and processed at the start of the next iteration
        self.human_input_queue: List[Dict[str, Any]] = []
        self.shared_news_queue: List[Dict[str, Any]] = []  # Queue for !share_news commands
        self.iteration_in_progress = False  # Flag to track if we're mid-iteration
        
        # Phase timing configuration - NO TIMEOUTS (VERY LONG WAIT TIMES)
        self.phase_delays = {
            'systematic_market_surveillance': 300,  # 5 minutes
            'multi_strategy_opportunity_synthesis': 300,
            'quantitative_opportunity_validation': 300,
            'investment_committee_review': 300,
            'portfolio_implementation_planning': 300,
            'performance_analytics_and_refinement': 300,
            'chief_investment_officer_oversight': 300,
            'pre_market_readiness_assessment': 300,
            'opening_bell_execution': 300,
            'position_surveillance_initialization': 300,
            'active_position_management': 300,
            'dynamic_portfolio_adjustment': 300,
            'execution_of_portfolio_changes': 300
        }

        self._initialize_workflow_commands()

        # Initialize AlertManager instance
        self.alert_manager = alert_manager

        # Set orchestrator reference in AlertManager for Discord notifications
        alert_manager.set_orchestrator(self)

        # Initialize ConsensusPoller for agent consensus workflows
        from src.workflows.consensus_poller import ConsensusPoller
        self.consensus_poller = ConsensusPoller(
            poll_interval=30,  # 30 second intervals
            default_timeout=300,  # 5 minute timeout
            min_confidence=0.6,  # 60% confidence threshold
            persistence_file="data/consensus_polls.json",
            alert_manager=alert_manager
        )

        # Add callback for consensus state changes to notify Discord
        self.consensus_poller.add_state_change_callback(self._handle_consensus_state_change)

    def _sanitize_user_input(self, input_text: str) -> str:
        """
        Sanitize user input to prevent injection attacks and ensure safe processing.
        Removes potentially harmful content while preserving legitimate analysis requests.
        """
        if not isinstance(input_text, str):
            return ""

        # Remove excessive whitespace and control characters
        import re
        sanitized = re.sub(r'\s+', ' ', input_text.strip())

        # Remove or escape potentially harmful patterns
        harmful_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'data:',  # Data URLs
            r'vbscript:',  # VBScript
            r'on\w+\s*=',  # Event handlers
            r'\\x[0-9a-fA-F]{2}',  # Hex escapes
            r'\\u[0-9a-fA-F]{4}',  # Unicode escapes
        ]

        for pattern in harmful_patterns:
            sanitized = re.sub(pattern, '[FILTERED]', sanitized, flags=re.IGNORECASE)

        # Limit length to prevent abuse
        if len(sanitized) > 2000:
            sanitized = sanitized[:2000] + "...[TRUNCATED]"

        # Remove attempts to override system behavior
        system_override_patterns = [
            r'SYSTEM:', r'ASSISTANT:', r'USER:',
            r'IGNORE.*INSTRUCTIONS', r'FORGET.*INSTRUCTIONS',
            r'NEW.*INSTRUCTIONS', r'SYSTEM.*PROMPT',
            r'You are now', r'FROM NOW ON', r'HENCEFORTH',
            r'STARTING NOW', r'BEGINNING NOW'
        ]

        for pattern in system_override_patterns:
            sanitized = re.sub(pattern, '[FILTERED]', sanitized, flags=re.IGNORECASE)

        return sanitized

    def _sanitize_agent_output(self, output_text: str) -> str:
        """
        Sanitize agent output before displaying to users.
        Ensures no sensitive information or harmful content is exposed.
        """
        if not isinstance(output_text, str):
            return str(output_text)

        # Remove potential sensitive patterns
        sensitive_patterns = [
            r'API_KEY[=:]\s*[\w-]+',  # API keys
            r'TOKEN[=:]\s*[\w-]+',  # Tokens
            r'PASSWORD[=:]\s*[\w-]+',  # Passwords
            r'SECRET[=:]\s*[\w-]+',  # Secrets
            r'PRIVATE_KEY[=:]\s*[\w-]+',  # Private keys
        ]

        sanitized = output_text
        for pattern in sensitive_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)

        # Limit output length for display
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000] + "...[TRUNCATED]"

        return sanitized

    def _initialize_agents(self):
        """Initialize direct agent instances for enhanced integration (sync placeholder)"""
        # Agent initialization will be done asynchronously in initialize_agents_async
        pass

    async def initialize_agents_async(self):
        """Asynchronously initialize direct agent instances for enhanced integration with retries"""
        if not AGENTS_AVAILABLE:
            print("⚠️ Agent classes not available - running in Discord-only mode")
            return
            
        print("🤖 Starting agent initialization (this may take 20-30 seconds)...")
        start_time = time.time()
            
        max_retries = 3
        retry_delay = 5  # seconds
            
        try:
            # Initialize each agent with A2A protocol
            agent_classes = {
                'macro': MacroAgent,
                'data': DataAgent,
                'strategy': StrategyAgent,
                'risk': RiskAgent,
                'reflection': ReflectionAgent,
                'execution': ExecutionAgent,
                'learning': LearningAgent
            }

            for agent_key, agent_class in agent_classes.items():
                if agent_class is None:
                    print(f"⚠️ Skipping {agent_key} agent - class not available")
                    continue
                    
                success = False
                for attempt in range(max_retries):
                    try:
                        print(f"🔧 Initializing {agent_key} agent (attempt {attempt + 1}/{max_retries})...")
                        agent_instance = agent_class(a2a_protocol=self.a2a_protocol)
                        
                        # Initialize LLM asynchronously
                        await agent_instance.async_initialize_llm()
                        
                        self.agent_instances[agent_key] = agent_instance
                        self.a2a_protocol.register_agent(agent_key, agent_instance)
                        print(f"✅ Initialized {agent_key} agent with A2A protocol")
                        success = True
                        break
                    except Exception as e:
                        print(f"⚠️ Failed to initialize {agent_key} agent (attempt {attempt + 1}): {e}")
                        if attempt < max_retries - 1:
                            print(f"Retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                        else:
                            print(f"❌ Giving up on {agent_key} agent after {max_retries} attempts")
                    
                if not success:
                    print(f"⚠️ Proceeding without {agent_key} agent")
                    
        except Exception as e:
            print(f"⚠️ Agent initialization failed: {e}")
            # Continue without direct agents - Discord-only mode
            
        end_time = time.time()
        print(f"🎯 Agent initialization completed in {end_time - start_time:.1f} seconds")

        # Post system event
        await self._post_system_event("agent_init", f"Agent initialization completed", f"Agents: {len(self.agent_instances)}, Duration: {end_time - start_time:.1f}s")

        self._initialize_agents()

    def get_command_channel(self, command: str) -> Optional[discord.TextChannel]:
        """Route commands to appropriate channels based on content and workflow phase"""
        command_lower = command.lower()

        # Route ACTUAL trade execution commands to alerts channel (not planning/analysis)
        if any(keyword in command_lower for keyword in [
            'execute trade', 'place order', 'buy signal', 'sell signal',
            'trade alert', 'execution validation', 'order placed', 'order filled'
        ]):
            if self.alerts_channel:
                print(f"🚨 Routing to alerts channel: {command[:50]}...")
                return self.alerts_channel

        # Route ACTUAL trade proposals and ranking RESULTS to ranked trades channel (not analysis/evaluation)
        if any(keyword in command_lower for keyword in [
            'rank opportunities', 'prioritize actions', 'trade proposal',
            'top ranked trades', 'ranked trade list', 'trade ranking results'
        ]):
            if self.ranked_trades_channel:
                print(f"📊 Routing to ranked trades channel: {command[:50]}...")
                return self.ranked_trades_channel

        # Default to general channel for analysis and coordination
        print(f"💬 Using general channel: {command[:50]}...")
        return self.channel

    async def send_trade_alert(self, alert_message: str, alert_type: str = "trade"):
        """Send a trade alert to the dedicated alerts channel with retries"""
        max_retries = 2
        retry_delay = 1  # seconds

        for attempt in range(max_retries + 1):
            try:
                if not self.alerts_channel:
                    print(f"⚠️ Alerts channel not available, sending to general channel")
                    if self.channel and hasattr(self.channel, 'send'):
                        general_channel = cast(discord.TextChannel, self.channel)
                        await general_channel.send(f"🚨 **TRADE ALERT** 🚨\n{alert_message}")
                    return

                # Format alert with appropriate emoji based on type
                emoji_map = {
                    "trade": "🚨",
                    "execution": "✅",
                    "risk": "⚠️",
                    "error": "❌",
                    "success": "🎯"
                }
                emoji = emoji_map.get(alert_type, "🚨")

                alerts_channel = cast(discord.TextChannel, self.alerts_channel)
                formatted_alert = f"{emoji} **TRADE ALERT** {emoji}\n{alert_message}"
                await alerts_channel.send(formatted_alert)
                print(f"✅ Trade alert sent to alerts channel: {alert_type}")

                # Post system event for trade activity
                await self._post_system_event(
                    "trade_alert",
                    f"Trade Alert Sent ({alert_type.upper()})",
                    f"Alert delivered to #{alerts_channel.name}",
                    {"alert_type": alert_type, "channel": alerts_channel.name}
                )

                return  # Success, exit

            except Exception as e:
                print(f"⚠️ Failed to send trade alert (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    # Final fallback to general channel
                    print("❌ All retries failed, falling back to general channel")
                    if self.channel and hasattr(self.channel, 'send'):
                        general_channel = cast(discord.TextChannel, self.channel)
                        await general_channel.send(f"🚨 **TRADE ALERT** 🚨\n{alert_message}")

    async def send_ranked_trade_info(self, trade_message: str, trade_type: str = "proposal"):
        """Send ranked trade information to the dedicated ranked trades channel with retries"""
        max_retries = 2
        retry_delay = 1  # seconds

        for attempt in range(max_retries + 1):
            try:
                if not self.ranked_trades_channel:
                    print(f"⚠️ Ranked trades channel not available, sending to general channel")
                    if self.channel and hasattr(self.channel, 'send'):
                        general_channel = cast(discord.TextChannel, self.channel)
                        await general_channel.send(f"📊 **RANKED TRADE {trade_type.upper()}** 📊\n{trade_message}")
                    return

                # Format trade info with appropriate emoji based on type
                emoji_map = {
                    "proposal": "📈",
                    "ranking": "🏆",
                    "analysis": "🔍",
                    "summary": "📊",
                    "update": "🔄"
                }
                emoji = emoji_map.get(trade_type, "📊")

                ranked_channel = cast(discord.TextChannel, self.ranked_trades_channel)
                formatted_trade = f"{emoji} **RANKED TRADE {trade_type.upper()}** {emoji}\n{trade_message}"
                await ranked_channel.send(formatted_trade)
                print(f"✅ Ranked trade info sent to ranked trades channel: {trade_type}")
                return  # Success, exit

            except Exception as e:
                print(f"⚠️ Failed to send ranked trade info (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    # Final fallback to general channel
                    print("❌ All retries failed, falling back to general channel")
                    if self.channel and hasattr(self.channel, 'send'):
                        general_channel = cast(discord.TextChannel, self.channel)
                        await general_channel.send(f"📊 **RANKED TRADE {trade_type.upper()}** 📊\n{trade_message}")

    def is_trade_related_message(self, message: str) -> bool:
        """Determine if a message contains trade-related content that should go to alerts"""
        trade_keywords = [
            'trade proposal', 'trade execution', 'buy signal', 'sell signal',
            'order placed', 'order filled', 'trade alert', 'execution validation',
            'trade recommendation', 'position opened', 'position closed'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in trade_keywords)

    def rank_trade_proposals(self, proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank trade proposals by confidence (descending) and expected return"""
        if not proposals:
            return proposals

        # Sort by confidence (descending), then by expected_return if available
        def sort_key(proposal):
            confidence = proposal.get('confidence', 0)
            if isinstance(confidence, str):
                # Try to parse as float
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0
            expected_return = proposal.get('expected_return', 0)
            if isinstance(expected_return, str):
                try:
                    expected_return = float(expected_return)
                except ValueError:
                    expected_return = 0
            return (-confidence, -expected_return)  # Negative for descending

        return sorted(proposals, key=sort_key)

    def _extract_trade_alert_info(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Extract key trade information from agent response for alerts, with enhanced string parsing"""
        try:
            agent_name = response_data.get('agent', 'Unknown')
            response = response_data.get('response', {})

            alert_parts = []

            if isinstance(response, dict):
                # Look for trade proposals in structured responses
                if 'trade_proposals' in response:
                    proposals = response['trade_proposals']
                    if isinstance(proposals, list) and proposals:
                        # Rank proposals before alerting
                        ranked_proposals = self.rank_trade_proposals(proposals)
                        alert_parts.append(f"**{agent_name.title()} Agent** generated {len(ranked_proposals)} ranked trade proposal(s):")
                        for i, proposal in enumerate(ranked_proposals[:3], 1):  # Limit to 3 proposals
                            if isinstance(proposal, dict):
                                instrument = proposal.get('instrument', 'Unknown')
                                action = proposal.get('action', 'Unknown')
                                confidence = proposal.get('confidence', 'Unknown')
                                alert_parts.append(f"• #{i} {action.upper()} {instrument} (Confidence: {confidence})")

                # Look for execution validation
                elif 'execution_plan' in response:
                    plan = response['execution_plan']
                    if isinstance(plan, dict):
                        instrument = plan.get('instrument', 'Unknown')
                        quantity = plan.get('quantity', 'Unknown')
                        alert_parts.append(f"**{agent_name.title()} Agent** validated execution:")
                        alert_parts.append(f"• Instrument: {instrument}")
                        alert_parts.append(f"• Quantity: {quantity}")

                # Look for risk assessment
                elif 'risk_assessment' in response:
                    assessment = response['risk_assessment']
                    if isinstance(assessment, dict):
                        risk_level = assessment.get('overall_risk', 'Unknown')
                        alert_parts.append(f"**{agent_name.title()} Agent** risk assessment:")
                        alert_parts.append(f"• Risk Level: {risk_level.upper()}")

            else:
                # Enhanced parsing for string responses
                response_str = str(response).lower()
                import re

                # Look for trade proposals in text (e.g., "BUY AAPL Confidence: 0.8")
                proposal_pattern = r'(buy|sell|hold)\s+(\w+)\s+confidence:\s*([\d.]+)'
                matches = re.findall(proposal_pattern, response_str, re.IGNORECASE)
                if matches:
                    alert_parts.append(f"**{agent_name.title()} Agent** has {len(matches)} trade proposal(s) in text:")
                    for action, instrument, confidence in matches[:3]:
                        alert_parts.append(f"• {action.upper()} {instrument} (Confidence: {confidence})")

                # Fallback to keyword detection
                elif 'trade proposal' in response_str:
                    alert_parts.append(f"**{agent_name.title()} Agent** has trade proposals ready for review")
                elif 'execution' in response_str:
                    alert_parts.append(f"**{agent_name.title()} Agent** provided execution guidance")

            if alert_parts:
                return "\n".join(alert_parts)
            else:
                return None

        except Exception as e:
            print(f"⚠️ Error extracting trade alert info: {e}")
            return None

    async def check_agent_health(self) -> Dict[str, Any]:
        """Check health status of all agents using BaseAgent methods"""
        if not AGENTS_AVAILABLE:
            return {
                'healthy_agents': [],
                'unhealthy_agents': [],
                'total_agents': 0,
                'overall_health': 'discord_only'
            }
            
        health_status = {
            'healthy_agents': [],
            'unhealthy_agents': [],
            'total_agents': len(self.agent_instances),
            'overall_health': 'unknown'
        }
        
        if not self.agent_instances:
            health_status['overall_health'] = 'no_agents'
            return health_status
            
        for agent_name, agent in self.agent_instances.items():
            try:
                # Use BaseAgent get_status method
                status = await agent.get_status()
                health_status_val = status.get('health_status', {}).get('overall_health', 'unknown')
                if health_status_val in ['healthy', 'good', 'online']:
                    health_status['healthy_agents'].append(agent_name)
                else:
                    health_status['unhealthy_agents'].append(agent_name)
            except Exception as e:
                print(f"⚠️ Health check failed for {agent_name}: {e}")
                health_status['unhealthy_agents'].append(agent_name)
        
        # Determine overall health
        healthy_count = len(health_status['healthy_agents'])
        total_count = health_status['total_agents']
        
        if healthy_count == total_count:
            health_status['overall_health'] = 'healthy'
        elif healthy_count >= total_count * 0.5:
            health_status['overall_health'] = 'degraded'
        else:
            health_status['overall_health'] = 'critical'
            
        return health_status

    async def send_direct_agent_command(self, agent_name: str, command: str, 
                                      data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Send command directly to agent using BaseAgent methods via A2A - SIMPLIFIED for full context sharing"""
        if agent_name not in self.agent_instances:
            print(f"⚠️ Agent {agent_name} not available for direct command")
            return None
            
        agent = self.agent_instances[agent_name]
        
        try:
            # Handle debate commands specially
            if 'debate' in command.lower():
                return await self._handle_debate_command(agent, command, data)
            
            # No more prefix stripping - agents respond based on content
            analysis_query = command
            if data:
                analysis_query += f" {data}"
                
            result = await agent.analyze(analysis_query)
            return result
                
        except Exception as e:
            print(f"⚠️ Direct agent command failed for {agent_name}: {e}")
            return {'error': str(e), 'agent': agent_name}

    async def _handle_debate_command(self, initiating_agent, command: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle debate commands by coordinating multiple agents via A2A"""
        try:
            # Parse debate command: '!m debate "topic" agent1 agent2 agent3'
            parts = command.split()
            if len(parts) < 4:
                return {'error': 'Invalid debate command format', 'expected': '!agent debate "topic" agent1 agent2 ...'}
            
            debate_topic = parts[2].strip('"')  # Remove quotes
            participant_names = parts[3:]  # Remaining agents
            
            # Validate participants
            participants = []
            for name in participant_names:
                if name in self.agent_instances:
                    participants.append(self.agent_instances[name])
                else:
                    return {'error': f'Agent {name} not available for debate'}
            
            if not participants:
                return {'error': 'No valid participants found for debate'}
            
            # Create debate session
            debate_session = {
                'topic': debate_topic,
                'participants': [p.role for p in participants],
                'initiator': initiating_agent.role,
                'timestamp': datetime.now().isoformat()
            }
            
            # Collect initial positions from all participants
            debate_responses = []
            for participant in participants:
                try:
                    # Ask each agent for their position on the debate topic
                    position = await participant.analyze(f"Debate Topic: {debate_topic}. What is your position and reasoning?")
                    debate_responses.append({
                        'agent': participant.role,
                        'position': position
                    })
                except Exception as e:
                    debate_responses.append({
                        'agent': participant.role,
                        'error': str(e)
                    })
            
            # Have agents respond to each other's positions
            for i, participant in enumerate(participants):
                other_positions = [
                    resp for j, resp in enumerate(debate_responses) 
                    if j != i and 'position' in resp
                ]
                
                if other_positions:
                    try:
                        # Ask agent to respond to other positions
                        response_prompt = f"Review these positions from other agents on '{debate_topic}': " + \
                                        ". ".join([f"{p['agent']}: {str(p['position'])[:200]}..." for p in other_positions]) + \
                                        ". Do you agree, disagree, or want to modify your position?"
                        
                        counter_response = await participant.analyze(response_prompt)
                        debate_responses[i]['counter_response'] = counter_response
                    except Exception as e:
                        debate_responses[i]['counter_error'] = str(e)
            
            # Synthesize debate results
            debate_result = {
                'debate_topic': debate_topic,
                'participants': len(participants),
                'responses': debate_responses,
                'consensus_level': self._calculate_debate_consensus(debate_responses),
                'key_insights': self._extract_debate_insights(debate_responses),
                'session_info': debate_session
            }
            
            return debate_result
            
        except Exception as e:
            return {'error': f'Debate coordination failed: {str(e)}'}

    def _calculate_debate_consensus(self, responses: List[Dict[str, Any]]) -> str:
        """Calculate consensus level from debate responses"""
        if not responses:
            return 'none'
        
        # Simple consensus calculation based on response similarity
        # In a real implementation, this could use NLP similarity metrics
        successful_responses = [r for r in responses if 'position' in r and 'error' not in r]
        
        if len(successful_responses) < 2:
            return 'insufficient_data'
        
        # For now, assume moderate consensus if all agents responded
        consensus_ratio = len(successful_responses) / len(responses)
        
        if consensus_ratio >= 0.8:
            return 'high'
        elif consensus_ratio >= 0.6:
            return 'moderate'
        else:
            return 'low'

    def _extract_debate_insights(self, responses: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from debate responses"""
        insights = []
        
        for response in responses:
            if 'position' in response and isinstance(response['position'], dict):
                # Look for key insights in structured responses
                if 'recommendations' in response['position']:
                    recs = response['position']['recommendations']
                    if isinstance(recs, list):
                        insights.extend(recs[:2])  # Take up to 2 insights per agent
        
        return insights[:5]  # Limit to 5 total insights

    async def create_collaborative_session(self, topic: str) -> bool:
        """Create a collaborative session using BaseAgent methods"""
        if not self.agent_instances:
            return False
            
        try:
            # Use the first available agent to create the session
            first_agent = next(iter(self.agent_instances.values()))
            session_id = await first_agent.create_collaborative_session(
                topic=topic,
                max_participants=len(self.agent_instances),
                session_timeout=3600  # 1 hour
            )
            
            if session_id:
                self.collaborative_session_id = session_id
                print(f"✅ Created collaborative session: {session_id}")
                
                # Join all agents to the session
                join_tasks = []
                for agent_name, agent in self.agent_instances.items():
                    if agent != first_agent:  # Creator already joined
                        task = agent.join_collaborative_session(session_id, {
                            'role': agent.role,
                            'capabilities': ['analysis', 'debate', 'consensus']
                        })
                        join_tasks.append(task)
                        
                await asyncio.gather(*join_tasks, return_exceptions=True)
                return True
            else:
                print("⚠️ Failed to create collaborative session")
                return False
                
        except Exception as e:
            print(f"⚠️ Collaborative session creation failed: {e}")
            return False

    async def share_workflow_context(self, context_key: str, context_data: Any):
        """Share workflow context using BaseAgent shared memory methods"""
        if not self.collaborative_session_id or not self.agent_instances:
            return
            
        try:
            # Use first agent to share context
            first_agent = next(iter(self.agent_instances.values()))
            success = await first_agent.update_session_context(
                self.collaborative_session_id,
                context_key,
                context_data
            )
            
            if success:
                print(f"✅ Shared workflow context: {context_key}")
            else:
                print(f"⚠️ Failed to share context: {context_key}")
                
        except Exception as e:
            print(f"⚠️ Context sharing failed: {e}")

    async def _share_full_workflow_context(self, phase_key: str, phase_title: str):
        """Share complete workflow context with all agents for enhanced collaboration"""
        if not self.agent_instances or not self.collaborative_session_id:
            return

        # Compile comprehensive context from all previous phases
        full_context = {
            'current_phase': phase_key,
            'phase_title': phase_title,
            'timestamp': datetime.now().isoformat(),
            'workflow_progress': self._get_workflow_progress_summary(),
            'all_previous_responses': self.responses_collected,
            'human_interventions': self.human_interventions,
            'agent_health_status': await self.check_agent_health(),
            'collaborative_session_id': self.collaborative_session_id
        }

        # Share with all agents via A2A protocol
        context_sharing_tasks = []
        for agent_name, agent in self.agent_instances.items():
            task = agent.update_session_context(
                self.collaborative_session_id,
                'full_workflow_context',
                full_context
            )
            context_sharing_tasks.append(task)

        await asyncio.gather(*context_sharing_tasks, return_exceptions=True)
        print(f"✅ Shared full workflow context with {len(self.agent_instances)} agents")

    async def _share_position_context(self):
        """Share current position data with all agents for position-aware trade proposals"""
        if not self.agent_instances or not self.collaborative_session_id:
            return

        # Get current position data (this would integrate with your trading platform)
        position_data = await self._get_current_positions()

        # Share position context with all agents
        position_sharing_tasks = []
        for agent_name, agent in self.agent_instances.items():
            task = agent.update_session_context(
                self.collaborative_session_id,
                'current_positions',
                position_data
            )
            position_sharing_tasks.append(task)

        await asyncio.gather(*position_sharing_tasks, return_exceptions=True)
        print(f"✅ Shared position context with {len(self.agent_instances)} agents")

    async def _get_current_positions(self) -> Dict[str, Any]:
        """Get current position data from trading platform with error handling"""
        try:
            # Integrate with IBKR connector
            from src.integrations.ibkr_connector import get_ibkr_connector
            connector = get_ibkr_connector()

            positions = await connector.get_positions()
            account_summary = await connector.get_account_summary()
            cash_balance = account_summary.get('NetLiquidation', 0) if account_summary else 0  # Use NetLiquidation as cash balance

            return {
                'timestamp': datetime.now().isoformat(),
                'positions': positions,
                'cash_balance': cash_balance
            }
            
        except ImportError as e:
            print(f"⚠️ IBKR integration not available: {e}")
            return {
                'error': 'IBKR integration missing',
                'timestamp': datetime.now().isoformat(),
                'positions': [],
                'cash_balance': 0.0
            }
        except Exception as e:
            print(f"⚠️ Failed to get position data: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'positions': [],
                'cash_balance': 0.0
            }

    async def _execute_commands_parallel(self, commands: List[str], phase_key: str) -> List[Dict[str, Any]]:
        """Execute all commands in parallel across agents using INTELLIGENT SCOPE-AWARE ROUTING with rate limiting"""
        if not self.agent_instances:
            return []

        all_responses = []
        max_wait_time = self.phase_delays.get(phase_key, 86400)

        # Get agent scope definitions for intelligent routing
        scope_definitions = get_agent_scope_definitions()

        # Create tasks for command-agent combinations with intelligent filtering
        execution_tasks = []

        for command in commands:
            # Determine which agents should process this command based on scope
            relevant_agents = self._filter_agents_by_command_scope(command, scope_definitions)

            for agent_name in relevant_agents:
                if agent_name in self.agent_instances:
                    agent = self.agent_instances[agent_name]
                    task = self._send_context_aware_command(agent_name, agent, command, phase_key)
                    execution_tasks.append((agent_name, command, task))

        # If no agents were deemed relevant, fall back to sending to all agents (with low confidence)
        if not execution_tasks:
            print(f"⚠️ No agents deemed relevant for commands, falling back to universal broadcast")
            for command in commands:
                for agent_name, agent in self.agent_instances.items():
                    task = self._send_context_aware_command(agent_name, agent, command, phase_key)
                    execution_tasks.append((agent_name, command, task))

        # Execute tasks in batches to prevent overwhelming resources (rate limiting)
        batch_size = 5  # Process 5 tasks at a time
        for i in range(0, len(execution_tasks), batch_size):
            batch = execution_tasks[i:i + batch_size]
            
            # Execute batch with timeout
            start_time = time.time()
            
            try:
                results = await asyncio.gather(
                    *[task for _, _, task in batch],
                    return_exceptions=True
                )

                # Process results
                for (agent_name, command, _), result in zip(batch, results):
                    if isinstance(result, Exception):
                        print(f"⚠️ Parallel execution failed for {agent_name}: {result}")

                        # Alert on agent command failure
                        self.alert_manager.error(
                            Exception(f"Agent command failed: {agent_name} - {str(result)}"),
                            {
                                "agent": agent_name,
                                "command": command[:100],
                                "phase": phase_key,
                                "error_type": type(result).__name__
                            },
                            "orchestrator"
                        )

                        # Post system event for agent error
                        await self._post_system_event(
                            "agent_error",
                            f"Agent Execution Failed: {agent_name}",
                            f"Command: {command[:50]}...\nError: {str(result)[:100]}",
                            {"agent": agent_name, "command_type": command.split()[0] if command else "unknown"}
                        )

                        continue

                    if result:
                        response_data = {
                            'agent': agent_name,
                            'method': 'intelligent_scope_routing',
                            'response': result,
                            'phase': phase_key,
                            'command': command,
                            'execution_time': time.time() - start_time
                        }
                        all_responses.append(response_data)
                        print(f"✅ Intelligent response from {agent_name} for command: {command[:50]}...")

            except asyncio.TimeoutError:
                print(f"⏰ Intelligent parallel execution timed out after {max_wait_time}s")

            # Brief delay between batches for rate limiting
            await asyncio.sleep(1)

        # Sort responses by agent for consistent presentation
        all_responses.sort(key=lambda x: x['agent'])

        return all_responses

    def _filter_agents_by_command_scope(self, command: str, scope_definitions) -> List[str]:
        """
        Intelligently filter which agents should process a command based on scope definitions.
        Enables hive mind collaboration while maintaining boundaries.
        """
        relevant_agents = []

        for agent_name in self.agent_instances.keys():
            # Use the scope system's command filtering logic
            decision = scope_definitions.should_agent_process_command(agent_name, command)

            if decision['should_process']:
                relevant_agents.append(agent_name)
                print(f"🎯 Agent {agent_name} deemed relevant for command (confidence: {decision['confidence']:.2f})")
            elif decision['action'] == 'delegate':
                # Allow delegation - the agent can still see the command and delegate to others
                relevant_agents.append(agent_name)
                print(f"👥 Agent {agent_name} will receive command for delegation to {decision['delegate_to']}")
            else:
                print(f"🚫 Agent {agent_name} filtered out for command (action: {decision['action']})")

        # Ensure at least some agents get the command (fallback for edge cases)
        if not relevant_agents:
            print("⚠️ No agents deemed relevant, enabling all agents for maximum collaboration")
            relevant_agents = list(self.agent_instances.keys())

        return relevant_agents

    async def _send_context_aware_command(self, agent_name: str, agent, command: str, phase_key: str) -> Optional[Dict[str, Any]]:
        """Send command to agent with full context awareness"""
        try:
            # Enrich command with context
            enriched_command = self._enrich_command_with_context(command, agent_name, phase_key)

            # Send via A2A protocol
            result = await self.send_direct_agent_command(agent_name, enriched_command)

            # Allow agent to see and respond to other agents' analyses
            if result and self.collaborative_session_id:
                # Share this agent's response with others for cross-agent learning
                await agent.contribute_session_insight(
                    self.collaborative_session_id,
                    {
                        'type': 'phase_contribution',
                        'phase': phase_key,
                        'agent': agent_name,
                        'command': command,
                        'analysis': result,
                        'timestamp': datetime.now().isoformat()
                    }
                )

            return result

        except Exception as e:
            print(f"⚠️ Context-aware command failed for {agent_name}: {e}")
            return None

    def _enrich_command_with_context(self, command: str, agent_name: str, phase_key: str) -> str:
        """Enrich command with relevant context for the specific agent"""
        # Get phase-specific context enrichment
        phase_context = self._get_phase_context_enrichment(phase_key, agent_name)

        # Add position awareness to trade-related commands
        if any(word in command.lower() for word in ['trade', 'proposal', 'strategy', 'position']):
            command += " Consider current portfolio positions and risk constraints."

        # Add cross-agent collaboration context
        command += f" {phase_context}"

        return command

    def _get_phase_context_enrichment(self, phase_key: str, agent_name: str) -> str:
        """Get phase-specific context enrichment for agents"""
        phase_contexts = {
            'macro_foundation_analysis': "Build on the collected economic and market data. Consider how macro regime affects sector opportunities.",
            'intelligence_gathering': "Use technical, fundamental, and sentiment data to identify specific trade setups. Cross-validate with institutional activity.",
            'strategy_development': "Develop actionable trade proposals considering current market regime and position constraints.",
            'risk_assessment': "Evaluate risk-adjusted returns for proposed trades, considering portfolio diversification and tail risk.",
            'consensus': "Synthesize all agent perspectives to identify the strongest trade opportunities with highest conviction.",
            'execution_validation': "Ensure trade proposals are executable given current market conditions and position limits.",
            'learning': "Learn from historical trade performance to improve future proposal quality."
        }

        base_context = phase_contexts.get(phase_key, "Contribute your specialized analysis to the collaborative decision process.")

        # Add agent-specific context
        agent_contexts = {
            'macro': "Focus on broader market regime and sector allocation opportunities.",
            'data': "Provide data-driven insights and validate information quality.",
            'strategy': "Develop specific, actionable trade proposals with clear entry/exit criteria.",
            'risk': "Assess risk/reward profiles and identify potential failure modes.",
            'reflection': "Synthesize perspectives and identify consensus vs disagreement.",
            'execution': "Validate practical execution feasibility and costs.",
            'learning': "Apply historical lessons to improve current proposals."
        }

        agent_specific = agent_contexts.get(agent_name, "")
        return f"{base_context} {agent_specific}"

    def _get_workflow_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive workflow progress summary"""
        completed_phases = []
        current_responses = {}

        # Analyze responses by phase
        for response in self.responses_collected:
            phase = response.get('phase', 'unknown')
            if phase not in current_responses:
                current_responses[phase] = []
            current_responses[phase].append(response)

        # Identify completed phases
        for phase_key in self.phase_commands.keys():
            if phase_key in current_responses and len(current_responses[phase_key]) > 0:
                completed_phases.append(phase_key)

        return {
            'completed_phases': completed_phases,
            'current_phase': self.current_phase,
            'total_responses': len(self.responses_collected),
            'responses_by_phase': {phase: len(responses) for phase, responses in current_responses.items()},
            'human_interventions': len(self.human_interventions),
            'key_insights': self._extract_key_workflow_insights()
        }

    def _extract_key_workflow_insights(self) -> List[str]:
        """Extract key insights from workflow responses"""
        insights = []

        # Look for trade proposals across all responses
        for response in self.responses_collected:
            response_data = response.get('response', {})
            if isinstance(response_data, dict):
                # Look for trade proposals in structured responses
                if 'trade_proposals' in response_data:
                    proposals = response_data['trade_proposals']
                    if isinstance(proposals, list):
                        for proposal in proposals[:2]:  # Limit per response
                            if isinstance(proposal, dict) and 'description' in proposal:
                                insights.append(f"{response.get('agent', 'Unknown')}: {proposal['description'][:100]}...")

        return insights[:10]  # Limit total insights

    async def execute_phase_with_agents(self, phase_key: str, phase_title: str):
        """Execute a phase using ENHANCED PARALLEL COLLABORATION - agents share full workflow context and collaborate simultaneously"""
        logger.info(f"Discord bot connection status - ready: {self.discord_ready.is_set()}, channel: {self.channel is not None}, client: {self.client is not None}")
        if not self.channel:
            logger.error(f"No channel available for phase {phase_key}")
            self.alert_manager.error(
                Exception(f"No Discord channel available for phase execution: {phase_key}"),
                {"phase": phase_key, "phase_title": phase_title},
                "orchestrator"
            )
            return

        general_channel = cast(discord.TextChannel, self.channel)
        self.current_phase = phase_key

        # Announce phase start in general channel
        await general_channel.send(f"\n{phase_title}")
        await general_channel.send("─" * 50)

        commands = self.phase_commands.get(phase_key, [])

        # Post system event - COMMENTED OUT FOR TROUBLESHOOTING
        # await self._post_system_event("phase_start", f"{phase_key.replace('_', ' ').title()} phase beginning", f"Commands: {len(commands)}, Agents: {len(self.agent_instances)}")

        # ENHANCED PARALLEL EXECUTION: Share complete workflow context with all agents
        await self._share_full_workflow_context(phase_key, phase_title)

        # POSITION AWARENESS: Include current position data in context
        await self._share_position_context()

        # SEND ALL COMMANDS TO ALL AGENTS SIMULTANEOUSLY for maximum collaboration
        agent_responses = await self._execute_commands_parallel(commands, phase_key)
        self.responses_collected.extend(agent_responses)

        # Generate and post debate summary - COMMENTED OUT FOR TROUBLESHOOTING
        # await self._post_debate_summary(general_channel, agent_responses, phase_key)

        # Announce parallel execution with clear separation
        await general_channel.send(f"\n🎯 **PARALLEL EXECUTION:** {len(commands)} commands sent to {len(self.agent_instances)} agents simultaneously!")
        await general_channel.send("🤝 **Agents collaborating with full shared context - no silos!**")

        # Format and display agent responses with enhanced readability
        if agent_responses:
            logger.info(f"Agent responses populated with {len(agent_responses)} items: {[r.get('agent', 'unknown') for r in agent_responses]}")
            # Test Discord connection before sending responses
            try:
                await general_channel.send("🔄 **Testing Discord connection before sending agent responses...**")
                logger.info("Discord test message sent successfully")
            except Exception as e:
                logger.error(f"Failed to send test message to Discord: {e}")
                await self.alert_manager.error(
                    Exception(f"Discord connection test failed: {e}"),
                    {"phase": "discord_test"},
                    "discord_integration"
                )
                return  # Don't proceed if Discord is broken
            await self._present_agent_responses_enhanced(general_channel, agent_responses, phase_key)
        else:
            await general_channel.send("\n⏰ **No agent responses received** - but they're thinking hard! 🤔")

        await general_channel.send(f"\n✅ **{phase_title} Complete! Alpha discovered!** 🎉")

        # Post system event - COMMENTED OUT FOR TROUBLESHOOTING
        # await self._post_system_event("phase_end", f"{phase_key.replace('_', ' ').title()} phase completed", f"Responses: {len(agent_responses)}, Duration: ~{(time.time() - time.time()) if 'start_time' in locals() else 'unknown'}s")

        await asyncio.sleep(2)  # Reduced from 3 to speed up

    async def _present_agent_responses_enhanced(self, channel: discord.TextChannel, responses: List[Dict[str, Any]], phase_key: str):
        """Present agent responses in a professional format with logical segmentation"""
        logger.info(f"_present_agent_responses_enhanced called with {len(responses)} responses for phase {phase_key}")
        if not responses:
            return

        # Verify channel is valid and bot has permissions
        if not channel:
            logger.error("general_channel is None, cannot send agent responses to Discord")
            return
        if not isinstance(channel, discord.TextChannel):
            logger.error(f"general_channel is not a TextChannel: {type(channel)}")
            return
        if not channel.permissions_for(channel.guild.me).send_messages:
            logger.error("Bot does not have send_messages permission in the general channel")
            return

        # Group responses by agent
        agent_summaries = {}
        detailed_responses = []

        for response_data in responses:
            agent_name = response_data['agent']
            response = response_data['response']
            command = response_data.get('command', '')

            # Initialize agent summary if not exists
            if agent_name not in agent_summaries:
                agent_summaries[agent_name] = []

            # SECURITY: Sanitize agent output before display
            sanitized_response = self._sanitize_agent_output(response)

            # Create professional agent summary
            if isinstance(sanitized_response, dict):
                # Handle structured agent responses
                response_dict = cast(Dict[str, Any], sanitized_response)
                if 'analysis_type' in response_dict:
                    summary = f"{agent_name.title()} Agent: {response_dict.get('analysis_type', 'analysis')} analysis"
                    if 'confidence_level' in response_dict:
                        summary += f" (Confidence: {response_dict['confidence_level']})"
                elif 'error' in response_dict:
                    summary = f"{agent_name.title()} Agent: Working through {response_dict['error']}"
                else:
                    summary = f"{agent_name.title()} Agent: Analysis complete"
            else:
                # Handle string responses
                summary = f"{agent_name.title()} Agent: {str(sanitized_response)[:100]}..."

            agent_summaries[agent_name].append(summary)

            # Prepare detailed response with professional formatting and logical segmentation
            if isinstance(sanitized_response, dict):
                detailed = f"**{agent_name.title()} Agent Analysis**\n"
                detailed += f"Command: {command}\n"
                for key, value in sanitized_response.items():
                    if key not in ['timestamp', 'agent_role']:
                        formatted_value = self._format_response_value(key, value)
                        emoji_map = {
                            'trade_proposals': '📈',
                            'confidence_level': '🎯',
                            'risk_assessment': '⚠️',
                            'analysis': '🔍',
                            'recommendations': '💡'
                        }
                        emoji = emoji_map.get(key, '•')
                        detailed += f"{emoji} **{key.replace('_', ' ').title()}:** {formatted_value}\n"
                detailed_responses.append(detailed.rstrip())
            else:
                # Format string responses with logical segmentation
                formatted_response = self._format_text_response_professionally(str(sanitized_response))
                detailed_responses.append(f"**{agent_name.title()} Agent:** {formatted_response}")

        # Send professional summary first
        summary_text = f"Collaborative analysis complete: {len(responses)} responses from {len(agent_summaries)} agents\n"
        for agent_name, summaries in agent_summaries.items():
            summary_text += f"• **{agent_name.title()}:** {len(summaries)} contributions\n"
        try:
            await channel.send(summary_text)
            logger.info("Successfully sent agent response summary to Discord")
        except Exception as e:
            logger.error(f"Failed to send agent response summary to Discord: {e}")
        await asyncio.sleep(1)  # Rate limit prevention

        # Send detailed responses with logical segmentation
        for detailed in detailed_responses:
            # Break into logical segments instead of hard character limits
            segments = self._break_into_logical_segments(detailed)
            for segment in segments:
                if segment.strip():  # Only send non-empty segments
                    try:
                        await channel.send(segment)
                        logger.debug("Successfully sent agent response segment to Discord")
                    except Exception as e:
                        logger.error(f"Failed to send agent response segment to Discord: {e}")
                    await asyncio.sleep(1)  # Rate limit prevention

        # Check for trade-related content and send alerts
        for response_data in responses:
            response = response_data['response']
            if isinstance(response, dict):
                response_text = str(response)
            else:
                response_text = str(response)

            if self.is_trade_related_message(response_text):
                alert_content = self._extract_trade_alert_info(response_data)
                if alert_content:
                    await self.send_trade_alert(alert_content, "trade")

        # Check for ranked trade proposals and send to dedicated channel
        for response_data in responses:
            response = response_data['response']
            agent_name = response_data.get('agent', 'Unknown')

            if isinstance(response, dict):
                # Look for trade proposals in structured responses
                if 'trade_proposals' in response:
                    proposals = response['trade_proposals']
                    if isinstance(proposals, list) and proposals:
                        # Rank proposals before displaying
                        ranked_proposals = self.rank_trade_proposals(proposals)
                        ranked_trade_message = f"**{agent_name.title()} Agent** presented {len(ranked_proposals)} ranked trade proposal(s):\n\n"
                        for i, proposal in enumerate(ranked_proposals[:5], 1):  # Limit to 5 proposals
                            if isinstance(proposal, dict):
                                instrument = proposal.get('instrument', 'Unknown')
                                action = proposal.get('action', 'Unknown')
                                confidence = proposal.get('confidence', 'Unknown')
                                reasoning = proposal.get('reasoning', 'No reasoning provided')

                                ranked_trade_message += f"**#{i} {action.upper()} {instrument}**\n"
                                ranked_trade_message += f"• Confidence: {confidence}\n"
                                ranked_trade_message += f"• Reasoning: {reasoning[:200]}...\n\n"

                        await self.send_ranked_trade_info(ranked_trade_message, "proposal")

        await channel.send("Parallel collaboration complete.")
        await asyncio.sleep(1)  # Rate limit prevention

    async def _post_debate_summary(self, channel: discord.TextChannel, responses: List[Dict[str, Any]], phase_key: str):
        """Generate and post a comprehensive summary of agent debates during collaborative sessions."""
        if not responses:
            return

        try:
            # Collect comprehensive insights from responses
            key_insights = []
            recommendations = []
            risk_assessments = []
            data_findings = []
            strategy_proposals = []
            consensus_points = []
            active_debates = []
            agent_contributions = {}
            confidence_levels = []
            execution_readiness = []

            for response_data in responses:
                agent_name = response_data['agent']
                response = response_data['response']
                command = response_data.get('command', '')

                # Track agent contributions
                if agent_name not in agent_contributions:
                    agent_contributions[agent_name] = 0
                agent_contributions[agent_name] += 1

                # Extract comprehensive insights from response
                if isinstance(response, dict):
                    # Extract structured data
                    if 'key_insights' in response:
                        key_insights.extend([f"{agent_name}: {insight}" for insight in response['key_insights'][:3]])
                    if 'recommendations' in response:
                        recommendations.extend([f"{agent_name}: {rec}" for rec in response['recommendations'][:2]])
                    if 'risk_assessment' in response:
                        risk_assessments.append(f"{agent_name}: {response['risk_assessment']}")
                    if 'data_findings' in response:
                        data_findings.extend([f"{agent_name}: {finding}" for finding in response['data_findings'][:2]])
                    if 'strategy_proposals' in response:
                        strategy_proposals.extend([f"{agent_name}: {prop}" for prop in response['strategy_proposals'][:2]])
                    if 'consensus' in response:
                        consensus_points.append(f"{agent_name}: {response['consensus']}")
                    if 'disagreements' in response or 'debates' in response:
                        debates = response.get('disagreements', response.get('debates', []))
                        active_debates.extend([f"{agent_name}: {debate}" for debate in debates[:2]])
                    if 'confidence_level' in response:
                        confidence_levels.append(f"{agent_name}: {response['confidence_level']}")
                    if 'execution_readiness' in response:
                        execution_readiness.append(f"{agent_name}: {response['execution_readiness']}")
                else:
                    # Extract from unstructured text response
                    response_text = str(response).lower()
                    if len(response_text) > 20:
                        # Extract meaningful insights based on content
                        if any(word in response_text for word in ['recommend', 'suggest', 'propose']):
                            recommendations.append(f"{agent_name}: {response_text[:150]}...")
                        elif any(word in response_text for word in ['risk', 'caution', 'warning']):
                            risk_assessments.append(f"{agent_name}: {response_text[:150]}...")
                        elif any(word in response_text for word in ['data', 'analysis', 'metric']):
                            data_findings.append(f"{agent_name}: {response_text[:150]}...")
                        elif any(word in response_text for word in ['strategy', 'approach', 'plan']):
                            strategy_proposals.append(f"{agent_name}: {response_text[:150]}...")
                        else:
                            key_insights.append(f"{agent_name}: {response_text[:150]}...")

            # Generate comprehensive summary
            summary_parts = []
            embed_fields = []

            # Header with participation stats
            active_agents = len(agent_contributions)
            total_responses = sum(agent_contributions.values())
            avg_confidence = "N/A"
            if confidence_levels:
                try:
                    confidences = [float(level.split(':')[1].strip().rstrip('%')) for level in confidence_levels if '%' in level.split(':')[1]]
                    if confidences:
                        avg_confidence = f"{sum(confidences)/len(confidences):.1f}%"
                except:
                    pass

            summary_parts.append(f"🤝 **Comprehensive Debate Summary**")
            summary_parts.append(f"📊 **Participation:** {active_agents} agents, {total_responses} responses")
            if avg_confidence != "N/A":
                summary_parts.append(f"🎯 **Avg Confidence:** {avg_confidence}")

            # Key Insights (most important section)
            if key_insights:
                embed_fields.append({
                    "name": "💡 Key Insights",
                    "value": "\n".join(key_insights[:5]) + (f"\n... and {len(key_insights)-5} more" if len(key_insights) > 5 else ""),
                    "inline": False
                })

            # Strategic Recommendations
            if recommendations:
                embed_fields.append({
                    "name": "🎯 Strategic Recommendations",
                    "value": "\n".join(recommendations[:4]) + (f"\n... and {len(recommendations)-4} more" if len(recommendations) > 4 else ""),
                    "inline": False
                })

            # Risk Assessments
            if risk_assessments:
                embed_fields.append({
                    "name": "⚠️ Risk Assessments",
                    "value": "\n".join(risk_assessments[:3]) + (f"\n... and {len(risk_assessments)-3} more" if len(risk_assessments) > 3 else ""),
                    "inline": False
                })

            # Data Findings
            if data_findings:
                embed_fields.append({
                    "name": "📈 Data & Analysis Findings",
                    "value": "\n".join(data_findings[:3]) + (f"\n... and {len(data_findings)-3} more" if len(data_findings) > 3 else ""),
                    "inline": False
                })

            # Strategy Proposals
            if strategy_proposals:
                embed_fields.append({
                    "name": "🚀 Strategy Proposals",
                    "value": "\n".join(strategy_proposals[:3]) + (f"\n... and {len(strategy_proposals)-3} more" if len(strategy_proposals) > 3 else ""),
                    "inline": False
                })

            # Consensus & Active Debates
            consensus_debate_parts = []
            if consensus_points:
                consensus_debate_parts.append(f"✅ **Consensus:** {consensus_points[0][:100]}{'...' if len(consensus_points[0]) > 100 else ''}")
            if active_debates:
                consensus_debate_parts.append(f"⚖️ **Active Debates:** {len(active_debates)} discussion points")

            if consensus_debate_parts:
                embed_fields.append({
                    "name": "🤝 Consensus & Discussions",
                    "value": "\n".join(consensus_debate_parts),
                    "inline": False
                })

            # Execution Readiness
            if execution_readiness:
                embed_fields.append({
                    "name": "⚡ Execution Readiness",
                    "value": "\n".join(execution_readiness[:2]) + (f"\n... and {len(execution_readiness)-2} more" if len(execution_readiness) > 2 else ""),
                    "inline": False
                })

            # Phase context
            embed_fields.append({
                "name": "📊 Phase Context",
                "value": f"**Current Phase:** {phase_key.replace('_', ' ').title()}\n**Timestamp:** <t:{int(time.time())}:f>",
                "inline": True
            })

            # Create and send embed
            embed = discord.Embed(
                title="🤖 **Real-Time Comprehensive Debate Summary**",
                description="\n".join(summary_parts),
                color=0x3498db,
                timestamp=datetime.now()
            )

            for field in embed_fields:
                embed.add_field(**field)

            embed.set_footer(text="ABC Application - Agent Collaboration Summary")

            await channel.send(embed=embed)

        except Exception as e:
            # Enhanced fallback with more details
            try:
                embed = discord.Embed(
                    title="🤝 **Debate Summary**",
                    description=f"Processed {len(responses)} agent responses in {phase_key.replace('_', ' ').title()} phase\nError in detailed analysis: {str(e)[:100]}",
                    color=0xffa500,
                    timestamp=datetime.now()
                )
                embed.add_field(name="📊 Basic Stats", value=f"**Agents:** {len(set(r['agent'] for r in responses))}\n**Responses:** {len(responses)}", inline=True)
                await channel.send(embed=embed)
            except:
                await channel.send(f"🤝 **Debate Summary:** {len(responses)} agent responses processed in {phase_key} phase")

    async def _post_system_event(self, event_type: str, message: str, details: str = "", embed_data: Optional[Dict[str, Any]] = None):
        """Post comprehensive real-time system event summaries to Discord."""
        try:
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)

                # Enhanced event emojis and colors
                event_config = {
                    "workflow_start": {"emoji": "🚀", "color": 0x3498db, "title": "Workflow Initiated"},
                    "workflow_end": {"emoji": "✅", "color": 0x2ecc71, "title": "Workflow Completed"},
                    "workflow_pause": {"emoji": "⏸️", "color": 0xf39c12, "title": "Workflow Paused"},
                    "workflow_resume": {"emoji": "▶️", "color": 0x3498db, "title": "Workflow Resumed"},
                    "phase_start": {"emoji": "📊", "color": 0x9b59b6, "title": "Phase Started"},
                    "phase_end": {"emoji": "🎯", "color": 0x2ecc71, "title": "Phase Completed"},
                    "agent_init": {"emoji": "🤖", "color": 0x1abc9c, "title": "Agent Initialization"},
                    "agent_error": {"emoji": "⚠️", "color": 0xe74c3c, "title": "Agent Error"},
                    "health_check": {"emoji": "🔍", "color": 0x95a5a6, "title": "Health Check"},
                    "trade_alert": {"emoji": "🚨", "color": 0xe67e22, "title": "Trade Alert"},
                    "data_update": {"emoji": "📈", "color": 0x27ae60, "title": "Data Update"},
                    "error": {"emoji": "❌", "color": 0xc0392b, "title": "System Error"},
                    "success": {"emoji": "✅", "color": 0x27ae60, "title": "Success"},
                    "warning": {"emoji": "⚠️", "color": 0xf39c12, "title": "Warning"},
                    "info": {"emoji": "ℹ️", "color": 0x3498db, "title": "Information"}
                }

                config = event_config.get(event_type, {"emoji": "ℹ️", "color": 0x95a5a6, "title": "System Event"})

                # Create embed for comprehensive events
                embed = discord.Embed(
                    title=f"{config['emoji']} **{config['title']}**",
                    description=f"**{message}**",
                    color=config['color'],
                    timestamp=datetime.now()
                )

                if details:
                    embed.add_field(name="📋 Details", value=details[:1000], inline=False)

                # Add embed data if provided
                if embed_data:
                    for key, value in embed_data.items():
                        if isinstance(value, dict):
                            # Handle nested data
                            formatted_value = "\n".join([f"• {k}: {v}" for k, v in value.items()][:5])
                        elif isinstance(value, list):
                            formatted_value = "\n".join([f"• {item}" for item in value][:5])
                        else:
                            formatted_value = str(value)

                        if len(formatted_value) > 1000:
                            formatted_value = formatted_value[:997] + "..."

                        embed.add_field(name=f"📊 {key.replace('_', ' ').title()}", value=formatted_value, inline=True)

                # Add system context
                embed.add_field(
                    name="🔧 System Context",
                    value=f"**Active Agents:** {len(self.agent_instances)}\n**Current Phase:** {getattr(self, 'current_phase', 'None')}\n**Timestamp:** <t:{int(time.time())}:f>",
                    inline=True
                )

                embed.set_footer(text="ABC Application - Real-Time System Monitor")

                await general_channel.send(embed=embed)

        except Exception as e:
            print(f"Failed to post system event: {e}")
            # Fallback to simple message
            try:
                if self.channel and hasattr(self.channel, 'send'):
                    general_channel = cast(discord.TextChannel, self.channel)
                    await general_channel.send(f"ℹ️ **System Event:** {message}")
            except:
                pass

    def _format_response_value(self, key: str, value: Any) -> str:
        """Format individual response values with appropriate Discord markdown"""
        if isinstance(value, (int, float)):
            # Format numbers appropriately
            if isinstance(value, float) and key.lower() in ['pnl', 'return', 'profit', 'loss', 'percentage', 'pct']:
                return f"{value:.3f}"  # 3 decimal places for percentages
            elif isinstance(value, float):
                return f"{value:.2f}"  # 2 decimal places for other floats
            else:
                return str(value)
        elif isinstance(value, str) and len(value) < 200:
            return value
        elif isinstance(value, list) and len(value) <= 5:
            # Format small lists nicely
            return ', '.join(str(x) for x in value)
        elif isinstance(value, dict) and len(value) <= 3:
            # Format small dicts as inline key-value pairs
            pairs = [f"{k}: {v}" for k, v in value.items() if len(str(v)) < 50]
            return ' | '.join(pairs) if pairs else str(value)[:100]
        else:
            # For complex data, truncate appropriately
            str_value = str(value)
            return str_value[:200] + "..." if len(str_value) > 200 else str_value

    def _format_text_response(self, text: str) -> str:
        """Format text responses with table/chart detection and proper markdown"""
        # Check for table-like content (multiple lines with similar structure)
        lines = text.strip().split('\n')
        if len(lines) > 3:
            # Look for potential table patterns
            if self._is_table_content(lines):
                # Format as code block for better table display
                return f"```\n{text}\n```"
            elif self._is_chart_content(text):
                # Format charts/data as code block
                return f"```\n{text}\n```"

        # Check for JSON-like content
        if text.strip().startswith(('{', '[')):
            try:
                import json
                parsed = json.loads(text)
                # Format JSON as code block
                return f"```json\n{json.dumps(parsed, indent=2)}\n```"
            except:
                pass

        # Return as regular text if no special formatting needed
        return text

    def _is_table_content(self, lines: List[str]) -> bool:
        """Detect if content appears to be tabular data"""
        if len(lines) < 3:
            return False

        # Check for common table indicators
        separators = ['|', '\t', ',', ';']
        has_separator = any(any(sep in line for sep in separators) for line in lines)

        # Check for consistent column structure
        if has_separator:
            first_line_separators = sum(1 for sep in separators if sep in lines[0])
            consistent_structure = all(
                sum(1 for sep in separators if sep in line) == first_line_separators
                for line in lines[1:3]  # Check first few lines
            )
            return consistent_structure

        return False

    def _is_chart_content(self, text: str) -> bool:
        """Detect if content appears to be chart/data visualization"""
        chart_indicators = [
            '│', '─', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼',  # Box drawing chars
            '█', '▌', '▐', '░', '▒', '▓',  # Block elements
            '↑', '↓', '→', '←',  # Arrows
            'plot', 'chart', 'graph', 'axis', 'scale'  # Keywords
        ]

        return any(indicator in text.lower() for indicator in chart_indicators)

    def _smart_chunk_message(self, message: str, max_length: int) -> List[str]:
        """Smartly chunk messages at natural break points"""
        if len(message) <= max_length:
            return [message]

        chunks = []
        remaining = message

        while len(remaining) > max_length:
            # Find the best break point
            chunk = remaining[:max_length]

            # Try to break at paragraph breaks first
            last_paragraph = chunk.rfind('\n\n')
            if last_paragraph > max_length * 0.7:  # Don't break too early
                chunk = remaining[:last_paragraph]
                remaining = remaining[last_paragraph:].lstrip()
            else:
                # Try to break at line breaks
                last_line = chunk.rfind('\n')
                if last_line > max_length * 0.8:
                    chunk = remaining[:last_line]
                    remaining = remaining[last_line:].lstrip()
                else:
                    # Force break at word boundary
                    last_space = chunk.rfind(' ')
                    if last_space > max_length * 0.9:
                        chunk = remaining[:last_space]
                        remaining = remaining[last_space:].lstrip()
                    else:
                        # Hard break if necessary
                        chunk = remaining[:max_length]
                        remaining = remaining[max_length:]

            chunks.append(chunk)

        if remaining:
            chunks.append(remaining)

        return chunks

    def _break_into_logical_segments(self, text: str) -> List[str]:
        """Break text into logical segments for better readability"""
        if not text or not text.strip():
            return []

        segments = []
        lines = text.split('\n')
        current_segment = []

        for line in lines:
            line = line.strip()
            if not line:
                # Empty line - check if we should end current segment
                if current_segment:
                    segments.append('\n'.join(current_segment))
                    current_segment = []
                continue

            # Check for segment break indicators
            is_new_segment = False

            # Numbered steps (1., 2., (1), (2), etc.)
            if re.match(r'^\d+\.|\(\d+\)|\d+\)', line):
                is_new_segment = True

            # Bullet points (-, •, *, +, etc.)
            elif re.match(r'^[-•*+]\s', line):
                is_new_segment = True

            # Headings (**, ##, etc.)
            elif line.startswith(('**', '##', '###', '####')):
                is_new_segment = True

            # Analysis keywords that indicate new thinking
            elif any(keyword in line.lower() for keyword in [
                'firstly', 'secondly', 'thirdly', 'next', 'then', 'furthermore',
                'however', 'therefore', 'consequently', 'additionally',
                'analysis:', 'assessment:', 'evaluation:', 'conclusion:',
                'recommendation:', 'summary:', 'key points:', 'findings:'
            ]):
                is_new_segment = True

            # Section headers in agent responses
            elif ':' in line and len(line.split(':')[0]) < 30 and line[0].isupper():
                is_new_segment = True

            if is_new_segment and current_segment:
                # End current segment and start new one
                segments.append('\n'.join(current_segment))
                current_segment = [line]
            else:
                # Continue current segment
                current_segment.append(line)

        # Add final segment if any
        if current_segment:
            segments.append('\n'.join(current_segment))

        # Filter out empty segments and ensure minimum length for readability
        filtered_segments = []
        for segment in segments:
            segment = segment.strip()
            if segment and len(segment) > 10:  # Minimum 10 chars to be meaningful
                filtered_segments.append(segment)

        # If no logical segments found, fall back to paragraph-based splitting
        if len(filtered_segments) <= 1:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if len(paragraphs) > 1:
                return paragraphs
            else:
                # Last resort: return as single segment
                return [text] if text else []

        return filtered_segments

    def _format_text_response_professionally(self, text: str) -> str:
        """Format text response for professional segment-based presentation"""
        # Basic formatting - remove excessive whitespace but preserve structure
        formatted = text.strip()

        # Ensure proper line endings
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)

        # Return as clean text without excessive emojis or styling
        return formatted

    def _initialize_workflow_commands(self):
        """Initialize the INSTITUTIONAL ALPHA DISCOVERY FRAMEWORK - Professional multi-agent trading analysis framework"""
        self.phase_commands = {
            'systematic_market_surveillance': [
                # Comprehensive market scanning for institutional-grade alpha opportunities
                "Conduct systematic multi-asset class market surveillance. Identify anomalous price movements, unusual volume patterns, emerging momentum trends, and potential regime shifts across equities, fixed income, commodities, and currencies that may represent exploitable alpha opportunities.",
                
                "Apply advanced technical analysis and quantitative signal processing. Evaluate momentum divergences, volatility regime changes, statistical arbitrage opportunities, and machine learning-based pattern recognition for predictive trading edges.",
                
                "Perform macroeconomic regime analysis and impact assessment. Evaluate central bank policy implications, economic data surprises, geopolitical risk developments, and their cascading effects on asset valuations and market correlations.",
                
                "Execute cross-sectional relative strength analysis. Compare sector performance, style factor exposures, geographic market leadership, and identify convergence/divergence patterns that signal potential alpha generation opportunities.",
                
                "Monitor institutional order flow and positioning. Analyze unusual accumulation/distribution patterns, large trader activity, and positioning data that may indicate informed trading or potential market impact events.",
                
                "Implement statistical modeling and anomaly detection. Apply time-series analysis, cointegration testing, and outlier identification to uncover non-obvious trading opportunities with quantifiable edge."
            ],
            
            'multi_strategy_opportunity_synthesis': [
                # Advanced cross-agent validation and signal integration
                "Perform comprehensive inter-methodological signal validation. Compare and contrast findings from technical, fundamental, quantitative, and sentiment-based approaches to identify reinforcing signals and potential methodological biases.",
                
                "Conduct multi-dimensional data source triangulation. Cross-reference signals across Bloomberg, Reuters, proprietary datasets, and alternative data sources to establish signal robustness and eliminate false positives.",
                
                "Synthesize unified investment theses from diverse analytical perspectives. Combine macro regime analysis with micro opportunity identification to construct coherent, well-supported trading hypotheses.",
                
                "Implement conviction-weighted opportunity prioritization. Rank potential trades by expected alpha capture probability, considering signal strength, historical precedent validation, and risk-adjusted return potential.",
                
                "Optimize trade structuring for execution efficiency. Develop position sizing frameworks that account for slippage costs, market impact, timing considerations, and portfolio-level risk constraints.",
                
                "Establish comprehensive knowledge base documentation. Record collaborative insights, successful analytical patterns, and decision frameworks for continuous learning and process improvement."
            ],
            
            'quantitative_opportunity_validation': [
                # Rigorous opportunity validation and quantification framework
                "Execute thorough opportunity validation against historical analogs. Compare current market conditions with precedent events, evaluating outcome distributions and conditional probabilities of success.",
                
                "Perform comprehensive risk decomposition and stress testing. Analyze downside volatility, tail risk exposures, correlation stress scenarios, and maximum drawdown potential under various market conditions.",
                
                "Quantify expected value and risk-adjusted metrics. Calculate Sharpe ratios, Sortino ratios, maximum drawdown expectations, and probabilistic return distributions for each opportunity.",
                
                "Develop institutional-grade execution frameworks. Establish precise entry/exit criteria, position scaling protocols, risk management overlays, and contingency planning for adverse market movements.",
                
                "Validate market timing and liquidity considerations. Assess current market microstructure, trading volumes, bid-ask spreads, and execution feasibility given prevailing market conditions.",
                
                "Create comprehensive opportunity documentation packages. Develop detailed trade theses, risk memos, execution playbooks, and monitoring protocols for institutional review and approval."
            ],
            
            'investment_committee_review': [
                # Efficient consensus formation with structured decision frameworks
                "Apply multi-criteria opportunity evaluation frameworks. Score opportunities against established investment criteria including alpha potential, execution feasibility, risk-adjusted returns, and portfolio fit.",
                
                "Assess market regime alignment and timing optimization. Evaluate whether identified opportunities are congruent with current market cycles, volatility regimes, and macroeconomic conditions.",
                
                "Implement cross-framework signal validation protocols. Ensure opportunities receive support from technical, fundamental, quantitative, and risk management perspectives before advancement.",
                
                "Refine trade construction and risk management parameters. Optimize entry/exit levels, position sizing algorithms, stop-loss mechanisms, and profit-taking protocols based on consensus feedback.",
                
                "Validate operational readiness and infrastructure capabilities. Confirm broker connectivity, market access, data feeds, and execution systems are fully prepared for trade implementation.",
                
                "Document consensus-driven decision rationale. Record the analytical framework, participant inputs, and decision criteria that led to final opportunity selection and prioritization."
            ],
            
            'portfolio_implementation_planning': [
                # Professional execution preparation and risk management
                "Calculate optimal capital allocation and position sizing. Determine appropriate position sizes considering portfolio beta, risk limits, liquidity constraints, and expected volatility-adjusted returns.",
                
                "Establish comprehensive risk management protocols. Define multi-layer stop-loss mechanisms, position limit thresholds, volatility-based adjustments, and catastrophic risk controls.",
                
                "Determine precise execution timing and methodology optimization. Select optimal trading algorithms, time horizons, and execution strategies to minimize market impact and transaction costs.",
                
                "Develop detailed execution playbooks with contingency planning. Create step-by-step execution protocols, communication procedures, and response frameworks for various market scenarios.",
                
                "Validate complete execution infrastructure readiness. Confirm API connectivity, order routing systems, market data feeds, and monitoring dashboards are fully operational.",
                
                "Conduct comprehensive pre-execution review and approval. Perform final validation of all trade parameters, risk controls, and market conditions before deployment authorization."
            ],
            
            'performance_analytics_and_refinement': [
                # Systematic performance analysis and continuous improvement
                "Evaluate analytical process effectiveness and outcome quality. Assess signal generation accuracy, opportunity identification success rates, and overall alpha discovery framework performance.",
                
                "Analyze collaborative intelligence integration quality. Evaluate how effectively diverse analytical perspectives were synthesized and whether consensus formation improved decision quality.",
                
                "Review predictive accuracy and outcome validation. Compare identified opportunities against actual market developments, tracking hit rates, false positive rates, and learning opportunities.",
                
                "Document key insights and process optimization opportunities. Record successful patterns, methodological improvements, and framework enhancements identified during the analysis cycle.",
                
                "Develop recommendations for enhanced alpha discovery capabilities. Identify technology upgrades, data source expansions, analytical methodology improvements, and process optimizations.",
                
                "Update analytical frameworks and decision criteria databases. Incorporate validated learnings into future analysis protocols, risk models, and investment decision frameworks."
            ],
            
            'chief_investment_officer_oversight': [
                # Executive oversight and final investment decision authority
                "Conduct comprehensive opportunity portfolio review. Evaluate the complete set of identified opportunities against investment objectives, risk tolerance, and portfolio construction requirements.",
                
                "Assess aggregate portfolio impact and risk implications. Analyze diversification benefits, correlation effects, liquidity considerations, and overall risk-adjusted portfolio enhancement potential.",
                
                "Make definitive execution decision: EXECUTE, HOLD, or RESTART. Provide detailed rationale considering opportunity quality, market conditions, risk assessments, and strategic alignment.",
                
                "Specify exact trade execution parameters if proceeding. Define precise position sizes, entry/exit levels, risk management protocols, and monitoring requirements for approved opportunities.",
                
                "Establish comprehensive monitoring and adjustment frameworks. Define performance tracking metrics, risk threshold triggers, and protocols for position management and exit strategies.",
                
                "Document executive decision framework and oversight process. Record the evaluation criteria, risk assessments, and strategic considerations that informed the final investment decision."
            ],
            'pre_market_readiness_assessment': [
                # Ultra-fast market readiness validation at open
                "Verify market opening status and session initialization. Confirm exchange systems are operational, market data feeds are streaming, and trading sessions have commenced across relevant venues.",
                
                "Validate pre-market analysis relevance and persistence. Assess whether identified opportunities remain viable given overnight developments, economic data releases, or geopolitical events.",
                
                "Evaluate immediate market conditions and volatility assessment. Analyze opening price action, initial volatility levels, bid-ask spreads, and early trading patterns that may affect execution.",
                
                "Confirm execution infrastructure operational readiness. Validate broker API connectivity, order routing systems, market data quality, and all execution platforms are fully functional.",
                
                "Review risk management parameters for market open conditions. Ensure position limits, stop-loss levels, and risk controls are appropriately calibrated for opening volatility and liquidity conditions.",
                
                "Provide definitive go/no-go execution recommendation. Determine whether current market conditions support immediate execution of pre-approved trading strategies or require holding/reassessment."
            ],
            
            'opening_bell_execution': [
                # Precision trade execution at market open
                "Execute pre-approved trading strategies at optimal market open timing. Place orders for validated opportunities utilizing predetermined execution algorithms and timing protocols.",
                
                "Monitor real-time execution quality and market impact assessment. Track fill rates, price improvement, slippage metrics, and execution costs during the volatile market opening period.",
                
                "Confirm successful position establishment and order completion. Verify all trades have executed according to plan, positions are correctly established, and confirmations are received.",
                
                "Document comprehensive execution details and performance metrics. Record exact execution prices, timestamps, slippage costs, and any deviations from planned execution parameters.",
                
                "Assess initial market reaction and price action response. Monitor post-execution price behavior, volatility response, and early performance indicators for executed positions.",
                
                "Transition seamlessly to active position monitoring protocols. Initialize real-time tracking systems, alert thresholds, and risk management monitoring for established positions."
            ],
            
            'position_surveillance_initialization': [
                # Comprehensive position monitoring infrastructure establishment
                "Configure multi-dimensional position monitoring parameters. Establish P&L tracking, risk metric calculations, performance attribution, and real-time position valuation systems.",
                
                "Initialize high-frequency market data streaming infrastructure. Establish continuous price feeds, volume data, order book depth, and market microstructure monitoring for active positions.",
                
                "Implement intelligent automated alerting systems. Configure threshold-based notifications for P&L deviations, risk limit breaches, volatility spikes, and predefined exit triggers.",
                
                "Establish quantitative performance tracking frameworks. Set up metrics for tracking execution quality, slippage analysis, market impact assessment, and risk-adjusted performance measurement.",
                
                "Define structured monitoring schedule and escalation protocols. Establish monitoring frequency parameters, communication hierarchies, and decision-making frameworks for different market conditions.",
                
                "Validate complete monitoring ecosystem operational readiness. Confirm all data feeds, calculation engines, alerting systems, and reporting dashboards are fully functional and calibrated."
            ],
            
            'active_position_management': [
                # Continuous real-time position surveillance and risk management
                "Monitor position performance metrics in real-time. Track P&L development, execution slippage, market impact costs, and performance against predefined expectations and benchmarks.",
                
                "Assess dynamic market condition evolution continuously. Evaluate volatility regime changes, liquidity fluctuations, correlation shifts, and broader market impacts on active positions.",
                
                "Maintain active risk parameter surveillance and adjustment. Monitor position sizes against current volatility, correlation changes, and ensure risk limits remain within acceptable thresholds.",
                
                "Identify profit-taking opportunities and exit signals. Analyze price targets achievement, momentum exhaustion patterns, and deteriorating trade setups requiring position reduction or closure.",
                
                "Track execution quality and market microstructure continuously. Monitor for adverse selection, price manipulation concerns, and execution inefficiencies requiring strategy adjustment.",
                
                "Maintain comprehensive position status documentation. Record performance metrics, market condition assessments, risk parameter status, and position management rationale throughout the holding period."
            ],
            
            'dynamic_portfolio_adjustment': [
                # Data-driven position management and exit strategy formulation
                "Evaluate position performance against original investment theses. Assess whether trades are performing according to expected risk-reward profiles and market condition assumptions.",
                
                "Develop position adjustment and exit strategies dynamically. Determine whether positions should be maintained, scaled, or closed based on current performance, risk metrics, and market developments.",
                
                "Assess risk management requirement evolution. Evaluate whether stop-loss levels need tightening, position limits require adjustment, or additional hedging strategies are necessary.",
                
                "Review profit target achievement and scaling opportunities. Determine whether profit-taking thresholds have been reached or if scaling out of positions is appropriate given current gains.",
                
                "Evaluate market regime change implications for positions. Assess whether broader market condition shifts warrant position adjustments, closures, or strategic repositioning.",
                
                "Formulate clear position management recommendations. Provide specific guidance on trade adjustments, risk controls, and next steps based on comprehensive monitoring analysis."
            ],
            
            'execution_of_portfolio_changes': [
                # Precision execution of monitoring-based position adjustments
                "Implement approved position adjustment strategies systematically. Execute predetermined position scaling, stop-loss modifications, or closure orders according to monitoring-based decisions.",
                
                "Close positions according to established exit criteria. Execute exit orders for positions meeting profit targets, stop-loss triggers, or deteriorating fundamental setups.",
                
                "Adjust risk management parameters dynamically. Modify stop-loss levels, position size limits, or implement additional risk mitigation strategies based on current market conditions.",
                
                "Document all position adjustments with comprehensive audit trail. Record execution details, price levels, timestamps, and decision rationale for all position management actions.",
                
                "Verify successful execution and position status confirmation. Confirm all adjustments executed correctly, positions reflect intended changes, and all orders processed successfully.",
                
                "Update monitoring parameters and thresholds post-adjustment. Recalibrate alert levels, risk limits, and monitoring criteria based on executed position changes and current portfolio structure."
            ]
        }

    def setup_scheduler(self):
        """Setup scheduled tasks for premarket and other workflows."""
        # Schedule pre-market prep at 6:00 AM ET, Mon-Fri
        self.scheduler.add_job(
            self.run_premarket_analysis,
            CronTrigger(hour=6, minute=0, day_of_week='mon-fri', timezone='America/New_York')
        )
        # Add more schedules as needed, e.g., midday check
        self.scheduler.add_job(
            self.run_midday_check,
            CronTrigger(hour=12, minute=0, day_of_week='mon-fri', timezone='America/New_York')
        )
        self.scheduler.start()
        print("📅 Scheduler started with premarket and scheduled tasks.")

    def schedule_iterative_reasoning(self, interval_hours: int = 4, trigger_time: Optional[str] = None):
        """
        Schedule iterative reasoning workflow execution.

        Args:
            interval_hours: Hours between executions (default: 4)
            trigger_time: Specific time in HH:MM format (optional, overrides interval)
        """
        if trigger_time:
            # Parse trigger time (e.g., "09:30")
            hour, minute = map(int, trigger_time.split(':'))
            trigger = CronTrigger(hour=hour, minute=minute, day_of_week='mon-fri', timezone='America/New_York')
            job_name = f"iterative_reasoning_{trigger_time}"
        else:
            # Use interval-based scheduling
            trigger = CronTrigger(hour=f"*/{interval_hours}", day_of_week='mon-fri', timezone='America/New_York')
            job_name = f"iterative_reasoning_every_{interval_hours}h"

        self.scheduler.add_job(
            self._scheduled_iterative_reasoning,
            trigger,
            id=job_name,
            name=f"Scheduled Iterative Reasoning ({trigger_time or f'every {interval_hours}h'})",
            replace_existing=True
        )
        logger.info(f"📅 Scheduled iterative reasoning: {job_name}")

    def schedule_continuous_workflow(self, interval_minutes: int = 60):
        """
        Schedule continuous alpha discovery workflow.

        Args:
            interval_minutes: Minutes between executions (default: 60)
        """
        trigger = CronTrigger(minute=f"*/{interval_minutes}", day_of_week='mon-fri', timezone='America/New_York')
        job_name = f"continuous_workflow_every_{interval_minutes}min"

        self.scheduler.add_job(
            self._scheduled_continuous_workflow,
            trigger,
            id=job_name,
            name=f"Scheduled Continuous Workflow (every {interval_minutes}min)",
            replace_existing=True
        )
        logger.info(f"📅 Scheduled continuous workflow: {job_name}")

    def schedule_health_check(self, interval_minutes: int = 30):
        """
        Schedule periodic health checks.

        Args:
            interval_minutes: Minutes between health checks (default: 30)
        """
        trigger = CronTrigger(minute=f"*/{interval_minutes}", timezone='America/New_York')
        job_name = f"health_check_every_{interval_minutes}min"

        self.scheduler.add_job(
            self._scheduled_health_check,
            trigger,
            id=job_name,
            name=f"Scheduled Health Check (every {interval_minutes}min)",
            replace_existing=True
        )
        logger.info(f"📅 Scheduled health check: {job_name}")

    async def _scheduled_iterative_reasoning(self):
        """Internal method for scheduled iterative reasoning execution."""
        try:
            # Check system health before running
            if not await self._check_health_before_schedule():
                logger.warning("⚠️ Skipping scheduled iterative reasoning due to health check failure")
                return

            logger.info("🚀 Starting scheduled iterative reasoning workflow")
            await self.start_workflow()
        except Exception as e:
            logger.error(f"❌ Scheduled iterative reasoning failed: {e}")
            await self.alert_manager.error(f"Scheduled workflow failed: {e}")

    async def _scheduled_continuous_workflow(self):
        """Internal method for scheduled continuous workflow execution."""
        try:
            # Check system health before running
            if not await self._check_health_before_schedule():
                logger.warning("⚠️ Skipping scheduled continuous workflow due to health check failure")
                return

            logger.info("🔄 Starting scheduled continuous alpha discovery")
            # Implement continuous workflow logic here
            await self.execute_phase('systematic_market_surveillance', "🔍 CONTINUOUS MARKET SURVEILLANCE")
        except Exception as e:
            logger.error(f"❌ Scheduled continuous workflow failed: {e}")
            await self.alert_manager.error(f"Scheduled continuous workflow failed: {e}")

    async def _scheduled_health_check(self):
        """Internal method for scheduled health check execution."""
        try:
            logger.info("🔍 Running scheduled health check")
            await self.perform_system_health_check()
        except Exception as e:
            logger.error(f"❌ Scheduled health check failed: {e}")
            await self.alert_manager.error(f"Scheduled health check failed: {e}")

    async def _check_health_before_schedule(self) -> bool:
        """
        Perform quick health check before running scheduled jobs.
        Returns True if system is healthy enough to run workflows.
        """
        try:
            # Quick Redis check
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            r.ping()

            # Check for critical alerts
            critical_count = len([alert for alert in self.alert_manager.error_queue if alert['level'] == 'critical'])
            if critical_count >= 5:
                logger.warning(f"⚠️ High critical alert count ({critical_count}), pausing scheduled workflows")
                return False

            return True
        except Exception as e:
            logger.error(f"❌ Health check failed before scheduled job: {e}")
            return False

    async def run_premarket_analysis(self):
        """Run premarket analysis workflow."""
        if self.workflow_active:
            print("⚠️ Workflow already active, skipping premarket.")
            return
        print("🚀 Starting premarket analysis.")
        # Implement premarket specific phases
        await self.execute_phase('pre_market_readiness_assessment', "🔍 PRE-MARKET READINESS ASSESSMENT")
        # Add more premarket phases as needed

    async def run_midday_check(self):
        """Run midday check workflow."""
        print("🕛 Starting midday check.")
        # Implement midday specific logic
        await self.execute_phase('active_position_management', "📊 MIDDAY POSITION CHECK")

    async def perform_system_health_check(self):
        """Perform comprehensive system health check before starting workflows."""
        print("🔍 Performing system health check...")
        if self.channel and hasattr(self.channel, 'send'):
            await self.channel.send("🔍 **System Health Check Starting...**")

        health_issues = []
        health_warnings = []
        total_checks = 0
        passed_checks = 0

        # Check Redis connection
        total_checks += 1
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            r.ping()
            print("✅ Redis connection: OK")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("✅ Redis connection: OK")
            passed_checks += 1
        except Exception as e:
            health_issues.append(f"Redis connection failed: {e}")

        # Check TigerBeetle connection
        total_checks += 1
        try:
            # Suppress warnings during TigerBeetle client creation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import tigerbeetle as tb
                client = tb.ClientSync(cluster_id=0, replica_addresses="127.0.0.1:3000")
            # Just test client creation - don't query to avoid hanging
            print("✅ TigerBeetle client creation: OK")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("✅ TigerBeetle client creation: OK")
            client.close()
            passed_checks += 1
        except Exception as e:
            health_issues.append(f"TigerBeetle connection failed: {e}")

        # Check LangChain imports
        total_checks += 1
        try:
            import langchain
            import langchain_community
            import langchain_core
            print("✅ LangChain imports: OK")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("✅ LangChain imports: OK")
            passed_checks += 1
        except ImportError as e:
            health_warnings.append(f"LangChain import warning: {e}")

        # Check IBKR connection (actual connection test via bridge)
        total_checks += 1
        try:
            from src.integrations.ibkr_connector import get_ibkr_connector
            connector = get_ibkr_connector()
            # Test connection by attempting to get positions with timeout
            positions = await asyncio.wait_for(connector.get_positions(), timeout=5.0)
            print("✅ IBKR connection: OK")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("✅ IBKR connection: OK")
            passed_checks += 1
        except asyncio.TimeoutError:
            health_warnings.append("IBKR connection timeout")
        except Exception as e:
            health_warnings.append(f"IBKR connection failed: {e}")

        # Check MarketDataApp API
        total_checks += 1
        try:
            from src.utils.vault_client import get_vault_secret
            api_key = get_vault_secret('MARKETDATAAPP_API_KEY')
            if api_key:
                print("✅ MarketDataApp API key available")
                if self.channel and hasattr(self.channel, 'send'):
                    general_channel = cast(discord.TextChannel, self.channel)
                    await general_channel.send("✅ MarketDataApp API key available")
                passed_checks += 1
            else:
                health_warnings.append("MarketDataApp API key not configured")
        except Exception as e:
            health_warnings.append(f"MarketDataApp API check failed: {e}")

        # Check FinanceDatabase
        total_checks += 1
        try:
            import financedatabase as fd
            print("✅ FinanceDatabase import: OK")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("✅ FinanceDatabase import: OK")
            passed_checks += 1
        except ImportError as e:
            health_warnings.append(f"FinanceDatabase import failed: {e}")

        # Check Stable-Baselines3
        total_checks += 1
        try:
            import stable_baselines3
            print("✅ Stable-Baselines3 import: OK")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("✅ Stable-Baselines3 import: OK")
            passed_checks += 1
        except ImportError as e:
            health_warnings.append(f"Stable-Baselines3 import failed: {e}")

        # Check scikit-learn
        total_checks += 1
        try:
            import sklearn
            print("✅ scikit-learn import: OK")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("✅ scikit-learn import: OK")
            passed_checks += 1
        except ImportError as e:
            health_warnings.append(f"scikit-learn import failed: {e}")

        # Check TensorFlow
        total_checks += 1
        try:
            import tensorflow as tf
            print("✅ TensorFlow import: OK")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("✅ TensorFlow import: OK")
            passed_checks += 1
        except ImportError as e:
            health_warnings.append(f"TensorFlow import failed: {e}")

        # Check yfinance
        total_checks += 1
        try:
            import yfinance as yf
            print("✅ yfinance import: OK")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("✅ yfinance import: OK")
            passed_checks += 1
        except ImportError as e:
            health_warnings.append(f"yfinance import failed: {e}")

        # Check FRED API
        total_checks += 1
        try:
            import fredapi
            print("✅ FRED API import: OK")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("✅ FRED API import: OK")
            passed_checks += 1
        except ImportError as e:
            health_warnings.append(f"FRED API import failed: {e}")

        # Check tools count
        tool_count = 0
        try:
            from src.utils.tools import get_available_tools
            tools = get_available_tools()
            tool_count = len(tools)
            print(f"🛠️ Tools available: {tool_count}")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send(f"🛠️ Tools available: {tool_count}")
            if tool_count < 10:  # Arbitrary minimum
                health_warnings.append(f"Low tool count: {tool_count} tools available")
        except Exception as e:
            health_issues.append(f"Tools check failed: {e}")

        # Check AlertManager error queue
        total_checks += 1
        try:
            alert_health = alert_manager.check_health()
            recent_critical = alert_health.get('recent_critical_alerts', 0)
            recent_errors = alert_health.get('recent_error_alerts', 0)

            if recent_critical > 5:
                health_issues.append(f"High critical alert count: {recent_critical} in last 50 alerts")
                # Auto-pause workflow
                self.workflow_active = False
                print("🚨 Auto-pausing workflow due to high alert count")
            elif recent_errors > 10:
                health_warnings.append(f"High error alert count: {recent_errors} in last 50 alerts")
            else:
                print("✅ Alert system: OK")
                passed_checks += 1
        except Exception as e:
            health_warnings.append(f"Alert system check failed: {e}")

        # Report connection stability
        stability_ratio = f"{passed_checks}/{total_checks}"
        alert_status = "ALERTING" if any("High critical alert count" in issue for issue in health_issues) else "NORMAL"

        if passed_checks == total_checks and alert_status == "NORMAL":
            print(f"🎯 System Stability: {stability_ratio} dependencies stable - FULLY OPERATIONAL")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send(f"🎯 **System Stability: {stability_ratio} dependencies stable - FULLY OPERATIONAL**")
        elif passed_checks >= total_checks * 0.8 and alert_status != "ALERTING":
            print(f"⚠️ System Stability: {stability_ratio} dependencies stable - MOSTLY OPERATIONAL")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send(f"⚠️ **System Stability: {stability_ratio} dependencies stable - MOSTLY OPERATIONAL**")
        else:
            status_msg = f"❌ System Stability: {stability_ratio} dependencies stable - LIMITED OPERATIONALITY"
            if alert_status == "ALERTING":
                status_msg += " (ALERTING - Workflow Auto-Paused)"
            print(status_msg)
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send(f"**{status_msg}**")

        # Report results
        if health_issues:
            print("❌ Critical health issues found:")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("❌ **Critical health issues found:**")
                for issue in health_issues:
                    await general_channel.send(f"  - {issue}")
            for issue in health_issues:
                print(f"  - {issue}")
        else:
            print("✅ All critical dependencies OK")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("✅ **All critical dependencies OK**")

        if health_warnings:
            print("⚠️ Health warnings:")
            if self.channel and hasattr(self.channel, 'send'):
                general_channel = cast(discord.TextChannel, self.channel)
                await general_channel.send("⚠️ **Health warnings:**")
                for warning in health_warnings:
                    await general_channel.send(f"  - {warning}")
            for warning in health_warnings:
                print(f"  - {warning}")

        # Post system event for health check completion
        total_checks = 11  # Based on health check implementation
        passed_checks = total_checks - len(health_issues) - len(health_warnings)
        health_status = "success" if passed_checks == total_checks else "warning" if len(health_issues) == 0 else "error"

        await self._post_system_event(
            health_status,
            f"System Health Check Completed ({passed_checks}/{total_checks} systems operational)",
            f"Issues: {len(health_issues)}, Warnings: {len(health_warnings)}",
            {
                "stability_ratio": f"{passed_checks}/{total_checks}",
                "critical_issues": len(health_issues),
                "warnings": len(health_warnings),
                "operational_status": "FULLY OPERATIONAL" if passed_checks == total_checks else "DEGRADED" if len(health_issues) == 0 else "CRITICAL"
            }
        )

        return health_issues, health_warnings

    async def recheck_errors(self):
        """
        On-demand error queue checking for manual health verification.
        Returns alert health status and triggers notifications if needed.
        """
        try:
            alert_health = self.alert_manager.check_health()
            critical_count = alert_health.get('recent_critical_alerts', 0)
            error_count = alert_health.get('recent_error_alerts', 0)

            if critical_count > 5:
                self.workflow_active = False
                self.alert_manager.critical(
                    Exception(f"System auto-paused due to {critical_count} critical alerts in queue"),
                    {
                        "action": "workflow_paused",
                        "critical_alerts": critical_count,
                        "total_errors": error_count
                    },
                    "orchestrator"
                )

            return {
                "alert_status": "ALERTING" if critical_count > 0 else "NORMAL",
                "critical_count": critical_count,
                "error_count": error_count,
                "workflow_active": self.workflow_active
            }

        except Exception as e:
            self.alert_manager.critical(
                Exception(f"Failed to recheck error queue: {str(e)}"),
                {"error_type": type(e).__name__},
                "orchestrator"
            )
            return {
                "alert_status": "ERROR",
                "critical_count": 0,
                "error_count": 0,
                "workflow_active": self.workflow_active
            }

    async def send_health_status(self, message):
        """Send system health status as a Discord embed."""
        try:
            # Perform health check
            health_issues, health_warnings = await self.perform_system_health_check()

            # Calculate stability metrics
            total_checks = 11  # Based on the health check method
            passed_checks = total_checks - len(health_issues) - len(health_warnings)
            stability_ratio = f"{passed_checks}/{total_checks}"

            # Determine status color and message
            if passed_checks == total_checks:
                color = 0x00FF00  # Green
                status_msg = "🎯 FULLY OPERATIONAL"
            elif passed_checks >= total_checks * 0.8:
                color = 0xFFFF00  # Yellow
                status_msg = "⚠️ MOSTLY OPERATIONAL"
            else:
                color = 0xFF0000  # Red
                status_msg = "❌ LIMITED OPERATIONALITY"

            # Create embed
            embed = discord.Embed(
                title="🔍 System Health Status",
                description=f"**Stability:** {stability_ratio} dependencies stable\n**Status:** {status_msg}",
                color=color,
                timestamp=datetime.now()
            )

            # Add critical issues field
            if health_issues:
                issues_text = "\n".join(f"❌ {issue}" for issue in health_issues[:5])  # Limit to 5
                if len(health_issues) > 5:
                    issues_text += f"\n... and {len(health_issues) - 5} more"
                embed.add_field(name="Critical Issues", value=issues_text, inline=False)
            else:
                embed.add_field(name="Critical Issues", value="✅ None found", inline=False)

            # Add warnings field
            if health_warnings:
                warnings_text = "\n".join(f"⚠️ {warning}" for warning in health_warnings[:5])  # Limit to 5
                if len(health_warnings) > 5:
                    warnings_text += f"\n... and {len(health_warnings) - 5} more"
                embed.add_field(name="Warnings", value=warnings_text, inline=False)
            else:
                embed.add_field(name="Warnings", value="✅ None found", inline=False)

            # Add workflow status
            workflow_status = "🟢 Active" if self.workflow_active else "🔴 Inactive"
            monitoring_status = "🟢 Active" if getattr(self, 'monitoring_active', False) else "🔴 Inactive"
            embed.add_field(
                name="Workflow Status",
                value=f"**Main Workflow:** {workflow_status}\n**Trade Monitoring:** {monitoring_status}",
                inline=True
            )

            # Add agent count
            agent_count = len(self.agents) if hasattr(self, 'agents') else 0
            embed.add_field(
                name="Agents",
                value=f"**Active Agents:** {agent_count}",
                inline=True
            )

            # Add timestamp
            embed.set_footer(text="ABC Application Health Check")

            await message.channel.send(embed=embed)

        except Exception as e:
            error_embed = discord.Embed(
                title="❌ Health Check Error",
                description=f"Failed to perform health check: {str(e)}",
                color=0xFF0000
            )
            await message.channel.send(embed=error_embed)

    async def initialize_discord_client(self):
        """Initialize Discord client for live orchestration"""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        self.client = discord.Client(intents=intents)
        self.tree = discord.app_commands.CommandTree(self.client)

        # Slash Commands Setup
        @self.tree.command(name="consensus_status", description="Show current consensus poll status")
        async def consensus_status_slash(interaction: discord.Interaction):
            """Slash command version of consensus status"""
            try:
                active_polls = self.consensus_poller.get_active_polls()
                completed_polls = self.consensus_poller.get_completed_polls(limit=5)

                if not active_polls and not completed_polls:
                    await interaction.response.send_message("🤝 **No consensus polls found**")
                    return

                embed = discord.Embed(
                    title="🤝 Consensus Poll Status",
                    color=0x3498DB,
                    timestamp=datetime.now()
                )

                if active_polls:
                    active_list = []
                    for poll in active_polls[:3]:  # Show up to 3 active polls
                        status = f"**{poll.state.value.replace('_', ' ').title()}**"
                        votes = sum(1 for v in poll.votes.values() if v.get("status") == "responded")
                        total = len(poll.votes)
                        active_list.append(f"• {poll.question[:50]}... ({votes}/{total} votes) - {status}")
                    embed.add_field(name="Active Polls", value="\n".join(active_list), inline=False)

                if completed_polls:
                    completed_list = []
                    for poll in completed_polls:
                        result = "✅" if poll.state.value == "consensus_reached" else "⏰" if poll.state.value == "timeout" else "❌"
                        completed_list.append(f"{result} {poll.question[:40]}...")
                    embed.add_field(name="Recent Results", value="\n".join(completed_list), inline=False)

                await interaction.response.send_message(embed=embed)

            except Exception as e:
                logger.error(f"Error in consensus_status slash command: {e}")
                await interaction.response.send_message(f"❌ **Error:** {str(e)}")

        @self.tree.command(name="poll_consensus", description="Start a new consensus poll")
        @discord.app_commands.describe(
            question="The question to ask agents",
            agents="Comma-separated list of agents to poll (e.g., risk_agent,strategy_agent)"
        )
        async def poll_consensus_slash(interaction: discord.Interaction, question: str, agents: str):
            """Slash command version of poll consensus"""
            try:
                # Parse agents
                agents_to_poll = [agent.strip() for agent in agents.split(",") if agent.strip()]

                if len(agents_to_poll) < 2:
                    await interaction.response.send_message("❌ **Need at least 2 agents to poll**")
                    return

                # Validate agents exist
                valid_agents = ["risk_agent", "strategy_agent", "data_agent", "execution_agent", "reflection_agent"]
                invalid_agents = [agent for agent in agents_to_poll if agent not in valid_agents]

                if invalid_agents:
                    await interaction.response.send_message(f"❌ **Invalid agents:** {', '.join(invalid_agents)}\n**Valid agents:** {', '.join(valid_agents)}")
                    return

                # Create the poll
                poll_id = await self.consensus_poller.create_poll(question, agents_to_poll)

                # Start the poll
                success = await self.consensus_poller.start_poll(poll_id)

                if success:
                    embed = discord.Embed(
                        title="🤝 Consensus Poll Started",
                        description=f"**Question:** {question}",
                        color=0x00ff00,
                        timestamp=datetime.now()
                    )
                    embed.add_field(name="Poll ID", value=poll_id, inline=True)
                    embed.add_field(name="Agents Polled", value=", ".join(agents_to_poll), inline=True)
                    embed.add_field(name="Timeout", value="5 minutes", inline=True)

                    await interaction.response.send_message(embed=embed)
                else:
                    await interaction.response.send_message(f"❌ **Failed to start consensus poll**")

            except Exception as e:
                logger.error(f"Error in poll_consensus slash command: {e}")
                await interaction.response.send_message(f"❌ **Poll consensus failed:** {str(e)}")

        @self.tree.command(name="commands", description="Show all available Discord commands")
        async def commands_slash(interaction: discord.Interaction):
            """Slash command to display all available commands"""
            try:
                from src.utils.command_registry import get_command_registry
                registry = get_command_registry()
                embed_data = registry.generate_discord_embed_data()

                embed = discord.Embed(
                    title=embed_data["title"],
                    description=embed_data["description"],
                    color=0x3498DB,
                    timestamp=datetime.now()
                )

                for field in embed_data["fields"]:
                    embed.add_field(
                        name=field["name"],
                        value=field["value"],
                        inline=field["inline"]
                    )

                embed.set_footer(text="💡 Use /commands for slash command version | Visit #commands for detailed docs")

                await interaction.response.send_message(embed=embed)

            except Exception as e:
                logger.error(f"Error in commands slash command: {e}")
                await interaction.response.send_message(f"❌ **Error displaying commands:** {str(e)}")

        @self.client.event
        async def on_ready():
            if not self.client or not self.client.user:
                print("❌ Client not properly initialized")
                return

            print(f"🎯 Live Workflow Orchestrator connected as {self.client.user}")
            print("🤖 Ready to orchestrate iterative reasoning workflow!")
            print("💡 You can ask questions or intervene at any time during the process.")

            # Find the target channels
            guild_id = int(os.getenv('DISCORD_GUILD_ID', '0'))
            if guild_id and self.client:
                guild = self.client.get_guild(guild_id)
                if guild:
                    # Set up general channel for summaries
                    general_channel_id = os.getenv('DISCORD_GENERAL_CHANNEL_ID')
                    if general_channel_id:
                        try:
                            self.channel = guild.get_channel(int(general_channel_id))
                            if self.channel:
                                print(f"📝 General channel configured: #{self.channel.name}")
                            else:
                                print(f"⚠️ General channel ID {general_channel_id} not found, using fallback")
                        except ValueError:
                            print(f"⚠️ Invalid general channel ID: {general_channel_id}")
                    
                    if not self.channel:
                        # Fallback to finding by name
                        for ch in guild.text_channels:
                            if ch.name in ['general', 'workflow', 'analysis', 'trading']:
                                self.channel = ch
                                print(f"📝 General channel (fallback): #{ch.name}")
                                break

                    # Set up specialized channels
                    await self._setup_discord_channels(guild)

                    if not self.channel and guild.text_channels:
                        self.channel = guild.text_channels[0]
                        print(f"📝 Using default general channel: #{self.channel.name}")

                    # Initialize agents after Discord connection
                    if not hasattr(self, 'agents_initialized') or not self.agents_initialized:
                        print("🤖 Initializing agent instances...")
                        try:
                            await self.initialize_agents_async()
                            print(f"✅ Agent initialization complete: {len(self.agent_instances)} agents ready")
                            self.agents_initialized = True
                        except Exception as e:
                            alert_manager.critical(e, {"component": "agent_initialization"}, "orchestrator")

                    # Perform system health check
                    health_issues, health_warnings = await self.perform_system_health_check()

                    # Report health issues to Discord if critical
                    if isinstance(self.channel, discord.TextChannel):
                        if health_issues:
                            issues_text = "\n".join(f"• {issue}" for issue in health_issues)
                            await self.channel.send(f"❌ **Critical Health Issues Detected:**\n{issues_text}\n\n⚠️ System may not function properly. Please resolve these issues before starting workflows.")
                        if health_warnings:
                            warnings_text = "\n".join(f"• {warning}" for warning in health_warnings)
                            await self.channel.send(f"⚠️ **Health Warnings:**\n{warnings_text}\n\n💡 Some features may be limited.")

                    # Announce orchestrator presence
                    if isinstance(self.channel, discord.TextChannel):
                        channel_status = "✅ Channel separation active" if self.alerts_channel else "⚠️ Alerts channel not configured"
                        ranked_trades_status = "✅ Ranked trades channel active" if self.ranked_trades_channel else "⚠️ Ranked trades channel not configured"
                        await self.channel.send("**Live Workflow Orchestrator Online**\nReady to begin iterative reasoning workflow with unified agent coordination. Type `!start_workflow` to begin analysis, `!start_premarket_analysis` for premarket prep, `!start_market_open_execution` for fast execution, or `!start_trade_monitoring` for position monitoring!")
                        await self.channel.send(f"📢 **Channel Configuration:** {channel_status}")
                        if self.alerts_channel and isinstance(self.alerts_channel, discord.TextChannel):
                            await self.channel.send(f"🚨 Trade alerts will be sent to: #{self.alerts_channel.name}")
                        if self.ranked_trades_channel and isinstance(self.ranked_trades_channel, discord.TextChannel):
                            await self.channel.send(f"📊 Ranked trade proposals will be sent to: #{self.ranked_trades_channel.name}")

                    # Signal that Discord is ready
                    self.discord_ready.set()
                    print("🎯 Discord client fully ready - channels configured and orchestrator online")

                    # Sync slash commands
                    try:
                        await self.tree.sync()
                        print("🔄 Slash commands synced successfully")
                    except Exception as e:
                        print(f"⚠️ Failed to sync slash commands: {e}")

                    self.setup_scheduler()  # Start scheduler after Discord is ready
                    
#                    # Automatically start the continuous workflow
#                    print("🚀 Auto-starting continuous alpha discovery workflow...")
#                    asyncio.create_task(self.start_workflow())

        @self.client.event
        async def on_message(message):
            if not self.client or not self.client.user:
                return

            # Don't respond to own messages
            if message.author == self.client.user:
                return

            # Handle workflow control commands (only in general channel)
            content = message.content.strip()

            if content == "!start_workflow" and not self.workflow_active:
                await self.start_workflow()
                return

            if content == "!start_premarket_analysis" and not self.workflow_active:
                await self.start_premarket_analysis_workflow()
                return

            if content == "!start_market_open_execution" and not self.workflow_active:
                await self.start_market_open_execution_workflow()
                return

            if content == "!start_trade_monitoring" and not getattr(self, 'monitoring_active', False):
                await self.start_trade_monitoring_workflow()
                return

            if content == "!pause_workflow" and self.workflow_active:
                await self.pause_workflow()
                return

            if content == "!resume_workflow":
                await self.resume_workflow()
                return

            if content == "!stop_workflow":
                await self.stop_workflow()
                return

            if content == "!stop_monitoring" and getattr(self, 'monitoring_active', False):
                await self.complete_monitoring_workflow()
                return

            if content == "!workflow_status":
                await self.send_status_update()
                return

            if content == "!status":
                await self.send_health_status(message)
                return

            if content == "!scheduler_status":
                await self.send_scheduler_status(message)
                return

            # Alert system commands
            if content == "!alert_test":
                await self.handle_alert_test_command(message)
                return

            if content == "!check_health_now":
                await self.handle_check_health_now_command(message)
                return

            if content == "!alert_history":
                await self.handle_alert_history_command(message)
                return

            if content == "!alert_stats":
                await self.handle_alert_stats_command(message)
                return

            if content == "!commands" or content == "!help":
                await self.handle_commands_command(message)
                return

            if content == "!consensus_status":
                await self.handle_consensus_status_command(message)
                return

            # Handle !poll_consensus command: !poll_consensus "Question?" agent1 agent2 agent3
            if content.startswith("!poll_consensus"):
                await self.handle_poll_consensus_command(message)
                return

            # Handle !schedule_workflow command: !schedule_workflow <type> <time>
            if content.startswith("!schedule_workflow"):
                await self.handle_schedule_workflow_command(message)
                return

            # Handle !share_news command: !share_news <link> [optional description]
            if content.startswith("!share_news"):
                await self.handle_share_news_command(message)
                return

            # Handle !set_commands_channel command: !set_commands_channel <channel_id>
            if content.startswith("!set_commands_channel"):
                await self.handle_set_commands_channel_command(message)
                return

            # Handle human questions/interventions during active workflow
            elif self.workflow_active and not message.author.bot:
                await self.handle_human_intervention(message)

        token = get_vault_secret('DISCORD_ORCHESTRATOR_TOKEN')
        if not token:
            raise ValueError("❌ DISCORD_ORCHESTRATOR_TOKEN not found. Please create a separate Discord bot for the orchestrator.")

    def _validate_url(self, url: str) -> bool:
        """
        Validate that a URL is safe to process (HTTP/HTTPS only).

        Args:
            url: The URL string to validate

        Returns:
            True if URL is valid and safe, False otherwise
        """
        try:
            parsed = urlparse(url)
            # Only allow HTTP and HTTPS schemes
            if parsed.scheme not in ('http', 'https'):
                return False
            # Must have a netloc (domain)
            if not parsed.netloc:
                return False
            # Block potentially dangerous domains
            dangerous_patterns = ['localhost', '127.0.0.1', '0.0.0.0', '[::1]', 'internal', '.local']
            netloc_lower = parsed.netloc.lower()
            if any(pattern in netloc_lower for pattern in dangerous_patterns):
                return False
            return True
        except Exception:
            return False

    async def _setup_discord_channels(self, guild):
        """Set up specialized Discord channels for alerts and trade proposals."""
        # Set up alerts channel for trade notifications
        alerts_channel_id = os.getenv('DISCORD_ALERTS_CHANNEL_ID')
        if alerts_channel_id:
            try:
                self.alerts_channel = guild.get_channel(int(alerts_channel_id))
                if self.alerts_channel:
                    print(f"🚨 Alerts channel configured: #{self.alerts_channel.name}")
                else:
                    print(f"⚠️ Alerts channel ID {alerts_channel_id} not found")
                    await self._send_channel_warning("Alerts channel not found. Trade alerts will use general channel.")
            except ValueError:
                print(f"⚠️ Invalid alerts channel ID: {alerts_channel_id}")
                await self._send_channel_warning("Invalid alerts channel ID. Trade alerts will use general channel.")
        else:
            print("⚠️ DISCORD_ALERTS_CHANNEL_ID not set, trade alerts will go to general channel")
            await self._send_channel_warning("DISCORD_ALERTS_CHANNEL_ID not set. Trade alerts will use general channel.")

        # Set up ranked trades channel for trade proposals
        ranked_trades_channel_id = os.getenv('DISCORD_RANKED_TRADES_CHANNEL_ID')
        if ranked_trades_channel_id:
            try:
                self.ranked_trades_channel = guild.get_channel(int(ranked_trades_channel_id))
                if self.ranked_trades_channel:
                    print(f"📊 Ranked trades channel configured: #{self.ranked_trades_channel.name}")
                else:
                    print(f"⚠️ Ranked trades channel ID {ranked_trades_channel_id} not found")
                    await self._send_channel_warning("Ranked trades channel not found. Trade proposals will use general channel.")
            except ValueError:
                print(f"⚠️ Invalid ranked trades channel ID: {ranked_trades_channel_id}")
                await self._send_channel_warning("Invalid ranked trades channel ID. Trade proposals will use general channel.")
        else:
            print("⚠️ DISCORD_RANKED_TRADES_CHANNEL_ID not set, ranked trades will go to general channel")
            await self._send_channel_warning("DISCORD_RANKED_TRADES_CHANNEL_ID not set. Trade proposals will use general channel.")

    async def _send_channel_warning(self, message: str):
        """Send a channel configuration warning to the general channel."""
        if self.channel and isinstance(self.channel, discord.TextChannel):
            await self.channel.send(f"⚠️ **Configuration Warning**: {message}")

    async def handle_share_news_command(self, message):
        """
        Handle the !share_news command for sharing news links.
        
        Format: !share_news <link> [optional description]
        Example: !share_news https://example.com/news "Potential market impact on tech sector"
        
        If mid-iteration, queues the news for the next iteration start.
        """
        content = message.content.strip()
        logger.info(f"Processing !share_news command from {message.author.display_name}: {content[:100]}")
        
        # Parse the command
        parts = content.split(maxsplit=2)  # Split into: ["!share_news", "<link>", "[description]"]
        
        if len(parts) < 2:
            await message.channel.send("❌ **Invalid format.** Usage: `!share_news <link> [optional description]`\n"
                                      "Example: `!share_news https://example.com/news \"Market update\"`")
            return
        
        link = parts[1].strip()
        description = parts[2].strip().strip('"\'') if len(parts) > 2 else ""
        
        # Validate the URL
        if not self._validate_url(link):
            await message.channel.send("❌ **Invalid link.** Only HTTP/HTTPS URLs are allowed. "
                                      "Please provide a valid news URL.")
            logger.warning(f"Invalid URL rejected from {message.author.display_name}: {link}")
            return
        
        # Sanitize the description
        sanitized_description = self._sanitize_user_input(description) if description else ""
        
        # Create the news share entry
        news_entry = {
            'link': link,
            'description': sanitized_description,
            'user': message.author.display_name[:50],
            'user_id': str(message.author.id),
            'timestamp': datetime.now().isoformat(),
            'message_id': str(message.id),
            'channel_id': str(message.channel.id)
        }
        
        # Check if we're mid-iteration
        if self.iteration_in_progress:
            # Queue for next iteration
            self.shared_news_queue.append(news_entry)
            await message.add_reaction("📥")
            await message.channel.send(
                f"📰 **Input noted** - News link will be processed at the start of the next iteration.\n"
                f"Link: {link[:100]}{'...' if len(link) > 100 else ''}\n"
                f"Queued items: {len(self.shared_news_queue)}"
            )
            logger.info(f"News link queued for next iteration: {link[:100]}")
        else:
            # Process immediately (at iteration start or when no workflow active)
            await message.add_reaction("⏳")
            await self._process_shared_news(news_entry, message.channel)

    async def _process_shared_news(self, news_entry: Dict[str, Any], channel):
        """
        Process a shared news link by forwarding it to the Data Agent for analysis.
        
        Args:
            news_entry: Dict with link, description, and metadata
            channel: Discord channel to send responses to
        """
        link = news_entry['link']
        description = news_entry.get('description', '')
        
        try:
            await channel.send(f"🔍 **Processing news link...**\nAnalyzing: {link[:100]}{'...' if len(link) > 100 else ''}")
            
            # Check if Data Agent is available
            if 'data' not in self.agent_instances:
                await channel.send("⚠️ Data Agent not available. News link logged but not processed.")
                logger.warning("Data Agent not available for news processing")
                return
            
            data_agent = self.agent_instances['data']
            
            # Try to process the news link using the NewsDataAnalyzer
            try:
                # Check if NewsDataAnalyzer has the process_shared_news_link method
                if hasattr(data_agent, 'news_sub') and hasattr(data_agent.news_sub, 'process_shared_news_link'):
                    result = await data_agent.news_sub.process_shared_news_link(link, description)
                else:
                    # Fallback: Use the data agent to analyze the link
                    result = await self._fetch_and_analyze_news_link(link, description, data_agent)
                
                # Format and send the response
                if result.get('success', False):
                    summary = result.get('summary', 'News processed successfully.')
                    sentiment = result.get('sentiment', 'neutral')
                    key_entities = result.get('key_entities', [])
                    
                    response_msg = f"✅ **News processed:**\n"
                    response_msg += f"📝 **Summary:** {summary[:500]}{'...' if len(summary) > 500 else ''}\n"
                    response_msg += f"📊 **Sentiment:** {sentiment.capitalize()}\n"
                    if key_entities:
                        response_msg += f"🏷️ **Key Entities:** {', '.join(key_entities[:5])}\n"
                    response_msg += "✅ Included in current analysis."
                    
                    await channel.send(response_msg)
                    logger.info(f"News link processed successfully: {link[:100]}")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    await channel.send(f"⚠️ **News processing failed:** {error_msg}")
                    logger.error(f"News processing failed for {link}: {error_msg}")
                    
            except Exception as e:
                await channel.send(f"❌ **Error processing news:** {str(e)[:200]}")
                logger.error(f"Exception processing news link {link}: {e}")
                
        except Exception as e:
            await channel.send(f"❌ **Error:** {str(e)[:200]}")
            logger.error(f"Error in _process_shared_news: {e}")

    async def _fetch_and_analyze_news_link(self, link: str, description: str, data_agent) -> Dict[str, Any]:
        """
        Fetch and analyze a news link using requests and BeautifulSoup.
        
        Args:
            link: URL to fetch
            description: User-provided description
            data_agent: The Data Agent instance for LLM analysis
            
        Returns:
            Dict with success status, summary, sentiment, and key entities
        """
        import requests
        from bs4 import BeautifulSoup
        
        try:
            # Fetch the content with timeout and headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ABC-Application/1.0; +https://github.com/nvickers04/ABC-Application)'
            }
            response = requests.get(link, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string or ""
            
            # Extract main content - try common article selectors
            article_text = ""
            article_selectors = ['article', '.article-content', '.post-content', 
                               '.entry-content', 'main', '.content', '#content']
            for selector in article_selectors:
                article = soup.select_one(selector)
                if article:
                    # Get text content, removing scripts and styles
                    for tag in article(['script', 'style', 'nav', 'footer', 'aside']):
                        tag.decompose()
                    article_text = article.get_text(separator=' ', strip=True)
                    break
            
            # Fallback to body text if no article found
            if not article_text:
                body = soup.body
                if body:
                    for tag in body(['script', 'style', 'nav', 'footer', 'aside', 'header']):
                        tag.decompose()
                    article_text = body.get_text(separator=' ', strip=True)
            
            # Truncate content for LLM analysis
            content_for_analysis = article_text[:5000] if article_text else ""
            
            if not content_for_analysis:
                return {
                    'success': False,
                    'error': 'Could not extract meaningful content from the page'
                }
            
            # Analyze with LLM if available
            analysis_result = await self._analyze_news_content_with_llm(
                title, content_for_analysis, description, link, data_agent
            )
            
            return analysis_result
            
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timed out (15s limit)'}
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f'Failed to fetch URL: {str(e)[:100]}'}
        except Exception as e:
            return {'success': False, 'error': f'Processing error: {str(e)[:100]}'}

    async def _analyze_news_content_with_llm(self, title: str, content: str, 
                                             description: str, link: str, 
                                             data_agent) -> Dict[str, Any]:
        """
        Analyze news content using the Data Agent's LLM.
        
        Args:
            title: Article title
            content: Article content
            description: User-provided description
            link: Original URL
            data_agent: Data Agent instance
            
        Returns:
            Dict with analysis results
        """
        try:
            # Prepare the analysis prompt
            analysis_prompt = f"""
Analyze the following news article for market relevance:

Title: {title}
User Note: {description if description else 'None provided'}
URL: {link}

Content (excerpt):
{content[:3000]}

Please provide:
1. A brief summary (2-3 sentences)
2. Market sentiment (bullish, bearish, or neutral)
3. Key entities mentioned (companies, sectors, people, etc.)
4. Potential market impact (high, medium, low)
5. Relevance to trading decisions

Format your response as JSON:
{{
    "summary": "...",
    "sentiment": "bullish/bearish/neutral",
    "key_entities": ["entity1", "entity2", ...],
    "market_impact": "high/medium/low",
    "relevance": "..."
}}
"""
            
            # Use the data agent's LLM if available
            if hasattr(data_agent, 'llm') and data_agent.llm:
                llm_response = await data_agent.llm.ainvoke(analysis_prompt)
                response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                
                # Try to parse JSON response
                try:
                    # Find JSON in response
                    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group())
                        return {
                            'success': True,
                            'summary': analysis.get('summary', 'News analyzed successfully'),
                            'sentiment': analysis.get('sentiment', 'neutral'),
                            'key_entities': analysis.get('key_entities', []),
                            'market_impact': analysis.get('market_impact', 'medium'),
                            'relevance': analysis.get('relevance', 'General market news')
                        }
                except (json.JSONDecodeError, AttributeError):
                    # Return raw summary if JSON parsing fails
                    return {
                        'success': True,
                        'summary': response_text[:500],
                        'sentiment': 'neutral',
                        'key_entities': [],
                        'market_impact': 'medium',
                        'relevance': 'News analyzed'
                    }
            else:
                # Basic analysis without LLM
                return {
                    'success': True,
                    'summary': f"News article: {title[:200]}",
                    'sentiment': 'neutral',
                    'key_entities': [],
                    'market_impact': 'medium',
                    'relevance': 'LLM not available for deep analysis'
                }
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {
                'success': True,
                'summary': f"News article: {title[:200]}",
                'sentiment': 'neutral',
                'key_entities': [],
                'market_impact': 'medium',
                'relevance': 'Basic analysis only'
            }

    async def process_queued_human_inputs(self, channel):
        """
        Process all queued human inputs at the start of an iteration.
        This is called at the beginning of each workflow iteration.
        
        Args:
            channel: Discord channel to send responses to
        """
        if not self.human_input_queue and not self.shared_news_queue:
            return
        
        # Process general human inputs
        if self.human_input_queue:
            await channel.send(f"📬 **Processing {len(self.human_input_queue)} queued human input(s)...**")
            for input_entry in self.human_input_queue:
                self.human_interventions.append(input_entry)
                self.workflow_log.append(f"👤 [Queued] {input_entry['user']}: {input_entry['content'][:100]}...")
            self.human_input_queue.clear()
            await channel.send("✅ Queued inputs processed and included in this iteration.")
        
        # Process shared news links
        if self.shared_news_queue:
            await channel.send(f"📰 **Processing {len(self.shared_news_queue)} queued news link(s)...**")
            for news_entry in self.shared_news_queue:
                await self._process_shared_news(news_entry, channel)
            self.shared_news_queue.clear()
            await channel.send("✅ Queued news links processed.")

    async def handle_schedule_workflow_command(self, message):
        """
        Handle the !schedule_workflow command for scheduling automated workflows.

        Format: !schedule_workflow <type> <time>
        Types: iterative_reasoning, continuous_workflow, health_check
        Time formats:
        - For iterative_reasoning: HH:MM (e.g., 09:30) or interval in hours (e.g., 4h)
        - For continuous_workflow: interval in minutes (e.g., 60m)
        - For health_check: interval in minutes (e.g., 30m)

        Examples:
        - !schedule_workflow iterative_reasoning 09:30
        - !schedule_workflow iterative_reasoning 4h
        - !schedule_workflow continuous_workflow 60m
        - !schedule_workflow health_check 30m
        """
        content = message.content.strip()
        logger.info(f"Processing !schedule_workflow command from {message.author.display_name}: {content}")

        # Parse the command
        parts = content.split()

        if len(parts) < 3:
            await message.channel.send(
                "❌ **Invalid format.** Usage: `!schedule_workflow <type> <time>`\n\n"
                "**Types:**\n"
                "• `iterative_reasoning` - Full workflow (time: HH:MM or Xh)\n"
                "• `continuous_workflow` - Market surveillance (time: Xm)\n"
                "• `health_check` - System health (time: Xm)\n\n"
                "**Examples:**\n"
                "• `!schedule_workflow iterative_reasoning 09:30`\n"
                "• `!schedule_workflow iterative_reasoning 4h`\n"
                "• `!schedule_workflow continuous_workflow 60m`\n"
                "• `!schedule_workflow health_check 30m`"
            )
            return

        workflow_type = parts[1].lower()
        time_param = parts[2].lower()

        try:
            if workflow_type == "iterative_reasoning":
                if ":" in time_param:  # Specific time like 09:30
                    self.schedule_iterative_reasoning(trigger_time=time_param)
                    await message.channel.send(f"✅ **Scheduled iterative reasoning workflow** at {time_param} ET (Mon-Fri)")
                elif time_param.endswith("h"):  # Interval like 4h
                    hours = int(time_param[:-1])
                    self.schedule_iterative_reasoning(interval_hours=hours)
                    await message.channel.send(f"✅ **Scheduled iterative reasoning workflow** every {hours} hours (Mon-Fri)")
                else:
                    await message.channel.send("❌ **Invalid time format.** Use HH:MM (e.g., 09:30) or Xh (e.g., 4h)")

            elif workflow_type == "continuous_workflow":
                if time_param.endswith("m"):
                    minutes = int(time_param[:-1])
                    self.schedule_continuous_workflow(interval_minutes=minutes)
                    await message.channel.send(f"✅ **Scheduled continuous workflow** every {minutes} minutes (Mon-Fri)")
                else:
                    await message.channel.send("❌ **Invalid time format.** Use Xm (e.g., 60m)")

            elif workflow_type == "health_check":
                if time_param.endswith("m"):
                    minutes = int(time_param[:-1])
                    self.schedule_health_check(interval_minutes=minutes)
                    await message.channel.send(f"✅ **Scheduled health check** every {minutes} minutes")
                else:
                    await message.channel.send("❌ **Invalid time format.** Use Xm (e.g., 30m)")

            else:
                await message.channel.send(f"❌ **Unknown workflow type:** {workflow_type}")

        except ValueError as e:
            await message.channel.send(f"❌ **Invalid time parameter:** {time_param}")
        except Exception as e:
            logger.error(f"Error scheduling workflow: {e}")
            await message.channel.send(f"❌ **Scheduling failed:** {str(e)}")

    async def send_scheduler_status(self, message):
        """
        Send current scheduler status and active jobs to Discord.
        """
        try:
            jobs = []
            for job in self.scheduler.get_jobs():
                jobs.append(f"• **{job.name}** (ID: {job.id})")

            if jobs:
                status_msg = "**📅 Active Scheduled Jobs:**\n" + "\n".join(jobs)
            else:
                status_msg = "**📅 No active scheduled jobs**"

            # Add scheduler state
            if self.scheduler.running:
                status_msg += "\n\n✅ **Scheduler Status:** Running"
            else:
                status_msg += "\n\n❌ **Scheduler Status:** Stopped"

            await message.channel.send(status_msg)

        except Exception as e:
            logger.error(f"Error getting scheduler status: {e}")
            await message.channel.send(f"❌ **Error retrieving scheduler status:** {str(e)}")

    async def handle_alert_test_command(self, message):
        """Handle the !alert_test command to test alert system functionality."""
        try:
            from src.utils.alert_manager import get_alert_manager
            alert_manager = get_alert_manager()

            # Send a test alert
            alert_manager.send_alert(
                level="INFO",
                component="DiscordCommand",
                message="Test alert triggered via !alert_test command",
                context={"user": str(message.author), "channel": str(message.channel)}
            )

            await message.channel.send("✅ **Alert test sent!** Check for notification in alerts channel.")

        except Exception as e:
            logger.error(f"Error in alert test command: {e}")
            await message.channel.send(f"❌ **Alert test failed:** {str(e)}")

    async def handle_check_health_now_command(self, message):
        """Handle the !check_health_now command to run immediate health check."""
        try:
            from src.utils.alert_manager import get_alert_manager
            alert_manager = get_alert_manager()

            # Run health check
            health_status = alert_manager.check_health()

            # Format response
            embed = discord.Embed(
                title="🔍 Health Check Results",
                color=0x00FF00 if health_status.get('orchestrator_connected') else 0xFF0000,
                timestamp=datetime.now()
            )

            embed.add_field(
                name="Alert Queue",
                value=f"{health_status.get('alert_queue_size', 0)} alerts",
                inline=True
            )
            embed.add_field(
                name="Recent Critical",
                value=f"{health_status.get('recent_critical_alerts', 0)} alerts",
                inline=True
            )
            embed.add_field(
                name="Recent Errors",
                value=f"{health_status.get('recent_error_alerts', 0)} alerts",
                inline=True
            )
            embed.add_field(
                name="Orchestrator",
                value="✅ Connected" if health_status.get('orchestrator_connected') else "❌ Disconnected",
                inline=True
            )
            embed.add_field(
                name="Discord",
                value="✅ Enabled" if health_status.get('discord_enabled') else "❌ Disabled",
                inline=True
            )

            await message.channel.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in health check command: {e}")
            await message.channel.send(f"❌ **Health check failed:** {str(e)}")

    async def handle_alert_history_command(self, message):
        """Handle the !alert_history command to show recent alerts."""
        try:
            from src.utils.alert_manager import get_alert_manager
            alert_manager = get_alert_manager()

            recent_alerts = alert_manager.get_recent_alerts(limit=10)

            if not recent_alerts:
                await message.channel.send("📭 **No recent alerts**")
                return

            embed = discord.Embed(
                title="📋 Recent Alert History",
                color=0xFFA500,
                timestamp=datetime.now()
            )

            for i, alert in enumerate(recent_alerts, 1):
                timestamp = alert.timestamp.strftime('%m/%d %H:%M')
                level_emoji = {
                    "CRITICAL": "🚨",
                    "ERROR": "❌",
                    "WARNING": "⚠️",
                    "INFO": "ℹ️"
                }.get(alert.level.value if hasattr(alert.level, 'value') else str(alert.level), "❓")

                field_name = f"{level_emoji} {alert.component}"
                field_value = f"**{timestamp}**\n{alert.message[:200]}"
                if len(alert.message) > 200:
                    field_value += "..."

                embed.add_field(name=field_name, value=field_value, inline=False)

            await message.channel.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in alert history command: {e}")
            await message.channel.send(f"❌ **Alert history failed:** {str(e)}")

    async def handle_alert_stats_command(self, message):
        """Handle the !alert_stats command to show alert statistics."""
        try:
            from src.utils.alert_manager import get_alert_manager
            alert_manager = get_alert_manager()

            # Get all alerts for statistics
            all_alerts = alert_manager.error_queue

            if not all_alerts:
                await message.channel.send("📊 **No alerts recorded yet**")
                return

            # Calculate statistics
            total_alerts = len(all_alerts)
            critical_count = sum(1 for a in all_alerts if a.level == "CRITICAL" or (hasattr(a.level, 'value') and a.level.value == "CRITICAL"))
            error_count = sum(1 for a in all_alerts if a.level in ["ERROR", "CRITICAL"] or (hasattr(a.level, 'value') and a.level.value in ["ERROR", "CRITICAL"]))
            warning_count = sum(1 for a in all_alerts if a.level == "WARNING" or (hasattr(a.level, 'value') and a.level.value == "WARNING"))
            info_count = sum(1 for a in all_alerts if a.level == "INFO" or (hasattr(a.level, 'value') and a.level.value == "INFO"))

            # Time-based stats (last 24 hours)
            now = datetime.now()
            last_24h = [a for a in all_alerts if (now - a.timestamp).total_seconds() < 86400]
            alerts_24h = len(last_24h)
            critical_24h = sum(1 for a in last_24h if a.level == "CRITICAL" or (hasattr(a.level, 'value') and a.level.value == "CRITICAL"))

            embed = discord.Embed(
                title="📊 Alert Statistics",
                color=0x3498DB,
                timestamp=datetime.now()
            )

            embed.add_field(
                name="Total Alerts",
                value=f"{total_alerts}",
                inline=True
            )
            embed.add_field(
                name="Last 24 Hours",
                value=f"{alerts_24h}",
                inline=True
            )
            embed.add_field(
                name="Critical (24h)",
                value=f"{critical_24h}",
                inline=True
            )

            embed.add_field(
                name="By Severity",
                value=f"🚨 Critical: {critical_count}\n❌ Error: {error_count}\n⚠️ Warning: {warning_count}\nℹ️ Info: {info_count}",
                inline=False
            )

            # Component breakdown (top 5)
            component_counts = {}
            for alert in all_alerts[-100:]:  # Last 100 alerts
                component = alert.component
                component_counts[component] = component_counts.get(component, 0) + 1

            top_components = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_components:
                component_str = "\n".join(f"• {comp}: {count}" for comp, count in top_components)
                embed.add_field(
                    name="Top Components (Last 100)",
                    value=component_str,
                    inline=False
                )

            await message.channel.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in alert stats command: {e}")
            await message.channel.send(f"❌ **Alert stats failed:** {str(e)}")

    async def handle_commands_command(self, message):
        """Handle the !commands command to show all available commands."""
        try:
            from src.utils.command_registry import get_command_registry
            registry = get_command_registry()
            embed_data = registry.generate_discord_embed_data()

            embed = discord.Embed(
                title=embed_data["title"],
                description=embed_data["description"],
                color=0x3498DB,
                timestamp=datetime.now()
            )

            for field in embed_data["fields"]:
                embed.add_field(
                    name=field["name"],
                    value=field["value"],
                    inline=field["inline"]
                )

            embed.set_footer(text="💡 Use /commands for slash command version | Visit #commands for detailed docs")

            await message.channel.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in commands command: {e}")
            await message.channel.send(f"❌ **Commands display failed:** {str(e)}")

    async def handle_set_commands_channel_command(self, message):
        """Handle the !set_commands_channel command to configure the dedicated commands channel."""
        try:
            # Parse channel ID from command: !set_commands_channel <channel_id>
            parts = message.content.strip().split()
            if len(parts) != 2:
                await message.channel.send("❌ **Usage:** `!set_commands_channel <channel_id>`\n\nGet the channel ID by right-clicking the #commands channel and selecting 'Copy ID' (Developer Mode must be enabled).")
                return

            channel_id_str = parts[1]
            try:
                channel_id = int(channel_id_str)
            except ValueError:
                await message.channel.send("❌ **Invalid channel ID:** Must be a numeric Discord channel ID.")
                return

            # Get the channel
            commands_channel = self.client.get_channel(channel_id)
            if not commands_channel:
                await message.channel.send(f"❌ **Channel not found:** Could not find channel with ID `{channel_id}`. Make sure the bot has access to this channel.")
                return

            # Set the commands channel
            self.commands_channel = commands_channel

            # Generate and post detailed documentation
            from src.utils.command_registry import get_command_registry
            registry = get_command_registry()
            markdown_docs = registry.generate_markdown_docs()

            # Send as a file attachment to avoid message length limits
            import io
            file_content = f"# 🤖 ABC-Application Command Reference\n\n{markdown_docs}"
            file_obj = io.BytesIO(file_content.encode('utf-8'))
            file_obj.seek(0)

            await self.commands_channel.send(
                content="📚 **Command Reference Documentation**",
                file=discord.File(file_obj, filename="command_reference.md")
            )

            # Confirm to user
            await message.channel.send(f"✅ **Commands channel configured!**\n\n📚 Detailed command documentation has been posted to {self.commands_channel.mention}\n\nUsers can now visit {self.commands_channel.mention} for comprehensive command help.")

        except Exception as e:
            logger.error(f"Error in set commands channel command: {e}")
            await message.channel.send(f"❌ **Failed to configure commands channel:** {str(e)}")

    async def handle_consensus_status_command(self, message):
        """Handle the !consensus_status command to show current consensus polls."""
        try:
            active_polls = self.consensus_poller.get_active_polls()
            completed_polls = self.consensus_poller.get_completed_polls(limit=5)

            if not active_polls and not completed_polls:
                await message.channel.send("🤝 **No consensus polls found**")
                return

            embed = discord.Embed(
                title="🤝 Consensus Poll Status",
                color=0x3498DB,
                timestamp=datetime.now()
            )

            if active_polls:
                active_list = []
                for poll in active_polls[:3]:  # Show up to 3 active polls
                    status = f"**{poll.state.value.replace('_', ' ').title()}**"
                    votes = sum(1 for v in poll.votes.values() if v.get("status") == "responded")
                    total = len(poll.votes)
                    active_list.append(f"• `{poll.poll_id}`: {status} ({votes}/{total} votes)")

                embed.add_field(
                    name="Active Polls",
                    value="\n".join(active_list) if active_list else "None",
                    inline=False
                )

            if completed_polls:
                completed_list = []
                for poll in completed_polls:
                    result = "✅" if poll.state.value == "consensus_reached" else "⏰" if poll.state.value == "timeout" else "❌"
                    status = poll.state.value.replace("_", " ").title()
                    completed_list.append(f"{result} `{poll.poll_id}`: {status}")

                embed.add_field(
                    name="Recent Results",
                    value="\n".join(completed_list[:3]) if completed_list else "None",
                    inline=False
                )

            await message.channel.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in consensus status command: {e}")
            await message.channel.send(f"❌ **Consensus status failed:** {str(e)}")

    async def handle_poll_consensus_command(self, message):
        """Handle the !poll_consensus command to start a new consensus poll."""
        try:
            # Parse command: !poll_consensus "Question?" agent1 agent2 agent3
            content = message.content.strip()
            parts = content.split('"')

            if len(parts) < 3:
                await message.channel.send("❌ **Usage:** `!poll_consensus \"Your question?\" agent1 agent2 agent3`")
                return

            question = parts[1].strip()
            if not question:
                await message.channel.send("❌ **Question cannot be empty**")
                return

            # Parse agents (everything after the closing quote)
            remaining = parts[2].strip()
            agents_to_poll = [agent.strip() for agent in remaining.split() if agent.strip()]

            if len(agents_to_poll) < 2:
                await message.channel.send("❌ **Need at least 2 agents to poll**")
                return

            # Validate agents exist (basic check)
            valid_agents = ["risk_agent", "strategy_agent", "data_agent", "execution_agent", "reflection_agent"]
            invalid_agents = [agent for agent in agents_to_poll if agent not in valid_agents]

            if invalid_agents:
                await message.channel.send(f"❌ **Invalid agents:** {', '.join(invalid_agents)}\n**Valid agents:** {', '.join(valid_agents)}")
                return

            # Create the poll
            poll_id = await self.consensus_poller.create_poll(question, agents_to_poll)

            # Start the poll
            success = await self.consensus_poller.start_poll(poll_id)

            if success:
                embed = discord.Embed(
                    title="🤝 Consensus Poll Started",
                    description=f"**Question:** {question}",
                    color=0x00ff00,
                    timestamp=datetime.now()
                )
                embed.add_field(name="Poll ID", value=poll_id, inline=True)
                embed.add_field(name="Agents Polled", value=", ".join(agents_to_poll), inline=True)
                embed.add_field(name="Timeout", value="5 minutes", inline=True)

                await message.channel.send(embed=embed)
            else:
                await message.channel.send(f"❌ **Failed to start consensus poll**")

        except Exception as e:
            logger.error(f"Error in poll consensus command: {e}")
            await message.channel.send(f"❌ **Poll consensus failed:** {str(e)}")

    async def handle_human_intervention(self, message):
        """Handle human questions or interventions during workflow via A2A.
        
        If mid-iteration, queues the input for the next iteration start.
        Otherwise, processes immediately.
        """
        # SECURITY: Validate message structure
        if not message or not hasattr(message, 'content') or not hasattr(message, 'author'):
            return

        content = message.content.strip()
        if len(content) > 2000:  # Discord message limit + safety buffer
            await message.channel.send("⚠️ Message too long. Please keep interventions under 2000 characters.")
            return

        # SECURITY: Sanitize user input
        sanitized_content = self._sanitize_user_input(content)
        if not sanitized_content:
            await message.add_reaction("❌")
            await message.channel.send("⚠️ Message contains invalid content and was rejected.")
            return

        intervention = {
            'user': message.author.display_name[:50],  # Limit username length
            'user_id': str(message.author.id),  # Track user ID for audit
            'content': sanitized_content,
            'timestamp': message.created_at.isoformat(),
            'phase': self.current_phase
        }
        
        # Check if we're mid-iteration
        if self.iteration_in_progress:
            # Queue for next iteration
            self.human_input_queue.append(intervention)
            await message.add_reaction("📥")
            await message.channel.send(
                f"📝 **Input noted** - will be considered at the start of the next iteration.\n"
                f"Queued items: {len(self.human_input_queue)}"
            )
            self.workflow_log.append(f"📥 [Queued] {message.author.display_name[:50]}: {sanitized_content[:100]}...")
            logger.info(f"Human input queued for next iteration from {message.author.display_name}")
            return
        
        # Process immediately (at iteration start or between iterations)
        self.human_interventions.append(intervention)
        self.workflow_log.append(f"👤 {message.author.display_name[:50]}: {sanitized_content[:100]}...")

        # Acknowledge the intervention
        await message.add_reaction("👀")

        # If it's a question, consult reflection agent via A2A
        if any(word in message.content.lower() for word in ['?', 'what', 'how', 'why', 'can you', 'explain']):
            await message.channel.send(f"Human intervention noted: `{message.content[:100]}...`\nConsulting reflection agent...")

            # Ask reflection agent to address the question via A2A
            if 'reflection' in self.agent_instances:
                try:
                    reflection_response = await self.agent_instances['reflection'].analyze(
                        f"Human Question: {message.content}. Please analyze this question in the context of our current workflow phase ({self.current_phase}) and provide relevant insights or recommendations."
                    )
                    
                    # Format and present reflection agent's response
                    await message.channel.send("**Reflection Agent Analysis:**")
                    if isinstance(reflection_response, dict):
                        for key, value in reflection_response.items():
                            if key not in ['timestamp', 'agent_role']:
                                # SECURITY: Sanitize agent output before display
                                sanitized_value = self._sanitize_agent_output(str(value))
                                await message.channel.send(f"• **{key.replace('_', ' ').title()}:** {sanitized_value[:500]}")
                    else:
                        # SECURITY: Sanitize agent output before display
                        sanitized_response = self._sanitize_agent_output(str(reflection_response))
                        await message.channel.send(sanitized_response[:1000])
                        
                except Exception as e:
                    await message.channel.send(f"Reflection agent consultation failed: {str(e)}")
            else:
                await message.channel.send("Reflection agent not available for consultation")

            await message.channel.send("Continuing workflow...")
        else:
            # For non-questions, just log and continue
            await message.channel.send("Intervention logged. Continuing workflow...")

    async def start_workflow(self):
        """Start the CONTINUOUS ALPHA DISCOVERY WORKFLOW - agents collaborate continuously!"""
        if not self.channel:
            print("❌ No channel available for workflow")
            return

        channel = cast(discord.TextChannel, self.channel)

        if self.workflow_active:
            await channel.send("⚠️ Workflow already active!")
            return

        # Post system event - COMMENTED OUT FOR TROUBLESHOOTING
        # await self._post_system_event("workflow_start", "Continuous Alpha Discovery Workflow initiated", f"Agents: {len(self.agent_instances)} ready for collaboration")

        # Check agent health before starting
        health_status = await self.check_agent_health()
        if health_status['overall_health'] == 'critical':
            await channel.send("🚨 **CRITICAL: System health prevents workflow start**")
            await channel.send(f"Healthy agents: {len(health_status['healthy_agents'])}/{health_status['total_agents']}")
            return
        elif health_status['overall_health'] == 'degraded':
            await channel.send("⚠️ **WARNING: System health degraded - proceeding with caution**")
            await channel.send(f"Healthy agents: {len(health_status['healthy_agents'])}/{health_status['total_agents']}")

        await channel.send("**COLLABORATIVE ANALYSIS SESSION ESTABLISHED**")
        await channel.send("All agents now share full workflow context for unified analysis.")

        # Create collaborative session for enhanced cross-agent communication
        session_created = await self.create_collaborative_session("Continuous Alpha Discovery with Full Context Sharing")
        if session_created:
            await channel.send("Enhanced A2A session active - agents share insights and build on each other's analysis")
        else:
            await channel.send("Limited collaboration mode - agents operating with reduced context sharing")

        self.workflow_active = True
        self.current_phase = "systematic_market_surveillance"
        self.workflow_log = []
        self.responses_collected = []
        self.human_interventions = []

        await channel.send("**CONTINUOUS ANALYSIS WORKFLOW STARTED**")
        await channel.send("Multi-agent collaborative analysis mode activated.")
       
       
        alpha_hunt_cycles = 0
        max_cycles = 10  # Prevent infinite loops this may need to be configurable with scheduler

        while self.workflow_active and alpha_hunt_cycles < max_cycles:
            alpha_hunt_cycles += 1
            await channel.send(f"\n**ANALYSIS CYCLE {alpha_hunt_cycles}**")
            
            # Process any queued human inputs at the START of this iteration
            await self.process_queued_human_inputs(channel)
            
            # Mark iteration as in-progress (new inputs will be queued)
            self.iteration_in_progress = True
            logger.info(f"Starting iteration {alpha_hunt_cycles} - human inputs will now be queued")

            # Phase 1: SYSTEMATIC MARKET SURVEILLANCE - Deep dive
            await self.execute_systematic_market_surveillance(channel)

            # Phase 2: PARALLEL COLLABORATION - Agents synergize
            await self.execute_parallel_collaboration(channel)

            # Phase 3: QUANTITATIVE OPPORTUNITY VALIDATION - Data-driven checks
            await self.execute_quantitative_opportunity_validation(channel)

            # Phase 4: RAPID CONSENSUS BUILDING - Align on top ideas
            await self.execute_rapid_consensus(channel)

            # Phase 5: PORTFOLIO IMPLEMENTATION PLANNING - Strategy formulation
            await self.execute_portfolio_implementation_planning(channel)

            # Phase 6: PERFORMANCE ANALYTICS & REFINEMENT - Learn and adapt
            await self.execute_performance_analytics_and_refinement(channel)

            # Phase 7: CHIEF INVESTMENT OFFICER OVERSIGHT - Final review
            execution_decision = await self.execute_chief_investment_officer_oversight(channel)
            
            # Mark iteration as complete (inputs will be processed immediately until next iteration starts)
            self.iteration_in_progress = False
            logger.info(f"Iteration {alpha_hunt_cycles} complete - human inputs will now be processed immediately")

            # Check execution decision
            if execution_decision == "EXECUTE":
                await channel.send("**EXECUTION APPROVED**")
                # Execute trades here
                await self.execute_alpha_deployment(channel)
                break  # Exit loop after execution
            elif execution_decision == "HOLD":
                await channel.send("**HOLDING POSITION** - monitoring for better conditions")
                await channel.send("Continuing analysis in next cycle")
                await asyncio.sleep(60)  # Brief pause before next cycle
            elif execution_decision == "RESTART":
                await channel.send("**RESTARTING ANALYSIS** - applying lessons learned")
                # Continue loop with improvements
                await asyncio.sleep(30)
            else:
                await channel.send("**DECISION PENDING** - continuing analysis")
                await asyncio.sleep(30)

        # Complete workflow
        self.iteration_in_progress = False  # Ensure flag is reset
        await channel.send("**ANALYSIS MISSION COMPLETE**")
        await self.complete_continuous_workflow(channel)

    async def start_premarket_analysis_workflow(self):
        """Start the premarket analysis workflow - focused on early market preparation"""
        if not self.channel:
            print("❌ No channel available for workflow")
            return

        channel = cast(discord.TextChannel, self.channel)

        if self.workflow_active:
            await channel.send("⚠️ Workflow already active! Complete current workflow first.")
            return

        # Check agent health before starting
        health_status = await self.check_agent_health()
        if health_status['overall_health'] == 'critical':
            await channel.send("🚨 **CRITICAL: System health prevents workflow start**")
            return

        await channel.send("**PREMARKET ANALYSIS WORKFLOW STARTED**")
        await channel.send("Early market preparation and data collection.")
        await channel.send("Analyzing pre-market conditions and preparing for market open.")

        # Create collaborative session for premarket analysis
        session_created = await self.create_collaborative_session("Premarket Analysis")
        if session_created:
            await channel.send("🎯 **Premarket Session Active** - Early market analysis")

        self.workflow_active = True
        self.current_phase = "pre_market_readiness_assessment"
        self.workflow_log = []
        self.responses_collected = []
        self.human_interventions = []

        # Premarket readiness assessment
        await self.execute_phase_with_agents('pre_market_readiness_assessment', "🔍 PRE-MARKET READINESS ASSESSMENT")

        # Additional premarket phases if needed (e.g., data collection, strategy prep)
        await self.execute_phase_with_agents('systematic_market_surveillance', "📊 SYSTEMATIC MARKET SURVEILLANCE")

        # Complete premarket analysis
        await channel.send("**PREMARKET ANALYSIS COMPLETE**")
        await channel.send("Ready for market open execution or further analysis.")

        self.workflow_active = False

    async def start_market_open_execution_workflow(self):
        """Start the market open execution workflow - fast-track execution leveraging premarket analysis"""
        if not self.channel:
            print("❌ No channel available for workflow")
            return

        channel = cast(discord.TextChannel, self.channel)

        if self.workflow_active:
            await channel.send("⚠️ Workflow already active! Complete current workflow first.")
            return

        # Check agent health before starting
        health_status = await self.check_agent_health()
        if health_status['overall_health'] == 'critical':
            await channel.send("🚨 **CRITICAL: System health prevents workflow start**")
            return

        await channel.send("**MARKET OPEN EXECUTION WORKFLOW STARTED**")
        await channel.send("Fast-track execution leveraging premarket analysis.")
        await channel.send("Executing pre-approved trades at market open.")

        # Create collaborative session for execution coordination
        session_created = await self.create_collaborative_session("Market Open Execution")
        if session_created:
            await channel.send("🎯 **Execution Session Active** - Coordinated trade execution")

        self.workflow_active = True
        self.current_phase = "opening_bell_execution"
        self.workflow_log = []
        self.responses_collected = []
        self.human_interventions = []

        # Quick market check phase
        await self.execute_phase_with_agents('pre_market_readiness_assessment', "🔍 PRE-MARKET READINESS ASSESSMENT")

        # Execution phase
        await self.execute_phase_with_agents('opening_bell_execution', "💹 OPENING BELL EXECUTION")

        # Complete and transition to monitoring
        await channel.send("**MARKET OPEN EXECUTION COMPLETE**")
        await channel.send("Auto-transitioning to trade monitoring.")

        # Automatically start trade monitoring
        await self.start_trade_monitoring_workflow()

    async def start_trade_monitoring_workflow(self):
        """Start the trade monitoring workflow for active positions"""
        if not self.channel:
            print("❌ No channel available for monitoring")
            return

        channel = cast(discord.TextChannel, self.channel)

        # Don't check workflow_active here - monitoring can run alongside other activities
        # But ensure we don't have conflicting workflows

        await channel.send("**TRADE MONITORING WORKFLOW ACTIVATED**")
        await channel.send("Continuous monitoring of executed positions.")
        await channel.send("Real-time risk management and exit opportunity assessment.")

        # Create monitoring session
        session_created = await self.create_collaborative_session("Active Trade Monitoring")
        if session_created:
            await channel.send("🎯 **Monitoring Session Active** - Continuous position oversight")

        self.monitoring_active = True
        self.current_phase = "trade_monitoring"
        self.monitoring_log = []
        self.monitoring_responses = []

        # Setup monitoring parameters
        await self.execute_phase_with_agents('position_surveillance_initialization', "⚙️ POSITION SURVEILLANCE INITIALIZATION")

        # Start continuous monitoring loop
        await channel.send("Starting continuous monitoring loop.")
        await channel.send("Monitoring will run continuously until positions are closed.")

        # Run monitoring phases in a loop
        monitoring_cycle = 0
        while self.monitoring_active:
            monitoring_cycle += 1
            await channel.send(f"\n**MONITORING CYCLE {monitoring_cycle}**")

            # Active monitoring phase
            await self.execute_phase_with_agents('active_position_management', f"Active Position Management - Cycle {monitoring_cycle}")

            # Decision phase
            await self.execute_phase_with_agents('dynamic_portfolio_adjustment', f"Dynamic Portfolio Adjustment - Cycle {monitoring_cycle}")

            # Check if we need to execute any adjustments
            await self.execute_phase_with_agents('execution_of_portfolio_changes', f"Execution of Portfolio Changes - Cycle {monitoring_cycle}")

            # Check if all positions are closed
            if await self._check_positions_closed():
                await channel.send("All positions closed - monitoring complete.")
                self.monitoring_active = False
                break

            # Wait before next monitoring cycle (e.g., 5 minutes)
            await channel.send("Waiting 5 minutes before next monitoring cycle...")
            await asyncio.sleep(300)  # 5 minutes

        await self.complete_monitoring_workflow()

    async def _check_positions_closed(self) -> bool:
        """Check if all monitored positions have been closed"""
        try:
            # Get current positions from IBKR
            from src.integrations.ibkr_connector import get_ibkr_connector
            ibkr_connector = get_ibkr_connector()

            positions = await ibkr_connector.get_positions()
            active_positions = [p for p in positions if p.get('position', 0) != 0]

            return len(active_positions) == 0
        except Exception as e:
            print(f"Error checking positions: {e}")
            return False

    async def complete_monitoring_workflow(self):
        """Complete the monitoring workflow"""
        if not self.channel:
            return

        channel = cast(discord.TextChannel, self.channel)
        self.monitoring_active = False
        self.current_phase = "monitoring_completed"

        await channel.send("**TRADE MONITORING COMPLETED**")

        # Save monitoring results
        monitoring_results = {
            'completed_at': datetime.now().astimezone().isoformat(),
            'monitoring_cycles': len(self.monitoring_log) if hasattr(self, 'monitoring_log') else 0,
            'monitoring_responses': self.monitoring_responses if hasattr(self, 'monitoring_responses') else [],
            'final_positions': await self._get_current_positions()
        }

        with open('data/trade_monitoring_results.json', 'w') as f:
            json.dump(monitoring_results, f, indent=2, default=str)

        await channel.send("💾 Monitoring results saved to `data/trade_monitoring_results.json`")
        await channel.send("🔄 Ready for next analysis workflow! Type `!start_workflow` to begin.")

    async def execute_phase(self, phase_key: str, phase_title: str):
        """Execute a single phase of the workflow"""
        if not self.channel:
            print(f"❌ No channel available for phase {phase_key}")
            await self.alert_manager.error(
                Exception(f"No Discord channel available for phase execution: {phase_key}"),
                {"phase": phase_key, "phase_title": phase_title},
                "orchestrator"
            )
            return

        general_channel = cast(discord.TextChannel, self.channel)
        self.current_phase = phase_key

        # Announce phase start in general channel
        await general_channel.send(f"\n{phase_title}")
        await general_channel.send("─" * 50)

        commands = self.phase_commands.get(phase_key, [])
        max_wait_time = self.phase_delays.get(phase_key, 86400)

        for i, command in enumerate(commands, 1):
            if not self.workflow_active:
                return  # Allow for pausing/stopping

            # Determine which channel to send this command to
            target_channel = self.get_command_channel(command)
            channel_name = target_channel.name if target_channel and target_channel != general_channel else "general"
            channel_emoji = "🚨" if target_channel == self.alerts_channel else "📊" if target_channel == self.ranked_trades_channel else "💬"

            # Announce command routing in general channel
            await general_channel.send(f"**Command {i}/{len(commands)}:** {command}")
            await general_channel.send(f"📤 Routing to {channel_emoji} #{channel_name}")

            # Send the command to the appropriate channel
            if target_channel:
                print(f"Routing command to #{channel_name}: {command[:50]}...")
                await target_channel.send(command)
            else:
                # Fallback to general channel if target not found
                print("ERROR: No target channel, using general")
                await general_channel.send(command)

            # Wait for responses with extended timing for complex analysis
            await general_channel.send(f"⏳ Waiting up to {max_wait_time}s for agent responses...")

            # Track responses for this specific command
            initial_response_count = len(self.responses_collected)
            start_time = time.time()
            responses_received = 0

            while time.time() - start_time < max_wait_time and self.workflow_active:
                await asyncio.sleep(5)  # Check every 5 seconds for better performance

                # Count new responses for this phase
                current_responses = len([r for r in self.responses_collected if r['phase'] == phase_key])
                if current_responses > initial_response_count:
                    responses_received = current_responses - initial_response_count
                    await general_channel.send(f"📥 Received {responses_received} response(s) so far...")

                    # If we got responses, wait a bit longer for additional ones
                    if responses_received >= 1:
                        await asyncio.sleep(10)  # Wait 10 more seconds for additional responses
                        break

            # Final count
            final_responses = len([r for r in self.responses_collected if r['phase'] == phase_key]) - initial_response_count
            if final_responses > 0:
                await general_channel.send(f"✅ **Phase {phase_key}**: {final_responses} responses received")
            else:
                await general_channel.send(f"⏰ **Phase {phase_key}**: No responses within {max_wait_time}s")
                await self.alert_manager.warning(
                    f"No agent responses received for phase: {phase_key}",
                    {"phase": phase_key, "phase_title": phase_title, "timeout_seconds": max_wait_time},
                    "orchestrator"
                )

        # Phase complete
        await general_channel.send(f"✅ **{phase_title} Complete!**")

        # Brief pause between phases
        await asyncio.sleep(3)

    async def pause_workflow(self):
        """Pause the current workflow"""
        if self.channel is None:
            return

        channel = cast(discord.TextChannel, self.channel)
        self.workflow_active = False
        await channel.send("Workflow paused. Type `!resume_workflow` to continue.")

        # Post system event
        await self._post_system_event(
            "workflow_pause",
            "Workflow Execution Paused",
            "Manual intervention requested. Awaiting resume command.",
            {"current_phase": getattr(self, 'current_phase', 'Unknown'), "active_agents": len(self.agent_instances)}
        )

    async def resume_workflow(self):
        """Resume a paused workflow"""
        if self.channel is None:
            return

        channel = cast(discord.TextChannel, self.channel)
        if not self.workflow_active:
            self.workflow_active = True
            await channel.send("Workflow resumed. Continuing from current phase.")

            # Post system event
            await self._post_system_event(
                "workflow_resume",
                "Workflow Execution Resumed",
                "Continuing from paused state with full agent collaboration.",
                {"current_phase": getattr(self, 'current_phase', 'Unknown'), "active_agents": len(self.agent_instances)}
            )

            # Could implement logic to resume from current phase

    async def stop_workflow(self):
        """Stop the current workflow"""
        if self.channel is None:
            return

        channel = cast(discord.TextChannel, self.channel)
        self.workflow_active = False
        self.current_phase = "stopped"
        await channel.send("Workflow stopped. All progress saved.")

    async def send_status_update(self):
        """Send current workflow status"""
        if self.channel is None:
            return

        channel = cast(discord.TextChannel, self.channel)
        status_msg = f"📊 **Workflow Status:** {self.current_phase.replace('_', ' ').title()}\n"
        status_msg += f"🤖 Active: {'Yes' if self.workflow_active else 'No'}\n"
        status_msg += f"💬 Responses Collected: {len(self.responses_collected)}\n"
        status_msg += f"👤 Human Interventions: {len(self.human_interventions)}\n"

        if self.workflow_log:
            status_msg += f"\n📝 Recent Activity:\n"
            for log_entry in self.workflow_log[-3:]:  # Last 3 entries
                status_msg += f"• {log_entry[:80]}...\n"

        await channel.send(status_msg)

    async def complete_workflow(self):
        """Complete the workflow and provide summary with audit logging"""
        if self.channel is None:
            return

        channel = cast(discord.TextChannel, self.channel)
        self.workflow_active = False
        self.current_phase = "completed"

        await channel.send("🎉 **WORKFLOW COMPLETED!**")
        
        # Get final agent health status
        final_health = await self.check_agent_health()
        
        await channel.send("📊 **Final Summary:**")
        await channel.send(f"• Agent Interactions: {len(self.responses_collected)} A2A communications")
        await channel.send(f"• Human Interventions: {len(self.human_interventions)}")
        await channel.send(f"• Phases Completed: 9 phases (1 iteration)")
        await channel.send(f"• Agent Health: {final_health['overall_health'].title()} ({len(final_health['healthy_agents'])}/{final_health['total_agents']} healthy)")
        await channel.send(f"• Architecture: Unified A2A coordination (no separate Discord bots)")
        
        if self.collaborative_session_id:
            await channel.send(f"• Collaborative Session: {self.collaborative_session_id}")
            
            # Archive the collaborative session
            try:
                if self.agent_instances:
                    first_agent = next(iter(self.agent_instances.values()))
                    await first_agent.archive_session(self.collaborative_session_id)
                    await channel.send("• Session Archived: ✅")
            except Exception as e:
                await channel.send(f"• Session Archive: ❌ ({str(e)})")

        # Save comprehensive results with audit log
        results = {
            'completed_at': datetime.now().astimezone().isoformat(),
            'total_responses': len(self.responses_collected),
            'human_interventions': len(self.human_interventions),
            'responses': self.responses_collected,
            'interventions': self.human_interventions,
            'workflow_log': self.workflow_log,
            'agent_health': final_health,
            'collaborative_session_id': self.collaborative_session_id,
            'direct_agent_integration': bool(self.agent_instances),
            'phases_completed': 9
        }

        # Generate or load encryption key
        key_path = 'data/encryption_key.key'
        if not os.path.exists(key_path):
            key = Fernet.generate_key()
            with open(key_path, 'wb') as key_file:
                key_file.write(key)
        else:
            with open(key_path, 'rb') as key_file:
                key = key_file.read()
        fernet = Fernet(key)

        # Encrypt and save results
        encrypted_results = fernet.encrypt(json.dumps(results, indent=2, default=str).encode())
        with open('data/live_workflow_results.enc', 'wb') as f:
            f.write(encrypted_results)

        # Save encrypted audit log
        audit_log = {
            'timestamp': datetime.now().astimezone().isoformat(),
            'decisions': [r for r in self.responses_collected if 'decision' in str(r.get('response', ''))],
            'trades': [r for r in self.responses_collected if 'trade' in str(r.get('response', ''))],
            'risk_assessments': [r for r in self.responses_collected if 'risk' in str(r.get('response', ''))]
        }
        encrypted_audit = fernet.encrypt(json.dumps(audit_log, indent=2, default=str).encode())
        with open('data/workflow_audit_log.enc', 'wb') as f:
            f.write(encrypted_audit)

        await channel.send("💾 Results saved to `data/live_workflow_results.json`")
        await channel.send("📝 Audit log saved to `data/workflow_audit_log.json`")
        await channel.send("\n🔄 Ready for next workflow! Type `!start_workflow` to begin again.")

    # CONTINUOUS ALPHA DISCOVERY METHODS

    async def execute_workflow_phase(self, channel, phase_title: str, phase_key: str, commands: List[str], completion_message: str):
        """Generic method for executing workflow phases with standardized structure."""
        await channel.send(f"\n{phase_title}")
        await channel.send("─" * 50)
        await channel.send(f"Executing {len(commands)} analysis tasks simultaneously.")

        # ENHANCED PARALLEL EXECUTION: Share complete workflow context with all agents
        await self._share_full_workflow_context(phase_key, phase_title)

        # POSITION AWARENESS: Include current position data in context
        await self._share_position_context()

        # SEND ALL COMMANDS TO ALL AGENTS SIMULTANEOUSLY for maximum collaboration
        agent_responses = await self._execute_commands_parallel(commands, phase_key)
        self.responses_collected.extend(agent_responses)

        # Announce parallel execution with clear separation
        await channel.send(f"\n🎯 **PARALLEL EXECUTION:** {len(commands)} commands sent to {len(self.agent_instances)} agents simultaneously!")
        await channel.send("🤝 **Agents collaborating with full shared context - no silos!**")

        # Format and display agent responses with enhanced readability
        if agent_responses:
            await self._present_agent_responses_enhanced(channel, agent_responses, phase_key)
        else:
            await channel.send("\n⏰ **No agent responses received** - but they're thinking hard! 🤔")

        await channel.send(f"\n✅ **{phase_title} Complete! {completion_message}** 🎉")
        await asyncio.sleep(2)  # Reduced from 3 to speed up

    async def execute_systematic_market_surveillance(self, channel):
        """Phase 1: CONTINUOUS ALPHA DISCOVERY - All agents hunt together"""
        commands = [
            "Analyze current market conditions and identify unusual price action, volume patterns, and emerging trends.",
            "Review technical indicators and quantitative signals for momentum shifts and statistical anomalies.",
            "Evaluate macroeconomic factors and their potential market impact.",
            "Assess relative strength across sectors and identify potential opportunities."
        ]
        await self.execute_workflow_phase(channel, "PHASE 1: MARKET ANALYSIS", "alpha_discovery", commands, "Alpha discovered!")

    async def execute_parallel_collaboration(self, channel):
        """Phase 2: PARALLEL AGENT COLLABORATION - Build on each other's insights"""
        commands = [
            "Cross-validate findings across different analytical approaches and data sources.",
            "Identify complementary signals and assess overall conviction levels.",
            "Evaluate signal strength and identify potential false positives.",
            "Prioritize opportunities based on combined analytical perspectives."
        ]
        await self.execute_workflow_phase(channel, "PHASE 2: COLLABORATIVE ANALYSIS", "parallel_collaboration", commands, "Signals validated!")

    async def execute_quantitative_opportunity_validation(self, channel):
        """Phase 3: OPPORTUNITY ASSESSMENT - Comprehensive evaluation"""
        commands = [
            "Perform comprehensive opportunity assessment against historical precedents.",
            "Conduct detailed risk analysis including downside potential and volatility.",
            "Quantify expected returns and calculate risk-adjusted metrics.",
            "Develop execution plans with entry/exit criteria and position sizing."
        ]
        await self.execute_workflow_phase(channel, "PHASE 3: OPPORTUNITY ASSESSMENT", "quantitative_opportunity_validation", commands, "Opportunities validated!")

    async def execute_rapid_consensus(self, channel):
        """Phase 4: CONSENSUS BUILDING - Fast-track to decisions"""
        commands = [
            "Evaluate opportunities against established criteria and ranking frameworks.",
            "Assess market regime alignment and current condition suitability.",
            "Cross-validate signals across technical, fundamental, and quantitative frameworks.",
            "Refine trade structures and optimize risk management parameters."
        ]
        await self.execute_workflow_phase(channel, "PHASE 4: CONSENSUS BUILDING", "rapid_consensus", commands, "Consensus reached!")

    async def execute_portfolio_implementation_planning(self, channel):
        """Phase 5: EXECUTION PREPARATION - Get ready to deploy"""
        commands = [
            "Calculate optimal position sizing considering portfolio impact and risk limits.",
            "Define comprehensive risk management parameters and stop-loss levels.",
            "Determine precise entry timing and execution methodology.",
            "Prepare detailed execution playbook with all trade parameters."
        ]
        await self.execute_workflow_phase(channel, "PHASE 5: EXECUTION PREPARATION", "portfolio_implementation_planning", commands, "Ready for execution!")

    async def execute_performance_analytics_and_refinement(self, channel):
        """Phase 6: PROCESS IMPROVEMENT - Systematic learning"""
        commands = [
            "Evaluate the effectiveness of our analysis process and identify success patterns.",
            "Assess agent collaboration quality and information integration effectiveness.",
            "Review signal quality and predictive accuracy against market outcomes.",
            "Identify process improvements and develop recommendations for enhanced analysis."
        ]
        await self.execute_workflow_phase(channel, "PHASE 6: PROCESS IMPROVEMENT", "performance_analytics_and_refinement", commands, "Process optimized!")

    async def execute_chief_investment_officer_oversight(self, channel):
        """Phase 7: EXECUTIVE OVERSIGHT - Final decision authority"""
        commands = [
            "Based on comprehensive analysis, recommend EXECUTE, HOLD, or RESTART. Provide detailed rationale for the recommended course of action considering opportunity quality, risk assessment, and market conditions."
        ]
        await self.execute_workflow_phase(channel, "PHASE 7: EXECUTIVE OVERSIGHT", "chief_investment_officer_oversight", commands, "Decision made!")

        # Analyze responses for execution decision
        recent_responses = [r for r in self.responses_collected if r.get('phase') == 'chief_investment_officer_oversight']
        if recent_responses:
            # Simple decision logic - look for keywords
            last_response = recent_responses[-1]['response'].upper()
            if 'EXECUTE' in last_response:
                return "EXECUTE"
            elif 'HOLD' in last_response:
                return "HOLD"
            elif 'RESTART' in last_response:
                return "RESTART"
            else:
                return "PENDING"
        return "PENDING"

    async def execute_alpha_deployment(self, channel):
        """Execute the alpha deployment - actually place trades"""
        await channel.send("**TRADE EXECUTION INITIATED**")
        await channel.send("Executing approved trading strategies.")

        # Execute deployment commands
        deployment_commands = [
            "Execute approved trades and establish positions.",
            "Verify all trades executed successfully.",
            "Initialize position monitoring and tracking.",
            "Begin performance tracking for executed trades."
        ]

        max_wait_time = self.phase_delays.get("opening_bell_execution", 86400)
        await self._execute_commands_parallel_old(deployment_commands, channel, "alpha_deployment", max_wait_time)

    async def complete_continuous_workflow(self, channel):
        """Complete the continuous alpha discovery workflow"""
        await channel.send("**ANALYSIS SESSION COMPLETE**")

        # Get final stats
        total_responses = len(self.responses_collected)
        total_cycles = sum(1 for r in self.responses_collected if 'ALPHA HUNT CYCLE' in str(r.get('phase', '')))

        await channel.send("**Session Summary:**")
        await channel.send(f"• Analysis cycles: {total_cycles}")
        await channel.send(f"• Agent interactions: {total_responses} communications")
        await channel.send(f"• Human interventions: {len(self.human_interventions)}")
        await channel.send("• Architecture: Continuous collaboration with full context sharing")

        # Save results
        results = {
            'completed_at': datetime.now().astimezone().isoformat(),
            'mission_type': 'systematic_market_surveillance',
            'total_cycles': total_cycles,
            'total_responses': total_responses,
            'human_interventions': len(self.human_interventions),
            'responses': self.responses_collected,
            'interventions': self.human_interventions,
            'workflow_log': self.workflow_log
        }

        with open('data/systematic_market_surveillance_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        await channel.send("Results saved to `data/systematic_market_surveillance_results.json`")
        await channel.send("Ready for next analysis session. Type `!start_workflow` to begin.")

    # HELPER METHODS FOR CONTINUOUS WORKFLOW

    async def _execute_commands_parallel_old(self, commands, channel, phase_key, max_wait_time):
        """Execute commands sequentially and display responses below each command like a train of thought"""
        await channel.send(f"Processing {len(commands)} commands sequentially...")

        for i, command in enumerate(commands, 1):
            if not self.workflow_active:
                return

            target_channel = self.get_command_channel(command)
            channel_name = target_channel.name if target_channel else "general"

            await channel.send(f"\n**Command {i}/{len(commands)}: {command}** → #{channel_name}")

            if target_channel:
                await target_channel.send(command)

            # Wait for and display responses for this specific command
            start_time = time.time()
            initial_count = len(self.responses_collected)
            displayed_responses = 0

            await channel.send(f"⏳ Waiting for agent responses to this command... (up to {max_wait_time}s)")

            while time.time() - start_time < max_wait_time and self.workflow_active:
                await asyncio.sleep(2)

                # Get new responses that match this command
                new_responses = [
                    r for r in self.responses_collected[initial_count:]
                    if r.get('phase') == phase_key and r.get('command', '') == command
                ]

                # Display new responses immediately below the command
                for response_data in new_responses:
                    agent_name = response_data.get('agent', 'Unknown')
                    response = response_data.get('response', '')

                    await channel.send(f"**📥 Response from {agent_name.title()} Agent:**")

                    if isinstance(response, dict):
                        for key, value in response.items():
                            if key not in ['timestamp', 'agent_role']:
                                formatted_value = self._format_response_value(key, value)
                                emoji_map = {
                                    'trade_proposals': '📈',
                                    'confidence_level': '🎯',
                                    'risk_assessment': '⚠️',
                                    'analysis': '🔍',
                                    'recommendations': '💡'
                                }
                                emoji = emoji_map.get(key, '•')
                                await channel.send(f"{emoji} **{key.replace('_', ' ').title()}:** {formatted_value}")
                    else:
                        sanitized_response = self._sanitize_agent_output(str(response))
                        segments = self._break_into_logical_segments(sanitized_response)
                        for segment in segments:
                            if segment.strip():
                                await channel.send(segment)

                    displayed_responses += 1

                if displayed_responses > 0:
                    # If we got responses, wait a bit longer for more
                    await asyncio.sleep(5)

            # Announce completion for this command
            if displayed_responses > 0:
                await channel.send(f"✅ Responses collected for command {i}: {displayed_responses}")
            else:
                await channel.send(f"⏰ No responses received for command {i} within {max_wait_time}s")

        await channel.send("✅ All commands processed sequentially")

    async def _execute_single_command(self, command, channel, phase_key, max_wait_time):
        """Execute a single command and display responses in real-time"""
        target_channel = self.get_command_channel(command)
        channel_name = target_channel.name if target_channel else "general"

        await channel.send(f"Supervision Command: {command} → #{channel_name}")

        if target_channel:
            await target_channel.send(command)

        # Wait for response and display in real-time
        await channel.send(f"⏳ Waiting up to {max_wait_time}s for supervision decision...")

        start_time = time.time()
        initial_count = len(self.responses_collected)
        displayed_responses = 0

        while time.time() - start_time < max_wait_time and self.workflow_active:
            await asyncio.sleep(2)  # Check frequently for real-time updates

            # Get new responses for this phase
            phase_responses = [r for r in self.responses_collected if r.get('phase') == phase_key]
            new_responses = phase_responses[displayed_responses:]

            # Display new responses immediately
            for response_data in new_responses:
                agent_name = response_data.get('agent', 'Unknown')
                response = response_data.get('response', '')

                # Display response in Discord immediately
                await channel.send(f"**📥 {agent_name.title()} Agent Decision:**")

                # Format and display the response
                if isinstance(response, dict):
                    # Handle structured responses
                    for key, value in response.items():
                        if key not in ['timestamp', 'agent_role']:
                            formatted_value = self._format_response_value(key, value)
                            emoji_map = {
                                'decision': '🎯',
                                'rationale': '💡',
                                'recommendation': '📋',
                                'analysis': '🔍'
                            }
                            emoji = emoji_map.get(key, '•')
                            await channel.send(f"{emoji} **{key.replace('_', ' ').title()}:** {formatted_value}")
                else:
                    # Handle string responses
                    sanitized_response = self._sanitize_agent_output(str(response))
                    # Break into logical segments for better readability
                    segments = self._break_into_logical_segments(sanitized_response)
                    for segment in segments:
                        if segment.strip():
                            await channel.send(segment)

                displayed_responses += 1

        final_count = len([r for r in self.responses_collected if r.get('phase') == phase_key])
        if final_count > initial_count:
            await channel.send("✅ **Supervision decision received and displayed**")
        else:
            await channel.send(f"⏰ **No supervision response received within {max_wait_time}s timeout**")

    async def request_consensus(self, question: str, requesting_agent: str, target_agents: List[str], timeout_seconds: int = 300) -> str:
        """Allow agents to request consensus polls for risk/strategy decisions"""
        try:
            # Validate requesting agent can make requests
            allowed_requesters = ["risk_agent", "strategy_agent", "execution_agent"]
            if requesting_agent not in allowed_requesters:
                raise ValueError(f"Agent {requesting_agent} not authorized to request consensus")

            # Validate target agents exist
            valid_agents = ["risk_agent", "strategy_agent", "data_agent", "execution_agent", "reflection_agent"]
            invalid_agents = [agent for agent in target_agents if agent not in valid_agents]
            if invalid_agents:
                raise ValueError(f"Invalid target agents: {invalid_agents}")

            # Create the poll
            poll_id = await self.consensus_poller.create_poll(
                question=f"[{requesting_agent}] {question}",
                agents_to_poll=target_agents,
                timeout_seconds=timeout_seconds,
                metadata={"requesting_agent": requesting_agent, "agent_requested": True}
            )

            # Start the poll
            success = await self.consensus_poller.start_poll(poll_id)
            if not success:
                raise RuntimeError("Failed to start consensus poll")

            logger.info(f"Agent {requesting_agent} requested consensus poll {poll_id}: {question}")
            return poll_id

        except Exception as e:
            logger.error(f"Error in agent consensus request from {requesting_agent}: {e}")
            raise

    async def _handle_consensus_state_change(self, poll):
        """Handle consensus poll state changes and status updates, notify Discord"""
        try:
            if not self.channel:
                return

            is_status_update = poll.metadata.get("status_update", False)

            if is_status_update:
                # Status update during voting - less prominent
                embed = discord.Embed(
                    title="🤝 Consensus Poll Progress",
                    description=f"**Question:** {poll.question}",
                    color=0xffa500,  # Orange for in-progress
                    timestamp=datetime.now()
                )
                embed.set_footer(text="Periodic status update")
            else:
                # State change - more prominent
                embed = discord.Embed(
                    title="🤝 Consensus Poll Update",
                    description=f"**Question:** {poll.question}",
                    color=0x00ff00 if poll.state.value == "consensus_reached" else 0xffa500 if poll.state.value == "voting" else 0xff0000,
                    timestamp=datetime.now()
                )

            embed.add_field(name="Poll ID", value=poll.poll_id, inline=True)
            embed.add_field(name="Status", value=poll.state.value.replace("_", " ").title(), inline=True)
            embed.add_field(name="Total Votes", value=str(poll.total_votes), inline=True)

            if poll.state.value == "consensus_reached":
                embed.add_field(name="Consensus", value=f"**{poll.consensus_vote}** ({poll.consensus_confidence:.1%} confidence)", inline=False)
                embed.add_field(name="Supporting Agents", value=", ".join(poll.supporting_agents), inline=False)
            elif poll.state.value == "timeout":
                embed.add_field(name="Result", value="⏰ Poll timed out without consensus", inline=False)
            elif poll.state.value == "failed":
                embed.add_field(name="Error", value=poll.metadata.get("error", "Unknown error"), inline=False)

            # Show vote breakdown if available
            if poll.votes:
                vote_summary = []
                responded_count = 0
                for agent, vote_data in poll.votes.items():
                    if vote_data.get("status") == "responded":
                        vote = vote_data.get("vote", "unknown")
                        confidence = vote_data.get("confidence", 0)
                        vote_summary.append(f"{agent}: {vote} ({confidence:.1%})")
                        responded_count += 1
                    else:
                        vote_summary.append(f"{agent}: pending")

                total_agents = len(poll.votes)
                embed.add_field(name="Progress", value=f"{responded_count}/{total_agents} agents responded", inline=True)

                if vote_summary and (not is_status_update or responded_count > 0):  # Show votes for state changes or if there are responses
                    embed.add_field(name="Current Votes", value="\n".join(vote_summary[:5]), inline=False)  # Limit to 5 for embed size

            # Only send status updates if there has been some progress (responses received)
            if is_status_update and responded_count == 0:
                return  # Skip status updates with no progress

            await self.channel.send(embed=embed)

        except Exception as e:
            logger.error(f"Error handling consensus state change: {e}")

    async def run_orchestrator(self):
        """Run the live workflow orchestrator"""
        print("🎯 Starting Live Workflow Orchestrator...")
        print("📋 Commands available in Discord:")
        print("  !start_workflow              - Begin full analysis workflow")
        print("  !start_market_open_execution - Fast-track execution at market open")
        print("  !start_trade_monitoring      - Start position monitoring workflow")
        print("  !pause_workflow              - Pause current workflow")
        print("  !resume_workflow             - Resume paused workflow")
        print("  !stop_workflow               - Stop current workflow")
        print("  !stop_monitoring             - Stop monitoring workflow")
        print("  !workflow_status             - Get current status")
        print("💡 You can ask questions or intervene at any time!")
        print()
        
        while True:  # Keep trying to reconnect
            try:
                print("🔧 Initializing Discord client...")
                try:
                    await self.initialize_discord_client()
                except Exception as e:
                    alert_manager.critical(e, {"component": "discord_initialization"}, "orchestrator")

                token = get_vault_secret('DISCORD_ORCHESTRATOR_TOKEN')
                if not token:
                    raise ValueError("❌ DISCORD_ORCHESTRATOR_TOKEN not found")
                if not self.client:
                    raise ValueError("❌ Discord client not initialized")
                print("🚀 Starting Discord client...")
                max_retries = 5
                retry_delay = 5
                for attempt in range(max_retries):
                    try:
                        await self.client.start(token)
                        break
                    except Exception as e:
                        print(f"❌ Connection attempt {attempt+1}/{max_retries} failed: {e}")
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"🔄 Retrying in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                        else:
                            raise
            except KeyboardInterrupt:
                print("\n🛑 Orchestrator shutting down...")
                if self.client:
                    await self.client.close()
                break
            except Exception as e:
                print(f"❌ Orchestrator error: {e}")
                import traceback
                traceback.print_exc()
                if self.client:
                    await self.client.close()
                print("🔄 Attempting to reconnect in 5 seconds...")
                await asyncio.sleep(5)

async def main():
    """Run the live workflow orchestrator"""
    print("🎯 Starting Live Workflow Orchestrator...")
    print("📋 Commands available in Discord:")
    print("  !start_workflow              - Begin full analysis workflow")
    print("  !start_market_open_execution - Fast-track execution at market open")
    print("  !start_trade_monitoring      - Start position monitoring workflow")
    print("  !pause_workflow              - Pause current workflow")
    print("  !resume_workflow             - Resume paused workflow")
    print("  !stop_workflow               - Stop current workflow")
    print("  !stop_monitoring             - Stop monitoring workflow")
    print("  !workflow_status             - Get current status")
    print("💡 You can ask questions or intervene at any time!")

    print("🔍 Checking Discord token...")
    token = get_vault_secret('DISCORD_ORCHESTRATOR_TOKEN')
    if token:
        print(f"✅ Discord token found (length: {len(token)})")
    else:
        print("❌ Discord token not found")

    # Start component health monitoring
    print("🏥 Starting component health monitoring...")
    try:
        from src.utils.component_health_monitor import start_component_health_monitoring
        start_component_health_monitoring(check_interval=60)  # Check every minute
        print("✅ Component health monitoring started")
    except Exception as e:
        print(f"⚠️  Failed to start component health monitoring: {e}")

    orchestrator = LiveWorkflowOrchestrator()
    await orchestrator.run_orchestrator()

if __name__ == "__main__":
    asyncio.run(main())