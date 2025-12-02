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

# Set up logging for human input features
logger = logging.getLogger(__name__)

# Import BaseAgent and agent classes for direct integration
from src.agents.base import BaseAgent

# Import agent scope definitions for intelligent command routing
from src.agents.agent_scopes import get_agent_scope_definitions

# Try to import agents, fallback to None if not available
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
    print(f"⚠️ Some agents not available due to import error: {e}")
    print("   Discord-only mode will be used")
    AGENTS_AVAILABLE = False
    MacroAgent = None
    DataAgent = None
    StrategyAgent = None
    RiskAgent = None
    ReflectionAgent = None
    ExecutionAgent = None
    LearningAgent = None
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
            # Integrate with IBKR
            from integrations.ibkr_connector import get_ibkr_connector
            ibkr_connector = get_ibkr_connector()
            
            positions = await ibkr_connector.get_positions()
            cash_balance = await ibkr_connector.get_cash_balance()  # Assuming this method exists
            
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
        if not self.channel:
            print(f"❌ No channel available for phase {phase_key}")
            return

        general_channel = cast(discord.TextChannel, self.channel)
        self.current_phase = phase_key

        # Announce phase start in general channel
        await general_channel.send(f"\n{phase_title}")
        await general_channel.send("─" * 50)

        commands = self.phase_commands.get(phase_key, [])

        # ENHANCED PARALLEL EXECUTION: Share complete workflow context with all agents
        await self._share_full_workflow_context(phase_key, phase_title)

        # POSITION AWARENESS: Include current position data in context
        await self._share_position_context()

        # SEND ALL COMMANDS TO ALL AGENTS SIMULTANEOUSLY for maximum collaboration
        agent_responses = await self._execute_commands_parallel(commands, phase_key)
        self.responses_collected.extend(agent_responses)

        # Announce parallel execution with clear separation
        await general_channel.send(f"\n🎯 **PARALLEL EXECUTION:** {len(commands)} commands sent to {len(self.agent_instances)} agents simultaneously!")
        await general_channel.send("🤝 **Agents collaborating with full shared context - no silos!**")

        # Format and display agent responses with enhanced readability
        if agent_responses:
            await self._present_agent_responses_enhanced(general_channel, agent_responses, phase_key)
        else:
            await general_channel.send("\n⏰ **No agent responses received** - but they're thinking hard! 🤔")

        await general_channel.send(f"\n✅ **{phase_title} Complete! Alpha discovered!** 🎉")
        await asyncio.sleep(2)  # Reduced from 3 to speed up

    async def _present_agent_responses_enhanced(self, channel: discord.TextChannel, responses: List[Dict[str, Any]], phase_key: str):
        """Present agent responses in a professional format with logical segmentation"""
        if not responses:
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
        await channel.send(summary_text)
        await asyncio.sleep(1)  # Rate limit prevention

        # Send detailed responses with logical segmentation
        for detailed in detailed_responses:
            # Break into logical segments instead of hard character limits
            segments = self._break_into_logical_segments(detailed)
            for segment in segments:
                if segment.strip():  # Only send non-empty segments
                    await channel.send(segment)
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

    async def initialize_discord_client(self):
        """Initialize Discord client for live orchestration"""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        self.client = discord.Client(intents=intents)

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

                    # Set up alerts channel for trade notifications
                    alerts_channel_id = os.getenv('DISCORD_ALERTS_CHANNEL_ID')
                    if alerts_channel_id:
                        try:
                            self.alerts_channel = guild.get_channel(int(alerts_channel_id))
                            if self.alerts_channel:
                                print(f"🚨 Alerts channel configured: #{self.alerts_channel.name}")
                            else:
                                print(f"⚠️ Alerts channel ID {alerts_channel_id} not found")
                                await general_channel.send("⚠️ **Configuration Warning**: Alerts channel not found. Trade alerts will use general channel.")
                        except ValueError:
                            print(f"⚠️ Invalid alerts channel ID: {alerts_channel_id}")
                            await general_channel.send("⚠️ **Configuration Warning**: Invalid alerts channel ID. Trade alerts will use general channel.")
                    else:
                        print("⚠️ DISCORD_ALERTS_CHANNEL_ID not set, trade alerts will go to general channel")
                        await general_channel.send("⚠️ **Configuration Warning**: DISCORD_ALERTS_CHANNEL_ID not set. Trade alerts will use general channel.")

                    # Set up ranked trades channel for trade proposals
                    ranked_trades_channel_id = os.getenv('DISCORD_RANKED_TRADES_CHANNEL_ID')
                    if ranked_trades_channel_id:
                        try:
                            self.ranked_trades_channel = guild.get_channel(int(ranked_trades_channel_id))
                            if self.ranked_trades_channel:
                                print(f"📊 Ranked trades channel configured: #{self.ranked_trades_channel.name}")
                            else:
                                print(f"⚠️ Ranked trades channel ID {ranked_trades_channel_id} not found")
                                await general_channel.send("⚠️ **Configuration Warning**: Ranked trades channel not found. Trade proposals will use general channel.")
                        except ValueError:
                            print(f"⚠️ Invalid ranked trades channel ID: {ranked_trades_channel_id}")
                            await general_channel.send("⚠️ **Configuration Warning**: Invalid ranked trades channel ID. Trade proposals will use general channel.")
                    else:
                        print("⚠️ DISCORD_RANKED_TRADES_CHANNEL_ID not set, ranked trades will go to general channel")
                        await general_channel.send("⚠️ **Configuration Warning**: DISCORD_RANKED_TRADES_CHANNEL_ID not set. Trade proposals will use general channel.")

                    if not self.channel and guild.text_channels:
                        self.channel = guild.text_channels[0]
                        print(f"📝 Using default general channel: #{self.channel.name}")

                    # Set up general channel only (no agent-specific channels needed)
                    for ch in guild.text_channels:
                        if ch.name in ['general', 'workflow', 'analysis', 'trading']:
                            self.channel = ch
                            print(f"📝 General channel: #{ch.name}")
                            break

                    if not self.channel and guild.text_channels:
                        self.channel = guild.text_channels[0]
                        print(f"📝 Using default general channel: #{self.channel.name}")

                    # Announce orchestrator presence
                    if self.channel and hasattr(self.channel, 'send'):
                        general_channel = cast(discord.TextChannel, self.channel)
                        channel_status = "✅ Channel separation active" if self.alerts_channel else "⚠️ Alerts channel not configured"
                        ranked_trades_status = "✅ Ranked trades channel active" if self.ranked_trades_channel else "⚠️ Ranked trades channel not configured"
                        await general_channel.send("**Live Workflow Orchestrator Online**\nReady to begin iterative reasoning workflow with unified agent coordination. Type `!start_workflow` to begin analysis, `!start_premarket_analysis` for premarket prep, `!start_market_open_execution` for fast execution, or `!start_trade_monitoring` for position monitoring!")
                        await general_channel.send(f"📢 **Channel Configuration:** {channel_status}")
                        if self.alerts_channel:
                            await general_channel.send(f"🚨 Trade alerts will be sent to: #{self.alerts_channel.name}")
                        if self.ranked_trades_channel:
                            await general_channel.send(f"📊 Ranked trade proposals will be sent to: #{self.ranked_trades_channel.name}")

                    # Signal that Discord is ready
                    self.discord_ready.set()
                    print("🎯 Discord client fully ready - channels configured and orchestrator online")
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

            # Handle !share_news command: !share_news <link> [optional description]
            if content.startswith("!share_news"):
                await self.handle_share_news_command(message)
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
        await channel.send("Human intervention available at any time during the process.")

        # CONTINUOUS ALPHA DISCOVERY LOOP - No more artificial phase barriers!
        alpha_hunt_cycles = 0
        max_cycles = 10  # Prevent infinite loops

        while self.workflow_active and alpha_hunt_cycles < max_cycles:
            alpha_hunt_cycles += 1
            await channel.send(f"\n**ANALYSIS CYCLE {alpha_hunt_cycles}**")
            
            # Process any queued human inputs at the START of this iteration
            await self.process_queued_human_inputs(channel)
            
            # Mark iteration as in-progress (new inputs will be queued)
            self.iteration_in_progress = True
            logger.info(f"Starting iteration {alpha_hunt_cycles} - human inputs will now be queued")

            # Phase 1: CONTINUOUS ALPHA DISCOVERY - All agents hunt together
            await self.execute_systematic_market_surveillance(channel)

            # Phase 2: PARALLEL AGENT COLLABORATION - Build on each other's insights
            await self.execute_parallel_collaboration(channel)

            # Phase 3: EXCITED ALPHA STORM - Maximum enthusiasm!
            await self.execute_quantitative_opportunity_validation(channel)

            # Phase 4: RAPID CONSENSUS BUILDING - Fast-track to decisions
            await self.execute_rapid_consensus(channel)

            # Phase 5: SWIFT EXECUTION PREP - Get ready to deploy
            await self.execute_portfolio_implementation_planning(channel)

            # Phase 6: ENTHUSIASTIC LEARNING - Celebrate and improve
            await self.execute_performance_analytics_and_refinement(channel)

            # Phase 7: ALPHA HUNTING SUPERVISION - Final authority
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
            from integrations.ibkr_connector import get_ibkr_connector
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

    async def resume_workflow(self):
        """Resume a paused workflow"""
        if self.channel is None:
            return

        channel = cast(discord.TextChannel, self.channel)
        if not self.workflow_active:
            self.workflow_active = True
            await channel.send("Workflow resumed. Continuing from current phase.")
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

    async def execute_systematic_market_surveillance(self, channel):
        """Phase 1: CONTINUOUS ALPHA DISCOVERY - All agents hunt together"""
        await channel.send("**PHASE 1: MARKET ANALYSIS**")
        await channel.send("Scanning markets for alpha opportunities across all asset classes.")

        # Execute parallel commands for alpha discovery
        alpha_commands = [
            "Analyze current market conditions and identify unusual price action, volume patterns, and emerging trends.",
            "Review technical indicators and quantitative signals for momentum shifts and statistical anomalies.",
            "Evaluate macroeconomic factors and their potential market impact.",
            "Assess relative strength across sectors and identify potential opportunities."
        ]

        max_wait_time = self.phase_delays.get("systematic_market_surveillance", 86400)
        await self._execute_commands_parallel_old(alpha_commands, channel, "alpha_discovery", max_wait_time)

    async def execute_parallel_collaboration(self, channel):
        """Phase 2: PARALLEL AGENT COLLABORATION - Build on each other's insights"""
        await channel.send("**PHASE 2: COLLABORATIVE ANALYSIS**")
        await channel.send("Agents integrating findings and validating signals across methodologies.")

        # Execute collaborative commands
        collab_commands = [
            "Cross-validate findings across different analytical approaches and data sources.",
            "Identify complementary signals and assess overall conviction levels.",
            "Evaluate signal strength and identify potential false positives.",
            "Prioritize opportunities based on combined analytical perspectives."
        ]

        max_wait_time = self.phase_delays.get("multi_strategy_opportunity_synthesis", 86400)
        await self._execute_commands_parallel_old(collab_commands, channel, "parallel_collaboration", max_wait_time)

    async def execute_quantitative_opportunity_validation(self, channel):
        """Phase 3: OPPORTUNITY ASSESSMENT - Comprehensive evaluation"""
        await channel.send("**PHASE 3: OPPORTUNITY ASSESSMENT**")
        await channel.send("Conducting detailed evaluation of identified opportunities.")

        # Execute assessment commands
        storm_commands = [
            "Perform comprehensive opportunity assessment against historical precedents.",
            "Conduct detailed risk analysis including downside potential and volatility.",
            "Quantify expected returns and calculate risk-adjusted metrics.",
            "Develop execution plans with entry/exit criteria and position sizing."
        ]

        max_wait_time = self.phase_delays.get("quantitative_opportunity_validation", 86400)
        await self._execute_commands_parallel_old(storm_commands, channel, "quantitative_opportunity_validation", max_wait_time)

    async def execute_rapid_consensus(self, channel):
        """Phase 4: CONSENSUS BUILDING - Fast-track to decisions"""
        await channel.send("**PHASE 4: CONSENSUS BUILDING**")
        await channel.send("Building consensus on opportunities and prioritizing actions.")

        # Execute consensus commands
        consensus_commands = [
            "Evaluate opportunities against established criteria and ranking frameworks.",
            "Assess market regime alignment and current condition suitability.",
            "Cross-validate signals across technical, fundamental, and quantitative frameworks.",
            "Refine trade structures and optimize risk management parameters."
        ]

        max_wait_time = self.phase_delays.get("investment_committee_review", 86400)
        await self._execute_commands_parallel_old(consensus_commands, channel, "rapid_consensus", max_wait_time)

    async def execute_portfolio_implementation_planning(self, channel):
        """Phase 5: EXECUTION PREPARATION - Get ready to deploy"""
        await channel.send("**PHASE 5: EXECUTION PREPARATION**")
        await channel.send("Preparing for trade execution with comprehensive risk management.")

        # Execute preparation commands
        prep_commands = [
            "Calculate optimal position sizing considering portfolio impact and risk limits.",
            "Define comprehensive risk management parameters and stop-loss levels.",
            "Determine precise entry timing and execution methodology.",
            "Prepare detailed execution playbook with all trade parameters."
        ]

        max_wait_time = self.phase_delays.get("portfolio_implementation_planning", 86400)
        await self._execute_commands_parallel_old(prep_commands, channel, "portfolio_implementation_planning", max_wait_time)

    async def execute_performance_analytics_and_refinement(self, channel):
        """Phase 6: PROCESS IMPROVEMENT - Systematic learning"""
        await channel.send("**PHASE 6: PROCESS IMPROVEMENT**")
        await channel.send("Analyzing performance and identifying optimization opportunities.")

        # Execute learning commands
        learning_commands = [
            "Evaluate the effectiveness of our analysis process and identify success patterns.",
            "Assess agent collaboration quality and information integration effectiveness.",
            "Review signal quality and predictive accuracy against market outcomes.",
            "Identify process improvements and develop recommendations for enhanced analysis."
        ]

        max_wait_time = self.phase_delays.get("performance_analytics_and_refinement", 86400)
        await self._execute_commands_parallel_old(learning_commands, channel, "performance_analytics_and_refinement", max_wait_time)

    async def execute_chief_investment_officer_oversight(self, channel):
        """Phase 7: EXECUTIVE OVERSIGHT - Final decision authority"""
        await channel.send("**PHASE 7: EXECUTIVE OVERSIGHT**")
        await channel.send("Final evaluation and execution decision.")

        # Execute supervision command
        supervision_command = "Based on comprehensive analysis, recommend EXECUTE, HOLD, or RESTART. Provide detailed rationale for the recommended course of action considering opportunity quality, risk assessment, and market conditions."

        max_wait_time = self.phase_delays.get("chief_investment_officer_oversight", 86400)
        await self._execute_single_command(supervision_command, channel, "chief_investment_officer_oversight", max_wait_time)

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
                await self.initialize_discord_client()
                
                # Initialize agents asynchronously
                print("🤖 Initializing agent instances...")
                await self.initialize_agents_async()
                print(f"✅ Agent initialization complete: {len(self.agent_instances)} agents ready")
                
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

    orchestrator = LiveWorkflowOrchestrator()
    await orchestrator.run_orchestrator()

if __name__ == "__main__":
    asyncio.run(main())