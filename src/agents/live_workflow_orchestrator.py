#!/usr/bin/env python3
"""
Live Workflow Orchestrator - Real-time Interactive Iterative Reasoning
Watches Discord and orchestrates the collaborative reasoning workflow automatically,
while allowing human intervention and questions during the process.

Enhanced to support both Discord-based operation and direct agent method calls
for improved reliability, testing, and integration with BaseAgent architecture.
"""

import asyncio
import discord
import os
import time
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, cast
from dotenv import load_dotenv

# Import BaseAgent and agent classes for direct integration
from src.agents.base import BaseAgent

# Try to import agents, fallback to None if not available
try:
    from src.agents.macro import MacroAgent
    from src.agents.data import DataAgent
    from src.agents.strategy import StrategyAgent
    # Temporarily skip RiskAgent due to TensorFlow import issues
    # from src.agents.risk import RiskAgent
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
    RiskAgent = None  # Explicitly set to None
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
        self.agent_channels = {}  # Map agent types to their channels
        
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
        
        # Phase timing configuration
        self.phase_delays = {
            'macro_foundation_data_collection': 300,  # Maximum wait time (300s = 5min) for data collection
            'macro_foundation_analysis': 300,  # Maximum wait time (300s = 5min) for macro analysis responses
            'intelligence_gathering': 240,  # Maximum wait time (240s = 4min) per command
            'strategy_development': 300,  # Maximum wait time (300s = 5min) for strategy development
            'debate': 360,  # Maximum wait time (360s = 6min) for debates
            'risk_assessment': 240,  # Maximum wait time (240s = 4min) for risk assessment
            'consensus': 300,  # Maximum wait time (300s = 5min) for consensus building
            'execution_validation': 240,  # Maximum wait time (240s = 4min) for execution validation
            'learning': 240,  # Maximum wait time (240s = 4min) for learning
            'executive_review': 420,  # Maximum wait time (420s = 7min) for enhanced iteration 2 analysis
            'supreme_oversight': 480  # Maximum wait time (480s = 8min) for final decisions
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
        """Asynchronously initialize direct agent instances for enhanced integration"""
        if not AGENTS_AVAILABLE:
            print("⚠️ Agent classes not available - running in Discord-only mode")
            return
            
        print("🤖 Starting agent initialization (this may take 20-30 seconds)...")
        start_time = time.time()
            
        try:
            # Initialize each agent with A2A protocol
            agent_classes = {
                'macro': MacroAgent,
                'data': DataAgent,
                'strategy': StrategyAgent,
                # 'risk': RiskAgent,  # Temporarily disabled due to TensorFlow issues
                'reflection': ReflectionAgent,
                'execution': ExecutionAgent,
                'learning': LearningAgent
            }

            for agent_key, agent_class in agent_classes.items():
                if agent_class is None:
                    print(f"⚠️ Skipping {agent_key} agent - class not available")
                    continue
                    
                try:
                    print(f"🔧 Initializing {agent_key} agent...")
                    agent_instance = agent_class(a2a_protocol=self.a2a_protocol)
                    
                    # Initialize LLM asynchronously
                    await agent_instance.async_initialize_llm()
                    
                    self.agent_instances[agent_key] = agent_instance
                    self.a2a_protocol.register_agent(agent_key, agent_instance)
                    print(f"✅ Initialized {agent_key} agent with A2A protocol")
                except Exception as e:
                    print(f"⚠️ Failed to initialize {agent_key} agent: {e}")
                    
        except Exception as e:
            print(f"⚠️ Agent initialization failed: {e}")
            # Continue without direct agents - Discord-only mode
            
        end_time = time.time()
        print(f"🎯 Agent initialization completed in {end_time - start_time:.1f} seconds")
        self._initialize_agents()

    def get_command_channel(self, command: str) -> Optional[discord.TextChannel]:
        """Get the appropriate channel for a command based on agent type"""
        # Map command prefixes to agent types
        prefix_to_agent = {
            '!m': 'macro',
            '!d': 'data',
            '!s': 'strategy', 
            '!r': 'risk',
            '!ref': 'reflection',
            '!exec': 'execution',
            '!l': 'learning'
        }
        
        # Extract prefix from command
        prefix = command.split()[0].lower() if command.split() else ''
        agent_type = prefix_to_agent.get(prefix)
        
        # Special handling for debate commands
        if 'debate' in command.lower():
            if 'debates' in self.agent_channels:
                return self.agent_channels['debates']
        
        # Return agent-specific channel if available, otherwise general channel
        if agent_type and agent_type in self.agent_channels:
            return self.agent_channels[agent_type]
        else:
            return self.channel

    async def send_trade_alert(self, alert_message: str, alert_type: str = "trade"):
        """Send a trade alert to the dedicated alerts channel"""
        if not self.alerts_channel:
            print(f"⚠️ Alerts channel not available, sending to general channel")
            if self.channel:
                await self.channel.send(f"🚨 **TRADE ALERT** 🚨\n{alert_message}")
            return

        try:
            # Format alert with appropriate emoji based on type
            emoji_map = {
                "trade": "🚨",
                "execution": "✅",
                "risk": "⚠️",
                "error": "❌",
                "success": "🎯"
            }
            emoji = emoji_map.get(alert_type, "🚨")
            
            formatted_alert = f"{emoji} **TRADE ALERT** {emoji}\n{alert_message}"
            await self.alerts_channel.send(formatted_alert)
            print(f"✅ Trade alert sent to alerts channel: {alert_type}")
        except Exception as e:
            print(f"⚠️ Failed to send trade alert: {e}")
            # Fallback to general channel
            if self.channel:
                await self.channel.send(f"🚨 **TRADE ALERT** 🚨\n{alert_message}")

    def is_trade_related_message(self, message: str) -> bool:
        """Determine if a message contains trade-related content that should go to alerts"""
        trade_keywords = [
            'trade proposal', 'trade execution', 'buy signal', 'sell signal',
            'entry point', 'exit point', 'position sizing', 'stop loss',
            'take profit', 'order placed', 'order filled', 'trade alert',
            'execution validation', 'trade recommendation'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in trade_keywords)

    def _extract_trade_alert_info(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Extract key trade information from agent response for alerts"""
        try:
            agent_name = response_data.get('agent', 'Unknown')
            response = response_data.get('response', {})
            
            alert_parts = []
            
            if isinstance(response, dict):
                # Look for trade proposals in structured responses
                if 'trade_proposals' in response:
                    proposals = response['trade_proposals']
                    if isinstance(proposals, list) and proposals:
                        alert_parts.append(f"**{agent_name.title()} Agent** generated {len(proposals)} trade proposal(s):")
                        for i, proposal in enumerate(proposals[:3], 1):  # Limit to 3 proposals
                            if isinstance(proposal, dict):
                                instrument = proposal.get('instrument', 'Unknown')
                                action = proposal.get('action', 'Unknown')
                                confidence = proposal.get('confidence', 'Unknown')
                                alert_parts.append(f"• {action.upper()} {instrument} (Confidence: {confidence})")
                
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
                # For string responses, extract key trade info
                response_str = str(response)
                if 'trade proposal' in response_str.lower():
                    alert_parts.append(f"**{agent_name.title()} Agent** has trade proposals ready for review")
                elif 'execution' in response_str.lower():
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
        """Send command directly to agent using BaseAgent methods via A2A"""
        if agent_name not in self.agent_instances:
            print(f"⚠️ Agent {agent_name} not available for direct command")
            return None
            
        agent = self.agent_instances[agent_name]
        
        try:
            # Handle debate commands specially
            if 'debate' in command.lower():
                return await self._handle_debate_command(agent, command, data)
            
            # Strip agent prefix for generic analysis
            stripped_command = command
            prefixes = ["!d ", "!m ", "!s ", "!ref ", "!exec ", "!l "]
            for prefix in prefixes:
                if stripped_command.startswith(prefix):
                    stripped_command = stripped_command[len(prefix):]
                    break
            
            # Parse command for direct agent method calls
            if stripped_command.startswith('!analyze') or stripped_command.startswith('analyze'):
                # Extract analysis query
                query = stripped_command.replace('!analyze', '').replace('analyze', '').strip()
                if data:
                    query += f" {data}"
                    
                result = await agent.analyze(query)
                return result
                
            elif stripped_command.startswith('!status') or stripped_command.startswith('status'):
                result = await agent.get_status()
                return result
                
            else:
                # For other commands, use generic analyze method
                result = await agent.analyze(stripped_command)
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
        """Get current position data from trading platform"""
        try:
            # This would integrate with your IBKR or other trading platform
            raise NotImplementedError("Real trading platform integration required - no mock position data allowed in production")

        except Exception as e:
            print(f"⚠️ Failed to get position data: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'positions': [],
                'cash_balance': 0.0
            }

    async def _execute_command_with_full_context(self, command: str, phase_key: str) -> List[Dict[str, Any]]:
        """Execute command with all agents having full context"""
        agent_responses = []

        # Send command to ALL agents (not just target) so they can collaborate
        command_tasks = []
        for agent_name, agent in self.agent_instances.items():
            task = self._send_context_aware_command(agent_name, agent, command, phase_key)
            command_tasks.append(task)

        # Wait for all agent responses
        command_results = await asyncio.gather(*command_tasks, return_exceptions=True)

        # Process results
        for agent_name, result in zip(self.agent_instances.keys(), command_results):
            if isinstance(result, Exception):
                print(f"⚠️ Agent {agent_name} failed: {result}")
                continue

            if result:
                agent_responses.append({
                    'agent': agent_name,
                    'method': 'a2a_enhanced',
                    'response': result,
                    'phase': phase_key,
                    'command': command
                })
                print(f"✅ Enhanced A2A response from {agent_name}")

        return agent_responses

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
        """Execute a phase using enhanced A2A protocol - agents share full workflow context and collaborate"""
        if not self.channel:
            print(f"❌ No channel available for phase {phase_key}")
            return

        general_channel = cast(discord.TextChannel, self.channel)
        self.current_phase = phase_key

        # Announce phase start
        await general_channel.send(f"\n{phase_title}")
        await general_channel.send("─" * 50)

        commands = self.phase_commands.get(phase_key, [])

        # ENHANCED CONTEXT SHARING: Share complete workflow context with all agents
        await self._share_full_workflow_context(phase_key, phase_title)

        # POSITION AWARENESS: Include current position data in context
        await self._share_position_context()

        for i, command in enumerate(commands, 1):
            if not self.workflow_active:
                return

            # Send command to ALL agents with full context, not just target agent
            agent_responses = await self._execute_command_with_full_context(command, phase_key)

            # Announce command and show responses
            await general_channel.send(f"📤 **Command {i}/{len(commands)}:** `{command}`")

            # Format and display agent responses
            if agent_responses:
                await self._present_agent_responses(general_channel, agent_responses, i)
            else:
                await general_channel.send("⏰ **No agent responses received**")

        await general_channel.send(f"✅ **{phase_title} Complete!**")
        await asyncio.sleep(3)

    async def _present_agent_responses(self, channel: discord.TextChannel, responses: List[Dict[str, Any]], command_index: int):
        """Present agent responses in a unified, readable format with proper Discord markdown"""
        if not responses:
            return

        # Group responses by agent
        agent_summaries = []
        detailed_responses = []

        for response_data in responses:
            agent_name = response_data['agent']
            response = response_data['response']

            # SECURITY: Sanitize agent output before display
            sanitized_response = self._sanitize_agent_output(response)

            # Create agent summary
            if isinstance(sanitized_response, dict):
                # Handle structured agent responses
                response_dict = cast(Dict[str, Any], sanitized_response)
                if 'analysis_type' in response_dict:
                    summary = f"**{agent_name.title()} Agent Analysis:** {response_dict.get('analysis_type', 'general')}"
                    if 'confidence_level' in response_dict:
                        summary += f" (Confidence: {response_dict['confidence_level']})"
                elif 'error' in response_dict:
                    summary = f"**{agent_name.title()} Agent:** Error - {response_dict['error']}"
                else:
                    summary = f"**{agent_name.title()} Agent:** Analysis complete"
            else:
                # Handle string responses
                summary = f"**{agent_name.title()} Agent:** {str(sanitized_response)[:100]}..."

            agent_summaries.append(summary)

            # Prepare detailed response with proper formatting
            if isinstance(sanitized_response, dict):
                detailed = f"🤖 **{agent_name.title()} Agent Details:**\n"
                for key, value in sanitized_response.items():
                    if key not in ['timestamp', 'agent_role']:  # Skip metadata
                        formatted_value = self._format_response_value(key, value)
                        detailed += f"• **{key.replace('_', ' ').title()}:** {formatted_value}\n"
                detailed_responses.append(detailed.rstrip())
            else:
                # Format string responses with potential table/chart detection
                formatted_response = self._format_text_response(str(sanitized_response))
                detailed_responses.append(f"🤖 **{agent_name.title()} Agent:** {formatted_response}")

        # Send summary first
        summary_text = f"📊 **Agent Responses Summary:**\n" + "\n".join(f"• {summary}" for summary in agent_summaries)
        await channel.send(summary_text)

        # Send detailed responses with proper chunking
        for detailed in detailed_responses:
            # Split long messages if needed (Discord limit is 2000 chars)
            if len(detailed) > 1900:
                chunks = self._smart_chunk_message(detailed, 1900)
                for chunk in chunks:
                    await channel.send(chunk)
            else:
                await channel.send(detailed)

        # Check for trade-related content and send alerts
        for response_data in responses:
            response = response_data['response']
            if isinstance(response, dict):
                # Check structured responses for trade content
                response_text = str(response)
            else:
                response_text = str(response)
            
            if self.is_trade_related_message(response_text):
                # Extract key trade information for alert
                alert_content = self._extract_trade_alert_info(response_data)
                if alert_content:
                    await self.send_trade_alert(alert_content, "trade")

        await channel.send(f"✅ **Command {command_index} processing complete**")

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

    def _initialize_workflow_commands(self):
        """Initialize the command sequences for each workflow phase with realistic data-driven tasks"""
        self.phase_commands = {
            'macro_foundation_data_collection': [
                # Phase 0a: Data Collection (prerequisite for macro analysis)
                "!d analyze Fetch current SPY, QQQ, IWM prices and calculate market breadth indicators (advancers/decliners, new highs/lows)",
                "!d analyze Pull economic data: Fed Funds Rate, Treasury yields (2Y/10Y/30Y), VIX, USD index, oil/gold prices",
                "!d analyze Calculate sector ETF performance: XLY, XLC, XLF, XLB, XLE, XLK, XLU, XLV, XLRE, XLP, XLI vs SPY"
            ],
            'macro_foundation_analysis': [
                # Phase 0b: Macro Analysis (using collected data)
                "!m analyze Based on collected economic and market data, assess market regime: Identify if we're in risk-on/risk-off, trending/range-bound, bull/bear market. Generate 5 specific trade proposals based on macro regime (e.g., sector rotation, duration positioning, currency trades) with Probability of Profit (PoP) calculations. Consider current portfolio positions and risk constraints.",
                "!m analyze Analyze sector performance data to generate top 5 sector opportunities based on relative strength, momentum, and risk-adjusted returns. For each sector, propose specific ETF or stock trades with entry/exit criteria and PoP estimates. Ensure proposals complement existing positions without excessive concentration."
            ],

            'intelligence_gathering': [
                "!d analyze Pull detailed data for top 5 sectors: price action, volume, technical indicators (RSI, MACD, moving averages). Identify 5 specific trade setups with support/resistance levels, momentum signals, and Probability of Profit (PoP) calculations based on historical patterns.",
                "!d analyze Fetch institutional holdings data for sector leaders (13F filings, ETF flows, options positioning). Generate 5 trade proposals based on institutional accumulation/distribution patterns with PoP estimates derived from institutional activity correlations.",
                "!d analyze Gather news sentiment and social media metrics for market-moving events and sector catalysts. Propose 5 trades based on sentiment extremes or breaking news developments with PoP calculations considering sentiment impact history.",
                "!d analyze Calculate volatility metrics and options data (put/call ratios, VIX term structure, gamma exposure). Develop 5 volatility-based trade proposals (straddles, spreads, directional bets) with PoP assessments based on volatility regime analysis.",
                "!d analyze Cross-validate data sources and identify any discrepancies or data quality issues. Refine 5 trade proposals based on data reliability and market microstructure analysis, updating PoP calculations accordingly."
            ],

            'strategy_development': [
                "!s analyze Based on macro regime and sector data, develop 5 specific trading strategies with entry/exit criteria. Each strategy must include: instrument selection, position sizing, entry triggers, profit targets, stop losses, and Probability of Profit (PoP) calculations. Consider current portfolio positions and ensure proposals maintain proper diversification and risk limits.",
                "!m analyze Provide macro perspective on proposed strategies - how do they align with current economic regime and market cycle? Validate that strategies are appropriate for current macro conditions.",
                "!d analyze Validate strategy proposals with current market data - check technical levels, volume patterns, and institutional activity. Ensure entry/exit criteria are supported by current market structure.",
                "!s analyze Incorporate technical analysis: support/resistance levels, trend channels, momentum divergence signals. Generate 5 specific trade proposals with technical entry/exit points, risk/reward ratios, and PoP estimates based on historical success rates. Account for existing position sizes and avoid over-concentration in similar assets.",
                "!s analyze Design risk management overlays: position sizing, stop losses, hedging strategies using options/futures. Propose 5 complete trade structures with defined risk parameters and PoP calculations considering win rate and risk/reward profiles. Ensure total portfolio risk remains within acceptable limits.",
                "!m analyze Assess how proposed strategies fit within broader portfolio allocation - sector balance, duration positioning, and risk parity considerations.",
                "!d analyze Cross-validate strategy timing with market microstructure - order flow, liquidity conditions, and execution feasibility.",
                "!s analyze Consider market timing: optimal entry points, holding periods, exit triggers based on technicals. Develop 5 timing-based trade proposals with specific entry windows, holding periods, and PoP assessments. Factor in current market exposure and rebalancing needs.",
                "!s analyze Evaluate strategy robustness across different market scenarios (bull/bear/sideways). Generate 5 scenario-specific trade proposals with conditional execution criteria and PoP calculations for each scenario. Consider how proposals perform relative to current portfolio composition."
            ],

            'debate': [
                '!m debate "Evaluate strategy robustness: Which approaches work best in current regime? Consider alternatives." strategy reflection data execution',
                '!m debate "Market timing and execution: When to enter/exit? What are the practical constraints?" strategy execution data'
            ],

            'risk_assessment': [
                "!ref analyze Calculate Value at Risk (VaR) for each proposed trade using historical simulation and parametric methods. Rank trades by risk-adjusted return potential and refine Probability of Profit (PoP) calculations. Consider how new trades affect total portfolio VaR and correlation with existing positions.",
                "!d analyze Provide current market data context for risk assessment - volatility levels, liquidity conditions, and correlation matrices for proposed instruments.",
                "!ref analyze Assess tail risk: Black Swan scenarios, correlation breakdowns, liquidity crunch possibilities. Identify which trade proposals are most vulnerable to extreme events and adjust PoP estimates downward accordingly. Evaluate impact on current portfolio resilience.",
                "!m analyze Evaluate macroeconomic risk factors - interest rate changes, currency impacts, geopolitical risks that could affect proposed trades.",
                "!ref analyze Evaluate strategy drawdown potential and maximum loss scenarios under stress conditions. Propose risk mitigation strategies for high-risk trades and recalibrate PoP calculations based on stress testing results. Assess combined drawdown risk with existing positions.",
                "!d analyze Validate risk metrics with current market data - check volatility surfaces, option pricing, and institutional positioning that affects risk calculations.",
                "!ref analyze Consider systemic risks: Fed policy changes, geopolitical events, economic data surprises. Generate contingency trade proposals for different risk scenarios with updated PoP assessments. Review current portfolio exposure to these risks.",
                "!ref analyze Generate risk-adjusted return metrics and Sharpe/Sortino ratios for strategy comparison. Recommend trade sizing based on risk tolerance and finalize PoP calculations. Ensure proposals align with current portfolio risk profile."
            ],

            'consensus': [
                "!ref analyze Synthesize all agent inputs: Which trade proposals pass risk/reward hurdles? What are the trade-offs between different proposals? Consider portfolio impact, diversification benefits, and refine Probability of Profit (PoP) calculations based on consensus confidence.",
                "!m analyze Provide macro-level consensus on trade proposals - which align best with current economic regime and long-term market outlook?",
                "!d analyze Validate consensus with current market data - which proposals have the strongest technical and fundamental support?",
                "!ref analyze Evaluate strategy consensus: Where do all agents agree on trade proposals? Where are there material disagreements on specific trades? Assess alignment with current position strategy and adjust PoP estimates based on consensus levels.",
                "!ref analyze Consider implementation feasibility: Capital requirements, slippage costs, execution complexity for each proposed trade. Evaluate funding needs relative to current cash balance and factor feasibility into final PoP calculations.",
                "!d analyze Assess market capacity: Can proposed trades be scaled without moving markets or exhausting liquidity? Consider position size limits relative to average daily volume and adjust PoP estimates for scalability concerns.",
                "!m analyze Evaluate portfolio-level implications of consensus trades - sector allocation changes, risk profile shifts, and macroeconomic positioning.",
                "!ref analyze Generate final trade proposal rankings with confidence levels and implementation priorities. Ensure selected trades complement current portfolio without excessive risk concentration, with final PoP assessments."
            ],

            'execution_validation': [
                "!exec analyze CONDITIONAL EXECUTION: Review supreme_oversight decision - if EXECUTE approved, model transaction costs, spreads, and market impact for proposed position sizes. If RESTART or HOLD, skip execution analysis.",
                "!exec analyze CONDITIONAL EXECUTION: If executing, assess execution logistics, trading hours, venue selection, and algorithmic requirements. Validate execution feasibility and check position limits. If not executing, analyze what would need to change for future execution.",
                "!exec analyze CONDITIONAL EXECUTION: If executing, evaluate position management, rebalancing triggers, and exit strategies. If restarting workflow, identify required re-analysis areas. If holding, specify monitoring triggers.",
                "!exec analyze CONDITIONAL EXECUTION: If executing, consider tax implications and optimize trade structure. If not executing, document execution constraints for future reference.",
                "!exec analyze CONDITIONAL EXECUTION: Generate execution playbook only if trades approved. Include position sizing and risk parameters. If not executing, provide restart criteria for next workflow cycle.",
                "!exec execute CONDITIONAL EXECUTION: Execute trades ONLY if supreme_oversight approved execution. Execute top approved trades through IBKR with full logging. If restart/hold decision, log decision and prepare for next cycle."
            ],

            'learning': [
                "!l analyze Review historical performance of similar trade proposals in current market regime. Identify patterns in successful vs failed trades. Consider how similar trades performed in current portfolio context.",
                "!l analyze Identify key success factors and common failure modes from past trade implementations. Update trade proposal templates based on historical outcomes. Learn from current position performance history.",
                "!l analyze Update strategy templates and decision frameworks based on current analysis. Refine trade proposal generation criteria. Incorporate lessons from current portfolio composition.",
                "!l analyze Document lessons learned and update institutional knowledge base with trade-specific insights. Include position management learnings from current holdings.",
                "!l analyze Generate trade improvement recommendations for future workflow iterations based on current proposal outcomes. Suggest portfolio rebalancing strategies based on historical performance."
            ],

            'supreme_oversight': [
                "!ref analyze EXECUTION DECISION: Review all workflow analysis and premarket developments - should we EXECUTE consensus trades, RESTART workflow cycle, or HOLD positions? Consider market conditions, news events, and risk assessment. Make definitive recommendation with reasoning.",
                "!ref analyze EXECUTION DECISION: If EXECUTING - provide final position sizes, entry timing, and risk parameters for approved trades. If RESTARTING - identify what aspects need re-analysis. If HOLDING - specify monitoring triggers.",
                "!ref analyze EXECUTION DECISION: Final execution authority - approve trade execution, mandate workflow restart, or implement holding pattern. Include specific execution instructions or restart parameters based on current market context."
            ],

            # MARKET OPEN EXECUTION WORKFLOW - Fast-track execution leveraging premarket analysis
            'market_open_quick_check': [
                "!d analyze MARKET OPEN CHECK: Pull current market data - SPY, QQQ, VIX, Treasury yields. Compare to premarket analysis expectations. Identify any significant deviations requiring execution delay.",
                "!d analyze MARKET OPEN CHECK: Check order book depth and liquidity for approved trade symbols. Assess if market conditions support planned execution sizes without excessive slippage.",
                "!ref analyze MARKET OPEN CHECK: Review premarket news and overnight developments. Determine if any breaking news invalidates approved trade thesis. Quick risk reassessment.",
                "!exec analyze MARKET OPEN CHECK: Validate execution logistics - IBKR connectivity, margin availability, position limits. Confirm all technical requirements for immediate execution."
            ],

            'market_open_execution': [
                "!exec execute MARKET OPEN EXECUTION: Execute approved trades immediately using premarket analysis parameters. Execute top 3 priority trades with market orders, full position sizes as approved.",
                "!exec execute MARKET OPEN EXECUTION: Log all execution details - entry prices, timestamps, order IDs. Update portfolio positions and calculate realized P&L.",
                "!exec execute MARKET OPEN EXECUTION: Set initial stop losses and profit targets based on premarket risk parameters. Configure automated risk management.",
                "!exec execute MARKET OPEN EXECUTION: Send execution alerts and initiate trade monitoring workflow. Transition from analysis mode to active position management."
            ],

            # TRADE MONITORING WORKFLOW - Ongoing position management after execution
            'trade_monitoring_setup': [
                "!exec analyze MONITORING SETUP: Establish monitoring parameters for executed positions - entry prices, stop levels, profit targets, holding periods.",
                "!exec analyze MONITORING SETUP: Configure automated alerts - price triggers, volume spikes, news sentiment changes, technical level breaches.",
                "!exec analyze MONITORING SETUP: Set up position scaling rules - partial profit taking, stop loss tightening, position size adjustments based on performance.",
                "!exec analyze MONITORING SETUP: Initialize risk monitoring - portfolio VaR, correlation changes, drawdown limits, margin utilization tracking."
            ],

            'trade_monitoring_active': [
                "!exec analyze ACTIVE MONITORING: Track position performance in real-time - P&L, unrealized gains/losses, risk metrics, market exposure.",
                "!exec analyze ACTIVE MONITORING: Monitor market conditions - volatility changes, sector rotations, macroeconomic developments affecting positions.",
                "!exec analyze ACTIVE MONITORING: Evaluate exit opportunities - profit targets hit, stop losses triggered, time-based exits, fundamental changes.",
                "!exec analyze ACTIVE MONITORING: Assess portfolio impact - overall risk profile, diversification, correlation changes, rebalancing needs."
            ],

            'trade_monitoring_decisions': [
                "!exec analyze MONITORING DECISIONS: Evaluate position adjustments - scale out profits, tighten stops, add to winners, cut losers.",
                "!exec analyze MONITORING DECISIONS: Consider new opportunities - related trades, hedging strategies, portfolio rebalancing trades.",
                "!exec analyze MONITORING DECISIONS: Risk management actions - reduce position sizes, add protective options, implement trailing stops.",
                "!exec analyze MONITORING DECISIONS: Exit decisions - full position closes, partial exits, time-based liquidations, fundamental-driven exits."
            ],

            'trade_monitoring_execution': [
                "!exec execute MONITORING EXECUTION: Execute approved position adjustments - scale outs, stop tightening, protective additions.",
                "!exec execute MONITORING EXECUTION: Execute new trades identified during monitoring - hedges, rebalancing, opportunity captures.",
                "!exec execute MONITORING EXECUTION: Execute exit orders - profit taking, stop loss execution, time-based closures.",
                "!exec execute MONITORING EXECUTION: Update portfolio records and risk metrics after all executions. Log comprehensive trade history."
            ]
        }

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
                        except ValueError:
                            print(f"⚠️ Invalid alerts channel ID: {alerts_channel_id}")
                    else:
                        print("⚠️ DISCORD_ALERTS_CHANNEL_ID not set, trade alerts will go to general channel")

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
                    if self.channel:
                        channel_status = "✅ Channel separation active" if self.alerts_channel else "⚠️ Alerts channel not configured"
                        await self.channel.send("🎯 **Live Workflow Orchestrator Online**\n🤖 Ready to begin iterative reasoning workflow with unified agent coordination. Type `!start_workflow` to begin analysis, `!start_market_open_execution` for fast execution, or `!start_trade_monitoring` for position monitoring!")
                        await self.channel.send(f"📢 **Channel Configuration:** {channel_status}")
                        if self.alerts_channel:
                            await self.channel.send(f"🚨 Trade alerts will be sent to: #{self.alerts_channel.name}")

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

            # Handle human questions/interventions during active workflow
            elif self.workflow_active and not message.author.bot:
                await self.handle_human_intervention(message)

        token = get_vault_secret('DISCORD_ORCHESTRATOR_TOKEN')
        if not token:
            raise ValueError("❌ DISCORD_ORCHESTRATOR_TOKEN not found. Please create a separate Discord bot for the orchestrator.")

    async def handle_human_intervention(self, message):
        """Handle human questions or interventions during workflow via A2A"""
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
        self.human_interventions.append(intervention)
        self.workflow_log.append(f"👤 {message.author.display_name[:50]}: {sanitized_content[:100]}...")

        # Acknowledge the intervention
        await message.add_reaction("👀")

        # If it's a question, consult reflection agent via A2A
        if any(word in message.content.lower() for word in ['?', 'what', 'how', 'why', 'can you', 'explain']):
            await message.channel.send(f"🤔 Human intervention noted: `{message.content[:100]}...`\n⏸️  Consulting reflection agent...")

            # Ask reflection agent to address the question via A2A
            if 'reflection' in self.agent_instances:
                try:
                    reflection_response = await self.agent_instances['reflection'].analyze(
                        f"Human Question: {message.content}. Please analyze this question in the context of our current workflow phase ({self.current_phase}) and provide relevant insights or recommendations."
                    )
                    
                    # Format and present reflection agent's response
                    await message.channel.send("🤖 **Reflection Agent Analysis:**")
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
                    await message.channel.send(f"⚠️ Reflection agent consultation failed: {str(e)}")
            else:
                await message.channel.send("⚠️ Reflection agent not available for consultation")

            await message.channel.send("▶️ Continuing workflow...")
        else:
            # For non-questions, just log and continue
            await message.channel.send(f"📝 Intervention logged. Continuing workflow...")

    async def start_workflow(self):
        """Start the complete iterative reasoning workflow"""
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

        await channel.send("🤝 **Collaborative session established for unified agent coordination**")

        # Create collaborative session for enhanced cross-agent communication
        session_created = await self.create_collaborative_session("Enhanced Trading Analysis with Position Awareness")
        if session_created:
            await channel.send("🎯 **Enhanced A2A Session Active** - Agents now share full workflow context and position data")
        else:
            await channel.send("⚠️ **Limited Collaboration Mode** - Agents operating with reduced context sharing")

        self.workflow_active = True
        self.current_phase = "starting"
        self.workflow_log = []
        self.responses_collected = []
        self.human_interventions = []

        await channel.send("🚀 **STARTING SEQUENTIAL ITERATIVE REASONING WORKFLOW**")
        await channel.send("📊 Complete analysis of market conditions, strategies, and risks for immediate trading decisions")
        await channel.send("💡 You can ask questions or intervene at any time!")

        # Phase 0: Macro Foundation (Data Collection + Analysis)
        await self.execute_phase_with_agents('macro_foundation_data_collection', "🏛️ PHASE 0a: DATA COLLECTION")
        await self.execute_phase_with_agents('macro_foundation_analysis', "🏛️ PHASE 0b: MACRO ANALYSIS")

        # Single Iteration: Comprehensive Deliberation
        await channel.send("\n🔄 **SINGLE ITERATION: COMPREHENSIVE ANALYSIS**")
        await channel.send("📊 Complete analysis of market conditions, strategies, and risks for immediate trading decisions")

        phases = [
            ('intelligence_gathering', "📊 Phase 1: Intelligence Gathering"),
            ('strategy_development', "🎯 Phase 2: Strategy Development"),
            ('debate', "⚔️ Phase 3: Multi-Agent Debate"),
            ('risk_assessment', "⚠️ Phase 4: Risk Assessment & Refinement"),
            ('consensus', "🤝 Phase 5: Consensus Building"),
            ('supreme_oversight', "🎯 Phase 6: Final Execution Decision"),
            ('execution_validation', "✅ Phase 7: Execution Validation"),
            ('learning', "🧠 Phase 8: Learning Integration")
        ]

        for phase_key, phase_title in phases:
            await self.execute_phase_with_agents(phase_key, phase_title)

        # Complete workflow with final decision
        await channel.send("\n🎯 **WORKFLOW COMPLETE**")
        await channel.send("📊 Analysis complete - decision made in supreme_oversight phase")

        # Complete workflow
        await self.complete_workflow()

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

        await channel.send("🏁 **MARKET OPEN EXECUTION WORKFLOW STARTED**")
        await channel.send("⚡ Fast-track execution leveraging premarket analysis")
        await channel.send("💡 This workflow executes pre-approved trades at market open")

        # Create collaborative session for execution coordination
        session_created = await self.create_collaborative_session("Market Open Execution")
        if session_created:
            await channel.send("🎯 **Execution Session Active** - Coordinated trade execution")

        self.workflow_active = True
        self.current_phase = "market_open_execution"
        self.workflow_log = []
        self.responses_collected = []
        self.human_interventions = []

        # Quick market check phase
        await self.execute_phase_with_agents('market_open_quick_check', "🔍 MARKET OPEN QUICK CHECK")

        # Execution phase
        await self.execute_phase_with_agents('market_open_execution', "💹 MARKET OPEN EXECUTION")

        # Complete and transition to monitoring
        await channel.send("✅ **MARKET OPEN EXECUTION COMPLETE**")
        await channel.send("🔄 **AUTO-TRANSITIONING TO TRADE MONITORING**")

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

        await channel.send("👁️ **TRADE MONITORING WORKFLOW ACTIVATED**")
        await channel.send("📊 Continuous monitoring of executed positions")
        await channel.send("⚡ Real-time risk management and exit opportunities")

        # Create monitoring session
        session_created = await self.create_collaborative_session("Active Trade Monitoring")
        if session_created:
            await channel.send("🎯 **Monitoring Session Active** - Continuous position oversight")

        self.monitoring_active = True
        self.current_phase = "trade_monitoring"
        self.monitoring_log = []
        self.monitoring_responses = []

        # Setup monitoring parameters
        await self.execute_phase_with_agents('trade_monitoring_setup', "⚙️ MONITORING SETUP")

        # Start continuous monitoring loop
        await channel.send("🔄 **STARTING CONTINUOUS MONITORING LOOP**")
        await channel.send("📡 Monitoring will run continuously until positions are closed")

        # Run monitoring phases in a loop
        monitoring_cycle = 0
        while self.monitoring_active:
            monitoring_cycle += 1
            await channel.send(f"\n📊 **MONITORING CYCLE {monitoring_cycle}**")

            # Active monitoring phase
            await self.execute_phase_with_agents('trade_monitoring_active', f"👁️ ACTIVE MONITORING - Cycle {monitoring_cycle}")

            # Decision phase
            await self.execute_phase_with_agents('trade_monitoring_decisions', f"🎯 MONITORING DECISIONS - Cycle {monitoring_cycle}")

            # Check if we need to execute any adjustments
            await self.execute_phase_with_agents('trade_monitoring_execution', f"⚡ MONITORING EXECUTION - Cycle {monitoring_cycle}")

            # Check if all positions are closed
            if await self._check_positions_closed():
                await channel.send("✅ **ALL POSITIONS CLOSED - MONITORING COMPLETE**")
                self.monitoring_active = False
                break

            # Wait before next monitoring cycle (e.g., 5 minutes)
            await channel.send("⏳ Waiting 5 minutes before next monitoring cycle...")
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

        await channel.send("🎉 **TRADE MONITORING COMPLETED**")

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
        max_wait_time = self.phase_delays.get(phase_key, 30)

        for i, command in enumerate(commands, 1):
            if not self.workflow_active:
                return  # Allow for pausing/stopping

            # Determine which channel to send this command to
            target_channel = self.get_command_channel(command)
            channel_name = target_channel.name if target_channel and target_channel != general_channel else "general"
            
            # Debug: Print routing info
            print(f"🎯 Routing command: '{command[:50]}...' to #{channel_name}")
            print(f"   Available agent channels: {list(self.agent_channels.keys())}")
            
            # Announce command in general channel
            await general_channel.send(f"📤 **Command {i}/{len(commands)}:** `{command}` → #{channel_name}")

            # Send the command to the appropriate channel
            if target_channel:
                print(f"   Sending to channel: {target_channel.name} (ID: {target_channel.id})")
                await target_channel.send(command)
            else:
                # Fallback to general channel if target not found
                print("   ERROR: No target channel, using general")
                await general_channel.send(command)

            # Wait for responses with dynamic timing
            await general_channel.send(f"⏳ Waiting for agent responses (max {max_wait_time}s)...")

            # Track responses for this specific command
            initial_response_count = len(self.responses_collected)
            start_time = time.time()
            responses_received = 0

            while time.time() - start_time < max_wait_time and self.workflow_active:
                await asyncio.sleep(2)  # Check every 2 seconds

                # Count new responses for this phase
                current_responses = len([r for r in self.responses_collected if r['phase'] == phase_key])
                if current_responses > initial_response_count:
                    responses_received = current_responses - initial_response_count
                    await general_channel.send(f"📥 Received {responses_received} response(s) so far...")

                    # If we got at least one response, wait a bit longer for others
                    if responses_received >= 1:
                        await asyncio.sleep(5)  # Wait 5 more seconds for additional responses
                        break

            # Final count
            final_responses = len([r for r in self.responses_collected if r['phase'] == phase_key]) - initial_response_count
            if final_responses > 0:
                await general_channel.send(f"✅ **Received {final_responses} response(s) for command {i}**")
            else:
                await general_channel.send(f"⏰ **No responses received within {max_wait_time}s timeout**")

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
        await channel.send("⏸️ **Workflow Paused** - Type `!resume_workflow` to continue")

    async def resume_workflow(self):
        """Resume a paused workflow"""
        if self.channel is None:
            return

        channel = cast(discord.TextChannel, self.channel)
        if not self.workflow_active:
            self.workflow_active = True
            await channel.send("▶️ **Workflow Resumed** - Continuing from current phase...")
            # Could implement logic to resume from current phase

    async def stop_workflow(self):
        """Stop the current workflow"""
        if self.channel is None:
            return

        channel = cast(discord.TextChannel, self.channel)
        self.workflow_active = False
        self.current_phase = "stopped"
        await channel.send("🛑 **Workflow Stopped** - All progress saved")

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
        """Complete the workflow and provide summary"""
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

        # Save comprehensive results
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

        with open('data/live_workflow_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        await channel.send("💾 Results saved to `data/live_workflow_results.json`")
        await channel.send("\n🔄 Ready for next workflow! Type `!start_workflow` to begin again.")

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
                await self.client.start(token)
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