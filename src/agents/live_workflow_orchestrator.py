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
    ReflectionAgent = None
    ExecutionAgent = None
    LearningAgent = None
from src.utils.a2a_protocol import A2AProtocol

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

    def _initialize_agents(self):
        """Initialize direct agent instances for enhanced integration (sync placeholder)"""
        # Agent initialization will be done asynchronously in initialize_agents_async
        pass

    async def initialize_agents_async(self):
        """Asynchronously initialize direct agent instances for enhanced integration"""
        if not AGENTS_AVAILABLE:
            print("⚠️ Agent classes not available - running in Discord-only mode")
            return
            
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
                    
                try:
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
        """Send command directly to agent using BaseAgent methods"""
        if agent_name not in self.agent_instances:
            print(f"⚠️ Agent {agent_name} not available for direct command")
            return None
            
        agent = self.agent_instances[agent_name]
        
        try:
            # Parse command for direct agent method calls
            if command.startswith('!analyze') or command.startswith('analyze'):
                # Extract analysis query
                query = command.replace('!analyze', '').replace('analyze', '').strip()
                if data:
                    query += f" {data}"
                    
                result = await agent.analyze(query)
                return result
                
            elif command.startswith('!status') or command.startswith('status'):
                result = await agent.get_status()
                return result
                
            else:
                # For other commands, use generic analyze method
                result = await agent.analyze(command)
                return result
                
        except Exception as e:
            print(f"⚠️ Direct agent command failed for {agent_name}: {e}")
            return {'error': str(e), 'agent': agent_name}

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

    async def collect_agent_insights(self, phase_key: str) -> List[Dict[str, Any]]:
        """Collect insights from all agents using A2A protocol"""
        insights = []
        
        if not self.collaborative_session_id:
            return insights
            
        try:
            # Get insights from collaborative session
            first_agent = next(iter(self.agent_instances.values()))
            session_insights = await first_agent.get_session_insights(self.collaborative_session_id)
            insights.extend(session_insights)
            
        except Exception as e:
            print(f"⚠️ Failed to collect agent insights: {e}")
            
        return insights

    async def execute_phase_with_agents(self, phase_key: str, phase_title: str):
        """Execute a phase using both Discord and direct agent methods"""
        if not self.channel:
            print(f"❌ No channel available for phase {phase_key}")
            return

        general_channel = cast(discord.TextChannel, self.channel)
        self.current_phase = phase_key

        # Announce phase start
        await general_channel.send(f"\n{phase_title}")
        await general_channel.send("─" * 50)

        commands = self.phase_commands.get(phase_key, [])
        
        # Share phase context with agents
        await self.share_workflow_context('current_phase', {
            'phase_key': phase_key,
            'phase_title': phase_title,
            'timestamp': datetime.now().isoformat()
        })

        for i, command in enumerate(commands, 1):
            if not self.workflow_active:
                return

            # Try direct agent communication first, then Discord
            agent_responses = []
            
            # Extract target agent from command
            if command.startswith('!'):
                prefix = command.split()[0].replace('!', '')
                agent_name = {'m': 'macro', 'd': 'data', 's': 'strategy', 
                            'r': 'risk', 'ref': 'reflection', 'exec': 'execution', 
                            'l': 'learning'}.get(prefix)
                            
                if agent_name and agent_name in self.agent_instances:
                    # Try direct agent call
                    direct_response = await self.send_direct_agent_command(agent_name, command)
                    if direct_response:
                        agent_responses.append({
                            'agent': agent_name,
                            'method': 'direct',
                            'response': direct_response,
                            'phase': phase_key
                        })
                        print(f"✅ Direct agent response from {agent_name}")

            # Also send via Discord for broader participation
            target_channel = self.get_command_channel(command)
            channel_name = target_channel.name if target_channel else "general"
            
            await general_channel.send(f"📤 **Command {i}/{len(commands)}:** `{command}` → #{channel_name}")

            if target_channel:
                await target_channel.send(command)

            # Wait for responses (both direct and Discord)
            max_wait_time = self.phase_delays.get(phase_key, 30)
            await general_channel.send(f"⏳ Waiting for responses (max {max_wait_time}s)...")

            start_time = time.time()
            discord_responses = 0
            direct_responses = len(agent_responses)

            while time.time() - start_time < max_wait_time and self.workflow_active:
                await asyncio.sleep(2)
                
                # Check for new Discord responses
                current_discord = len([r for r in self.responses_collected if r['phase'] == phase_key])
                if current_discord > discord_responses:
                    discord_responses = current_discord
                    await general_channel.send(f"📥 Discord responses: {discord_responses}")

            # Combine responses
            total_responses = direct_responses + discord_responses
            if total_responses > 0:
                await general_channel.send(f"✅ **Received {total_responses} response(s) for command {i}** ({direct_responses} direct, {discord_responses} Discord)")
            else:
                await general_channel.send(f"⏰ **No responses received within {max_wait_time}s timeout**")

        await general_channel.send(f"✅ **{phase_title} Complete!**")
        await asyncio.sleep(3)

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
                "!m analyze Based on collected economic and market data, assess market regime: Identify if we're in risk-on/risk-off, trending/range-bound, bull/bear market",
                "!m analyze Analyze sector performance data to generate top 5 sector opportunities based on relative strength, momentum, and risk-adjusted returns"
            ],

            'intelligence_gathering': [
                "!d analyze Pull detailed data for top 5 sectors: price action, volume, technical indicators (RSI, MACD, moving averages)",
                "!d analyze Fetch institutional holdings data for sector leaders (13F filings, ETF flows, options positioning)",
                "!d analyze Gather news sentiment and social media metrics for market-moving events and sector catalysts",
                "!d analyze Calculate volatility metrics and options data (put/call ratios, VIX term structure, gamma exposure)",
                "!d analyze Cross-validate data sources and identify any discrepancies or data quality issues"
            ],

            'strategy_development': [
                "!s analyze Based on macro regime and sector data, develop 3-5 specific trading strategies with entry/exit criteria",
                "!s analyze Incorporate technical analysis: support/resistance levels, trend channels, momentum divergence signals",
                "!s analyze Design risk management overlays: position sizing, stop losses, hedging strategies using options/futures",
                "!s analyze Consider market timing: optimal entry points, holding periods, exit triggers based on technicals",
                "!s analyze Evaluate strategy robustness across different market scenarios (bull/bear/sideways)"
            ],

            'debate': [
                '!m debate "Evaluate strategy robustness: Which approaches work best in current regime? Consider alternatives." strategy reflection data execution',
                '!m debate "Market timing and execution: When to enter/exit? What are the practical constraints?" strategy execution data'
            ],

            'risk_assessment': [
                "!ref analyze Calculate Value at Risk (VaR) for each strategy using historical simulation and parametric methods",
                "!ref analyze Assess tail risk: Black Swan scenarios, correlation breakdowns, liquidity crunch possibilities",
                "!ref analyze Evaluate strategy drawdown potential and maximum loss scenarios under stress conditions",
                "!ref analyze Consider systemic risks: Fed policy changes, geopolitical events, economic data surprises",
                "!ref analyze Generate risk-adjusted return metrics and Sharpe/Sortino ratios for strategy comparison"
            ],

            'consensus': [
                "!ref analyze Synthesize all agent inputs: Which strategies pass risk/reward hurdles? What are the trade-offs?",
                "!ref analyze Evaluate strategy consensus: Where do all agents agree? Where are there material disagreements?",
                "!ref analyze Consider implementation feasibility: Capital requirements, slippage costs, execution complexity",
                "!ref analyze Assess market capacity: Can strategies be scaled without moving markets or exhausting liquidity?",
                "!ref analyze Generate final strategy rankings with confidence levels and implementation priorities"
            ],

            'execution_validation': [
                "!exec analyze Model transaction costs: Commissions, spreads, market impact for proposed position sizes",
                "!exec analyze Assess execution logistics: Trading hours, venue selection, algorithmic execution requirements",
                "!exec analyze Evaluate position management: Rebalancing triggers, scaling in/out protocols, exit strategies",
                "!exec analyze Consider tax implications and wash sale rules for strategy implementation",
                "!exec analyze Generate detailed execution playbook with step-by-step implementation guide"
            ],

            'learning': [
                "!l analyze Review historical performance of similar strategies in current market regime",
                "!l analyze Identify key success factors and common failure modes from past implementations",
                "!l analyze Update strategy templates and decision frameworks based on current analysis",
                "!l analyze Document lessons learned and update institutional knowledge base",
                "!l analyze Generate strategy improvement recommendations for future workflow iterations"
            ],

            'executive_review': [
                "!ref analyze ITERATION 2: Review Iteration 1 results and identify gaps, assumptions, or alternative perspectives that were missed",
                "!ref analyze ITERATION 2: Reassess risks using Iteration 1 findings - are there hidden risks or overconfidence in the initial analysis?",
                "!s analyze ITERATION 2: Develop enhanced strategies building on Iteration 1 - consider counter-trend approaches, timing refinements, or position sizing adjustments",
                "!ref analyze ITERATION 2: Challenge Iteration 1 consensus - what dissenting views or devil's advocate positions should be considered?",
                "!exec analyze ITERATION 2: Evaluate execution feasibility of Iteration 1 strategies under various market conditions and liquidity scenarios",
                "!l analyze ITERATION 2: Apply historical lessons to Iteration 1 strategies - what similar market conditions produced different outcomes?",
                "!m analyze ITERATION 2: Consider macroeconomic regime changes - how would Iteration 1 strategies perform if the regime shifts?"
            ],

            'supreme_oversight': [
                "!ref analyze SUPREME OVERSIGHT: Compare Iteration 1 vs Iteration 2 analyses - which insights are most robust and actionable?",
                "!ref analyze SUPREME OVERSIGHT: Synthesize both iterations into final recommendations - what is the strongest consensus position?",
                "!ref analyze SUPREME OVERSIGHT: Crisis detection across both iterations - identify any black swan risks or critical assumptions",
                "!ref analyze SUPREME OVERSIGHT: Final veto authority - approve/reject strategies or mandate additional analysis based on both iterations",
                "!ref analyze SUPREME OVERSIGHT: Implementation prioritization - rank strategies by conviction level considering both iteration insights"
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
                    for ch in guild.text_channels:
                        if ch.name in ['general', 'workflow', 'analysis', 'trading']:
                            self.channel = ch
                            print(f"📝 General summaries in: #{ch.name}")
                            break

                    if not self.channel and guild.text_channels:
                        self.channel = guild.text_channels[0]
                        print(f"📝 Using default general channel: #{self.channel.name}")

                    # Set up agent-specific channels
                    agent_channel_map = {
                        'macro': 'macro',
                        'data': 'data', 
                        'strategy': 'strategy',
                        'risk': 'risk',
                        'reflection': 'reflection',
                        'execution': 'execution',
                        'learning': 'learning',
                        'debates': 'debates',  # For debate commands
                        'alerts': 'alerts'     # For system alerts
                    }
                    
                    for agent_type, channel_name in agent_channel_map.items():
                        for ch in guild.text_channels:
                            if ch.name == channel_name:
                                self.agent_channels[agent_type] = ch
                                print(f"🤖 {agent_type.title()} agent channel: #{ch.name}")
                                break
                    
                    # Report missing channels
                    missing_channels = []
                    for agent_type in agent_channel_map.keys():
                        if agent_type not in self.agent_channels:
                            missing_channels.append(f"#{agent_type}")
                    
                    if missing_channels:
                        print(f"⚠️  Missing agent channels: {', '.join(missing_channels)}")
                        print("   Agents will respond in general channel instead")
                    else:
                        print(f"✅ All {len(self.agent_channels)} agent channels configured!")
                        print(f"   Agent channels: {list(self.agent_channels.keys())}")

                    # Announce orchestrator presence
                    if self.channel:
                        await self.channel.send("🎯 **Live Workflow Orchestrator Online**\n🤖 Ready to begin iterative reasoning workflow. Type `!start_workflow` to begin, or ask questions at any time!")

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

            if content == "!pause_workflow" and self.workflow_active:
                await self.pause_workflow()
                return

            if content == "!resume_workflow":
                await self.resume_workflow()
                return

            if content == "!stop_workflow":
                await self.stop_workflow()
                return

            if content == "!workflow_status":
                await self.send_status_update()
                return

            # Collect agent responses (from bots in any channel)
            if message.author.bot and message.author != self.client.user:
                self.responses_collected.append({
                    'agent': message.author.display_name,
                    'content': message.content,
                    'channel': message.channel.name,
                    'timestamp': message.created_at.isoformat(),
                    'phase': self.current_phase
                })
                self.workflow_log.append(f"🤖 {message.author.display_name} (#{message.channel.name}): {message.content[:100]}...")

            # Handle human questions/interventions during active workflow
            elif self.workflow_active and not message.author.bot:
                await self.handle_human_intervention(message)

        token = os.getenv('DISCORD_ORCHESTRATOR_TOKEN')
        if not token:
            raise ValueError("❌ DISCORD_ORCHESTRATOR_TOKEN not found. Please create a separate Discord bot for the orchestrator.")

    async def handle_human_intervention(self, message):
        """Handle human questions or interventions during workflow"""
        intervention = {
            'user': message.author.display_name,
            'content': message.content,
            'timestamp': message.created_at.isoformat(),
            'phase': self.current_phase
        }
        self.human_interventions.append(intervention)
        self.workflow_log.append(f"👤 {message.author.display_name}: {message.content}")

        # Acknowledge the intervention
        await message.add_reaction("👀")

        # If it's a question, pause briefly to allow agents to respond
        if any(word in message.content.lower() for word in ['?', 'what', 'how', 'why', 'can you', 'explain']):
            await message.channel.send(f"🤔 Human intervention noted: `{message.content[:100]}...`\n⏸️  Pausing workflow briefly for consideration...")
            await asyncio.sleep(5)  # Give time for consideration

            # Ask reflection agent to address the question
            await message.channel.send("!ref analyze Please address the human question above and incorporate it into our reasoning process.")

            # Wait for reflection agent response
            await asyncio.sleep(25)

            await message.channel.send("▶️ Resuming workflow...")
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

        # Create collaborative session for direct agent coordination
        session_created = await self.create_collaborative_session("Live Workflow Orchestration")
        if session_created:
            await channel.send("🤝 **Collaborative session established for direct agent coordination**")
        else:
            await channel.send("⚠️ **Discord-only mode: Direct agent coordination unavailable**")

        self.workflow_active = True
        self.current_phase = "starting"
        self.workflow_log = []
        self.responses_collected = []
        self.human_interventions = []

        await channel.send("🚀 **STARTING SEQUENTIAL ITERATIVE REASONING WORKFLOW**")
        await channel.send("📊 This will run through 2 sequential iterations where each builds upon the previous")
        await channel.send("💡 You can ask questions or intervene at any time!")

        # Phase 0: Macro Foundation (Data Collection + Analysis)
        await self.execute_phase_with_agents('macro_foundation_data_collection', "🏛️ PHASE 0a: DATA COLLECTION")
        await self.execute_phase_with_agents('macro_foundation_analysis', "🏛️ PHASE 0b: MACRO ANALYSIS")

        # Iteration 1: Comprehensive Deliberation
        await channel.send("\n🔄 **ITERATION 1: INITIAL COMPREHENSIVE ANALYSIS**")
        await channel.send("📊 First complete analysis of market conditions, strategies, and risks")

        phases_1 = [
            ('intelligence_gathering', "📊 Phase 1: Intelligence Gathering"),
            ('strategy_development', "🎯 Phase 2: Strategy Development"),
            ('debate', "⚔️ Phase 3: Multi-Agent Debate"),
            ('risk_assessment', "⚠️ Phase 4: Risk Assessment & Refinement"),
            ('consensus', "🤝 Phase 5: Consensus Building"),
            ('execution_validation', "✅ Phase 6: Execution Validation"),
            ('learning', "🧠 Phase 7: Learning Integration")
        ]

        for phase_key, phase_title in phases_1:
            await self.execute_phase_with_agents(phase_key, phase_title)

        # Iteration 2: Enhanced Analysis Building on Iteration 1
        await channel.send("\n🔄 **ITERATION 2: ENHANCED ANALYSIS (Building on Iteration 1)**")
        await channel.send("📊 Second iteration that challenges, refines, and enhances Iteration 1 findings")
        await self.execute_phase_with_agents('executive_review', "🎯 Iteration 2: Enhanced Strategic Review")

        # Supreme Oversight: Synthesis of Both Iterations
        await channel.send("\n👑 **SUPREME OVERSIGHT: FINAL SYNTHESIS**")
        await channel.send("📊 Supreme synthesis comparing and combining insights from both iterations")
        await self.execute_phase_with_agents('supreme_oversight', "🎯 Supreme Oversight: Final Decision")

        # Complete workflow
        await self.complete_workflow()

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
        await channel.send(f"• Total Agent Responses: {len(self.responses_collected)}")
        await channel.send(f"• Human Interventions: {len(self.human_interventions)}")
        await channel.send(f"• Phases Completed: All 11 phases")
        await channel.send(f"• Agent Health: {final_health['overall_health'].title()} ({len(final_health['healthy_agents'])}/{final_health['total_agents']} healthy)")
        
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
            'completed_at': datetime.now().isoformat(),
            'total_responses': len(self.responses_collected),
            'human_interventions': len(self.human_interventions),
            'responses': self.responses_collected,
            'interventions': self.human_interventions,
            'workflow_log': self.workflow_log,
            'agent_health': final_health,
            'collaborative_session_id': self.collaborative_session_id,
            'direct_agent_integration': bool(self.agent_instances),
            'phases_completed': 11
        }

        with open('data/live_workflow_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        await channel.send("💾 Results saved to `data/live_workflow_results.json`")
        await channel.send("\n🔄 Ready for next workflow! Type `!start_workflow` to begin again.")

    async def run_orchestrator(self):
        """Run the live workflow orchestrator"""
        while True:  # Keep trying to reconnect
            try:
                print("🔧 Initializing Discord client...")
                await self.initialize_discord_client()
                
                # Initialize agents asynchronously
                print("🤖 Initializing agent instances...")
                await self.initialize_agents_async()
                print(f"✅ Agent initialization complete: {len(self.agent_instances)} agents ready")
                
                token = os.getenv('DISCORD_ORCHESTRATOR_TOKEN')
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
    print("  !start_workflow  - Begin the iterative reasoning process")
    print("  !pause_workflow  - Pause the current workflow")
    print("  !resume_workflow - Resume a paused workflow")
    print("  !stop_workflow   - Stop the current workflow")
    print("  !workflow_status - Get current status")
    print("💡 You can ask questions or intervene at any time!")

    orchestrator = LiveWorkflowOrchestrator()
    await orchestrator.run_orchestrator()

if __name__ == "__main__":
    asyncio.run(main())