#!/usr/bin/env python3
"""
Discord Bot Interface for Agent Communication
Provides Discord-based communication interfaces for AI agents.
Each bot interface wraps an agent and exposes its functionality through Discord commands.
"""

import os
import sys
import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import discord
from discord.ext import commands, tasks
import pandas as pd

from src.agents.base import BaseAgent
from src.agents.macro import MacroAgent
from src.agents.data import DataAgent
from src.agents.strategy import StrategyAgent
from src.agents.risk import RiskAgent
from src.agents.reflection import ReflectionAgent
from src.agents.execution import ExecutionAgent
from src.agents.learning import LearningAgent
from src.utils.a2a_protocol import A2AProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiscordBotInterface(commands.Bot):
    """
    Discord bot interface for a specific agent.
    Provides a chat-based interface to access agent functionality.
    Each interface wraps an agent instance and exposes its methods via Discord commands.
    """

    # Class variable to track active bot interfaces for collaboration
    _active_interfaces: List['DiscordBotInterface'] = []
    _debate_channels: Dict[str, discord.TextChannel] = {}
    _human_participants: Dict[str, List[int]] = {}  # channel_id -> list of human user IDs

    def __init__(self, agent: BaseAgent, token: str, interface_config: Dict[str, Any]):
        self.agent = agent
        self.interface_config = interface_config
        self.agent_name = interface_config['name']
        self.agent_role = interface_config['role']
        self.status_channel_id = interface_config.get('status_channel_id')
        self.command_prefix = interface_config.get('command_prefix', '!')

        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        super().__init__(
            command_prefix=self.command_prefix,
            intents=intents,
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=f"markets | {self.command_prefix}help"
            )
        )

        # Add this interface to the active interfaces list
        DiscordBotInterface._active_interfaces.append(self)

        self.setup_commands()

    def setup_commands(self):
        """Set up agent-specific commands"""

        @self.command(name='status')
        async def status(ctx):
            """Get agent status"""
            try:
                status_info = await self.agent.get_status()
                embed = discord.Embed(
                    title=f"{self.agent_name} Status",
                    color=self.interface_config['color'],
                    timestamp=datetime.now()
                )

                for key, value in status_info.items():
                    embed.add_field(name=key.replace('_', ' ').title(), value=str(value), inline=True)

                await ctx.send(embed=embed)
            except Exception as e:
                await ctx.send(f"❌ Error getting status: {str(e)}")

        @self.command(name='memory')
        async def memory(ctx, limit: int = 5):
            """Get recent memory entries"""
            try:
                # Try to get memories from advanced memory system first
                if hasattr(self.agent, 'advanced_memory') and self.agent.advanced_memory:
                    # Search for recent memories for this agent
                    agent_scope = f"agent_{self.agent_role}"
                    search_results = await self.agent.advanced_memory.search_memories(
                        query=f"agent:{self.agent_role}",  # Search for this agent's memories
                        memory_type=None,
                        limit=limit
                    )
                    
                    if search_results:
                        embed = discord.Embed(
                            title=f"{self.agent_name} Recent Memories (Advanced)",
                            color=self.interface_config['color'],
                            timestamp=datetime.now()
                        )

                        for i, memory in enumerate(search_results[:limit], 1):
                            content = memory.get('data', memory.get('content', 'N/A'))
                            if isinstance(content, dict):
                                content = str(content)
                            embed.add_field(
                                name=f"Memory {i}",
                                value=content[:200] + "..." if len(str(content)) > 200 else str(content),
                                inline=False
                            )
                    else:
                        # Fallback to basic agent memory
                        memories = self.agent.get_recent_memories(limit)
                        embed = discord.Embed(
                            title=f"{self.agent_name} Recent Memories",
                            color=self.interface_config['color'],
                            timestamp=datetime.now()
                        )

                        for i, memory in enumerate(memories, 1):
                            embed.add_field(
                                name=f"Memory {i}",
                                value=memory.get('content', 'N/A')[:200] + "...",
                                inline=False
                            )
                else:
                    # Use basic agent memory
                    memories = self.agent.get_recent_memories(limit)
                    embed = discord.Embed(
                        title=f"{self.agent_name} Recent Memories",
                        color=self.interface_config['color'],
                        timestamp=datetime.now()
                    )

                    for i, memory in enumerate(memories, 1):
                        embed.add_field(
                            name=f"Memory {i}",
                            value=memory.get('content', 'N/A')[:200] + "...",
                            inline=False
                        )

                await ctx.send(embed=embed)
            except Exception as e:
                await ctx.send(f"❌ Error getting memories: {str(e)}")

        @self.command(name='analyze')
        async def analyze(ctx, *, query: str):
            """Request analysis from agent"""
            try:
                # Send typing indicator
                async with ctx.typing():
                    analysis_result = await self.agent.analyze(query)

                # Extract the analysis text from the result dictionary
                if isinstance(analysis_result, dict):
                    analysis_text = analysis_result.get('llm_analysis', 
                                                      analysis_result.get('fallback_analysis', 
                                                                        str(analysis_result)))
                else:
                    analysis_text = str(analysis_result)

                embed = discord.Embed(
                    title=f"{self.agent_name} Analysis",
                    description=analysis_text[:2000],  # Discord embed limit
                    color=self.interface_config['color'],
                    timestamp=datetime.now()
                )
                embed.set_footer(text=f"Requested by {ctx.author.display_name}")

                await ctx.send(embed=embed)
            except Exception as e:
                await ctx.send(f"❌ Error performing analysis: {str(e)}")

        @self.command(name='debate')
        async def debate(ctx, topic: str, *agents):
            """Start a debate on a topic with specified agents"""
            try:
                # Get available agents
                available_agents = [bot.agent_name.lower() for bot in self.__class__._active_interfaces]
                
                # Convert tuple to list for easier manipulation
                agents_list = list(agents) if agents else []
                
                # If no agents specified, include all
                if not agents_list:
                    agents_list = available_agents
                else:
                    agents_list = [agent.lower() for agent in agents_list]
                
                # Validate agents
                valid_agents = [agent for agent in agents_list if agent in available_agents]
                invalid_agents = [agent for agent in agents_list if agent not in available_agents]
                
                if invalid_agents:
                    await ctx.send(f"⚠️ Warning: Agents not found: {', '.join(invalid_agents)}")
                
                if not valid_agents:
                    await ctx.send("❌ No valid agents specified for debate")
                    return
                
                # Create debate embed
                embed = discord.Embed(
                    title=f"🤝 Debate Started: {topic}",
                    description=f"**Moderator:** {ctx.author.mention}\n**Participants:** {', '.join(valid_agents).title()}",
                    color=0xffd700,  # Gold color for debates
                    timestamp=datetime.now()
                )
                embed.add_field(name="Topic", value=topic, inline=False)
                embed.set_footer(text="Use !join_debate to participate as a human")
                
                debate_msg = await ctx.send(embed=embed)
                
                # Store debate info
                debate_id = f"{ctx.channel.id}_{debate_msg.id}"
                self.__class__._debate_channels[debate_id] = ctx.channel
                
                # Notify participating agents
                for bot in self.__class__._active_interfaces:
                    if bot.agent_name.lower() in valid_agents:
                        try:
                            agent_embed = discord.Embed(
                                title=f"🎯 Debate Invitation: {topic}",
                                description=f"You've been invited to debate by {ctx.author.mention}",
                                color=bot.interface_config['color'],
                                timestamp=datetime.now()
                            )
                            agent_embed.add_field(name="Topic", value=topic, inline=False)
                            agent_embed.add_field(name="Channel", value=ctx.channel.mention, inline=True)
                            agent_embed.set_footer(text=f"Debate ID: {debate_id}")
                            
                            if bot.status_channel_id:
                                channel = bot.get_channel(bot.status_channel_id)
                                if channel:
                                    await channel.send(embed=agent_embed)
                        except Exception as e:
                            logger.error(f"Failed to notify {bot.agent_name}: {e}")
                
                await ctx.send(f"✅ Debate started! {len(valid_agents)} agents invited.")
                
            except Exception as e:
                await ctx.send(f"❌ Error starting debate: {str(e)}")

        @self.command(name='join_debate')
        async def join_debate(ctx):
            """Join the current debate as a human participant"""
            try:
                # Find active debate in this channel
                debate_id = None
                for db_id, channel in self.__class__._debate_channels.items():
                    if channel.id == ctx.channel.id:
                        debate_id = db_id
                        break
                
                if not debate_id:
                    await ctx.send("❌ No active debate in this channel. Start one with `!debate <topic>`")
                    return
                
                # Add user to participants
                if debate_id not in self.__class__._human_participants:
                    self.__class__._human_participants[debate_id] = []
                
                if ctx.author.id not in self.__class__._human_participants[debate_id]:
                    self.__class__._human_participants[debate_id].append(ctx.author.id)
                    
                    embed = discord.Embed(
                        title="👤 Human Joined Debate",
                        description=f"{ctx.author.mention} has joined the debate!",
                        color=0x00ff00,
                        timestamp=datetime.now()
                    )
                    await ctx.send(embed=embed)
                else:
                    await ctx.send("ℹ️ You're already participating in this debate")
                    
            except Exception as e:
                await ctx.send(f"❌ Error joining debate: {str(e)}")

        @self.command(name='broadcast')
        async def broadcast(ctx, *agents: str):
            """Broadcast a message to multiple agents"""
            try:
                # Get the message content after the command
                message = ctx.message.content.replace(f"{ctx.prefix}broadcast", "").strip()
                if not agents:
                    await ctx.send("❌ Please specify agents to broadcast to. Usage: `!broadcast agent1 agent2 Your message here`")
                    return
                
                message = message.replace(" ".join(agents), "", 1).strip()
                if not message:
                    await ctx.send("❌ Please provide a message to broadcast")
                    return
                
                # Find target agents
                sent_count = 0
                failed_agents = []
                
                for agent_name in agents:
                    target_bot = None
                    for bot in self.__class__._active_interfaces:
                        if bot.agent_name.lower() == agent_name.lower():
                            target_bot = bot
                            break
                    
                    if target_bot:
                        try:
                            embed = discord.Embed(
                                title=f"📢 Broadcast from {ctx.author.display_name}",
                                description=message,
                                color=0xff6b6b,  # Red for broadcasts
                                timestamp=datetime.now()
                            )
                            embed.set_footer(text=f"Broadcast via {self.agent_name}")
                            
                            if target_bot.status_channel_id:
                                channel = target_bot.get_channel(target_bot.status_channel_id)
                                if channel and isinstance(channel, discord.TextChannel):
                                    await channel.send(embed=embed)
                                    sent_count += 1
                        except Exception as e:
                            logger.error(f"Failed to broadcast to {agent_name}: {e}")
                            failed_agents.append(agent_name)
                    else:
                        failed_agents.append(agent_name)
                
                response = f"✅ Broadcast sent to {sent_count} agents"
                if failed_agents:
                    response += f"\n❌ Failed to reach: {', '.join(failed_agents)}"
                
                await ctx.send(response)
                
            except Exception as e:
                await ctx.send(f"❌ Error broadcasting: {str(e)}")

        @self.command(name='discuss')
        async def discuss(ctx, topic: str):
            """Start a group discussion on a topic with all agents"""
            try:
                embed = discord.Embed(
                    title=f"💬 Group Discussion: {topic}",
                    description=f"**Initiated by:** {ctx.author.mention}\n**All agents invited to participate**",
                    color=0x9b59b6,  # Purple for discussions
                    timestamp=datetime.now()
                )
                embed.add_field(name="Topic", value=topic, inline=False)
                embed.set_footer(text="Agents will respond based on their expertise")
                
                await ctx.send(embed=embed)
                
                # Notify all agents
                for bot in self.__class__._active_interfaces:
                    try:
                        agent_embed = discord.Embed(
                            title=f"💬 Discussion Invitation: {topic}",
                            description=f"Discussion started by {ctx.author.mention} in {ctx.channel.mention}",
                            color=bot.interface_config['color'],
                            timestamp=datetime.now()
                        )
                        agent_embed.add_field(name="Topic", value=topic, inline=False)
                        
                        if bot.status_channel_id:
                            channel = bot.get_channel(bot.status_channel_id)
                            if channel and isinstance(channel, discord.TextChannel):
                                await channel.send(embed=agent_embed)
                    except Exception as e:
                        logger.error(f"Failed to notify {bot.agent_name}: {e}")
                
                await ctx.send("📨 All agents have been notified of the discussion!")
                
            except Exception as e:
                await ctx.send(f"❌ Error starting discussion: {str(e)}")

        @self.command(name='human_input')
        async def human_input(ctx, *, message: str):
            """Share human insights with all agents"""
            try:
                embed = discord.Embed(
                    title="🧠 Human Insight Shared",
                    description=message,
                    color=0x3498db,  # Blue for human input
                    timestamp=datetime.now()
                )
                embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.avatar.url if ctx.author.avatar else None)
                embed.set_footer(text="Human perspective for agent consideration")
                
                await ctx.send(embed=embed)
                
                # Forward to all agent channels
                for bot in self.__class__._active_interfaces:
                    try:
                        if bot.status_channel_id and bot != self:  # Don't send to self
                            channel = bot.get_channel(bot.status_channel_id)
                            if channel and isinstance(channel, discord.TextChannel):
                                forward_embed = embed.copy()
                                forward_embed.title = f"🧠 Human Input from {ctx.author.display_name}"
                                forward_embed.set_footer(text=f"Shared via {ctx.channel.mention}")
                                await channel.send(embed=forward_embed)
                    except Exception as e:
                        logger.error(f"Failed to forward human input to {bot.agent_name}: {e}")
                
                await ctx.send("✅ Insight shared with all agents!")
                
            except Exception as e:
                await ctx.send(f"❌ Error sharing human input: {str(e)}")

        @self.command(name='agent_question')
        async def agent_question(ctx, agent_name: str, *, question: str):
            """Ask a specific agent a question"""
            try:
                # Find target agent
                target_bot = None
                for bot in self.__class__._active_interfaces:
                    if bot.agent_name.lower() == agent_name.lower():
                        target_bot = bot
                        break
                
                if not target_bot:
                    await ctx.send(f"❌ Agent '{agent_name}' not found")
                    return
                
                # Send question to agent
                embed = discord.Embed(
                    title=f"❓ Question from {ctx.author.display_name}",
                    description=question,
                    color=0xf39c12,  # Orange for questions
                    timestamp=datetime.now()
                )
                embed.set_footer(text=f"Asked in {ctx.channel.mention}")
                
                if target_bot.status_channel_id:
                    channel = target_bot.get_channel(target_bot.status_channel_id)
                    if channel and isinstance(channel, discord.TextChannel):
                        await channel.send(embed=embed)
                        await ctx.send(f"✅ Question sent to {agent_name}!")
                        
                        # Also send to current channel for visibility
                        response_embed = discord.Embed(
                            title=f"Question Sent to {target_bot.agent_name}",
                            description=f"**Question:** {question}",
                            color=target_bot.interface_config['color'],
                            timestamp=datetime.now()
                        )
                        await ctx.send(embed=response_embed)
                    else:
                        await ctx.send(f"❌ Could not reach {agent_name}'s channel")
                else:
                    await ctx.send(f"❌ {agent_name} has no configured channel")
                    
            except Exception as e:
                await ctx.send(f"❌ Error sending question: {str(e)}")

        @self.command(name='end_debate')
        async def end_debate(ctx):
            """End the current debate in this channel"""
            try:
                # Find and remove active debate
                debate_id = None
                for db_id, channel in self.__class__._debate_channels.items():
                    if channel.id == ctx.channel.id:
                        debate_id = db_id
                        break
                
                if debate_id:
                    # Remove debate
                    del self.__class__._debate_channels[debate_id]
                    
                    # Remove participants
                    if debate_id in self.__class__._human_participants:
                        del self.__class__._human_participants[debate_id]
                    
                    embed = discord.Embed(
                        title="🏁 Debate Ended",
                        description="The debate in this channel has been concluded.",
                        color=0x95a5a6,
                        timestamp=datetime.now()
                    )
                    embed.set_footer(text=f"Ended by {ctx.author.display_name}")
                    await ctx.send(embed=embed)
                    
                    # Notify agents that debate ended
                    for bot in self.__class__._active_interfaces:
                        try:
                            if bot.status_channel_id:
                                channel = bot.get_channel(bot.status_channel_id)
                                if channel and isinstance(channel, discord.TextChannel):
                                    await channel.send(embed=embed)
                        except Exception as e:
                            logger.error(f"Failed to notify {bot.agent_name} of debate end: {e}")
                else:
                    await ctx.send("❌ No active debate found in this channel")
                    
            except Exception as e:
                await ctx.send(f"❌ Error ending debate: {str(e)}")

        @self.command(name='debate_summary')
        async def debate_summary(ctx):
            """Get a summary of the current debate"""
            try:
                debate_id = None
                for db_id, channel in self.__class__._debate_channels.items():
                    if channel.id == ctx.channel.id:
                        debate_id = db_id
                        break
                
                if not debate_id:
                    await ctx.send("❌ No active debate in this channel")
                    return
                
                embed = discord.Embed(
                    title="📊 Debate Summary",
                    color=0xffd700,
                    timestamp=datetime.now()
                )
                
                # Count messages in the debate (rough estimate)
                # This would need more sophisticated tracking in a real implementation
                embed.add_field(
                    name="Status",
                    value="Active",
                    inline=True
                )
                
                participants = self.__class__._human_participants.get(debate_id, [])
                embed.add_field(
                    name="Human Participants",
                    value=str(len(participants)),
                    inline=True
                )
                
                embed.add_field(
                    name="Agent Participants",
                    value=str(len(self.__class__._active_interfaces)),
                    inline=True
                )
                
                embed.set_footer(text=f"Debate ID: {debate_id}")
                await ctx.send(embed=embed)
                
            except Exception as e:
                await ctx.send(f"❌ Error getting debate summary: {str(e)}")

        # Agent-specific commands
        if self.agent_role == 'macro':
            @self.command(name='economy')
            async def economy(ctx):
                """Get macroeconomic analysis"""
                try:
                    analysis = await self.agent.analyze_economy()
                    embed = discord.Embed(
                        title="Macroeconomic Analysis",
                        description=analysis[:2000],
                        color=self.interface_config['color'],
                        timestamp=datetime.now()
                    )
                    await ctx.send(embed=embed)
                except Exception as e:
                    await ctx.send(f"❌ Error getting economic analysis: {str(e)}")

        elif self.agent_role == 'data':
            @self.command(name='fetch')
            async def fetch(ctx, symbol: str, data_type: str = "quotes"):
                """Fetch market data"""
                try:
                    data = await self.agent.fetch_market_data(symbol, data_type)
                    embed = discord.Embed(
                        title=f"Market Data: {symbol.upper()}",
                        color=self.interface_config['color'],
                        timestamp=datetime.now()
                    )

                    if isinstance(data, dict):
                        for key, value in data.items():
                            embed.add_field(name=key.title(), value=str(value), inline=True)
                    else:
                        embed.description = str(data)[:2000]

                    await ctx.send(embed=embed)
                except Exception as e:
                    await ctx.send(f"❌ Error fetching data: {str(e)}")

        elif self.agent_role == 'strategy':
            @self.command(name='propose')
            async def propose(ctx, *, context: str = ""):
                """Generate trading strategy proposal"""
                try:
                    proposal = await self.agent.generate_strategy(context)
                    embed = discord.Embed(
                        title="Strategy Proposal",
                        description=proposal[:2000],
                        color=self.interface_config['color'],
                        timestamp=datetime.now()
                    )
                    await ctx.send(embed=embed)
                except Exception as e:
                    await ctx.send(f"❌ Error generating strategy: {str(e)}")

        elif self.agent_role == 'risk':
            @self.command(name='assess')
            async def assess(ctx, *, portfolio_data: str):
                """Assess risk for portfolio"""
                try:
                    # Parse portfolio data (simplified)
                    risk_assessment = await self.agent.assess_risk(json.loads(portfolio_data))
                    embed = discord.Embed(
                        title="Risk Assessment",
                        color=self.interface_config['color'],
                        timestamp=datetime.now()
                    )

                    for key, value in risk_assessment.items():
                        embed.add_field(name=key.title(), value=str(value), inline=True)

                    await ctx.send(embed=embed)
                except Exception as e:
                    await ctx.send(f"❌ Error assessing risk: {str(e)}")

        elif self.agent_role == 'execution':
            @self.command(name='execute')
            async def execute(ctx, *, trade_details: str):
                """Execute a trade"""
                try:
                    result = await self.agent.execute_trade(json.loads(trade_details))
                    embed = discord.Embed(
                        title="Trade Execution",
                        description=f"✅ Trade executed successfully\n{result}",
                        color=discord.Color.green(),
                        timestamp=datetime.now()
                    )
                    await ctx.send(embed=embed)
                except Exception as e:
                    await ctx.send(f"❌ Error executing trade: {str(e)}")

        elif self.agent_role == 'reflection':
            @self.command(name='audit')
            async def audit(ctx, period: str = "daily"):
                """Perform performance audit"""
                try:
                    audit_result = await self.agent.perform_audit(period)
                    embed = discord.Embed(
                        title=f"{period.title()} Audit Results",
                        description=str(audit_result)[:2000],
                        color=self.interface_config['color'],
                        timestamp=datetime.now()
                    )
                    await ctx.send(embed=embed)
                except Exception as e:
                    await ctx.send(f"❌ Error performing audit: {str(e)}")

        elif self.agent_role == 'learning':
            @self.command(name='learn')
            async def learn(ctx, *, feedback: str):
                """Provide learning feedback"""
                try:
                    learning_result = await self.agent.process_feedback(feedback)
                    embed = discord.Embed(
                        title="Learning Update",
                        description=learning_result[:2000],
                        color=self.interface_config['color'],
                        timestamp=datetime.now()
                    )
                    await ctx.send(embed=embed)
                except Exception as e:
                    await ctx.send(f"❌ Error processing learning: {str(e)}")

    async def on_ready(self):
        """Called when bot is ready"""
        try:
            logger.info(f"{self.agent_name} bot is ready! Logged in as {self.user}")
            # Start status updates if we have a channel configured (with delay to ensure bot is fully ready)
            if hasattr(self, 'status_update') and self.status_channel_id:
                try:
                    # Delay starting the status update to ensure bot is fully connected
                    await asyncio.sleep(5)
                    self.status_update.start()
                except Exception as e:
                    logger.error(f"Failed to start status updates for {self.agent_name}: {e}")
        except Exception as e:
            logger.error(f"Error in on_ready for {self.agent_name}: {e}")
            raise

    @tasks.loop(minutes=5)
    async def status_update(self):
        """Periodic status updates"""
        try:
            if self.status_channel_id:
                channel = self.get_channel(self.status_channel_id)
                if channel and isinstance(channel, discord.TextChannel):
                    status_info = await self.agent.get_status()
                    embed = discord.Embed(
                        title=f"{self.agent_name} Status Update",
                        color=self.interface_config['color'],
                        timestamp=datetime.now()
                    )

                    # Add key metrics
                    for key, value in status_info.items():
                        if key in ['health', 'status', 'alerts']:
                            embed.add_field(name=key.title(), value=str(value), inline=True)

                    await channel.send(embed=embed)
        except Exception as e:
            logger.error(f"Error in status update for {self.agent_name}: {str(e)}")

    async def on_message(self, message):
        """Handle messages"""
        # Don't respond to own messages
        if message.author == self.user:
            return

        # Check if this is part of an active debate
        debate_id = None
        for db_id, channel in self.__class__._debate_channels.items():
            if channel.id == message.channel.id:
                debate_id = db_id
                break

        if debate_id and debate_id in self.__class__._human_participants:
            # This is a debate channel with human participants
            if message.author.id in self.__class__._human_participants[debate_id]:
                # Human participant is speaking - forward to relevant agents
                await self._handle_human_debate_message(message, debate_id)

        # Check if mentioned
        if self.user in message.mentions:
            try:
                # Remove mention from content
                content = message.content.replace(f'<@{self.user.id}>', '').strip()

                if content:
                    # Direct analysis request
                    async with message.channel.typing():
                        response = await self.agent.analyze(content)

                    embed = discord.Embed(
                        title=f"{self.agent_name} Response",
                        description=str(response)[:2000],
                        color=self.interface_config['color'],
                        timestamp=datetime.now()
                    )
                    await message.reply(embed=embed)
            except Exception as e:
                await message.reply(f"❌ Error processing request: {str(e)}")

        # Process commands
        await self.process_commands(message)

    async def _handle_human_debate_message(self, message: discord.Message, debate_id: str):
        """Handle human messages in debate channels"""
        try:
            # Create embed for human input in debate
            embed = discord.Embed(
                title="💭 Human Input in Debate",
                description=message.content,
                color=0x3498db,
                timestamp=datetime.now()
            )
            embed.set_author(
                name=message.author.display_name,
                icon_url=message.author.avatar.url if message.author.avatar else None
            )
            embed.set_footer(text=f"Debate ID: {debate_id}")

            # Forward to all participating agents
            forwarded_count = 0
            for bot in self.__class__._active_interfaces:
                try:
                    if bot.status_channel_id:
                        channel = bot.get_channel(bot.status_channel_id)
                        if channel and isinstance(channel, discord.TextChannel):
                            # Create agent-specific version
                            agent_embed = embed.copy()
                            agent_embed.title = f"💭 Human Debate Input - {message.author.display_name}"
                            agent_embed.add_field(
                                name="Context",
                                value=f"Message in debate channel {message.channel.mention}",
                                inline=False
                            )
                            await channel.send(embed=agent_embed)
                            forwarded_count += 1
                except Exception as e:
                    logger.error(f"Failed to forward human message to {bot.agent_name}: {e}")

            # Acknowledge in the debate channel
            if forwarded_count > 0:
                await message.add_reaction("✅")  # Confirm message was forwarded

        except Exception as e:
            logger.error(f"Error handling human debate message: {e}")
            await message.add_reaction("❌")  # Indicate error

        @self.command(name='agent_vote')
        async def agent_vote(ctx, proposal: str, *options):
            """Create a voting poll for agents on a proposal"""
            try:
                if not options:
                    await ctx.send("❌ Please provide voting options. Usage: `!agent_vote 'Proposal text' 'Option 1' 'Option 2' 'Option 3'`")
                    return
                
                embed = discord.Embed(
                    title="🗳️ Agent Voting Poll",
                    description=f"**Proposal:** {proposal}",
                    color=0x3498db,
                    timestamp=datetime.now()
                )
                embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.avatar.url if ctx.author.avatar else None)
                
                # Add voting options
                option_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']
                for i, option in enumerate(options[:10]):  # Limit to 10 options
                    embed.add_field(
                        name=f"{option_emojis[i]} Option {i+1}",
                        value=option,
                        inline=False
                    )
                
                poll_msg = await ctx.send(embed=embed)
                
                # Add reactions for voting
                for i in range(min(len(options), 10)):
                    await poll_msg.add_reaction(option_emojis[i])
                
                # Notify agents of the poll
                for bot in self.__class__._active_interfaces:
                    try:
                        if bot.status_channel_id:
                            channel = bot.get_channel(bot.status_channel_id)
                            if channel and isinstance(channel, discord.TextChannel):
                                agent_embed = discord.Embed(
                                    title="🗳️ Voting Poll Available",
                                    description=f"A voting poll has been created by {ctx.author.mention}",
                                    color=bot.interface_config['color'],
                                    timestamp=datetime.now()
                                )
                                agent_embed.add_field(name="Proposal", value=proposal, inline=False)
                                agent_embed.add_field(name="Channel", value=ctx.channel.mention, inline=True)
                                agent_embed.add_field(name="Options", value=", ".join(options), inline=False)
                                await channel.send(embed=agent_embed)
                    except Exception as e:
                        logger.error(f"Failed to notify {bot.agent_name} of poll: {e}")
                
                await ctx.send("✅ Poll created and agents notified!")
                
            except Exception as e:
                await ctx.send(f"❌ Error creating poll: {str(e)}")

        @self.command(name='request_analysis')
        async def request_analysis(ctx, analysis_type: str, *, details: str = ""):
            """Request specific type of analysis from relevant agents"""
            try:
                analysis_types = {
                    'macro': ['macro'],
                    'data': ['data'],
                    'strategy': ['strategy'],
                    'risk': ['risk'],
                    'technical': ['data'],
                    'fundamental': ['data'],
                    'sentiment': ['data'],
                    'execution': ['execution'],
                    'performance': ['reflection', 'learning'],
                    'all': ['macro', 'data', 'strategy', 'risk', 'reflection', 'execution', 'learning']
                }
                
                target_roles = analysis_types.get(analysis_type.lower())
                if not target_roles:
                    available_types = ", ".join(analysis_types.keys())
                    await ctx.send(f"❌ Unknown analysis type. Available: {available_types}")
                    return
                
                embed = discord.Embed(
                    title=f"🔍 {analysis_type.title()} Analysis Request",
                    description=f"**Requested by:** {ctx.author.mention}\n**Details:** {details or 'General analysis requested'}",
                    color=0x9b59b6,
                    timestamp=datetime.now()
                )
                
                await ctx.send(embed=embed)
                
                # Notify relevant agents
                notified_count = 0
                for bot in self.__class__._active_interfaces:
                    if bot.agent_role in target_roles:
                        try:
                            if bot.status_channel_id:
                                channel = bot.get_channel(bot.status_channel_id)
                                if channel and isinstance(channel, discord.TextChannel):
                                    agent_embed = discord.Embed(
                                        title=f"🔍 {analysis_type.title()} Analysis Requested",
                                        description=f"Analysis request from {ctx.author.mention} in {ctx.channel.mention}",
                                        color=bot.interface_config['color'],
                                        timestamp=datetime.now()
                                    )
                                    if details:
                                        agent_embed.add_field(name="Details", value=details, inline=False)
                                    await channel.send(embed=agent_embed)
                                    notified_count += 1
                        except Exception as e:
                            logger.error(f"Failed to notify {bot.agent_name}: {e}")
                
                await ctx.send(f"📨 Analysis request sent to {notified_count} relevant agents!")
                
            except Exception as e:
                await ctx.send(f"❌ Error requesting analysis: {str(e)}")

        @self.command(name='create_poll')
        async def create_poll(ctx, question: str, *options):
            """Create a Discord poll for collaborative decision-making"""
            try:
                if len(options) < 2:
                    await ctx.send("❌ Please provide at least 2 options. Usage: `!create_poll 'Question?' 'Option 1' 'Option 2' 'Option 3'`")
                    return
                
                if len(options) > 10:
                    await ctx.send("❌ Maximum 10 options allowed for polls.")
                    return
                
                # Create the poll using Discord's native polling feature
                poll = await ctx.channel.create_poll(
                    question=question,
                    options=options,
                    duration=3600,  # 1 hour default
                    multiple=False  # Single choice poll
                )
                
                # Create embed with poll details
                embed = discord.Embed(
                    title="📊 Poll Created",
                    description=f"**Question:** {question}",
                    color=self.interface_config['color'],
                    timestamp=datetime.now()
                )
                embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.avatar.url if ctx.author.avatar else None)
                embed.add_field(name="Duration", value="1 hour", inline=True)
                embed.add_field(name="Options", value=str(len(options)), inline=True)
                embed.add_field(name="Multiple Choice", value="No", inline=True)
                
                # List all options
                options_text = "\n".join([f"• {opt}" for opt in options])
                embed.add_field(name="Choices", value=options_text, inline=False)
                
                await ctx.send(embed=embed)
                
                # Notify all agents about the poll
                for bot in self.__class__._active_interfaces:
                    try:
                        if bot.status_channel_id and bot != self:  # Don't notify ourselves
                            channel = bot.get_channel(bot.status_channel_id)
                            if channel and isinstance(channel, discord.TextChannel):
                                agent_embed = discord.Embed(
                                    title="📊 New Poll Available",
                                    description=f"A poll has been created by {ctx.author.mention} in {ctx.channel.mention}",
                                    color=bot.interface_config['color'],
                                    timestamp=datetime.now()
                                )
                                agent_embed.add_field(name="Question", value=question, inline=False)
                                agent_embed.add_field(name="Options", value=", ".join(options), inline=False)
                                agent_embed.add_field(name="Channel", value=ctx.channel.mention, inline=True)
                                agent_embed.add_field(name="Duration", value="1 hour", inline=True)
                                await channel.send(embed=agent_embed)
                    except Exception as e:
                        logger.error(f"Failed to notify {bot.agent_name} of poll: {e}")
                
                # Optional: Mention everyone if it's an important decision
                if "@important" in question.lower() or "@critical" in question.lower():
                    try:
                        await ctx.send("@everyone An important poll has been created! Please vote.")
                    except Exception as e:
                        logger.error(f"Failed to mention everyone: {e}")
                
            except Exception as e:
                await ctx.send(f"❌ Error creating poll: {str(e)}")

        @self.command(name='alert_everyone')
        async def alert_everyone(ctx, *, message: str):
            """Send an alert to everyone (admin/reflection agent only)"""
            try:
                # Only allow admins or the reflection agent to use this
                is_admin = ctx.author.guild_permissions.administrator
                is_reflection = self.agent_role == 'reflection'
                
                if not (is_admin or is_reflection):
                    await ctx.send("❌ This command requires administrator permissions or Reflection Agent access")
                    return
                
                embed = discord.Embed(
                    title="🚨 IMPORTANT ALERT",
                    description=message,
                    color=0xe74c3c,
                    timestamp=datetime.now()
                )
                embed.set_author(
                    name=f"{self.agent_name} Alert",
                    icon_url=self.user.avatar.url if self.user and self.user.avatar else None
                )
                
                # Send the alert with @everyone mention
                alert_msg = f"@everyone **CRITICAL ALERT from {self.agent_name}:**\n\n"
                await ctx.send(alert_msg, embed=embed)
                
                # Log the alert
                logger.warning(f"Alert sent by {ctx.author.display_name}: {message}")
                
            except Exception as e:
                await ctx.send(f"❌ Error sending alert: {str(e)}")

        @self.command(name='system_health')
        async def system_health(ctx):
            """Check overall system health and agent status"""
            try:
                embed = discord.Embed(
                    title="🏥 System Health Check",
                    color=0x2ecc71,
                    timestamp=datetime.now()
                )
                
                # Agent status
                healthy_agents = 0
                total_agents = len(self.__class__._active_interfaces)
                
                agent_statuses = []
                for bot in self.__class__._active_interfaces:
                    try:
                        # Try to get agent status
                        status = await bot.agent.get_status()
                        health = status.get('health', 'unknown')
                        if health in ['good', 'healthy', 'online']:
                            healthy_agents += 1
                            agent_statuses.append(f"✅ {bot.agent_name}")
                        else:
                            agent_statuses.append(f"⚠️ {bot.agent_name} ({health})")
                    except Exception as e:
                        agent_statuses.append(f"❌ {bot.agent_name} (error)")
                
                embed.add_field(
                    name="Agent Health",
                    value=f"{healthy_agents}/{total_agents} agents healthy",
                    inline=True
                )
                
                embed.add_field(
                    name="Active Debates",
                    value=str(len(self.__class__._debate_channels)),
                    inline=True
                )
                
                embed.add_field(
                    name="Human Participants",
                    value=str(sum(len(participants) for participants in self.__class__._human_participants.values())),
                    inline=True
                )
                
                # Detailed agent statuses
                if agent_statuses:
                    status_text = "\n".join(agent_statuses[:10])  # Limit to 10
                    if len(agent_statuses) > 10:
                        status_text += f"\n... and {len(agent_statuses) - 10} more"
                    embed.add_field(
                        name="Agent Details",
                        value=status_text,
                        inline=False
                    )
                
                # System metrics
                embed.add_field(
                    name="System Metrics",
                    value=f"• Memory channels: {len(self.__class__._debate_channels)}\n• Active discussions: {len(self.__class__._human_participants)}",
                    inline=False
                )
                
                await ctx.send(embed=embed)
                
            except Exception as e:
                await ctx.send(f"❌ Error checking system health: {str(e)}")

        @self.command(name='emergency_stop')
        async def emergency_stop(ctx):
            """Emergency stop all agent activities (admin only)"""
            try:
                # Check if user has admin permissions (you might want to customize this)
                if not ctx.author.guild_permissions.administrator:
                    await ctx.send("❌ This command requires administrator permissions")
                    return
                
                embed = discord.Embed(
                    title="🚨 EMERGENCY STOP ACTIVATED",
                    description=f"**Initiated by:** {ctx.author.mention}\n**All agent activities halted**",
                    color=0xe74c3c,
                    timestamp=datetime.now()
                )
                
                await ctx.send(embed=embed)
                
                # Notify all agents of emergency stop
                for bot in self.__class__._active_interfaces:
                    try:
                        if bot.status_channel_id:
                            channel = bot.get_channel(bot.status_channel_id)
                            if channel and isinstance(channel, discord.TextChannel):
                                emergency_embed = discord.Embed(
                                    title="🚨 EMERGENCY STOP",
                                    description="All activities halted by administrator command",
                                    color=0xe74c3c,
                                    timestamp=datetime.now()
                                )
                                await channel.send(embed=emergency_embed)
                    except Exception as e:
                        logger.error(f"Failed to notify {bot.agent_name} of emergency stop: {e}")
                
                # Clear debates and participants
                self.__class__._debate_channels.clear()
                self.__class__._human_participants.clear()
                
                await ctx.send("🛑 Emergency stop complete. All debates cleared.")
                
            except Exception as e:
                await ctx.send(f"❌ Error executing emergency stop: {str(e)}")


class DiscordInterfaceSystem:
    """
    System for managing multiple Discord bot interfaces.
    Coordinates multiple agent interfaces and provides system-level commands.
    """

    def __init__(self):
        self.agents_config = self.load_config()
        self.interfaces: List[DiscordBotInterface] = []
        self.agent_instances: Dict[str, BaseAgent] = {}
        
        # Initialize A2A protocol with Discord monitoring
        self.a2a_protocol = A2AProtocol(max_agents=50)
        self.monitoring_channel_id = os.getenv('DISCORD_A2A_MONITORING_CHANNEL_ID')

    def load_config(self) -> Dict[str, Any]:
        """Load Discord agent configuration from environment variables"""
        try:
            # Load configuration from environment variables
            agents_config = {
                "macro": {
                    "name": "Macro Analyst",
                    "role": "macro",
                    "token": os.getenv('DISCORD_MACRO_AGENT_TOKEN', 'YOUR_MACRO_BOT_TOKEN'),
                    "color": 0x3498db,
                    "status_channel_id": None,
                    "command_prefix": "!m"
                },
                "data": {
                    "name": "Data Collector",
                    "role": "data",
                    "token": os.getenv('DISCORD_DATA_AGENT_TOKEN', 'YOUR_DATA_BOT_TOKEN'),
                    "color": 0x2ecc71,
                    "status_channel_id": None,
                    "command_prefix": "!d"
                },
                "strategy": {
                    "name": "Strategy Advisor",
                    "role": "strategy",
                    "token": os.getenv('DISCORD_STRATEGY_AGENT_TOKEN', 'YOUR_STRATEGY_BOT_TOKEN'),
                    "color": 0xe67e22,
                    "status_channel_id": None,
                    "command_prefix": "!s"
                },
                "reflection": {
                    "name": "Reflection Agent",
                    "role": "reflection",
                    "token": os.getenv('DISCORD_REFLECTION_AGENT_TOKEN', 'YOUR_REFLECTION_BOT_TOKEN'),
                    "color": 0x9b59b6,
                    "status_channel_id": None,
                    "command_prefix": "!ref"
                },
                "execution": {
                    "name": "Trade Executor",
                    "role": "execution",
                    "token": os.getenv('DISCORD_EXECUTION_AGENT_TOKEN', 'YOUR_EXECUTION_BOT_TOKEN'),
                    "color": 0x1abc9c,
                    "status_channel_id": None,
                    "command_prefix": "!exec"
                }
            }

            # Conditionally add risk agent if token is available
            if os.getenv('DISCORD_RISK_AGENT_TOKEN'):
                agents_config["risk"] = {
                    "name": "Risk Manager",
                    "role": "risk",
                    "token": os.getenv('DISCORD_RISK_AGENT_TOKEN'),
                    "color": 0xe74c3c,
                    "status_channel_id": None,
                    "command_prefix": "!r"
                }

            # Conditionally add learning agent if token is available
            if os.getenv('DISCORD_LEARNING_AGENT_TOKEN'):
                agents_config["learning"] = {
                    "name": "Learning Agent",
                    "role": "learning",
                    "token": os.getenv('DISCORD_LEARNING_AGENT_TOKEN'),
                    "color": 0xf39c12,
                    "status_channel_id": None,
                    "command_prefix": "!l"
                }

            config = {
                "guild_id": os.getenv('DISCORD_GUILD_ID', 'YOUR_GUILD_ID'),
                "agents": agents_config
            }

            logger.info("Loaded Discord configuration from environment variables")
            return config

        except Exception as e:
            logger.error(f"Error loading Discord configuration: {str(e)}")
            # Return default config as fallback
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration as fallback"""
        return {
            "guild_id": "YOUR_GUILD_ID",
            "agents": {
                "macro": {
                    "name": "Macro Analyst",
                    "role": "macro",
                    "token": "YOUR_MACRO_BOT_TOKEN",
                    "color": 0x3498db,
                    "status_channel_id": None,
                    "command_prefix": "!m"
                },
                "data": {
                    "name": "Data Collector",
                    "role": "data",
                    "token": "YOUR_DATA_BOT_TOKEN",
                    "color": 0x2ecc71,
                    "status_channel_id": None,
                    "command_prefix": "!d"
                },
                "strategy": {
                    "name": "Strategy Advisor",
                    "role": "strategy",
                    "token": "YOUR_STRATEGY_BOT_TOKEN",
                    "color": 0xe67e22,
                    "status_channel_id": None,
                    "command_prefix": "!s"
                },
                "risk": {
                    "name": "Risk Manager",
                    "role": "risk",
                    "token": "YOUR_RISK_BOT_TOKEN",
                    "color": 0xe74c3c,
                    "status_channel_id": None,
                    "command_prefix": "!r"
                },
                "reflection": {
                    "name": "Reflection Agent",
                    "role": "reflection",
                    "token": "YOUR_REFLECTION_BOT_TOKEN",
                    "color": 0x9b59b6,
                    "status_channel_id": None,
                    "command_prefix": "!ref"
                },
                "execution": {
                    "name": "Trade Executor",
                    "role": "execution",
                    "token": "YOUR_EXECUTION_BOT_TOKEN",
                    "color": 0x1abc9c,
                    "status_channel_id": None,
                    "command_prefix": "!exec"
                },
                "learning": {
                    "name": "Learning Agent",
                    "role": "learning",
                    "token": "YOUR_LEARNING_BOT_TOKEN",
                    "color": 0xf39c12,
                    "status_channel_id": None,
                    "command_prefix": "!l"
                }
            }
        }

    async def initialize_agents(self):
        """Initialize all agent instances asynchronously"""
        logger.info("Initializing agents...")

        # Initialize each agent
        agent_classes = {
            'macro': MacroAgent,
            'data': DataAgent,
            'strategy': StrategyAgent,
            'risk': RiskAgent,
            'reflection': ReflectionAgent,
            'execution': ExecutionAgent,
            'learning': LearningAgent
        }

        for agent_key, interface_config in self.agents_config['agents'].items():
            try:
                agent_class = agent_classes.get(agent_key)
                if agent_class:
                    # Pass A2A protocol to each agent for monitored communication
                    agent_instance = agent_class(a2a_protocol=self.a2a_protocol)
                    
                    # Initialize LLM asynchronously
                    await agent_instance.async_initialize_llm()
                    
                    self.agent_instances[agent_key] = agent_instance
                    logger.info(f"Initialized {agent_key} agent with LLM and A2A protocol")
                else:
                    logger.warning(f"No agent class found for {agent_key}")
            except Exception as e:
                logger.error(f"Failed to initialize {agent_key} agent: {str(e)}")

    def create_interfaces(self):
        """Create Discord bot interfaces for each agent"""
        logger.info("Creating Discord bot interfaces...")

        for agent_key, interface_config in self.agents_config['agents'].items():
            try:
                agent_instance = self.agent_instances.get(agent_key)
                if agent_instance:
                    interface = DiscordBotInterface(agent_instance, interface_config['token'], interface_config)
                    self.interfaces.append(interface)
                    logger.info(f"Created interface for {agent_key} agent")
                else:
                    logger.warning(f"No agent instance found for {agent_key}")
            except Exception as e:
                logger.error(f"Failed to create interface for {agent_key}: {str(e)}")

    async def start_interfaces(self):
        """Start all Discord bot interfaces"""
        logger.info("Starting Discord bot interfaces...")

        tasks = []
        for interface in self.interfaces:
            task = asyncio.create_task(interface.start(interface.interface_config['token']))
            tasks.append(task)

        # Wait for all interfaces to start
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All Discord bot interfaces started")

    async def run(self):
        """Run the Discord interface system"""
        await self.initialize_agents()
        self.create_interfaces()
        
        # Configure A2A protocol with Discord monitoring
        if self.monitoring_channel_id and self.interfaces:
            # Use the first interface for monitoring (they all have access to the same channels)
            self.a2a_protocol.set_discord_monitoring(self.interfaces[0], self.monitoring_channel_id)
            logger.info(f"A2A protocol configured with Discord monitoring on channel {self.monitoring_channel_id}")
        
        # Register agents with A2A protocol
        for agent_key, agent_instance in self.agent_instances.items():
            self.a2a_protocol.register_agent(agent_key, agent_instance)
            logger.info(f"Registered {agent_key} agent with A2A protocol")

        # Start all interfaces concurrently and let them run indefinitely
        logger.info("Starting Discord bot interfaces concurrently...")
        
        # Create tasks for all interfaces to start
        tasks = []
        for interface in self.interfaces:
            task = asyncio.create_task(interface.start(interface.interface_config['token']))
            tasks.append(task)
            logger.info(f"Created startup task for {interface.interface_config['name']}")

        # Let all interfaces run indefinitely - don't wait for completion
        logger.info("All Discord bot interfaces are now running. Press Ctrl+C to stop.")
        
        # Keep the event loop running forever
        try:
            while True:
                await asyncio.sleep(1)  # Check every second for any issues
        except KeyboardInterrupt:
            logger.info("Received shutdown signal, stopping interfaces...")
            # Gracefully stop all interfaces
            for interface in self.interfaces:
                try:
                    await interface.close()
                    logger.info(f"Stopped {interface.interface_config['name']}")
                except Exception as e:
                    logger.error(f"Error stopping {interface.interface_config['name']}: {e}")
            logger.info("All Discord bot interfaces have stopped")

if __name__ == "__main__":
    system = DiscordInterfaceSystem()
    asyncio.run(system.run())
