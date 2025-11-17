#!/usr/bin/env python3
"""
Live Workflow Orchestrator - Real-time Interactive Iterative Reasoning
Watches Discord and orchestrates the collaborative reasoning workflow automatically,
while allowing human intervention and questions during the process.
"""

import asyncio
import discord
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, cast
from dotenv import load_dotenv

load_dotenv()

class LiveWorkflowOrchestrator:
    """
    Orchestrates the iterative reasoning workflow in real-time on Discord,
    allowing human participation and intervention during the process.
    """

    def __init__(self):
        self.client = None
        self.channel = None
        self.workflow_active = False
        self.current_phase = "waiting"
        self.phase_commands = {}
        self.responses_collected = []
        self.human_interventions = []
        self.workflow_log = []
        self.phase_delays = {
            'macro_foundation': 15,  # 15 seconds for macro analysis
            'intelligence_gathering': 12,  # 12 seconds per command
            'strategy_development': 12,
            'debate': 25,  # Longer for debates
            'risk_assessment': 12,
            'consensus': 15,
            'execution_validation': 10,
            'learning': 10,
            'executive_review': 12,
            'supreme_oversight': 20  # Longer for final decisions
        }

        self._initialize_workflow_commands()

    def _initialize_workflow_commands(self):
        """Initialize the command sequences for each workflow phase with realistic data-driven tasks"""
        self.phase_commands = {
            'macro_foundation': [
                "!m analyze Fetch current SPY, QQQ, IWM prices and calculate market breadth indicators (advancers/decliners, new highs/lows)",
                "!m analyze Pull economic data: Fed Funds Rate, Treasury yields (2Y/10Y/30Y), VIX, USD index, oil/gold prices",
                "!m analyze Calculate sector ETF performance: XLY, XLC, XLF, XLB, XLE, XLK, XLU, XLV, XLRE, XLP, XLI vs SPY",
                "!m analyze Assess market regime: Identify if we're in risk-on/risk-off, trending/range-bound, bull/bear market",
                "!m analyze Generate top 5 sector opportunities based on relative strength, momentum, and risk-adjusted returns"
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
                '!m debate "Evaluate strategy robustness: Which approaches work best in current regime? Consider alternatives." strategy risk data execution',
                '!m debate "Risk-reward analysis: Which strategies offer best asymmetric opportunities? Challenge assumptions." strategy risk reflection',
                '!m debate "Market timing and execution: When to enter/exit? What are the practical constraints?" strategy execution data'
            ],

            'risk_assessment': [
                "!r analyze Calculate Value at Risk (VaR) for each strategy using historical simulation and parametric methods",
                "!r analyze Assess tail risk: Black Swan scenarios, correlation breakdowns, liquidity crunch possibilities",
                "!r analyze Evaluate strategy drawdown potential and maximum loss scenarios under stress conditions",
                "!r analyze Consider systemic risks: Fed policy changes, geopolitical events, economic data surprises",
                "!r analyze Generate risk-adjusted return metrics and Sharpe/Sortino ratios for strategy comparison"
            ],

            'consensus': [
                "!ref analyze Synthesize all agent inputs: Which strategies pass risk/reward hurdles? What are the trade-offs?",
                "!ref analyze Evaluate strategy consensus: Where do all agents agree? Where are there material disagreements?",
                "!ref analyze Consider implementation feasibility: Capital requirements, slippage costs, execution complexity",
                "!ref analyze Assess market capacity: Can strategies be scaled without moving markets or exhausting liquidity?",
                "!ref analyze Generate final strategy rankings with confidence levels and implementation priorities"
            ],

            'execution_validation': [
                "!e analyze Model transaction costs: Commissions, spreads, market impact for proposed position sizes",
                "!e analyze Assess execution logistics: Trading hours, venue selection, algorithmic execution requirements",
                "!e analyze Evaluate position management: Rebalancing triggers, scaling in/out protocols, exit strategies",
                "!e analyze Consider tax implications and wash sale rules for strategy implementation",
                "!e analyze Generate detailed execution playbook with step-by-step implementation guide"
            ],

            'learning': [
                "!l analyze Review historical performance of similar strategies in current market regime",
                "!l analyze Identify key success factors and common failure modes from past implementations",
                "!l analyze Update strategy templates and decision frameworks based on current analysis",
                "!l analyze Document lessons learned and update institutional knowledge base",
                "!l analyze Generate strategy improvement recommendations for future workflow iterations"
            ],

            'executive_review': [
                "!ref analyze Executive summary: Present top 3 strategies with risk/reward profiles and implementation confidence",
                "!r analyze Conservative stress testing: Apply more stringent risk thresholds and worst-case assumptions",
                "!s analyze Strategic implications: How do these strategies fit broader portfolio objectives and market outlook?",
                "!e analyze Operational readiness: Team capabilities, system requirements, compliance considerations",
                "!l analyze Historical precedent analysis: Similar strategies in comparable market conditions and outcomes"
            ],

            'supreme_oversight': [
                "!ref analyze Supreme audit: Review all data sources, assumptions, and analytical methods for completeness",
                "!ref analyze Scenario stress testing: Evaluate strategies against Fed hikes, earnings misses, geopolitical shocks",
                "!ref analyze Crisis detection: Identify 'canary in the coal mine' indicators and potential black swan events",
                "!ref analyze Logical consistency check: Ensure all conclusions follow from evidence and reasoning chains",
                "!ref analyze Final veto authority: Approve/reject strategies or mandate additional analysis if warranted"
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

            # Find the target channel
            guild_id = int(os.getenv('DISCORD_GUILD_ID', '0'))
            if guild_id and self.client:
                guild = self.client.get_guild(guild_id)
                if guild:
                    # Try to find workflow channel, fallback to general
                    for ch in guild.text_channels:
                        if ch.name in ['general', 'workflow', 'analysis', 'trading']:
                            self.channel = ch
                            print(f"📝 Orchestrating in: #{ch.name}")
                            break

                    if not self.channel and guild.text_channels:
                        self.channel = guild.text_channels[0]
                        print(f"📝 Using default channel: #{self.channel.name}")

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

            # Handle workflow control commands
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

            # Collect agent responses (from bots)
            if message.author.bot and message.author != self.client.user:
                self.responses_collected.append({
                    'agent': message.author.display_name,
                    'content': message.content,
                    'timestamp': message.created_at.isoformat(),
                    'phase': self.current_phase
                })
                self.workflow_log.append(f"🤖 {message.author.display_name}: {message.content[:100]}...")

            # Handle human questions/interventions during active workflow
            elif self.workflow_active and not message.author.bot:
                await self.handle_human_intervention(message)

        token = os.getenv('DISCORD_MACRO_AGENT_TOKEN')  # Use macro token for orchestration
        if token:
            await self.client.start(token)
        else:
            raise ValueError("❌ No Discord token found for orchestrator")

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
            await asyncio.sleep(10)

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

        self.workflow_active = True
        self.current_phase = "starting"
        self.workflow_log = []
        self.responses_collected = []
        self.human_interventions = []

        await channel.send("🚀 **STARTING ITERATIVE REASONING WORKFLOW**")
        await channel.send("📊 This will run through 2 iterations with supreme oversight")
        await channel.send("💡 You can ask questions or intervene at any time!")

        # Phase 0: Macro Foundation
        await self.execute_phase('macro_foundation', "🏛️ PHASE 0: MACRO FOUNDATION")

        # Iteration 1: Comprehensive Deliberation
        await channel.send("\n🔄 **ITERATION 1: COMPREHENSIVE MULTI-AGENT DELIBERATION**")

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
            await self.execute_phase(phase_key, phase_title)

        # Iteration 2: Executive Oversight
        await channel.send("\n👔 **ITERATION 2: EXECUTIVE-LEVEL STRATEGIC OVERSIGHT**")
        await self.execute_phase('executive_review', "🎯 Executive Strategic Review")

        # Supreme Oversight
        await channel.send("\n👑 **SUPREME OVERSIGHT**")
        await self.execute_phase('supreme_oversight', "🎯 Final Reflection & Decision")

        # Complete workflow
        await self.complete_workflow()

    async def execute_phase(self, phase_key: str, phase_title: str):
        """Execute a single phase of the workflow"""
        if not self.channel:
            print(f"❌ No channel available for phase {phase_key}")
            return

        channel = cast(discord.TextChannel, self.channel)
        self.current_phase = phase_key

        await channel.send(f"\n{phase_title}")
        await channel.send("─" * 50)

        commands = self.phase_commands.get(phase_key, [])
        delay = self.phase_delays.get(phase_key, 10)

        for i, command in enumerate(commands, 1):
            if not self.workflow_active:
                return  # Allow for pausing/stopping

            await channel.send(f"📤 **Command {i}/{len(commands)}:** `{command}`")

            # Send the command
            await channel.send(command)

            # Wait for responses
            await channel.send(f"⏳ Waiting {delay} seconds for responses...")
            await asyncio.sleep(delay)

            # Check for responses
            phase_responses = [r for r in self.responses_collected if r['phase'] == phase_key]
            await channel.send(f"📥 Received {len(phase_responses)} responses so far in this phase")

        # Phase complete
        await channel.send(f"✅ **{phase_title} Complete!**")

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
        await channel.send("📊 **Final Summary:**")
        await channel.send(f"• Total Agent Responses: {len(self.responses_collected)}")
        await channel.send(f"• Human Interventions: {len(self.human_interventions)}")
        await channel.send(f"• Phases Completed: All 10 phases")

        # Save results
        results = {
            'completed_at': datetime.now().isoformat(),
            'total_responses': len(self.responses_collected),
            'human_interventions': len(self.human_interventions),
            'responses': self.responses_collected,
            'interventions': self.human_interventions,
            'workflow_log': self.workflow_log
        }

        with open('live_workflow_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        await channel.send("💾 Results saved to `live_workflow_results.json`")
        await channel.send("\n🔄 Ready for next workflow! Type `!start_workflow` to begin again.")

    async def run_orchestrator(self):
        """Run the live workflow orchestrator"""
        try:
            await self.initialize_discord_client()
        except KeyboardInterrupt:
            print("\n🛑 Orchestrator shutting down...")
            if self.client:
                await self.client.close()
        except Exception as e:
            print(f"❌ Orchestrator error: {e}")
            if self.client:
                await self.client.close()

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