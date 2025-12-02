#!/usr/bin/env python3
"""
Iterative Reasoning Workflow Implementation
Implements the 22-agent collaborative reasoning framework through orchestrated Discord commands
"""

import asyncio
import discord
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class IterativeReasoningWorkflow:
    """
    Implements the collaborative reasoning workflow with macro foundation and two iterations
    """

    def __init__(self):
        self.client = None
        self.channel = None
        self.workflow_state = {}
        self.responses = []

    async def initialize_discord_client(self):
        """Initialize Discord client for workflow orchestration"""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready():
            print(f"🤖 Workflow orchestrator connected as {self.client.user}")

            # Find the target channel
            guild = self.client.get_guild(int(os.getenv('DISCORD_GUILD_ID', '0')))
            if guild:
                # Try to find a workflow or general channel
                for ch in guild.text_channels:
                    if ch.name in ['general', 'workflow', 'analysis', 'trading']:
                        self.channel = ch
                        print(f"📝 Using channel: #{ch.name}")
                        break

                if not self.channel:
                    self.channel = guild.text_channels[0]  # Use first available
                    print(f"📝 Using default channel: #{self.channel.name}")

        @self.client.event
        async def on_message(message):
            # Collect responses from agents (exclude our own messages)
            if message.author != self.client.user and not message.author.bot:
                return

            if message.author.bot and message.author != self.client.user:
                self.responses.append({
                    'agent': message.author.display_name,
                    'content': message.content,
                    'timestamp': message.created_at.isoformat()
                })

        token = os.getenv('DISCORD_ORCHESTRATOR_TOKEN')  # Use orchestrator token for all agents
        if token:
            await self.client.start(token)
        else:
            raise ValueError("No Discord orchestrator token found")

    async def send_command_and_wait(self, command: str, wait_seconds: int = 5) -> List[Dict]:
        """Send a command and wait for agent responses"""
        if not self.channel:
            print("❌ No channel available")
            return []

        print(f"📤 Sending: {command}")
        await self.channel.send(command)

        # Clear previous responses
        self.responses = []

        # Wait for responses
        await asyncio.sleep(wait_seconds)

        responses = self.responses.copy()
        print(f"📥 Received {len(responses)} responses")

        for resp in responses:
            print(f"  🤖 {resp['agent']}: {resp['content'][:100]}...")

        return responses

    async def execute_macro_foundation(self) -> Dict[str, Any]:
        """
        Phase 0: Macro Foundation - Market Regime Assessment & Opportunity Identification
        """
        print("\n🏛️  PHASE 0: MACRO FOUNDATION")
        print("=" * 50)

        # MacroAgent establishes strategic foundation
        responses = await self.send_command_and_wait(
            "!m analyze Assess current market regime, volatility levels, and macroeconomic trends. Identify top 5 sectors/assets with highest relative strength, momentum, and risk-adjusted returns.",
            wait_seconds=8
        )

        macro_context = {
            'market_regime': 'extracted from macro analysis',
            'top_opportunities': ['extracted sectors/assets'],
            'risk_environment': 'baseline risk parameters',
            'responses': responses
        }

        print("✅ Macro foundation established")
        return macro_context

    async def execute_iteration_1_comprehensive(self, macro_context: Dict) -> Dict[str, Any]:
        """
        Iteration 1: Comprehensive Multi-Agent Deliberation (All 7 Agents)
        7-Phase Process
        """
        print("\n🔄 ITERATION 1: COMPREHENSIVE MULTI-AGENT DELIBERATION")
        print("=" * 60)

        iteration_results = {
            'phase_1_intelligence': [],
            'phase_2_strategy_dev': [],
            'phase_3_debate': [],
            'phase_4_risk_assessment': [],
            'phase_5_consensus': [],
            'phase_6_execution_validation': [],
            'phase_7_learning': []
        }

        # Phase 1: Integrated Intelligence Gathering & Analysis
        print("\n📊 Phase 1: Integrated Intelligence Gathering & Analysis")
        commands = [
            "!d analyze Gather and validate multi-source market data for the identified opportunities",
            "!m analyze Provide market regime context and sector analysis",
            "!s analyze Begin forming initial hypotheses based on market data",
            "!r analyze Evaluate data quality and identify potential risk signals"
        ]

        for cmd in commands:
            responses = await self.send_command_and_wait(cmd, wait_seconds=6)
            iteration_results['phase_1_intelligence'].extend(responses)

        # Phase 2: Collaborative Strategy Development
        print("\n🎯 Phase 2: Collaborative Strategy Development")
        commands = [
            "!s analyze Develop comprehensive trading strategies informed by complete intelligence picture",
            "!d analyze Provide specific insights and validation for proposed approaches",
            "!r analyze Integrate risk constraints and probability assessments into strategy design",
            "!e analyze Evaluate practical feasibility and market impact considerations"
        ]

        for cmd in commands:
            responses = await self.send_command_and_wait(cmd, wait_seconds=6)
            iteration_results['phase_2_strategy_dev'].extend(responses)

        # Phase 3: Comprehensive Multi-Agent Debate & Challenge
        print("\n⚔️  Phase 3: Comprehensive Multi-Agent Debate & Challenge")
        debate_topic = f"Debate the proposed strategies considering market regime {macro_context.get('market_regime', 'analysis')}"
        responses = await self.send_command_and_wait(
            f"!m debate \"{debate_topic}\" strategy risk reflection execution",
            wait_seconds=10
        )
        iteration_results['phase_3_debate'].extend(responses)

        # Phase 4: Integrated Risk Assessment & Strategy Refinement
        print("\n⚠️  Phase 4: Integrated Risk Assessment & Strategy Refinement")
        commands = [
            "!r analyze Conduct comprehensive probabilistic analysis and risk assessment",
            "!s analyze Refine strategies based on risk analysis and agent feedback",
            "!m analyze Ensure strategies align with broader market regime analysis"
        ]

        for cmd in commands:
            responses = await self.send_command_and_wait(cmd, wait_seconds=6)
            iteration_results['phase_4_risk_assessment'].extend(responses)

        # Phase 5: Consensus Building & Decision Finalization
        print("\n🤝 Phase 5: Consensus Building & Decision Finalization")
        responses = await self.send_command_and_wait(
            "!ref analyze Synthesize all inputs and mediate conflicts to reach consensus on optimal strategies",
            wait_seconds=8
        )
        iteration_results['phase_5_consensus'].extend(responses)

        # Phase 6: Execution Validation & Final Review
        print("\n✅ Phase 6: Execution Validation & Final Review")
        commands = [
            "!e analyze Validate practical feasibility, timing, and market impact",
            "!m analyze Final sanity checks and market timing validation"
        ]

        for cmd in commands:
            responses = await self.send_command_and_wait(cmd, wait_seconds=6)
            iteration_results['phase_6_execution_validation'].extend(responses)

        # Phase 7: Learning Integration & Continuous Improvement
        print("\n🧠 Phase 7: Learning Integration & Continuous Improvement")
        responses = await self.send_command_and_wait(
            "!l analyze Incorporate outcomes into future reasoning processes and identify improvement areas",
            wait_seconds=6
        )
        iteration_results['phase_7_learning'].extend(responses)

        print("✅ Iteration 1 completed")
        return iteration_results

    async def execute_iteration_2_executive(self, iteration_1_results: Dict) -> Dict[str, Any]:
        """
        Iteration 2: Executive-Level Strategic Oversight (Main 8 Agents - adapted for 7 available)
        """
        print("\n👔 ITERATION 2: EXECUTIVE-LEVEL STRATEGIC OVERSIGHT")
        print("=" * 55)

        iteration_results = {
            'executive_synthesis': [],
            'risk_sensitivity': [],
            'strategic_focus': [],
            'implementation_focus': [],
            'historical_context': []
        }

        # Executive Synthesis
        print("\n🎯 Executive Synthesis")
        responses = await self.send_command_and_wait(
            "!ref analyze Synthesize comprehensive inputs into cohesive strategic narratives with elevated perspective",
            wait_seconds=8
        )
        iteration_results['executive_synthesis'].extend(responses)

        # Risk Sensitivity Amplification
        print("\n⚠️  Risk Sensitivity Amplification")
        responses = await self.send_command_and_wait(
            "!r analyze Apply more conservative probability thresholds and heightened risk sensitivity",
            wait_seconds=6
        )
        iteration_results['risk_sensitivity'].extend(responses)

        # Strategic Focus & Implementation
        print("\n📈 Strategic Focus & Implementation")
        commands = [
            "!s analyze Consider broader market implications and systemic risks",
            "!e analyze Emphasize practical constraints and market impact",
            "!l analyze Provide deeper pattern recognition and historical precedent analysis"
        ]

        for cmd in commands:
            responses = await self.send_command_and_wait(cmd, wait_seconds=6)
            iteration_results['strategic_focus'].extend(responses)

        print("✅ Iteration 2 completed")
        return iteration_results

    async def execute_reflection_supreme_oversight(self, all_results: Dict) -> Dict[str, Any]:
        """
        Reflection Agent's Supreme Oversight Authority
        """
        print("\n👑 REFLECTION AGENT'S SUPREME OVERSIGHT")
        print("=" * 45)

        oversight_results = {
            'data_audit': [],
            'scenario_stress_test': [],
            'pattern_recognition': [],
            'logical_consistency': [],
            'final_decision': []
        }

        # Comprehensive Data Audit
        print("\n🔍 Comprehensive Data Audit")
        responses = await self.send_command_and_wait(
            "!ref analyze Conduct comprehensive audit of all data points and analysis from both iterations",
            wait_seconds=8
        )
        oversight_results['data_audit'].extend(responses)

        # Scenario Stress Testing
        print("\n🌪️  Scenario Stress Testing")
        responses = await self.send_command_and_wait(
            "!ref analyze Evaluate decisions against multiple potential market scenarios and stress conditions",
            wait_seconds=8
        )
        oversight_results['scenario_stress_test'].extend(responses)

        # Pattern Recognition & Warning Signs
        print("\n🚨 Pattern Recognition & Warning Signs")
        responses = await self.send_command_and_wait(
            "!ref analyze Identify subtle warning signals, historical precedents, and potential catastrophic scenarios",
            wait_seconds=8
        )
        oversight_results['pattern_recognition'].extend(responses)

        # Logical Consistency Validation
        print("\n🧮 Logical Consistency Validation")
        responses = await self.send_command_and_wait(
            "!ref analyze Ensure all conclusions follow from established premises and logical reasoning",
            wait_seconds=8
        )
        oversight_results['logical_consistency'].extend(responses)

        # Final Decision with Veto Authority
        print("\n🎯 FINAL DECISION")
        responses = await self.send_command_and_wait(
            "!ref analyze Render final decision with authority to veto any strategy or mandate additional iterations if concerning patterns emerge",
            wait_seconds=10
        )
        oversight_results['final_decision'].extend(responses)

        print("✅ Supreme oversight completed")
        return oversight_results

    async def run_complete_workflow(self) -> Dict[str, Any]:
        """
        Execute the complete iterative reasoning workflow
        """
        print("🚀 STARTING COMPLETE ITERATIVE REASONING WORKFLOW")
        print("=" * 60)

        workflow_results = {}

        try:
            # Initialize Discord connection
            await self.initialize_discord_client()

            # Phase 0: Macro Foundation
            workflow_results['macro_foundation'] = await self.execute_macro_foundation()

            # Iteration 1: Comprehensive Deliberation
            workflow_results['iteration_1'] = await self.execute_iteration_1_comprehensive(
                workflow_results['macro_foundation']
            )

            # Iteration 2: Executive Oversight
            workflow_results['iteration_2'] = await self.execute_iteration_2_executive(
                workflow_results['iteration_1']
            )

            # Supreme Oversight
            workflow_results['supreme_oversight'] = await self.execute_reflection_supreme_oversight(
                workflow_results
            )

            print("\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            print("=" * 60)

            # Summary
            total_responses = sum(len(phase_results) for iteration in workflow_results.values()
                                for phase_results in iteration.values() if isinstance(phase_results, list))

            print(f"📊 Workflow Summary:")
            print(f"   • Macro Foundation: ✅ Completed")
            print(f"   • Iteration 1 (7 phases): ✅ Completed")
            print(f"   • Iteration 2 (Executive): ✅ Completed")
            print(f"   • Supreme Oversight: ✅ Completed")
            print(f"   • Total Agent Responses: {total_responses}")

        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            workflow_results['error'] = str(e)

        finally:
            if self.client:
                await self.client.close()

        return workflow_results

async def main():
    """Run the complete iterative reasoning workflow"""
    workflow = IterativeReasoningWorkflow()
    results = await workflow.run_complete_workflow()

    # Save results to file
    import json
    with open('workflow_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Results saved to workflow_results.json")

if __name__ == "__main__":
    asyncio.run(main())