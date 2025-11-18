#!/usr/bin/env python3
"""
Test Discord Workflow Orchestrator Integration
Tests the full Discord integration with live workflow orchestration
"""
import asyncio
import os
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator

async def test_discord_workflow_orchestrator():
    """Test the Discord workflow orchestrator with full agent integration"""
    print("🎯 Testing Discord Workflow Orchestrator Integration...")

    # Initialize orchestrator
    orchestrator = LiveWorkflowOrchestrator()

    # Initialize agents
    print("🤖 Initializing agents...")
    await orchestrator.initialize_agents_async()
    print(f"✅ Initialized {len(orchestrator.agent_instances)} agents")

    # Check health
    health = await orchestrator.check_agent_health()
    print(f"🏥 Health check: {health['overall_health']} ({len(health['healthy_agents'])}/{health['total_agents']} healthy)")

    # Create collaborative session
    session_created = await orchestrator.create_collaborative_session('Discord Integration Test')
    print(f"🤝 Session created: {session_created}")

    # Test agent direct communication (without Discord UI)
    if orchestrator.agent_instances:
        print("📤 Testing direct agent communication...")

        # Test macro agent
        if 'macro' in orchestrator.agent_instances:
            macro_response = await orchestrator.send_direct_agent_command('macro', '!m analyze Test macro analysis for Discord integration')
            print(f"📥 Macro agent response: {'Success' if macro_response else 'Failed'}")

        # Test data agent
        if 'data' in orchestrator.agent_instances:
            data_response = await orchestrator.send_direct_agent_command('data', '!d analyze Test data collection for AAPL')
            print(f"📥 Data agent response: {'Success' if data_response else 'Failed'}")

        # Test strategy agent
        if 'strategy' in orchestrator.agent_instances:
            strategy_response = await orchestrator.send_direct_agent_command('strategy', '!s analyze Test strategy development')
            print(f"📥 Strategy agent response: {'Success' if strategy_response else 'Failed'}")

    # Test workflow phase execution (simulated)
    if session_created:
        print("⚙️ Testing workflow phase execution...")
        await orchestrator.execute_phase_with_agents('macro_foundation_data_collection', 'TEST: Discord Data Collection')
        print("✅ Phase execution completed")

    print("🎉 Discord Workflow Orchestrator test completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Start the Discord orchestrator: python src/agents/live_workflow_orchestrator.py")
    print("2. In Discord, use !start_workflow to begin live orchestration")
    print("3. Test human intervention during active workflows")
    print("4. Monitor agent responses in dedicated channels")

if __name__ == "__main__":
    asyncio.run(test_discord_workflow_orchestrator())