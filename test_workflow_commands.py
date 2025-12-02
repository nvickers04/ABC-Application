#!/usr/bin/env python3
"""
Test script for workflow commands without Discord
"""
import asyncio
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator

async def test_workflow_commands():
    print('🧪 Testing workflow commands...')

    # Initialize orchestrator
    orchestrator = LiveWorkflowOrchestrator()

    # Initialize agents
    await orchestrator.initialize_agents_async()
    print(f'✅ Initialized {len(orchestrator.agent_instances)} agents')

    # Test health check
    health = await orchestrator.check_agent_health()
    print(f'🏥 Health check: {health["overall_health"]} ({len(health["healthy_agents"])}/{health["total_agents"]} healthy)')

    # Test workflow commands by calling the methods directly
    print('\n🧪 Testing !start_workflow command...')

    # Mock a channel for testing
    class MockChannel:
        async def send(self, message):
            print(f'📢 Channel message: {message}')

    orchestrator.channel = MockChannel()

    # Test start_workflow
    try:
        await orchestrator.start_workflow()
        print('✅ !start_workflow command executed successfully')
        print(f'🔄 Workflow active: {orchestrator.workflow_active}')
        print(f'📊 Current phase: {orchestrator.current_phase}')
    except Exception as e:
        print(f'❌ !start_workflow failed: {e}')

    # Test pause/resume if workflow is active
    if orchestrator.workflow_active:
        print('\n🧪 Testing !pause_workflow command...')
        try:
            await orchestrator.pause_workflow()
            print('✅ !pause_workflow command executed successfully')
        except Exception as e:
            print(f'❌ !pause_workflow failed: {e}')

        print('\n🧪 Testing !resume_workflow command...')
        try:
            await orchestrator.resume_workflow()
            print('✅ !resume_workflow command executed successfully')
        except Exception as e:
            print(f'❌ !resume_workflow failed: {e}')

    print('\n🎉 Workflow command tests completed!')

if __name__ == "__main__":
    asyncio.run(test_workflow_commands())