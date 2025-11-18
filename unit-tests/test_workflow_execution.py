#!/usr/bin/env python3
"""
Test script for complete workflow execution
"""
import asyncio
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator

async def test_workflow():
    print('🎯 Testing complete workflow execution...')
    orchestrator = LiveWorkflowOrchestrator()

    # Initialize agents
    await orchestrator.initialize_agents_async()
    print(f'✅ Initialized {len(orchestrator.agent_instances)} agents')

    # Check health
    health = await orchestrator.check_agent_health()
    print(f'🏥 Health check: {health["overall_health"]} ({len(health["healthy_agents"])}/{health["total_agents"]} healthy)')

    # Create collaborative session
    session_created = await orchestrator.create_collaborative_session('Test Workflow')
    print(f'🤝 Session created: {session_created}')

    # Test a single phase execution (macro foundation data collection)
    if session_created:
        print('📊 Testing phase execution...')
        await orchestrator.execute_phase_with_agents('macro_foundation_data_collection', 'TEST: Data Collection')
        print('✅ Phase execution completed')

    print('🎉 Workflow test completed successfully!')

if __name__ == "__main__":
    asyncio.run(test_workflow())