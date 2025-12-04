import asyncio
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator

async def test_enhanced_orchestrator():
    print('Testing enhanced orchestrator with cross-agent communication...')
    orchestrator = LiveWorkflowOrchestrator()
    await orchestrator.initialize_agents_async()

    # Test collaborative session creation
    session_created = await orchestrator.create_collaborative_session('Test Session')
    print(f'Collaborative session created: {session_created}')
    print(f'Session ID: {orchestrator.collaborative_session_id}')

    # Test position data retrieval
    position_data = await orchestrator._get_current_positions()
    positions_count = len(position_data.get('positions', []))
    print(f'Position data retrieved: {positions_count} positions')

    # Test context sharing
    if session_created:
        await orchestrator._share_position_context()
        print('Position context shared with agents')

        # Test full workflow context sharing
        await orchestrator._share_full_workflow_context('test_phase', 'Test Phase')
        print('Full workflow context shared with agents')

    print('Enhanced orchestrator test completed successfully')

if __name__ == "__main__":
    asyncio.run(test_enhanced_orchestrator())