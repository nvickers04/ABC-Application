#!/usr/bin/env python3
"""
Performance test for optimized DataAgent with Phase 2 optimizations.
"""
import asyncio
import time
from src.agents.data import DataAgent

async def test_performance():
    print('Testing optimized DataAgent performance...')
    start_time = time.time()

    agent = DataAgent()
    result = await agent.process_input({'symbols': ['SPY']})

    end_time = time.time()
    duration = end_time - start_time

    print('.2f')
    print(f'Memory manager active: {hasattr(agent, "memory_manager")}')
    print(f'Pipeline processor active: {hasattr(agent, "pipeline_processor")}')

    # Check if we achieved sub-30 second target
    if duration < 30:
        print('✅ SUCCESS: Achieved sub-30 second processing target!')
    elif duration < 60:
        print('⚠️  PARTIAL: Under 60 seconds but above 30 second target')
    else:
        print('❌ ISSUE: Still above 60 seconds')

    # Show memory stats if available
    if hasattr(agent, 'memory_manager'):
        try:
            stats = agent.memory_manager.get_comprehensive_stats()
            print(f'Memory stats: Peak usage {stats["memory_stats"]["peak_usage_mb"]:.1f}MB, Pool hits {stats["memory_stats"]["pool_hits"]}')
        except Exception as e:
            print(f'Could not get memory stats: {e}')

    return duration

if __name__ == "__main__":
    # Run the test
    duration = asyncio.run(test_performance())