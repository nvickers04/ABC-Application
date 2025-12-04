#!/usr/bin/env python3
"""
Test script for concurrent pipeline processing.
"""
import asyncio
import time
from src.agents.data import DataAgent

async def test_concurrent_pipeline():
    print('Testing concurrent pipeline processing...')
    start_time = time.time()

    agent = DataAgent()

    # Test with multiple symbols to trigger concurrent processing
    result = await agent.process_input({'symbols': ['SPY', 'AAPL']})

    end_time = time.time()
    duration = end_time - start_time

    print('.2f')
    print(f'Pipeline used: {hasattr(agent, "pipeline_processor")}')
    print(f'Symbols processed: {result.get("symbols_processed", [])}')
    print(f'Results count: {len(result.get("symbol_data", {}))}')

    # Check if concurrent processing worked
    if duration < 45:  # Should be faster than sequential
        print('✅ SUCCESS: Concurrent pipeline working!')
    else:
        print('⚠️  SLOWER: May still be using sequential fallback')

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
    duration = asyncio.run(test_concurrent_pipeline())