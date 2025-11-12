import asyncio
import sys
import os
sys.path.append('src')

from agents.data import DataAgent

async def test_integration():
    print("Testing enhanced DataAgent integration...")
    agent = DataAgent()

    # Test with multiple symbols to trigger cross-symbol analysis
    result = await agent.process_input({'symbols': ['SPY', 'AAPL']})

    print("Integration test completed!")
    print(f"Cross-symbol analysis present: {'cross_symbol_analysis' in result}")

    if 'cross_symbol_analysis' in result:
        analysis = result['cross_symbol_analysis']
        print(f"Symbols analyzed: {analysis.get('symbols_analyzed', [])}")
        print(f"Total enhanced analyses: {analysis.get('total_enhanced_analyses', 0)}")
        print(f"Analysis breakdown length: {len(analysis.get('analysis_breakdown', []))}")

    # Check enhanced subagent counts
    if 'enhanced_subagent_counts' in result:
        counts = result['enhanced_subagent_counts']
        print(f"Enhanced subagent counts: {counts}")

    return result

if __name__ == "__main__":
    result = asyncio.run(test_integration())