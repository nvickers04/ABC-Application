import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.data import DataAgent
from src.agents.strategy_subs.multi_instrument_strategy_sub import MultiInstrumentStrategySub
import asyncio

async def comprehensive_test():
    print('=== COMPREHENSIVE MULTI-SYMBOL SYSTEM TEST ===')

    # Step 1: Test Data Agent with multiple symbols
    print('\n1. Testing Data Agent with multiple symbols...')
    data_agent = DataAgent()
    data_input = {'symbols': ['SPY', 'AAPL'], 'period': '6mo'}
    data_result = await data_agent.process_input(data_input)

    if 'dataframe' in data_result and not data_result['dataframe'].empty:
        df = data_result['dataframe']
        print(f'   ✅ Data Agent: Created dataframe with shape {df.shape}')
        print(f'   ✅ Columns: {df.columns.tolist()}')
    else:
        print('   ❌ Data Agent: Failed to create dataframe')
        return

    # Step 2: Test Multi-Instrument Strategy Subagent
    print('\n2. Testing Multi-Instrument Strategy Subagent...')
    strategy_input = {
        'dataframe': data_result['dataframe'],
        'symbols': data_result.get('symbols', ['SPY', 'AAPL']),
        'sentiment': {'score': 0.6},
        'economic': {'indicators': {'GDP': 0.5}},
        'institutional': {'holdings': [{'symbol': 'SPY', 'shares': 1000000}]}
    }

    strategy_agent = MultiInstrumentStrategySub()
    strategy_result = await strategy_agent.process_input(strategy_input)

    if 'multi_instrument' in strategy_result and strategy_result['multi_instrument']:
        strategy = strategy_result['multi_instrument']
        print(f'   ✅ Strategy Agent: Generated {strategy.get("strategy_type", "unknown")} strategy')
        print(f'   ✅ Instruments: {strategy.get("instruments", [])}')
        print(f'   ✅ Expected ROI: {strategy.get("roi_estimate", 0):.1%}')
    else:
        print('   ❌ Strategy Agent: Failed to generate strategy')
        return

    print('\n=== ALL TESTS PASSED! ===')
    print('Multi-symbol processing system is fully operational.')

if __name__ == "__main__":
    # Run the comprehensive test
    result = asyncio.run(comprehensive_test())