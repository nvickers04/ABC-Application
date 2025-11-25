import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.agents.strategy_analyzers.multi_instrument_strategy_analyzer import MultiInstrumentStrategyAnalyzer
import pandas as pd
import numpy as np
import asyncio

async def test_multi_instrument_strategy():
    # Create test data similar to what the data agent produces
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'Close_SPY': np.random.randn(100).cumsum() + 400,
        'Close_AAPL': np.random.randn(100).cumsum() + 150
    }
    df = pd.DataFrame(data, index=dates)

    # Create test input data
    input_data = {
        'dataframe': df,
        'symbols': ['SPY', 'AAPL'],
        'sentiment': {'score': 0.6},
        'economic': {'indicators': {'GDP': 0.5, 'inflation': 0.02}},
        'institutional': {'holdings': [{'symbol': 'SPY', 'shares': 1000000}]}
    }

    # Test the multi-instrument strategy subagent
    agent = MultiInstrumentStrategyAnalyzer()
    result = await agent.process_input(input_data)

    print('Multi-instrument strategy result:')
    if 'multi_instrument' in result and result['multi_instrument']:
        strategy = result['multi_instrument']
        print(f'  Strategy type: {strategy.get("strategy_type", "none")}')
        print(f'  Instruments: {strategy.get("instruments", [])}')
        print(f'  ROI estimate: {strategy.get("roi_estimate", 0):.1%}')
        print(f'  Description: {strategy.get("description", "")[:100]}...')
    else:
        print('  No strategy generated')

    return result

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_multi_instrument_strategy())