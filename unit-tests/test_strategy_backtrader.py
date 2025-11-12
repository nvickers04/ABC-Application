import sys
sys.path.insert(0, '.')
from src.agents.strategy import StrategyAgent
import asyncio
import pandas as pd

# Test the strategy agent with backtrader integration
agent = StrategyAgent()
sample_input = {
    'dataframe': pd.DataFrame({'Close': [100, 105, 102, 108, 106, 110, 115, 112, 118, 120]}),
    'sentiment': {'sentiment': 'bullish', 'confidence': 0.8},
    'symbols': ['SPY']
}

try:
    result = asyncio.run(agent.process_input(sample_input))
    print('SUCCESS: Strategy Agent Test Result')
    print(f'Strategy Type: {result.get("strategy_type", "unknown")}')
    print(f'Validation Confidence: {result.get("validation_confidence", "none")}')
    print(f'Backtrader Validation: {result.get("backtrader_validation", {}).get("backtest_validated", "not_run")}')
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()