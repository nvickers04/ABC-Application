#!/usr/bin/env python3
"""
Test Paper Trade Execution
Execute a small test trade through the ABC Application system
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.execution import ExecutionAgent

async def test_paper_trade():
    print('🧪 Testing Paper Trade Execution...')
    print('=' * 40)

    # Create execution agent
    agent = ExecutionAgent(historical_mode=False)

    # Test trade parameters (very small position)
    test_trade = {
        'symbol': 'SPY',
        'quantity': 1,  # Just 1 share for testing
        'roi_estimate': 0.02,  # 2% expected return
        'simulated_pop': 0.75,  # Good probability
        'alpha_estimate': 0.15  # Good alpha
    }

    print('📊 Test Trade: BUY 1 share of SPY')
    roi_pct = test_trade['roi_estimate'] * 100
    print(f'🎯 Expected ROI: {roi_pct:.1f}%')
    print('')

    try:
        # Execute the trade
        result = await agent.process_input(test_trade)

        print('📋 Execution Result:')
        executed = result.get('executed', False)
        status = "SUCCESS" if executed else "FAILED"
        print(f'   Status: {status}')
        print(f'   Symbol: {result.get("symbol", "N/A")}')
        print(f'   Quantity: {result.get("quantity", 0)}')
        price = result.get('price', 0)
        total_value = result.get('total_value', 0)
        print(f'   Price: ${price:.2f}')
        print(f'   Total Value: ${total_value:.2f}')
        print(f'   Simulated: {result.get("simulated", True)}')
        print(f'   Source: {result.get("source", "unknown")}')

        if executed:
            print('')
            print('🎉 PAPER TRADE SUCCESSFUL!')
            print('The ABC Application system is fully operational!')
        else:
            print('')
            reason = result.get('reason', 'Unknown error')
            print(f'❌ Trade failed: {reason}')

    except Exception as e:
        print(f'❌ Error during trade execution: {e}')
        import traceback
        traceback.print_exc()

    print('')
    print('🏁 Test completed')

if __name__ == "__main__":
    asyncio.run(test_paper_trade())