import asyncio
import sys
import os
sys.path.append('src')

from agents.strategy import StrategyAgent

async def test_portfolio_rebalancing():
    agent = StrategyAgent()
    result = await agent.analyze_portfolio_rebalancing()

    print('Portfolio Rebalancing Analysis Results:')
    print(f"Recommendation Type: {result.get('recommendation_type', 'N/A')}")
    print(f"Implementation Priority: {result.get('implementation_priority', 'N/A')}")
    print()

    print('Current Portfolio Allocation:')
    current = result.get('current_portfolio', {})
    for sector, allocation in current.get('sector_allocation', {}).items():
        print(f"  {sector}: {allocation:.1%}")
    print()

    print('Tactical Rebalancing Recommendations:')
    tactical = result.get('rebalancing_recommendations', {}).get('tactical_rebalancing', {})
    for sector, rec in tactical.items():
        action = rec.get('action', 'N/A')
        target = rec.get('target_allocation', 0)
        reason = rec.get('reason', 'N/A')
        print(f"  {sector}: {action} to {target:.1%} - {reason}")
    print()

    print('Expected Impact:')
    impact = result.get('impact_analysis', {})
    print(f"  Beta Improvement: {impact.get('beta_improvement', 0):.2f}")
    print(f"  Sharpe Ratio Improvement: {impact.get('expected_sharpe_improvement', 0):.2f}")
    print(f"  Confidence Level: {impact.get('confidence_level', 0):.1%}")

if __name__ == "__main__":
    asyncio.run(test_portfolio_rebalancing())