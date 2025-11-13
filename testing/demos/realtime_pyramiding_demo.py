#!/usr/bin/env python3
"""
Real-Time Pyramiding Demo
Demonstrates the live real-time pyramiding system in action
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.realtime_pyramiding import RealTimePyramidingMonitor
from src.utils.pyramiding import PyramidingEngine

async def mock_market_data_callback():
    """Mock market data callback that simulates price movements"""
    # Simulate AAPL price starting at $150 and moving up
    base_price = 150.0
    volatility = 0.02  # 2% volatility

    while True:
        # Simulate price movement
        price_change = random.uniform(-volatility, volatility)
        current_price = base_price * (1 + price_change)

        market_data = {
            'AAPL': {
                'price': current_price,
                'timestamp': datetime.now()
            }
        }

        await asyncio.sleep(1)  # Update every second
        return market_data

async def run_realtime_pyramiding_demo():
    """Run the real-time pyramiding demonstration"""

    print("🚀 REAL-TIME PYRAMIDING SYSTEM DEMO")
    print("=" * 60)
    print()

    # Initialize pyramiding engine
    pyramiding_engine = PyramidingEngine()

    # Create pyramiding plan for AAPL
    pyramiding_plan = pyramiding_engine.calculate_pyramiding_plan(
        current_price=150.0,
        entry_price=150.0,
        volatility=0.20,  # 20% volatility
        trend_strength=0.7,  # Strong uptrend
        current_pnl_pct=0.0,  # At entry
        max_drawdown_pct=0.05,  # 5% max drawdown
        portfolio_value=100000  # $100k portfolio
    )

    print("📊 PYRAMIDING PLAN:")
    print(f"   Symbol: AAPL")
    print(f"   Entry Price: $150.00")
    print(f"   Initial Position: 1000 shares")
    print(f"   Price Triggers: {pyramiding_plan['price_triggers']}")
    print(f"   Scaling Factors: {pyramiding_plan['scaling_factors']}")
    print()

    # Initialize real-time monitor
    monitor = RealTimePyramidingMonitor(pyramiding_engine)

    # Add position to monitoring
    monitor.add_position(
        symbol='AAPL',
        entry_price=150.0,
        quantity=1000,
        pyramiding_plan=pyramiding_plan
    )

    print("🔍 ACTIVE TRIGGERS:")
    for i, trigger in enumerate(monitor.active_triggers.get('AAPL', [])):
        print(f"   {i+1}. {trigger.trigger_type}: {trigger.threshold_value} -> {trigger.action}")
    print()

    print("⚡ STARTING REAL-TIME MONITORING...")
    print("   (Monitoring for price movements and trigger execution)")
    print()

    # Start monitoring for 30 seconds
    monitoring_task = asyncio.create_task(monitor.start_monitoring(mock_market_data_callback))

    # Let it run for 30 seconds
    await asyncio.sleep(30)

    # Stop monitoring
    monitor.stop_monitoring()
    monitoring_task.cancel()

    print()
    print("🏁 DEMO COMPLETE")
    print("   Real-time pyramiding system successfully demonstrated!")
    print()

    # Show final position status
    position = monitor.positions.get('AAPL')
    if position:
        print("📈 FINAL POSITION STATUS:")
        print(f"   Symbol: {position.symbol}")
        print(f"   Entry Price: ${position.entry_price:.2f}")
        print(f"   Current Price: ${position.current_price:.2f}")
        print(f"   Quantity: {position.quantity}")
        print(".2f")
        print(f"   Pyramiding Tiers Executed: {position.pyramiding_tiers_executed}")
        print()

if __name__ == "__main__":
    asyncio.run(run_realtime_pyramiding_demo())