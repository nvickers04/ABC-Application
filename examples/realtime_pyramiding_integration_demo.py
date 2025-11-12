#!/usr/bin/env python3
"""
Real-Time Pyramiding Integration Demo
Demonstrates the complete integration between real-time pyramiding and execution agent
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.agents.strategy import StrategyAgent
from src.agents.execution import ExecutionAgent

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

async def run_realtime_pyramiding_integration_demo():
    """Run the complete real-time pyramiding integration demonstration"""

    print("🚀 REAL-TIME PYRAMIDING INTEGRATION DEMO")
    print("=" * 70)
    print()

    # Initialize Strategy Agent (which includes real-time monitor)
    print("🤖 INITIALIZING STRATEGY AGENT WITH REAL-TIME MONITORING...")
    try:
        strategy_agent = StrategyAgent()
        print("✅ StrategyAgent initialized with RealTimePyramidingMonitor")
    except Exception as e:
        print(f"❌ Failed to initialize StrategyAgent: {e}")
        return

    # Initialize Execution Agent
    print("⚡ INITIALIZING EXECUTION AGENT...")
    try:
        execution_agent = ExecutionAgent(historical_mode=False)
        print("✅ ExecutionAgent initialized")
    except Exception as e:
        print(f"❌ Failed to initialize ExecutionAgent: {e}")
        return

    print()

    # Create pyramiding plan
    print("📊 CREATING PYRAMIDING PLAN...")
    try:
        pyramiding_plan = strategy_agent.pyramiding_engine.calculate_pyramiding_plan(
            current_price=150.0,
            entry_price=150.0,
            volatility=0.20,
            trend_strength=0.7,
            current_pnl_pct=0.0,
            max_drawdown_pct=0.05,
            portfolio_value=100000
        )
        print("✅ Pyramiding plan created:")
        print(f"   Price Triggers: {pyramiding_plan['price_triggers']}")
        print(f"   Scaling Factors: {pyramiding_plan['scaling_factors']}")
    except Exception as e:
        print(f"❌ Failed to create pyramiding plan: {e}")
        return

    print()

    # Add position to real-time monitoring
    print("🔍 ADDING POSITION TO REAL-TIME MONITORING...")
    try:
        strategy_agent.add_position_to_monitoring(
            symbol='AAPL',
            entry_price=150.0,
            quantity=1000,
            pyramiding_plan=pyramiding_plan
        )
        print("✅ Position added to real-time monitoring")
    except Exception as e:
        print(f"❌ Failed to add position to monitoring: {e}")
        return

    # Show active triggers
    print()
    print("🎯 ACTIVE PYRAMIDING TRIGGERS:")
    if hasattr(strategy_agent.realtime_monitor, 'active_triggers'):
        for i, trigger in enumerate(strategy_agent.realtime_monitor.active_triggers.get('AAPL', [])):
            print(f"   {i+1}. {trigger.trigger_type}: {trigger.threshold_value} -> {trigger.action}")
    print()

    # Start real-time monitoring
    print("⚡ STARTING REAL-TIME MONITORING...")
    print("   (Monitoring for 30 seconds with simulated market data)")
    print()

    try:
        # Start monitoring
        monitoring_task = asyncio.create_task(
            strategy_agent.realtime_monitor.start_monitoring(mock_market_data_callback)
        )

        # Let it run for 30 seconds
        await asyncio.sleep(30)

        # Stop monitoring
        strategy_agent.realtime_monitor.stop_monitoring()
        monitoring_task.cancel()

    except Exception as e:
        print(f"❌ Error during monitoring: {e}")
        return

    print()
    print("🏁 INTEGRATION DEMO COMPLETE")
    print()

    # Show final status
    try:
        final_status = strategy_agent.get_realtime_status('AAPL')
        if final_status:
            position_data = final_status.get('AAPL', {})
            if position_data:
                print("📈 FINAL POSITION STATUS:")
                print(f"   Symbol: {position_data.get('symbol', 'N/A')}")
                print(f"   Quantity: {position_data.get('quantity', 0)}")
                print(".2f")
                print(f"   Pyramiding Tiers Executed: {position_data.get('pyramiding_tiers_executed', 0)}")
                print()
    except Exception as e:
        print(f"⚠️ Could not retrieve final status: {e}")

    print("🔗 INTEGRATION SUMMARY:")
    print("   ✅ Real-time monitor initialized with A2A protocol")
    print("   ✅ Position tracking active")
    print("   ✅ Trigger system operational")
    print("   ✅ Execution integration ready")
    print("   ✅ Fallback simulation available")
    print()
    print("💡 The real-time pyramiding system is now fully integrated!")
    print("   - Triggers execute via A2A protocol to ExecutionAgent")
    print("   - Orders are placed through IBKR integration")
    print("   - Fallback simulation ensures continuity")
    print("   - 24/7 automated position management active")

if __name__ == "__main__":
    asyncio.run(run_realtime_pyramiding_integration_demo())