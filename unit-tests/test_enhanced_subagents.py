#!/usr/bin/env python3
"""
Test script for enhanced strategy subagents with LLM integration and collaborative memory.
Tests the OptionsStrategySub, FlowStrategySub, and MLStrategySub agents.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_basic_functionality():
    """Test basic functionality without full LLM/memory integration."""
    print("🧪 Testing Enhanced Strategy Subagents - Basic Functionality")
    print("=" * 60)

    try:
        # Test imports
        from src.agents.strategy_subs.options_strategy_sub import OptionsStrategySub
        print("✅ OptionsStrategySub imported successfully")

        from src.agents.strategy_subs.flow_strategy_sub import FlowStrategySub
        print("✅ FlowStrategySub imported successfully")

        from src.agents.strategy_subs.ml_strategy_sub import MLStrategySub
        print("✅ MLStrategySub imported successfully")

        # Test basic instantiation (without LLM calls)
        print("\nTesting basic instantiation...")

        # Create agents (this will fail if there are import issues)
        options_agent = OptionsStrategySub()
        print("✅ OptionsStrategySub instantiated")

        flow_agent = FlowStrategySub()
        print("✅ FlowStrategySub instantiated")

        ml_agent = MLStrategySub()
        print("✅ MLStrategySub instantiated")

        # Test basic attributes
        print(f"\nOptions agent role: {options_agent.role}")
        print(f"Flow agent role: {flow_agent.role}")
        print(f"ML agent role: {ml_agent.role}")

        # Test collaborative memory initialization
        print(f"\nOptions subagent memory initialized: {hasattr(options_agent, 'subagent_memory')}")
        print(f"Flow subagent memory initialized: {hasattr(flow_agent, 'subagent_memory')}")
        print(f"ML subagent memory initialized: {hasattr(ml_agent, 'subagent_memory')}")

        print("\n" + "=" * 60)
        print("✅ Basic functionality test passed!")
        print("\nNote: Full LLM integration requires GROK_API_KEY environment variable")
        print("and complete memory system setup for full functionality testing.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run basic functionality test."""
    await test_basic_functionality()

if __name__ == "__main__":
    asyncio.run(main())