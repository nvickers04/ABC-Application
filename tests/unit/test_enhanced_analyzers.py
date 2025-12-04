#!/usr/bin/env python3
"""
Test script for enhanced strategy analyzers with LLM integration and collaborative memory.
Tests the OptionsStrategyAnalyzer, FlowStrategyAnalyzer, and MLStrategyAnalyzer agents.
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
        from src.agents.strategy_analyzers.options_strategy_analyzer import OptionsStrategyAnalyzer
        print("✅ OptionsStrategyAnalyzer imported successfully")

        from src.agents.strategy_analyzers.flow_strategy_analyzer import FlowStrategyAnalyzer
        print("✅ FlowStrategyAnalyzer imported successfully")

        from src.agents.strategy_analyzers.ai_strategy_analyzer import AIStrategyAnalyzer
        print("✅ AIStrategyAnalyzer imported successfully")

        # Test basic instantiation (without LLM calls)
        print("\nTesting basic instantiation...")

        # Create agents (this will fail if there are import issues)
        options_agent = OptionsStrategyAnalyzer()
        print("✅ OptionsStrategyAnalyzer instantiated")

        flow_agent = FlowStrategyAnalyzer()
        print("✅ FlowStrategyAnalyzer instantiated")

        ai_agent = AIStrategyAnalyzer()
        print("✅ AIStrategyAnalyzer instantiated")

        # Test basic attributes
        print(f"\nOptions agent role: {options_agent.role}")
        print(f"Flow agent role: {flow_agent.role}")
        print(f"AI agent role: {ai_agent.role}")

        # Test collaborative memory initialization
        print(f"\nOptions analyzer memory initialized: {hasattr(options_agent, 'analyzer_memory')}")
        print(f"Flow analyzer memory initialized: {hasattr(flow_agent, 'analyzer_memory')}")
        print(f"AI analyzer memory initialized: {hasattr(ai_agent, 'analyzer_memory')}")

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