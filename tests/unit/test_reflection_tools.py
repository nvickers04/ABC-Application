# [LABEL:TEST:reflection_agent] [LABEL:TEST:unit] [LABEL:FRAMEWORK:python]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Unit tests for reflection agent tools and functionality
# Dependencies: ReflectionAgent, src.agents.reflection
# Related: src/agents/reflection.py, docs/AGENTS/main-agents/reflection-agent.md
#
#!/usr/bin/env python3
"""
Test script for reflection agent tools
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.reflection import ReflectionAgent

def main():
    print("Testing Reflection Agent Tools...")

    # Initialize agent with default constructor
    agent = ReflectionAgent()

    # Check tools
    print(f'Agent initialized with {len(agent.tools)} tools')
    print('Tools available:')
    for i, tool in enumerate(agent.tools):
        tool_name = tool.__name__ if hasattr(tool, '__name__') else str(tool)
        print(f'  {i+1}. {tool_name}')

    # Test sanity_check_tool
    print('\nTesting sanity_check_tool...')
    test_proposal = {'symbol': 'AAPL', 'quantity': 100, 'direction': 'buy', 'price': 150.0}
    # Convert proposal dict to string format expected by tool
    proposal_str = f"{test_proposal.get('direction', 'trade').title()} {test_proposal.get('quantity', 0)} shares of {test_proposal.get('symbol', 'UNKNOWN')} at ${test_proposal.get('price', 0):.2f}"
    result = agent.tools[3].invoke(proposal_str)  # sanity_check_tool is at index 3
    print(f'Sanity check result keys: {list(result.keys())}')
    print(f'Overall sanity: {result.get("overall_sanity", "unknown")}')

    # Test convergence_check_tool
    print('\nTesting convergence_check_tool...')
    test_data = {'metrics': {}, 'learning_history': []}  # The actual performance data dict
    result = agent.tools[4].invoke({"performance_data": test_data})  # convergence_check_tool is at index 4
    print(f'Convergence check result keys: {list(result.keys())}')
    print(f'Overall convergence: {result.get("overall_convergence", "unknown")}')

    print("\n✅ Tool testing completed successfully!")

if __name__ == "__main__":
    main()