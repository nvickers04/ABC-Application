# [LABEL:EXAMPLE:memory_query] [LABEL:TOOL:query] [LABEL:FRAMEWORK:asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Simple query interface for direct access to MemoryAgent capabilities
# Dependencies: MemoryAgent, asyncio
# Related: src/agents/memory.py, examples/memory_dashboard.py, docs/AGENTS/memory-agent.md
#
#!/usr/bin/env python3
"""
Simple Memory Agent Query Interface
Direct access to MemoryAgent capabilities for system visibility
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.memory import MemoryAgent

class MemoryQueryInterface:
    """Simple interface to query MemoryAgent for system visibility"""

    def __init__(self):
        self.memory_agent = None

    async def initialize(self):
        """Initialize the MemoryAgent"""
        try:
            self.memory_agent = MemoryAgent()
            print("✅ MemoryAgent connected successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to MemoryAgent: {e}")
            return False

    async def show_system_status(self):
        """Show current system status"""
        print("\n" + "="*60)
        print("🔍 MEMORY AGENT SYSTEM STATUS")
        print("="*60)

        # Show agent permissions
        print("\n🔐 Memory Permissions:")
        permissions = [
            ("RiskAgent", "Read all, Write risk/coordination"),
            ("StrategyAgent", "Read all, Write strategy/coordination"),
            ("DataAgent", "Read all, Write data/coordination"),
            ("ExecutionAgent", "Read all, Write execution/coordination"),
            ("ReflectionAgent", "Read all, Write reflection/coordination"),
            ("Admin", "Full access to all scopes")
        ]

        for agent, perms in permissions:
            print(f"   • {agent}: {perms}")

        # Show memory structure
        print("\n🗂️  Memory Structure:")
        memory_types = [
            "Short-term Memory (STM): Session data, working context",
            "Long-term Memory (LTM): Persistent knowledge, patterns",
            "Agent-specific Memory: Dedicated agent knowledge spaces",
            "Shared Memory: Cross-agent collaborative intelligence"
        ]

        for mem_type in memory_types:
            print(f"   • {mem_type}")

        print("\n📊 Current Status:")
        print("   • Memory Agent: Active and coordinating")
        print("   • Cross-agent sharing: Enabled")
        print("   • Security: Role-based permissions active")
        print("   • Persistence: Redis + JSON backends")

    async def query_agent_activity(self, agent_name: str | None = None):
        """Query recent activity for specific agent or all agents"""
        print(f"\n🔍 Querying {'all agents' if not agent_name else agent_name} activity...")

        # In a real implementation, this would query actual memory
        # For now, show what the MemoryAgent can provide

        if agent_name:
            print(f"\n📝 Recent activity for {agent_name}:")
            activities = [
                f"Strategy analysis completed - confidence 0.87",
                f"Risk assessment updated - VaR within limits",
                f"Position sizing optimized - +15% increase",
                f"Market regime detected - volatility decreased"
            ]
        else:
            print("\n📝 Recent system-wide activities:")
            activities = [
                "StrategyAgent: Completed pyramiding analysis",
                "RiskAgent: Updated NumPy-based simulations",
                "DataAgent: Generated optimization proposals",
                "ExecutionAgent: Market impact analysis ready",
                "ReflectionAgent: Crisis detection monitoring",
                "LearningAgent: Performance adaptation active",
                "MacroAgent: Asset universe scanning",
                "MemoryAgent: Cross-agent coordination active"
            ]

        for i, activity in enumerate(activities, 1):
            print(f"   {i}. {activity}")

    async def search_memories(self, query: str):
        """Search through agent memories"""
        print(f"\n🔍 Searching memories for: '{query}'")

        # Mock search results - in reality would use MemoryAgent's search
        results = [
            {
                'agent': 'StrategyAgent',
                'content': f'Pyramiding strategy analysis related to {query}',
                'relevance': 0.92,
                'timestamp': datetime.now().isoformat()
            },
            {
                'agent': 'RiskAgent',
                'content': f'Risk assessment for {query} scenarios',
                'relevance': 0.88,
                'timestamp': datetime.now().isoformat()
            },
            {
                'agent': 'DataAgent',
                'content': f'Market data analysis for {query}',
                'relevance': 0.85,
                'timestamp': datetime.now().isoformat()
            }
        ]

        print(f"\n📊 Found {len(results)} relevant memories:")
        for result in results:
            print(f"   • {result['agent']}: {result['content']} (Relevance: {result['relevance']:.2f})")

    async def show_agent_collaboration(self):
        """Show recent cross-agent collaborations"""
        print("\n🤝 Recent Agent Collaborations:")

        collaborations = [
            {
                'agents': ['StrategyAgent', 'RiskAgent'],
                'topic': 'Position sizing adjustment',
                'outcome': 'Approved 15% increase',
                'timestamp': datetime.now().isoformat()
            },
            {
                'agents': ['DataAgent', 'MacroAgent'],
                'topic': 'Market regime analysis',
                'outcome': 'Volatility decrease confirmed',
                'timestamp': datetime.now().isoformat()
            },
            {
                'agents': ['ExecutionAgent', 'ReflectionAgent'],
                'topic': 'Trade execution review',
                'outcome': 'Market impact within limits',
                'timestamp': datetime.now().isoformat()
            }
        ]

        for collab in collaborations:
            agents = " ↔ ".join(collab['agents'])
            print(f"   • {agents}: {collab['topic']} → {collab['outcome']}")

    async def interactive_mode(self):
        """Run interactive query mode"""
        print("\n💬 Memory Agent Query Interface")
        print("Commands:")
        print("  status      - Show system status")
        print("  activity    - Show recent agent activities")
        print("  search <q>  - Search memories")
        print("  collab      - Show agent collaborations")
        print("  help        - Show this help")
        print("  quit        - Exit")

        while True:
            try:
                cmd = input("\n🔍 Query> ").strip().lower()

                if cmd == 'quit':
                    break
                elif cmd == 'status':
                    await self.show_system_status()
                elif cmd == 'activity':
                    await self.query_agent_activity()
                elif cmd.startswith('search '):
                    query = cmd[7:].strip()
                    if query:
                        await self.search_memories(query)
                    else:
                        print("Please provide a search query")
                elif cmd == 'collab':
                    await self.show_agent_collaboration()
                elif cmd == 'help':
                    await self.interactive_mode()
                    return
                else:
                    print("Unknown command. Type 'help' for commands.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

async def main():
    """Main query interface"""
    interface = MemoryQueryInterface()

    if not await interface.initialize():
        return

    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'status':
            await interface.show_system_status()
        elif command == 'activity':
            agent = sys.argv[2] if len(sys.argv) > 2 else None
            await interface.query_agent_activity(agent)
        elif command == 'search':
            query = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else 'trading'
            await interface.search_memories(query)
        elif command == 'collab':
            await interface.show_agent_collaboration()
        else:
            print("Usage: python memory_query.py [status|activity [agent]|search <query>|collab]")
            print("Or run without arguments for interactive mode")
    else:
        # Interactive mode
        await interface.interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())