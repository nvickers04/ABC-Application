#!/usr/bin/env python3
"""Test memory system initialization"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_memory_system():
    """Test memory system initialization"""
    try:
        print("Testing memory system initialization...")

        # Test advanced memory manager
        from src.utils.advanced_memory import get_advanced_memory_manager
        memory_manager = get_advanced_memory_manager()
        print("✅ Advanced memory manager initialized")

        # Test memory health
        health = memory_manager.get_memory_health_status()
        print(f"✅ Memory health: {health.get('overall_healthy', 'unknown')}")
        print(f"   Backends: {list(health.get('backends', {}).keys())}")

        # Test shared memory coordinator
        from src.utils.shared_memory import get_multi_agent_coordinator
        coordinator = get_multi_agent_coordinator()
        print("✅ Multi-agent coordinator initialized")

        # Test basic memory operations
        test_key = "test_key"
        test_data = {"message": "test data", "timestamp": "2024-01-01"}

        # Store
        success = await memory_manager.store_memory(test_key, test_data)
        print(f"✅ Memory store: {success}")

        # Retrieve
        retrieved = await memory_manager.retrieve_memory(test_key)
        print(f"✅ Memory retrieve: {retrieved is not None}")

        # Test collaborative session
        session_id = await coordinator.create_collaborative_session("test_agent", "test topic")
        print(f"✅ Collaborative session created: {session_id}")

        print("\n🎉 All memory system tests passed!")
        return True

    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_memory_system())