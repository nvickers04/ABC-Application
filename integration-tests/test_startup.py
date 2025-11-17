# [LABEL:TEST:startup] [LABEL:TEST:integration] [LABEL:FRAMEWORK:asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Integration test for ABC Application startup and basic functionality
# Dependencies: src.main, asyncio, all core modules
# Related: src/main.py, docs/IMPLEMENTATION/setup.md
#
#!/usr/bin/env python3
"""
Quick test script for ABC Application startup
"""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_startup():
    """Test basic application startup"""
    try:
        print("Testing ABC Application startup...")

        # Test imports
        from src.main import main_loop
        print("✓ Main module imported successfully")

        # Test agent imports
        from src.agents.data import DataAgent
        from src.agents.strategy import StrategyAgent
        from src.agents.risk import RiskAgent
        from src.agents.execution import ExecutionAgent
        print("✓ All agent imports successful")

        # Test A2A protocol
        from src.utils.a2a_protocol import A2AProtocol
        print("✓ A2A protocol import successful")

        print("🎉 All startup tests passed!")
        return True

    except Exception as e:
        print(f"❌ Startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_startup())
    sys.exit(0 if success else 1)