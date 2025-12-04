#!/usr/bin/env python3
"""
Test script for system health check
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the health check method
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator

async def test_health_check():
    orchestrator = LiveWorkflowOrchestrator()
    issues, warnings = await orchestrator.perform_system_health_check()
    print(f"Health check completed. Issues: {len(issues)}, Warnings: {len(warnings)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_health_check())