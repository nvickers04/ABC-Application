# [LABEL:TOOL:test_24_6] [LABEL:FRAMEWORK:discord] [LABEL:INFRA:vps]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-19] [LABEL:REVIEWED:pending]
#
# Purpose: Test script for unified workflow orchestrator continuous operation setup
# Dependencies: Discord integration, environment configuration
# Related: src/agents/unified_workflow_orchestrator.py, docs/IMPLEMENTATION/24_6_CONTINUOUS_OPERATION.md
#
#!/usr/bin/env python3
"""
Test script for 24/6 Workflow Orchestrator
Verifies configuration and basic functionality before full deployment.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.vault_client import get_vault_secret
from src.agents.unified_workflow_orchestrator import UnifiedWorkflowOrchestrator, WorkflowMode

async def test_discord_configuration():
    """Test Discord bot configuration"""
    print("🔍 Testing Discord Configuration...")

    # Check required environment variables
    token = get_vault_secret('DISCORD_ORCHESTRATOR_TOKEN')
    guild_id = get_vault_secret('DISCORD_GUILD_ID')

    if not token:
        print("❌ DISCORD_ORCHESTRATOR_TOKEN not found")
        return False

    if not guild_id:
        print("❌ DISCORD_GUILD_ID not found")
        return False

    print("✅ Discord credentials configured")
    print(f"   Guild ID: {guild_id}")
    print(f"   Token: {'*' * 10}...{'*' * 10}")  # Partial mask

    return True

async def test_agent_initialization():
    """Test agent initialization"""
    print("\n🔍 Testing Agent Initialization...")

    orchestrator = UnifiedWorkflowOrchestrator(mode=WorkflowMode.HYBRID, enable_discord=False)

    try:
        await orchestrator.initialize()

        agent_count = len(orchestrator.agent_instances)
        print(f"✅ Initialized {agent_count} agents")

        if agent_count == 0:
            print("⚠️ No agents initialized - Discord-only mode")
        else:
            # Test agent health
            health = await orchestrator.check_agent_health()
            print(f"   Health Status: {health['overall_health']}")
            print(f"   Healthy Agents: {len(health['healthy_agents'])}/{health['total_agents']}")

        return True

    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

async def test_market_calendar():
    """Test market calendar functionality"""
    print("\n🔍 Testing Market Calendar...")

    try:
        import exchange_calendars as ecals
        calendar = ecals.get_calendar('NYSE')

        from datetime import datetime
        today = datetime.now().date()
        is_trading_day = calendar.is_session(today)

        print(f"✅ Market calendar loaded")
        print(f"   Today is trading day: {is_trading_day}")

        # Check next 3 trading days
        next_sessions = calendar.sessions_in_range(today, today.replace(day=today.day + 7))[:3]
        print(f"   Next trading days: {next_sessions.strftime('%Y-%m-%d').tolist()}")

        return True

    except ImportError:
        print("❌ exchange_calendars not installed")
        return False
    except Exception as e:
        print(f"❌ Market calendar test failed: {e}")
        return False

async def test_scheduled_workflows():
    """Test scheduled workflow configuration"""
    print("\n🔍 Testing Scheduled Workflows...")

    try:
        import schedule

        # Test unified orchestrator scheduling
        orchestrator = UnifiedWorkflowOrchestrator(mode=WorkflowMode.HYBRID, enable_discord=False)
        await orchestrator.initialize()

        # Check if scheduler jobs were created
        jobs = orchestrator.scheduler.get_jobs()
        print(f"✅ Created {len(jobs)} scheduled jobs")

        for job in jobs[:3]:  # Show first 3
            print(f"   {job.name}: {job.trigger}")

        if len(jobs) < 5:
            print("⚠️ Fewer scheduled jobs than expected")

        return True

    except ImportError:
        print("❌ schedule library not installed")
        return False
    except Exception as e:
        print(f"❌ Scheduled workflow test failed: {e}")
        return False

async def run_tests():
    """Run all tests"""
    print("🧪 ABC Application - 24/6 Orchestrator Test Suite")
    print("=" * 55)

    tests = [
        ("Discord Configuration", test_discord_configuration),
        ("Agent Initialization", test_agent_initialization),
        ("Market Calendar", test_market_calendar),
        ("Scheduled Workflows", test_scheduled_workflows),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 55)
    print("📊 Test Results Summary:")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready for 24/6 deployment.")
        print("\nNext steps:")
        print("1. Run: sudo systemctl start abc-24-6-orchestrator")
        print("2. Check Discord for startup messages")
        print("3. Wait for scheduled workflows to trigger")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before deployment.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install discord.py schedule exchange-calendars")
        print("- Configure Discord credentials in vault")
        print("- Check agent imports and dependencies")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)