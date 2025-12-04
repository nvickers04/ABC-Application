#!/usr/bin/env python3
"""
Live Trading Setup Script - TWS and Discord Integration
Sets up live trading with Interactive Brokers TWS and Discord orchestration
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load existing environment
load_dotenv()

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_tws_status():
    """Check if TWS is running"""
    print_header("Checking TWS Status")

    try:
        # Run the diagnostic script
        result = subprocess.run([
            sys.executable, "integration-tests/diagnose_api.py"
        ], capture_output=True, text=True, cwd=os.getcwd())

        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error checking TWS: {e}")
        return False

def setup_discord_bot():
    """Guide user through Discord bot setup"""
    print_header("Discord Bot Setup")

    print("To set up Discord integration, you need to:")
    print("1. Create a Discord application at https://discord.com/developers/applications")
    print("2. Create a bot user in the application")
    print("3. Copy the bot token")
    print("4. Invite the bot to your server")
    print()

    # Check if token already exists
    existing_token = os.getenv('DISCORD_ORCHESTRATOR_TOKEN')
    if existing_token:
        print(f"✅ Discord token already configured: {existing_token[:20]}...")
        return True

    print("Please provide your Discord bot token:")
    token = input("Bot Token: ").strip()

    if not token:
        print("❌ No token provided. Discord setup skipped.")
        return False

    # Update .env file
    env_file = Path('.env')
    if not env_file.exists():
        env_file.touch()

    # Read existing content
    content = env_file.read_text() if env_file.exists() else ""

    # Add or update the token
    if 'DISCORD_ORCHESTRATOR_TOKEN=' in content:
        # Replace existing
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('DISCORD_ORCHESTRATOR_TOKEN='):
                lines[i] = f'DISCORD_ORCHESTRATOR_TOKEN={token}'
                break
        content = '\n'.join(lines)
    else:
        # Add new
        if content and not content.endswith('\n'):
            content += '\n'
        content += f'DISCORD_ORCHESTRATOR_TOKEN={token}\n'

    env_file.write_text(content)
    print("✅ Discord token saved to .env file")

    # Ask for guild ID
    print("\nPlease provide your Discord server (guild) ID:")
    print("(Right-click server name → Copy Server ID - requires Developer Mode)")
    guild_id = input("Guild ID: ").strip()

    if guild_id:
        # Update content again
        if 'DISCORD_GUILD_ID=' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('DISCORD_GUILD_ID='):
                    lines[i] = f'DISCORD_GUILD_ID={guild_id}'
                    break
            content = '\n'.join(lines)
        else:
            content += f'DISCORD_GUILD_ID={guild_id}\n'

        env_file.write_text(content)
        print("✅ Guild ID saved to .env file")

    return True

def test_discord_connection():
    """Test Discord bot connection"""
    print_header("Testing Discord Connection")

    try:
        # Try to import and test the orchestrator
        from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator

        print("✅ Discord orchestrator imports successfully")

        # Check if token is available
        token = os.getenv('DISCORD_ORCHESTRATOR_TOKEN')
        if not token:
            print("❌ No Discord token found in environment")
            return False

        print(f"✅ Discord token found: {token[:20]}...")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def test_ibkr_connection():
    """Test IBKR connection"""
    print_header("Testing IBKR Connection")

    try:
        # Import the bridge
        from src.integrations.nautilus_ibkr_bridge import get_nautilus_ibkr_bridge

        print("✅ IBKR bridge imports successfully")

        # Try to get bridge instance
        bridge = get_nautilus_ibkr_bridge()
        print("✅ IBKR bridge initialized")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def test_execution_agent():
    """Test execution agent integration"""
    print_header("Testing Execution Agent")

    try:
        from src.agents.execution import ExecutionAgent

        print("✅ Execution agent imports successfully")

        # Check if it has IBKR integration
        agent = ExecutionAgent()
        if hasattr(agent, 'ibkr_connector'):
            print("✅ Execution agent has IBKR connector")
        else:
            print("⚠️ Execution agent missing IBKR connector")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def start_live_workflow():
    """Start the live workflow orchestrator"""
    print_header("Starting Live Workflow Orchestrator")

    print("🚀 Starting Discord-based live trading workflow...")
    print()
    print("Commands available in Discord:")
    print("• !start_workflow - Begin the iterative reasoning process")
    print("• !pause_workflow - Pause current workflow")
    print("• !resume_workflow - Resume paused workflow")
    print("• !stop_workflow - Stop workflow")
    print("• !workflow_status - Check status")
    print("• !status - Agent health check")
    print()
    print("During active workflow, you can ask questions and intervene!")
    print()

    try:
        # Import and run the orchestrator
        from src.agents.live_workflow_orchestrator import main
        import asyncio

        print("Starting orchestrator...")
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n🛑 Orchestrator stopped by user")
    except Exception as e:
        print(f"❌ Failed to start orchestrator: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main setup function"""
    print("🎯 ABC Application - Live Trading Setup")
    print("Setting up TWS (IBKR) and Discord integration for live trading")
    print()

    # Check current status
    tws_ok = check_tws_status()

    if not tws_ok:
        print("\n❌ TWS is not running. Please start Trader Workstation:")
        print("1. Launch IBKR Trader Workstation")
        print("2. Log in with your paper trading account")
        print("3. Go to File → Global Configuration → API")
        print("4. Enable 'Enable ActiveX and Socket Clients'")
        print("5. Set Socket port to 7497")
        print("6. Save and restart TWS")
        print()
        print("Then re-run this script.")
        return

    # Setup Discord
    discord_ok = setup_discord_bot()

    if not discord_ok:
        print("\n⚠️ Discord setup incomplete. You can still test IBKR integration.")
        proceed = input("Continue with IBKR testing? (y/n): ").lower().strip()
        if proceed != 'y':
            return

    # Test components
    print_header("Component Testing")

    discord_test = test_discord_connection()
    ibkr_test = test_ibkr_connection()
    agent_test = test_execution_agent()

    print("\n📊 Test Results:")
    print(f"• TWS Status: {'✅' if tws_ok else '❌'}")
    print(f"• Discord Config: {'✅' if discord_ok else '❌'}")
    print(f"• Discord Test: {'✅' if discord_test else '❌'}")
    print(f"• IBKR Test: {'✅' if ibkr_test else '❌'}")
    print(f"• Agent Test: {'✅' if agent_test else '❌'}")

    all_good = tws_ok and discord_ok and discord_test and ibkr_test and agent_test

    if all_good:
        print("\n🎉 All systems ready for live trading!")
        start_now = input("Start live workflow orchestrator now? (y/n): ").lower().strip()
        if start_now == 'y':
            start_live_workflow()
        else:
            print("\nTo start later, run:")
            print("python tools/start_unified_workflow.py --mode hybrid --symbols SPY")
    else:
        print("\n⚠️ Some components need attention. Please fix issues and re-run.")

if __name__ == "__main__":
    main()