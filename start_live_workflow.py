#!/usr/bin/env python3
"""
Quick launcher for the Live Workflow Orchestrator
"""

import subprocess
import sys
import os

def main():
    print("🎯 ABC Application - Live Workflow Orchestrator")
    print("=" * 50)
    print("🤖 This will start an interactive workflow orchestrator in Discord")
    print("💡 Features:")
    print("  • Automatic iterative reasoning workflow")
    print("  • Real-time execution in Discord")
    print("  • Human intervention and questioning")
    print("  • Live progress tracking")
    print("")
    print("📋 Discord Commands:")
    print("  !start_workflow  - Begin the process")
    print("  !pause_workflow  - Pause mid-workflow")
    print("  !workflow_status - Check progress")
    print("  !stop_workflow   - End workflow")
    print("  💬 Ask questions anytime during execution!")
    print("")

    confirm = input("Start Live Workflow Orchestrator? (y/N): ").strip().lower()

    if confirm == 'y':
        print("\n🚀 Starting Live Workflow Orchestrator...")
        print("📝 Check your Discord server for the orchestrator bot!")
        print("💡 Type '!start_workflow' in Discord to begin")
        print("")

        try:
            # Run the orchestrator
            result = subprocess.run([sys.executable, "live_workflow_orchestrator.py"],
                                  cwd=os.getcwd())

            if result.returncode == 0:
                print("\n✅ Orchestrator completed successfully")
            else:
                print(f"\n❌ Orchestrator exited with code {result.returncode}")

        except KeyboardInterrupt:
            print("\n🛑 Orchestrator stopped by user")
        except Exception as e:
            print(f"\n❌ Error starting orchestrator: {e}")

        print("\n💾 Check 'live_workflow_results.json' for results")
    else:
        print("❌ Orchestrator not started")

if __name__ == "__main__":
    main()