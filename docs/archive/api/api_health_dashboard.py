#!/usr/bin/env python3
"""
API Health Dashboard for ABC Application

Displays real-time API health status and monitoring information.
Run with: python api_health_dashboard.py
"""

import json
import time
from datetime import datetime
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.api_health_monitor import get_api_health_summary, check_api_health_now

def print_health_dashboard():
    """Print a formatted health dashboard"""
    print("\n" + "="*80)
    print("🔍 ABC Application API HEALTH DASHBOARD")
    print("="*80)
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        health_data = get_api_health_summary()

        # Overall status
        overall_status = health_data.get("overall_status", "unknown")
        summary = health_data.get("summary", {})

        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌",
            "unknown": "❓"
        }

        print(f"\n🌐 Overall Status: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
        print(f"📊 APIs Monitored: {summary.get('total_apis', 0)}")
        print(f"✅ Healthy: {summary.get('healthy', 0)}")
        print(f"⚠️  Degraded: {summary.get('degraded', 0)}")
        print(f"❌ Unhealthy: {summary.get('unhealthy', 0)}")

        # Individual API status
        print(f"\n📋 API DETAILS:")
        print("-" * 80)

        api_details = health_data.get("api_details", {})
        for api_name, details in api_details.items():
            status = details.get("status", "unknown")
            response_time = details.get("response_time", 0)
            success_rate = details.get("success_rate", 0)
            error_count = details.get("error_count", 0)
            last_check = details.get("last_check", "")
            circuit_state = details.get("circuit_breaker_state", "UNKNOWN")

            # Format status with emoji
            api_emoji = status_emoji.get(status, "❓")

            # Format response time
            if response_time > 0:
                rt_str = f"{response_time:.2f}s"
            else:
                rt_str = "N/A"

            # Format success rate
            sr_str = f"{success_rate:.1%}"

            # Circuit breaker status
            cb_emoji = "🟢" if circuit_state == "CLOSED" else "🔴" if circuit_state == "OPEN" else "🟡"

            print(f"{api_emoji} {api_name:<15} | {rt_str:<6} | {sr_str:<6} | {error_count:<3} errs | {cb_emoji} {circuit_state:<6} | {last_check[-8:]}")

        print("-" * 80)

        # Recommendations
        if overall_status == "unhealthy":
            print("🚨 CRITICAL: Multiple APIs are failing. Check network connectivity and API keys.")
        elif overall_status == "degraded":
            print("⚠️  WARNING: Some APIs are experiencing issues. Monitor closely.")
        else:
            print("✅ All systems operational. API health is good.")

    except Exception as e:
        print(f"❌ Error retrieving health data: {e}")
        print("💡 Make sure the health monitoring service is running.")

    print("="*80)

def interactive_dashboard():
    """Run an interactive dashboard"""
    print("🔄 Starting API Health Dashboard...")
    print("Press Ctrl+C to exit")

    try:
        while True:
            print_health_dashboard()
            print("\n⏰ Refreshing in 30 seconds... (Ctrl+C to exit)")
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--check-now":
        print("🔍 Performing immediate API health check...")
        try:
            result = check_api_health_now()
            print("✅ Health check completed!")
            print_health_dashboard()
        except Exception as e:
            print(f"❌ Health check failed: {e}")
    else:
        interactive_dashboard()

if __name__ == "__main__":
    main()