#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for ABC Trading System
Provides comprehensive visibility into system status and trading activity
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import requests

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.agents.memory import MemoryAgent
    from src.agents.learning import LearningAgent
    from src.integrations.live_trading_safeguards import LiveTradingSafeguards
    from src.utils.config import Config
    import yfinance as yf
except ImportError as e:
    print(f"Import error: {e}")
    print("Some components may not be available")
    Config = None

class MonitoringDashboard:
    def __init__(self):
        self.config = Config() if Config else None
        self.log_dir = Path("logs")
        self.data_dir = Path("data")

    async def get_system_status(self):
        """Get overall system health status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_health": "unknown",
            "components": {},
            "warnings": [],
            "errors": []
        }

        # Check Python processes
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if any('abc' in arg.lower() or 'trading' in arg.lower() for arg in cmdline):
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'cmdline': ' '.join(cmdline),
                            'cpu_percent': proc.cpu_percent(),
                            'memory_mb': proc.memory_info().rss / 1024 / 1024
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        status["components"]["python_processes"] = python_processes

        # Check TigerBeetle
        tigerbeetle_running = False
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'tigerbeetle' in proc.info['name'].lower():
                    tigerbeetle_running = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        status["components"]["tigerbeetle"] = {
            "running": tigerbeetle_running,
            "status": "running" if tigerbeetle_running else "not running"
        }

        if not tigerbeetle_running:
            status["warnings"].append("TigerBeetle database is not running")

        # Check Redis
        redis_running = False
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            redis_running = True
        except:
            redis_running = False

        status["components"]["redis"] = {
            "running": redis_running,
            "status": "running" if redis_running else "not running"
        }

        if not redis_running:
            status["warnings"].append("Redis cache is not available")

        return status

    async def get_market_data_status(self):
        """Get current market data status"""
        market_status = {
            "timestamp": datetime.now().isoformat(),
            "market_open": False,
            "vix_level": None,
            "spx_level": None,
            "last_update": None
        }

        try:
            # Check if market is open (simplified - weekdays 9:30-16:00 ET)
            now = datetime.now()
            if now.weekday() < 5 and 14 <= now.hour < 21:  # 9:30-16:00 ET in UTC
                market_status["market_open"] = True

            # Get VIX data
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d", interval="1m")
            if not vix_data.empty:
                market_status["vix_level"] = float(vix_data['Close'].iloc[-1])
                market_status["last_update"] = vix_data.index[-1].isoformat()

            # Get SPX data
            spx = yf.Ticker("^GSPC")
            spx_data = spx.history(period="1d", interval="1m")
            if not spx_data.empty:
                market_status["spx_level"] = float(spx_data['Close'].iloc[-1])

        except Exception as e:
            market_status["error"] = str(e)

        return market_status

    async def get_agent_status(self):
        """Get status of AI agents"""
        agent_status = {
            "timestamp": datetime.now().isoformat(),
            "agents": {}
        }

        # Check Memory Agent
        try:
            memory_agent = MemoryAgent()
            stats = await memory_agent._calculate_memory_performance_stats()
            agent_status["agents"]["memory"] = {
                "status": "operational",
                "performance_stats": stats
            }
        except Exception as e:
            agent_status["agents"]["memory"] = {
                "status": "error",
                "error": str(e)
            }

        # Check Learning Agent
        try:
            learning_agent = LearningAgent()
            # Try to get some basic status
            agent_status["agents"]["learning"] = {
                "status": "operational"
            }
        except Exception as e:
            agent_status["agents"]["learning"] = {
                "status": "error",
                "error": str(e)
            }

        # Check Trading Safeguards
        try:
            safeguards = LiveTradingSafeguards()
            agent_status["agents"]["safeguards"] = {
                "status": "operational"
            }
        except Exception as e:
            agent_status["agents"]["safeguards"] = {
                "status": "error",
                "error": str(e)
            }

        return agent_status

    async def get_recent_activity(self):
        """Get recent trading and system activity"""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "recent_logs": [],
            "trading_activity": [],
            "system_events": []
        }

        # Read recent log entries
        log_files = [
            self.log_dir / "abc_application.log",
            self.log_dir / "audit.log",
            self.log_dir / "24_6_orchestrator.log"
        ]

        for log_file in log_files:
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[-10:]  # Last 10 lines
                        activity["recent_logs"].extend([{
                            "file": log_file.name,
                            "line": line.strip()
                        } for line in lines if line.strip()])
                except Exception as e:
                    activity["recent_logs"].append({
                        "file": log_file.name,
                        "error": str(e)
                    })

        # Check for trading data files
        trading_files = [
            "live_workflow_results.json",
            "trade_monitoring_results.json",
            "trading_day_report_*.json"
        ]

        for pattern in trading_files:
            if "*" in pattern:
                # Handle wildcards
                import glob
                matches = glob.glob(str(self.data_dir / pattern))
                for match in matches[-3:]:  # Last 3 matching files
                    try:
                        with open(match, 'r') as f:
                            data = json.load(f)
                            activity["trading_activity"].append({
                                "file": Path(match).name,
                                "timestamp": data.get("timestamp"),
                                "summary": f"Found {len(data.get('results', []))} results"
                            })
                    except:
                        pass
            else:
                file_path = self.data_dir / pattern
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            activity["trading_activity"].append({
                                "file": pattern,
                                "timestamp": data.get("timestamp"),
                                "summary": f"Found {len(data.get('results', []))} results"
                            })
                    except:
                        pass

        return activity

    async def get_langfuse_status(self):
        """Check Langfuse monitoring status"""
        langfuse_status = {
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "metrics": {}
        }

        try:
            # Check if Langfuse config exists
            config_path = Path("config/langfuse_config.yaml")
            if config_path.exists():
                langfuse_status["config_exists"] = True

                # Try to check Langfuse API (if configured)
                # This would require actual Langfuse credentials
                langfuse_status["status"] = "configured_but_not_checked"
            else:
                langfuse_status["status"] = "not_configured"
                langfuse_status["config_exists"] = False

        except Exception as e:
            langfuse_status["error"] = str(e)

        return langfuse_status

    async def display_dashboard(self):
        """Display the monitoring dashboard"""
        print("\n" + "="*80)
        print("ABC TRADING SYSTEM - MONITORING DASHBOARD")
        print("="*80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # System Status
        print("\n🔧 SYSTEM STATUS:")
        system_status = await self.get_system_status()
        print(f"  Overall Health: {system_status.get('system_health', 'unknown')}")

        print("  Components:")
        for component, info in system_status["components"].items():
            if isinstance(info, list):  # Python processes
                print(f"    {component}: {len(info)} processes running")
                for proc in info[:2]:  # Show first 2
                    print(f"      PID {proc['pid']}: {proc['cmdline'][:60]}...")
            else:
                status = info.get("status", "unknown")
                emoji = "✅" if "running" in status else "❌"
                print(f"    {component}: {emoji} {status}")

        if system_status["warnings"]:
            print("  ⚠️  Warnings:")
            for warning in system_status["warnings"]:
                print(f"    - {warning}")

        # Market Data
        print("\n📊 MARKET DATA:")
        market_status = await self.get_market_data_status()
        market_emoji = "🟢" if market_status["market_open"] else "🔴"
        print(f"  Market Status: {market_emoji} {'Open' if market_status['market_open'] else 'Closed'}")
        if market_status["vix_level"]:
            print(f"  VIX Level: {market_status['vix_level']:.2f}")
        if market_status["spx_level"]:
            print(f"  SPX Level: {market_status['spx_level']:.2f}")
        # Agent Status
        print("\n🤖 AI AGENTS:")
        agent_status = await self.get_agent_status()
        for agent_name, info in agent_status["agents"].items():
            status = info["status"]
            emoji = "✅" if status == "operational" else "❌"
            print(f"  {agent_name.title()}: {emoji} {status}")
            if "error" in info:
                print(f"    Error: {info['error']}")

        # Langfuse Status
        print("\n📈 LANGFUSE MONITORING:")
        langfuse_status = await self.get_langfuse_status()
        status = langfuse_status["status"]
        emoji = "✅" if "configured" in status else "❌"
        print(f"  Status: {emoji} {status.replace('_', ' ').title()}")

        # Recent Activity
        print("\n📋 RECENT ACTIVITY:")
        activity = await self.get_recent_activity()

        print("  Recent Logs:")
        if activity["recent_logs"]:
            for log_entry in activity["recent_logs"][-5:]:  # Show last 5
                file_name = log_entry.get("file", "unknown")
                if "error" in log_entry:
                    print(f"    {file_name}: Error reading - {log_entry['error']}")
                else:
                    line = log_entry.get("line", "")[:80]
                    print(f"    {file_name}: {line}")
        else:
            print("    No recent log activity")

        print("  Trading Activity:")
        if activity["trading_activity"]:
            for trade in activity["trading_activity"][-3:]:  # Show last 3
                print(f"    {trade['file']}: {trade.get('summary', 'No summary')}")
        else:
            print("    No recent trading activity")

        print("\n" + "="*80)

    async def run_continuous_monitoring(self, interval_seconds=30):
        """Run continuous monitoring"""
        print("Starting continuous monitoring... (Ctrl+C to stop)")
        try:
            while True:
                await self.display_dashboard()
                print(f"\nNext update in {interval_seconds} seconds...")
                await asyncio.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

async def main():
    dashboard = MonitoringDashboard()

    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        await dashboard.run_continuous_monitoring()
    else:
        await dashboard.display_dashboard()

if __name__ == "__main__":
    asyncio.run(main())