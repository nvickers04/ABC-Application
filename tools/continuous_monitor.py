#!/usr/bin/env python3
"""
Continuous Trading Day Monitor - Runs throughout trading session
Monitors system health, errors, and performance for improvement data.
"""

import asyncio
import logging
import time
import signal
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('src'))

from src.utils.trading_day_monitor import get_monitor, check_system_health, save_monitoring_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousMonitor:
    """Runs continuous monitoring throughout the trading day."""

    def __init__(self):
        self.monitor = get_monitor()
        self.running = True
        self.check_interval = 300  # 5 minutes
        self.save_interval = 3600  # 1 hour

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    async def run_monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("🚀 Starting Continuous Trading Day Monitor")
        logger.info(f"📊 Health check interval: {self.check_interval}s")
        logger.info(f"💾 Report save interval: {self.save_interval}s")

        last_save = time.time()
        check_count = 0

        try:
            while self.running:
                check_count += 1
                logger.info(f"🔍 Performing system health check #{check_count}")

                # Perform comprehensive health check
                await check_system_health()

                # Save periodic report
                current_time = time.time()
                if current_time - last_save >= self.save_interval:
                    logger.info("💾 Saving periodic monitoring report...")
                    try:
                        filename = await save_monitoring_report()
                        logger.info(f"✅ Report saved: {filename}")
                        last_save = current_time
                    except Exception as e:
                        logger.error(f"❌ Failed to save report: {e}")

                # Wait for next check
                await asyncio.sleep(self.check_interval)

        except Exception as e:
            logger.error(f"❌ Monitoring loop error: {e}")
            await self._emergency_save()

        finally:
            await self._shutdown()

    async def _emergency_save(self):
        """Emergency save of monitoring data."""
        try:
            logger.info("🚨 Performing emergency save of monitoring data...")
            filename = await save_monitoring_report("data/emergency_monitoring_report.json")
            logger.info(f"✅ Emergency report saved: {filename}")
        except Exception as e:
            logger.error(f"❌ Emergency save failed: {e}")

    async def _shutdown(self):
        """Graceful shutdown with final report."""
        logger.info("🔄 Shutting down Continuous Monitor...")

        try:
            # Final health check
            logger.info("🔍 Performing final system health check...")
            await check_system_health()

            # Generate final comprehensive report
            logger.info("📊 Generating final trading day report...")
            filename = await save_monitoring_report()
            logger.info(f"✅ Final report saved: {filename}")

            # Log summary
            report = await self.monitor.generate_report()
            total_errors = sum(len(e) for e in report['error_analysis'].values())
            total_warnings = sum(len(w) for w in report['warning_analysis'].values())
            rate_limits = report['rate_limiting']['total_events']

            logger.info("📈 Trading Day Summary:")
            logger.info(f"   • Runtime: {report['session_info']['runtime_hours']:.2f} hours")
            logger.info(f"   • Total Errors: {total_errors}")
            logger.info(f"   • Total Warnings: {total_warnings}")
            logger.info(f"   • Rate Limit Events: {rate_limits}")
            logger.info(f"   • Recommendations: {len(report['recommendations'])}")

            if report['recommendations']:
                logger.info("🎯 Key Recommendations:")
                for rec in report['recommendations'][:5]:  # Show top 5
                    logger.info(f"   • {rec}")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

        logger.info("✅ Continuous Monitor shutdown complete")

def main():
    """Main entry point."""
    print("🎯 ABC-Application Trading Day Monitor")
    print("=" * 50)
    print("This monitor will run continuously throughout the trading day,")
    print("tracking errors, warnings, and system health for improvement data.")
    print()
    print("Press Ctrl+C to stop and generate final report.")
    print()

    monitor = ContinuousMonitor()

    try:
        asyncio.run(monitor.run_monitoring_loop())
    except KeyboardInterrupt:
        print("\n⏹️  Monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()