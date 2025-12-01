# [LABEL:TOOL:24_6_orchestrator] [LABEL:FRAMEWORK:discord] [LABEL:FRAMEWORK:systemd]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-19] [LABEL:REVIEWED:pending]
#
# Purpose: 24/6 continuous workflow orchestrator for production deployment
# Dependencies: Discord integration, systemd, market calendar
# Related: src/agents/live_workflow_orchestrator.py, setup/deploy-vultr.sh
#
#!/usr/bin/env python3
"""
24/6 Continuous Workflow Orchestrator
Runs the trading analysis workflow continuously with Discord output and market-aware scheduling.
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, time as dt_time
from typing import Dict, Any, Optional
import schedule
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psutil
from simulations.idle_training_workflow import IdleTrainingWorkflow

from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator
from src.utils.vault_client import get_vault_secret
import exchange_calendars as ecals

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/24_6_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousWorkflowOrchestrator(LiveWorkflowOrchestrator):
    """
    Extended orchestrator for 24/6 continuous operation with market-aware scheduling.
    """

    def __init__(self):
        super().__init__()
        self.market_calendar = ecals.get_calendar('NYSE')
        self.last_workflow_date = None
        self.continuous_mode = True
        self.idle_trainer = IdleTrainingWorkflow()
        self.idle_trainer = IdleTrainingWorkflow()

        # 24/6 Schedule configuration - All times in Eastern Time (ET)
        self.schedules = {
            # Pre-market sessions - Start early for 2+ hours prep time
            'pre_market_prep': {'time': '06:00', 'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'], 'description': 'Early pre-market analysis and data collection'},
            'early_monday_prep': {'time': '05:30', 'days': ['monday'], 'description': 'Extra early Monday market regime assessment'},

            # Market preparation - 2+ hours before open
            'market_open_prep': {'time': '07:30', 'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'], 'description': 'Final pre-open analysis and position setup (2+ hours before open)'},
            'midday_check': {'time': '12:00', 'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'], 'description': 'Intraday performance and adjustment analysis'},
            'market_close_review': {'time': '16:30', 'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'], 'description': 'End-of-day performance review'},

            # Post-market sessions
            'post_market_review': {'time': '17:00', 'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'], 'description': 'Post-market analysis and next-day preparation'},

            # Weekend sessions
            'weekend_analysis': {'time': '10:00', 'days': ['saturday'], 'description': 'Weekly market trend analysis'},
            'sunday_prep': {'time': '18:00', 'days': ['sunday'], 'description': 'Sunday evening market preparation'}
        }

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now(self.market_calendar.tz)
        return self.market_calendar.is_open_on_timestamp(now)

    def is_trading_day(self) -> bool:
        """Check if today is a trading day"""
        today = datetime.now().date()
        return self.market_calendar.is_session(today)

    async def run_scheduled_workflow(self, workflow_type: str = 'full'):
        """Run a workflow based on schedule type"""
        if not self.channel:
            logger.error("No Discord channel available for workflow")
            return

        channel = self.channel

        # Get schedule config for this workflow type
        schedule_config = self.schedules.get(workflow_type, {})
        description = schedule_config.get('description', workflow_type.replace('_', ' ').title())

        # Check if we should run based on market conditions
        trading_day_workflows = ['pre_market_prep', 'market_open_prep', 'midday_check', 'market_close_review', 'post_market_review']
        weekend_workflows = ['weekend_analysis', 'sunday_prep', 'early_monday_prep']

        if workflow_type in trading_day_workflows:
            if not self.is_trading_day():
                await channel.send(f"📅 Skipping {description} - not a trading day")
                return
            if workflow_type == 'midday_check' and not self.is_market_open():
                await channel.send(f"🏢 Skipping {description} - market not open")
                return

        # Prevent duplicate workflows on same day for certain types
        today = datetime.now().date()
        if workflow_type in weekend_workflows and self.last_workflow_date == today:
            await channel.send(f"📅 Skipping {description} - already ran today")
            return

        await channel.send(f"🤖 **24/6 Scheduled Workflow: {description}**")
        await channel.send("⏰ Starting automated analysis workflow...")

        try:
            # Start the workflow
            await self.start_workflow()
            self.last_workflow_date = today

        except Exception as e:
            logger.error(f"Scheduled workflow failed: {e}")
            await channel.send(f"❌ Scheduled workflow failed: {str(e)}")

    async def monitor_market_conditions(self):
        """Continuously monitor market conditions and trigger workflows as needed"""
        while self.continuous_mode:
            try:
                # Check for significant market events (placeholder - integrate with your data feeds)
                market_events = await self._check_market_events()

                if market_events:
                    await self.channel.send("🚨 **Market Event Detected**")
                    for event in market_events:
                        await self.channel.send(f"• {event}")
                    await self.channel.send("🔄 Triggering emergency analysis workflow...")
                    await self.run_scheduled_workflow('emergency_analysis')

                # Check system health every hour
                if datetime.now().minute == 0:
                    health = await self.check_agent_health()
                    if health['overall_health'] in ['critical', 'degraded']:
                        await self.channel.send("⚠️ **System Health Alert**")
                        await self.channel.send(f"Status: {health['overall_health'].title()}")
                        await self.channel.send(f"Healthy agents: {len(health['healthy_agents'])}/{health['total_agents']}")

            except Exception as e:
                logger.error(f"Market monitoring error: {e}")

            # Check and run idle training if conditions allow
            if self.idle_trainer.is_safe_to_run():
                asyncio.create_task(self.idle_trainer.run_simulation_and_training())

            await asyncio.sleep(300)  # Check every 5 minutes

    async def _check_market_events(self) -> list:
        """Check for significant market events that warrant immediate analysis"""
        events = []

        try:
            # Placeholder - integrate with your market data feeds
            # This could check for:
            # - VIX spikes
            # - Major index moves
            # - Economic data releases
            # - News sentiment changes

            # Example checks (replace with real implementations)
            vix_level = await self._get_vix_level()
            if vix_level and vix_level > 30:  # High volatility
                events.append(f"High VIX: {vix_level}")

            market_move = await self._get_market_move()
            if market_move and abs(market_move) > 2.0:  # 2%+ move
                events.append(f"Market Move: {market_move:+.2f}%")

        except Exception as e:
            logger.warning(f"Market event check failed: {e}")

        return events

    async def _get_vix_level(self) -> Optional[float]:
        """Get current VIX level (placeholder)"""
        # Integrate with your data sources
        return None

    async def _get_market_move(self) -> Optional[float]:
        """Get today's market move (placeholder)"""
        # Integrate with your data sources
        return None

    def setup_schedules(self):
        """Set up scheduled workflows"""
        for schedule_name, config in self.schedules.items():
            schedule_time = config['time']
            days = config['days']

            # Create schedule for each day
            for day in days:
                try:
                    # Get the day method from schedule.every()
                    day_method = getattr(schedule.every(), day)
                    day_method.at(schedule_time).do(
                        lambda s=schedule_name: asyncio.create_task(self.run_scheduled_workflow(s))
                    )
                    logger.info(f"✅ Scheduled {schedule_name} for {day} at {schedule_time} ET")
                except Exception as e:
                    logger.error(f"Failed to schedule {schedule_name} for {day}: {e}")

    async def run_24_6_orchestrator(self):
        """Main 24/6 orchestrator loop"""
        logger.info("🚀 Starting 24/6 Continuous Workflow Orchestrator")

        # Initialize Discord client
        await self.initialize_discord_client()

        # Initialize agents
        await self.initialize_agents_async()

        # Setup scheduled workflows
        self.setup_schedules()
        logger.info("📅 Scheduled workflows configured")

        # Start market monitoring
        monitor_task = asyncio.create_task(self.monitor_market_conditions())

        # Announce 24/6 mode
        if self.channel:
            await self.channel.send("🤖 **24/6 Continuous Orchestrator Online**")
            await self.channel.send("📅 Automated workflows scheduled for market hours")
            await self.channel.send("👀 Monitoring market conditions continuously")
            await self.channel.send("💡 Manual commands still available: `!start_workflow`, `!status`, etc.")

        try:
            # Start Discord client (this will run forever)
            token = get_vault_secret('DISCORD_ORCHESTRATOR_TOKEN')
            if not token:
                raise ValueError("❌ DISCORD_ORCHESTRATOR_TOKEN not found")

            # Run schedule checker in parallel
            async def run_scheduler():
                while self.continuous_mode:
                    schedule.run_pending()
                    await asyncio.sleep(60)  # Check every minute

            scheduler_task = asyncio.create_task(run_scheduler())

            # Start Discord client
            await self.client.start(token)

        except KeyboardInterrupt:
            logger.info("🛑 24/6 Orchestrator shutting down...")
            self.continuous_mode = False
            monitor_task.cancel()
            if self.client:
                await self.client.close()
        except Exception as e:
            logger.error(f"24/6 Orchestrator error: {e}")
            self.continuous_mode = False
            monitor_task.cancel()
            if self.client:
                await self.client.close()
            raise

async def main():
    """Main entry point for 24/6 orchestrator"""
    print("🤖 ABC Application - 24/6 Continuous Workflow Orchestrator")
    print("=" * 60)
    print("🎯 Features:")
    print("  • Continuous 24/6 operation with Discord output")
    print("  • Market-aware scheduled workflows")
    print("  • Real-time market condition monitoring")
    print("  • Automated emergency analysis triggers")
    print("  • Manual intervention capabilities")
    print("")
    print("📅 Scheduled Workflows (Eastern Time - ET):")
    print("  • Early Monday Prep (Mon 5:30 AM ET)")
    print("  • Pre-Market Prep (Mon-Fri 6:00 AM ET)")
    print("  • Market Open Prep (Mon-Fri 7:30 AM ET) - 2+ hours before open")
    print("  • Midday Check (Mon-Fri 12:00 PM ET)")
    print("  • Market Close Review (Mon-Fri 4:30 PM ET)")
    print("  • Post-Market Review (Mon-Fri 5:00 PM ET)")
    print("  • Weekend Analysis (Sat 10:00 AM ET)")
    print("  • Sunday Prep (Sun 6:00 PM ET)")
    print("")
    print("💡 Manual Discord Commands:")
    print("  !start_workflow  - Manual workflow trigger")
    print("  !workflow_status - Check current status")
    print("  !status         - System health check")
    print("  💬 Questions and interventions anytime!")
    print("")

    # Check environment
    required_env = ['DISCORD_ORCHESTRATOR_TOKEN', 'DISCORD_GUILD_ID']
    missing = [env for env in required_env if not get_vault_secret(env)]
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        print("   Please configure these in your vault or .env file")
        return

    orchestrator = ContinuousWorkflowOrchestrator()
    await orchestrator.run_24_6_orchestrator()

if __name__ == "__main__":
    asyncio.run(main())