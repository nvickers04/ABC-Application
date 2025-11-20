"""
AdaptiveScheduler - Dynamic agent coordination based on market conditions and earnings events.

This module provides intelligent scheduling of agent activities based on:
- Earnings calendar events
- Market volatility conditions
- Risk thresholds
- Agent performance metrics
"""

import asyncio
import datetime
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for adaptive scheduling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class MarketCondition(Enum):
    """Market condition classifications."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    EARNINGS_SEASON = "earnings_season"
    BLACK_SWAN = "black_swan"


@dataclass
class EarningsEvent:
    """Represents an earnings event."""
    symbol: str
    event_date: datetime.datetime
    event_time: str  # "before_market", "after_market", "during_market"
    expected_impact: str  # "high", "medium", "low"
    analyst_consensus: Optional[float] = None
    previous_quarter_beat: Optional[bool] = None


@dataclass
class AgentSchedule:
    """Represents a scheduled agent activity."""
    agent_name: str
    frequency_minutes: int
    risk_threshold: RiskLevel
    market_conditions: List[MarketCondition]
    last_run: Optional[datetime.datetime] = None
    next_run: Optional[datetime.datetime] = None
    is_active: bool = True


class AdaptiveScheduler:
    """
    Adaptive scheduler that dynamically adjusts agent frequencies based on market conditions,
    earnings events, and risk levels.
    """

    def __init__(self, a2a_protocol=None):
        """
        Initialize the AdaptiveScheduler.

        Args:
            a2a_protocol: Agent-to-agent communication protocol
        """
        self.a2a_protocol = a2a_protocol
        self.logger = logging.getLogger(__name__)

        # Agent schedules with default configurations
        self.agent_schedules = {
            'RiskAgent': AgentSchedule(
                agent_name='RiskAgent',
                frequency_minutes=15,  # Base frequency
                risk_threshold=RiskLevel.MEDIUM,
                market_conditions=[MarketCondition.NORMAL, MarketCondition.VOLATILE]
            ),
            'ExecutionAgent': AgentSchedule(
                agent_name='ExecutionAgent',
                frequency_minutes=5,
                risk_threshold=RiskLevel.LOW,
                market_conditions=[MarketCondition.NORMAL]
            ),
            'StrategyAgent': AgentSchedule(
                agent_name='StrategyAgent',
                frequency_minutes=30,
                risk_threshold=RiskLevel.MEDIUM,
                market_conditions=[MarketCondition.NORMAL, MarketCondition.VOLATILE]
            ),
            'LearningAgent': AgentSchedule(
                agent_name='LearningAgent',
                frequency_minutes=60,
                risk_threshold=RiskLevel.LOW,
                market_conditions=[MarketCondition.NORMAL]
            )
        }

        # Earnings calendar
        self.earnings_calendar: List[EarningsEvent] = []

        # Market state tracking
        self.current_market_condition = MarketCondition.NORMAL
        self.current_risk_level = RiskLevel.MEDIUM
        self.vix_level = 20.0  # Default VIX level

        # Background tasks
        self._scheduler_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Callbacks
        self.on_schedule_change: Optional[Callable] = None

        self.logger.info("AdaptiveScheduler initialized")

    async def start_scheduler(self):
        """Start the adaptive scheduling system."""
        if self._is_running:
            self.logger.warning("AdaptiveScheduler already running")
            return

        self._is_running = True
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        self.logger.info("AdaptiveScheduler started")

    async def stop_scheduler(self):
        """Stop the adaptive scheduling system."""
        self._is_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        self.logger.info("AdaptiveScheduler stopped")

    async def _run_scheduler(self):
        """Main scheduler loop."""
        while self._is_running:
            try:
                # Update market conditions
                await self._update_market_conditions()

                # Update agent schedules based on current conditions
                await self._update_agent_schedules()

                # Check for earnings events
                await self._check_earnings_events()

                # Execute scheduled agent activities
                await self._execute_scheduled_activities()

                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)

    async def _update_market_conditions(self):
        """Update current market conditions and risk levels."""
        try:
            # Get VIX level (volatility indicator)
            vix_level = await self._get_vix_level()
            self.vix_level = vix_level

            # Determine market condition
            if vix_level > 30:
                self.current_market_condition = MarketCondition.BLACK_SWAN
            elif vix_level > 20:
                self.current_market_condition = MarketCondition.VOLATILE
            else:
                self.current_market_condition = MarketCondition.NORMAL

            # Check if we're in earnings season (simplified)
            current_month = datetime.datetime.now().month
            earnings_months = [1, 4, 7, 10]  # Quarterly earnings seasons
            if current_month in earnings_months:
                # Check if there are earnings events in the next 7 days
                upcoming_earnings = self._get_upcoming_earnings(days_ahead=7)
                if upcoming_earnings:
                    self.current_market_condition = MarketCondition.EARNINGS_SEASON

            # Determine risk level based on conditions
            if self.current_market_condition == MarketCondition.BLACK_SWAN:
                self.current_risk_level = RiskLevel.EXTREME
            elif self.current_market_condition == MarketCondition.EARNINGS_SEASON:
                self.current_risk_level = RiskLevel.HIGH
            elif self.current_market_condition == MarketCondition.VOLATILE:
                self.current_risk_level = RiskLevel.HIGH
            else:
                self.current_risk_level = RiskLevel.MEDIUM

            self.logger.debug(f"Market conditions updated: condition={self.current_market_condition.value}, "
                            f"risk={self.current_risk_level.value}, vix={self.vix_level}")

        except Exception as e:
            self.logger.error(f"Error updating market conditions: {e}")

    async def _update_agent_schedules(self):
        """Update agent schedules based on current market conditions and risk levels."""
        try:
            for agent_name, schedule in self.agent_schedules.items():
                old_frequency = schedule.frequency_minutes

                # Adjust frequency based on risk level
                if self.current_risk_level == RiskLevel.EXTREME:
                    # Maximum frequency during extreme risk
                    schedule.frequency_minutes = max(1, schedule.frequency_minutes // 4)
                elif self.current_risk_level == RiskLevel.HIGH:
                    # Increased frequency during high risk
                    schedule.frequency_minutes = max(2, schedule.frequency_minutes // 2)
                elif self.current_risk_level == RiskLevel.MEDIUM:
                    # Normal frequency
                    schedule.frequency_minutes = schedule.frequency_minutes  # Keep base frequency
                else:
                    # Reduced frequency during low risk
                    schedule.frequency_minutes = schedule.frequency_minutes * 2

                # Special handling for earnings season
                if self.current_market_condition == MarketCondition.EARNINGS_SEASON:
                    if agent_name == 'RiskAgent':
                        # RiskAgent gets highest priority during earnings
                        schedule.frequency_minutes = max(1, schedule.frequency_minutes // 2)
                    elif agent_name == 'ExecutionAgent':
                        # ExecutionAgent more cautious during earnings
                        schedule.frequency_minutes = schedule.frequency_minutes * 2

                # Check if frequency changed
                if old_frequency != schedule.frequency_minutes:
                    self.logger.info(f"Updated {agent_name} frequency: {old_frequency}min -> {schedule.frequency_minutes}min "
                                   f"(condition: {self.current_market_condition.value}, risk: {self.current_risk_level.value})")

                    # Notify callback if set
                    if self.on_schedule_change:
                        await self.on_schedule_change(agent_name, old_frequency, schedule.frequency_minutes)

        except Exception as e:
            self.logger.error(f"Error updating agent schedules: {e}")

    async def _check_earnings_events(self):
        """Check for upcoming earnings events and adjust schedules accordingly."""
        try:
            now = datetime.datetime.now()
            upcoming_events = []

            # Find events in the next 24 hours
            for event in self.earnings_calendar:
                time_diff = event.event_date - now
                if 0 < time_diff.total_seconds() < 86400:  # Next 24 hours
                    upcoming_events.append(event)

            if upcoming_events:
                self.logger.info(f"Found {len(upcoming_events)} upcoming earnings events in next 24 hours")

                # Increase RiskAgent frequency for high-impact earnings
                high_impact_events = [e for e in upcoming_events if e.expected_impact == 'high']
                if high_impact_events:
                    risk_schedule = self.agent_schedules['RiskAgent']
                    risk_schedule.frequency_minutes = max(1, risk_schedule.frequency_minutes // 2)
                    self.logger.info("Increased RiskAgent frequency due to high-impact earnings events")

        except Exception as e:
            self.logger.error(f"Error checking earnings events: {e}")

    async def _execute_scheduled_activities(self):
        """Execute scheduled agent activities based on their schedules."""
        try:
            now = datetime.datetime.now()

            for agent_name, schedule in self.agent_schedules.items():
                if not schedule.is_active:
                    continue

                # Check if it's time to run this agent
                if schedule.next_run is None or now >= schedule.next_run:
                    # Execute agent activity
                    await self._execute_agent_activity(agent_name)

                    # Schedule next run
                    schedule.last_run = now
                    schedule.next_run = now + datetime.timedelta(minutes=schedule.frequency_minutes)

        except Exception as e:
            self.logger.error(f"Error executing scheduled activities: {e}")

    async def _execute_agent_activity(self, agent_name: str):
        """Execute a specific agent's scheduled activity."""
        try:
            self.logger.debug(f"Executing scheduled activity for {agent_name}")

            # This would integrate with the actual agent execution
            # For now, we'll simulate the activity
            if agent_name == 'RiskAgent':
                # Risk assessment activity
                await self._perform_risk_assessment()
            elif agent_name == 'ExecutionAgent':
                # Execution monitoring activity
                await self._perform_execution_monitoring()
            elif agent_name == 'StrategyAgent':
                # Strategy review activity
                await self._perform_strategy_review()
            elif agent_name == 'LearningAgent':
                # Learning update activity
                await self._perform_learning_update()

        except Exception as e:
            self.logger.error(f"Error executing {agent_name} activity: {e}")

    async def _perform_risk_assessment(self):
        """Perform scheduled risk assessment."""
        # This would call the actual RiskAgent
        self.logger.debug("Performing scheduled risk assessment")

    async def _perform_execution_monitoring(self):
        """Perform scheduled execution monitoring."""
        # This would call the actual ExecutionAgent
        self.logger.debug("Performing scheduled execution monitoring")

    async def _perform_strategy_review(self):
        """Perform scheduled strategy review."""
        # This would call the actual StrategyAgent
        self.logger.debug("Performing scheduled strategy review")

    async def _perform_learning_update(self):
        """Perform scheduled learning update."""
        # This would call the actual LearningAgent
        self.logger.debug("Performing scheduled learning update")

    async def _get_vix_level(self) -> float:
        """Get current VIX level from market data."""
        try:
            # In a real implementation, this would get VIX from IBKR or other data source
            # For now, return a simulated value based on time of day
            hour = datetime.datetime.now().hour
            if 9 <= hour <= 16:  # Market hours
                return 18.0 + (datetime.datetime.now().minute % 10)  # Simulate some variation
            else:
                return 20.0  # Default level
        except Exception as e:
            self.logger.error(f"Error getting VIX level: {e}")
            return 20.0

    def add_earnings_event(self, event: EarningsEvent):
        """Add an earnings event to the calendar."""
        self.earnings_calendar.append(event)
        self.logger.info(f"Added earnings event: {event.symbol} on {event.event_date}")

    def remove_earnings_event(self, symbol: str, event_date: datetime.datetime):
        """Remove an earnings event from the calendar."""
        self.earnings_calendar = [
            e for e in self.earnings_calendar
            if not (e.symbol == symbol and e.event_date.date() == event_date.date())
        ]
        self.logger.info(f"Removed earnings event: {symbol} on {event_date}")

    def _get_upcoming_earnings(self, days_ahead: int = 7) -> List[EarningsEvent]:
        """Get earnings events in the next N days."""
        now = datetime.datetime.now()
        cutoff = now + datetime.timedelta(days=days_ahead)

        return [
            event for event in self.earnings_calendar
            if now <= event.event_date <= cutoff
        ]

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status and configuration."""
        return {
            'is_running': self._is_running,
            'current_market_condition': self.current_market_condition.value,
            'current_risk_level': self.current_risk_level.value,
            'vix_level': self.vix_level,
            'agent_schedules': {
                name: {
                    'frequency_minutes': schedule.frequency_minutes,
                    'is_active': schedule.is_active,
                    'last_run': schedule.last_run.isoformat() if schedule.last_run else None,
                    'next_run': schedule.next_run.isoformat() if schedule.next_run else None
                }
                for name, schedule in self.agent_schedules.items()
            },
            'upcoming_earnings': [
                {
                    'symbol': e.symbol,
                    'event_date': e.event_date.isoformat(),
                    'event_time': e.event_time,
                    'expected_impact': e.expected_impact
                }
                for e in self._get_upcoming_earnings(7)
            ]
        }

    def set_schedule_change_callback(self, callback: Callable):
        """Set callback for schedule changes."""
        self.on_schedule_change = callback

    def update_agent_schedule(self, agent_name: str, frequency_minutes: int = None,
                            is_active: bool = None):
        """Update an agent's schedule parameters."""
        if agent_name in self.agent_schedules:
            schedule = self.agent_schedules[agent_name]
            if frequency_minutes is not None:
                old_freq = schedule.frequency_minutes
                schedule.frequency_minutes = frequency_minutes
                self.logger.info(f"Updated {agent_name} frequency: {old_freq}min -> {frequency_minutes}min")
            if is_active is not None:
                schedule.is_active = is_active
                self.logger.info(f"Updated {agent_name} active status: {is_active}")
        else:
            self.logger.warning(f"Agent {agent_name} not found in schedules")