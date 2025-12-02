# src/utils/alert_manager.py
"""
AlertManager - Centralized alerting system for the ABC-Application

Provides unified error handling, logging, and notification across all components.
Supports Discord notifications, health check integration, and fail-fast error handling.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import os
import sys

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure"""
    level: AlertLevel
    message: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = "unknown"
    error_id: Optional[str] = None


class HealthAlertError(Exception):
    """Exception raised for critical health alerts that should abort operations"""

    def __init__(self, message: str, alert: Alert):
        super().__init__(message)
        self.alert = alert


class AlertManager:
    """
    Singleton AlertManager for centralized error handling and notifications.

    Features:
    - Unified logging with structured context
    - Discord notifications via orchestrator integration
    - Health check triggering on critical errors
    - Error queue for monitoring and analysis
    - Fail-fast critical error handling
    """

    _instance: Optional['AlertManager'] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> 'AlertManager':
        if cls._instance is None:
            with cls._lock:  # Note: This is sync lock, for singleton
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.error_queue: List[Alert] = []
        self.max_queue_size = 1000
        self.orchestrator = None  # Will be set by orchestrator
        self.discord_alerts_channel = os.getenv('DISCORD_ALERTS_CHANNEL', 'alerts')

        # Email/Slack integration (optional)
        self.email_enabled = bool(os.getenv('ALERT_EMAIL_ENABLED', False))
        self.slack_enabled = bool(os.getenv('ALERT_SLACK_ENABLED', False))

        logger.info("AlertManager initialized")

    def set_orchestrator(self, orchestrator):
        """Set orchestrator reference for Discord notifications"""
        self.orchestrator = orchestrator
        logger.info("Orchestrator reference set for Discord notifications")

    def critical(self, error: Exception, context: Optional[Dict[str, Any]] = None, component: str = "unknown"):
        """Handle critical errors - log, alert, and trigger health check"""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            message=str(error),
            context=context or {},
            component=component
        )

        # Schedule async processing
        asyncio.create_task(self._process_alert(alert))

        # Trigger health check if orchestrator available (sync call)
        if self.orchestrator and hasattr(self.orchestrator, 'perform_system_health_check'):
            try:
                # Create task for health check
                asyncio.create_task(self.orchestrator.perform_system_health_check())
            except Exception as e:
                logger.error(f"Failed to trigger health check: {e}")

        # Raise HealthAlertError for fail-fast behavior
        raise HealthAlertError(f"Critical alert: {error}", alert)

    async def error(self, error: Exception, context: Optional[Dict[str, Any]] = None, component: str = "unknown"):
        """Handle error-level alerts"""
        alert = Alert(
            level=AlertLevel.ERROR,
            message=str(error),
            context=context or {},
            component=component
        )
        await self._process_alert(alert)

    async def warning(self, message: str, context: Optional[Dict[str, Any]] = None, component: str = "unknown"):
        """Handle warning-level alerts"""
        alert = Alert(
            level=AlertLevel.WARNING,
            message=message,
            context=context or {},
            component=component
        )
        await self._process_alert(alert)

    async def info(self, message: str, context: Optional[Dict[str, Any]] = None, component: str = "unknown"):
        """Handle info-level alerts"""
        alert = Alert(
            level=AlertLevel.INFO,
            message=message,
            context=context or {},
            component=component
        )
        await self._process_alert(alert)

    async def debug(self, message: str, context: Optional[Dict[str, Any]] = None, component: str = "unknown"):
        """Handle debug-level alerts"""
        alert = Alert(
            level=AlertLevel.DEBUG,
            message=message,
            context=context or {},
            component=component
        )
        await self._process_alert(alert)

    async def _process_alert(self, alert: Alert):
        """Process an alert: log, queue, notify"""
        # Log the alert
        log_message = f"[{alert.level.value.upper()}] {alert.component}: {alert.message}"
        if alert.context:
            log_message += f" | Context: {alert.context}"

        if alert.level == AlertLevel.CRITICAL:
            logger.critical(log_message)
        elif alert.level == AlertLevel.ERROR:
            logger.error(log_message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_message)
        elif alert.level == AlertLevel.INFO:
            logger.info(log_message)
        else:
            logger.debug(log_message)

        # Add to error queue
        self.error_queue.append(alert)
        if len(self.error_queue) > self.max_queue_size:
            self.error_queue.pop(0)  # Remove oldest

        # Send Discord notification
        await self._send_discord_alert(alert)

        # Send email/Slack if enabled
        await self._send_external_alerts(alert)

    async def _send_discord_alert(self, alert: Alert):
        """Send alert to Discord alerts channel"""
        if not self.orchestrator:
            return

        try:
            # Find alerts channel
            alerts_channel = None
            if hasattr(self.orchestrator, 'channel') and self.orchestrator.channel:
                # Check if current channel is alerts
                if hasattr(self.orchestrator.channel, 'name') and self.orchestrator.channel.name == self.discord_alerts_channel:
                    alerts_channel = self.orchestrator.channel
                else:
                    # Try to find alerts channel in guild
                    if hasattr(self.orchestrator, 'guild') and self.orchestrator.guild:
                        for ch in self.orchestrator.guild.text_channels:
                            if ch.name == self.discord_alerts_channel:
                                alerts_channel = ch
                                break

            if not alerts_channel:
                logger.warning(f"Alerts channel '{self.discord_alerts_channel}' not found")
                return

            # Create embed
            embed_color = {
                AlertLevel.CRITICAL: 0xFF0000,  # Red
                AlertLevel.ERROR: 0xFF6600,     # Orange
                AlertLevel.WARNING: 0xFFFF00,   # Yellow
                AlertLevel.INFO: 0x00FF00,      # Green
                AlertLevel.DEBUG: 0x666666     # Gray
            }.get(alert.level, 0xFFFFFF)

            embed = {
                'title': f"🚨 {alert.level.value.upper()} Alert",
                'description': alert.message,
                'color': embed_color,
                'fields': [
                    {'name': 'Component', 'value': alert.component, 'inline': True},
                    {'name': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'inline': True}
                ],
                'footer': {'text': f"Alert ID: {alert.error_id or 'N/A'}"}
            }

            if alert.context:
                context_str = "\n".join(f"{k}: {v}" for k, v in alert.context.items())
                embed['fields'].append({'name': 'Context', 'value': context_str[:1024], 'inline': False})

            # Send embed
            await alerts_channel.send(embed=embed)

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    async def _send_external_alerts(self, alert: Alert):
        """Send alerts to external services (email, Slack)"""
        # Placeholder for email/Slack integration
        if self.email_enabled:
            logger.info("Email alerts enabled but not implemented yet")

        if self.slack_enabled:
            logger.info("Slack alerts enabled but not implemented yet")

    def check_health(self) -> Dict[str, Any]:
        """Check alert system health and return status"""
        recent_critical = [a for a in self.error_queue[-50:] if a.level == AlertLevel.CRITICAL]
        recent_errors = [a for a in self.error_queue[-50:] if a.level in (AlertLevel.CRITICAL, AlertLevel.ERROR)]

        return {
            'alert_queue_size': len(self.error_queue),
            'recent_critical_alerts': len(recent_critical),
            'recent_error_alerts': len(recent_errors),
            'orchestrator_connected': self.orchestrator is not None,
            'discord_enabled': bool(self.orchestrator),
            'email_enabled': self.email_enabled,
            'slack_enabled': self.slack_enabled
        }

    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts"""
        return self.error_queue[-limit:]

    def clear_alerts(self):
        """Clear all alerts from queue"""
        self.error_queue.clear()
        logger.info("Alert queue cleared")


# Global instance
alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    """Get the global AlertManager instance"""
    return alert_manager