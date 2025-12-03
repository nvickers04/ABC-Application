# src/utils/alert_manager.py
"""
AlertManager - Centralized alerting system for the ABC-Application

Provides unified error handling, logging, and notification across all components.
Supports Discord notifications, health check integration, and fail-fast error handling.
"""

import logging
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable
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


class ConnectionError(Exception):
    """Exception raised for connection failures to external services"""

    def __init__(self, message: str, service: str, retryable: bool = True):
        super().__init__(message)
        self.service = service
        self.retryable = retryable


class AuthenticationError(Exception):
    """Exception raised for authentication failures"""

    def __init__(self, message: str, service: str, credentials_expired: bool = False):
        super().__init__(message)
        self.service = service
        self.credentials_expired = credentials_expired


class RateLimitError(Exception):
    """Exception raised when API rate limits are exceeded"""

    def __init__(self, message: str, service: str, retry_after_seconds: int = None):
        super().__init__(message)
        self.service = service
        self.retry_after_seconds = retry_after_seconds


class DataQualityError(Exception):
    """Exception raised when data quality validation fails"""

    def __init__(self, message: str, data_source: str, validation_errors: List[str] = None):
        super().__init__(message)
        self.data_source = data_source
        self.validation_errors = validation_errors or []


class ValidationGateError(Exception):
    """Exception raised when validation gates fail"""

    def __init__(self, message: str, gate_name: str, validation_results: Dict[str, Any] = None):
        super().__init__(message)
        self.gate_name = gate_name
        self.validation_results = validation_results or {}


class ValidationGate:
    """Validation gate for checking conditions before operations"""

    def __init__(self, name: str, alert_manager: 'AlertManager'):
        self.name = name
        self.alert_manager = alert_manager

    async def validate(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate conditions and return results

        Args:
            conditions: Dictionary of validation conditions

        Returns:
            Dict with validation results
        """
        results = {"passed": True, "checks": {}}

        for check_name, check_func in conditions.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    check_result = await check_func()
                else:
                    check_result = check_func()

                results["checks"][check_name] = {"passed": True, "result": check_result}
            except Exception as e:
                results["passed"] = False
                results["checks"][check_name] = {"passed": False, "error": str(e)}

                # Alert on validation failure
                await self.alert_manager.warning(
                    f"Validation gate '{self.name}' check '{check_name}' failed",
                    {"gate": self.name, "check": check_name, "error": str(e)},
                    "validation_gate"
                )

        return results

    async def enforce(self, conditions: Dict[str, Any], fail_fast: bool = True) -> None:
        """
        Enforce validation conditions, raising exception if any fail

        Args:
            conditions: Dictionary of validation conditions
            fail_fast: Whether to raise exception immediately on first failure
        """
        results = await self.validate(conditions)

        if not results["passed"]:
            failed_checks = [
                check_name for check_name, check_result in results["checks"].items()
                if not check_result["passed"]
            ]

            raise ValidationGateError(
                f"Validation gate '{self.name}' failed: {', '.join(failed_checks)}",
                self.name,
                results
            )


class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""

    def __init__(self, service_name: str, failure_threshold: int = 5,
                 recovery_timeout: int = 60, alert_manager: Optional['AlertManager'] = None):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.alert_manager = alert_manager

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return False
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout

    async def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection

        Args:
            func: Function to call
            *args, **kwargs: Arguments for the function

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise ConnectionError(
                    f"Circuit breaker OPEN for {self.service_name}",
                    self.service_name,
                    retryable=True
                )

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - reset failure count and close circuit
            self.failure_count = 0
            self.state = "CLOSED"
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                if self.alert_manager:
                    await self.alert_manager.error(
                        Exception(f"Circuit breaker opened for {self.service_name} after {self.failure_count} failures"),
                        {"service": self.service_name, "failures": self.failure_count},
                        "circuit_breaker"
                    )

            raise e

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "service": self.service_name,
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


async def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0,
                           max_delay: float = 60.0, backoff_factor: float = 2.0,
                           alert_manager: Optional['AlertManager'] = None,
                           service_name: str = "unknown"):
    """
    Retry function with exponential backoff

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Exponential backoff factor
        alert_manager: Optional alert manager for notifications
        service_name: Service name for alerts

    Returns:
        Function result

    Raises:
        Exception: Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)

                if alert_manager:
                    await alert_manager.warning(
                        f"Retry attempt {attempt + 1}/{max_retries + 1} for {service_name}",
                        {"attempt": attempt + 1, "max_retries": max_retries + 1,
                         "delay": delay, "error": str(e)},
                        "retry_mechanism"
                    )

                await asyncio.sleep(delay)
            else:
                # All retries exhausted
                if alert_manager:
                    await alert_manager.error(
                        Exception(f"All {max_retries + 1} attempts failed for {service_name}"),
                        {"attempts": max_retries + 1, "final_error": str(e)},
                        "retry_mechanism"
                    )

    raise last_exception


def graceful_degradation(func):
    """
    Decorator for graceful degradation when services are unavailable

    Args:
        func: Function to decorate

    Returns:
        Decorated function with fallback behavior
    """
    async def wrapper(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except (ConnectionError, AuthenticationError, RateLimitError) as e:
            # Log the degradation but continue with fallback
            logger.warning(f"Service degradation for {func.__name__}: {e}")

            # Return fallback result based on function name
            if "market_data" in func.__name__.lower():
                return {"degraded": True, "error": str(e), "fallback_data": {}}
            elif "analysis" in func.__name__.lower():
                return {"degraded": True, "error": str(e), "fallback_analysis": {}}
            elif "trade" in func.__name__.lower():
                return {"degraded": True, "error": str(e), "simulated": True}
            else:
                return {"degraded": True, "error": str(e)}

    return wrapper


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
    _lock = threading.Lock()

    def __new__(cls) -> 'AlertManager':
        if cls._instance is None:
            with cls._lock:  # Thread-safe singleton creation
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

        # Escalation and routing configuration
        self.escalation_policies = self._load_escalation_policies()
        self.notification_filters = self._load_notification_filters()
        self.escalation_history = []  # Track escalation events

        # Metrics collection
        self.alert_metrics = {
            'total_alerts': 0,
            'alerts_by_level': {level.value: 0 for level in AlertLevel},
            'alerts_by_component': {},
            'alerts_by_hour': {},  # Hourly frequency tracking
            'alert_response_times': [],  # Processing time tracking
            'false_positives': [],  # False positive tracking
            'alert_patterns': {},  # Pattern detection
            'performance_stats': {
                'avg_response_time': 0.0,
                'max_response_time': 0.0,
                'min_response_time': float('inf'),
                'false_positive_rate': 0.0
            }
        }

        logger.info("AlertManager initialized")

    def _load_escalation_policies(self) -> Dict[str, Any]:
        """Load escalation policies from configuration"""
        return {
            'default': {
                'levels': [
                    {
                        'name': 'immediate',
                        'channels': ['discord'],
                        'delay_minutes': 0,
                        'conditions': {'min_level': 'warning'}
                    },
                    {
                        'name': 'escalated',
                        'channels': ['discord', 'email'],
                        'delay_minutes': 5,
                        'conditions': {'min_level': 'error', 'frequency_threshold': 3}
                    },
                    {
                        'name': 'critical',
                        'channels': ['discord', 'email', 'sms'],
                        'delay_minutes': 10,
                        'conditions': {'min_level': 'critical', 'pattern_match': ['security', 'data_breach']}
                    }
                ]
            },
            'business_hours': {
                'active_hours': {'start': '09:00', 'end': '17:00', 'timezone': 'UTC'},
                'levels': [
                    {
                        'name': 'immediate',
                        'channels': ['discord'],
                        'delay_minutes': 0,
                        'conditions': {'min_level': 'warning'}
                    },
                    {
                        'name': 'escalated',
                        'channels': ['discord', 'email'],
                        'delay_minutes': 15,
                        'conditions': {'min_level': 'error', 'frequency_threshold': 5}
                    }
                ]
            }
        }

    def _load_notification_filters(self) -> Dict[str, Any]:
        """Load notification filters and preferences"""
        return {
            'component_filters': {
                'health_monitor': {'min_level': 'warning', 'quiet_hours': True},
                'data_processor': {'min_level': 'error', 'rate_limit': 10},  # Max 10 alerts per hour
                'ibkr_connector': {'min_level': 'info', 'patterns': ['connection_failed']},
                'api_client': {'min_level': 'warning', 'exclude_patterns': ['rate_limited']}
            },
            'user_preferences': {
                'default': {
                    'channels': ['discord'],
                    'quiet_hours': {'start': '22:00', 'end': '08:00', 'timezone': 'UTC'},
                    'alert_types': ['error', 'warning', 'critical']
                }
            },
            'suppression_rules': [
                {
                    'name': 'duplicate_suppression',
                    'condition': {'same_message_within': 300},  # 5 minutes
                    'action': 'suppress'
                },
                {
                    'name': 'maintenance_mode',
                    'condition': {'component': 'health_monitor', 'pattern': 'maintenance'},
                    'action': 'suppress'
                }
            ]
        }

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

    async def send_alert(self, level: AlertLevel, component: str, message: str,
                        context: Optional[Dict[str, Any]] = None):
        """Send an alert with specified level, component, and message"""
        alert = Alert(
            level=level,
            message=message,
            context=context or {},
            component=component
        )
        await self._process_alert(alert)

    async def _process_alert(self, alert: Alert):
        """Process an alert: log, queue, notify, and collect metrics"""
        start_time = datetime.now()

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

        # Apply escalation and routing logic
        await self._apply_escalation_policies(alert)

        # Send notifications based on routing rules
        await self._route_notifications(alert)

        # Collect metrics
        await self._collect_alert_metrics(alert, start_time)

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

    async def _collect_alert_metrics(self, alert: Alert, start_time: datetime):
        """Collect metrics for alert processing"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000  # ms

        # Update total count
        self.alert_metrics['total_alerts'] += 1

        # Update level counts
        self.alert_metrics['alerts_by_level'][alert.level.value] += 1

        # Update component counts
        if alert.component not in self.alert_metrics['alerts_by_component']:
            self.alert_metrics['alerts_by_component'][alert.component] = 0
        self.alert_metrics['alerts_by_component'][alert.component] += 1

        # Update hourly frequency
        hour_key = alert.timestamp.strftime('%Y-%m-%d %H:00')
        if hour_key not in self.alert_metrics['alerts_by_hour']:
            self.alert_metrics['alerts_by_hour'][hour_key] = 0
        self.alert_metrics['alerts_by_hour'][hour_key] += 1

        # Track response times
        self.alert_metrics['alert_response_times'].append(processing_time)
        if len(self.alert_metrics['alert_response_times']) > 1000:  # Keep last 1000
            self.alert_metrics['alert_response_times'].pop(0)

        # Update performance stats
        self._update_performance_stats()

        # Detect alert patterns
        self._detect_alert_patterns(alert)

    def _update_performance_stats(self):
        """Update performance statistics"""
        response_times = self.alert_metrics['alert_response_times']
        if response_times:
            self.alert_metrics['performance_stats']['avg_response_time'] = sum(response_times) / len(response_times)
            self.alert_metrics['performance_stats']['max_response_time'] = max(response_times)
            self.alert_metrics['performance_stats']['min_response_time'] = min(response_times)

        # Calculate false positive rate (alerts marked as false positives / total alerts)
        total_false_positives = len(self.alert_metrics['false_positives'])
        if self.alert_metrics['total_alerts'] > 0:
            self.alert_metrics['performance_stats']['false_positive_rate'] = total_false_positives / self.alert_metrics['total_alerts']

    def _detect_alert_patterns(self, alert: Alert):
        """Detect patterns in alerts for analysis"""
        # Simple pattern detection based on message content
        message_lower = alert.message.lower()

        patterns = {
            'connection_failed': ['connection', 'connect', 'timeout', 'refused'],
            'authentication_failed': ['auth', 'login', 'credential', 'token'],
            'rate_limited': ['rate limit', 'too many requests', 'throttle'],
            'data_quality': ['invalid data', 'missing', 'corrupt', 'quality'],
            'system_resource': ['memory', 'cpu', 'disk', 'resource']
        }

        for pattern_name, keywords in patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                if pattern_name not in self.alert_metrics['alert_patterns']:
                    self.alert_metrics['alert_patterns'][pattern_name] = 0
                self.alert_metrics['alert_patterns'][pattern_name] += 1
                break

    def mark_false_positive(self, alert_index: int = -1):
        """Mark an alert as a false positive for metrics tracking"""
        if self.error_queue and 0 <= alert_index < len(self.error_queue):
            alert = self.error_queue[alert_index]
            self.alert_metrics['false_positives'].append({
                'alert': alert,
                'marked_at': datetime.now()
            })
            self._update_performance_stats()
            logger.info(f"Alert marked as false positive: {alert.message}")

    def get_alert_metrics(self) -> Dict[str, Any]:
        """Get comprehensive alert metrics"""
        return {
            'summary': {
                'total_alerts': self.alert_metrics['total_alerts'],
                'alerts_by_level': self.alert_metrics['alerts_by_level'],
                'top_components': dict(sorted(
                    self.alert_metrics['alerts_by_component'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                'alert_patterns': self.alert_metrics['alert_patterns']
            },
            'performance': self.alert_metrics['performance_stats'],
            'recent_activity': {
                'last_24h_alerts': len([a for a in self.error_queue[-100:]
                                      if (datetime.now() - a.timestamp).total_seconds() < 86400]),
                'hourly_frequency': dict(list(self.alert_metrics['alerts_by_hour'].items())[-24:])
            },
            'quality': {
                'false_positives_count': len(self.alert_metrics['false_positives']),
                'false_positive_rate': self.alert_metrics['performance_stats']['false_positive_rate']
            }
        }

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data for alert statistics"""
        metrics = self.get_alert_metrics()
        health = self.check_health()

        return {
            'timestamp': datetime.now().isoformat(),
            'alert_system_health': health,
            'metrics_summary': metrics['summary'],
            'performance_indicators': {
                'avg_response_time_ms': round(metrics['performance']['avg_response_time'], 2),
                'max_response_time_ms': round(metrics['performance']['max_response_time'], 2),
                'false_positive_rate_percent': round(metrics['performance']['false_positive_rate'] * 100, 2),
                'alerts_per_hour': self._calculate_alerts_per_hour()
            },
            'alert_distribution': {
                'by_level': metrics['summary']['alerts_by_level'],
                'by_component': metrics['summary']['top_components'],
                'by_pattern': metrics['summary']['alert_patterns']
            },
            'recent_alerts': [
                {
                    'level': alert.level.value,
                    'component': alert.component,
                    'message': alert.message[:100] + '...' if len(alert.message) > 100 else alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'age_minutes': round((datetime.now() - alert.timestamp).total_seconds() / 60, 1)
                }
                for alert in self.get_recent_alerts(10)
            ],
            'recommendations': self._generate_monitoring_recommendations(metrics, health)
        }

    def _calculate_alerts_per_hour(self) -> float:
        """Calculate average alerts per hour over the last 24 hours"""
        recent_alerts = [a for a in self.error_queue[-200:]
                        if (datetime.now() - a.timestamp).total_seconds() < 86400]
        if not recent_alerts:
            return 0.0

        hours_span = (datetime.now() - recent_alerts[0].timestamp).total_seconds() / 3600
        return len(recent_alerts) / max(hours_span, 1)

    def _generate_monitoring_recommendations(self, metrics: Dict, health: Dict) -> List[str]:
        """Generate monitoring recommendations based on metrics and health"""
        recommendations = []

        # Check alert frequency
        alerts_per_hour = self._calculate_alerts_per_hour()
        if alerts_per_hour > 10:
            recommendations.append(f"⚠️ High alert frequency: {alerts_per_hour:.1f} alerts/hour - investigate root causes")
        elif alerts_per_hour > 5:
            recommendations.append(f"ℹ️ Moderate alert frequency: {alerts_per_hour:.1f} alerts/hour - monitor closely")

        # Check false positive rate
        fp_rate = metrics['performance']['false_positive_rate']
        if fp_rate > 0.3:
            recommendations.append(f"⚠️ High false positive rate: {fp_rate:.1%} - review alert thresholds")
        elif fp_rate > 0.1:
            recommendations.append(f"ℹ️ Moderate false positive rate: {fp_rate:.1%} - consider tuning")

        # Check response times
        avg_response = metrics['performance']['avg_response_time']
        if avg_response > 1000:  # > 1 second
            recommendations.append(f"⚠️ Slow alert processing: {avg_response:.0f}ms average - optimize delivery")
        elif avg_response > 500:  # > 0.5 seconds
            recommendations.append(f"ℹ️ Moderate response time: {avg_response:.0f}ms - monitor performance")

        # Check queue health
        if health['alert_queue_size'] > 500:
            recommendations.append(f"⚠️ Large alert queue: {health['alert_queue_size']} alerts - review processing")

        # Check error patterns
        error_patterns = metrics['summary']['alert_patterns']
        if error_patterns.get('connection_failed', 0) > 10:
            recommendations.append("🔄 Frequent connection failures detected - check network stability")
        if error_patterns.get('rate_limited', 0) > 5:
            recommendations.append("⏱️ Frequent rate limiting - review API usage patterns")

        if not recommendations:
            recommendations.append("✅ Alert system operating normally")

        return recommendations

    async def _apply_escalation_policies(self, alert: Alert):
        """Apply escalation policies based on alert characteristics"""
        current_time = datetime.now()

        # Check if alert should be escalated based on policies
        escalation_needed = self._check_escalation_conditions(alert)

        if escalation_needed:
            escalation_event = {
                'alert': alert,
                'escalation_level': escalation_needed['level'],
                'escalation_time': current_time,
                'channels': escalation_needed['channels'],
                'reason': escalation_needed['reason']
            }
            self.escalation_history.append(escalation_event)

            # Keep only last 100 escalation events
            if len(self.escalation_history) > 100:
                self.escalation_history.pop(0)

            logger.info(f"Alert escalated: {alert.message} -> {escalation_needed['level']}")

    def _check_escalation_conditions(self, alert: Alert) -> Optional[Dict[str, Any]]:
        """Check if alert meets escalation conditions"""
        policy = self.escalation_policies.get('default', {})  # Use default policy

        for level in policy.get('levels', []):
            conditions = level.get('conditions', {})

            # Check minimum alert level
            if 'min_level' in conditions:
                min_level_str = conditions['min_level']
                current_level_str = alert.level.value
                level_hierarchy = {'debug': 0, 'info': 1, 'warning': 2, 'error': 3, 'critical': 4}

                if level_hierarchy.get(current_level_str, 0) < level_hierarchy.get(min_level_str, 0):
                    continue

            # Check frequency threshold
            if 'frequency_threshold' in conditions:
                recent_alerts = [a for a in self.error_queue[-50:]
                               if a.component == alert.component and
                               (datetime.now() - a.timestamp).total_seconds() < 3600]  # Last hour
                if len(recent_alerts) < conditions['frequency_threshold']:
                    continue

            # Check pattern matching
            if 'pattern_match' in conditions:
                message_lower = alert.message.lower()
                if not any(pattern.lower() in message_lower for pattern in conditions['pattern_match']):
                    continue

            # If all conditions met, return escalation info
            return {
                'level': level['name'],
                'channels': level['channels'],
                'delay_minutes': level['delay_minutes'],
                'reason': f"Met conditions: {conditions}"
            }

        return None

    async def _route_notifications(self, alert: Alert):
        """Route notifications based on filters and escalation policies"""
        # Check if alert should be filtered/suppressed
        if self._should_suppress_alert(alert):
            logger.debug(f"Alert suppressed: {alert.message}")
            return

        # Determine target channels based on escalation and filters
        target_channels = self._determine_target_channels(alert)

        # Send to each target channel
        for channel in target_channels:
            if channel == 'discord':
                await self._send_discord_alert(alert)
            elif channel == 'email':
                await self._send_email_alert(alert)
            elif channel == 'sms':
                await self._send_sms_alert(alert)
            elif channel == 'slack':
                await self._send_slack_alert(alert)

    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed based on filters"""
        # Check suppression rules
        for rule in self.notification_filters.get('suppression_rules', []):
            condition = rule.get('condition', {})

            # Check time-based suppression (same message within time window)
            if 'same_message_within' in condition:
                time_window = condition['same_message_within']
                recent_similar = [a for a in self.error_queue[-20:]
                                if a.message == alert.message and
                                (datetime.now() - a.timestamp).total_seconds() < time_window]
                if recent_similar:
                    return True

            # Check component and pattern based suppression
            if 'component' in condition and alert.component == condition['component']:
                if 'pattern' in condition:
                    if condition['pattern'].lower() in alert.message.lower():
                        return True

        # Check component-specific filters
        component_filters = self.notification_filters.get('component_filters', {}).get(alert.component, {})

        # Check minimum level filter
        if 'min_level' in component_filters:
            min_level_value = AlertLevel[component_filters['min_level']].value
            current_level_value = alert.level.value
            level_hierarchy = {'debug': 0, 'info': 1, 'warning': 2, 'error': 3, 'critical': 4}

            if level_hierarchy.get(current_level_value, 0) < level_hierarchy.get(min_level_value, 0):
                return True

        # Check rate limiting
        if 'rate_limit' in component_filters:
            recent_component_alerts = [a for a in self.error_queue[-50:]
                                     if a.component == alert.component and
                                     (datetime.now() - a.timestamp).total_seconds() < 3600]  # Last hour
            if len(recent_component_alerts) >= component_filters['rate_limit']:
                return True

        return False

    def _determine_target_channels(self, alert: Alert) -> List[str]:
        """Determine which channels should receive the alert"""
        channels = set()

        # Check escalation history for this alert
        escalation_info = None
        for escalation in self.escalation_history[-10:]:  # Check recent escalations
            if escalation['alert'].message == alert.message and \
               escalation['alert'].component == alert.component and \
               (datetime.now() - escalation['escalation_time']).total_seconds() < 300:  # Within 5 minutes
                escalation_info = escalation
                break

        if escalation_info:
            # Use escalated channels
            channels.update(escalation_info['channels'])
        else:
            # Use default channels based on alert level
            if alert.level in (AlertLevel.CRITICAL, AlertLevel.ERROR):
                channels.add('discord')
                if self.email_enabled:
                    channels.add('email')
            elif alert.level == AlertLevel.WARNING:
                channels.add('discord')
            else:
                channels.add('discord')  # Info and debug still go to Discord

        return list(channels)

    async def _send_email_alert(self, alert: Alert):
        """Send alert via email (placeholder)"""
        logger.info(f"Email alert: {alert.level.value.upper()} - {alert.message}")
        # TODO: Implement actual email sending
        pass

    async def _send_sms_alert(self, alert: Alert):
        """Send alert via SMS (placeholder)"""
        logger.info(f"SMS alert: {alert.level.value.upper()} - {alert.message}")
        # TODO: Implement actual SMS sending
        pass

    async def _send_slack_alert(self, alert: Alert):
        """Send alert via Slack (placeholder)"""
        logger.info(f"Slack alert: {alert.level.value.upper()} - {alert.message}")
        # TODO: Implement actual Slack sending
        pass

    def get_escalation_status(self) -> Dict[str, Any]:
        """Get current escalation status and history"""
        return {
            'active_escalations': len([e for e in self.escalation_history[-20:]
                                     if (datetime.now() - e['escalation_time']).total_seconds() < 3600]),  # Last hour
            'recent_escalations': [
                {
                    'level': e['escalation_level'],
                    'component': e['alert'].component,
                    'message': e['alert'].message[:50] + '...' if len(e['alert'].message) > 50 else e['alert'].message,
                    'channels': e['channels'],
                    'time': e['escalation_time'].isoformat()
                }
                for e in self.escalation_history[-10:]
            ],
            'escalation_policies': self.escalation_policies,
            'notification_filters': self.notification_filters
        }


# Exception Classes for Specific Error Handling
class ConnectionError(Exception):
    """Raised when connection to external services fails"""
    pass

class AuthenticationError(Exception):
    """Raised when authentication with external services fails"""
    pass

class RateLimitError(Exception):
    """Raised when API rate limits are exceeded"""
    pass

class DataQualityError(Exception):
    """Raised when data quality validation fails"""
    pass

class ValidationGateError(Exception):
    """Raised when validation gate enforcement fails"""
    pass


class ValidationGate:
    """Pre-operation validation gate for critical operations"""

    def __init__(self, name: str, alert_manager: Optional[AlertManager] = None):
        self.name = name
        self.alert_manager = alert_manager or get_alert_manager()
        self.validation_results: List[Dict[str, Any]] = []

    def validate(self, checks: Dict[str, Callable[[], bool]]) -> Dict[str, Any]:
        """Run validation checks and return results"""
        results = {}
        all_passed = True

        for check_name, check_func in checks.items():
            try:
                passed = check_func()
                results[check_name] = {
                    'passed': passed,
                    'timestamp': datetime.now(),
                    'error': None
                }
                if not passed:
                    all_passed = False
            except Exception as e:
                results[check_name] = {
                    'passed': False,
                    'timestamp': datetime.now(),
                    'error': str(e)
                }
                all_passed = False

        self.validation_results.append({
            'gate': self.name,
            'timestamp': datetime.now(),
            'results': results,
            'all_passed': all_passed
        })

        return {
            'gate': self.name,
            'all_passed': all_passed,
            'results': results
        }

    def enforce(self, checks: Dict[str, Callable[[], bool]], raise_on_failure: bool = True) -> bool:
        """Run validation checks and raise ValidationGateError if any fail"""
        validation_result = self.validate(checks)

        if not validation_result['all_passed'] and raise_on_failure:
            failed_checks = [k for k, v in validation_result['results'].items() if not v['passed']]
            error_msg = f"Validation gate '{self.name}' failed: {', '.join(failed_checks)}"

            # Send alert
            if self.alert_manager:
                # Create task to send alert asynchronously (only if event loop exists)
                try:
                    asyncio.create_task(self.alert_manager.send_alert(
                        level=AlertLevel.ERROR,
                        component=f"ValidationGate:{self.name}",
                        message=error_msg,
                        context={'failed_checks': failed_checks}
                    ))
                except RuntimeError:
                    # No event loop, skip async alert (for testing)
                    pass

            raise ValidationGateError(error_msg)

        return validation_result['all_passed']

    def get_recent_results(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent validation results"""
        return self.validation_results[-limit:]


class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60,
                 alert_manager: Optional[AlertManager] = None):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.alert_manager = alert_manager or get_alert_manager()

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise ConnectionError(f"Circuit breaker '{self.name}' is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            if self.alert_manager:
                try:
                    asyncio.create_task(self.alert_manager.send_alert(
                        level=AlertLevel.INFO,
                        component=f"CircuitBreaker:{self.name}",
                        message=f"Circuit breaker reset to CLOSED"
                    ))
                except RuntimeError:
                    # No event loop, skip async alert (for testing)
                    pass

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            if self.alert_manager:
                try:
                    asyncio.create_task(self.alert_manager.send_alert(
                        level=AlertLevel.WARNING,
                        component=f"CircuitBreaker:{self.name}",
                        message=f"Circuit breaker opened after {self.failure_count} failures"
                    ))
                except RuntimeError:
                    # No event loop, skip async alert (for testing)
                    pass

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'name': self.name,
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'next_reset_attempt': (self.last_failure_time + timedelta(seconds=self.recovery_timeout)).isoformat() if self.last_failure_time else None
        }


async def retry_with_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0,
                            max_delay: float = 60.0, backoff_factor: float = 2.0,
                            alert_manager: Optional[AlertManager] = None):
    """Retry function with exponential backoff"""
    alert_mgr = alert_manager or get_alert_manager()

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
                if attempt == max_retries:
                    if alert_mgr:
                        asyncio.create_task(alert_mgr.send_alert(
                            level=AlertLevel.ERROR,
                            component="RetryMechanism",
                            message=f"Function failed after {max_retries + 1} attempts",
                            context={'error': str(e), 'max_retries': max_retries}
                        ))
                    raise

                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                if alert_mgr:
                    asyncio.create_task(alert_mgr.send_alert(
                        level=AlertLevel.WARNING,
                        component="RetryMechanism",
                        message=f"Retry attempt {attempt + 1}/{max_retries + 1} after {delay:.1f}s",
                        context={'error': str(e), 'delay': delay}
                    ))
                await asyncio.sleep(delay)


def graceful_degradation(func: Callable):
    """Decorator for graceful degradation on failures"""
    async def wrapper(*args, **kwargs):
        alert_manager = get_alert_manager()
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Log the error
            logger.error(f"Function {func.__name__} failed, entering graceful degradation: {e}")

            # Send alert
            if alert_manager:
                asyncio.create_task(alert_manager.send_alert(
                    level=AlertLevel.WARNING,
                    component=f"GracefulDegradation:{func.__name__}",
                    message=f"Entering graceful degradation mode",
                    context={'error': str(e), 'function': func.__name__}
                ))

            # Return degraded response or None
            return None

    return wrapper


# Global instances
alert_manager = AlertManager()

# Initialize component health monitor and connect to alert manager
try:
    from .component_health_monitor import get_component_health_monitor
    component_health_monitor = get_component_health_monitor()
    component_health_monitor.set_alert_manager(alert_manager)
    logger.info("ComponentHealthMonitor initialized and connected to AlertManager")
except Exception as e:
    logger.warning(f"Failed to initialize ComponentHealthMonitor: {e}")
    component_health_monitor = None


def get_alert_manager() -> AlertManager:
    """Get the global AlertManager instance"""
    return alert_manager


def get_component_health_monitor():
    """Get the global ComponentHealthMonitor instance"""
    return component_health_monitor