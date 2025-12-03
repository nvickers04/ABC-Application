#!/usr/bin/env python3
"""
Component Health Monitor - Comprehensive health checking for all system components.

Provides detailed health checks for agents, external services, and internal components
with automatic recovery mechanisms and monitoring capabilities.
"""

import asyncio
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of components that can be monitored"""
    AGENT = "agent"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL_SERVICE = "internal_service"
    DATABASE = "database"
    API = "api"


@dataclass
class ComponentHealth:
    """Health information for a single component"""
    name: str
    component_type: ComponentType
    status: ComponentStatus
    last_check: datetime
    response_time: float = 0.0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    last_recovery_attempt: Optional[datetime] = None


@dataclass
class HealthCheckResult:
    """Result of a health check operation"""
    component_name: str
    success: bool
    response_time: float
    status: ComponentStatus
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class ComponentHealthMonitor:
    """
    Comprehensive health monitor for all system components.

    Monitors agents, external services, databases, and APIs with automatic
    recovery mechanisms and detailed health reporting.
    """

    def __init__(self, check_interval: int = 60):  # 1 minute default
        self.check_interval = check_interval
        self.components: Dict[str, ComponentHealth] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alert_manager = None

        # Component check functions
        self.check_functions: Dict[str, Callable[[], HealthCheckResult]] = {
            'execution_agent': self._check_execution_agent,
            'live_workflow_orchestrator': self._check_live_workflow_orchestrator,
            'redis': self._check_redis,
            'tigerbeetle': self._check_tigerbeetle,
            'ibkr_bridge': self._check_ibkr_bridge,
            'alert_manager': self._check_alert_manager,
            'api_health_monitor': self._check_api_health_monitor,
            'discord_bot': self._check_discord_bot,
        }

        # Recovery functions
        self.recovery_functions: Dict[str, Callable[[], bool]] = {
            'redis': self._recover_redis,
            'tigerbeetle': self._recover_tigerbeetle,
            'ibkr_bridge': self._recover_ibkr_bridge,
            'discord_bot': self._recover_discord_bot,
        }

        # Initialize component tracking
        self._initialize_components()

    def _initialize_components(self):
        """Initialize tracking for all known components"""
        components_config = {
            'execution_agent': ComponentType.AGENT,
            'live_workflow_orchestrator': ComponentType.AGENT,
            'redis': ComponentType.DATABASE,
            'tigerbeetle': ComponentType.DATABASE,
            'ibkr_bridge': ComponentType.EXTERNAL_SERVICE,
            'alert_manager': ComponentType.INTERNAL_SERVICE,
            'api_health_monitor': ComponentType.INTERNAL_SERVICE,
            'discord_bot': ComponentType.EXTERNAL_SERVICE,
        }

        for name, comp_type in components_config.items():
            self.components[name] = ComponentHealth(
                name=name,
                component_type=comp_type,
                status=ComponentStatus.UNKNOWN,
                last_check=datetime.now()
            )

    def set_alert_manager(self, alert_manager):
        """Set alert manager for health notifications"""
        self.alert_manager = alert_manager

    def start_monitoring(self):
        """Start the background monitoring thread"""
        if self.monitoring_active:
            logger.warning("Component health monitoring is already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Component health monitoring started with {self.check_interval}s interval")

    def stop_monitoring(self):
        """Stop the background monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Component health monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self.perform_health_checks()
                self._attempt_recoveries()
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

            time.sleep(self.check_interval)

    def perform_health_checks(self) -> Dict[str, ComponentHealth]:
        """Perform health checks on all components"""
        results = {}

        for component_name, check_func in self.check_functions.items():
            try:
                start_time = time.time()
                result = check_func()
                response_time = time.time() - start_time

                # Update component health
                component = self.components[component_name]
                component.last_check = datetime.now()
                component.response_time = response_time
                component.status = result.status
                component.error_message = result.error_message
                component.metrics.update(result.metrics)

                results[component_name] = component

                # Send alerts for status changes
                self._handle_status_change(component, result)

            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                component = self.components[component_name]
                component.status = ComponentStatus.UNHEALTHY
                component.error_message = str(e)
                component.last_check = datetime.now()
                results[component_name] = component

        return results

    def _handle_status_change(self, component: ComponentHealth, result: HealthCheckResult):
        """Handle component status changes and send alerts"""
        if not self.alert_manager:
            return

        # Alert on status changes to unhealthy or degraded
        if result.status in [ComponentStatus.UNHEALTHY, ComponentStatus.DEGRADED]:
            asyncio.create_task(
                self.alert_manager.send_alert(
                    level="WARNING" if result.status == ComponentStatus.DEGRADED else "ERROR",
                    component=component.name,
                    message=f"Component {component.name} is {result.status.value}",
                    context={
                        "component_type": component.component_type.value,
                        "response_time": result.response_time,
                        "error_message": result.error_message,
                        "metrics": result.metrics
                    }
                )
            )

    def _attempt_recoveries(self):
        """Attempt recovery for unhealthy components"""
        for component_name, component in self.components.items():
            if component.status in [ComponentStatus.UNHEALTHY, ComponentStatus.DEGRADED]:
                # Check if we should attempt recovery (not too frequent)
                if (component.last_recovery_attempt is None or
                    datetime.now() - component.last_recovery_attempt > timedelta(minutes=5)):

                    recovery_func = self.recovery_functions.get(component_name)
                    if recovery_func:
                        try:
                            component.recovery_attempts += 1
                            component.last_recovery_attempt = datetime.now()

                            if recovery_func():
                                logger.info(f"Successfully recovered component {component_name}")
                                component.status = ComponentStatus.HEALTHY
                                component.error_message = None
                            else:
                                logger.warning(f"Recovery attempt failed for {component_name}")

                        except Exception as e:
                            logger.error(f"Recovery failed for {component_name}: {e}")

    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health information for a specific component"""
        return self.components.get(component_name)

    def get_all_component_health(self) -> Dict[str, ComponentHealth]:
        """Get health information for all components"""
        return self.components.copy()

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        total_components = len(self.components)
        healthy_count = sum(1 for c in self.components.values() if c.status == ComponentStatus.HEALTHY)
        degraded_count = sum(1 for c in self.components.values() if c.status == ComponentStatus.DEGRADED)
        unhealthy_count = sum(1 for c in self.components.values() if c.status == ComponentStatus.UNHEALTHY)

        # Calculate overall status
        if unhealthy_count > 0:
            overall_status = ComponentStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = ComponentStatus.DEGRADED
        elif healthy_count == total_components:
            overall_status = ComponentStatus.HEALTHY
        else:
            overall_status = ComponentStatus.UNKNOWN

        return {
            'overall_status': overall_status.value,
            'total_components': total_components,
            'healthy_components': healthy_count,
            'degraded_components': degraded_count,
            'unhealthy_components': unhealthy_count,
            'component_details': {
                name: {
                    'status': comp.status.value,
                    'last_check': comp.last_check.isoformat(),
                    'response_time': comp.response_time,
                    'error_message': comp.error_message,
                    'recovery_attempts': comp.recovery_attempts
                }
                for name, comp in self.components.items()
            },
            'timestamp': datetime.now().isoformat()
        }

    # Component-specific health check implementations

    def _check_execution_agent(self) -> HealthCheckResult:
        """Check ExecutionAgent health"""
        try:
            from src.agents.execution import ExecutionAgent

            # Try to create agent instance (lightweight check)
            agent = ExecutionAgent()
            metrics = {
                'memory_initialized': agent.memory is not None,
                'scheduler_available': hasattr(agent, 'scheduler'),
                'tigerbeetle_connected': agent.tb_client is not None
            }

            return HealthCheckResult(
                component_name='execution_agent',
                success=True,
                response_time=0.0,
                status=ComponentStatus.HEALTHY,
                metrics=metrics
            )

        except Exception as e:
            return HealthCheckResult(
                component_name='execution_agent',
                success=False,
                response_time=0.0,
                status=ComponentStatus.UNHEALTHY,
                error_message=str(e)
            )

    def _check_live_workflow_orchestrator(self) -> HealthCheckResult:
        """Check LiveWorkflowOrchestrator health"""
        try:
            from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator

            # Try to create orchestrator instance
            orchestrator = LiveWorkflowOrchestrator()
            metrics = {
                'alert_manager_connected': orchestrator.alert_manager is not None,
                'discord_ready': getattr(orchestrator, 'discord_ready', False),
                'scheduler_active': hasattr(orchestrator, 'scheduler') and orchestrator.scheduler is not None
            }

            return HealthCheckResult(
                component_name='live_workflow_orchestrator',
                success=True,
                response_time=0.0,
                status=ComponentStatus.HEALTHY,
                metrics=metrics
            )

        except Exception as e:
            return HealthCheckResult(
                component_name='live_workflow_orchestrator',
                success=False,
                response_time=0.0,
                status=ComponentStatus.UNHEALTHY,
                error_message=str(e)
            )

    def _check_redis(self) -> HealthCheckResult:
        """Check Redis health"""
        try:
            from src.utils.redis_cache import RedisCacheManager
            cache = RedisCacheManager()

            # Perform actual health check
            health_result = cache.health_check()

            status = ComponentStatus.HEALTHY if health_result.get('status') == 'connected' else ComponentStatus.UNHEALTHY

            return HealthCheckResult(
                component_name='redis',
                success=status == ComponentStatus.HEALTHY,
                response_time=health_result.get('response_time', 0.0),
                status=status,
                metrics=health_result
            )

        except Exception as e:
            return HealthCheckResult(
                component_name='redis',
                success=False,
                response_time=0.0,
                status=ComponentStatus.UNHEALTHY,
                error_message=str(e)
            )

    def _check_tigerbeetle(self) -> HealthCheckResult:
        """Check TigerBeetle health"""
        try:
            # Try to import and connect to TigerBeetle
            import tigerbeetle as tb

            # Attempt connection (this is a basic connectivity check)
            try:
                client = tb.ClientSync(cluster_id=0, replica_addresses="127.0.0.1:3000")
                client.close()  # Close immediately after test

                return HealthCheckResult(
                    component_name='tigerbeetle',
                    success=True,
                    response_time=0.0,
                    status=ComponentStatus.HEALTHY,
                    metrics={'connection_test': 'successful'}
                )
            except Exception as conn_e:
                return HealthCheckResult(
                    component_name='tigerbeetle',
                    success=False,
                    response_time=0.0,
                    status=ComponentStatus.UNHEALTHY,
                    error_message=f"Connection failed: {conn_e}"
                )

        except ImportError:
            return HealthCheckResult(
                component_name='tigerbeetle',
                success=False,
                response_time=0.0,
                status=ComponentStatus.DEGRADED,
                error_message="TigerBeetle library not available"
            )

    def _check_ibkr_bridge(self) -> HealthCheckResult:
        """Check IBKR bridge health"""
        try:
            from src.integrations.nautilus_ibkr_bridge import get_nautilus_ibkr_bridge

            # Try to get bridge instance
            bridge = get_nautilus_ibkr_bridge()
            metrics = {
                'bridge_available': bridge is not None,
                'connection_status': getattr(bridge, 'is_connected', lambda: False)() if bridge else False
            }

            status = ComponentStatus.HEALTHY if metrics['connection_status'] else ComponentStatus.DEGRADED

            return HealthCheckResult(
                component_name='ibkr_bridge',
                success=status == ComponentStatus.HEALTHY,
                response_time=0.0,
                status=status,
                metrics=metrics
            )

        except Exception as e:
            return HealthCheckResult(
                component_name='ibkr_bridge',
                success=False,
                response_time=0.0,
                status=ComponentStatus.UNHEALTHY,
                error_message=str(e)
            )

    def _check_alert_manager(self) -> HealthCheckResult:
        """Check AlertManager health"""
        try:
            from src.utils.alert_manager import get_alert_manager

            alert_manager = get_alert_manager()
            queue_size = len(alert_manager.error_queue)

            # Alert manager is healthy if it exists and queue is manageable
            status = ComponentStatus.HEALTHY if queue_size < 1000 else ComponentStatus.DEGRADED

            return HealthCheckResult(
                component_name='alert_manager',
                success=True,
                response_time=0.0,
                status=status,
                metrics={'queue_size': queue_size}
            )

        except Exception as e:
            return HealthCheckResult(
                component_name='alert_manager',
                success=False,
                response_time=0.0,
                status=ComponentStatus.UNHEALTHY,
                error_message=str(e)
            )

    def _check_api_health_monitor(self) -> HealthCheckResult:
        """Check API Health Monitor health"""
        try:
            from src.utils.api_health_monitor import get_api_health_summary

            summary = get_api_health_summary()
            metrics = {
                'apis_monitored': len(summary.get('api_status', {})),
                'overall_status': summary.get('overall_status', 'unknown')
            }

            # Consider healthy if monitoring is working
            status = ComponentStatus.HEALTHY

            return HealthCheckResult(
                component_name='api_health_monitor',
                success=True,
                response_time=0.0,
                status=status,
                metrics=metrics
            )

        except Exception as e:
            return HealthCheckResult(
                component_name='api_health_monitor',
                success=False,
                response_time=0.0,
                status=ComponentStatus.UNHEALTHY,
                error_message=str(e)
            )

    def _check_discord_bot(self) -> HealthCheckResult:
        """Check Discord bot health"""
        try:
            # Check if Discord bot interface exists and is connected
            from src.integrations.discord.discord_bot_interface import DiscordBotInterface

            active_interfaces = DiscordBotInterface._active_interfaces
            connected_interfaces = [iface for iface in active_interfaces if hasattr(iface, 'is_ready') and iface.is_ready()]

            metrics = {
                'active_interfaces': len(active_interfaces),
                'connected_interfaces': len(connected_interfaces)
            }

            status = ComponentStatus.HEALTHY if connected_interfaces else ComponentStatus.DEGRADED

            return HealthCheckResult(
                component_name='discord_bot',
                success=status == ComponentStatus.HEALTHY,
                response_time=0.0,
                status=status,
                metrics=metrics
            )

        except Exception as e:
            return HealthCheckResult(
                component_name='discord_bot',
                success=False,
                response_time=0.0,
                status=ComponentStatus.UNHEALTHY,
                error_message=str(e)
            )

    # Recovery function implementations

    def _recover_redis(self) -> bool:
        """Attempt to recover Redis connection"""
        try:
            from src.utils.redis_cache import RedisCacheManager
            cache = RedisCacheManager()
            # Attempt reconnection by checking health
            health = cache.health_check()
            return health.get('status') == 'connected'
        except Exception as e:
            logger.error(f"Redis recovery failed: {e}")
            return False

    def _recover_tigerbeetle(self) -> bool:
        """Attempt to recover TigerBeetle connection"""
        try:
            # This would typically involve restarting TigerBeetle service
            # For now, just log that recovery was attempted
            logger.info("TigerBeetle recovery attempted - manual intervention may be required")
            return False  # Manual recovery typically needed
        except Exception as e:
            logger.error(f"TigerBeetle recovery failed: {e}")
            return False

    def _recover_ibkr_bridge(self) -> bool:
        """Attempt to recover IBKR bridge connection"""
        try:
            # For now, just log that recovery was attempted
            # In a real implementation, this would attempt to reinitialize the bridge
            logger.info("IBKR bridge recovery attempted - manual intervention may be required")
            return False  # Manual recovery typically needed
        except Exception as e:
            logger.error(f"IBKR bridge recovery failed: {e}")
            return False

    def _recover_discord_bot(self) -> bool:
        """Attempt to recover Discord bot connection"""
        try:
            # For now, just log that recovery was attempted
            # In a real implementation, this would attempt to restart the Discord bot
            logger.info("Discord bot recovery attempted - manual intervention may be required")
            return False  # Manual recovery typically needed
        except Exception as e:
            logger.error(f"Discord bot recovery failed: {e}")
            return False


# Global instance - now initialized in alert_manager.py
_component_monitor: Optional[ComponentHealthMonitor] = None


def get_component_health_monitor() -> ComponentHealthMonitor:
    """Get the global component health monitor instance"""
    global _component_monitor
    if _component_monitor is None:
        _component_monitor = ComponentHealthMonitor()
        # Try to connect to alert manager if available
        try:
            from .alert_manager import get_alert_manager
            alert_manager = get_alert_manager()
            _component_monitor.set_alert_manager(alert_manager)
        except Exception as e:
            logger.warning(f"Could not connect ComponentHealthMonitor to AlertManager: {e}")
    return _component_monitor


def start_component_health_monitoring(check_interval: int = 60):
    """Start component health monitoring"""
    monitor = get_component_health_monitor()
    monitor.start_monitoring()
    logger.info("Component health monitoring started")


def stop_component_health_monitoring():
    """Stop component health monitoring"""
    global _component_monitor
    if _component_monitor:
        _component_monitor.stop_monitoring()
        _component_monitor = None
    logger.info("Component health monitoring stopped")


def get_component_health_summary() -> Dict[str, Any]:
    """Get comprehensive component health summary"""
    monitor = get_component_health_monitor()
    return monitor.get_system_health_summary()