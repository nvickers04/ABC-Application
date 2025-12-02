#!/usr/bin/env python3
"""
Trading Day Monitoring System - Comprehensive Error/Warning Tracking
Monitors all system components during live trading for improvement data collection.
"""

import asyncio
import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('src'))

from src.utils.advanced_memory import get_memory_health_status
from src.utils.api_health_monitor import get_api_health_summary
from src.integrations.nautilus_ibkr_bridge import get_nautilus_ibkr_bridge

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_day_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingDayMonitor:
    """Comprehensive monitoring system for trading day operations."""

    def __init__(self):
        self.start_time = datetime.now()
        self.monitoring_data = {
            'session_start': self.start_time.isoformat(),
            'errors': defaultdict(list),
            'warnings': defaultdict(list),
            'rate_limits': [],
            'api_health': [],
            'memory_health': [],
            'ibkr_status': [],
            'performance_metrics': [],
            'system_events': []
        }

        # Error patterns to track
        self.error_patterns = {
            'mem0_gpt4_mini': 'Mem0 backend gpt-4o-mini model access issues',
            'discord_rate_limit': 'Discord API rate limiting',
            'ibkr_connection': 'IBKR connectivity issues',
            'api_failures': 'External API failures',
            'memory_backend': 'Memory backend failures',
            'llm_timeouts': 'LLM API timeouts',
            'position_data': 'Position data retrieval failures'
        }

    async def log_error(self, error_type: str, message: str, context: Dict[str, Any] = None):
        """Log an error with context for analysis."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': message,
            'context': context or {},
            'pattern': self._classify_error(message)
        }

        self.monitoring_data['errors'][error_type].append(error_entry)
        logger.error(f"[{error_type}] {message}")

        # Immediate alerts for critical errors
        if error_type in ['ibkr_connection', 'api_failures']:
            await self._send_alert(f"🚨 CRITICAL: {error_type} - {message}")

    async def log_warning(self, warning_type: str, message: str, context: Dict[str, Any] = None):
        """Log a warning with context."""
        warning_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': warning_type,
            'message': message,
            'context': context or {},
            'pattern': self._classify_error(message)
        }

        self.monitoring_data['warnings'][warning_type].append(warning_entry)
        logger.warning(f"[{warning_type}] {message}")

    async def log_rate_limit(self, service: str, retry_time: float, endpoint: str = None):
        """Log rate limiting events."""
        rate_limit_entry = {
            'timestamp': datetime.now().isoformat(),
            'service': service,
            'retry_time': retry_time,
            'endpoint': endpoint
        }

        self.monitoring_data['rate_limits'].append(rate_limit_entry)
        logger.warning(f"Rate limited: {service} - retrying in {retry_time}s")

    def _classify_error(self, message: str) -> str:
        """Classify error messages into patterns."""
        message_lower = message.lower()

        if 'gpt-4o-mini' in message_lower and 'mem0' in message_lower:
            return 'mem0_gpt4_mini'
        elif 'rate limited' in message_lower or '429' in message_lower:
            return 'discord_rate_limit'
        elif 'ibkr' in message_lower and ('connection' in message_lower or 'connect' in message_lower):
            return 'ibkr_connection'
        elif 'api' in message_lower and ('fail' in message_lower or 'error' in message_lower):
            return 'api_failures'
        elif 'memory' in message_lower and ('backend' in message_lower or 'store' in message_lower):
            return 'memory_backend'
        elif 'timeout' in message_lower and ('llm' in message_lower or 'api' in message_lower):
            return 'llm_timeouts'
        elif 'position' in message_lower and ('fail' in message_lower or 'mock' in message_lower):
            return 'position_data'

        return 'other'

    async def check_system_health(self):
        """Perform comprehensive system health check."""
        try:
            # API Health
            api_health = get_api_health_summary()
            self.monitoring_data['api_health'].append({
                'timestamp': datetime.now().isoformat(),
                'health': api_health
            })

            # Memory Health
            memory_health = get_memory_health_status()
            self.monitoring_data['memory_health'].append({
                'timestamp': datetime.now().isoformat(),
                'health': memory_health
            })

            # IBKR Status
            try:
                bridge = get_nautilus_ibkr_bridge()
                connector = bridge.ibkr_connector  # Access internal connector for status
                ibkr_status = {
                    'connected': connector.connected if hasattr(connector, 'connected') else False,
                    'account_id': getattr(connector, 'account_id', 'unknown'),
                    'last_check': datetime.now().isoformat()
                }
                self.monitoring_data['ibkr_status'].append(ibkr_status)
            except Exception as e:
                await self.log_error('ibkr_check', f"Failed to check IBKR status: {e}")

            # Performance metrics
            runtime = (datetime.now() - self.start_time).total_seconds()
            error_count = sum(len(errors) for errors in self.monitoring_data['errors'].values())
            warning_count = sum(len(warnings) for warnings in self.monitoring_data['warnings'].values())

            perf_metrics = {
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': runtime,
                'total_errors': error_count,
                'total_warnings': warning_count,
                'rate_limits_hit': len(self.monitoring_data['rate_limits']),
                'api_health_score': api_health.get('summary', {}).get('overall_score', 0),
                'memory_healthy': memory_health.get('overall_healthy', False)
            }

            self.monitoring_data['performance_metrics'].append(perf_metrics)

        except Exception as e:
            await self.log_error('health_check', f"Health check failed: {e}")

    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive trading day report."""
        end_time = datetime.now()
        runtime = (end_time - self.start_time).total_seconds()

        # Error analysis
        error_summary = {}
        for error_type, errors in self.monitoring_data['errors'].items():
            error_summary[error_type] = {
                'count': len(errors),
                'patterns': Counter(e['pattern'] for e in errors),
                'first_occurrence': min(e['timestamp'] for e in errors),
                'last_occurrence': max(e['timestamp'] for e in errors)
            }

        # Warning analysis
        warning_summary = {}
        for warning_type, warnings in self.monitoring_data['warnings'].items():
            warning_summary[warning_type] = {
                'count': len(warnings),
                'patterns': Counter(w['pattern'] for w in warnings),
                'first_occurrence': min(w['timestamp'] for w in warnings),
                'last_occurrence': max(w['timestamp'] for w in warnings)
            }

        # Rate limiting analysis
        rate_limit_summary = {
            'total_events': len(self.monitoring_data['rate_limits']),
            'services_affected': list(set(rl['service'] for rl in self.monitoring_data['rate_limits'])),
            'average_retry_time': sum(rl['retry_time'] for rl in self.monitoring_data['rate_limits']) / max(len(self.monitoring_data['rate_limits']), 1)
        }

        # System health trends
        api_health_trend = []
        memory_health_trend = []

        for health in self.monitoring_data['api_health']:
            api_health_trend.append({
                'timestamp': health['timestamp'],
                'score': health['health'].get('summary', {}).get('overall_score', 0)
            })

        for health in self.monitoring_data['memory_health']:
            memory_health_trend.append({
                'timestamp': health['timestamp'],
                'healthy': health['health'].get('overall_healthy', False)
            })

        # Recommendations
        recommendations = self._generate_recommendations(error_summary, warning_summary, rate_limit_summary)

        report = {
            'session_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'runtime_seconds': runtime,
                'runtime_hours': runtime / 3600
            },
            'error_analysis': error_summary,
            'warning_analysis': warning_summary,
            'rate_limiting': rate_limit_summary,
            'system_health': {
                'api_health_trend': api_health_trend,
                'memory_health_trend': memory_health_trend,
                'ibkr_status_history': self.monitoring_data['ibkr_status']
            },
            'performance_metrics': self.monitoring_data['performance_metrics'],
            'recommendations': recommendations,
            'raw_data': self.monitoring_data
        }

        return report

    def _generate_recommendations(self, errors: Dict, warnings: Dict, rate_limits: Dict) -> List[str]:
        """Generate improvement recommendations based on monitoring data."""
        recommendations = []

        # Mem0 issues
        if 'mem0_gpt4_mini' in [p for e in errors.values() for p in e.get('patterns', {})]:
            recommendations.append("🔧 FIX: Mem0 backend disabled due to gpt-4o-mini model access issues. Consider alternative AI memory providers or fix Mem0 configuration.")

        # Discord rate limiting
        if rate_limits['total_events'] > 0:
            recommendations.append(f"⚡ RATE LIMIT: {rate_limits['total_events']} Discord rate limit events. Implement message batching or reduce update frequency.")

        # API failures
        api_errors = sum(len(e) for e in errors.get('api_failures', []))
        if api_errors > 0:
            recommendations.append(f"🌐 API: {api_errors} API failures detected. Review API key validity and implement better error handling.")

        # Memory issues
        memory_warnings = sum(len(w) for w in warnings.get('memory_backend', []))
        if memory_warnings > 0:
            recommendations.append(f"💾 MEMORY: {memory_warnings} memory backend warnings. Ensure Redis is running and consider memory optimization.")

        # IBKR issues
        ibkr_errors = sum(len(e) for e in errors.get('ibkr_connection', []))
        if ibkr_errors > 0:
            recommendations.append(f"📊 IBKR: {ibkr_errors} connection issues. Verify TWS is running and API connections are enabled.")

        # Position data issues
        position_warnings = sum(len(w) for w in warnings.get('position_data', []))
        if position_warnings > 0:
            recommendations.append("📈 POSITIONS: Mock position data being used. Implement real position retrieval with proper async handling.")

        # General recommendations
        total_errors = sum(len(e) for e in errors.values())
        total_warnings = sum(len(w) for w in warnings.values())

        if total_errors > 10:
            recommendations.append(f"🚨 HIGH ERROR RATE: {total_errors} total errors. Consider implementing circuit breakers and graceful degradation.")

        if total_warnings > 20:
            recommendations.append(f"⚠️ HIGH WARNING RATE: {total_warnings} total warnings. Review system configuration and dependencies.")

        return recommendations

    async def _send_alert(self, message: str):
        """Send critical alert (placeholder for actual alerting system)."""
        logger.critical(f"ALERT: {message}")
        # TODO: Implement actual alerting (Discord DM, email, etc.)

    async def save_report(self, filename: str = None):
        """Save monitoring report to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/trading_day_report_{timestamp}.json"

        report = await self.generate_report()

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Trading day report saved to {filename}")
        return filename

# Global monitor instance
_monitor_instance = None

def get_monitor() -> TradingDayMonitor:
    """Get singleton monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = TradingDayMonitor()
    return _monitor_instance

# Convenience functions for easy integration
async def log_error(error_type: str, message: str, context: Dict[str, Any] = None):
    """Log an error to the monitor."""
    await get_monitor().log_error(error_type, message, context)

async def log_warning(warning_type: str, message: str, context: Dict[str, Any] = None):
    """Log a warning to the monitor."""
    await get_monitor().log_warning(warning_type, message, context)

async def log_rate_limit(service: str, retry_time: float, endpoint: str = None):
    """Log a rate limit event."""
    await get_monitor().log_rate_limit(service, retry_time, endpoint)

async def check_system_health():
    """Perform system health check."""
    await get_monitor().check_system_health()

async def generate_monitoring_report() -> Dict[str, Any]:
    """Generate comprehensive monitoring report."""
    return await get_monitor().generate_report()

async def save_monitoring_report(filename: str = None) -> str:
    """Save monitoring report and return filename."""
    return await get_monitor().save_report(filename)

if __name__ == "__main__":
    # Continuous monitoring for trading day
    async def continuous_monitoring():
        monitor = get_monitor()
        logger.info("🎯 Trading Day Monitor started - running continuous monitoring")

        try:
            while True:
                # Check system health every 5 minutes
                await monitor.check_system_health()

                # Log current status
                runtime = (datetime.now() - monitor.start_time).total_seconds()
                error_count = sum(len(errors) for errors in monitor.monitoring_data['errors'].values())
                warning_count = sum(len(warnings) for warnings in monitor.monitoring_data['warnings'].values())

                logger.info(f"📊 Monitor Status - Runtime: {runtime/3600:.1f}h, Errors: {error_count}, Warnings: {warning_count}")

                # Save periodic report every hour
                if int(runtime) % 3600 == 0 and runtime > 0:
                    filename = await monitor.save_report()
                    logger.info(f"💾 Hourly report saved: {filename}")

                # Wait 5 minutes before next check
                await asyncio.sleep(300)

        except KeyboardInterrupt:
            logger.info("🛑 Trading Day Monitor stopping...")
            # Save final report
            final_report = await monitor.save_report("data/trading_day_final_report.json")
            logger.info(f"💾 Final report saved: {final_report}")

        except Exception as e:
            logger.error(f"❌ Trading Day Monitor error: {e}")
            # Save emergency report
            emergency_report = await monitor.save_report("data/trading_day_emergency_report.json")
            logger.info(f"🚨 Emergency report saved: {emergency_report}")

    asyncio.run(continuous_monitoring())