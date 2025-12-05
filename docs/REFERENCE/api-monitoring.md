# API Monitoring Guide

## 📊 API Health Monitoring

This guide covers comprehensive API monitoring strategies for ABC-Application, including health checks, performance monitoring, error tracking, and alerting.

## 🏥 Health Check Endpoints

### Comprehensive Health Checks
```python
# src/monitoring/health_checks.py
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import aiohttp
import aioredis
import psycopg2
import logging

logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive API health monitoring."""

    def __init__(self):
        self.checks = {}
        self.last_results = {}
        self.check_interval = 30  # seconds

    def register_check(self, name: str, check_func: callable, critical: bool = False):
        """Register a health check function."""
        self.checks[name] = {
            'function': check_func,
            'critical': critical,
            'last_run': None,
            'last_result': None
        }
        logger.info(f"Registered health check: {name}")

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_healthy = True

        for name, check_info in self.checks.items():
            try:
                start_time = time.time()
                result = await check_info['function']()
                duration = time.time() - start_time

                check_info['last_run'] = datetime.utcnow()
                check_info['last_result'] = result

                results[name] = {
                    'status': 'healthy' if result['healthy'] else 'unhealthy',
                    'response_time': duration,
                    'details': result.get('details', {}),
                    'timestamp': datetime.utcnow().isoformat()
                }

                if not result['healthy']:
                    if check_info['critical']:
                        overall_healthy = False
                    logger.warning(f"Health check failed: {name} - {result.get('message', 'Unknown error')}")

            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                if check_info['critical']:
                    overall_healthy = False
                logger.error(f"Health check error: {name} - {e}")

        return {
            'overall_status': 'healthy' if overall_healthy else 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results
        }

    async def database_health_check(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            # Assuming PostgreSQL connection
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'abc_app'),
                user=os.getenv('DB_USER', 'abc_user'),
                password=os.getenv('DB_PASSWORD'),
                connect_timeout=5
            )

            cursor = conn.cursor()

            # Simple query to test connectivity
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            # Performance test
            start_time = time.time()
            cursor.execute("SELECT COUNT(*) FROM information_schema.tables")
            count = cursor.fetchone()[0]
            query_time = time.time() - start_time

            cursor.close()
            conn.close()

            return {
                'healthy': True,
                'details': {
                    'connection': 'successful',
                    'tables_count': count,
                    'query_time': query_time
                }
            }

        except Exception as e:
            return {
                'healthy': False,
                'message': f'Database connection failed: {str(e)}'
            }

    async def redis_health_check(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        try:
            redis = aioredis.from_url(
                os.getenv('REDIS_URL', 'redis://localhost:6379'),
                socket_connect_timeout=5
            )

            # Test connection
            await redis.ping()

            # Test performance
            start_time = time.time()
            await redis.set('health_check', 'ok')
            await redis.get('health_check')
            await redis.delete('health_check')
            operation_time = time.time() - start_time

            # Get Redis info
            info = await redis.info()

            await redis.close()

            return {
                'healthy': True,
                'details': {
                    'connection': 'successful',
                    'operation_time': operation_time,
                    'used_memory': info.get('used_memory_human', 'unknown'),
                    'connected_clients': info.get('connected_clients', 0)
                }
            }

        except Exception as e:
            return {
                'healthy': False,
                'message': f'Redis connection failed: {str(e)}'
            }

    async def external_api_health_check(self) -> Dict[str, Any]:
        """Check external API dependencies."""
        apis_to_check = [
            {'name': 'Alpha Vantage', 'url': 'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey=demo'},
            {'name': 'Yahoo Finance', 'url': 'https://query1.finance.yahoo.com/v8/finance/chart/AAPL'},
            {'name': 'IEX Cloud', 'url': 'https://cloud.iexapis.com/stable/stock/AAPL/quote?token=pk_test_token'}
        ]

        results = {}

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for api in apis_to_check:
                try:
                    start_time = time.time()
                    async with session.get(api['url']) as response:
                        response_time = time.time() - start_time

                        results[api['name']] = {
                            'status': response.status,
                            'response_time': response_time,
                            'healthy': response.status == 200
                        }

                except Exception as e:
                    results[api['name']] = {
                        'status': 'error',
                        'error': str(e),
                        'healthy': False
                    }

        all_healthy = all(result['healthy'] for result in results.values())

        return {
            'healthy': all_healthy,
            'details': results
        }

    async def memory_health_check(self) -> Dict[str, Any]:
        """Check application memory usage."""
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        # Define thresholds
        critical_threshold = 90  # 90%
        warning_threshold = 75   # 75%

        healthy = memory_percent < critical_threshold

        status = 'healthy'
        if memory_percent >= critical_threshold:
            status = 'critical'
        elif memory_percent >= warning_threshold:
            status = 'warning'

        return {
            'healthy': healthy,
            'details': {
                'memory_used_mb': memory_info.rss / 1024 / 1024,
                'memory_percent': memory_percent,
                'status': status
            }
        }

# Global health checker instance
health_checker = HealthChecker()

# Register default checks
health_checker.register_check('database', health_checker.database_health_check, critical=True)
health_checker.register_check('redis', health_checker.redis_health_check, critical=True)
health_checker.register_check('external_apis', health_checker.external_api_health_check, critical=False)
health_checker.register_check('memory', health_checker.memory_health_check, critical=True)

def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    return health_checker
```

### Health Check API Endpoint
```python
# src/api/health.py
from flask import Blueprint, jsonify
from src.monitoring.health_checks import get_health_checker
import logging

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)

@health_bp.route('/health')
async def health_check():
    """Basic health check endpoint."""
    try:
        checker = get_health_checker()
        results = await checker.run_all_checks()

        status_code = 200 if results['overall_status'] == 'healthy' else 503

        return jsonify(results), status_code

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'overall_status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@health_bp.route('/health/detailed')
async def detailed_health_check():
    """Detailed health check with all component status."""
    try:
        checker = get_health_checker()
        results = await checker.run_all_checks()

        # Add system information
        import psutil
        import platform

        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            'disk_usage': psutil.disk_usage('/').percent
        }

        results['system_info'] = system_info

        status_code = 200 if results['overall_status'] == 'healthy' else 503

        return jsonify(results), status_code

    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return jsonify({
            'overall_status': 'error',
            'error': str(e)
        }), 500

@health_bp.route('/health/<component>')
async def component_health_check(component: str):
    """Check specific component health."""
    try:
        checker = get_health_checker()

        if component not in checker.checks:
            return jsonify({
                'error': f'Unknown component: {component}',
                'available_components': list(checker.checks.keys())
            }), 404

        # Run specific check
        check_info = checker.checks[component]
        result = await check_info['function']()

        response = {
            'component': component,
            'status': 'healthy' if result['healthy'] else 'unhealthy',
            'critical': check_info['critical'],
            'details': result.get('details', {}),
            'timestamp': datetime.utcnow().isoformat()
        }

        if not result['healthy']:
            response['message'] = result.get('message', 'Component unhealthy')

        status_code = 200 if result['healthy'] else 503

        return jsonify(response), status_code

    except Exception as e:
        logger.error(f"Component health check failed for {component}: {e}")
        return jsonify({
            'component': component,
            'status': 'error',
            'error': str(e)
        }), 500
```

## 📈 API Performance Monitoring

### Request/Response Monitoring
```python
# src/monitoring/api_monitor.py
import time
import threading
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class APIMonitor:
    """Monitor API performance and usage patterns."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.requests = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'errors': 0,
            'last_request': None
        })
        self._lock = threading.Lock()

    def record_request(
        self,
        method: str,
        endpoint: str,
        response_time: float,
        status_code: int,
        user_id: Optional[str] = None
    ):
        """Record API request metrics."""
        with self._lock:
            timestamp = datetime.utcnow()

            request_data = {
                'method': method,
                'endpoint': endpoint,
                'response_time': response_time,
                'status_code': status_code,
                'user_id': user_id,
                'timestamp': timestamp
            }

            self.requests.append(request_data)
            self.response_times.append(response_time)

            # Update endpoint statistics
            endpoint_key = f"{method} {endpoint}"
            stats = self.endpoint_stats[endpoint_key]
            stats['count'] += 1
            stats['total_time'] += response_time
            stats['last_request'] = timestamp

            if status_code >= 400:
                stats['errors'] += 1
                self.errors.append(request_data)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current API performance metrics."""
        with self._lock:
            if not self.requests:
                return {'total_requests': 0}

            response_times_list = list(self.response_times)

            return {
                'total_requests': len(self.requests),
                'avg_response_time': sum(response_times_list) / len(response_times_list),
                'min_response_time': min(response_times_list),
                'max_response_time': max(response_times_list),
                'p95_response_time': self._percentile(response_times_list, 95),
                'p99_response_time': self._percentile(response_times_list, 99),
                'error_rate': len(self.errors) / len(self.requests) if self.requests else 0,
                'requests_per_second': self._calculate_rps()
            }

    def get_endpoint_metrics(self) -> Dict[str, Any]:
        """Get per-endpoint performance metrics."""
        with self._lock:
            metrics = {}

            for endpoint, stats in self.endpoint_stats.items():
                if stats['count'] > 0:
                    avg_time = stats['total_time'] / stats['count']
                    error_rate = stats['errors'] / stats['count']

                    metrics[endpoint] = {
                        'total_requests': stats['count'],
                        'avg_response_time': avg_time,
                        'error_rate': error_rate,
                        'last_request': stats['last_request'].isoformat() if stats['last_request'] else None
                    }

            return metrics

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _calculate_rps(self) -> float:
        """Calculate requests per second."""
        if len(self.requests) < 2:
            return 0

        # Get timestamps from last 60 seconds
        now = datetime.utcnow()
        recent_requests = [
            r for r in self.requests
            if (now - r['timestamp']).total_seconds() <= 60
        ]

        if len(recent_requests) < 2:
            return 0

        time_span = (recent_requests[-1]['timestamp'] - recent_requests[0]['timestamp']).total_seconds()
        if time_span == 0:
            return 0

        return len(recent_requests) / time_span

    def get_error_analysis(self) -> Dict[str, Any]:
        """Analyze API errors."""
        with self._lock:
            error_counts = defaultdict(int)
            endpoint_errors = defaultdict(int)

            for error in self.errors:
                error_counts[error['status_code']] += 1
                endpoint_key = f"{error['method']} {error['endpoint']}"
                endpoint_errors[endpoint_key] += 1

            return {
                'total_errors': len(self.errors),
                'error_counts_by_status': dict(error_counts),
                'errors_by_endpoint': dict(endpoint_errors),
                'recent_errors': [
                    {
                        'method': e['method'],
                        'endpoint': e['endpoint'],
                        'status_code': e['status_code'],
                        'timestamp': e['timestamp'].isoformat()
                    } for e in list(self.errors)[-10:]  # Last 10 errors
                ]
            }

# Global API monitor instance
api_monitor = APIMonitor()

def get_api_monitor() -> APIMonitor:
    """Get global API monitor instance."""
    return api_monitor
```

### Flask Middleware for Monitoring
```python
# src/middleware/monitoring_middleware.py
from flask import request, g
import time
import logging

logger = logging.getLogger(__name__)

class MonitoringMiddleware:
    """Flask middleware for API monitoring."""

    def __init__(self, app, api_monitor):
        self.app = app
        self.api_monitor = api_monitor
        self.setup_middleware()

    def setup_middleware(self):
        """Setup Flask middleware."""

        @self.app.before_request
        def before_request():
            """Record request start time."""
            g.start_time = time.time()
            g.request_method = request.method
            g.request_path = request.path

        @self.app.after_request
        def after_request(response):
            """Record request metrics after completion."""
            if hasattr(g, 'start_time'):
                response_time = time.time() - g.start_time

                # Get user ID if available (depends on authentication system)
                user_id = getattr(g, 'user_id', None)

                # Record metrics
                self.api_monitor.record_request(
                    method=g.request_method,
                    endpoint=g.request_path,
                    response_time=response_time,
                    status_code=response.status_code,
                    user_id=user_id
                )

                # Add performance headers
                response.headers['X-Response-Time'] = f"{response_time:.3f}s"

            return response

        @self.app.errorhandler(Exception)
        def handle_error(error):
            """Handle and log application errors."""
            logger.error(f"Application error: {error}", exc_info=True)

            # Record error metrics
            if hasattr(g, 'start_time'):
                response_time = time.time() - g.start_time
                user_id = getattr(g, 'user_id', None)

                self.api_monitor.record_request(
                    method=getattr(g, 'request_method', 'UNKNOWN'),
                    endpoint=getattr(g, 'request_path', '/'),
                    response_time=response_time,
                    status_code=500,  # Internal server error
                    user_id=user_id
                )

            return {"error": "Internal server error"}, 500

def setup_monitoring(app):
    """Setup monitoring middleware for Flask app."""
    from src.monitoring.api_monitor import get_api_monitor

    monitor = get_api_monitor()
    MonitoringMiddleware(app, monitor)

    logger.info("API monitoring middleware initialized")
```

## 🚨 Alerting System

### Automated Alerting
```python
# src/monitoring/alerts.py
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertManager:
    """Manage and send alerts for system issues."""

    def __init__(self):
        self.alert_channels = []
        self.alert_history = []
        self.cooldown_period = 300  # 5 minutes between similar alerts

    def add_email_channel(self, smtp_server: str, smtp_port: int, username: str, password: str, recipients: List[str]):
        """Add email alert channel."""
        self.alert_channels.append({
            'type': 'email',
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'recipients': recipients
        })

    def add_slack_channel(self, webhook_url: str, channel: str = None):
        """Add Slack alert channel."""
        self.alert_channels.append({
            'type': 'slack',
            'webhook_url': webhook_url,
            'channel': channel
        })

    def add_discord_channel(self, webhook_url: str):
        """Add Discord alert channel."""
        self.alert_channels.append({
            'type': 'discord',
            'webhook_url': webhook_url
        })

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        details: Optional[Dict[str, Any]] = None
    ):
        """Send alert through all configured channels."""
        alert_data = {
            'title': title,
            'message': message,
            'severity': severity.value,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details or {}
        }

        # Check cooldown
        if self._is_on_cooldown(alert_data):
            logger.debug(f"Alert on cooldown: {title}")
            return

        self.alert_history.append(alert_data)

        # Send through all channels
        tasks = []
        for channel in self.alert_channels:
            if channel['type'] == 'email':
                tasks.append(self._send_email_alert(channel, alert_data))
            elif channel['type'] == 'slack':
                tasks.append(self._send_slack_alert(channel, alert_data))
            elif channel['type'] == 'discord':
                tasks.append(self._send_discord_alert(channel, alert_data))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Alert sent: {title} ({severity.value})")

    def _is_on_cooldown(self, alert_data: Dict[str, Any]) -> bool:
        """Check if similar alert is on cooldown."""
        now = datetime.utcnow()

        for historical_alert in reversed(self.alert_history[-10:]):  # Check last 10 alerts
            alert_time = datetime.fromisoformat(historical_alert['timestamp'])
            time_diff = (now - alert_time).total_seconds()

            if (time_diff < self.cooldown_period and
                historical_alert['title'] == alert_data['title'] and
                historical_alert['severity'] == alert_data['severity']):
                return True

        return False

    async def _send_email_alert(self, channel: Dict[str, Any], alert_data: Dict[str, Any]):
        """Send alert via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = channel['username']
            msg['To'] = ', '.join(channel['recipients'])
            msg['Subject'] = f"[{alert_data['severity'].upper()}] {alert_data['title']}"

            body = f"""
{alert_data['message']}

Severity: {alert_data['severity']}
Timestamp: {alert_data['timestamp']}

Details:
{json.dumps(alert_data['details'], indent=2)}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(channel['smtp_server'], channel['smtp_port'])
            server.starttls()
            server.login(channel['username'], channel['password'])
            text = msg.as_string()
            server.sendmail(channel['username'], channel['recipients'], text)
            server.quit()

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def _send_slack_alert(self, channel: Dict[str, Any], alert_data: Dict[str, Any]):
        """Send alert to Slack."""
        try:
            payload = {
                'text': f"*{alert_data['severity'].upper()}*: {alert_data['title']}\n{alert_data['message']}",
                'attachments': [{
                    'color': self._get_severity_color(alert_data['severity']),
                    'fields': [
                        {'title': 'Timestamp', 'value': alert_data['timestamp'], 'short': True},
                        {'title': 'Details', 'value': f"```{json.dumps(alert_data['details'], indent=2)}```", 'short': False}
                    ]
                }]
            }

            if channel.get('channel'):
                payload['channel'] = channel['channel']

            async with aiohttp.ClientSession() as session:
                async with session.post(channel['webhook_url'], json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Slack alert failed: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def _send_discord_alert(self, channel: Dict[str, Any], alert_data: Dict[str, Any]):
        """Send alert to Discord."""
        try:
            embed = {
                'title': alert_data['title'],
                'description': alert_data['message'],
                'color': self._get_severity_color(alert_data['severity']),
                'timestamp': alert_data['timestamp'],
                'fields': [
                    {'name': 'Severity', 'value': alert_data['severity'], 'inline': True},
                    {'name': 'Details', 'value': f"```json\n{json.dumps(alert_data['details'], indent=2)}\n```", 'inline': False}
                ]
            }

            payload = {
                'embeds': [embed]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(channel['webhook_url'], json=payload) as response:
                    if response.status != 204:
                        logger.error(f"Discord alert failed: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    def _get_severity_color(self, severity: str) -> int:
        """Get color code for severity level."""
        colors = {
            'info': 0x3498db,      # Blue
            'warning': 0xf39c12,   # Orange
            'error': 0xe74c3c,     # Red
            'critical': 0x9b59b6   # Purple
        }
        return colors.get(severity, 0x95a5a6)  # Default gray

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return self.alert_history[-limit:]

# Global alert manager instance
alert_manager = AlertManager()

def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    return alert_manager
```

### Automated Alert Rules
```python
# src/monitoring/alert_rules.py
import asyncio
from typing import Dict, Any, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AlertRule:
    """Define alert rules for automatic monitoring."""

    def __init__(self, name: str, condition_func: Callable, severity: str, message: str, cooldown: int = 300):
        self.name = name
        self.condition_func = condition_func
        self.severity = severity
        self.message = message
        self.cooldown = cooldown
        self.last_triggered = None

    async def check_and_alert(self, metrics: Dict[str, Any]) -> bool:
        """Check condition and send alert if triggered."""
        try:
            should_alert, details = await self.condition_func(metrics)

            if should_alert and self._should_trigger():
                from src.monitoring.alerts import get_alert_manager, AlertSeverity

                manager = get_alert_manager()
                severity_enum = AlertSeverity[self.severity.upper()]

                await manager.send_alert(
                    title=f"Alert Rule: {self.name}",
                    message=self.message,
                    severity=severity_enum,
                    details=details
                )

                self.last_triggered = datetime.utcnow()
                return True

        except Exception as e:
            logger.error(f"Alert rule check failed for {self.name}: {e}")

        return False

    def _should_trigger(self) -> bool:
        """Check if alert should be triggered based on cooldown."""
        if self.last_triggered is None:
            return True

        time_since_last = (datetime.utcnow() - self.last_triggered).total_seconds()
        return time_since_last >= self.cooldown

class AlertRuleEngine:
    """Engine for managing and executing alert rules."""

    def __init__(self):
        self.rules = []

    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    async def evaluate_rules(self, metrics: Dict[str, Any]):
        """Evaluate all alert rules against metrics."""
        tasks = [rule.check_and_alert(metrics) for rule in self.rules]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        triggered_count = sum(1 for result in results if result is True)
        if triggered_count > 0:
            logger.info(f"Triggered {triggered_count} alert rules")

# Predefined alert rules
async def high_error_rate_condition(metrics: Dict[str, Any]) -> tuple:
    """Check for high API error rate."""
    error_rate = metrics.get('error_rate', 0)
    threshold = 0.05  # 5%

    if error_rate > threshold:
        return True, {
            'error_rate': error_rate,
            'threshold': threshold,
            'total_requests': metrics.get('total_requests', 0)
        }

    return False, {}

async def slow_response_condition(metrics: Dict[str, Any]) -> tuple:
    """Check for slow API responses."""
    avg_response_time = metrics.get('avg_response_time', 0)
    threshold = 2.0  # 2 seconds

    if avg_response_time > threshold:
        return True, {
            'avg_response_time': avg_response_time,
            'threshold': threshold,
            'p95_response_time': metrics.get('p95_response_time', 0)
        }

    return False, {}

async def high_memory_usage_condition(metrics: Dict[str, Any]) -> tuple:
    """Check for high memory usage."""
    # This would come from system metrics
    memory_percent = metrics.get('memory_percent', 0)
    threshold = 85  # 85%

    if memory_percent > threshold:
        return True, {
            'memory_percent': memory_percent,
            'threshold': threshold
        }

    return False, {}

# Setup default alert rules
def setup_default_alert_rules(engine: AlertRuleEngine):
    """Setup default alert rules."""
    engine.add_rule(AlertRule(
        name="High Error Rate",
        condition_func=high_error_rate_condition,
        severity="error",
        message="API error rate has exceeded threshold"
    ))

    engine.add_rule(AlertRule(
        name="Slow Responses",
        condition_func=slow_response_condition,
        severity="warning",
        message="API response times are too slow"
    ))

    engine.add_rule(AlertRule(
        name="High Memory Usage",
        condition_func=high_memory_usage_condition,
        severity="critical",
        message="System memory usage is critically high"
    ))

# Global alert rule engine
alert_engine = AlertRuleEngine()
setup_default_alert_rules(alert_engine)

def get_alert_engine() -> AlertRuleEngine:
    """Get global alert rule engine."""
    return alert_engine
```

---

*These monitoring and alerting systems provide comprehensive visibility into ABC-Application's API health and performance. Regular review of metrics and alerts helps maintain system reliability and quickly identify issues.*