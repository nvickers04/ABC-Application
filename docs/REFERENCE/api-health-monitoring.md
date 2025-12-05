# API Health Monitoring

## Overview

The ABC Application system implements comprehensive API health monitoring to ensure reliable operation of all external data sources, trading platforms, and internal services. This monitoring system provides real-time visibility into system health, automatic failure detection, and proactive issue resolution.

## Monitoring Architecture

### Core Components

1. **Health Check Engine** - Central monitoring coordinator
2. **Service Monitors** - Individual service health trackers
3. **Alert System** - Notification and escalation system
4. **Metrics Collector** - Performance and health metrics aggregation
5. **Dashboard** - Real-time monitoring visualization

### Health Check Types

#### Liveness Checks
- Basic connectivity and responsiveness
- Process health verification
- Resource availability confirmation

#### Readiness Checks
- Full functionality verification
- Dependency availability
- Configuration validation

#### Deep Health Checks
- Performance metrics analysis
- Data quality assessment
- Business logic validation

## API Health Monitoring Implementation

### Core Health Check Classes

```python
# src/utils/api_health_monitor.py
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum

class APIStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class APIHealthMetrics:
    """Health metrics for an API endpoint"""
    api_name: str
    status: APIStatus
    response_time: float
    success_rate: float
    error_count: int
    total_requests: int
    last_check: datetime
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    circuit_breaker_state: str = "CLOSED"

class APIHealthMonitor:
    """Monitors health of all API endpoints"""

    def __init__(self, check_interval: int = 300):  # 5 minutes default
        self.check_interval = check_interval
        self.metrics: Dict[str, APIHealthMetrics] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # API endpoints to monitor
        self.api_endpoints = {
            'marketdataapp_api': self._check_marketdataapp_api,
            'kalshi_api': self._check_kalshi_api,
            'yfinance': self._check_yfinance,
            'news_api': self._check_news_api,
            'economic_data': self._check_economic_data,
            'currents_news': self._check_currents_news,
            'twitter_api': self._check_twitter_api,
            'whale_wisdom': self._check_whale_wisdom,
            'grok_api': self._check_grok_api
        }

        # Initialize metrics for all APIs
        for api_name in self.api_endpoints.keys():
            self.metrics[api_name] = APIHealthMetrics(
                api_name=api_name,
                status=APIStatus.UNKNOWN,
                response_time=0.0,
                success_rate=0.0,
                error_count=0,
                total_requests=0,
                last_check=datetime.now()
            )

    def start_monitoring(self):
        """Start the background monitoring thread"""
        if self.monitoring_active:
            logger.warning("API health monitoring is already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"API health monitoring started with {self.check_interval}s interval")

    def stop_monitoring(self):
        """Stop the background monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("API health monitoring stopped")

    def _check_all_apis(self):
        """Check health of all APIs"""
        logger.info("Starting API health checks...")

        for api_name, check_func in self.api_endpoints.items():
            try:
                start_time = time.time()
                success = check_func()
                response_time = time.time() - start_time

                self._update_metrics(api_name, success, response_time)

            except Exception as e:
                logger.error(f"Error checking {api_name}: {e}")
                self._update_metrics(api_name, False, 0.0, str(e))

        logger.info("API health checks completed")

    def _update_metrics(self, api_name: str, success: bool, response_time: float, error_msg: Optional[str] = None):
        """Update health metrics for an API"""
        metrics = self.metrics[api_name]
        metrics.total_requests += 1
        metrics.last_check = datetime.now()

        if success:
            metrics.response_time = response_time
            metrics.consecutive_failures = 0
            if error_msg:
                metrics.last_error = None
        else:
            metrics.error_count += 1
            metrics.consecutive_failures += 1
            if error_msg:
                metrics.last_error = error_msg

        # Calculate success rate (rolling window of last 100 requests)
        window_size = min(100, metrics.total_requests)
        recent_successes = metrics.total_requests - metrics.error_count
        metrics.success_rate = recent_successes / window_size if window_size > 0 else 0.0

        # Determine status
        if metrics.consecutive_failures >= 5:
            metrics.status = APIStatus.UNHEALTHY
            metrics.circuit_breaker_state = "OPEN"
        elif metrics.consecutive_failures >= 2:
            metrics.status = APIStatus.DEGRADED
            metrics.circuit_breaker_state = "HALF_OPEN"
        elif metrics.success_rate >= 0.95:
            metrics.status = APIStatus.HEALTHY
            metrics.circuit_breaker_state = "CLOSED"
        else:
            metrics.status = APIStatus.DEGRADED
            metrics.circuit_breaker_state = "CLOSED"
                        response_time_ms=0,
                        timestamp=current_time,
                        message=f"Check failed with exception: {str(e)}"
                    )
                    results[name] = health_result
                    self.last_results[name] = health_result

        return results

    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Check all services and return comprehensive health status"""
        results = await self.run_health_checks()

        # Aggregate overall health
        healthy_count = sum(1 for r in results.values() if r.is_healthy)
        total_count = len(results)

        overall_status = HealthStatus.HEALTHY
        if healthy_count < total_count * 0.8:  # Less than 80% healthy
            overall_status = HealthStatus.UNHEALTHY
        elif healthy_count < total_count:  # Some services unhealthy
            overall_status = HealthStatus.DEGRADED

        results['overall'] = HealthCheckResult(
            service_name='overall',
            status=overall_status,
            response_time_ms=0,
            timestamp=time.time(),
            message=f"{healthy_count}/{total_count} services healthy",
            details={'healthy_count': healthy_count, 'total_count': total_count}
        )

        return results
```

### Data Source Health Checks

```python
# src/monitoring/data_source_health.py
import aiohttp
import asyncpg
import redis.asyncio as redis
from .health_checker import HealthChecker, HealthStatus, HealthCheckResult

class DataSourceHealthChecker:
    def __init__(self, config):
        self.config = config
        self.session = None

    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def check_ibkr_api(self) -> Dict:
        """Check IBKR API connectivity and functionality"""
        try:
            # This would integrate with actual IBKR API client
            # For demonstration, we'll simulate the check
            session = await self._get_session()

            # Check IBKR API endpoint (mock)
            async with session.get('https://api.ibkr.com/health') as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'operational':
                        return {
                            'status': 'healthy',
                            'message': 'IBKR API operational',
                            'details': {
                                'latency_ms': response.headers.get('X-Response-Time', 0),
                                'version': data.get('version', 'unknown')
                            }
                        }

            return {
                'status': 'unhealthy',
                'message': 'IBKR API not responding properly'
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'IBKR API check failed: {str(e)}'
            }

    async def check_alpha_vantage_api(self) -> Dict:
        """Check Alpha Vantage API health"""
        try:
            session = await self._get_session()
            api_key = self.config.get('ALPHA_VANTAGE_API_KEY')

            # Test with a simple quote request
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={api_key}"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    if 'Global Quote' in data and data['Global Quote']:
                        return {
                            'status': 'healthy',
                            'message': 'Alpha Vantage API responding',
                            'details': {
                                'remaining_calls': response.headers.get('X-RateLimit-Remaining', 'unknown'),
                                'reset_time': response.headers.get('X-RateLimit-Reset', 'unknown')
                            }
                        }

                    elif 'Note' in data:
                        return {
                            'status': 'degraded',
                            'message': 'Alpha Vantage API rate limited',
                            'details': {'rate_limit_note': data['Note']}
                        }

            return {
                'status': 'unhealthy',
                'message': 'Alpha Vantage API not accessible'
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Alpha Vantage API check failed: {str(e)}'
            }

    async def check_database_connection(self) -> Dict:
        """Check PostgreSQL database connectivity"""
        try:
            conn = await asyncpg.connect(
                host=self.config.get('DB_HOST', 'localhost'),
                port=self.config.get('DB_PORT', 5432),
                user=self.config.get('DB_USER'),
                password=self.config.get('DB_PASSWORD'),
                database=self.config.get('DB_NAME')
            )

            # Test basic query
            result = await conn.fetchval("SELECT 1")
            await conn.close()

            if result == 1:
                return {
                    'status': 'healthy',
                    'message': 'Database connection successful',
                    'details': {'connection_pool_size': 10}  # Would get from actual pool
                }

            return {
                'status': 'unhealthy',
                'message': 'Database query failed'
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Database connection failed: {str(e)}'
            }

    async def check_redis_connection(self) -> Dict:
        """Check Redis connectivity"""
        try:
            r = redis.Redis(
                host=self.config.get('REDIS_HOST', 'localhost'),
                port=self.config.get('REDIS_PORT', 6379),
                password=self.config.get('REDIS_PASSWORD'),
                decode_responses=True
            )

            # Test basic operations
            await r.set('health_check', 'ok')
            result = await r.get('health_check')
            await r.delete('health_check')

            if result == 'ok':
                info = await r.info()
                return {
                    'status': 'healthy',
                    'message': 'Redis connection successful',
                    'details': {
                        'version': info.get('redis_version'),
                        'connected_clients': info.get('connected_clients'),
                        'used_memory_human': info.get('used_memory_human')
                    }
                }

            return {
                'status': 'unhealthy',
                'message': 'Redis operations failed'
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Redis connection failed: {str(e)}'
            }

    async def check_news_api(self) -> Dict:
        """Check NewsAPI health"""
        try:
            session = await self._get_session()
            api_key = self.config.get('NEWSAPI_KEY')

            url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get('status') == 'ok' and 'articles' in data:
                        return {
                            'status': 'healthy',
                            'message': 'NewsAPI responding',
                            'details': {
                                'article_count': len(data.get('articles', [])),
                                'total_results': data.get('totalResults', 0)
                            }
                        }

                    elif data.get('code') == 'rateLimited':
                        return {
                            'status': 'degraded',
                            'message': 'NewsAPI rate limited',
                            'details': {'reset_time': data.get('message', 'unknown')}
                        }

            return {
                'status': 'unhealthy',
                'message': 'NewsAPI not accessible'
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'NewsAPI check failed: {str(e)}'
            }

    async def check_twitter_api(self) -> Dict:
        """Check Twitter API health"""
        try:
            # This would use tweepy or similar library
            # For demonstration, we'll simulate the check
            return {
                'status': 'healthy',
                'message': 'Twitter API accessible',
                'details': {
                    'rate_limit_remaining': 250,
                    'rate_limit_reset': '2024-01-15T15:00:00Z'
                }
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Twitter API check failed: {str(e)}'
            }
```

### Agent Health Monitoring

```python
# src/monitoring/agent_health.py
from .health_checker import HealthChecker, HealthStatus

class AgentHealthChecker:
    def __init__(self, agent_manager):
        self.agent_manager = agent_manager

    async def check_data_agent_health(self) -> Dict:
        """Check data agent health and analyzers"""
        try:
            data_agent = self.agent_manager.get_agent('data_agent')

            if not data_agent:
                return {
                    'status': 'unhealthy',
                    'message': 'Data agent not found'
                }

            # Check agent responsiveness
            health_status = await data_agent.health_check()

            # Check analyzer status
            analyzer_status = {}
            for analyzer_name in ['market_data', 'sentiment', 'economic', 'options_data']:
                analyzer = getattr(data_agent, f'{analyzer_name}_analyzer', None)
                if analyzer:
                    analyzer_status[analyzer_name] = await analyzer.health_check()

            healthy_analyzers = sum(1 for s in analyzer_status.values() if s.get('status') == 'healthy')
            total_analyzers = len(analyzer_status)

            overall_status = 'healthy' if healthy_analyzers == total_analyzers else 'degraded'

            return {
                'status': overall_status,
                'message': f'Data agent: {healthy_analyzers}/{total_analyzers} analyzers healthy',
                'details': {
                    'agent_status': health_status,
                    'subagent_status': subagent_status,
                    'healthy_subagents': healthy_subagents,
                    'total_subagents': total_subagents
                }
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Data agent health check failed: {str(e)}'
            }

    async def check_strategy_agent_health(self) -> Dict:
        """Check strategy agent health"""
        try:
            strategy_agent = self.agent_manager.get_agent('strategy_agent')

            if not strategy_agent:
                return {
                    'status': 'unhealthy',
                    'message': 'Strategy agent not found'
                }

            # Check recent strategy generation performance
            recent_performance = await strategy_agent.get_recent_performance()

            # Check subagent status
            subagents = ['options', 'ml_models', 'pairs_trading', 'arbitrage']
            subagent_status = {}

            for subagent in subagents:
                subagent_instance = getattr(strategy_agent, f'{subagent}_subagent', None)
                if subagent_instance:
                    status = await subagent_instance.health_check()
                    subagent_status[subagent] = status

            # Determine overall health
            healthy_subagents = sum(1 for s in subagent_status.values() if s.get('status') == 'healthy')

            if healthy_subagents == len(subagents):
                overall_status = 'healthy'
            elif healthy_subagents >= len(subagents) // 2:
                overall_status = 'degraded'
            else:
                overall_status = 'unhealthy'

            return {
                'status': overall_status,
                'message': f'Strategy agent: {healthy_subagents}/{len(subagents)} subagents healthy',
                'details': {
                    'subagent_status': subagent_status,
                    'recent_performance': recent_performance
                }
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Strategy agent health check failed: {str(e)}'
            }

    async def check_execution_agent_health(self) -> Dict:
        """Check execution agent health"""
        try:
            execution_agent = self.agent_manager.get_agent('execution_agent')

            if not execution_agent:
                return {
                    'status': 'unhealthy',
                    'message': 'Execution agent not found'
                }

            # Check IBKR connection status
            connection_status = await execution_agent.check_ibkr_connection()

            # Check recent execution performance
            recent_executions = await execution_agent.get_recent_executions()

            # Calculate execution quality metrics
            if recent_executions:
                avg_slippage = sum(e.get('slippage_bps', 0) for e in recent_executions) / len(recent_executions)
                success_rate = sum(1 for e in recent_executions if e.get('status') == 'filled') / len(recent_executions)
            else:
                avg_slippage = 0
                success_rate = 0

            # Determine health based on metrics
            if connection_status.get('connected') and success_rate > 0.95 and avg_slippage < 5:
                status = 'healthy'
            elif connection_status.get('connected') and success_rate > 0.90:
                status = 'degraded'
            else:
                status = 'unhealthy'

            return {
                'status': status,
                'message': f'Execution agent: {success_rate:.1%} success rate, {avg_slippage:.1f}bps avg slippage',
                'details': {
                    'connection_status': connection_status,
                    'success_rate': success_rate,
                    'avg_slippage_bps': avg_slippage,
                    'recent_execution_count': len(recent_executions)
                }
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Execution agent health check failed: {str(e)}'
            }

    async def check_risk_agent_health(self) -> Dict:
        """Check risk agent health"""
        try:
            risk_agent = self.agent_manager.get_agent('risk_agent')

            if not risk_agent:
                return {
                    'status': 'unhealthy',
                    'message': 'Risk agent not found'
                }

            # Check risk calculations
            portfolio_var = await risk_agent.calculate_portfolio_var()
            stress_test_results = await risk_agent.run_stress_tests()

            # Check compliance status
            compliance_status = await risk_agent.check_compliance()

            # Determine health based on risk metrics
            if (portfolio_var < 0.15 and  # 15% VaR limit
                all(st['breached'] == False for st in stress_test_results.values()) and
                compliance_status.get('compliant', False)):
                status = 'healthy'
            elif portfolio_var < 0.20:  # 20% VaR warning
                status = 'degraded'
            else:
                status = 'unhealthy'

            return {
                'status': status,
                'message': f'Risk agent: VaR {portfolio_var:.1%}, compliance {compliance_status.get("status", "unknown")}',
                'details': {
                    'portfolio_var': portfolio_var,
                    'stress_test_results': stress_test_results,
                    'compliance_status': compliance_status
                }
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Risk agent health check failed: {str(e)}'
            }
```

### Alert System

```python
# src/monitoring/alert_system.py
import asyncio
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List
from .health_checker import HealthStatus, HealthCheckResult

class AlertSystem:
    def __init__(self, config):
        self.config = config
        self.alert_history = []
        self.alert_thresholds = {
            'max_alerts_per_hour': 10,
            'alert_cooldown_minutes': 15,
            'escalation_levels': {
                'warning': ['email'],
                'critical': ['email', 'sms', 'slack']
            }
        }

    async def process_health_results(self, results: Dict[str, HealthCheckResult]):
        """Process health check results and generate alerts"""
        alerts = []

        for service_name, result in results.items():
            if service_name == 'overall':
                continue

            alert = self._evaluate_alert_conditions(service_name, result)
            if alert:
                alerts.append(alert)

        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)

        return alerts

    def _evaluate_alert_conditions(self, service_name: str, result: HealthCheckResult) -> Dict:
        """Evaluate if an alert should be generated"""
        alert = None

        # Check status-based alerts
        if result.status == HealthStatus.UNHEALTHY:
            alert = {
                'level': 'critical',
                'service': service_name,
                'type': 'status',
                'message': f'Service {service_name} is unhealthy: {result.message}',
                'details': {
                    'status': result.status.value,
                    'response_time': result.response_time_ms,
                    'timestamp': result.timestamp
                }
            }

        elif result.status == HealthStatus.DEGRADED:
            # Only alert on degradation if not recently alerted
            if not self._recently_alerted(service_name, 'degraded'):
                alert = {
                    'level': 'warning',
                    'service': service_name,
                    'type': 'status',
                    'message': f'Service {service_name} is degraded: {result.message}',
                    'details': {
                        'status': result.status.value,
                        'response_time': result.response_time_ms,
                        'timestamp': result.timestamp
                    }
                }

        # Check performance-based alerts
        if result.response_time_ms > self.config.get('response_time_critical', 30000):
            if not self._recently_alerted(service_name, 'performance'):
                alert = {
                    'level': 'critical',
                    'service': service_name,
                    'type': 'performance',
                    'message': f'Service {service_name} response time critically high: {result.response_time_ms:.0f}ms',
                    'details': {
                        'response_time': result.response_time_ms,
                        'threshold': self.config.get('response_time_critical', 30000)
                    }
                }

        return alert

    def _recently_alerted(self, service: str, alert_type: str, minutes: int = 15) -> bool:
        """Check if service was recently alerted for same type"""
        cutoff_time = asyncio.get_event_loop().time() - (minutes * 60)

        recent_alerts = [
            a for a in self.alert_history
            if a['service'] == service and a['type'] == alert_type and a['timestamp'] > cutoff_time
        ]

        return len(recent_alerts) > 0

    async def _send_alert(self, alert: Dict):
        """Send alert through configured channels"""
        alert['timestamp'] = asyncio.get_event_loop().time()
        self.alert_history.append(alert)

        channels = self.alert_thresholds['escalation_levels'].get(alert['level'], ['email'])

        tasks = []
        if 'email' in channels:
            tasks.append(self._send_email_alert(alert))
        if 'sms' in channels:
            tasks.append(self._send_sms_alert(alert))
        if 'slack' in channels:
            tasks.append(self._send_slack_alert(alert))

        await asyncio.gather(*tasks)

    async def _send_email_alert(self, alert: Dict):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.get('alert_email_from')
            msg['To'] = ', '.join(self.config.get('alert_email_recipients', []))
            msg['Subject'] = f"[{alert['level'].upper()}] ABC Application Alert: {alert['service']}"

            body = f"""
            ABC Application System Alert

            Level: {alert['level'].upper()}
            Service: {alert['service']}
            Type: {alert['type']}
            Time: {alert['timestamp']}

            Message: {alert['message']}

            Details:
            {json.dumps(alert['details'], indent=2)}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.config.get('smtp_server', 'localhost'))
            server.send_message(msg)
            server.quit()

        except Exception as e:
            print(f"Failed to send email alert: {e}")

    async def _send_sms_alert(self, alert: Dict):
        """Send SMS alert (placeholder)"""
        # Implementation would integrate with Twilio or similar service
        print(f"SMS Alert: {alert['message']}")

    async def _send_slack_alert(self, alert: Dict):
        """Send Slack alert (placeholder)"""
        # Implementation would integrate with Slack API
        print(f"Slack Alert: {alert['message']}")

    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get recent alert history"""
        cutoff_time = asyncio.get_event_loop().time() - (hours * 3600)

        return [
            alert for alert in self.alert_history
            if alert['timestamp'] > cutoff_time
        ]
```

### Monitoring Dashboard

```python
# src/monitoring/dashboard.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
from datetime import datetime, timedelta
from .health_checker import HealthChecker
from .alert_system import AlertSystem

app = FastAPI(title="ABC Application Health Dashboard")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class MonitoringDashboard:
    def __init__(self, health_checker: HealthChecker, alert_system: AlertSystem):
        self.health_checker = health_checker
        self.alert_system = alert_system
        self.metrics_history = []
        self.setup_routes()

    def setup_routes(self):
        @app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard view"""
            health_results = await self.health_checker.check_all()
            recent_alerts = self.alert_system.get_alert_history(hours=24)

            # Calculate uptime statistics
            uptime_stats = self._calculate_uptime_stats()

            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "health_results": health_results,
                "recent_alerts": recent_alerts,
                "uptime_stats": uptime_stats,
                "last_update": datetime.now()
            })

        @app.get("/api/health")
        async def get_health_status():
            """API endpoint for health status"""
            results = await self.health_checker.check_all()
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_health": results['overall'].status.value,
                "services": {
                    name: {
                        "status": result.status.value,
                        "response_time": result.response_time_ms,
                        "message": result.message,
                        "last_check": datetime.fromtimestamp(result.timestamp).isoformat()
                    }
                    for name, result in results.items()
                }
            }

        @app.get("/api/metrics")
        async def get_metrics():
            """API endpoint for metrics data"""
            return {
                "timestamp": datetime.now().isoformat(),
                "metrics": self._get_current_metrics(),
                "history": self.metrics_history[-100:]  # Last 100 data points
            }

        @app.get("/api/alerts")
        async def get_alerts(hours: int = 24):
            """API endpoint for alerts"""
            alerts = self.alert_system.get_alert_history(hours=hours)
            return {
                "timestamp": datetime.now().isoformat(),
                "alerts": alerts,
                "summary": {
                    "total": len(alerts),
                    "critical": len([a for a in alerts if a['level'] == 'critical']),
                    "warning": len([a for a in alerts if a['level'] == 'warning'])
                }
            }

    def _calculate_uptime_stats(self) -> Dict:
        """Calculate uptime statistics"""
        # This would analyze historical health data
        return {
            "overall_uptime": 0.987,  # 98.7%
            "last_24h_uptime": 0.995,  # 99.5%
            "average_response_time": 245.5,  # ms
            "total_incidents": 3,
            "mttr": 12.5  # minutes
        }

    def _get_current_metrics(self) -> Dict:
        """Get current system metrics"""
        # This would collect real metrics
        metrics = {
            "cpu_usage": 45.2,
            "memory_usage": 62.8,
            "disk_usage": 34.1,
            "network_in": 125.5,  # Mbps
            "network_out": 89.3,  # Mbps
            "active_connections": 156,
            "queue_depth": 12,
            "error_rate": 0.023
        }

        # Store in history
        self.metrics_history.append({
            "timestamp": datetime.now().isoformat(),
            **metrics
        })

        # Keep only last 24 hours of data
        cutoff = datetime.now() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]

        return metrics

# HTML Template (dashboard.html)
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>ABC Application Health Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .status-healthy { color: green; }
        .status-degraded { color: orange; }
        .status-unhealthy { color: red; }
        .status-unknown { color: gray; }
        .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; display: inline-block; min-width: 200px; }
        .alert-critical { background: #ffe6e6; border-left: 4px solid red; }
        .alert-warning { background: #fff3cd; border-left: 4px solid orange; }
    </style>
</head>
<body>
    <h1>ABC Application System Health Dashboard</h1>

    <div class="metric-card">
        <h3>Overall System Health</h3>
        <div class="status-{{ 'healthy' if health_results.overall.is_healthy else 'unhealthy' }}">
            {{ health_results.overall.status.value.upper() }}
        </div>
        <p>{{ health_results.overall.message }}</p>
    </div>

    <div class="metric-card">
        <h3>Uptime Statistics</h3>
        <p>Overall: {{ "%.1f"|format(uptime_stats.overall_uptime * 100) }}%</p>
        <p>Last 24h: {{ "%.1f"|format(uptime_stats.last_24h_uptime * 100) }}%</p>
        <p>Avg Response Time: {{ "%.1f"|format(uptime_stats.average_response_time) }}ms</p>
    </div>

    <h2>Service Status</h2>
    {% for name, result in health_results.items() %}
        {% if name != 'overall' %}
        <div class="metric-card">
            <h4>{{ name.replace('_', ' ').title() }}</h4>
            <div class="status-{{ result.status.value }}">
                {{ result.status.value.upper() }}
            </div>
            <p>{{ result.message }}</p>
            <small>Response: {{ "%.0f"|format(result.response_time_ms) }}ms</small>
        </div>
        {% endif %}
    {% endfor %}

    <h2>Recent Alerts</h2>
    {% for alert in recent_alerts %}
    <div class="metric-card alert-{{ alert.level }}">
        <h4>{{ alert.service }} - {{ alert.level.upper() }}</h4>
        <p>{{ alert.message }}</p>
        <small>{{ alert.timestamp | strftime('%Y-%m-%d %H:%M:%S') }}</small>
    </div>
    {% endfor %}

    <script>
        // Real-time updates every 30 seconds
        setInterval(async () => {
            const response = await fetch('/api/health');
            const data = await response.json();

            // Update dashboard with new data
            location.reload();
        }, 30000);
    </script>
</body>
</html>
"""
```

### Health Check Configuration

```yaml
# config/health_monitoring.yaml
health_monitoring:
  # Health check intervals (seconds)
  check_intervals:
    data_sources: 60
    agents: 30
    database: 30
    redis: 30
    external_apis: 300  # 5 minutes

  # Alert thresholds
  alerts:
    response_time_warning_ms: 5000
    response_time_critical_ms: 30000
    error_rate_warning: 0.05
    error_rate_critical: 0.20
    max_alerts_per_hour: 10

  # Alert channels
  notification_channels:
    email:
      enabled: true
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      from_address: "alerts@abc-application.com"
      recipients: ["admin@abc-application.com", "devops@abc-application.com"]
    slack:
      enabled: true
      webhook_url: "${SLACK_WEBHOOK_URL}"
      channel: "#alerts"
    sms:
      enabled: false  # Enable for critical alerts only
      provider: "twilio"
      account_sid: "${TWILIO_ACCOUNT_SID}"
      auth_token: "${TWILIO_AUTH_TOKEN}"
      from_number: "${TWILIO_FROM_NUMBER}"
      recipients: ["+1234567890"]

  # Dashboard configuration
  dashboard:
    enabled: true
    host: "0.0.0.0"
    port: 8080
    update_interval_seconds: 30

  # Health check timeouts
  timeouts:
    api_call_timeout_seconds: 10
    database_timeout_seconds: 5
    redis_timeout_seconds: 5
    agent_timeout_seconds: 15

  # Service-specific configurations
  services:
    ibkr_api:
      endpoints:
        - "https://api.ibkr.com/health"
        - "https://gdcdyn.interactivebrokers.com/health"
      expected_response_codes: [200, 201]
      required_fields: ["status", "version"]

    alpha_vantage:
      rate_limit_buffer: 0.9  # Use 90% of limit
      test_symbol: "IBM"
      required_fields: ["Global Quote"]

    database:
      connection_pool_min: 5
      connection_pool_max: 20
      query_timeout_seconds: 30

    redis:
      max_memory_percent: 80
      min_available_connections: 5
```

### Health Check Integration

```python
# src/main.py
from src.monitoring.health_checker import HealthChecker
from src.monitoring.data_source_health import DataSourceHealthChecker
from src.monitoring.agent_health import AgentHealthChecker
from src.monitoring.alert_system import AlertSystem
from src.monitoring.dashboard import MonitoringDashboard

async def initialize_health_monitoring(config, agent_manager):
    """Initialize comprehensive health monitoring system"""

    # Create health checker
    health_checker = HealthChecker()

    # Initialize service-specific checkers
    data_checker = DataSourceHealthChecker(config)
    agent_checker = AgentHealthChecker(agent_manager)

    # Register health checks
    health_checker.register_check("ibkr_api", data_checker.check_ibkr_api, 60)
    health_checker.register_check("alpha_vantage_api", data_checker.check_alpha_vantage_api, 300)
    health_checker.register_check("database", data_checker.check_database_connection, 30)
    health_checker.register_check("redis", data_checker.check_redis_connection, 30)
    health_checker.register_check("news_api", data_checker.check_news_api, 300)
    health_checker.register_check("twitter_api", data_checker.check_twitter_api, 300)

    health_checker.register_check("data_agent", agent_checker.check_data_agent_health, 30)
    health_checker.register_check("strategy_agent", agent_checker.check_strategy_agent_health, 30)
    health_checker.register_check("execution_agent", agent_checker.check_execution_agent_health, 30)
    health_checker.register_check("risk_agent", agent_checker.check_risk_agent_health, 30)

    # Initialize alert system
    alert_system = AlertSystem(config)

    # Initialize dashboard
    dashboard = MonitoringDashboard(health_checker, alert_system)

    # Start background monitoring
    asyncio.create_task(run_health_monitoring_loop(health_checker, alert_system))

    return health_checker, alert_system, dashboard

async def run_health_monitoring_loop(health_checker, alert_system):
    """Background health monitoring loop"""
    while True:
        try:
            # Run health checks
            results = await health_checker.run_health_checks()

            # Process alerts
            alerts = await alert_system.process_health_results(results)

            # Log summary
            healthy_count = sum(1 for r in results.values() if r.is_healthy)
            total_count = len(results)

            print(f"Health check complete: {healthy_count}/{total_count} services healthy")

            if alerts:
                print(f"Generated {len(alerts)} alerts")

        except Exception as e:
            print(f"Health monitoring error: {e}")

        # Wait for next check cycle
        await asyncio.sleep(60)  # Check every minute
```

This comprehensive API health monitoring system ensures the ABC Application platform maintains high availability, performance, and reliability through proactive monitoring, alerting, and automated recovery mechanisms.

---

*For implementation details, see IMPLEMENTATION/setup-and-development.md. For testing procedures, see IMPLEMENTATION/testing.md.*