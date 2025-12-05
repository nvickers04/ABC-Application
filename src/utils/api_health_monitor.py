"""
API Health Monitoring System for ABC Application

This module provides ongoing monitoring of all API endpoints used in the system,
tracking health metrics, response times, success rates, and managing circuit breaker status.
"""

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

import pandas as pd
import requests

# Import centralized configuration
from src.utils.config import get_api_key, get_grok_api_key

# Logging configured centrally in logging_config.py
logger = logging.getLogger(__name__)

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

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['last_check'] = self.last_check.isoformat()
        return data

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

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_all_apis()
                self._log_health_summary()
                self._save_metrics_to_file()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.check_interval)

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

    def _check_marketdataapp_api(self) -> bool:
        """Check MarketDataApp API health"""
        api_key = get_api_key('marketdataapp')
        if not api_key:
            return False

        # Use correct MarketDataApp endpoint for quotes
        url = "https://api.marketdata.app/v1/stocks/quotes/AAPL/"
        params = {"token": api_key}

        try:
            response = requests.get(url, params=params, timeout=10)
            # MarketDataApp returns 203 for success, check for 's': 'ok' in response
            if response.status_code in [200, 203]:
                data = response.json()
                return data.get('s') == 'ok'
            return False
        except:
            return False

    def _check_kalshi_api(self) -> bool:
        """Check Kalshi API health"""
        api_key = get_api_key('kalshi')
        access_key_id = get_api_key('kalshi', 'KALSHI_ACCESS_KEY_ID')
        if not api_key or not access_key_id:
            return False

        # Try to get market list
        url = "https://api.elections.kalshi.com/trade-api/v2/markets"
        headers = {
            "KALSHI-ACCESS-KEY": api_key,
            "KALSHI-ACCESS-KEY-ID": access_key_id
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            return response.status_code == 200
        except:
            return False

    def _check_yfinance(self) -> bool:
        """Check yfinance health"""
        try:
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            # Try to get basic info
            return ticker.info is not None
        except:
            return False

    def _check_news_api(self) -> bool:
        """Check NewsAPI health"""
        api_key = get_api_key('news')
        if not api_key:
            return False

        url = "https://newsapi.org/v2/top-headlines"
        params = {"country": "us", "apiKey": api_key}

        try:
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except:
            return False

    def _check_economic_data(self) -> bool:
        """Check FRED API health"""
        api_key = get_api_key('fred')
        if not api_key:
            return False

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "UNRATE",
            "api_key": api_key,
            "file_type": "json",
            "limit": 1
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except:
            return False

    def _check_currents_news(self) -> bool:
        """Check Currents API health"""
        api_key = get_api_key('currents')
        if not api_key:
            return False

        url = "https://api.currentsapi.services/v1/latest-news"
        params = {"apiKey": api_key, "language": "en"}

        try:
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except:
            return False

    def _check_twitter_api(self) -> bool:
        """Check Twitter API health"""
        bearer_token = get_api_key('twitter', 'TWITTER_BEARER_TOKEN')
        if not bearer_token:
            return False

        try:
            import tweepy
            client = tweepy.Client(bearer_token=bearer_token)
            # Try a simple search
            tweets = client.search_recent_tweets(query="test", max_results=1)
            return tweets is not None
        except:
            return False

    def _check_whale_wisdom(self) -> bool:
        """Check Whale Wisdom API health"""
        api_key = get_api_key('whale_wisdom')
        if not api_key:
            return False

        url = "https://whalewisdom.com/api/v2/institutional-holdings"
        params = {"cik": "0001067983", "limit": 1, "api_key": api_key}

        try:
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except:
            return False

    def _check_grok_api(self) -> bool:
        """Check Grok API health"""
        api_key = get_grok_api_key()
        if not api_key:
            return False

        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "grok-4-fast-reasoning",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            return response.status_code == 200
        except:
            return False

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        total_apis = len(self.metrics)
        healthy_count = sum(1 for m in self.metrics.values() if m.status == APIStatus.HEALTHY)
        degraded_count = sum(1 for m in self.metrics.values() if m.status == APIStatus.DEGRADED)
        unhealthy_count = sum(1 for m in self.metrics.values() if m.status == APIStatus.UNHEALTHY)

        overall_status = APIStatus.HEALTHY
        if unhealthy_count > 0:
            overall_status = APIStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = APIStatus.DEGRADED

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status.value,
            "summary": {
                "total_apis": total_apis,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            },
            "api_details": {name: metrics.to_dict() for name, metrics in self.metrics.items()}
        }

    def _log_health_summary(self):
        """Log health summary"""
        summary = self.get_health_summary()
        status_counts = summary["summary"]

        logger.info(f"API Health Summary: {status_counts['healthy']} healthy, "
                   f"{status_counts['degraded']} degraded, {status_counts['unhealthy']} unhealthy")

        # Log unhealthy APIs
        for api_name, metrics in self.metrics.items():
            if metrics.status == APIStatus.UNHEALTHY:
                logger.warning(f"UNHEALTHY API: {api_name} - {metrics.last_error}")

    def _save_metrics_to_file(self):
        """Save metrics to JSON file for persistence"""
        try:
            summary = self.get_health_summary()
            filename = "data/api_health_metrics.json"

            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def get_api_status(self, api_name: str) -> Optional[Dict[str, Any]]:
        """Get status for specific API"""
        if api_name in self.metrics:
            return self.metrics[api_name].to_dict()
        return None

# Global monitor instance
_monitor: Optional[APIHealthMonitor] = None

def get_health_monitor() -> APIHealthMonitor:
    """Get the global health monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = APIHealthMonitor()
    return _monitor

def start_health_monitoring(check_interval: int = 300):
    """Start API health monitoring"""
    monitor = get_health_monitor()
    monitor.start_monitoring()

def stop_health_monitoring():
    """Stop API health monitoring"""
    global _monitor
    if _monitor:
        _monitor.stop_monitoring()
        _monitor = None

def get_api_health_summary() -> Dict[str, Any]:
    """Get current API health summary"""
    monitor = get_health_monitor()
    return monitor.get_health_summary()

def check_api_health_now():
    """Manually trigger health check for all APIs"""
    monitor = get_health_monitor()
    monitor._check_all_apis()
    return monitor.get_health_summary()

if __name__ == "__main__":
    # Example usage
    monitor = APIHealthMonitor(check_interval=60)  # Check every minute for testing
    monitor.start_monitoring()

    try:
        # Keep running for a bit
        time.sleep(120)  # 2 minutes
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop_monitoring()