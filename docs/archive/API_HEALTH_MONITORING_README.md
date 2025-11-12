# API Health Monitoring System

## Overview

The ABC Application system includes comprehensive API health monitoring to ensure reliable data availability and system resilience. The monitoring system tracks response times, success rates, error rates, and manages circuit breaker status for all integrated APIs.

## Features

- **Real-time Monitoring**: Continuous health checks every 5 minutes (configurable)
- **Circuit Breaker Integration**: Automatic failover protection for failing APIs
- **Comprehensive Metrics**: Response times, success rates, error counts, circuit breaker status
- **Alert System**: Logging and status tracking for degraded APIs
- **Dashboard**: Visual health status display
- **Persistence**: Health metrics saved to JSON file for analysis

## Monitored APIs

The system monitors the following APIs:

1. **Massive API** - Premium market data
2. **Kalshi API** - Prediction market data
3. **yFinance** - Free stock data
4. **NewsAPI** - Financial news
5. **FRED Economic Data** - Economic indicators
6. **Currents API** - Real-time news
7. **Twitter API** - Social sentiment
8. **Whale Wisdom** - Institutional holdings
9. **Grok API** - AI sentiment analysis

## Usage

### Starting Health Monitoring

The health monitoring starts automatically when the main application runs:

```python
from src.main import main_loop
import asyncio

# Health monitoring starts automatically
results = asyncio.run(main_loop())
```

### Manual Health Checks

```python
from utils.api_health_monitor import check_api_health_now, get_api_health_summary

# Perform immediate health check
result = check_api_health_now()
print(result['summary'])

# Get current health status
status = get_api_health_summary()
print(status['overall_status'])
```

### Health Dashboard

Run the interactive dashboard:

```bash
# From project root
python api_health_dashboard.py
```

Or perform a one-time check:

```bash
python api_health_dashboard.py --check-now
```

### Using the Health Tool

The health monitoring is available as a Langchain tool:

```python
from utils.api_health_tool import api_health_monitor_tool

# Get current status
status = api_health_monitor_tool.invoke({"action": "status"})

# Perform immediate check
result = api_health_monitor_tool.invoke({"action": "check_now"})

# Start/stop monitoring
api_health_monitor_tool.invoke({"action": "start_monitoring"})
```

## Health Status Levels

- **✅ HEALTHY**: API responding normally, success rate > 95%
- **⚠️ DEGRADED**: API experiencing issues, success rate 80-95% or recent errors
- **❌ UNHEALTHY**: API failing consistently, success rate < 80% or circuit breaker open

## Circuit Breaker States

- **🟢 CLOSED**: Normal operation
- **🟡 HALF_OPEN**: Testing recovery
- **🔴 OPEN**: API blocked due to failures

## Configuration

### Check Interval

Default check interval is 300 seconds (5 minutes). Modify in code:

```python
from utils.api_health_monitor import APIHealthMonitor

monitor = APIHealthMonitor(check_interval=600)  # 10 minutes
```

### Circuit Breaker Settings

Circuit breaker thresholds are configured per API in the tools.py file:

```python
@circuit_breaker("api_name", failure_threshold=3, recovery_timeout=300)
def api_function():
    # API code here
```

## Metrics Storage

Health metrics are automatically saved to `api_health_metrics.json` for analysis and persistence across restarts.

## Troubleshooting

### Common Issues

1. **API Key Missing**: Ensure all required API keys are in `.env` file
2. **Network Issues**: Check internet connectivity
3. **Rate Limits**: Some APIs have request limits
4. **Circuit Breaker Open**: Wait for recovery timeout or restart

### Logs

Check logs for detailed error information:
- Health check results
- API failures and errors
- Circuit breaker state changes

## Integration

The health monitoring integrates with:

- **Circuit Breakers**: Automatic protection against failing APIs
- **Fallback Systems**: Graceful degradation to backup data sources
- **Alert Systems**: Logging and status reporting
- **Dashboard**: Visual monitoring interface

## API Key Requirements

Ensure these API keys are configured in your `.env` file:

```
MASSIVE_API_KEY=your_key
KALSHI_API_KEY=your_key
KALSHI_ACCESS_KEY_ID=your_key
NEWS_API_KEY=your_key
FRED_API_KEY=your_key
CURRENTS_API_KEY=your_key
TWITTER_BEARER_TOKEN=your_key
WHALE_WISDOM_API_KEY=your_key
GROK_API_KEY=your_key
```

## Performance Impact

The health monitoring system has minimal performance impact:
- Checks run in background thread
- Lightweight API calls (simple health checks)
- 5-minute intervals minimize resource usage
- Metrics stored efficiently in memory and JSON