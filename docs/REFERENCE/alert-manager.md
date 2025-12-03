# AlertManager Documentation

## Overview

The AlertManager is a centralized alerting system for the ABC-Application that provides unified error handling, logging, and notifications across all components. It integrates with Discord for real-time notifications and triggers health checks on critical errors.

## Features

- **Centralized Error Handling**: Unified logging with structured context across all components
- **Discord Notifications**: Real-time alerts sent to Discord channels via orchestrator integration
- **Health Check Integration**: Automatic health checks triggered on critical errors
- **Multiple Alert Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL severity levels
- **Fail-Fast Behavior**: Critical errors can abort operations using HealthAlertError
- **Error Queue**: Maintains history of alerts for monitoring and analysis

## Architecture

### Core Components

#### AlertManager Class
Singleton implementation ensuring consistent alerting across the application.

#### Alert Data Structure
```python
@dataclass
class Alert:
    level: AlertLevel
    message: str
    context: Dict[str, Any]
    timestamp: datetime
    component: str
    error_id: Optional[str]
```

#### HealthAlertError
Exception raised for critical health alerts that should abort operations.

### Alert Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General information about system operations
- **WARNING**: Warning conditions that don't prevent operation
- **ERROR**: Error conditions that may affect functionality
- **CRITICAL**: Critical errors that require immediate attention

## Usage

### Basic Alerting

```python
from src.utils.alert_manager import get_alert_manager

alert_manager = get_alert_manager()

# Error alert
await alert_manager.error(
    Exception("Database connection failed"),
    {"database": "redis", "host": "localhost"},
    "data_persistence"
)

# Warning alert
await alert_manager.warning(
    "High memory usage detected",
    {"memory_percent": 85, "component": "data_analyzer"},
    "system_monitoring"
)

# Info alert
await alert_manager.info(
    "System startup completed",
    {"startup_time": "2025-12-02T10:30:00Z"},
    "system"
)
```

### Critical Error Handling

```python
try:
    # Risky operation
    result = await high_risk_operation()
except Exception as e:
    # This will trigger health check and may abort operation
    alert_manager.critical(e, {"operation": "high_risk"}, "trading_engine")
```

### HealthAlertError for Fail-Fast

```python
from src.utils.alert_manager import HealthAlertError

try:
    # Operation that should fail fast on critical issues
    await critical_operation()
except HealthAlertError as e:
    # Operation aborted due to health alert
    logger.error(f"Operation aborted: {e.alert.message}")
    return {"error": "System health check failed", "aborted": True}
```

## Integration Points

### Discord Notifications

The AlertManager integrates with the LiveWorkflowOrchestrator for Discord notifications:

- Alerts are sent to configured Discord channels
- Different alert levels can be routed to different channels
- Rich formatting with context information

### Health Check Integration

Critical alerts automatically trigger system health checks:

```python
# Critical alert triggers health check
alert_manager.critical(error, context, component)
# -> Automatically calls orchestrator.perform_system_health_check()
```

### Component Integration

All major components integrate with AlertManager:

- **IBKR Operations**: Alerts on connection failures, order placement issues
- **TigerBeetle**: Alerts on transaction logging failures
- **Data Analyzers**: Alerts on API failures, data quality issues
- **Discord Operations**: Alerts on message sending failures

## Configuration

### Environment Variables

```bash
# Discord integration
DISCORD_ALERTS_CHANNEL_ID=123456789012345678
DISCORD_ALERTS_CHANNEL=alerts

# Alert queue settings
ALERT_QUEUE_MAX_SIZE=1000

# Email integration (optional)
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_SMTP=smtp.gmail.com
ALERT_EMAIL_RECIPIENTS=admin@example.com,dev@example.com
```

### Runtime Configuration

```python
alert_manager = get_alert_manager()
alert_manager.discord_alerts_channel = "custom-alerts"
alert_manager.max_queue_size = 2000
```

## Monitoring and Troubleshooting

### Alert Queue Inspection

```python
alert_manager = get_alert_manager()

# Get recent alerts
recent_alerts = alert_manager.error_queue[-10:]

# Filter by level
critical_alerts = [a for a in alert_manager.error_queue if a.level == AlertLevel.CRITICAL]

# Filter by component
ibkr_alerts = [a for a in alert_manager.error_queue if a.component == "ibkr_integration"]
```

### Discord Notification Issues

**Problem**: Alerts not appearing in Discord
**Solutions**:
1. Verify orchestrator is set: `alert_manager.set_orchestrator(orchestrator)`
2. Check Discord bot permissions in the alerts channel
3. Verify `DISCORD_ALERTS_CHANNEL_ID` is set correctly
4. Check bot is online and connected

**Problem**: Too many alerts flooding Discord
**Solutions**:
1. Adjust alert levels (reduce INFO/DEBUG alerts in production)
2. Implement alert throttling/rate limiting
3. Use different channels for different alert types

### Health Check Integration Issues

**Problem**: Health checks not triggering on critical alerts
**Solutions**:
1. Ensure orchestrator has `perform_system_health_check` method
2. Verify orchestrator reference is set: `alert_manager.set_orchestrator(orchestrator)`
3. Check async task creation is working

## Best Practices

### Alert Design

1. **Use Appropriate Levels**: Don't use CRITICAL for routine issues
2. **Provide Context**: Include relevant data in alert context
3. **Be Specific**: Use descriptive component names and messages
4. **Avoid Alert Fatigue**: Don't alert on expected failures

### Error Handling

```python
# Good: Specific alerts with context
try:
    await ibkr_connector.place_order(symbol, quantity, action)
except Exception as e:
    await alert_manager.error(
        Exception(f"IBKR order failed for {symbol}"),
        {"symbol": symbol, "quantity": quantity, "action": action},
        "ibkr_integration"
    )

# Bad: Generic alerts without context
try:
    await ibkr_connector.place_order(symbol, quantity, action)
except Exception as e:
    await alert_manager.error(e, {}, "unknown")
```

### Testing Alerts

```python
# Test alert functionality
await alert_manager.warning("Test alert", {"test": True}, "testing")

# Verify Discord notifications are working
# Check alert queue: len(alert_manager.error_queue) > 0
```

## Troubleshooting Procedures

### Common Issues

#### 1. Alerts Not Sending to Discord
```bash
# Check orchestrator integration
python -c "
from src.utils.alert_manager import get_alert_manager
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator
alert_manager = get_alert_manager()
orchestrator = LiveWorkflowOrchestrator()
alert_manager.set_orchestrator(orchestrator)
print('Orchestrator set successfully')
"
```

#### 2. Alert Queue Growing Too Large
```python
# Clear old alerts
alert_manager.error_queue = alert_manager.error_queue[-500:]  # Keep last 500
```

#### 3. Critical Alerts Not Triggering Health Checks
```python
# Verify orchestrator has health check method
hasattr(orchestrator, 'perform_system_health_check')  # Should return True
```

### Performance Considerations

- Alert processing is asynchronous to avoid blocking operations
- Alert queue has maximum size limit (default: 1000)
- Discord notifications are batched to avoid rate limits
- Context data is minimized to reduce memory usage

## Future Enhancements

- Alert escalation policies (email after repeated failures)
- Alert correlation and deduplication
- Metrics and analytics dashboard
- Alert routing based on time/severity rules
- Integration with external monitoring systems (DataDog, New Relic)