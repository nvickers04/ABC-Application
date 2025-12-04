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

### Discord Commands

The AlertManager provides several Discord commands for monitoring and testing:

- **`!alert_test`**: Send a test alert to verify the system is working
- **`!alert_history`**: Show the last 10 alerts with details
- **`!alert_stats`**: Display alert statistics and trends
- **`!alert_dashboard`**: Show comprehensive monitoring dashboard with performance metrics

Example usage:
```
!alert_dashboard
```
Shows system health, performance indicators, alert distribution, and recommendations.

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

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily Monitoring
1. Check `!alert_dashboard` for system health and performance metrics
2. Review `!alert_stats` for alert trends and patterns
3. Monitor alert queue size (should not exceed 1000 alerts)

#### Weekly Maintenance
1. Review escalation history for patterns requiring policy updates
2. Check alert metrics for false positive rates > 5%
3. Verify Discord notification delivery
4. Clean up old alert logs if needed

#### Monthly Maintenance
1. Review and update escalation policies based on incident response
2. Audit alert suppression rules for effectiveness
3. Update notification filters based on team feedback
4. Check alert performance metrics and optimize if needed

### Configuration Updates

#### Updating Escalation Policies
```python
# Access alert manager
alert_manager = get_alert_manager()

# Update escalation policies
alert_manager.escalation_policies['default']['levels'][0]['delay_minutes'] = 2

# Reload policies (if dynamic reloading implemented)
alert_manager._load_escalation_policies()
```

#### Modifying Notification Filters
```python
# Update component filters
alert_manager.notification_filters['component_filters']['new_component'] = {
    'min_level': 'warning',
    'rate_limit': 5
}
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Alerts Not Appearing in Discord
**Symptoms**: Alerts logged but not sent to Discord channels
**Causes**:
- Orchestrator not connected to AlertManager
- Discord bot not initialized
- Wrong channel configuration
- Bot permissions issues

**Solutions**:
```bash
# Check orchestrator connection
python -c "
from src.utils.alert_manager import get_alert_manager
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator
alert_manager = get_alert_manager()
print(f'Orchestrator connected: {alert_manager.orchestrator is not None}')
"

# Verify Discord channel access
# Check DISCORD_ALERTS_CHANNEL environment variable
# Ensure bot has send_messages permission in alerts channel
```

#### 2. Alert Queue Growing Too Large
**Symptoms**: Alert queue exceeds 1000 items, memory usage high
**Causes**:
- High error rate in monitored components
- Alert suppression rules not working
- Escalation policies causing alert storms

**Solutions**:
```python
# Clear alert queue
alert_manager = get_alert_manager()
alert_manager.clear_alerts()

# Review and update suppression rules
# Check for alert storms from specific components
recent_alerts = alert_manager.get_recent_alerts(100)
component_counts = {}
for alert in recent_alerts:
    component_counts[alert.component] = component_counts.get(alert.component, 0) + 1
print("Alert counts by component:", component_counts)
```

#### 3. False Positive Alerts
**Symptoms**: Too many non-critical alerts, alert fatigue
**Causes**:
- Low alert thresholds
- Missing suppression rules
- Component misconfiguration

**Solutions**:
1. Review alert levels in component integrations
2. Add suppression rules for known benign errors
3. Update component-specific filters

#### 4. Escalation Not Working
**Symptoms**: Critical alerts not escalating to email/SMS
**Causes**:
- Email/SMS not configured
- Escalation conditions not met
- External service failures

**Solutions**:
```bash
# Check escalation configuration
alert_manager = get_alert_manager()
print("Escalation policies:", alert_manager.escalation_policies)

# Verify external service configuration
print(f"Email enabled: {alert_manager.email_enabled}")
print(f"Slack enabled: {alert_manager.slack_enabled}")
```

#### 5. High Response Times
**Symptoms**: Alert processing taking >1 second
**Causes**:
- Large alert queue
- Complex escalation logic
- External service delays

**Solutions**:
1. Monitor `!alert_dashboard` performance indicators
2. Reduce alert queue size
3. Optimize escalation conditions
4. Check external service response times

### Performance Tuning

#### Optimizing Alert Processing
- Keep alert queue size < 500 for best performance
- Use component-specific filters to reduce noise
- Limit escalation history to last 50 events
- Configure appropriate rate limits per component

#### Memory Management
- Alert objects are lightweight but accumulate in queue
- Set appropriate `max_queue_size` (default: 1000)
- Clear old alerts periodically
- Monitor memory usage in dashboard

### Emergency Procedures

#### Alert System Failure
1. **Immediate**: Check system logs for AlertManager errors
2. **Fallback**: Use direct logging if AlertManager fails
3. **Recovery**: Restart orchestrator to reconnect AlertManager
4. **Prevention**: Add health checks for AlertManager itself

#### Alert Storm Response
1. **Identify**: Use `!alert_dashboard` to find high-frequency component
2. **Suppress**: Add temporary suppression rule
3. **Investigate**: Check component logs for root cause
4. **Resolve**: Fix underlying issue and remove suppression

## Future Enhancements

- Alert escalation policies (email after repeated failures)
- Alert correlation and deduplication
- Alert routing based on time/severity rules
- Integration with external monitoring systems (DataDog, New Relic)