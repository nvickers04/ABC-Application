# Health API Documentation

## Overview

The ABC-Application Health Check API provides comprehensive monitoring endpoints for system health, component status, and performance metrics. Built with FastAPI, this API offers both human-readable and machine-readable health information for operational monitoring and automated health checks.

## Base URL
```
http://localhost:8080
```

## Authentication
No authentication required for health check endpoints.

## Endpoints

### GET /health

**Basic Health Check Endpoint**

Returns overall system health status with summary information.

**Response Model:**
```json
{
  "status": "healthy" | "degraded" | "unhealthy",
  "timestamp": "2025-12-04T12:00:00.000000",
  "uptime_seconds": 3600.0,
  "version": "1.0.0",
  "checks": {
    "components": {...},
    "api_endpoints": {...},
    "system": {...}
  }
}
```

**Status Codes:**
- `200` - System is healthy or degraded
- `503` - System is unhealthy

**Example Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-04T12:00:00.000000",
  "uptime_seconds": 3600.0,
  "version": "1.0.0",
  "checks": {
    "components": {
      "ibkr_bridge": {
        "status": "healthy",
        "last_check": "2025-12-04T12:00:00.000000",
        "response_time": 0.123,
        "error_message": null,
        "metrics": {}
      }
    },
    "api_endpoints": {...},
    "system": {
      "cpu_percent": 15.2,
      "memory_percent": 45.8,
      "memory_used_gb": 7.3,
      "memory_total_gb": 16.0,
      "disk_usage_percent": 65.4,
      "process_count": 124
    }
  }
}
```

### GET /health/components

**Component Health Check**

Provides detailed health status for all system components.

**Response:**
```json
{
  "components": {
    "component_name": {
      "status": "healthy" | "degraded" | "unhealthy",
      "last_check": "ISO8601 timestamp",
      "response_time": 0.123,
      "error_message": "string or null",
      "metrics": {}
    }
  }
}
```

**Status Codes:**
- `200` - Components retrieved successfully
- `503` - Component health check failed

### GET /health/api

**API Endpoints Health Check**

Monitors health of external API endpoints used by the system.

**Response:**
```json
{
  "api_endpoints": {
    "endpoint_name": {
      "status": "healthy" | "degraded" | "unhealthy",
      "last_check": "ISO8601 timestamp",
      "response_time": 0.123,
      "error_message": "string or null"
    }
  }
}
```

### GET /health/system

**System Metrics Health Check**

Returns current system resource usage metrics.

**Response:**
```json
{
  "system": {
    "cpu_percent": 15.2,
    "memory_percent": 45.8,
    "memory_used_gb": 7.3,
    "memory_total_gb": 16.0,
    "disk_usage_percent": 65.4,
    "process_count": 124
  }
}
```

### GET /health/ready

**Readiness Probe**

Kubernetes-style readiness check for container orchestration.

**Response:**
```json
{
  "status": "ready"
}
```

**Status Codes:**
- `200` - System is ready
- `503` - System is not ready

### GET /health/live

**Liveness Probe**

Kubernetes-style liveness check for container orchestration.

**Response:**
```json
{
  "status": "alive"
}
```

**Status Codes:**
- `200` - Application is alive

### GET /metrics

**Prometheus Metrics Endpoint**

Exposes system and component metrics in Prometheus format.

**Response:** Plain text in Prometheus exposition format

**Example:**
```
# HELP abc_app_cpu_percent CPU usage percentage
# TYPE abc_app_cpu_percent gauge
abc_app_cpu_percent 15.2

# HELP abc_app_memory_percent Memory usage percentage
# TYPE abc_app_memory_percent gauge
abc_app_memory_percent 45.8

# HELP abc_app_component_health Component health status (0=healthy, 1=degraded, 2=unhealthy)
# TYPE abc_app_component_health gauge
abc_app_component_health{component="ibkr_bridge"} 0
```

## Health Status Definitions

### Overall System Status
- **healthy**: All critical components operational, system resources within normal limits
- **degraded**: Some non-critical components unhealthy, or system resources elevated but manageable
- **unhealthy**: Critical components failed, or system resources critically high

### Component Status
- **healthy**: Component responding normally within acceptable time limits
- **degraded**: Component responding but with elevated response times or warnings
- **unhealthy**: Component not responding or returning errors

## Critical Components

The following components are considered critical for system operation:
- `ibkr_bridge`: Interactive Brokers API connection
- `redis`: Data caching and persistence
- `alert_manager`: Alert notification system

## Monitoring Integration

### Prometheus
Metrics endpoint (`/metrics`) provides Prometheus-compatible metrics for monitoring dashboards.

### Kubernetes
Readiness (`/health/ready`) and liveness (`/health/live`) probes support container orchestration.

### Alerting
Health status can trigger alerts through the integrated alert manager system.

## Usage Examples

### Basic Health Check
```bash
curl http://localhost:8080/health
```

### Component Status
```bash
curl http://localhost:8080/health/components
```

### System Metrics
```bash
curl http://localhost:8080/health/system
```

### Prometheus Scraping
```bash
curl http://localhost:8080/metrics
```

## Interactive Documentation

Full interactive API documentation is available at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## Configuration

The health server can be configured with command-line arguments:
```bash
python tools/health_server.py --host 0.0.0.0 --port 8080 --reload
```

## Dependencies

- FastAPI
- Uvicorn
- psutil
- pydantic</content>
</xai:function_call name="manage_todo_list">
<parameter name="todoList">[{"id":1,"title":"Update API documentation for all components","status":"completed"}]