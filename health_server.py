#!/usr/bin/env python3
"""
Health Check HTTP Server for ABC-Application
Provides HTTP endpoints for health monitoring and component status checks.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils.component_health_monitor import get_component_health_monitor
from src.utils.alert_manager import get_alert_manager
from src.utils.api_health_monitor import get_api_health_summary

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ABC-Application Health Check API",
    description="Health monitoring endpoints for ABC-Application components",
    version="1.0.0"
)

class HealthResponse(BaseModel):
    """Standard health check response"""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"
    checks: Dict[str, Any] = {}

class ComponentHealthResponse(BaseModel):
    """Detailed component health response"""
    component: str
    status: str
    last_check: str
    response_time: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = {}

# Global variables
start_time = time.time()
component_monitor = None
alert_manager = None

def get_uptime() -> float:
    """Get application uptime in seconds"""
    return time.time() - start_time

def get_system_metrics() -> Dict[str, Any]:
    """Get basic system metrics"""
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)

    return {
        "cpu_percent": cpu,
        "memory_percent": mem.percent,
        "memory_used_gb": mem.used / (1024**3),
        "memory_total_gb": mem.total / (1024**3),
        "disk_usage_percent": psutil.disk_usage('/').percent,
        "process_count": len(psutil.pids())
    }

async def get_component_health() -> Dict[str, Any]:
    """Get health status of all components"""
    global component_monitor

    if not component_monitor:
        try:
            component_monitor = get_component_health_monitor()
        except Exception as e:
            logger.error(f"Failed to get component monitor: {e}")
            return {"error": f"Component monitor unavailable: {str(e)}"}

    try:
        # Perform health checks
        results = component_monitor.perform_health_checks()

        # Convert to serializable format
        health_data = {}
        for component_name, health in results.items():
            health_data[component_name] = {
                "status": health.status.value,
                "last_check": health.last_check.isoformat(),
                "response_time": health.response_time,
                "error_message": health.error_message,
                "metrics": health.metrics
            }

        return health_data

    except Exception as e:
        logger.error(f"Error getting component health: {e}")
        return {"error": f"Health check failed: {str(e)}"}

async def get_api_health() -> Dict[str, Any]:
    """Get API health status"""
    try:
        api_health = get_api_health_summary()
        return api_health
    except Exception as e:
        logger.error(f"Error getting API health: {e}")
        return {"error": f"API health check failed: {str(e)}"}

def determine_overall_status(component_health: Dict, api_health: Dict, system_metrics: Dict) -> str:
    """Determine overall system health status"""
    # Check for critical issues
    critical_components = ['ibkr_bridge', 'redis', 'alert_manager']
    for comp in critical_components:
        if comp in component_health and component_health[comp].get('status') == 'unhealthy':
            return 'unhealthy'

    # Check system resources
    if system_metrics.get('memory_percent', 0) > 95:
        return 'unhealthy'
    if system_metrics.get('cpu_percent', 0) > 95:
        return 'degraded'

    # Check for any unhealthy components
    for comp_data in component_health.values():
        if isinstance(comp_data, dict) and comp_data.get('status') == 'unhealthy':
            return 'degraded'

    return 'healthy'

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    try:
        # Gather all health data
        component_health = await get_component_health()
        api_health = await get_api_health()
        system_metrics = get_system_metrics()

        # Determine overall status
        overall_status = determine_overall_status(component_health, api_health, system_metrics)

        response = HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=get_uptime(),
            checks={
                "components": component_health,
                "api_endpoints": api_health,
                "system": system_metrics
            }
        )

        # Return appropriate HTTP status
        status_code = 200 if overall_status == 'healthy' else 503 if overall_status == 'unhealthy' else 200

        return JSONResponse(content=response.dict(), status_code=status_code)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": get_uptime(),
                "error": str(e)
            },
            status_code=503
        )

@app.get("/health/components")
async def component_health_check():
    """Detailed component health check"""
    try:
        component_health = await get_component_health()
        return {"components": component_health}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Component health check failed: {str(e)}")

@app.get("/health/api")
async def api_health_check():
    """API endpoints health check"""
    try:
        api_health = await get_api_health()
        return {"api_endpoints": api_health}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"API health check failed: {str(e)}")

@app.get("/health/system")
async def system_health_check():
    """System metrics health check"""
    try:
        system_metrics = get_system_metrics()
        return {"system": system_metrics}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"System health check failed: {str(e)}")

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint"""
    try:
        # Check critical components
        component_health = await get_component_health()
        critical_components = ['alert_manager']

        for comp in critical_components:
            if comp in component_health:
                comp_data = component_health[comp]
                if isinstance(comp_data, dict) and comp_data.get('status') == 'unhealthy':
                    return JSONResponse(
                        content={"status": "not ready", "reason": f"Critical component {comp} unhealthy"},
                        status_code=503
                    )

        return {"status": "ready"}

    except Exception as e:
        return JSONResponse(
            content={"status": "not ready", "reason": str(e)},
            status_code=503
        )

@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    # Simple liveness check - if the server is responding, it's alive
    return {"status": "alive"}

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint"""
    try:
        component_health = await get_component_health()
        system_metrics = get_system_metrics()

        # Format as Prometheus metrics
        metrics_output = []

        # System metrics
        metrics_output.append("# HELP abc_app_cpu_percent CPU usage percentage")
        metrics_output.append("# TYPE abc_app_cpu_percent gauge")
        metrics_output.append(f"abc_app_cpu_percent {system_metrics['cpu_percent']}")

        metrics_output.append("# HELP abc_app_memory_percent Memory usage percentage")
        metrics_output.append("# TYPE abc_app_memory_percent gauge")
        metrics_output.append(f"abc_app_memory_percent {system_metrics['memory_percent']}")

        # Component health metrics
        metrics_output.append("# HELP abc_app_component_health Component health status (0=healthy, 1=degraded, 2=unhealthy)")
        metrics_output.append("# TYPE abc_app_component_health gauge")

        status_map = {'healthy': 0, 'degraded': 1, 'unhealthy': 2}
        for comp_name, comp_data in component_health.items():
            if isinstance(comp_data, dict):
                status = comp_data.get('status', 'unknown')
                value = status_map.get(status, -1)
                metrics_output.append(f"abc_app_component_health{{component=\"{comp_name}\"}} {value}")

        return "\n".join(metrics_output)

    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(status_code=503, detail=f"Metrics unavailable: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize health monitoring on startup"""
    global component_monitor, alert_manager

    try:
        # Initialize component monitor
        component_monitor = get_component_health_monitor()
        component_monitor.start_monitoring()
        logger.info("Component health monitoring started")

        # Get alert manager
        alert_manager = get_alert_manager()
        logger.info("Alert manager connected")

    except Exception as e:
        logger.error(f"Failed to initialize health monitoring: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    global component_monitor

    if component_monitor:
        component_monitor.stop_monitoring()
        logger.info("Component health monitoring stopped")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="ABC-Application Health Check Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    logger.info(f"Starting health check server on {args.host}:{args.port}")

    uvicorn.run(
        "health_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()