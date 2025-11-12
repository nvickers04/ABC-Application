"""
API Health Monitoring Tool for ABC Application

This module provides a Langchain-compatible tool for monitoring API health status.
"""

from langchain_core.tools import tool
from typing import Dict, Any
from .api_health_monitor import get_api_health_summary, check_api_health_now, start_health_monitoring, stop_health_monitoring

@tool
def api_health_monitor_tool(action: str = "status") -> Dict[str, Any]:
    """
    Monitor and manage API health status across all data sources.
    Args:
        action: Action to perform ('status', 'check_now', 'start_monitoring', 'stop_monitoring').
    Returns:
        Dict with API health information.
    """
    try:
        if action == "status":
            return get_api_health_summary()
        elif action == "check_now":
            return check_api_health_now()
        elif action == "start_monitoring":
            start_health_monitoring()
            return {"message": "API health monitoring started", "status": "active"}
        elif action == "stop_monitoring":
            stop_health_monitoring()
            return {"message": "API health monitoring stopped", "status": "inactive"}
        else:
            return {"error": f"Unknown action: {action}. Use 'status', 'check_now', 'start_monitoring', or 'stop_monitoring'"}
    except Exception as e:
        return {"error": f"Health monitoring error: {str(e)}"}