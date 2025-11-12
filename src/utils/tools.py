# src/utils/tools.py
# Purpose: Aggregates all utility tools and functions for agent operations.
# This file imports from specialized modules for better organization and maintainability.

from langchain_core.tools import tool
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import requests
from .config import get_marketdataapp_api_key, get_grok_api_key, get_kalshi_api_key, get_kalshi_access_key_id
import time
import threading
from functools import wraps
import logging
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
import base64
from .api_health_monitor import get_api_health_summary, check_api_health_now, start_health_monitoring, stop_health_monitoring

# Import from specialized modules
from .validation import (
    DataValidator, validate_tool_inputs, circuit_breaker,
    get_circuit_breaker_status, CircuitBreakerOpenException
)
from .financial_tools import (
    yfinance_data_tool, sentiment_analysis_tool, risk_calculation_tool, strategy_proposal_tool
)
from .news_tools import (
    news_data_tool, economic_data_tool, currents_news_tool
)
from .market_data_tools import (
    marketdataapp_api_tool, marketdataapp_websocket_tool, alpha_vantage_tool, financial_modeling_prep_tool
)
from .backtesting_tools import (
    pyfolio_metrics_tool, zipline_backtest_tool, backtrader_strategy_tool, risk_analytics_tool
)
from .social_media_tools import (
    twitter_sentiment_tool, social_media_monitor_tool, reddit_sentiment_tool, news_sentiment_aggregation_tool
)
from .agent_tools import (
    audit_poll_tool, agent_coordination_tool, shared_memory_broadcast_tool,
    agent_health_check_tool, collaborative_decision_tool
)

# Additional imports for backward compatibility
# These functions may need to be implemented in appropriate modules
try:
    from .news_tools import fred_data_tool
except ImportError:
    # Placeholder for fred_data_tool
    def fred_data_tool(*args, **kwargs):
        return {"error": "fred_data_tool not implemented"}

try:
    from .market_data_tools import institutional_holdings_analysis_tool, thirteen_f_filings_tool
except ImportError:
    # Placeholders
    def institutional_holdings_analysis_tool(*args, **kwargs):
        return {"error": "institutional_holdings_analysis_tool not implemented"}
    def thirteen_f_filings_tool(*args, **kwargs):
        return {"error": "thirteen_f_filings_tool not implemented"}

try:
    from .market_data_tools import fundamental_data_tool
except ImportError:
    def fundamental_data_tool(*args, **kwargs):
        return {"error": "fundamental_data_tool not implemented"}

try:
    from .financial_tools import fundamental_analysis_tool
except ImportError:
    def fundamental_analysis_tool(*args, **kwargs):
        return {"error": "fundamental_analysis_tool not implemented"}

try:
    from .market_data_tools import microstructure_analysis_tool
except ImportError:
    def microstructure_analysis_tool(*args, **kwargs):
        return {"error": "microstructure_analysis_tool not implemented"}

try:
    from .market_data_tools import kalshi_data_tool
except ImportError:
    def kalshi_data_tool(*args, **kwargs):
        return {"error": "kalshi_data_tool not implemented"}

try:
    from .backtesting_tools import tf_quant_monte_carlo_tool
except ImportError:
    def tf_quant_monte_carlo_tool(*args, **kwargs):
        return {"error": "tf_quant_monte_carlo_tool not implemented"}

try:
    from .financial_tools import options_greeks_calc_tool, flow_alpha_calc_tool
except ImportError:
    def options_greeks_calc_tool(*args, **kwargs):
        return {"error": "options_greeks_calc_tool not implemented"}
    def flow_alpha_calc_tool(*args, **kwargs):
        return {"error": "flow_alpha_calc_tool not implemented"}

logger = logging.getLogger(__name__)

# Additional utility functions not included in specialized modules

@tool
def load_yaml_tool(file_path: str) -> Dict[str, Any]:
    '''
    Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML file to load
        
    Returns:
        Dict containing the loaded YAML data
    '''
    try:
        from .config import load_yaml
        return load_yaml(file_path)
    except Exception as e:
        return {'error': f'Failed to load YAML file {file_path}: {str(e)}'}

@tool
def qlib_ml_refine_tool(data: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Refine machine learning models using Qlib framework.
    
    Args:
        data: Input data for ML refinement
        
    Returns:
        Dict containing refined ML results
    '''
    try:
        # Mock implementation for testing
        return {
            'refined_model': 'qlib_enhanced',
            'accuracy': 0.85,
            'features_used': list(data.keys()) if isinstance(data, dict) else ['mock_data'],
            'source': 'qlib_ml_refine'
        }
    except Exception as e:
        return {'error': f'Qlib ML refinement failed: {str(e)}'}

def get_available_tools() -> Dict[str, Any]:
    '''
    Get a dictionary of all available tools and their functions.

    Returns:
        Dict mapping tool names to tool functions
    '''
    # Import all tools to ensure they're available
    tools_dict = {}

    # List of all tool functions available from modules
    tool_functions = [
        yfinance_data_tool,
        sentiment_analysis_tool,
        risk_calculation_tool,
        strategy_proposal_tool,
        news_data_tool,
        economic_data_tool,
        marketdataapp_api_tool,
        marketdataapp_websocket_tool,
        audit_poll_tool,
        pyfolio_metrics_tool,
        zipline_backtest_tool,
        twitter_sentiment_tool,
        currents_news_tool,
        qlib_ml_refine_tool
    ]

    # Create mapping from function name to function
    for tool_func in tool_functions:
        try:
            # Handle both regular functions and StructuredTool objects
            if hasattr(tool_func, 'name'):
                tool_name = tool_func.name
            elif hasattr(tool_func, '__name__'):
                tool_name = tool_func.__name__
            else:
                # Fallback: create name from object type
                tool_name = str(type(tool_func).__name__).lower() + '_tool'
            
            tools_dict[tool_name] = tool_func
        except Exception:
            # Skip tools that can't be processed
            continue

    return tools_dict
