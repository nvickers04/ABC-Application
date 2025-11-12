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

@tool
@circuit_breaker("sanity_check", failure_threshold=3, recovery_timeout=3600)
def sanity_check_tool(proposal: str) -> Dict[str, Any]:
    """
    Perform sanity checks on trading proposals to ensure they make logical sense.
    Args:
        proposal: Trading proposal to validate.
    Returns:
        Dict with sanity check results and recommendations.
    """
    results = {
        "proposal": proposal,
        "timestamp": pd.Timestamp.now().isoformat(),
        "checks": {},
        "overall_sanity": "unknown",
        "recommendations": []
    }

    try:
        # Basic sanity checks
        checks = {
            "has_symbol": False,
            "has_direction": False,
            "has_quantity": False,
            "has_price_logic": False,
            "risk_reasonable": False,
            "time_horizon_reasonable": False
        }

        proposal_lower = proposal.lower()

        # Check for stock symbol (basic pattern)
        import re
        symbol_pattern = r'\b[A-Z]{1,5}\b'  # 1-5 uppercase letters
        symbols = re.findall(symbol_pattern, proposal)
        if symbols:
            # Filter out common words that might match
            valid_symbols = [s for s in symbols if s not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'HOT', 'BUT', 'SAY', 'WHO', 'EACH', 'WHICH', 'THEIR', 'TIME', 'WILL', 'ABOUT', 'WOULD', 'THERE', 'COULD', 'OTHER']]
            if valid_symbols:
                checks["has_symbol"] = True
                results["identified_symbols"] = valid_symbols

        # Check for direction
        if any(word in proposal_lower for word in ['buy', 'sell', 'long', 'short', 'purchase', 'acquire']):
            checks["has_direction"] = True

        # Check for quantity/position size
        if any(word in proposal_lower for word in ['shares', 'position', 'size', 'allocation', 'percent', '%', 'dollars', '$']):
            checks["has_quantity"] = True

        # Check for price logic
        if any(word in proposal_lower for word in ['price', 'valuation', 'pe', 'pb', 'growth', 'momentum', 'support', 'resistance']):
            checks["has_price_logic"] = True

        # Check for reasonable risk
        risk_indicators = ['stop loss', 'risk management', 'position size', 'volatility', 'drawdown']
        if any(indicator in proposal_lower for indicator in risk_indicators):
            checks["risk_reasonable"] = True

        # Check for time horizon
        if any(word in proposal_lower for word in ['days', 'weeks', 'months', 'years', 'hold', 'exit', 'target']):
            checks["time_horizon_reasonable"] = True

        results["checks"] = checks

        # Overall sanity assessment
        passed_checks = sum(checks.values())
        total_checks = len(checks)

        if passed_checks >= total_checks * 0.8:
            results["overall_sanity"] = "excellent"
        elif passed_checks >= total_checks * 0.6:
            results["overall_sanity"] = "good"
        elif passed_checks >= total_checks * 0.4:
            results["overall_sanity"] = "fair"
        else:
            results["overall_sanity"] = "poor"

        # Generate recommendations
        if not checks["has_symbol"]:
            results["recommendations"].append("Specify which stock symbol(s) to trade")

        if not checks["has_direction"]:
            results["recommendations"].append("Clearly state buy/sell direction")

        if not checks["has_quantity"]:
            results["recommendations"].append("Define position size or allocation")

        if not checks["has_price_logic"]:
            results["recommendations"].append("Explain valuation or entry logic")

        if not checks["risk_reasonable"]:
            results["recommendations"].append("Include risk management parameters")

        if not checks["time_horizon_reasonable"]:
            results["recommendations"].append("Specify holding period or exit strategy")

        return results

    except Exception as e:
        return {
            "error": f"Sanity check failed: {str(e)}",
            "proposal": proposal,
            "overall_sanity": "error",
            "recommendations": ["Unable to perform sanity check - review proposal manually"]
        }

@tool
@circuit_breaker("convergence_check", failure_threshold=3, recovery_timeout=3600)
def convergence_check_tool(performance_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if the system's learning and performance are converging toward optimal behavior.
    Args:
        performance_data: Dict containing performance metrics and learning data.
    Returns:
        Dict with convergence analysis and recommendations.
    """
    results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "convergence_metrics": {},
        "learning_progress": {},
        "recommendations": [],
        "overall_convergence": "unknown"
    }

    try:
        # Extract performance metrics
        metrics = performance_data.get("metrics", {})
        learning_history = performance_data.get("learning_history", [])

        # Convergence checks
        convergence_checks = {
            "sharpe_ratio_stable": False,
            "win_rate_improving": False,
            "drawdown_decreasing": False,
            "learning_loss_converging": False,
            "strategy_adaptation": False
        }

        # Sharpe ratio stability (should stabilize over time)
        sharpe_history = [m.get("sharpe_ratio", 0) for m in learning_history[-10:]] if learning_history else []
        if len(sharpe_history) >= 5:
            recent_sharpe = sharpe_history[-3:]
            older_sharpe = sharpe_history[:-3]
            recent_avg = sum(recent_sharpe) / len(recent_sharpe)
            older_avg = sum(older_sharpe) / len(older_sharpe)
            # Sharpe should be positive and relatively stable
            if recent_avg > 0.5 and abs(recent_avg - older_avg) < 0.5:
                convergence_checks["sharpe_ratio_stable"] = True

        # Win rate improvement
        win_rates = [m.get("win_rate", 0) for m in learning_history[-10:]] if learning_history else []
        if len(win_rates) >= 5:
            recent_win_rate = sum(win_rates[-3:]) / 3
            older_win_rate = sum(win_rates[:-3]) / len(win_rates[:-3])
            if recent_win_rate > older_win_rate and recent_win_rate > 0.5:
                convergence_checks["win_rate_improving"] = True

        # Drawdown reduction
        drawdowns = [m.get("max_drawdown", 0) for m in learning_history[-10:]] if learning_history else []
        if len(drawdowns) >= 5:
            recent_dd = sum(drawdowns[-3:]) / 3
            older_dd = sum(drawdowns[:-3]) / len(drawdowns[:-3])
            if recent_dd < older_dd and recent_dd < 0.15:  # Less than 15% drawdown
                convergence_checks["drawdown_decreasing"] = True

        # Learning loss convergence (if available)
        losses = [m.get("learning_loss", 0) for m in learning_history[-10:]] if learning_history else []
        if len(losses) >= 5:
            recent_loss = sum(losses[-3:]) / 3
            older_loss = sum(losses[:-3]) / len(losses[:-3])
            if recent_loss < older_loss * 0.9 and recent_loss < 0.1:  # Converging and low
                convergence_checks["learning_loss_converging"] = True

        # Strategy adaptation (diversity of strategies used)
        strategies_used = set()
        for m in learning_history[-20:]:
            strategy = m.get("strategy_type", "")
            if strategy:
                strategies_used.add(strategy)
        if len(strategies_used) >= 3:  # Using multiple strategy types
            convergence_checks["strategy_adaptation"] = True

        results["convergence_metrics"] = convergence_checks

        # Learning progress assessment
        passed_checks = sum(convergence_checks.values())
        total_checks = len(convergence_checks)

        if passed_checks >= total_checks * 0.8:
            results["overall_convergence"] = "excellent"
            results["learning_progress"]["status"] = "Well converged - system performing optimally"
        elif passed_checks >= total_checks * 0.6:
            results["overall_convergence"] = "good"
            results["learning_progress"]["status"] = "Converging well - continue current learning approach"
        elif passed_checks >= total_checks * 0.4:
            results["overall_convergence"] = "fair"
            results["learning_progress"]["status"] = "Partial convergence - may need parameter tuning"
        else:
            results["overall_convergence"] = "poor"
            results["learning_progress"]["status"] = "Not converging - review learning algorithm"

        # Generate recommendations
        if not convergence_checks["sharpe_ratio_stable"]:
            results["recommendations"].append("Sharpe ratio not stable - review risk-return optimization")

        if not convergence_checks["win_rate_improving"]:
            results["recommendations"].append("Win rate not improving - consider different entry/exit signals")

        if not convergence_checks["drawdown_decreasing"]:
            results["recommendations"].append("Drawdowns not decreasing - strengthen risk management")

        if not convergence_checks["learning_loss_converging"]:
            results["recommendations"].append("Learning not converging - adjust learning rate or architecture")

        if not convergence_checks["strategy_adaptation"]:
            results["recommendations"].append("Limited strategy diversity - explore additional strategy types")

        # Performance summary
        if learning_history:
            latest_metrics = learning_history[-1]
            results["current_performance"] = {
                "sharpe_ratio": latest_metrics.get("sharpe_ratio", 0),
                "win_rate": latest_metrics.get("win_rate", 0),
                "max_drawdown": latest_metrics.get("max_drawdown", 0),
                "total_return": latest_metrics.get("total_return", 0),
                "strategy_type": latest_metrics.get("strategy_type", "unknown")
            }

        return results

    except Exception as e:
        return {
            "error": f"Convergence check failed: {str(e)}",
            "overall_convergence": "error",
            "recommendations": ["Unable to assess convergence - check performance data format"]
        }

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
        qlib_ml_refine_tool,
        sanity_check_tool,
        convergence_check_tool
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
