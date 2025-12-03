# [LABEL:TEST:config] [LABEL:FRAMEWORK:fixtures] [LABEL:PACKAGE:init]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-02] [LABEL:REVIEWED:pending]
#
# Purpose: Test fixtures package initialization
# Dependencies: pytest
# Related: tests/fixtures/*.py

"""
Test Fixtures Package

This package provides standardized test fixtures for the ABC Application test suite.
Fixtures are organized by category for easy maintenance and discovery.

Categories:
- market_data: Market data and pricing fixtures
- trading_config: Trading configuration and settings fixtures
- mock_agents: Mock objects for system components
"""

__version__ = "1.0.0"
__author__ = "ABC Application Team"

# Import commonly used fixtures for convenience
from .market_data import (
    sample_symbols,
    market_data_template,
    historical_data,
    aapl_market_data,
    spy_market_data,
)

from .trading_config import (
    risk_constraints,
    trading_permissions,
    ibkr_config,
    conservative_risk_constraints,
    aggressive_risk_constraints,
)

from .mock_agents import (
    mock_execution_agent,
    mock_data_analyzer,
    mock_risk_manager,
    mock_memory_agent,
    mock_alert_manager,
)

__all__ = [
    # Market data fixtures
    'sample_symbols',
    'market_data_template',
    'historical_data',
    'aapl_market_data',
    'spy_market_data',

    # Trading config fixtures
    'risk_constraints',
    'trading_permissions',
    'ibkr_config',
    'conservative_risk_constraints',
    'aggressive_risk_constraints',

    # Mock agent fixtures
    'mock_execution_agent',
    'mock_data_analyzer',
    'mock_risk_manager',
    'mock_memory_agent',
    'mock_alert_manager',
]