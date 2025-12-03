# [LABEL:TEST:data] [LABEL:FRAMEWORK:fixtures] [LABEL:DATA:config]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-02] [LABEL:REVIEWED:pending]
#
# Purpose: Standardized trading configuration fixtures for testing
# Dependencies: pytest
# Related: tests/fixtures/*.py, config/*.yaml, config/*.ini

import pytest
from decimal import Decimal

# Standard risk constraints
RISK_CONSTRAINTS = {
    'max_position_size': Decimal('10000.00'),
    'max_portfolio_risk': Decimal('0.05'),  # 5% max portfolio risk
    'max_single_trade_risk': Decimal('0.02'),  # 2% max single trade risk
    'max_daily_loss': Decimal('500.00'),
    'max_drawdown': Decimal('0.10'),  # 10% max drawdown
    'min_liquidity_ratio': Decimal('0.50'),  # 50% minimum liquidity
    'max_leverage': Decimal('2.0'),
    'max_concentration': Decimal('0.25'),  # 25% max concentration in single asset
}

# Standard trading permissions
TRADING_PERMISSIONS = {
    'allowed_exchanges': ['NASDAQ', 'NYSE', 'AMEX'],
    'allowed_asset_classes': ['equity', 'option', 'future'],
    'max_order_quantity': 1000,
    'min_order_quantity': 1,
    'allowed_order_types': ['market', 'limit', 'stop', 'stop_limit'],
    'restricted_symbols': ['GME', 'AMC'],  # High risk symbols
    'trading_hours_only': True,
    'pre_market_allowed': False,
    'after_hours_allowed': False,
}

# IBKR configuration template
IBKR_CONFIG = {
    'host': '127.0.0.1',
    'port': 7497,
    'client_id': 1,
    'account': 'DU1234567',
    'timeout': 30,
    'readonly': False,
    'paper_trading': True,
}

# Profitability targets
PROFITABILITY_TARGETS = {
    'daily_target': Decimal('100.00'),
    'weekly_target': Decimal('500.00'),
    'monthly_target': Decimal('2000.00'),
    'sharpe_ratio_target': Decimal('1.5'),
    'max_drawdown_limit': Decimal('0.05'),
    'win_rate_target': Decimal('0.55'),
    'profit_factor_target': Decimal('1.3'),
}

# API cost reference
API_COST_REFERENCE = {
    'ibkr_per_contract': Decimal('0.005'),
    'ibkr_per_trade': Decimal('0.35'),
    'alpha_vantage_per_call': Decimal('0.01'),
    'polygon_per_request': Decimal('0.001'),
    'twelve_data_per_request': Decimal('0.01'),
    'monthly_budget': Decimal('100.00'),
}

@pytest.fixture
def risk_constraints():
    """Fixture providing standard risk constraints."""
    return RISK_CONSTRAINTS.copy()

@pytest.fixture
def trading_permissions():
    """Fixture providing standard trading permissions."""
    return TRADING_PERMISSIONS.copy()

@pytest.fixture
def ibkr_config():
    """Fixture providing IBKR configuration."""
    return IBKR_CONFIG.copy()

@pytest.fixture
def profitability_targets():
    """Fixture providing profitability targets."""
    return PROFITABILITY_TARGETS.copy()

@pytest.fixture
def api_cost_reference():
    """Fixture providing API cost reference."""
    return API_COST_REFERENCE.copy()

@pytest.fixture
def conservative_risk_constraints():
    """Fixture providing conservative risk constraints."""
    return {
        'max_position_size': Decimal('5000.00'),
        'max_portfolio_risk': Decimal('0.02'),  # 2% max portfolio risk
        'max_single_trade_risk': Decimal('0.005'),  # 0.5% max single trade risk
        'max_daily_loss': Decimal('200.00'),
        'max_drawdown': Decimal('0.05'),  # 5% max drawdown
        'min_liquidity_ratio': Decimal('0.75'),  # 75% minimum liquidity
        'max_leverage': Decimal('1.0'),
        'max_concentration': Decimal('0.10'),  # 10% max concentration in single asset
    }

@pytest.fixture
def aggressive_risk_constraints():
    """Fixture providing aggressive risk constraints."""
    return {
        'max_position_size': Decimal('25000.00'),
        'max_portfolio_risk': Decimal('0.10'),  # 10% max portfolio risk
        'max_single_trade_risk': Decimal('0.05'),  # 5% max single trade risk
        'max_daily_loss': Decimal('1000.00'),
        'max_drawdown': Decimal('0.20'),  # 20% max drawdown
        'min_liquidity_ratio': Decimal('0.25'),  # 25% minimum liquidity
        'max_leverage': Decimal('5.0'),
        'max_concentration': Decimal('0.50'),  # 50% max concentration in single asset
    }

@pytest.fixture
def test_user_config():
    """Fixture providing test user configuration."""
    return {
        'user_id': 'test_user_001',
        'account_type': 'paper_trading',
        'risk_profile': 'moderate',
        'preferred_exchanges': ['NASDAQ', 'NYSE'],
        'notification_preferences': {
            'email': True,
            'sms': False,
            'discord': True,
        },
        'trading_schedule': {
            'start_time': '09:30',
            'end_time': '16:00',
            'timezone': 'America/New_York',
        },
    }

@pytest.fixture
def production_user_config():
    """Fixture providing production user configuration."""
    return {
        'user_id': 'prod_user_001',
        'account_type': 'live_trading',
        'risk_profile': 'conservative',
        'preferred_exchanges': ['NASDAQ', 'NYSE', 'AMEX'],
        'notification_preferences': {
            'email': True,
            'sms': True,
            'discord': True,
        },
        'trading_schedule': {
            'start_time': '09:30',
            'end_time': '16:00',
            'timezone': 'America/New_York',
        },
    }