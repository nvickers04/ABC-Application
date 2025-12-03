# [LABEL:TEST:data] [LABEL:FRAMEWORK:fixtures] [LABEL:DATA:market]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-02] [LABEL:REVIEWED:pending]
#
# Purpose: Standardized market data fixtures for testing
# Dependencies: pytest
# Related: tests/fixtures/*.py, unit-tests/*, integration-tests/*

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

# Standard test symbols
TEST_SYMBOLS = {
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'exchange': 'NASDAQ'},
    'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'exchange': 'NASDAQ'},
    'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'exchange': 'NASDAQ'},
    'TSLA': {'name': 'Tesla Inc.', 'sector': 'Consumer Discretionary', 'exchange': 'NASDAQ'},
    'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Discretionary', 'exchange': 'NASDAQ'},
    'SPY': {'name': 'SPDR S&P 500 ETF', 'sector': 'Financials', 'exchange': 'NYSE'},
    'QQQ': {'name': 'Invesco QQQ Trust', 'sector': 'Financials', 'exchange': 'NASDAQ'},
}

# Standard market data templates
MARKET_DATA_TEMPLATES = {
    'pre_market': {
        'timestamp': datetime.now().replace(hour=8, minute=0, second=0),
        'bid': 150.00,
        'ask': 150.10,
        'last': 150.05,
        'volume': 1000,
        'high': 151.00,
        'low': 149.50,
        'open': 150.00,
        'close': 150.05,
        'vwap': 150.02,
    },
    'regular_hours': {
        'timestamp': datetime.now().replace(hour=10, minute=30, second=0),
        'bid': 152.50,
        'ask': 152.60,
        'last': 152.55,
        'volume': 50000,
        'high': 153.00,
        'low': 151.00,
        'open': 151.50,
        'close': 152.55,
        'vwap': 152.10,
    },
    'after_hours': {
        'timestamp': datetime.now().replace(hour=16, minute=30, second=0),
        'bid': 153.00,
        'ask': 153.10,
        'last': 153.05,
        'volume': 25000,
        'high': 153.50,
        'low': 152.00,
        'open': 152.50,
        'close': 153.05,
        'vwap': 152.80,
    },
}

# Historical price data for backtesting
HISTORICAL_DATA = {
    'AAPL': [
        {'date': '2024-01-01', 'open': 185.00, 'high': 186.00, 'low': 184.00, 'close': 185.50, 'volume': 1000000},
        {'date': '2024-01-02', 'open': 185.50, 'high': 187.00, 'low': 185.00, 'close': 186.75, 'volume': 1200000},
        {'date': '2024-01-03', 'open': 186.75, 'high': 188.00, 'low': 186.00, 'close': 187.25, 'volume': 950000},
        {'date': '2024-01-04', 'open': 187.25, 'high': 189.00, 'low': 187.00, 'close': 188.50, 'volume': 1100000},
        {'date': '2024-01-05', 'open': 188.50, 'high': 190.00, 'low': 188.00, 'close': 189.25, 'volume': 1300000},
    ],
    'GOOGL': [
        {'date': '2024-01-01', 'open': 135.00, 'high': 136.50, 'low': 134.50, 'close': 136.00, 'volume': 800000},
        {'date': '2024-01-02', 'open': 136.00, 'high': 137.50, 'low': 135.50, 'close': 137.25, 'volume': 900000},
        {'date': '2024-01-03', 'open': 137.25, 'high': 138.50, 'low': 137.00, 'close': 138.00, 'volume': 750000},
        {'date': '2024-01-04', 'open': 138.00, 'high': 139.50, 'low': 137.50, 'close': 139.25, 'volume': 850000},
        {'date': '2024-01-05', 'open': 139.25, 'high': 140.50, 'low': 139.00, 'close': 140.00, 'volume': 950000},
    ],
}

@pytest.fixture
def sample_symbols():
    """Fixture providing standard test symbols."""
    return TEST_SYMBOLS.copy()

@pytest.fixture
def market_data_template():
    """Fixture providing market data templates."""
    return MARKET_DATA_TEMPLATES.copy()

@pytest.fixture
def historical_data():
    """Fixture providing historical price data."""
    return HISTORICAL_DATA.copy()

@pytest.fixture
def aapl_market_data():
    """Fixture providing current market data for AAPL."""
    return {
        'symbol': 'AAPL',
        'timestamp': datetime.now(),
        'bid': Decimal('192.50'),
        'ask': Decimal('192.60'),
        'last': Decimal('192.55'),
        'volume': 100000,
        'high': Decimal('193.00'),
        'low': Decimal('191.50'),
        'open': Decimal('192.00'),
        'close': Decimal('192.55'),
        'vwap': Decimal('192.30'),
        'bid_size': 100,
        'ask_size': 200,
    }

@pytest.fixture
def spy_market_data():
    """Fixture providing current market data for SPY."""
    return {
        'symbol': 'SPY',
        'timestamp': datetime.now(),
        'bid': Decimal('450.50'),
        'ask': Decimal('450.60'),
        'last': Decimal('450.55'),
        'volume': 500000,
        'high': Decimal('451.00'),
        'low': Decimal('449.50'),
        'open': Decimal('450.00'),
        'close': Decimal('450.55'),
        'vwap': Decimal('450.30'),
        'bid_size': 500,
        'ask_size': 800,
    }

@pytest.fixture
def volatile_market_data():
    """Fixture providing market data for a volatile stock."""
    return {
        'symbol': 'TSLA',
        'timestamp': datetime.now(),
        'bid': Decimal('245.00'),
        'ask': Decimal('246.00'),
        'last': Decimal('245.50'),
        'volume': 2000000,
        'high': Decimal('250.00'),
        'low': Decimal('240.00'),
        'open': Decimal('242.00'),
        'close': Decimal('245.50'),
        'vwap': Decimal('244.80'),
        'bid_size': 1000,
        'ask_size': 1500,
    }

@pytest.fixture
def illiquid_market_data():
    """Fixture providing market data for an illiquid stock."""
    return {
        'symbol': 'SMALL',
        'timestamp': datetime.now(),
        'bid': Decimal('15.50'),
        'ask': Decimal('16.00'),
        'last': Decimal('15.75'),
        'volume': 500,
        'high': Decimal('16.50'),
        'low': Decimal('15.00'),
        'open': Decimal('15.25'),
        'close': Decimal('15.75'),
        'vwap': Decimal('15.60'),
        'bid_size': 10,
        'ask_size': 5,
    }