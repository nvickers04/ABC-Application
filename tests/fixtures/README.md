# Test Fixtures Documentation

## Overview

This directory contains standardized test fixtures for the ABC Application test suite. These fixtures provide consistent, reusable test data and mock objects to ensure reliable and maintainable tests.

## Structure

```
tests/fixtures/
├── README.md              # This documentation
├── market_data.py         # Market data fixtures
├── trading_config.py      # Trading configuration fixtures
├── mock_agents.py         # Mock agent fixtures
└── __init__.py           # Package initialization
```

## Available Fixtures

### Market Data Fixtures (`market_data.py`)

#### Data Fixtures
- `sample_symbols`: Standard test symbols with metadata
- `market_data_template`: Pre-defined market data templates for different market sessions
- `historical_data`: Historical price data for backtesting
- `aapl_market_data`: Current market data for AAPL
- `spy_market_data`: Current market data for SPY ETF
- `volatile_market_data`: Market data for high-volatility stocks
- `illiquid_market_data`: Market data for low-liquidity stocks

#### Usage Example
```python
def test_price_calculation(aapl_market_data):
    price = aapl_market_data['last']
    assert price > 0
```

### Trading Configuration Fixtures (`trading_config.py`)

#### Configuration Fixtures
- `risk_constraints`: Standard risk management constraints
- `trading_permissions`: Trading permission settings
- `ibkr_config`: IBKR API configuration
- `profitability_targets`: Performance target settings
- `api_cost_reference`: API cost tracking reference
- `conservative_risk_constraints`: Conservative risk settings
- `aggressive_risk_constraints`: Aggressive risk settings
- `test_user_config`: Test user configuration
- `production_user_config`: Production user configuration

#### Usage Example
```python
def test_risk_check(risk_constraints, sample_portfolio):
    max_risk = risk_constraints['max_portfolio_risk']
    assert max_risk <= 0.05  # 5% limit
```

### Mock Agent Fixtures (`mock_agents.py`)

#### Mock Objects
- `mock_execution_agent`: Mock trading execution agent
- `mock_data_analyzer`: Mock market data analyzer
- `mock_risk_manager`: Mock risk management agent
- `mock_memory_agent`: Mock memory/learning agent
- `mock_alert_manager`: Mock alert/notification system
- `mock_discord_bot`: Mock Discord bot interface
- `mock_redis_client`: Mock Redis cache client
- `mock_ibkr_connector`: Mock IBKR API connector
- `mock_tigerbeetle_client`: Mock TigerBeetle database client
- `mock_strategy_analyzer`: Mock trading strategy analyzer
- `mock_orchestrator`: Mock workflow orchestrator

#### Usage Example
```python
def test_trade_execution(mock_execution_agent):
    result = await mock_execution_agent.execute_trade(
        symbol='AAPL', quantity=100, action='BUY'
    )
    assert result['success'] is True
```

### Global Test Fixtures (`conftest.py`)

#### Utility Fixtures
- `temp_dir`: Temporary directory for file operations
- `sample_trade_request`: Sample trade request data
- `sample_portfolio`: Sample portfolio data
- `sample_api_response`: Sample API response data
- `performance_timer`: Simple performance timing utility

#### Mock Fixtures
- `mock_external_api`: Comprehensive external API mocking
- `mock_database`: Database operation mocking
- `mock_file_system`: File system operation mocking

#### Error Simulation
- `simulate_network_error`: Simulate network connectivity issues
- `simulate_api_rate_limit`: Simulate API rate limiting
- `simulate_database_error`: Simulate database failures

#### Data Generators
- `generate_random_trades`: Generate random trade data for testing
- `generate_market_data_series`: Generate time series market data

## Best Practices

### 1. Use Descriptive Fixture Names
```python
# Good
def test_portfolio_rebalancing(conservative_risk_constraints, sample_portfolio):
    pass

# Avoid
def test_rebalancing(risk, data):
    pass
```

### 2. Keep Fixtures Focused
Each fixture should provide one specific type of data or mock object. Don't create "kitchen sink" fixtures that try to do everything.

### 3. Use Appropriate Scopes
- `scope="function"`: Default, creates fresh fixture for each test
- `scope="class"`: Shared across tests in a class
- `scope="module"`: Shared across tests in a module
- `scope="session"`: Shared across entire test session

### 4. Document Fixture Behavior
```python
@pytest.fixture
def mock_execution_agent():
    """Mock execution agent that always returns successful trades."""
    # Implementation...
```

### 5. Use Parametrized Tests with Fixtures
```python
@pytest.mark.parametrize("market_data", [
    pytest.lazy_fixture("aapl_market_data"),
    pytest.lazy_fixture("spy_market_data"),
    pytest.lazy_fixture("volatile_market_data"),
])
def test_price_validation(market_data):
    assert market_data['last'] > 0
```

## Adding New Fixtures

1. **Choose the appropriate file** based on the fixture's purpose
2. **Follow naming conventions**: Use descriptive names with `mock_` prefix for mocks
3. **Add comprehensive documentation** including return value structure
4. **Include usage examples** in docstrings when helpful
5. **Test your fixtures** to ensure they work correctly

## Example Test Using Multiple Fixtures

```python
import pytest

class TestTradingWorkflow:
    @pytest.mark.asyncio
    async def test_complete_trade_workflow(
        self,
        mock_execution_agent,
        mock_data_analyzer,
        mock_risk_manager,
        aapl_market_data,
        risk_constraints
    ):
        # Analyze market data
        analysis = await mock_data_analyzer.analyze_market_data(aapl_market_data)
        assert analysis['trend'] in ['bullish', 'bearish', 'neutral']

        # Check risk constraints
        risk_assessment = await mock_risk_manager.assess_risk(
            symbol='AAPL',
            quantity=100,
            constraints=risk_constraints
        )
        assert risk_assessment['acceptable'] is True

        # Execute trade
        trade_result = await mock_execution_agent.execute_trade(
            symbol='AAPL',
            quantity=100,
            action='BUY'
        )
        assert trade_result['success'] is True
```

## Maintenance

- **Review fixtures regularly** to ensure they reflect current system behavior
- **Update fixtures** when the underlying system changes
- **Remove unused fixtures** to keep the test suite clean
- **Version control fixture changes** like any other code changes