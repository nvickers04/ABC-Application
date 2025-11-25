# Unit Tests Folder

This folder contains unit tests for individual components of the ABC Application system.

## Test Categories

### Backtesting & Simulation
- `test_backtesting.py` - General backtesting functionality
- `test_backtesting_validation.py` - Backtesting validation and edge cases
- `test_strategy_backtrader.py` - Strategy backtesting framework testing

### Memory & Analytics
- `test_batch_analytics_memory.py` - Batch analytics memory testing
- `test_memory_comprehensive.py` - Comprehensive memory system testing

### Agent Components
- `test_agents_core.py` - Core agent functionality
- `test_enhanced_subagents.py` - Enhanced subagent functionality testing
- `test_multi_instrument.py` - Multi-instrument strategy testing

### Core Systems
- `test_concurrent_pipeline.py` - Concurrent data pipeline testing
- `test_collaborative_sessions.py` - Agent collaboration testing
- `test_config.py` - Configuration system testing
- `test_ibkr_connection.py` - IBKR connection testing
- `test_imports.py` - Import/module testing

### Risk & Strategy
- `test_risk_analytics_framework.py` - Risk analytics framework testing

### Data & Market Analysis
- `test_data_analyzers.py` - Data analyzer components
- `test_enhanced_analyzers.py` - Enhanced analyzer functionality
- `test_yfinance_data_analyzer.py` - Yahoo Finance data analyzer

### Performance & Optimization
- `test_optimized_performance.py` - Performance optimization testing

## Running Tests

### Run All Unit Tests
```bash
python -m pytest unit-tests/ -v
```

### Run Specific Test
```bash
python -m pytest unit-tests/test_backtesting_validation.py -v
```

### Run with Coverage
```bash
python -m pytest --cov=src --cov-report=html unit-tests/
```

### Run Specific Categories
```bash
# Backtesting tests only
python -m pytest unit-tests/ -k "backtest" -v

# Agent tests only
python -m pytest unit-tests/ -k "agent" -v

# Risk tests only
python -m pytest unit-tests/ -k "risk" -v
```

## Test Structure

Each test file follows the naming convention `test_*.py` and contains:
- Unit tests for individual functions/classes
- Mocked dependencies where appropriate
- Clear test cases with descriptive names
- Setup/teardown methods as needed
- Edge case testing for robustness

## New Test Additions (2025-11-24)

### Backtesting Validation (`test_backtesting_validation.py`)
- Realistic performance validation
- Market crash scenario testing
- Benchmark comparison testing
- Transaction cost impact analysis
- Edge case handling (insufficient data, high volatility)

### Enhanced Strategy Testing (`test_strategy_backtrader.py`)
- Converted to proper pytest format
- Edge case testing (market crashes, high volatility)
- Backtrader validation integration
- Empty dataframe handling
- Invalid sentiment handling

## Integration with CI/CD

These unit tests are designed to run in automated CI/CD pipelines to ensure code quality and prevent regressions. All tests are optimized for fast execution and parallel running.

### CI/CD Commands
```bash
# Fast unit test run (no slow tests)
python -m pytest unit-tests/ -m "not slow" --tb=short

# Full unit test suite with coverage
python -m pytest unit-tests/ --cov=src --cov-report=xml --cov-fail-under=85
```