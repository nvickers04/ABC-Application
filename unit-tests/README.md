# Unit Tests Folder

This folder contains unit tests for individual components of the ABC Application system.

## Test Categories

### Memory & Analytics
- `test_batch_analytics_memory.py` - Batch analytics memory testing
- `test_memory_comprehensive.py` - Comprehensive memory system testing

### Agent Components
- `test_enhanced_subagents.py` - Enhanced subagent functionality testing
- `test_multi_instrument.py` - Multi-instrument strategy testing

### Core Systems
- `test_concurrent_pipeline.py` - Concurrent data pipeline testing
- `test_collaborative_sessions.py` - Agent collaboration testing
- `test_ibkr_connection.py` - IBKR connection testing
- `test_imports.py` - Import/module testing

### Risk & Strategy
- `test_risk_analytics_framework.py` - Risk analytics framework testing
- `test_strategy_backtrader.py` - Strategy backtesting framework testing

### Performance
- `test_optimized_performance.py` - Performance optimization testing

## Running Tests

### Run All Unit Tests
```bash
python -m pytest unit-tests/
```

### Run Specific Test
```bash
python -m pytest unit-tests/test_memory_comprehensive.py
```

### Run with Coverage
```bash
python -m pytest --cov=src --cov-report=html unit-tests/
```

## Test Structure

Each test file follows the naming convention `test_*.py` and contains:
- Unit tests for individual functions/classes
- Mocked dependencies where appropriate
- Clear test cases with descriptive names
- Setup/teardown methods as needed

## Integration with CI/CD

These unit tests are designed to run in automated CI/CD pipelines to ensure code quality and prevent regressions.