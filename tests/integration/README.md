# [LABEL:DOC:integration_tests] [LABEL:DOC:readme] [LABEL:TEST:integration]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Documentation for integration tests and system validation
# Dependencies: pytest, asyncio, ABC Application source code
# Related: unit-tests/, pytest.ini, src/
#
# Integration Tests Folder

This folder contains comprehensive integration and system testing scripts for the ABC Application system. These tests validate end-to-end functionality, agent interactions, and system integration across all components.

## Test Categories

### System Integration Tests
- `full_system_integration_test.py` - Complete system integration validation
- `system_integration_test.py` - Core system integration testing
- `priority7_integration_test.py` - Priority 7 system integration with performance targets
- `macro_to_micro_framework_test.py` - Macro-to-micro analysis framework validation

### IBKR Integration Tests
- `test_ibkr_simple.py` - Basic IBKR connection and trading functionality
- `test_ibkr_historical.py` - IBKR historical data provider testing
- `test_ibkr_paper_trading.py` - IBKR paper trading connection and functionality
- `test_paper_trade.py` - Paper trade execution validation
- `test_live_trading_integration.py` - Live trading integration with safeguards

### Agent Integration Tests
- `test_data_integration.py` - Data agent integration testing
- `test_shared_memory_integration.py` - Shared memory system integration
- `test_server_access.py` - Discord server access integration
- `test_startup.py` - Application startup integration testing
- `test_enhanced_orchestrator.py` - Enhanced orchestrator functionality

### Failover & Recovery Tests
- `test_failover_recovery.py` - System recovery from failures (agent crashes, network issues, IBKR disconnects)

### End-to-End Trading Workflow Tests
- `test_e2e_trading_workflow.py` - Complete trading workflows from data to execution

### Optimization Regression Tests
- `test_optimization_regression.py` - Regression testing for optimization proposals

### Edge Cases & Rare Events Tests
- `test_edge_cases.py` - Coverage for extreme volatility, API rate limits, multi-agent conflicts, timeout handling
- `test_long_timeout.py` - Extended timeout handling for IBKR connections

### Diagnostic and Simulation Tools
- `diagnose_ibkr.py` - IBKR connectivity diagnostic tool
- `grok_simulator.py` - Grok API simulation for testing
- `historical_simulation_demo.py` - Historical simulation demonstration

## Running Tests

### Individual Test Scripts
Run any test script directly from the project root:
```bash
# Basic IBKR connectivity test
python integration-tests/test_ibkr_simple.py


# Priority 7 performance test
python integration-tests/priority7_integration_test.py
```

### Using pytest
Run all integration tests with pytest:
```bash
# Run all integration tests
pytest integration-tests/

# Run specific test file
pytest integration-tests/test_ibkr_paper_trading.py

# Run with verbose output
pytest integration-tests/ -v

# Run with coverage
pytest integration-tests/ --cov=src --cov-report=html

# Run fast unit tests only
pytest -m fast

# Run tests with mocks (skip real IBKR)
pytest -m mocked

# Run IBKR tests (requires TWS running)
pytest -m ibkr

# Run in parallel
pytest -n auto

# Run only changed tests (testmon)
pytest --testmon

# Optimized parallel with testmon (start small on Windows)
pytest -n 4 --testmon --ff --dist=loadscope
```

### Test Dependencies
Before running integration tests, ensure:
1. All Python dependencies are installed: `pip install -r requirements.txt`
2. Environment variables are configured in `.env` file
3. Redis server is running (for memory tests)
4. IBKR TWS/Gateway is running (for IBKR integration tests)
5. API keys are properly configured

For IBKR tests, set RUN_IBKR_TESTS=true in .env

### Environment Setup
```bash
# Copy environment template
cp config/.env.template .env

# Edit .env with your actual API keys and configuration
# Required for most integration tests:
# - IBKR credentials (for trading tests)
# - LLM API keys (for agent tests)
# - Database connection (if applicable)
```

## Test Results and Logging

Integration tests generate detailed logs and results:
- Test output is displayed in console
- JSON result files are saved to `data/` directory
- Logs are written to `logs/` directory
- Performance metrics are tracked for system validation

### Example Test Output
```
🚀 PRIORITY 7: SYSTEM INTEGRATION & PERFORMANCE TEST
Testing enhanced LLM-powered trading AI system
Target: 10-20% monthly returns with <5% drawdown

🤖 INITIALIZING ALL AGENTS
✅ DataAgent initialized (enhanced with LLM)
✅ RiskAgent initialized
✅ StrategyAgent initialized
✅ ExecutionAgent initialized
✅ ReflectionAgent initialized
✅ LearningAgent initialized
✅ MacroAgent initialized

🎯 CONCLUSION:
🚀 EXCELLENT: System ready for production deployment!
💰 Expected: 10-20% monthly returns with <5% drawdown
```

## Continuous Integration

These integration tests are designed to run in CI/CD pipelines:
- All tests are automated and don't require manual intervention
- Tests include proper error handling and cleanup
- Results are machine-readable for automated reporting
- Performance benchmarks ensure system stability

## Troubleshooting

### Common Issues
- **IBKR Connection Failed**: Ensure TWS/Gateway is running with API enabled
- **Redis Connection Error**: Start Redis server: `redis-server`
- **Import Errors**: Ensure you're running from project root: `cd /path/to/abc-application`
- **API Key Errors**: Check `.env` file configuration

### Debug Mode
Run tests with debug logging:
```bash
export LOG_LEVEL=DEBUG
python integration-tests/test_ibkr_simple.py
```

## Contributing

When adding new integration tests:
1. Follow the naming convention: `test_*.py`
2. Include proper error handling and cleanup
3. Add documentation in this README
4. Ensure tests can run in CI environment
5. Include performance metrics where applicable

## Optimized Test Running

### Parallel Execution
Run integration tests in parallel for faster execution:
```bash
pytest -n auto
```

### Change-Based Testing with Testmon
Run only tests affected by recent code changes:
```bash
pytest --testmon
```
