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
- `comprehensive_test.py` - Full system testing of all agents and subagents
- `full_system_integration_test.py` - Complete system integration validation
- `system_integration_test.py` - Core system integration testing
- `priority7_integration_test.py` - Priority 7 system integration with performance targets
- `macro_to_micro_framework_test.py` - Macro-to-micro analysis framework validation

### IBKR Integration Tests
- `test_ibkr_simple.py` - Basic IBKR connection and trading functionality
- `test_ibkr_historical.py` - IBKR historical data provider testing
- `test_ibkr_paper_trading.py` - IBKR paper trading connection and functionality
- `test_paper_trade.py` - Paper trade execution validation

### Agent Integration Tests
- `test_data_integration.py` - Data agent integration testing
- `test_shared_memory_integration.py` - Shared memory system integration
- `test_server_access.py` - Discord server access integration
- `test_startup.py` - Application startup integration testing

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

# Full system integration test
python integration-tests/comprehensive_test.py

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
```

### Test Dependencies
Before running integration tests, ensure:
1. All Python dependencies are installed: `pip install -r requirements.txt`
2. Environment variables are configured in `.env` file
3. Redis server is running (for memory tests)
4. IBKR TWS/Gateway is running (for IBKR integration tests)
5. API keys are properly configured

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