# Integration Tests Folder

This folder contains all integration and system testing scripts for the ABC Application system.

## Test Scripts

### Core Tests
- `comprehensive_test.py` - Full system testing of all agents and subagents
- `full_system_integration_test.py` - Complete system integration validation
- `historical_simulation_demo.py` - Demonstration of historical simulation capabilities

### IBKR Tests
- `test_ibkr_simple.py` - Basic IBKR connection and trading functionality
- `test_ibkr_historical.py` - IBKR historical data provider testing
- `test_paper_trade.py` - Paper trade execution validation

### Diagnostic Tools
- `diagnose_ibkr.py` - IBKR connectivity diagnostic tool
- `grok_simulator.py` - Grok API simulation for testing

## Usage

Run any test script directly:
```bash
python integration-tests/test_ibkr_simple.py
```

Or run comprehensive tests:
```bash
python integration-tests/comprehensive_test.py
```