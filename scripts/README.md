# Scripts Directory

This directory contains utility scripts and diagnostic tools for the ABC-Application system.

## Diagnostic Scripts

### `test_grok.py`
Tests Grok API connectivity and initialization.

**Usage:**
```bash
python scripts/test_grok.py
```

**Purpose:**
- Validate Grok API key configuration
- Test LLM connectivity
- Verify model initialization

### `quick_workflow_test.py`
Quick workflow validation script for testing basic system functionality.

**Usage:**
```bash
python scripts/quick_workflow_test.py
```

**Purpose:**
- Basic system health checks
- Workflow validation
- Quick diagnostic testing

### `test_import.py`
Tests DataAgent imports with mocked dependencies.

**Usage:**
```bash
python scripts/test_import.py
```

**Purpose:**
- Validate DataAgent import paths
- Test with mocked analyzers
- Quick import validation

### `test_imports.py`
Tests TensorFlow and Nautilus Trader imports.

**Usage:**
```bash
python scripts/test_imports.py
```

**Purpose:**
- Validate ML framework imports
- Test trading platform connectivity
- Dependency verification

### `diagnose_ibkr.py`
Minimal IBKR connectivity diagnostic tool.

**Usage:**
```bash
python scripts/diagnose_ibkr.py
```

**Purpose:**
- Test IBKR TWS API connectivity
- Diagnose connection issues
- Validate trading platform setup

## Purpose

These scripts are designed for:
- **Diagnostics**: Testing system components and external services
- **Validation**: Quick checks of system functionality
- **Debugging**: Troubleshooting connectivity and configuration issues
- **Development**: Supporting development and testing workflows

## Running Scripts

All scripts can be run directly from the project root:
```bash
python scripts/<script_name>.py
```

Some scripts may require:
- API keys configured in `.env`
- External services running (Redis, IBKR, etc.)
- Network connectivity for API tests