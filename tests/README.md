# ABC-Application Testing Suite

This directory contains the consolidated testing suite for the ABC-Application, organized into subdirectories for different test types.

## Directory Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for component interactions
- `performance/` - Performance and load testing
- `security/` - Security vulnerability tests
- `system/` - Full system deployment and validation tests

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only
```bash
pytest tests/unit/
```

### Integration Tests Only
```bash
pytest tests/integration/
```

### With Coverage
```bash
pytest --cov=src --cov-report=html
```

### Specific Markers
```bash
pytest -m ibkr  # IBKR-related tests
pytest -m mocked  # Tests using mocks
pytest -m slow  # Slower tests
```

### Parallel Execution
```bash
pytest -n auto
```

## Performance Testing

Run load tests with Locust:
```bash
locust -f tests/performance/locustfile.py
```

## Security Testing

Run security scans:
```bash
python tests/security/run_security_scan.py
```

Or with Bandit:
```bash
bandit -c tests/security/bandit_config.yaml -r src/
```

## Coverage Goals

- Overall coverage: 80%+
- Unit tests: 90%+
- Integration tests: 75%+
- Critical paths (IBKR, workflows): 95%+

## CI/CD Integration

Tests are configured to run automatically on pushes and PRs. Key checks:
- Unit test suite
- Integration test suite (with IBKR mock)
- Security scans
- Coverage reports

## Adding New Tests

- Unit tests: `tests/unit/test_<component>.py`
- Integration tests: `tests/integration/test_<feature>.py`
- Performance tests: `tests/performance/`
- Security tests: `tests/security/`
- System tests: `tests/system/`

Follow pytest conventions and use appropriate markers.