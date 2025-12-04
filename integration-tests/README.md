# Integration Tests

This directory contains integration tests for the ABC-Application system. These tests verify that different components work together correctly and provide end-to-end validation of system functionality.

## Test Categories

### Unified Workflow Integration (`test_unified_workflow_integration.py`)
Tests the UnifiedWorkflowOrchestrator and its interaction with various system components:
- Orchestrator initialization and lifecycle
- Workflow mode switching
- Market schedule awareness
- Agent-to-agent communication
- Error handling and recovery
- Performance and scalability

### Health API Integration (`test_health_api_integration.py`)
Tests the health monitoring API endpoints:
- Basic health check endpoints
- Component health monitoring
- System metrics collection
- API health monitoring
- Kubernetes readiness/liveness probes
- Prometheus metrics endpoint
- OpenAPI documentation availability
- Concurrent request handling
- Data accuracy validation

### IBKR Integration (`test_ibkr_integration.py`)
Tests Interactive Brokers connectivity and trading integration:
- TWS connectivity verification
- IBKR bridge initialization
- Market data retrieval
- Historical data access
- Paper trading connectivity
- Order placement simulation
- Health monitoring integration
- Configuration validation
- Alert system integration

## Running Integration Tests

### Prerequisites
- Python environment with all dependencies installed
- Redis server running (for some tests)
- Optional: IBKR TWS/Gateway running for IBKR tests

### Basic Test Run
```bash
# Run all integration tests
pytest integration-tests/

# Run specific test file
pytest integration-tests/test_unified_workflow_integration.py

# Run with verbose output
pytest integration-tests/ -v
```

### IBKR-Specific Tests
IBKR integration tests require a running TWS/Gateway instance and are disabled by default:

```bash
# Run IBKR tests (requires TWS running)
pytest integration-tests/test_ibkr_integration.py --run-ibkr-tests

# Run all tests including IBKR
pytest integration-tests/ --run-ibkr-tests
```

### Test Configuration
Integration tests use the following configuration:
- Redis: localhost:6379
- IBKR TWS: localhost:7497 (paper trading)
- Test mode: Enabled
- Log level: INFO

## Test Fixtures

### `temp_data_dir`
Provides a temporary directory for test data that gets cleaned up after tests.

### `test_config`
Provides test configuration dictionary with common settings.

### `health_monitor`
Initializes the health monitoring system for tests.

### `unified_orchestrator`
Creates a UnifiedWorkflowOrchestrator instance configured for testing.

### `component_health_monitor`
Initializes component health monitoring.

### `alert_manager`
Provides access to the alert management system.

## Test Environment Setup

### Local Development
1. Ensure Redis is running
2. Install test dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest integration-tests/`

### CI/CD Environment
Integration tests are designed to run in automated environments:
- Use mock fixtures when external services are unavailable
- Skip IBKR tests when TWS is not available
- Provide clear error messages for missing dependencies

## Test Coverage

Integration tests cover:
- **System Integration**: End-to-end workflow execution
- **API Integration**: REST API functionality and responses
- **External Service Integration**: IBKR, Redis, and other external dependencies
- **Monitoring Integration**: Health checks, metrics, and alerting
- **Configuration Integration**: Settings loading and validation
- **Performance Integration**: Load testing and resource monitoring

## Best Practices

### Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Mock external dependencies when possible
- Clean up test data after execution

### Test Naming
- Use descriptive test names
- Follow `test_*` naming convention
- Group related tests in classes

### Assertions
- Use specific assertions
- Check both positive and negative cases
- Validate data types and ranges
- Test error conditions

### Performance
- Keep tests focused and fast
- Use appropriate timeouts
- Monitor resource usage
- Parallelize when possible

## Troubleshooting

### Common Issues
- **Redis Connection Failed**: Ensure Redis server is running on localhost:6379
- **IBKR Tests Skipped**: Use `--run-ibkr-tests` flag and ensure TWS is running
- **Import Errors**: Check Python path and virtual environment
- **Timeout Errors**: Increase timeout values or check system performance

### Debug Mode
Run tests with debug output:
```bash
pytest integration-tests/ -v -s --tb=long
```

## Contributing

When adding new integration tests:
1. Follow existing naming and structure conventions
2. Add appropriate fixtures for setup/teardown
3. Include both success and failure test cases
4. Update this README with new test descriptions
5. Ensure tests run in CI/CD environment</content>
</xai:function_call name="manage_todo_list">
<parameter name="todoList">[{"id":2,"title":"Add Integration Test Suite","status":"completed"}]