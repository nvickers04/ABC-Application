# System Testing Suite

This directory contains full system deployment simulation and end-to-end testing for the ABC trading system.

## Overview

System tests validate complete system functionality under production-like conditions with:
- Full deployment simulation
- End-to-end workflow testing
- Slow-running comprehensive tests (10s-5min)
- Production environment mocking
- Multi-day operation simulation

## Test Categories

### Full System Deployment Tests
- `test_full_system_deployment.py` - Complete system deployment simulation

## Running System Tests

### Run All System Tests
```bash
python -m pytest tests/system/ -v --tb=short
```

### Run Specific System Test
```bash
python -m pytest tests/system/test_full_system_deployment.py::TestFullSystemDeployment::test_complete_market_cycle_simulation -v -s
```

### Run with Performance Monitoring
```bash
# Run system tests with timing
time python -m pytest tests/system/ --durations=10

# Run specific slow test
python -m pytest tests/system/test_full_system_deployment.py::TestFullSystemDeployment::test_multi_day_simulation -v --timeout=400
```

## Test Scenarios

### Complete Market Cycle Simulation (2-3 minutes)
- Market open data ingestion
- Morning strategy analysis
- Risk assessment and portfolio review
- Trade execution
- Afternoon performance monitoring
- Market close position management
- End-of-day analysis and learning

### Multi-Day Simulation (3-4 minutes)
- 5-day continuous operation
- Daily routine execution
- System stability validation
- Performance consistency checking

### System Recovery Testing (2 minutes)
- Network outage recovery
- Database failure recovery
- Agent crash recovery
- Market data feed loss recovery

### Load Stress Testing (3 minutes)
- Light load (10 concurrent users)
- Medium load (50 concurrent users)
- Heavy load (100 concurrent users)
- Performance degradation monitoring

### Data Pipeline Endurance (5 minutes)
- Continuous data processing
- Throughput validation
- Memory leak detection
- Processing reliability

### Agent Collaboration Complexity (4 minutes)
- Multi-agent workflow execution
- Complex decision making
- Feedback loop validation
- Conflict resolution testing

## Test Configuration

### Environment Variables
```bash
export ABC_SYSTEM_TEST_MODE=true
export ABC_TEST_DURATION_MULTIPLIER=1.0  # Speed up/slow down tests
export ABC_MOCK_EXTERNAL_SERVICES=true
export ABC_PERFORMANCE_MONITORING=true
```

### Test Data Generation
System tests use realistic simulated data:
- Market prices with realistic volatility
- Trading volumes and patterns
- Multi-asset correlations
- Time-based market events

## Performance Benchmarks

### Expected Test Durations
- `test_complete_market_cycle_simulation`: 120-180 seconds
- `test_multi_day_simulation`: 180-240 seconds
- `test_system_recovery_from_failures`: 90-120 seconds
- `test_load_stress_under_production_conditions`: 150-200 seconds
- `test_data_pipeline_endurance`: 280-320 seconds
- `test_agent_collaboration_complexity`: 200-250 seconds

### Success Criteria
- All tests complete within expected time ranges
- No system crashes or unhandled exceptions
- Performance metrics meet minimum thresholds
- Recovery mechanisms work reliably
- Memory usage remains stable

## CI/CD Integration

### Automated System Testing Pipeline
```yaml
- name: System Tests
  run: |
    # Run system tests with extended timeout
    python -m pytest tests/system/ --timeout=600 --tb=short -q

    # Generate performance report
    python -m pytest tests/system/ --durations=0 > system_performance.txt

    # Validate test results
    if grep -q "FAILED" system_performance.txt; then
      echo "System tests failed!"
      exit 1
    fi
```

### Test Selection for CI
```bash
# Fast system validation (run in PR checks)
python -m pytest tests/system/test_full_system_deployment.py::TestFullSystemDeployment::test_system_recovery_from_failures -v

# Full system validation (run nightly)
python -m pytest tests/system/ --maxfail=3 --timeout=1200
```

## Monitoring & Reporting

### Performance Metrics Collected
- Test execution time
- Memory usage patterns
- CPU utilization
- Agent response times
- Error rates and types
- System recovery times

### Result Analysis
```bash
# Analyze test performance
python -c "
import json
with open('system_test_results.json', 'r') as f:
    results = json.load(f)
    print('Average test duration:', sum(r['duration'] for r in results['tests']) / len(results['tests']))
    print('Success rate:', sum(1 for r in results['tests'] if r['passed']) / len(results['tests']))
"
```

## Troubleshooting

### Common System Test Issues
- **Timeout errors**: Increase `--timeout` parameter or optimize test code
- **Memory issues**: Reduce concurrent operations or increase system memory
- **External service failures**: Ensure mock services are properly configured
- **Slow performance**: Check system resources and optimize test data generation

### Debugging System Tests
```bash
# Run with detailed logging
export LOG_LEVEL=DEBUG
python -m pytest tests/system/ -v -s --log-cli-level=DEBUG

# Profile memory usage
python -m memory_profiler pytest tests/system/test_full_system_deployment.py::TestFullSystemDeployment::test_data_pipeline_endurance

# Run single test with profiling
python -m cProfile -s time pytest tests/system/test_full_system_deployment.py::TestFullSystemDeployment::test_complete_market_cycle_simulation -v
```

## Test Maintenance

### Adding New System Tests
1. Follow naming convention: `test_*.py`
2. Include `@pytest.mark.slow` decorator for long-running tests
3. Implement proper setup/teardown with fixtures
4. Add performance assertions and timing validation
5. Update this README with new test descriptions

### Test Data Management
- Use deterministic random seeds for reproducible results
- Generate realistic market data patterns
- Include edge cases and failure scenarios
- Validate data quality and consistency

## Production Readiness Validation

System tests ensure:
- **Deployment readiness**: Full system can be deployed and operated
- **Performance validation**: System meets performance requirements
- **Reliability testing**: System handles failures gracefully
- **Scalability assessment**: System performs under load
- **Integration validation**: All components work together correctly

These tests provide confidence that the ABC trading system is ready for production deployment and can handle real-world trading conditions.