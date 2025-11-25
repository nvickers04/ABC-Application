# Performance Testing Suite

This directory contains load testing and performance validation scripts for the ABC trading system.

## Overview

The performance tests simulate realistic 24/6 trading conditions with:
- Concurrent API calls from multiple agents
- High-volume data ingestion
- Multi-agent workflow processing
- Realistic trading patterns and frequencies
- Load testing using Locust framework

## Test Scenarios

### 1. Load Testing with Locust
- **File**: `locustfile.py`
- **Purpose**: Simulate concurrent users performing trading operations
- **Scenarios**:
  - Data collection (30% of requests)
  - Strategy analysis (20%)
  - Risk analysis (15%)
  - Trade execution (10%)
  - Multi-agent workflows (5%)
  - Health monitoring (20%)

### 2. User Types
- **TradingSystemUser**: Standard trading operations
- **HighFrequencyUser**: High-frequency trading patterns (10 req/sec)
- **BatchProcessingUser**: Heavy batch processing workloads

## Running the Tests

### Prerequisites
```bash
pip install locust pandas numpy
```

### Basic Load Test
```bash
# Run with web interface
locust -f tests/performance/locustfile.py

# Run headless with 100 users, hatch rate of 10 users/second
locust -f tests/performance/locustfile.py --headless -u 100 -r 10 --run-time 5m
```

### Advanced Scenarios
```bash
# High-frequency trading simulation
locust -f tests/performance/locustfile.py --headless -u 50 -r 5 --run-time 10m --class-picker HighFrequencyUser

# Batch processing load
locust -f tests/performance/locustfile.py --headless -u 20 -r 2 --run-time 15m --class-picker BatchProcessingUser
```

## Performance Metrics

### Target Performance
- **Response Time**: <500ms for data requests, <2s for analysis
- **Throughput**: 100+ concurrent users
- **Error Rate**: <1%
- **Memory Usage**: <2GB per agent process
- **CPU Usage**: <80% during peak load

### Monitoring
- Response times by request type
- Error rates and failure patterns
- Resource utilization
- Agent coordination efficiency

## Configuration

### Environment Variables
```bash
export ABC_PERFORMANCE_MODE=true
export ABC_MAX_CONCURRENT_REQUESTS=100
export ABC_MEMORY_LIMIT=2GB
```

### Test Parameters
- **Concurrent Users**: 10-500 (configurable)
- **Ramp-up Rate**: 1-50 users/second
- **Test Duration**: 1-60 minutes
- **Request Mix**: Configurable via task weights

## Results Analysis

### Key Metrics to Monitor
1. **P95 Response Time**: Should remain under 2 seconds
2. **Request Success Rate**: Target >99%
3. **Memory Leak Detection**: Monitor for gradual memory increase
4. **Agent Coordination**: Ensure no deadlock conditions

### Sample Results
```
Type     Name                 # reqs    # fails    Avg     Min     Max    Med    |  req/s  failures/s
-------- ------------------- -------- -------- ------- ------- ------- ------- | ------- -----------
DATA     data_collection        1500      5      450     120    1200    400   |   30.0       0.1
STRATEGY strategy_analysis       900      2      890     300    2100    850   |   18.0       0.0
RISK     risk_analysis           675      1     1200     500    2800   1150   |   13.5       0.0
EXEC     trade_execution         450      3      340     100     800    320   |    9.0       0.1
WORKFLOW multi_agent_workflow    225      0     2500    1200    4800   2400   |    4.5       0.0
HEALTH   health_check            900      0      150      50     400    140   |   18.0       0.0
-------- ------------------- -------- -------- ------- ------- ------- ------- | ------- -----------
         Aggregated             4650     11      650     50    4800    520   |   93.0       0.2
```

## Integration with CI/CD

Add to your CI/CD pipeline:

```yaml
- name: Performance Tests
  run: |
    locust -f tests/performance/locustfile.py --headless -u 50 -r 5 --run-time 3m --csv results
    # Analyze results and fail if thresholds exceeded
```

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce concurrent users or increase system memory
2. **Timeout Errors**: Check agent responsiveness and network connectivity
3. **High Error Rates**: Investigate agent failures and error handling

### Debugging
```bash
# Run with detailed logging
locust -f tests/performance/locustfile.py --loglevel DEBUG

# Profile memory usage
python -m memory_profiler tests/performance/locustfile.py
```