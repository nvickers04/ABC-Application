# Test Environment Setup

This directory contains scripts and configurations for setting up test environments that mirror production for comprehensive integration testing.

## Overview

The test environment provides:
- **Redis**: In-memory data store and caching
- **PostgreSQL**: Relational database for structured data
- **Mock IBKR TWS**: Simulated Interactive Brokers API server
- **Health Monitor**: System health monitoring and metrics
- **Test Runner**: Automated test execution environment

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Python 3.11+
- At least 4GB RAM available

### Start Test Environment
```bash
# From project root
python setup/setup_test_environment.py --action start

# Or using Docker Compose directly
docker-compose -f setup/docker-compose.test.yml up -d
```

### Run Integration Tests
```bash
# Run tests in the environment
python setup/setup_test_environment.py --action test

# Run tests with custom arguments
python setup/setup_test_environment.py --action test --test-args --verbose --coverage
```

### Stop Test Environment
```bash
python setup/setup_test_environment.py --action stop
```

## Environment Configurations

### Test Environment (`config/environments/test.yaml`)
- **Purpose**: Full integration testing with mocked external services
- **IBKR**: Mock TWS server for API testing
- **Database**: Isolated PostgreSQL instance
- **Monitoring**: Full health monitoring enabled
- **Alerts**: Disabled (test environment)

### Staging Environment (`config/environments/staging.yaml`)
- **Purpose**: Pre-production validation
- **IBKR**: Paper trading account
- **Database**: Staging database instance
- **Monitoring**: Enhanced monitoring with alerts
- **Security**: Production-like security settings

## Services

### Redis
- **Port**: 6379
- **Purpose**: Caching, session storage, real-time data
- **Health Check**: `redis-cli ping`

### PostgreSQL
- **Port**: 5432
- **Database**: abc_test
- **Credentials**: abc_user / test_password
- **Health Check**: `pg_isready`

### Mock IBKR TWS
- **Port**: 7497
- **Purpose**: Simulate IBKR API responses
- **Features**:
  - Account information
  - Market data quotes
  - Order submission
  - Connection management

### Health Monitor
- **Port**: 8080
- **Endpoints**:
  - `/health`: Overall system health
  - `/health/components`: Component status
  - `/health/system`: System metrics
  - `/health/api`: API health
  - `/metrics`: Prometheus metrics

## Testing

### Running Tests Manually
```bash
# Start environment
python setup/setup_test_environment.py --action start

# Wait for services
python setup/setup_test_environment.py --action start --no-wait

# Run specific tests
pytest integration-tests/test_health_api_integration.py -v

# Run with coverage
pytest integration-tests/ --cov=src --cov-report=html
```

### Automated Testing
The test environment supports:
- **Unit Tests**: Fast, isolated component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Load and scalability testing
- **Security Tests**: Vulnerability scanning

### Test Data
Test data is automatically created and cleaned up. The environment includes:
- Sample market data
- Mock user accounts
- Test trading scenarios
- Historical price data

## Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check Docker status
docker ps -a

# View service logs
docker-compose -f setup/docker-compose.test.yml logs [service_name]

# Restart services
docker-compose -f setup/docker-compose.test.yml restart
```

#### Port Conflicts
If ports are already in use:
```bash
# Check port usage
netstat -tulpn | grep :[port]

# Modify docker-compose.test.yml to use different ports
# Then restart: docker-compose -f setup/docker-compose.test.yml up -d
```

#### Database Connection Issues
```bash
# Connect to PostgreSQL
docker exec -it abc-test_postgres_1 psql -U abc_user -d abc_test

# Reset database
docker-compose -f setup/docker-compose.test.yml down -v
docker-compose -f setup/docker-compose.test.yml up -d
```

#### Test Failures
```bash
# Run tests with detailed output
pytest integration-tests/ -v -s --tb=long

# Run specific failing test
pytest integration-tests/test_unified_workflow_integration.py::TestUnifiedWorkflowIntegration::test_orchestrator_initialization -v
```

### Environment Variables
Override default settings:
```bash
# Custom Redis host
export REDIS_HOST=custom.redis.host

# Custom database credentials
export POSTGRES_PASSWORD=my_password

# Run tests
python setup/setup_test_environment.py --action test
```

## Development

### Adding New Services
1. Add service definition to `docker-compose.test.yml`
2. Create Dockerfile in `setup/`
3. Update environment configurations
4. Add health checks
5. Update this documentation

### Modifying Mock Services
The mock IBKR server (`setup/mock_ibkr_server.py`) can be extended to:
- Add new API endpoints
- Simulate error conditions
- Provide realistic test data
- Support different market scenarios

### Environment Scaling
For larger test environments:
- Increase Docker resource limits
- Add more mock services
- Implement service discovery
- Add load balancing

## Security Considerations

### Test Environment Security
- **Network Isolation**: Services communicate through Docker networks
- **Credential Management**: Test credentials are clearly marked
- **Data Isolation**: Test data is separate from production
- **Access Control**: No external access to test services

### Production Parity
The test environment aims to match production:
- Same service versions
- Similar configuration structure
- Equivalent security settings
- Realistic data volumes

## Monitoring and Logging

### Service Logs
```bash
# View all service logs
docker-compose -f setup/docker-compose.test.yml logs -f

# View specific service logs
docker-compose -f setup/docker-compose.test.yml logs redis
```

### Health Monitoring
Access the health dashboard at: http://localhost:8080/docs

### Performance Metrics
Prometheus metrics available at: http://localhost:8080/metrics

## Cleanup

### Remove Test Environment
```bash
# Stop and remove containers
python setup/setup_test_environment.py --action stop

# Remove volumes (WARNING: deletes all data)
docker-compose -f setup/docker-compose.test.yml down -v

# Remove images
docker-compose -f setup/docker-compose.test.yml down --rmi all
```

### Reset Test Data
```bash
# Reset database
docker-compose -f setup/docker-compose.test.yml exec postgres psql -U abc_user -d abc_test -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# Clear Redis
docker-compose -f setup/docker-compose.test.yml exec redis redis-cli FLUSHALL
```</content>
</xai:function_call name="manage_todo_list">
<parameter name="todoList">[{"id":4,"title":"Create test environments mirroring production","status":"completed"}]