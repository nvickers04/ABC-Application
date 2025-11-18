# [LABEL:DOC:implementation] [LABEL:DOC:topic:deployment] [LABEL:DOC:audience:administrator]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Comprehensive implementation and deployment guide for ABC Application
# Dependencies: All system components, infrastructure requirements
# Related: docs/production_readiness_checklist.md, docs/IMPLEMENTATION/IBKR_PAPER_TRADING_DEPLOYMENT.md
#
# Implementation Guide

## Overview

This section provides comprehensive implementation details for deploying and operating the ABC Application multi-agent trading system. The implementation covers environment setup, configuration management, deployment strategies, and operational procedures.

## System Requirements

### Hardware Requirements

#### Minimum Configuration
- **CPU**: 8-core processor (Intel i7/AMD Ryzen 7 or equivalent)
- **RAM**: 16GB DDR4
- **Storage**: 500GB SSD (NVMe preferred)
- **Network**: 100Mbps stable internet connection

#### Recommended Configuration
- **CPU**: 16-core processor (Intel i9/AMD Ryzen 9 or equivalent)
- **RAM**: 32GB DDR4 (64GB for high-frequency operations)
- **Storage**: 1TB NVMe SSD + 2TB HDD for data archival
- **Network**: 1Gbps fiber connection with low latency
- **GPU**: NVIDIA RTX 30-series or equivalent (for ML acceleration)

### Software Requirements

#### Operating Systems
- **Primary**: Ubuntu 22.04 LTS or Windows 11 Pro
- **Alternative**: macOS Monterey or later (development only)
- **Container**: Docker 24.0+ with Docker Compose

#### Core Dependencies
```bash
# Python Environment
Python 3.11+
pip 23.0+
virtualenv 20.0+

# Security and Secrets Management
hvac>=2.0.0,<3.0.0        # HashiCorp Vault client
cryptography>=46.0.0,<47.0 # Encryption support

# System Tools
Redis 7.0+
PostgreSQL 15+
Node.js 18+ (for monitoring dashboard)

# Trading Platforms
IBKR TWS/Gateway 10.19+
Interactive Brokers API

# Data Sources
yfinance 0.2+
alpha_vantage 2.3+
fredapi 0.5+
newsapi 0.1+
tweepy 4.14+
```

## Environment Setup

### Development Environment

#### 1. Clone Repository
```bash
git clone https://github.com/your-org/abc-application.git
cd abc-application
```

#### 2. Python Environment Setup
```bash
# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate  # Linux/Mac
# myenv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

#### 3. Database Setup
```bash
# Install Redis
sudo apt-get install redis-server  # Ubuntu
# brew install redis               # macOS

# Start Redis service
sudo systemctl start redis-server

# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb grok_ibkr
sudo -u postgres psql -c "CREATE USER grok_user WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE grok_ibkr TO grok_user;"
```

#### 4. HashiCorp Vault Setup (Security Critical)
```bash
# Download Vault for Windows
# From: https://developer.hashicorp.com/vault/downloads
# Extract to C:\vault\

# Create Vault configuration
New-Item -ItemType Directory -Path vault-data -Force
@"
storage "file" {
  path = "./vault-data"
}

listener "tcp" {
  address = "127.0.0.1:8200"
  tls_disable = "true"  # Enable TLS in production
}

api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"
"@ | Out-File -FilePath vault-config.hcl -Encoding UTF8

# Start Vault in development mode
C:\vault\vault.exe server -dev -config=vault-config.hcl

# In another terminal, set environment and initialize
$env:VAULT_ADDR = "http://127.0.0.1:8200"
$env:VAULT_TOKEN = "dev-token-from-startup-output"

# Enable KV v2 secrets engine
vault secrets enable -path=secret kv-v2

# Import existing secrets (run the import script)
python tools/import_env_to_vault.py
```

#### 5. Redis Security Configuration
```bash
# Install Redis for Windows
# Download from: https://redis.io/download
# Or use chocolatey: choco install redis-64

# Configure Redis security
# Edit redis.windows.conf
@"
bind 127.0.0.1
port 6379
requirepass SecureRedisPass2025!
maxmemory 256mb
maxmemory-policy allkeys-lru

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command SHUTDOWN SHUTDOWN_REDIS
"@ | Out-File -FilePath redis\redis.windows.conf -Encoding UTF8

# Start Redis with config
redis-server.exe redis\redis.windows.conf
```

### Production Environment

#### Docker Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  abc-application:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://grok_user:password@postgres/grok_ibkr
    depends_on:
      - redis
      - postgres
    volumes:
      - ./config:/app/config
      - ./data:/app/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: grok_ibkr
      POSTGRES_USER: grok_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  redis_data:
  postgres_data:
```

#### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: abc-application
spec:
  replicas: 3
  selector:
    matchLabels:
      app: abc-application
  template:
    metadata:
      labels:
        app: abc-application
    spec:
      containers:
      - name: abc-application
        image: abc-application:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: database_url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Configuration Management

### Core Configuration Files

#### System Configuration (`config/system_config.yaml`)
```yaml
# System-wide settings
system:
  environment: production
  log_level: INFO
  timezone: UTC
  max_workers: 8

# Agent configuration
agents:
  max_concurrent_operations: 10
  message_timeout_seconds: 300
  retry_attempts: 3

# Trading parameters
trading:
  max_position_size: 100000
  max_daily_loss: 50000
  commission_model: ibkr_standard

# Data sources
data:
  cache_ttl_hours: 24
  max_api_calls_per_minute: 100
  fallback_providers: [alpha_vantage, yahoo_finance]
```

#### Agent-Specific Configuration
```yaml
# config/agents/data_agent.yaml
data_agent:
  analyzers:
    market_data:
      update_frequency_seconds: 60
      sources: [ibkr, yahoo_finance, alpha_vantage]
    sentiment:
      apis: [newsapi, twitter]
      processing_batch_size: 100
    economic:
      indicators: [GDP, CPI, Unemployment]
      update_frequency_hours: 24

# config/agents/strategy_agent.yaml
strategy_agent:
  analyzers:
    options: {enabled: true, max_complexity: 3}
    ml_models: {enabled: true, retrain_frequency_days: 7}
    pairs_trading: {enabled: true, correlation_threshold: 0.8}
    arbitrage: {enabled: true, min_spread_bps: 5}
```

#### Risk Configuration (`config/risk_config.yaml`)
```yaml
# Risk management settings
risk_limits:
  max_portfolio_var: 0.15  # 15% Value at Risk
  max_single_position: 0.10  # 10% of portfolio
  max_sector_exposure: 0.25  # 25% sector concentration
  max_daily_drawdown: 0.05  # 5% daily loss limit

# Stress testing scenarios
stress_tests:
  market_crash: {equity_drop: 0.20, vol_increase: 2.0}
  volatility_spike: {vix_increase: 50, correlation_breakdown: true}
  liquidity_crisis: {bid_ask_spread_increase: 5.0}

# Compliance settings
compliance:
  reg_t_margin: true
  pattern_day_trading_rules: true
  wash_sale_rules: true
```

### Environment Variables
```bash
# .env file (to be imported to Vault)
# Database
DATABASE_URL=postgresql://grok_user:secure_password@localhost/grok_ibkr
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=SecureRedisPass2025!

# HashiCorp Vault
VAULT_ADDR=http://127.0.0.1:8200
VAULT_TOKEN=your_vault_token_here

# IBKR API
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# API Keys (will be migrated to Vault)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWSAPI_KEY=your_newsapi_key
TWITTER_BEARER_TOKEN=your_twitter_token

# Discord Bot Tokens (will be migrated to Vault)
DISCORD_ORCHESTRATOR_TOKEN=your_orchestrator_token
DISCORD_DATA_AGENT_TOKEN=your_data_agent_token
DISCORD_STRATEGY_AGENT_TOKEN=your_strategy_agent_token
DISCORD_RISK_AGENT_TOKEN=your_risk_agent_token
DISCORD_EXECUTION_AGENT_TOKEN=your_execution_agent_token
DISCORD_REFLECTION_AGENT_TOKEN=your_reflection_agent_token
DISCORD_MACRO_AGENT_TOKEN=your_macro_agent_token
GUILD_ID=your_discord_guild_id

# System
LOG_LEVEL=INFO
SECRET_KEY=your_secret_key_for_sessions
```

## Deployment Strategies

### Development Deployment
```bash
# Quick start for development
make setup-dev
make run-dev

# Or manually
python -m src.main --config config/dev_config.yaml
```

### Production Deployment
```bash
# Build and deploy
make build
make deploy-prod

# Or with Docker
docker-compose up -d

# Or with Kubernetes
kubectl apply -f k8s/
```

### Blue-Green Deployment
```bash
# Deploy new version alongside existing
kubectl set image deployment/abc-application abc-application=abc-application:v2.0

# Test new version
curl -H "Host: abc-application-new.example.com" http://load-balancer/health

# Switch traffic
kubectl patch service abc-application -p '{"spec":{"selector":{"version":"v2.0"}}}'
```

## Monitoring and Observability

### Health Checks
```python
# src/health_check.py
from fastapi import FastAPI
from src.monitoring.health import HealthChecker

app = FastAPI()

@app.get("/health")
async def health_check():
    checker = HealthChecker()
    return await checker.check_all()

@app.get("/health/detailed")
async def detailed_health():
    checker = HealthChecker()
    return await checker.detailed_check()
```

### Metrics Collection
```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Performance metrics
REQUEST_COUNT = Counter('request_count', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')

# Trading metrics
TRADES_EXECUTED = Counter('trades_executed_total', 'Total trades executed')
PNL_REALIZED = Gauge('pnl_realized', 'Realized P&L')
PORTFOLIO_VALUE = Gauge('portfolio_value', 'Current portfolio value')

# Agent metrics
AGENT_MESSAGES = Counter('agent_messages_total', 'Agent messages', ['agent_type', 'message_type'])
AGENT_ERRORS = Counter('agent_errors_total', 'Agent errors', ['agent_type', 'error_type'])
```

### Logging Configuration
```python
# src/logging_config.py
import logging
from logging.config import dictConfig

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logs/grok_ibkr.log',
            'formatter': 'detailed',
            'level': 'DEBUG'
        },
        'agent_file': {
            'class': 'logging.FileHandler',
            'filename': 'logs/agents.log',
            'formatter': 'detailed',
            'level': 'DEBUG'
        }
    },
    'loggers': {
        'src.agents': {
            'handlers': ['console', 'agent_file'],
            'level': 'DEBUG',
            'propagate': False
        },
        'src.trading': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO'
    }
}

def setup_logging():
    dictConfig(LOGGING_CONFIG)
```

## Testing and Validation

### Unit Testing
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html

# Run specific agent tests
pytest tests/unit/test_data_agent.py -v
```

### Integration Testing
```bash
# Run integration tests
pytest tests/integration/ -v --tb=short

# Test with mock IBKR API
pytest tests/integration/test_trading_execution.py --mock-ibkr
```

### Performance Testing
```bash
# Load testing
locust -f tests/performance/locustfile.py

# Stress testing
pytest tests/performance/stress_test.py

# Memory profiling
mprof run src/main.py
mprof plot
```

### Backtesting Validation
```python
# tests/backtesting/validation.py
import pytest
from src.backtesting.engine import BacktestingEngine
from src.strategies.test_strategy import TestStrategy

def test_strategy_performance():
    engine = BacktestingEngine()
    strategy = TestStrategy()

    results = engine.run_backtest(
        strategy=strategy,
        start_date='2023-01-01',
        end_date='2024-01-01',
        initial_capital=100000
    )

    assert results.total_return > 0
    assert results.sharpe_ratio > 1.0
    assert results.max_drawdown < 0.20
```

## Security Considerations

### API Key Management
```python
# src/security/secrets.py
import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

class SecretManager:
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.client = SecretClient(
            vault_url="https://abc-application-keyvault.vault.azure.net/",
            credential=self.credential
        )

    def get_secret(self, name: str) -> str:
        secret = self.client.get_secret(name)
        return secret.value

# Usage
secrets = SecretManager()
api_key = secrets.get_secret('alpha-vantage-api-key')
```

### Network Security
```nginx
# nginx.conf for API gateway
server {
    listen 443 ssl http2;
    server_name api.abc-application.com;

    ssl_certificate /etc/ssl/certs/grok_ibkr.crt;
    ssl_certificate_key /etc/ssl/private/grok_ibkr.key;

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Rate limiting
        limit_req zone=api burst=10 nodelay;

        # Authentication
        auth_request /auth;
    }

    location /auth {
        proxy_pass http://localhost:8001;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URI $request_uri;
    }
}
```

### Data Encryption
```python
# src/security/encryption.py
from cryptography.fernet import Fernet
import base64
import os

class DataEncryption:
    def __init__(self, key: bytes = None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt_data(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        return self.cipher.decrypt(encrypted_data.encode()).decode()

# Database field encryption
from sqlalchemy import TypeDecorator, String
from sqlalchemy.engine import Dialect

class EncryptedString(TypeDecorator):
    impl = String

    def __init__(self, encryption_key: str, **kwargs):
        super().__init__(**kwargs)
        self.encryption = DataEncryption(encryption_key.encode())

    def process_bind_param(self, value: str, dialect: Dialect) -> str:
        if value is not None:
            return self.encryption.encrypt_data(value)
        return value

    def process_result_value(self, value: str, dialect: Dialect) -> str:
        if value is not None:
            return self.encryption.decrypt_data(value)
        return value
```

## Backup and Recovery

### Database Backup
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/abc_application"

# PostgreSQL backup
pg_dump -U abc_user -h localhost abc_application > $BACKUP_DIR/postgres_$DATE.sql

# Redis backup
redis-cli --rdb $BACKUP_DIR/redis_$DATE.rdb

# Configuration backup
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /app/config/

# Retention policy (keep last 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery
```yaml
# disaster_recovery.yaml
recovery:
  rto: 4  # Recovery Time Objective: 4 hours
  rpo: 1  # Recovery Point Objective: 1 hour

  procedures:
    - name: database_restore
      steps:
        - stop_application
        - restore_postgresql_from_backup
        - restore_redis_from_backup
        - validate_data_integrity
        - start_application

    - name: full_system_restore
      steps:
        - provision_infrastructure
        - restore_configurations
        - restore_database
        - deploy_application
        - run_health_checks
        - enable_traffic

  testing:
    quarterly_dr_drill: true
    backup_integrity_checks: daily
    failover_testing: monthly
```

## Performance Optimization

### Database Optimization
```sql
-- PostgreSQL performance tuning
-- postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

-- Indexing strategy
CREATE INDEX CONCURRENTLY idx_trades_symbol_time ON trades (symbol, timestamp);
CREATE INDEX CONCURRENTLY idx_portfolio_positions_symbol ON portfolio_positions (symbol);
CREATE INDEX CONCURRENTLY idx_market_data_symbol_time ON market_data (symbol, timestamp DESC);

-- Partitioning for large tables
CREATE TABLE trades_y2024m01 PARTITION OF trades
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### Caching Strategy
```python
# src/caching/cache_manager.py
from cachetools import TTLCache, LRUCache
import redis
import json

class CacheManager:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.local_cache = TTLCache(maxsize=1000, ttl=300)  # 5 min TTL

    def get_market_data(self, symbol: str, timeframe: str) -> dict:
        cache_key = f"market_data:{symbol}:{timeframe}"

        # Check local cache first
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]

        # Check Redis
        cached_data = self.redis.get(cache_key)
        if cached_data:
            data = json.loads(cached_data)
            self.local_cache[cache_key] = data
            return data

        # Fetch from source
        data = self._fetch_market_data(symbol, timeframe)

        # Cache in Redis (1 hour TTL)
        self.redis.setex(cache_key, 3600, json.dumps(data))
        self.local_cache[cache_key] = data

        return data
```

### Async Processing
```python
# src/async_processing/task_queue.py
import asyncio
from celery import Celery
from src.agents.data_agent import DataAgent

app = Celery('grok_ibkr', broker='redis://localhost:6379/0')

@app.task
def process_market_data_batch(symbols: list, timeframe: str):
    """Process market data for multiple symbols asynchronously"""
    agent = DataAgent()
    results = []

    for symbol in symbols:
        try:
            data = agent.fetch_market_data(symbol, timeframe)
            agent.store_data(data)
            results.append({'symbol': symbol, 'status': 'success'})
        except Exception as e:
            results.append({'symbol': symbol, 'status': 'error', 'error': str(e)})

    return results

# Usage
from .task_queue import process_market_data_batch

# Queue async processing
result = process_market_data_batch.delay(['AAPL', 'GOOGL', 'MSFT'], '1d')
```

## Troubleshooting Guide

### Common Issues

#### Agent Communication Failures
```python
# Diagnostic script
def diagnose_agent_communication():
    # Check Redis connectivity
    try:
        redis.ping()
        print("✓ Redis connection OK")
    except:
        print("✗ Redis connection failed")

    # Check agent health
    for agent in agents:
        try:
            response = agent.health_check()
            if response.status == 'healthy':
                print(f"✓ {agent.name} healthy")
            else:
                print(f"✗ {agent.name} unhealthy: {response.message}")
        except:
            print(f"✗ {agent.name} unreachable")
```

#### Memory Issues
```python
# Memory monitoring
import psutil
import gc

def check_memory_usage():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}%")

    if memory.percent > 85:
        print("High memory usage detected, triggering garbage collection")
        gc.collect()

    # Check for memory leaks in agents
    for agent in agents:
        if hasattr(agent, 'memory_usage'):
            usage = agent.memory_usage()
            if usage > 500 * 1024 * 1024:  # 500MB
                print(f"Warning: {agent.name} using {usage/1024/1024:.1f}MB")
```

#### Trading Execution Delays
```python
# Performance monitoring
def monitor_execution_latency():
    execution_times = []

    for trade in recent_trades:
        latency = trade.execution_time - trade.submission_time
        execution_times.append(latency.total_seconds())

    avg_latency = sum(execution_times) / len(execution_times)
    max_latency = max(execution_times)

    print(f"Average execution latency: {avg_latency:.2f}s")
    print(f"Maximum execution latency: {max_latency:.2f}s")

    if avg_latency > 5.0:  # More than 5 seconds
        print("Warning: High execution latency detected")
```

### Log Analysis
```bash
# Search for errors
grep "ERROR" logs/grok_ibkr.log | tail -20

# Analyze agent communication
grep "agent.*message" logs/agents.log | awk '{print $1, $7}' | sort | uniq -c | sort -nr

# Check trading performance
grep "TRADE_EXECUTED" logs/trading.log | jq -r '.pnl' | awk '{sum+=$1} END {print "Total P&L:", sum}'
```

## Maintenance Procedures

### Regular Maintenance Tasks
```bash
# Daily tasks
make backup-db
make update-market-data
make run-health-checks

# Weekly tasks
make optimize-database
make retrain-ml-models
make security-scan

# Monthly tasks
make compliance-audit
make performance-review
make update-dependencies
```

### Version Upgrades
```bash
# Safe upgrade procedure
# 1. Create backup
make full-backup

# 2. Test upgrade in staging
make deploy-staging
make run-integration-tests

# 3. Deploy to production with rollback plan
make deploy-prod-blue
make switch-traffic
make monitor-post-deploy

# 4. Verify and cleanup
make verify-deployment
make cleanup-old-version
```

## Security Validation and Testing

### Post-Setup Security Checks
```bash
# 1. Verify Vault connectivity
python -c "
from src.utils.vault_client import get_vault_secret
try:
    token = get_vault_secret('DISCORD_ORCHESTRATOR_TOKEN')
    print('✓ Vault connection successful')
except Exception as e:
    print(f'✗ Vault connection failed: {e}')
"

# 2. Test Redis authentication
python -c "
import redis
r = redis.Redis(host='127.0.0.1', port=6379, password='SecureRedisPass2025!')
try:
    r.ping()
    print('✓ Redis authentication successful')
except Exception as e:
    print(f'✗ Redis authentication failed: {e}')
"

# 3. Run security-focused unit tests
python -m pytest unit-tests/ -k "vault or redis or security" -v
```

### Security Migration Verification
After completing the security hardening:

1. **Confirm no hardcoded secrets** in source code
2. **Verify Vault stores all sensitive credentials**
3. **Test Redis requires authentication**
4. **Run integration tests** with secure configurations
5. **Validate error handling** for connection failures

### Production Security Checklist
- [ ] Vault running with proper TLS configuration
- [ ] Redis bound to localhost with authentication
- [ ] All API keys migrated to Vault
- [ ] Environment variables encrypted or removed
- [ ] Firewall configured to restrict access
- [ ] SSH hardened (key-based auth only)
- [ ] Regular security updates scheduled
- [ ] Audit logging enabled
- [ ] Backup encryption configured