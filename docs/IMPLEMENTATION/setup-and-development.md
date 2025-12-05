# Setup and Development Guide

## 🛠️ Environment Setup

### System Requirements

#### Hardware Requirements
- **CPU**: 8-core processor (Intel i7/AMD Ryzen 7 or equivalent) minimum, 16-core recommended
- **RAM**: 16GB DDR4 minimum, 32GB+ recommended
- **Storage**: 500GB SSD (NVMe preferred) minimum, 1TB+ recommended
- **Network**: 100Mbps stable internet connection minimum, 1Gbps recommended
- **GPU**: NVIDIA RTX 30-series or equivalent (for ML acceleration)

#### Software Requirements
- **OS**: Ubuntu 22.04 LTS, Windows 11 Pro, or macOS Monterey+
- **Python**: 3.11+
- **Git**: Latest version
- **Code Editor**: VS Code recommended

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

### Initial Setup
```bash
# Clone the repository
git clone https://github.com/nvickers04/ABC-Application.git
cd ABC-Application

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
source myenv/bin/activate  # Unix/Mac

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Copy environment template
cp .env.template .env
# Edit .env with your configuration
```

### Development Tools Setup
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Setup VS Code workspace
code .  # Opens in VS Code
# Install recommended extensions from .vscode/extensions.json
```

### Database Setup
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

## 📋 Development Workflow

### Branching Strategy
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Create bugfix branch
git checkout -b bugfix/issue-description

# Create hotfix branch (from main)
git checkout -b hotfix/critical-fix
```

### Code Standards

#### Python Code Style
- Follow PEP 8 style guide
- Use Black for code formatting
- Use isort for import sorting
- Maximum line length: 88 characters

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

#### Commit Messages
Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

Examples:
```
feat(agent): add new strategy analyzer agent
fix(api): resolve timeout issue in market data calls
docs(readme): update installation instructions
```

### Testing Strategy

#### Test Categories
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **API Tests**: Test external service integrations
- **End-to-End Tests**: Test complete workflows

#### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_agents.py

# Run tests matching pattern
pytest -k "test_agent" -v

# Fast incremental testing
pytest --testmon
```

#### Writing Tests
```python
import pytest
from src.agents.base import BaseAgent

class TestBaseAgent:
    def test_initialization(self):
        """Test agent initialization."""
        agent = BaseAgent()
        assert agent.name is not None

    def test_process_input(self):
        """Test input processing."""
        agent = BaseAgent()
        result = agent.process_input({"test": "data"})
        assert isinstance(result, dict)
```

### Code Review Process

#### Pull Request Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No sensitive data committed
- [ ] Branch up to date with main

#### Review Guidelines
- Focus on code quality and maintainability
- Check for security issues
- Verify test coverage
- Ensure documentation accuracy
- Consider performance implications

## 🏗️ Architecture Guidelines

### Agent Development
```python
from src.agents.base import BaseAgent
from src.utils.a2a_protocol import A2AProtocol

class CustomAgent(BaseAgent):
    """Custom agent implementation."""

    def __init__(self, a2a_protocol: A2AProtocol = None):
        super().__init__(name="custom_agent", a2a_protocol=a2a_protocol)

    async def process_input(self, input_data: dict) -> dict:
        """Process input data and return results."""
        # Implementation here
        return {"result": "processed"}
```

### Utility Module Structure
```python
"""
Utility module for specific functionality.

This module provides utilities for [specific purpose].
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class UtilityClass:
    """Utility class documentation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def utility_method(self, data: Any) -> Any:
        """Process data and return result."""
        return data
```

### Error Handling
```python
from src.utils.exceptions import AgentError, ValidationError

try:
    result = risky_operation()
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    raise
except AgentError as e:
    logger.error(f"Agent operation failed: {e}")
    # Handle gracefully
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise AgentError(f"Operation failed: {e}") from e
```

## 🔧 Debugging and Troubleshooting

### Common Issues

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify package installation
pip list | grep package-name

# Reinstall package
pip uninstall package-name
pip install package-name
```

#### Memory Issues
```python
# Check memory usage
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

#### API Timeouts
```python
# Add timeout handling
import asyncio
from aiohttp import ClientTimeout

timeout = ClientTimeout(total=30)
async with session.get(url, timeout=timeout) as response:
    data = await response.json()
```

### Logging Configuration
```python
import logging
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Log levels
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning messages")
logger.error("Error messages")
logger.critical("Critical errors")
```

## 📊 Performance Optimization

### Profiling Code
```python
import cProfile
import pstats

def profile_function():
    # Code to profile
    pass

# Profile execution
profiler = cProfile.Profile()
profiler.enable()
profile_function()
profiler.disable()

# Print statistics
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Memory Optimization
```python
# Use generators for large datasets
def process_large_dataset(data):
    for item in data:
        yield process_item(item)

# Use context managers
class ResourceManager:
    def __enter__(self):
        self.resource = acquire_resource()
        return self.resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        release_resource(self.resource)
```

## 🔒 Security Best Practices

### API Key Management
```python
# Use environment variables
import os
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API_KEY environment variable not set")

# Never hardcode secrets
# ❌ BAD
api_key = "sk-1234567890abcdef"

# ✅ GOOD
api_key = os.getenv('OPENAI_API_KEY')
```

### Input Validation
```python
from pydantic import BaseModel, validator
from typing import Optional

class UserInput(BaseModel):
    symbol: str
    quantity: int

    @validator('symbol')
    def validate_symbol(cls, v):
        if not v.isalpha() or len(v) > 10:
            raise ValueError('Invalid symbol')
        return v.upper()

    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0 or v > 1000000:
            raise ValueError('Invalid quantity')
        return v
```

### Secure Configuration
```yaml
# config/secure_config.yaml
database:
  host: "${DB_HOST}"
  password: "${DB_PASSWORD}"

api_keys:
  openai: "${OPENAI_API_KEY}"
  ibkr: "${IBKR_API_KEY}"
```

## 📚 Documentation Standards

### Code Documentation
```python
def complex_function(param1: str, param2: int = 0) -> dict:
    """
    Perform complex operation on input parameters.

    This function processes the input parameters through multiple
    transformations and returns a structured result.

    Args:
        param1 (str): Primary input parameter, should be a valid string
        param2 (int, optional): Secondary parameter, defaults to 0

    Returns:
        dict: Result containing processed data with keys:
            - 'result': Main processing result
            - 'metadata': Processing metadata
            - 'errors': Any errors encountered

    Raises:
        ValueError: If param1 is invalid
        ConnectionError: If external service unavailable

    Example:
        >>> result = complex_function("test", 42)
        >>> print(result['result'])
        'processed_test'

    Note:
        This function may take several seconds for large inputs.
        Consider using async version for better performance.
    """
    pass
```

### README Updates
- Update README.md for any API changes
- Add examples for new features
- Update installation instructions
- Document breaking changes

## 🚀 Deployment Strategies

### Option 1: Docker Deployment (Recommended)

#### Dockerfile
```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /home/app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "src/main.py"]
```

#### Docker Compose for Production
```yaml
version: '3.8'

services:
  abc-application:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

#### Build and Deploy
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f abc-application

# Scale the application
docker-compose up -d --scale abc-application=3
```

### Option 2: Direct Server Deployment

#### Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3.11 python3.11-venv python3-pip -y

# Install Redis
sudo apt install redis-server -y
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Create application user
sudo useradd -m -s /bin/bash abc-app

# Create directories
sudo mkdir -p /opt/abc-application
sudo chown abc-app:abc-app /opt/abc-application
```

#### Application Deployment
```bash
# Switch to application user
sudo -u abc-app bash

# Clone repository
cd /opt
git clone https://github.com/nvickers04/ABC-Application.git
cd ABC-Application

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with production values
```

#### Systemd Service
```bash
# Create systemd service file
sudo tee /etc/systemd/system/abc-application.service > /dev/null <<EOF
[Unit]
Description=ABC Application AI Portfolio Manager
After=network.target redis-server.service

[Service]
Type=simple
User=abc-app
WorkingDirectory=/opt/abc-application
Environment=PATH=/opt/abc-application/venv/bin
ExecStart=/opt/abc-application/venv/bin/python src/main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable abc-application
sudo systemctl start abc-application

# Check status
sudo systemctl status abc-application
```

### Horizontal Scaling
```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  abc-application:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
      restart_policy:
        condition: on-failure

  redis-cluster:
    image: redis:7-alpine
    deploy:
      replicas: 3
```

### Production Deployment Checklist
- [ ] Environment variables configured
- [ ] Database connections tested
- [ ] API keys validated
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Backup procedures tested
- [ ] Security hardening applied
- [ ] Systemd service created (for direct deployment)
- [ ] Log rotation configured
- [ ] Firewall rules applied
- [ ] SSL/TLS certificates configured

## 📈 Monitoring and Observability

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
        }
    },
    'loggers': {
        'src.agents': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
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

## 🔧 Configuration Management

### Configuration Files
```yaml
# config/system_config.yaml
system:
  environment: production
  log_level: INFO
  timezone: UTC
  max_workers: 8

# config/agents/data_agent.yaml
data_agent:
  analyzers:
    market_data:
      update_frequency_seconds: 60
      sources: [ibkr, yahoo_finance, alpha_vantage]
```

### Environment Variables
```bash
# .env file
DATABASE_URL=postgresql://grok_user:secure_password@localhost/grok_ibkr
REDIS_URL=redis://localhost:6379
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
OPENAI_API_KEY=your_key_here
```

## 🧪 Testing and Validation

### Unit Testing
```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
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
```

---

*This consolidated guide covers both setup/deployment and development workflows. For production deployment details, see `docs/production_readiness_checklist.md`. For security hardening, see `docs/security_hardening_guide.md`.*