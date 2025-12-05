# Troubleshooting Guide

## 🔍 Common Issues and Solutions

This guide covers frequently encountered issues when running ABC-Application and their resolutions.

## 🚫 Application Won't Start

### Issue: Import Errors
**Symptoms:**
```
ModuleNotFoundError: No module named 'langchain'
ImportError: cannot import name 'OpenAI' from 'langchain.llms'
```

**Solutions:**
```bash
# Check Python version
python --version  # Should be 3.11+

# Reinstall requirements
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Check virtual environment
which python  # Should point to venv
pip list | grep langchain  # Should show installed packages
```

### Issue: Configuration Errors
**Symptoms:**
```
ValueError: OPENAI_API_KEY not found
KeyError: 'redis_url'
```

**Solutions:**
```bash
# Check environment file
ls -la .env  # File should exist

# Validate .env content
cat .env | grep -E "(OPENAI_API_KEY|REDIS_URL)"  # Should show values

# Copy template if missing
cp .env.template .env
# Edit .env with your values
```

### Issue: Port Already in Use
**Symptoms:**
```
OSError: [Errno 48] Address already in use
Connection refused on port 8000
```

**Solutions:**
```bash
# Find process using port
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows

# Change port in configuration
export PORT=8001
```

## 🔄 Agent Communication Issues

### Issue: A2A Protocol Not Working
**Symptoms:**
```
A2AProtocol initialization failed
No agent responses in workflow
Timeout waiting for agent communication
```

**Solutions:**
```python
# Check agent registration
from src.utils.a2a_protocol import A2AProtocol
a2a = A2AProtocol()
print(f"Registered agents: {len(a2a.agents)}")
print(f"Agent queues: {list(a2a.agent_queues.keys())}")

# Test basic communication
import asyncio
async def test_a2a():
    await a2a.register_agent("test_agent")
    print("A2A basic functionality works")
asyncio.run(test_a2a())
```

### Issue: Agent Processing Timeout
**Symptoms:**
```
Agent processing timeout after 30 seconds
Workflow stuck at agent step
```

**Solutions:**
```python
# Increase timeout in configuration
# In .env or config
AGENT_TIMEOUT=60  # Increase from default 30

# Check agent health
from src.agents.base import BaseAgent
agent = BaseAgent()
health = await agent.health_check()
print(f"Agent health: {health}")
```

## 📊 Data and API Issues

### Issue: Market Data API Failures
**Symptoms:**
```
API rate limit exceeded
Connection timeout to market data
Invalid API response format
```

**Solutions:**
```python
# Check API health
from src.utils.api_health_monitor import check_api_health
health_status = await check_api_health()
print(f"API Health: {health_status}")

# Implement circuit breaker
from src.utils.circuit_breaker import CircuitBreaker
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

@breaker
async def call_api():
    # Your API call here
    pass
```

### Issue: Redis Connection Issues
**Symptoms:**
```
ConnectionError: Error 111 connecting to redis:6379
Redis connection timeout
Memory cache not working
```

**Solutions:**
```bash
# Check Redis service
redis-cli ping  # Should return PONG

# Start Redis if not running
sudo systemctl start redis-server  # Linux
brew services start redis  # Mac

# Check Redis configuration
redis-cli config get maxmemory
redis-cli config get maxmemory-policy
```

## 💰 Trading and Execution Issues

### Issue: IBKR Connection Problems
**Symptoms:**
```
IBKR API connection failed
TWS/Gateway not responding
Authentication failed
```

**Solutions:**
```python
# Check IBKR configuration
from src.integrations.ibkr import IBKRIntegration
ibkr = IBKRIntegration()
connection_status = await ibkr.test_connection()
print(f"IBKR Connection: {connection_status}")

# Validate credentials
# Check .env for IBKR_API_KEY, IBKR_HOST, IBKR_PORT
```

### Issue: Trade Execution Failures
**Symptoms:**
```
Order rejected by broker
Insufficient funds
Invalid order parameters
```

**Solutions:**
```python
# Validate order parameters
from src.agents.execution import ExecutionAgent
agent = ExecutionAgent()

# Test order validation
test_order = {
    "symbol": "SPY",
    "quantity": 100,
    "order_type": "MKT",
    "action": "BUY"
}

validation_result = await agent.validate_order(test_order)
print(f"Order validation: {validation_result}")
```

## 🔄 Workflow Execution Problems

### Issue: Workflow Not Starting
**Symptoms:**
```
Workflow initialization failed
No workflow orchestration
Agent coordination not working
```

**Solutions:**
```python
# Check workflow configuration
from src.agents.unified_workflow_orchestrator import UnifiedWorkflowOrchestrator
orchestrator = UnifiedWorkflowOrchestrator()

# Validate initialization
success = await orchestrator.initialize()
print(f"Orchestrator initialized: {success}")

# Check agent registration
print(f"Available agents: {list(orchestrator.agents.keys())}")
```

### Issue: Workflow Hanging
**Symptoms:**
```
Workflow stuck at specific step
No progress after agent execution
Timeout in workflow processing
```

**Solutions:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check workflow state
workflow_state = await orchestrator.get_workflow_state()
print(f"Current state: {workflow_state}")

# Manual step execution
await orchestrator.execute_step("data_agent")
```

## 🔍 Memory and Performance Issues

### Issue: High Memory Usage
**Symptoms:**
```
Memory usage > 90%
Out of memory errors
Application slowdown
```

**Solutions:**
```python
# Check memory usage
import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
print(f"VMS: {memory_info.vms / 1024 / 1024:.1f} MB")

# Force garbage collection
import gc
gc.collect()

# Check for memory leaks
from src.utils.memory_profiler import profile_memory
profile_memory()
```

### Issue: Slow Performance
**Symptoms:**
```
Operations taking >30 seconds
API calls timing out
High CPU usage
```

**Solutions:**
```python
# Profile performance
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Run slow operation
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## 📝 Logging and Debugging

### Issue: Insufficient Logging
**Symptoms:**
```
No error logs
Hard to debug issues
Unknown failure points
```

**Solutions:**
```python
# Enable debug logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add file logging
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(file_handler)
```

### Issue: Log Files Too Large
**Symptoms:**
```
Log files growing too large
Disk space issues
Hard to find recent logs
```

**Solutions:**
```python
# Implement log rotation
import logging.handlers

# Rotating file handler
handler = logging.handlers.RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
logging.getLogger().addHandler(handler)

# Compress old logs
import gzip
import glob

def compress_old_logs():
    for log_file in glob.glob('app.log.*'):
        if not log_file.endswith('.gz'):
            with open(log_file, 'rb') as f_in:
                with gzip.open(f'{log_file}.gz', 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(log_file)
```

## 🔧 Configuration Issues

### Issue: Environment Variables Not Loading
**Symptoms:**
```
Environment variables not recognized
Configuration not applied
Application using wrong settings
```

**Solutions:**
```bash
# Check environment loading
python -c "import os; print(os.environ.get('OPENAI_API_KEY', 'NOT SET'))"

# Validate .env file format
cat .env | head -10  # Check format

# Load .env explicitly
from dotenv import load_dotenv
load_dotenv()
print("Environment loaded")
```

### Issue: YAML Configuration Errors
**Symptoms:**
```
YAML syntax error
Configuration not loading
Invalid configuration values
```

**Solutions:**
```python
# Validate YAML syntax
import yaml
try:
    with open('config/risk_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("YAML is valid")
except yaml.YAMLError as e:
    print(f"YAML error: {e}")

# Check configuration values
from pydantic import BaseModel, ValidationError

class ConfigModel(BaseModel):
    max_drawdown: float
    risk_per_trade: float

try:
    config_obj = ConfigModel(**config)
    print("Configuration is valid")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## 🌐 Network and Connectivity Issues

### Issue: External API Timeouts
**Symptoms:**
```
Connection timeout
SSL certificate errors
DNS resolution failures
```

**Solutions:**
```python
# Test connectivity
import requests
try:
    response = requests.get('https://api.openai.com/v1/models', timeout=10)
    print(f"API reachable: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Connection error: {e}")

# Check DNS
import socket
try:
    ip = socket.gethostbyname('api.openai.com')
    print(f"DNS resolution: {ip}")
except socket.gaierror as e:
    print(f"DNS error: {e}")
```

### Issue: WebSocket Connection Issues
**Symptoms:**
```
WebSocket connection failed
Real-time data not updating
Connection dropped frequently
```

**Solutions:**
```python
# Test WebSocket connection
import websockets
import asyncio

async def test_websocket():
    try:
        async with websockets.connect('wss://echo.websocket.org') as websocket:
            await websocket.send('test')
            response = await websocket.recv()
            print(f"WebSocket test successful: {response}")
    except Exception as e:
        print(f"WebSocket error: {e}")

asyncio.run(test_websocket())
```

## 🆘 Emergency Procedures

### Complete System Reset
```bash
# Stop all services
docker-compose down  # If using Docker
sudo systemctl stop abc-application

# Clear caches
redis-cli FLUSHALL
rm -rf __pycache__/ */__pycache__/

# Reset logs
truncate -s 0 logs/*.log

# Restart services
docker-compose up -d
sudo systemctl start abc-application
```

### Data Recovery
```bash
# Restore from backup
tar -xzf backup.tar.gz -C /

# Rebuild search indices
python -c "from src.utils.embeddings import rebuild_indices; rebuild_indices()"

# Validate data integrity
python -c "from src.utils.data_validator import validate_all_data; validate_all_data()"
```

### Emergency Contacts
- **Development Team**: Create GitHub issue with `emergency` label
- **System Monitoring**: Check Discord alerts channel
- **API Support**: Contact respective API providers
- **Infrastructure**: Contact hosting provider support

## 📊 Diagnostic Tools

### System Health Check
```python
# src/diagnostics.py
import asyncio
import psutil
import redis
from typing import Dict, Any

async def comprehensive_health_check() -> Dict[str, Any]:
    """Run comprehensive system health check."""

    results = {
        'timestamp': asyncio.get_event_loop().time(),
        'checks': {}
    }

    # Memory check
    memory = psutil.virtual_memory()
    results['checks']['memory'] = {
        'available': memory.available / 1024 / 1024,  # MB
        'percent': memory.percent,
        'status': 'healthy' if memory.percent < 90 else 'critical'
    }

    # Disk check
    disk = psutil.disk_usage('/')
    results['checks']['disk'] = {
        'free': disk.free / 1024 / 1024 / 1024,  # GB
        'percent': disk.percent,
        'status': 'healthy' if disk.percent < 95 else 'critical'
    }

    # Redis check
    try:
        r = redis.Redis()
        r.ping()
        results['checks']['redis'] = {'status': 'healthy'}
    except Exception as e:
        results['checks']['redis'] = {'status': 'failed', 'error': str(e)}

    # Network check
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('https://httpbin.org/status/200', timeout=5) as resp:
                results['checks']['network'] = {
                    'status': 'healthy' if resp.status == 200 else 'degraded'
                }
    except Exception as e:
        results['checks']['network'] = {'status': 'failed', 'error': str(e)}

    return results

# Usage
if __name__ == '__main__':
    results = asyncio.run(comprehensive_health_check())
    print(f"Health Check Results: {results}")
```

### Performance Profiler
```python
# src/profiler.py
import time
import functools
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

def performance_profiler(func: Callable) -> Callable:
    """Decorator to profile function performance."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time

            if duration > 1.0:  # Log slow operations
                logger.warning(
                    f"SLOW OPERATION: {func.__name__} took {duration:.2f}s"
                )

            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"FAILED OPERATION: {func.__name__} took {duration:.2f}s - {e}"
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            if duration > 1.0:
                logger.warning(
                    f"SLOW OPERATION: {func.__name__} took {duration:.2f}s"
                )

            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"FAILED OPERATION: {func.__name__} took {duration:.2f}s - {e}"
            )
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
```

---

*This troubleshooting guide should resolve most common issues. For persistent problems, create a detailed GitHub issue with logs and reproduction steps.*