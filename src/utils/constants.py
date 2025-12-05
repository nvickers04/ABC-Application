# [LABEL:CONFIG:constants] [LABEL:FRAMEWORK:constants] [LABEL:AUTHOR:system]
# [LABEL:UPDATED:2025-12-05] [LABEL:REVIEWED:pending]
#
# Purpose: Centralized constants for the ABC Application
# Dependencies: None
# Related: All modules using these constants

# Timeout Constants (seconds)
DEFAULT_API_TIMEOUT = 30
DEFAULT_LLM_TIMEOUT = 60
DEFAULT_AGENT_TIMEOUT = 120
DEFAULT_WORKFLOW_TIMEOUT = 300  # 5 minutes
DEFAULT_BATCH_TIMEOUT = 60

# Data Quality Constants
MIN_DATA_POINTS = 10
MAX_NAN_RATIO = 0.1  # 10% maximum NaN values
REQUIRED_DATA_COLUMNS = ['Close']

# Circuit Breaker Constants
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_RECOVERY_TIMEOUT = 300  # 5 minutes

# Memory and Caching Constants
DEFAULT_MEMORY_POOL_SIZE = 1000
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_VECTOR_DIMENSION = 768

# Agent Constants
MAX_AGENT_RETRIES = 3
DEFAULT_AGENT_BATCH_SIZE = 10

# Trading Constants
DEFAULT_RISK_LIMIT = 0.05  # 5% max drawdown
DEFAULT_POSITION_SIZE = 0.1  # 10% of portfolio

# API Rate Limits
DEFAULT_RATE_LIMIT_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW = 60  # seconds

# Logging Constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_LOG_BACKUP_COUNT = 5

# Database Constants
DEFAULT_DB_POOL_SIZE = 10
DEFAULT_DB_TIMEOUT = 30

# Redis Constants
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0

# File System Constants
DEFAULT_DATA_DIR = "./data"
DEFAULT_LOGS_DIR = "./logs"
DEFAULT_CONFIG_DIR = "./config"

# Error Message Constants
ERROR_API_KEY_NOT_FOUND = "API key not found in Vault or environment variables"
ERROR_NO_DATA_FOUND = "No data found"
ERROR_INVALID_RESPONSE = "Invalid response from API"
ERROR_CONNECTION_FAILED = "Connection failed"
ERROR_VALIDATION_FAILED = "Validation failed"