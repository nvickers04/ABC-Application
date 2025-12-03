#!/usr/bin/env python3
"""
Custom Exception Hierarchy for ABC-Application
Provides standardized, specific exception types for better error handling.
"""

from typing import Dict, Any, Optional


class ABCApplicationError(Exception):
    """Base exception class for all ABC-Application errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(ABCApplicationError):
    """Configuration-related errors"""
    pass


class ValidationError(ABCApplicationError):
    """Data validation errors"""
    pass


class ConnectionError(ABCApplicationError):
    """Connection and network-related errors"""
    pass


class AuthenticationError(ABCApplicationError):
    """Authentication and authorization errors"""
    pass


class TradingError(ABCApplicationError):
    """Trading-related errors"""
    pass


class DataError(ABCApplicationError):
    """Data processing and storage errors"""
    pass


class APIError(ABCApplicationError):
    """External API errors"""
    pass


class ResourceError(ABCApplicationError):
    """Resource-related errors (memory, disk, etc.)"""
    pass


class AgentError(ABCApplicationError):
    """Agent-related errors"""
    pass


class WorkflowError(ABCApplicationError):
    """Workflow execution errors"""
    pass


# Specific trading errors
class IBKRError(TradingError):
    """IBKR-specific errors"""
    pass


class OrderError(TradingError):
    """Order placement and management errors"""
    pass


class MarketDataError(TradingError):
    """Market data retrieval errors"""
    pass


# Specific connection errors
class IBKRConnectionError(ConnectionError):
    """IBKR connection errors"""
    pass


class RedisConnectionError(ConnectionError):
    """Redis connection errors"""
    pass


class APIConnectionError(ConnectionError):
    """API connection errors"""
    pass


# Specific data errors
class DataValidationError(DataError):
    """Data validation errors"""
    pass


class DataPersistenceError(DataError):
    """Data persistence errors"""
    pass


# Specific agent errors
class AgentInitializationError(AgentError):
    """Agent initialization errors"""
    pass


class AgentExecutionError(AgentError):
    """Agent execution errors"""
    pass


# Specific workflow errors
class ConsensusError(WorkflowError):
    """Consensus workflow errors"""
    pass


class OrchestrationError(WorkflowError):
    """Orchestration errors"""
    pass


# Utility functions for error handling
def handle_exceptions(func):
    """
    Decorator for standardized exception handling.
    Converts common exceptions to ABCApplicationError types.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, TimeoutError, OSError) as e:
            raise ConnectionError(f"Connection failed: {str(e)}") from e
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Validation failed: {str(e)}") from e
        except (KeyError, IndexError) as e:
            raise DataError(f"Data access failed: {str(e)}") from e
        except PermissionError as e:
            raise AuthenticationError(f"Permission denied: {str(e)}") from e
        except Exception as e:
            # For unexpected errors, re-raise as ABCApplicationError
            raise ABCApplicationError(f"Unexpected error: {str(e)}") from e
    return wrapper


def safe_execute(func, default_return=None, log_errors=True):
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        default_return: Value to return on error
        log_errors: Whether to log errors

    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except ABCApplicationError as e:
        if log_errors:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"ABC Application error: {e}")
        return default_return
    except Exception as e:
        if log_errors:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Unexpected error: {e}")
        return default_return


def create_error_context(**kwargs) -> Dict[str, Any]:
    """
    Create error context dictionary for consistent error reporting.

    Args:
        **kwargs: Key-value pairs for error context

    Returns:
        Dict with error context
    """
    context = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'component': kwargs.get('component', 'unknown'),
    }
    context.update(kwargs)
    return context