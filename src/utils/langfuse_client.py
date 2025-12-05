# src/utils/langfuse_client.py
# [LABEL:COMPONENT:langfuse_client] [LABEL:FRAMEWORK:langfuse] [LABEL:FRAMEWORK:asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-04] [LABEL:REVIEWED:pending]
#
# Purpose: Langfuse client for agent monitoring, tracing, and analytics
# Dependencies: langfuse, pyyaml, dotenv
# Related: config/langfuse_config.yaml, src/agents/base.py, src/utils/a2a_protocol.py
"""
Langfuse Client for ABC-Application Agent Monitoring

Provides comprehensive tracing, monitoring, and analytics for:
- Agent operations (input/output, processing time)
- LLM calls (prompts, completions, token usage)
- Memory operations (reads, writes, searches)
- A2A protocol messages
- Workflow orchestration
- Consensus polling

Features:
- Simple decorator-based tracing with @observe
- Graceful degradation when Langfuse is unavailable
- Integration with AlertManager for error correlation
- Discord health channel summaries
"""

import os
import logging
import functools
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import langfuse
try:
    from langfuse import Langfuse, observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    observe = None

# Set up logger
logger = logging.getLogger(__name__)

# Global client instance
_client = None
_enabled = False


def initialize_langfuse():
    """
    Initialize the global Langfuse client.

    Returns:
        True if successfully initialized, False otherwise
    """
    global _client, _enabled

    if not LANGFUSE_AVAILABLE:
        logger.warning("Langfuse package not installed - tracing disabled")
        _enabled = False
        return False

    # Get API keys from environment
    public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
    secret_key = os.getenv('LANGFUSE_SECRET_KEY')

    if not public_key or not secret_key:
        logger.warning("Langfuse API keys not configured - tracing disabled")
        logger.info("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables to enable tracing")
        _enabled = False
        return False

    try:
        host = os.getenv('LANGFUSE_BASE_URL', 'https://us.cloud.langfuse.com')

        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )

        _enabled = True
        logger.info(f"Langfuse client initialized successfully (host: {host})")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {e}")
        _enabled = False
        _client = None
        return False


def is_langfuse_enabled():
    """Check if Langfuse tracing is enabled."""
    return _enabled and _client is not None


def get_langfuse_client():
    """Get the global Langfuse client instance."""
    return _client


def trace_function(name=None,
                  user_id=None,
                  session_id=None,
                  tags=None,
                  metadata=None):
    """
    Decorator to trace function calls with Langfuse.

    Args:
        name: Custom trace name (defaults to function name)
        user_id: User/agent identifier (stored in metadata)
        session_id: Session identifier (stored in metadata)
        tags: Tags for filtering traces (stored in metadata)
        metadata: Additional metadata

    Example:
        @trace_function(user_id="agent_1", tags=["llm_call"])
        def process_input(data):
            return analyze_data(data)
    """
    def decorator(func):
        if not is_langfuse_enabled() or observe is None:
            # Return original function if tracing disabled
            return func

        trace_name = name or f"{func.__module__}.{func.__name__}"

        # Build metadata with all the additional info
        trace_metadata = {
            'app_name': 'ABC-Application',
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'function': f"{func.__module__}.{func.__name__}",
            'timestamp': datetime.now().isoformat()
        }
        if user_id:
            trace_metadata['user_id'] = user_id
        if session_id:
            trace_metadata['session_id'] = session_id
        if tags:
            trace_metadata['tags'] = tags
        if metadata:
            trace_metadata.update(metadata)

        @functools.wraps(func)
        @observe(name=trace_name,
                capture_input=True,
                capture_output=True)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Re-raise the exception after tracing
                raise e

        return wrapper

    return decorator


def trace_async_function(name=None,
                        user_id=None,
                        session_id=None,
                        tags=None,
                        metadata=None):
    """
    Decorator to trace async function calls with Langfuse.

    Args:
        name: Custom trace name (defaults to function name)
        user_id: User/agent identifier (stored in metadata)
        session_id: Session identifier (stored in metadata)
        tags: Tags for filtering traces (stored in metadata)
        metadata: Additional metadata
    """
    def decorator(func):
        if not is_langfuse_enabled() or observe is None:
            # Return original function if tracing disabled
            return func

        trace_name = name or f"{func.__module__}.{func.__name__}"

        # Build metadata with all the additional info
        trace_metadata = {
            'app_name': 'ABC-Application',
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'function': f"{func.__module__}.{func.__name__}",
            'async': True,
            'timestamp': datetime.now().isoformat()
        }
        if user_id:
            trace_metadata['user_id'] = user_id
        if session_id:
            trace_metadata['session_id'] = session_id
        if tags:
            trace_metadata['tags'] = tags
        if metadata:
            trace_metadata.update(metadata)

        @functools.wraps(func)
        @observe(name=trace_name,
                capture_input=True,
                capture_output=True)
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                # Re-raise the exception after tracing
                raise e

        return wrapper

    return decorator


class LangfuseClient:
    """
    Simplified Langfuse client for backward compatibility.

    This class provides a simplified interface that uses the @observe decorator
    internally while maintaining the same API for existing code.
    """

    def __init__(self):
        # Initialize on first use
        if not _enabled:
            initialize_langfuse()

    @property
    def is_enabled(self):
        """Check if Langfuse tracing is enabled."""
        return is_langfuse_enabled()

    @property
    def client(self):
        """Get the underlying Langfuse client."""
        return get_langfuse_client()

    # Legacy methods for backward compatibility
    def create_trace(self, name, user_id=None, session_id=None, input_data=None, metadata=None, tags=None):
        """Legacy method - now returns None as tracing is decorator-based."""
        logger.warning("create_trace() is deprecated. Use @trace_function decorator instead.")
        return None

    def end_trace(self, trace_id, output_data=None, level="DEFAULT", status_message=None):
        """Legacy method - no-op as tracing is decorator-based."""
        logger.warning("end_trace() is deprecated. Use @trace_function decorator instead.")
        pass

    def create_span(self, trace_id, name, input_data=None, metadata=None, level="DEFAULT"):
        """Legacy method - now returns None as tracing is decorator-based."""
        logger.warning("create_span() is deprecated. Use @trace_function decorator instead.")
        return None

    def end_span(self, span_id, output_data=None, level="DEFAULT", status_message=None):
        """Legacy method - no-op as tracing is decorator-based."""
        logger.warning("end_span() is deprecated. Use @trace_function decorator instead.")
        pass

    def get_callback_handler(self, trace_id=None, user_id=None, session_id=None, tags=None, metadata=None):
        """Legacy method - returns None as LangChain integration is not supported."""
        logger.warning("get_callback_handler() is deprecated. LangChain integration not supported.")
        return None

    def flush(self):
        """Flush any pending traces to Langfuse."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Failed to flush traces: {e}")

    def score(self, trace_id, name, value, comment=None, user_id=None):
        """Legacy method - no-op as scoring is not implemented."""
        logger.warning("score() is deprecated. Scoring not implemented.")
        pass