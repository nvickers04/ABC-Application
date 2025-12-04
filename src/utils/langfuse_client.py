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
- Thread-safe singleton pattern
- Automatic span management with context managers
- Graceful degradation when Langfuse is unavailable
- Integration with AlertManager for error correlation
- Discord health channel summaries
"""

import os
import logging
import asyncio
import time
import functools
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from contextlib import contextmanager, asynccontextmanager
from pathlib import Path
import threading

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import langfuse
try:
    from langfuse import Langfuse
    from langfuse.decorators import langfuse_context, observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    langfuse_context = None
    observe = None

# CallbackHandler is not compatible with LangChain 1.x, so we don't import it
CallbackHandler = None

# Try to import yaml for config loading
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Set up logger early for import-time logging
logger = logging.getLogger(__name__)


def _load_config() -> Dict[str, Any]:
    """Load Langfuse configuration from YAML file."""
    config = {}
    
    # Find config file
    config_paths = [
        Path("config/langfuse_config.yaml"),
        Path(__file__).parent.parent.parent / "config" / "langfuse_config.yaml",
    ]
    
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path and YAML_AVAILABLE:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded Langfuse config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load Langfuse config: {e}")
    
    return config


def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variable references in config values."""
    if isinstance(value, str) and value.startswith("${") and "}" in value:
        # Parse ${VAR_NAME:-default}
        var_expr = value[2:value.index("}")]
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
        else:
            var_name, default = var_expr, ""
        return os.getenv(var_name, default)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


class LangfuseClient:
    """
    Singleton Langfuse client for agent monitoring and tracing.
    
    Thread-safe singleton pattern ensures consistent state across
    all agents and components.
    """
    
    _instance: Optional['LangfuseClient'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'LangfuseClient':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._client: Optional[Langfuse] = None
        self._config: Dict[str, Any] = {}
        self._enabled: bool = False
        self._active_traces: Dict[str, Any] = {}
        self._active_spans: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {
            'total_traces': 0,
            'total_spans': 0,
            'total_errors': 0,
            'traces_by_agent': {},
            'avg_response_times': {},
            'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        }
        
        # Load configuration
        self._load_configuration()
        
        # Initialize client if available and enabled
        self._initialize_client()
    
    def _load_configuration(self):
        """Load and process configuration."""
        self._config = _resolve_env_vars(_load_config())
        
        # Check if enabled
        connection_config = self._config.get('connection', {})
        enabled_str = str(connection_config.get('enabled', 'true')).lower()
        self._enabled = enabled_str in ('true', '1', 'yes')
    
    def _initialize_client(self):
        """Initialize the Langfuse client."""
        if not LANGFUSE_AVAILABLE:
            logger.warning("Langfuse package not installed - tracing disabled")
            self._enabled = False
            return
        
        if not self._enabled:
            logger.info("Langfuse disabled in configuration")
            return
        
        # Get API keys from environment
        public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
        secret_key = os.getenv('LANGFUSE_SECRET_KEY')
        
        if not public_key or not secret_key:
            logger.warning("Langfuse API keys not configured - tracing disabled")
            logger.info("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables to enable tracing")
            self._enabled = False
            return
        
        try:
            connection_config = self._config.get('connection', {})
            host = connection_config.get('host', 'https://cloud.langfuse.com')
            
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            
            logger.info(f"Langfuse client initialized successfully (host: {host})")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            self._enabled = False
            self._client = None
    
    @property
    def is_enabled(self) -> bool:
        """Check if Langfuse tracing is enabled."""
        return self._enabled and self._client is not None
    
    @property
    def client(self) -> Optional[Langfuse]:
        """Get the underlying Langfuse client."""
        return self._client
    
    def get_callback_handler(self, trace_id: Optional[str] = None,
                            user_id: Optional[str] = None,
                            session_id: Optional[str] = None,
                            tags: Optional[List[str]] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> Optional[CallbackHandler]:
        """
        Get a LangChain callback handler for tracing LLM calls.
        
        Args:
            trace_id: Optional trace ID to use (creates new trace if not provided)
            user_id: User/agent identifier
            session_id: Session identifier for grouping traces
            tags: Tags for filtering traces
            metadata: Additional metadata
            
        Returns:
            CallbackHandler instance or None if not available
        """
        if not self.is_enabled or CallbackHandler is None:
            return None
        
        try:
            handler = CallbackHandler(
                trace_id=trace_id,
                user_id=user_id,
                session_id=session_id,
                tags=tags or [],
                metadata=metadata or {}
            )
            return handler
        except Exception as e:
            logger.error(f"Failed to create Langfuse callback handler: {e}")
            return None
    
    def create_trace(self, name: str,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    input_data: Optional[Any] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    tags: Optional[List[str]] = None) -> Optional[str]:
        """
        Create a new trace for an operation.
        
        Args:
            name: Trace name (e.g., "agent_process_input")
            user_id: User/agent identifier
            session_id: Session identifier
            input_data: Input data for the trace
            metadata: Additional metadata
            tags: Tags for filtering
            
        Returns:
            Trace ID or None if tracing is disabled
        """
        if not self.is_enabled:
            return None
        
        try:
            # Build metadata
            trace_metadata = {
                'app_name': self._config.get('metadata', {}).get('app_name', 'ABC-Application'),
                'environment': self._config.get('metadata', {}).get('environment', 'development'),
                'version': self._config.get('metadata', {}).get('version', '1.0.0'),
                'timestamp': datetime.now().isoformat()
            }
            if metadata:
                trace_metadata.update(metadata)
            
            trace = self._client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                input=input_data,
                metadata=trace_metadata,
                tags=tags or []
            )
            
            trace_id = trace.id
            self._active_traces[trace_id] = {
                'trace': trace,
                'name': name,
                'start_time': time.time(),
                'user_id': user_id
            }
            
            self._metrics['total_traces'] += 1
            
            # Track by agent
            if user_id:
                if user_id not in self._metrics['traces_by_agent']:
                    self._metrics['traces_by_agent'][user_id] = 0
                self._metrics['traces_by_agent'][user_id] += 1
            
            logger.debug(f"Created trace {trace_id} for {name}")
            return trace_id
            
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            self._metrics['total_errors'] += 1
            return None
    
    def end_trace(self, trace_id: str,
                 output_data: Optional[Any] = None,
                 level: str = "DEFAULT",
                 status_message: Optional[str] = None):
        """
        End a trace and record output.
        
        Args:
            trace_id: Trace ID to end
            output_data: Output data from the operation
            level: Log level (DEBUG, DEFAULT, WARNING, ERROR)
            status_message: Optional status message
        """
        if not self.is_enabled or trace_id not in self._active_traces:
            return
        
        try:
            trace_data = self._active_traces.pop(trace_id)
            trace = trace_data['trace']
            
            # Calculate duration
            duration = time.time() - trace_data['start_time']
            
            # Update trace
            trace.update(
                output=output_data,
                level=level,
                status_message=status_message
            )
            
            # Track response time
            user_id = trace_data.get('user_id', 'unknown')
            if user_id not in self._metrics['avg_response_times']:
                self._metrics['avg_response_times'][user_id] = []
            self._metrics['avg_response_times'][user_id].append(duration)
            
            # Keep only last 100 response times
            if len(self._metrics['avg_response_times'][user_id]) > 100:
                self._metrics['avg_response_times'][user_id].pop(0)
            
            logger.debug(f"Ended trace {trace_id} (duration: {duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"Failed to end trace {trace_id}: {e}")
            self._metrics['total_errors'] += 1
    
    def create_span(self, trace_id: str, name: str,
                   input_data: Optional[Any] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   level: str = "DEFAULT") -> Optional[str]:
        """
        Create a span within a trace.
        
        Args:
            trace_id: Parent trace ID
            name: Span name
            input_data: Input data for the span
            metadata: Additional metadata
            level: Log level
            
        Returns:
            Span ID or None
        """
        if not self.is_enabled:
            return None
        
        trace_data = self._active_traces.get(trace_id)
        if not trace_data:
            logger.warning(f"Cannot create span: trace {trace_id} not found")
            return None
        
        try:
            trace = trace_data['trace']
            span = trace.span(
                name=name,
                input=input_data,
                metadata=metadata or {},
                level=level
            )
            
            span_id = span.id
            self._active_spans[span_id] = {
                'span': span,
                'trace_id': trace_id,
                'name': name,
                'start_time': time.time()
            }
            
            self._metrics['total_spans'] += 1
            logger.debug(f"Created span {span_id} in trace {trace_id}")
            return span_id
            
        except Exception as e:
            logger.error(f"Failed to create span: {e}")
            self._metrics['total_errors'] += 1
            return None
    
    def end_span(self, span_id: str,
                output_data: Optional[Any] = None,
                level: str = "DEFAULT",
                status_message: Optional[str] = None):
        """
        End a span and record output.
        
        Args:
            span_id: Span ID to end
            output_data: Output data
            level: Log level
            status_message: Optional status message
        """
        if not self.is_enabled or span_id not in self._active_spans:
            return
        
        try:
            span_data = self._active_spans.pop(span_id)
            span = span_data['span']
            
            span.update(
                output=output_data,
                level=level,
                status_message=status_message
            )
            span.end()
            
            logger.debug(f"Ended span {span_id}")
            
        except Exception as e:
            logger.error(f"Failed to end span {span_id}: {e}")
            self._metrics['total_errors'] += 1
    
    def log_generation(self, trace_id: str,
                      name: str,
                      model: str,
                      input_messages: Any,
                      output: Any,
                      usage: Optional[Dict[str, int]] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Log an LLM generation within a trace.
        
        Args:
            trace_id: Parent trace ID
            name: Generation name
            model: Model name/identifier
            input_messages: Input messages/prompt
            output: Generated output
            usage: Token usage dict (prompt_tokens, completion_tokens, total_tokens)
            metadata: Additional metadata
        """
        if not self.is_enabled:
            return
        
        trace_data = self._active_traces.get(trace_id)
        if not trace_data:
            logger.warning(f"Cannot log generation: trace {trace_id} not found")
            return
        
        try:
            trace = trace_data['trace']
            trace.generation(
                name=name,
                model=model,
                input=input_messages,
                output=output,
                usage=usage,
                metadata=metadata or {}
            )
            
            # Track token usage
            if usage:
                self._metrics['token_usage']['prompt_tokens'] += usage.get('prompt_tokens', 0)
                self._metrics['token_usage']['completion_tokens'] += usage.get('completion_tokens', 0)
                self._metrics['token_usage']['total_tokens'] += usage.get('total_tokens', 0)
            
            logger.debug(f"Logged generation {name} in trace {trace_id}")
            
        except Exception as e:
            logger.error(f"Failed to log generation: {e}")
            self._metrics['total_errors'] += 1
    
    def score_trace(self, trace_id: str, name: str, value: float,
                   comment: Optional[str] = None):
        """
        Add a score to a trace for quality evaluation.
        
        Args:
            trace_id: Trace ID to score
            name: Score name (e.g., "accuracy", "relevance")
            value: Score value (typically 0-1)
            comment: Optional comment
        """
        if not self.is_enabled:
            return
        
        try:
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment
            )
            
            logger.debug(f"Scored trace {trace_id}: {name}={value}")
            
        except Exception as e:
            logger.error(f"Failed to score trace: {e}")
            self._metrics['total_errors'] += 1
    
    @contextmanager
    def trace_context(self, name: str,
                     user_id: Optional[str] = None,
                     session_id: Optional[str] = None,
                     input_data: Optional[Any] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     tags: Optional[List[str]] = None):
        """
        Context manager for automatic trace management.
        
        Usage:
            with langfuse_client.trace_context("my_operation", user_id="agent") as trace_id:
                # Do something
                pass
        """
        trace_id = self.create_trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            input_data=input_data,
            metadata=metadata,
            tags=tags
        )
        
        try:
            yield trace_id
            self.end_trace(trace_id, level="DEFAULT")
        except Exception as e:
            self.end_trace(trace_id, level="ERROR", status_message=str(e))
            raise
    
    @contextmanager
    def span_context(self, trace_id: str, name: str,
                    input_data: Optional[Any] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for automatic span management.
        
        Usage:
            with langfuse_client.span_context(trace_id, "my_span") as span_id:
                # Do something
                pass
        """
        span_id = self.create_span(
            trace_id=trace_id,
            name=name,
            input_data=input_data,
            metadata=metadata
        )
        
        try:
            yield span_id
            self.end_span(span_id, level="DEFAULT")
        except Exception as e:
            self.end_span(span_id, level="ERROR", status_message=str(e))
            raise
    
    @asynccontextmanager
    async def async_trace_context(self, name: str,
                                 user_id: Optional[str] = None,
                                 session_id: Optional[str] = None,
                                 input_data: Optional[Any] = None,
                                 metadata: Optional[Dict[str, Any]] = None,
                                 tags: Optional[List[str]] = None):
        """Async context manager for trace management."""
        trace_id = self.create_trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            input_data=input_data,
            metadata=metadata,
            tags=tags
        )
        
        try:
            yield trace_id
            self.end_trace(trace_id, level="DEFAULT")
        except Exception as e:
            self.end_trace(trace_id, level="ERROR", status_message=str(e))
            raise
    
    @asynccontextmanager
    async def async_span_context(self, trace_id: str, name: str,
                                input_data: Optional[Any] = None,
                                metadata: Optional[Dict[str, Any]] = None):
        """Async context manager for span management."""
        span_id = self.create_span(
            trace_id=trace_id,
            name=name,
            input_data=input_data,
            metadata=metadata
        )
        
        try:
            yield span_id
            self.end_span(span_id, level="DEFAULT")
        except Exception as e:
            self.end_span(span_id, level="ERROR", status_message=str(e))
            raise
    
    def flush(self):
        """Flush any pending traces to Langfuse."""
        if self._client:
            try:
                self._client.flush()
                logger.debug("Flushed Langfuse traces")
            except Exception as e:
                logger.error(f"Failed to flush Langfuse traces: {e}")
    
    def shutdown(self):
        """Shutdown the Langfuse client cleanly."""
        if self._client:
            try:
                self._client.flush()
                self._client.shutdown()
                logger.info("Langfuse client shutdown complete")
            except Exception as e:
                logger.error(f"Error during Langfuse shutdown: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        metrics = self._metrics.copy()
        
        # Calculate average response times
        avg_times = {}
        for agent, times in metrics.get('avg_response_times', {}).items():
            if times:
                avg_times[agent] = sum(times) / len(times)
        metrics['avg_response_times_computed'] = avg_times
        
        return metrics
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a health summary for Discord integration.
        
        Returns:
            Dict with health summary data
        """
        metrics = self.get_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'enabled': self.is_enabled,
            'total_traces': metrics.get('total_traces', 0),
            'total_spans': metrics.get('total_spans', 0),
            'total_errors': metrics.get('total_errors', 0),
            'active_traces': len(self._active_traces),
            'active_spans': len(self._active_spans),
            'traces_by_agent': metrics.get('traces_by_agent', {}),
            'token_usage': metrics.get('token_usage', {}),
            'avg_response_times': metrics.get('avg_response_times_computed', {}),
            'error_rate': (
                metrics.get('total_errors', 0) / max(metrics.get('total_traces', 1), 1)
            ) * 100
        }
    
    def get_config(self, key: str = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Dot-separated config key (e.g., "tracing.enabled")
                 If None, returns full config
                 
        Returns:
            Config value or None
        """
        if key is None:
            return self._config
        
        parts = key.split('.')
        value = self._config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value


# Singleton accessor
_langfuse_client: Optional[LangfuseClient] = None


def get_langfuse_client() -> LangfuseClient:
    """Get the singleton Langfuse client instance."""
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = LangfuseClient()
    return _langfuse_client


def trace_agent_operation(name: Optional[str] = None,
                         include_input: bool = True,
                         include_output: bool = True):
    """
    Decorator for tracing agent operations.
    
    Args:
        name: Operation name (defaults to function name)
        include_input: Whether to include input in trace
        include_output: Whether to include output in trace
    
    Usage:
        @trace_agent_operation("process_input")
        async def _process_input(self, input_data):
            # Do something
            return result
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            client = get_langfuse_client()
            
            # Get agent role from self if available
            agent_role = None
            if args and hasattr(args[0], 'role'):
                agent_role = args[0].role
            
            op_name = name or func.__name__
            input_data = None
            if include_input and len(args) > 1:
                input_data = str(args[1])[:1000] if args[1] else None
            
            async with client.async_trace_context(
                name=op_name,
                user_id=agent_role,
                input_data=input_data,
                metadata={'function': func.__name__},
                tags=['agent_operation']
            ) as trace_id:
                result = await func(*args, **kwargs)
                
                if include_output and trace_id:
                    output_str = str(result)[:1000] if result else None
                    client.end_trace(trace_id, output_data=output_str)
                
                return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            client = get_langfuse_client()
            
            # Get agent role from self if available
            agent_role = None
            if args and hasattr(args[0], 'role'):
                agent_role = args[0].role
            
            op_name = name or func.__name__
            input_data = None
            if include_input and len(args) > 1:
                input_data = str(args[1])[:1000] if args[1] else None
            
            with client.trace_context(
                name=op_name,
                user_id=agent_role,
                input_data=input_data,
                metadata={'function': func.__name__},
                tags=['agent_operation']
            ) as trace_id:
                result = func(*args, **kwargs)
                
                if include_output and trace_id:
                    output_str = str(result)[:1000] if result else None
                    client.end_trace(trace_id, output_data=output_str)
                
                return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def trace_llm_call(model_name: str):
    """
    Decorator for tracing LLM calls.
    
    Args:
        model_name: Name of the LLM model
    
    Usage:
        @trace_llm_call("grok-4")
        async def call_llm(self, prompt):
            return response
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            client = get_langfuse_client()
            
            if not client.is_enabled:
                return await func(*args, **kwargs)
            
            # Get agent role
            agent_role = None
            if args and hasattr(args[0], 'role'):
                agent_role = args[0].role
            
            trace_id = client.create_trace(
                name=f"llm_call_{func.__name__}",
                user_id=agent_role,
                tags=['llm_call', model_name]
            )
            
            try:
                start_time = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log generation
                if trace_id:
                    # Extract prompt from args if possible
                    prompt = str(args[1])[:2000] if len(args) > 1 and args[1] is not None else "unknown"
                    output = str(result)[:2000] if result else "no_output"
                    
                    client.log_generation(
                        trace_id=trace_id,
                        name=func.__name__,
                        model=model_name,
                        input_messages=prompt,
                        output=output,
                        metadata={'duration': duration}
                    )
                    client.end_trace(trace_id)
                
                return result
                
            except Exception as e:
                if trace_id:
                    client.end_trace(trace_id, level="ERROR", status_message=str(e))
                raise
        
        return wrapper
    return decorator


def trace_a2a_message(message_type: str = "a2a"):
    """
    Decorator for tracing A2A protocol messages.
    
    Args:
        message_type: Type of A2A message
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            client = get_langfuse_client()
            
            if not client.is_enabled or not client.get_config("tracing.trace_a2a_messages"):
                return await func(*args, **kwargs)
            
            # Extract message details
            sender = None
            receiver = None
            if len(args) > 1 and hasattr(args[1], 'sender'):
                sender = args[1].sender
                receiver = args[1].receiver
            
            async with client.async_trace_context(
                name=f"a2a_{message_type}",
                user_id=sender,
                metadata={
                    'message_type': message_type,
                    'sender': sender,
                    'receiver': receiver
                },
                tags=['a2a', message_type]
            ) as trace_id:
                result = await func(*args, **kwargs)
                return result
        
        return wrapper
    return decorator


def trace_memory_operation(operation_type: str):
    """
    Decorator for tracing memory operations.
    
    Args:
        operation_type: Type of memory operation (read, write, search)
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            client = get_langfuse_client()
            
            if not client.is_enabled or not client.get_config("tracing.trace_memory_operations"):
                return await func(*args, **kwargs)
            
            # Get agent role
            agent_role = None
            if args and hasattr(args[0], 'role'):
                agent_role = args[0].role
            
            async with client.async_trace_context(
                name=f"memory_{operation_type}",
                user_id=agent_role,
                metadata={'operation': operation_type},
                tags=['memory', operation_type]
            ) as trace_id:
                result = await func(*args, **kwargs)
                return result
        
        return wrapper
    return decorator
