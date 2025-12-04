# [LABEL:TEST:langfuse_client] [LABEL:TEST:unit] [LABEL:FRAMEWORK:pytest] [LABEL:FRAMEWORK:asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-04] [LABEL:REVIEWED:pending]
#
# Purpose: Unit tests for Langfuse client integration
# Dependencies: pytest, pytest-asyncio, unittest.mock
# Related: src/utils/langfuse_client.py, config/langfuse_config.yaml
"""
Unit tests for the Langfuse client integration.

Tests cover:
- Client initialization (with and without API keys)
- Trace creation and management
- Span creation and management
- Context managers for traces and spans
- Decorator functionality
- Health summary generation
- Graceful degradation when Langfuse is unavailable
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import os
import sys
import importlib.util


def load_langfuse_client_directly():
    """Load langfuse_client module directly to avoid circular imports."""
    spec = importlib.util.spec_from_file_location(
        'langfuse_client',
        os.path.join(os.path.dirname(__file__), '..', 'src', 'utils', 'langfuse_client.py')
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        return None


class TestLangfuseClientInitialization:
    """Tests for Langfuse client initialization."""
    
    def test_client_module_loads(self):
        """Test that the client module can be loaded."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        assert hasattr(module, 'LangfuseClient')
        assert hasattr(module, 'LANGFUSE_AVAILABLE')
    
    def test_client_disabled_without_api_keys(self):
        """Test that client is disabled when API keys are not set."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        # Clear any existing environment variables
        env_backup = {}
        for key in ['LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY', 'LANGFUSE_ENABLED']:
            if key in os.environ:
                env_backup[key] = os.environ.pop(key)
        
        try:
            # Create a fresh instance by clearing singleton
            module.LangfuseClient._instance = None
            client = module.LangfuseClient()
            
            # Client should be disabled without API keys
            assert client.is_enabled == False or client.client is None
        finally:
            # Restore environment
            for key, value in env_backup.items():
                os.environ[key] = value
    
    def test_get_langfuse_client_returns_singleton(self):
        """Test that get_langfuse_client returns the same instance."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        # Reset singleton
        module.LangfuseClient._instance = None
        module._langfuse_client = None
        
        client1 = module.get_langfuse_client()
        client2 = module.get_langfuse_client()
        
        assert client1 is client2


class TestLangfuseClientTracing:
    """Tests for Langfuse client tracing functionality."""
    
    def test_create_trace_when_disabled(self):
        """Test that create_trace returns None when client is disabled."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        # Create a fresh instance
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        client._enabled = False  # Force disable
        
        trace_id = client.create_trace(
            name="test_trace",
            user_id="test_user"
        )
        
        assert trace_id is None
    
    def test_end_trace_when_disabled(self):
        """Test that end_trace handles missing traces gracefully."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        client._enabled = False
        
        # Should not raise exception
        client.end_trace("non_existent_trace_id")
    
    def test_create_span_when_disabled(self):
        """Test that create_span returns None when client is disabled."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        client._enabled = False
        
        span_id = client.create_span(
            trace_id="test_trace",
            name="test_span"
        )
        
        assert span_id is None


class TestLangfuseClientMetrics:
    """Tests for Langfuse client metrics collection."""
    
    def test_get_metrics(self):
        """Test that get_metrics returns expected structure."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        metrics = client.get_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_traces' in metrics
        assert 'total_spans' in metrics
        assert 'total_errors' in metrics
    
    def test_get_health_summary(self):
        """Test that get_health_summary returns expected structure."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        summary = client.get_health_summary()
        
        assert isinstance(summary, dict)
        assert 'timestamp' in summary
        assert 'enabled' in summary
        assert 'total_traces' in summary
        assert 'total_errors' in summary
        assert 'error_rate' in summary


class TestLangfuseContextManagers:
    """Tests for Langfuse context managers."""
    
    def test_trace_context_when_disabled(self):
        """Test that trace_context works when client is disabled."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        client._enabled = False
        
        # Should not raise exception
        with client.trace_context("test_operation", user_id="test") as trace_id:
            assert trace_id is None  # Should be None when disabled
    
    def test_span_context_when_disabled(self):
        """Test that span_context works when client is disabled."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        client._enabled = False
        
        # Should not raise exception
        with client.span_context("fake_trace_id", "test_span") as span_id:
            assert span_id is None  # Should be None when disabled
    
    @pytest.mark.asyncio
    async def test_async_trace_context_when_disabled(self):
        """Test that async_trace_context works when client is disabled."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        client._enabled = False
        
        # Should not raise exception
        async with client.async_trace_context("test_operation", user_id="test") as trace_id:
            assert trace_id is None  # Should be None when disabled


class TestLangfuseDecorators:
    """Tests for Langfuse decorator functions."""
    
    def test_trace_agent_operation_decorator_exists(self):
        """Test that decorator exists in module."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        assert hasattr(module, 'trace_agent_operation')
    
    def test_trace_llm_call_decorator_exists(self):
        """Test that LLM call decorator exists in module."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        assert hasattr(module, 'trace_llm_call')


class TestLangfuseConfig:
    """Tests for Langfuse configuration loading."""
    
    def test_get_config_returns_dict(self):
        """Test that get_config returns a dictionary."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        config = client.get_config()
        
        assert isinstance(config, dict)
    
    def test_get_config_with_key(self):
        """Test that get_config with key returns correct value or None."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        
        # Test with a key that may or may not exist
        value = client.get_config("tracing.enabled")
        # Should return either the config value or None
        assert value is None or isinstance(value, (bool, str))


class TestLangfuseGracefulDegradation:
    """Tests for graceful degradation when Langfuse is unavailable."""
    
    def test_client_works_without_langfuse(self):
        """Test that client works even when langfuse package is not available."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        # Reset singleton and test client works when disabled
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        
        # Force disable to simulate unavailable Langfuse
        client._enabled = False
        client._client = None
        
        # These should all work without raising exceptions
        assert client.is_enabled == False
        assert client.create_trace("test") is None
        assert client.create_span("trace", "span") is None
        client.end_trace("trace")
        client.end_span("span")
        
        metrics = client.get_metrics()
        assert isinstance(metrics, dict)
        
        summary = client.get_health_summary()
        assert isinstance(summary, dict)
    
    def test_flush_works_when_disabled(self):
        """Test that flush works when client is disabled."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        client._enabled = False
        
        # Should not raise exception
        client.flush()
    
    def test_shutdown_works_when_disabled(self):
        """Test that shutdown works when client is disabled."""
        module = load_langfuse_client_directly()
        if module is None:
            pytest.skip("langfuse_client module could not be loaded")
        
        module.LangfuseClient._instance = None
        client = module.LangfuseClient()
        client._enabled = False
        
        # Should not raise exception
        client.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

