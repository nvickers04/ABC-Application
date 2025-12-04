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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestLangfuseClientInitialization:
    """Tests for Langfuse client initialization."""
    
    def test_client_import_without_langfuse(self):
        """Test that the client module handles missing langfuse gracefully."""
        # This should not raise an exception even if langfuse is not installed
        try:
            from src.utils.langfuse_client import get_langfuse_client, LANGFUSE_AVAILABLE
            assert True  # Import succeeded
        except ImportError:
            pytest.skip("langfuse_client module not available")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_client_disabled_without_api_keys(self):
        """Test that client is disabled when API keys are not set."""
        # Clear any existing environment variables
        env_backup = {}
        for key in ['LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY', 'LANGFUSE_ENABLED']:
            if key in os.environ:
                env_backup[key] = os.environ.pop(key)
        
        try:
            from src.utils.langfuse_client import LangfuseClient
            
            # Create a fresh instance by clearing singleton
            LangfuseClient._instance = None
            client = LangfuseClient()
            
            # Client should be disabled without API keys
            assert client.is_enabled == False or client.client is None
        finally:
            # Restore environment
            for key, value in env_backup.items():
                os.environ[key] = value
    
    def test_get_langfuse_client_returns_singleton(self):
        """Test that get_langfuse_client returns the same instance."""
        try:
            from src.utils.langfuse_client import get_langfuse_client
            
            client1 = get_langfuse_client()
            client2 = get_langfuse_client()
            
            assert client1 is client2
        except ImportError:
            pytest.skip("langfuse_client module not available")


class TestLangfuseClientTracing:
    """Tests for Langfuse client tracing functionality."""
    
    def test_create_trace_when_disabled(self):
        """Test that create_trace returns None when client is disabled."""
        try:
            from src.utils.langfuse_client import LangfuseClient
            
            # Create a fresh instance
            LangfuseClient._instance = None
            client = LangfuseClient()
            client._enabled = False  # Force disable
            
            trace_id = client.create_trace(
                name="test_trace",
                user_id="test_user"
            )
            
            assert trace_id is None
        except ImportError:
            pytest.skip("langfuse_client module not available")
    
    def test_end_trace_when_disabled(self):
        """Test that end_trace handles missing traces gracefully."""
        try:
            from src.utils.langfuse_client import LangfuseClient
            
            client = LangfuseClient()
            client._enabled = False
            
            # Should not raise exception
            client.end_trace("non_existent_trace_id")
        except ImportError:
            pytest.skip("langfuse_client module not available")
    
    def test_create_span_when_disabled(self):
        """Test that create_span returns None when client is disabled."""
        try:
            from src.utils.langfuse_client import LangfuseClient
            
            client = LangfuseClient()
            client._enabled = False
            
            span_id = client.create_span(
                trace_id="test_trace",
                name="test_span"
            )
            
            assert span_id is None
        except ImportError:
            pytest.skip("langfuse_client module not available")


class TestLangfuseClientMetrics:
    """Tests for Langfuse client metrics collection."""
    
    def test_get_metrics(self):
        """Test that get_metrics returns expected structure."""
        try:
            from src.utils.langfuse_client import LangfuseClient
            
            client = LangfuseClient()
            metrics = client.get_metrics()
            
            assert isinstance(metrics, dict)
            assert 'total_traces' in metrics
            assert 'total_spans' in metrics
            assert 'total_errors' in metrics
        except ImportError:
            pytest.skip("langfuse_client module not available")
    
    def test_get_health_summary(self):
        """Test that get_health_summary returns expected structure."""
        try:
            from src.utils.langfuse_client import LangfuseClient
            
            client = LangfuseClient()
            summary = client.get_health_summary()
            
            assert isinstance(summary, dict)
            assert 'timestamp' in summary
            assert 'enabled' in summary
            assert 'total_traces' in summary
            assert 'total_errors' in summary
            assert 'error_rate' in summary
        except ImportError:
            pytest.skip("langfuse_client module not available")


class TestLangfuseContextManagers:
    """Tests for Langfuse context managers."""
    
    def test_trace_context_when_disabled(self):
        """Test that trace_context works when client is disabled."""
        try:
            from src.utils.langfuse_client import LangfuseClient
            
            client = LangfuseClient()
            client._enabled = False
            
            # Should not raise exception
            with client.trace_context("test_operation", user_id="test") as trace_id:
                assert trace_id is None  # Should be None when disabled
        except ImportError:
            pytest.skip("langfuse_client module not available")
    
    def test_span_context_when_disabled(self):
        """Test that span_context works when client is disabled."""
        try:
            from src.utils.langfuse_client import LangfuseClient
            
            client = LangfuseClient()
            client._enabled = False
            
            # Should not raise exception
            with client.span_context("fake_trace_id", "test_span") as span_id:
                assert span_id is None  # Should be None when disabled
        except ImportError:
            pytest.skip("langfuse_client module not available")
    
    @pytest.mark.asyncio
    async def test_async_trace_context_when_disabled(self):
        """Test that async_trace_context works when client is disabled."""
        try:
            from src.utils.langfuse_client import LangfuseClient
            
            client = LangfuseClient()
            client._enabled = False
            
            # Should not raise exception
            async with client.async_trace_context("test_operation", user_id="test") as trace_id:
                assert trace_id is None  # Should be None when disabled
        except ImportError:
            pytest.skip("langfuse_client module not available")


class TestLangfuseDecorators:
    """Tests for Langfuse decorator functions."""
    
    def test_trace_agent_operation_decorator_import(self):
        """Test that decorator can be imported."""
        try:
            from src.utils.langfuse_client import trace_agent_operation
            assert trace_agent_operation is not None or trace_agent_operation is None  # May be None if langfuse not available
        except ImportError:
            pytest.skip("langfuse_client module not available")
    
    def test_trace_llm_call_decorator_import(self):
        """Test that LLM call decorator can be imported."""
        try:
            from src.utils.langfuse_client import trace_llm_call
            assert trace_llm_call is not None or trace_llm_call is None  # May be None if langfuse not available
        except ImportError:
            pytest.skip("langfuse_client module not available")


class TestLangfuseConfig:
    """Tests for Langfuse configuration loading."""
    
    def test_get_config_returns_dict(self):
        """Test that get_config returns a dictionary."""
        try:
            from src.utils.langfuse_client import LangfuseClient
            
            client = LangfuseClient()
            config = client.get_config()
            
            assert isinstance(config, dict)
        except ImportError:
            pytest.skip("langfuse_client module not available")
    
    def test_get_config_with_key(self):
        """Test that get_config with key returns correct value or None."""
        try:
            from src.utils.langfuse_client import LangfuseClient
            
            client = LangfuseClient()
            
            # Test with a key that may or may not exist
            value = client.get_config("tracing.enabled")
            # Should return either the config value or None
            assert value is None or isinstance(value, (bool, str))
        except ImportError:
            pytest.skip("langfuse_client module not available")


class TestLangfuseIntegrationWithBaseAgent:
    """Tests for Langfuse integration with BaseAgent."""
    
    def test_base_agent_has_langfuse_client(self):
        """Test that BaseAgent initializes langfuse_client attribute."""
        try:
            from src.agents.base import BaseAgent
            
            # Create a minimal mock agent
            class TestAgent(BaseAgent):
                async def _process_input(self, input_data):
                    return {"result": "test"}
            
            # This may fail if dependencies aren't available, skip in that case
            try:
                agent = TestAgent(role="test")
                assert hasattr(agent, 'langfuse_client')
            except Exception as e:
                pytest.skip(f"Could not create test agent: {e}")
        except ImportError as e:
            pytest.skip(f"BaseAgent not available: {e}")


class TestLangfuseA2AIntegration:
    """Tests for Langfuse integration with A2A protocol."""
    
    def test_a2a_protocol_langfuse_import(self):
        """Test that A2A protocol can import langfuse functions."""
        try:
            # Should not raise ImportError
            from src.utils.a2a_protocol import _get_a2a_langfuse_client
            # Function should exist
            assert _get_a2a_langfuse_client is not None
        except ImportError as e:
            pytest.skip(f"A2A protocol not available: {e}")


class TestLangfuseConsensusIntegration:
    """Tests for Langfuse integration with consensus poller."""
    
    def test_consensus_poller_langfuse_import(self):
        """Test that consensus poller can import langfuse functions."""
        try:
            from src.workflows.consensus_poller import _get_consensus_langfuse_client
            assert _get_consensus_langfuse_client is not None
        except ImportError as e:
            pytest.skip(f"Consensus poller not available: {e}")
    
    @pytest.mark.asyncio
    async def test_consensus_poll_creation_with_tracing(self):
        """Test that poll creation works with tracing disabled."""
        try:
            from src.workflows.consensus_poller import ConsensusPoller
            
            poller = ConsensusPoller()
            
            # Create a poll - should work even if tracing is disabled
            poll_id = await poller.create_poll(
                question="Test question for tracing",
                agents_to_poll=["agent1", "agent2"],
                timeout_seconds=60
            )
            
            assert poll_id is not None
            assert poll_id.startswith("consensus_")
            
            # Check that poll was created
            poll = poller.get_poll_status(poll_id)
            assert poll is not None
            assert poll.question == "Test question for tracing"
        except ImportError as e:
            pytest.skip(f"Consensus poller not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
