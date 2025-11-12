#!/usr/bin/env python3
"""
Unit tests for configuration management functionality.
Tests API key loading, environment variable handling, and configuration validation.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import (
    get_api_key,
    get_marketdataapp_api_key,
    get_fred_api_key,
    get_news_api_key,
    get_grok_api_key,
    get_kalshi_api_key,
    get_kalshi_access_key_id,
    get_twitter_bearer_token
)


class TestConfigurationManagement:
    """Test cases for configuration management functionality."""

    @patch.dict(os.environ, {}, clear=True)
    def test_get_api_key_default_env_var(self):
        """Test get_api_key with default environment variable naming."""
        # Test with existing environment variable
        test_key = "test_api_key_value"
        os.environ["TEST_API_KEY"] = test_key

        result = get_api_key("test")
        assert result == test_key

    @patch.dict(os.environ, {}, clear=True)
    def test_get_api_key_custom_env_var(self):
        """Test get_api_key with custom environment variable name."""
        test_key = "custom_key_value"
        os.environ["CUSTOM_VAR"] = test_key

        result = get_api_key("test", "CUSTOM_VAR")
        assert result == test_key

    @patch.dict(os.environ, {}, clear=True)
    def test_get_api_key_missing(self):
        """Test get_api_key when environment variable is missing."""
        result = get_api_key("nonexistent")
        assert result == ""

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.utils.config.logger')
    def test_get_api_key_logging_success(self, mock_logger):
        """Test logging when API key is successfully loaded."""
        test_key = "success_key"
        os.environ["TEST_API_KEY"] = test_key

        result = get_api_key("test")

        assert result == test_key
        mock_logger.info.assert_called_with("Loaded API key for test")

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.utils.config.logger')
    def test_get_api_key_logging_missing(self, mock_logger):
        """Test logging when API key is missing."""
        result = get_api_key("missing")

        assert result == ""
        mock_logger.warning.assert_called_with("API key for missing not found in environment variables")

    @patch.dict(os.environ, {}, clear=True)
    def test_get_marketdataapp_api_key(self):
        """Test get_marketdataapp_api_key function."""
        test_key = "marketdata_key"
        os.environ["MARKETDATAAPP_API_KEY"] = test_key

        result = get_marketdataapp_api_key()
        assert result == test_key

    @patch.dict(os.environ, {}, clear=True)
    def test_get_fred_api_key(self):
        """Test get_fred_api_key function."""
        test_key = "fred_key"
        os.environ["FRED_API_KEY"] = test_key

        result = get_fred_api_key()
        assert result == test_key

    @patch.dict(os.environ, {}, clear=True)
    def test_get_news_api_key(self):
        """Test get_news_api_key function."""
        test_key = "news_key"
        os.environ["NEWS_API_KEY"] = test_key

        result = get_news_api_key()
        assert result == test_key

    @patch.dict(os.environ, {}, clear=True)
    def test_get_grok_api_key(self):
        """Test get_grok_api_key function."""
        test_key = "grok_key"
        os.environ["GROK_API_KEY"] = test_key

        result = get_grok_api_key()
        assert result == test_key

    @patch.dict(os.environ, {}, clear=True)
    def test_get_kalshi_api_key_formatted(self):
        """Test get_kalshi_api_key with PEM formatting."""
        # Test with raw key that needs PEM formatting
        raw_key = "MIIEpAIBAAKCAQEA..."  # Shortened for test
        os.environ["KALSHI_API_KEY"] = raw_key

        result = get_kalshi_api_key()

        # Should have PEM headers and footers
        assert result.startswith("-----BEGIN RSA PRIVATE KEY-----")
        assert result.endswith("-----END RSA PRIVATE KEY-----\n")
        # Should contain the original key
        assert raw_key in result.replace("\n", "")

    @patch.dict(os.environ, {}, clear=True)
    def test_get_kalshi_api_key_already_formatted(self):
        """Test get_kalshi_api_key when key is already PEM formatted."""
        pem_key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----\n"
        os.environ["KALSHI_API_KEY"] = pem_key

        result = get_kalshi_api_key()

        # Should return as-is since it's already formatted
        assert result == pem_key

    @patch.dict(os.environ, {}, clear=True)
    def test_get_kalshi_api_key_empty(self):
        """Test get_kalshi_api_key when key is empty."""
        result = get_kalshi_api_key()
        assert result == ""

    @patch.dict(os.environ, {}, clear=True)
    def test_get_kalshi_access_key_id(self):
        """Test get_kalshi_access_key_id function."""
        test_key_id = "kalshi_key_id_123"
        os.environ["KALSHI_ACCESS_KEY_ID"] = test_key_id

        result = get_kalshi_access_key_id()
        assert result == test_key_id

    @patch.dict(os.environ, {}, clear=True)
    def test_get_twitter_bearer_token(self):
        """Test get_twitter_bearer_token function."""
        test_token = "twitter_bearer_token_123"
        os.environ["TWITTER_BEARER_TOKEN"] = test_token

        result = get_twitter_bearer_token()
        assert result == test_token

    @patch('src.utils.config.load_dotenv')
    def test_dotenv_loading(self, mock_load_dotenv):
        """Test that dotenv is loaded on module import."""
        # This test verifies that load_dotenv is called during module initialization
        # We need to reload the module to test this properly
        import importlib
        import src.utils.config

        # Reload the module to trigger initialization
        importlib.reload(src.utils.config)

        # Verify load_dotenv was called
        mock_load_dotenv.assert_called_once()

    def test_pem_formatting_edge_cases(self):
        """Test PEM formatting edge cases."""
        from src.utils.config import get_kalshi_api_key

        # Test with empty string
        with patch.dict(os.environ, {"KALSHI_API_KEY": ""}):
            result = get_kalshi_api_key()
            assert result == ""

        # Test with very short key
        short_key = "ABC"
        with patch.dict(os.environ, {"KALSHI_API_KEY": short_key}):
            result = get_kalshi_api_key()
            assert result.startswith("-----BEGIN RSA PRIVATE KEY-----")
            assert result.endswith("-----END RSA PRIVATE KEY-----\n")
            assert short_key in result

        # Test with key that exactly fits line length
        exact_key = "A" * 64
        with patch.dict(os.environ, {"KALSHI_API_KEY": exact_key}):
            result = get_kalshi_api_key()
            lines = result.strip().split('\n')
            # Should have header, one data line, footer
            assert len(lines) == 3
            assert lines[1] == exact_key

        # Test with key longer than line length
        long_key = "A" * 100
        with patch.dict(os.environ, {"KALSHI_API_KEY": long_key}):
            result = get_kalshi_api_key()
            lines = result.strip().split('\n')
            # Should have header, two data lines (64 + 36), footer
            assert len(lines) == 4
            assert lines[1] == "A" * 64
            assert lines[2] == "A" * 36

    def test_environment_variable_precedence(self):
        """Test that environment variables take precedence over .env file."""
        # This is more of an integration test, but we can mock the behavior
        with patch.dict(os.environ, {"TEST_API_KEY": "env_value"}):
            with patch('os.getenv') as mock_getenv:
                mock_getenv.return_value = "env_value"

                result = get_api_key("test")
                assert result == "env_value"

    def test_service_name_case_insensitive(self):
        """Test that service names are handled case-insensitively for env vars."""
        # Environment variables are typically uppercase, but test the logic
        test_key = "case_test_key"
        os.environ["CASE_TEST_API_KEY"] = test_key

        result = get_api_key("case_test")
        assert result == test_key

    @patch.dict(os.environ, {}, clear=True)
    def test_multiple_services_independence(self):
        """Test that different services don't interfere with each other."""
        # Set up different keys for different services
        os.environ["SERVICE1_API_KEY"] = "key1"
        os.environ["SERVICE2_API_KEY"] = "key2"
        os.environ["SERVICE3_API_KEY"] = "key3"

        result1 = get_api_key("service1")
        result2 = get_api_key("service2")
        result3 = get_api_key("service3")

        assert result1 == "key1"
        assert result2 == "key2"
        assert result3 == "key3"

    def test_custom_env_var_override(self):
        """Test that custom env_var parameter overrides default naming."""
        test_key = "custom_override_key"
        os.environ["MY_CUSTOM_VAR"] = test_key

        # Use custom env var name
        result = get_api_key("service", "MY_CUSTOM_VAR")
        assert result == test_key

        # Verify default env var is not used
        assert "SERVICE_API_KEY" not in os.environ or os.environ["SERVICE_API_KEY"] != result


if __name__ == "__main__":
    pytest.main([__file__])