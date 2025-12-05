# src/utils/config.py
# Purpose: Centralized configuration management for API keys and environment variables.
# Loads sensitive data from .env file securely, ensuring API keys are never committed to version control.
# Structural Reasoning: Separates config from utilities for better organization; uses python-dotenv for secure loading.
# Ties to security best practices: Never store API keys in code; load from environment variables.

import os
from dotenv import load_dotenv
import logging
import yaml

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
_loaded = False

def get_api_key(service_name: str, env_var: str = None) -> str:
    """
    Retrieves an API key from environment variables.

    Args:
        service_name (str): Name of the service (e.g., 'massive', 'fred') for logging
        env_var (str): Environment variable name. If None, uses uppercase service_name + '_API_KEY'

    Returns:
        str: The API key if found, empty string if not found

    Reasoning: Centralized key retrieval with logging; fails gracefully for development/testing.
    """
    global _loaded
    if not _loaded:
        load_dotenv()
        _loaded = True

    if env_var is None:
        env_var = f"{service_name.upper()}_API_KEY"

    api_key = os.getenv(env_var, "")

    if api_key:
        logger.info(f"Loaded API key for {service_name}")
    else:
        logger.warning(f"API key for {service_name} not found in environment variables")

    return api_key

# Specific API key getters for convenience
def get_marketdataapp_api_key() -> str:
    """Get MarketDataApp API key"""
    return get_api_key("marketdataapp")

def get_fred_api_key() -> str:
    """Get FRED API key"""
    return get_api_key("fred")

def get_news_api_key() -> str:
    """Get News API key"""
    return get_api_key("news")

def get_grok_api_key() -> str:
    """Get Grok API key"""
    return get_api_key("grok")

def get_kalshi_api_key() -> str:
    """Get Kalshi API key (RSA private key)"""
    key = get_api_key("kalshi")
    # Add PEM headers if not present
    if key and not key.startswith('-----BEGIN'):
        # Format the key with proper line breaks for PEM
        formatted_key = "-----BEGIN RSA PRIVATE KEY-----\n"
        # Insert line breaks every 64 characters
        for i in range(0, len(key), 64):
            formatted_key += key[i:i+64] + "\n"
        formatted_key += "-----END RSA PRIVATE KEY-----\n"
        return formatted_key
    return key

def get_kalshi_access_key_id() -> str:
    """Get Kalshi API Key ID"""
    return get_api_key("kalshi", "KALSHI_ACCESS_KEY_ID")

def get_twitter_bearer_token() -> str:
    """Get Twitter Bearer Token"""
    return get_api_key("twitter", "TWITTER_BEARER_TOKEN")



# Example usage and testing
if __name__ == "__main__":
    # Test loading API keys
    marketdataapp_key = get_marketdataapp_api_key()
    fred_key = get_fred_api_key()
    news_key = get_news_api_key()
    grok_key = get_grok_api_key()

    print(f"MarketDataApp API Key loaded: {'Yes' if marketdataapp_key else 'No'}")
    print(f"FRED API Key loaded: {'Yes' if fred_key else 'No'}")
    print(f"News API Key loaded: {'Yes' if news_key else 'No'}")
    print(f"Grok API Key loaded: {'Yes' if grok_key else 'No'}")