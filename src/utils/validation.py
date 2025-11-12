#!/usr/bin/env python3
"""
Data validation and circuit breaker utilities.
Provides input validation, sanitization, and circuit breaker patterns for API resilience.
"""

import re
import time
import logging
from typing import Dict, Any, Callable, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation and sanitization utilities.
    """

    @staticmethod
    def sanitize_text_input(text: str, max_length: int = 10000) -> str:
        """
        Sanitize text input for processing by removing HTML/script tags and excessive whitespace.
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
        Returns:
            str: Sanitized text
        """
        import re

        if not isinstance(text, str):
            return ""

        # Remove HTML tags and script content
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)  # Remove script tags and content
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)  # Remove style tags and content

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text.strip()

    @staticmethod
    def validate_api_response(response_data: Any, required_fields: list = None) -> bool:
        """
        Validate API response structure and required fields.
        Args:
            response_data: Response data to validate
            required_fields: List of required field names
        Returns:
            bool: True if valid, False otherwise
        """
        if response_data is None:
            return False

        if required_fields:
            if not isinstance(response_data, dict):
                return False
            for field in required_fields:
                if field not in response_data:
                    return False

        return True

    @staticmethod
    def detect_data_anomalies(articles: list, sentiment_scores: list = None) -> list:
        """
        Detect anomalies in news data and sentiment analysis.
        Args:
            articles: List of news articles
            sentiment_scores: List of sentiment scores
        Returns:
            list: List of detected anomalies
        """
        anomalies = []

        if not articles:
            anomalies.append("No articles in news data")
            return anomalies

        # Validate article structure
        if not all(isinstance(article, dict) for article in articles):
            anomalies.append("Invalid article format - not all articles are dictionaries")

        # Check for empty articles
        empty_articles = sum(1 for article in articles if not article.get('title', '').strip())
        if empty_articles > len(articles) * 0.5:  # More than 50% empty
            anomalies.append(f"Too many empty articles: {empty_articles}/{len(articles)}")

        # Check for duplicate titles
        titles = [article.get('title', '') for article in articles if article.get('title')]
        if len(titles) != len(set(titles)):
            anomalies.append("Duplicate article titles detected")

        # Check for very old articles
        import pandas as pd
        try:
            if articles:
                current_time = pd.Timestamp.now()
                old_articles = 0
                for article in articles:
                    pub_date = article.get('published_at', '')
                    if pub_date:
                        try:
                            article_date = pd.to_datetime(pub_date)
                            if (current_time - article_date).days > 30:  # Older than 30 days
                                old_articles += 1
                        except (ValueError, KeyError, AttributeError, TypeError):
                            pass

                if old_articles > len(articles) * 0.8:  # More than 80% old
                    anomalies.append(f"Mostly old articles: {old_articles}/{len(articles)} older than 30 days")
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            anomalies.append(f"Error during anomaly detection: {str(e)}")

        return anomalies


def validate_tool_inputs(**validators):
    """
    Decorator to validate tool function inputs.
    Args:
        **validators: Validation functions for each parameter
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for parameter '{param_name}': {value}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for API resilience.
    """

    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: int = 300):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        """
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise CircuitBreakerOpenException(f"Circuit breaker '{self.name}' is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout
        }


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global circuit breaker registry
_circuit_breakers = {}


def get_circuit_breaker(name: str, failure_threshold: int = 3, recovery_timeout: int = 300) -> CircuitBreaker:
    """
    Get or create a circuit breaker instance.
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, failure_threshold, recovery_timeout)
    return _circuit_breakers[name]


def circuit_breaker(api_name: str, failure_threshold: int = 3, recovery_timeout: int = 300):
    """
    Decorator to apply circuit breaker pattern to API calls.
    """
    def decorator(func):
        breaker = get_circuit_breaker(api_name, failure_threshold, recovery_timeout)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        return wrapper
    return decorator


def get_circuit_breaker_status() -> Dict[str, Any]:
    """
    Get status of all circuit breakers.
    """
    return {name: breaker.get_status() for name, breaker in _circuit_breakers.items()}