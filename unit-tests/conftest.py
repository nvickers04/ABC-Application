# [LABEL:TEST:config] [LABEL:FRAMEWORK:pytest] [LABEL:FRAMEWORK:pytest_asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Pytest configuration and fixtures for unit test suite
# Dependencies: pytest, pytest-asyncio
# Related: unit-tests/*.py, pytest.ini
#
import pytest
import asyncio
import sys
import os

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio