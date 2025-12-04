# [LABEL:TEST:integration_config] [LABEL:FRAMEWORK:pytest] [LABEL:FRAMEWORK:pytest_asyncio]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-04] [LABEL:REVIEWED:pending]
#
# Purpose: Pytest configuration and fixtures for integration test suite
# Dependencies: pytest, pytest-asyncio
# Related: integration-tests/*.py, pytest.ini
#
import pytest
import asyncio
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure pytest-asyncio for integration tests
pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "redis_host": "localhost",
        "redis_port": 6379,
        "ibkr_host": "localhost",
        "ibkr_port": 7497,
        "log_level": "INFO",
        "test_mode": True
    }

@pytest.fixture
async def health_monitor():
    """Initialize health monitor for integration tests."""
    from src.utils.api_health_monitor import start_health_monitoring
    # Start monitoring with short interval for tests
    start_health_monitoring(check_interval=1)
    yield
    # Cleanup would happen here if needed

@pytest.fixture
async def unified_orchestrator(test_config):
    """Create UnifiedWorkflowOrchestrator instance for testing."""
    from src.agents.unified_workflow_orchestrator import UnifiedWorkflowOrchestrator, WorkflowMode

    orchestrator = UnifiedWorkflowOrchestrator(
        mode=WorkflowMode.ANALYSIS,
        enable_discord=False,
        config=test_config
    )
    yield orchestrator
    # Cleanup
    if orchestrator.is_running:
        await orchestrator.stop()

@pytest.fixture
async def component_health_monitor():
    """Initialize component health monitor."""
    from src.utils.component_health_monitor import get_component_health_monitor

    monitor = get_component_health_monitor()
    monitor.start_monitoring()
    yield monitor
    monitor.stop_monitoring()

@pytest.fixture
async def alert_manager():
    """Initialize alert manager."""
    from src.utils.alert_manager import get_alert_manager

    manager = get_alert_manager()
    yield manager</content>
</xai:function_call name="create_file">
<parameter name="filePath">c:\Users\nvick\ABC-Application\integration-tests\test_unified_workflow_integration.py