import pytest
from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator

@pytest.fixture
def orchestrator():
    return LiveWorkflowOrchestrator()

def test_initialization(orchestrator):
    assert orchestrator.workflow_active is False
    assert orchestrator.current_phase == "waiting"
    assert isinstance(orchestrator.phase_delays, dict)
    assert len(orchestrator.phase_commands) > 0
    assert orchestrator.collaborative_session_id is None

def test_phase_delays(orchestrator):
    for delay in orchestrator.phase_delays.values():
        assert delay == 300  # Phase delays are 5 minutes (300 seconds) in current implementation

def test_agent_initialization(orchestrator):
    # Assuming async init is called separately, test structure
    assert isinstance(orchestrator.agent_instances, dict)

# Test simulation mode (assuming it's implemented)
def test_simulation_mode(orchestrator):
    # This would need to mock the run method or check flags
    pass  # Implement actual test based on code

# Add more tests as needed
