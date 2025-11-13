# Testing Folder Structure

This folder contains all testing-related files for the ABC-Application project.

## Structure
- **unit-tests/**: Pure automated unit tests (e.g., test_tools.py, test_agents_core.py). Run with `pytest unit-tests/`.
- **integration-tests/**: Integration tests for system components (e.g., comprehensive_test.py). Run with `pytest integration-tests/`.
- **demos/**: Scripts that serve as both testable examples and demonstrations (e.g., trading_simulator.py). These can be run directly (e.g., `python testing/demos/trading_simulator.py`) to showcase features, and may include test assertions.

To run all tests: `pytest testing/`

For demo mode on a script: Run it directly or with a flag if implemented (e.g., `--demo` for verbose output).