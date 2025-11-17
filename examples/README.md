# [LABEL:DOC:examples] [LABEL:DOC:readme] [LABEL:FRAMEWORK:python]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Documentation for example scripts and demonstrations
# Dependencies: Python 3.11+, ABC Application source code
# Related: src/, docs/EXAMPLES/, README.md
#
# Examples Directory

This directory contains example scripts and demonstrations for the ABC Application system.

## Memory Agent Examples

### `memory_dashboard.py`
Real-time dashboard for monitoring MemoryAgent activities and system state.

**Usage:**
```bash
python examples/memory_dashboard.py
```

**Features:**
- Real-time system overview
- Agent activity monitoring
- Memory usage statistics
- Health metrics display
- Interactive dashboard mode

### `memory_query.py`
Simple query interface for direct access to MemoryAgent capabilities.

**Usage:**
```bash
# Show system status
python examples/memory_query.py status

# Query specific agent activity
python examples/memory_query.py agent DataAgent

# Search memories
python examples/memory_query.py search "trading strategy"

# Interactive mode
python examples/memory_query.py
```

**Features:**
- Direct MemoryAgent queries
- Agent activity analysis
- Memory search functionality
- System collaboration visualization

## Running Examples

All examples require the ABC Application source code to be available. Make sure to run from the project root:

```bash
cd /path/to/abc-application
python examples/memory_dashboard.py
```

## Dependencies

- Python 3.11+
- ABC Application source code (`src/`)
- Required packages from `requirements.txt`
- Optional: `rich` for enhanced terminal display