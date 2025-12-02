---
applyTo: '*ABC-Application*'
---
# ABC-Application File Organization Guide

## Directory Structure Overview

```
ABC-Application/
├── .github/                    # GitHub Actions and templates
├── config/                     # Configuration files
├── data/                       # Data files, logs, cache
├── docs/                       # Documentation
├── integration-tests/          # Integration tests
├── logs/                       # Application logs
├── myenv/                      # Python virtual environment
├── redis/                      # Redis server files
├── setup/                      # Setup and installation scripts
├── simulations/                # Backtesting and simulation scripts
├── src/                        # Source code (PRIMARY LOCATION)
│   └── integrations/           # External service integrations
├── tools/                      # Utility scripts and tools
├── unit-tests/                 # Unit tests
├── AI_DEVELOPMENT_INSTRUCTIONS.md  # Development guidelines
├── README.md                   # Project overview
├── requirements.txt            # Python dependencies
└── pytest.ini                  # Testing configuration
```

## Performance and Privacy Considerations

### Performance Files
- **Profiling Results**: Store in `data/` or `logs/` directories
- **Cache Files**: Use `data/cache/` for Redis dumps or local caches
- **Optimization Scripts**: Place in `tools/` for profiling utilities

### Privacy Files
- **Audit Logs**: Store in `logs/` with encryption
- **Consent Records**: Use `data/consent/` for user agreements
- **Anonymized Data**: Store processed data in `data/anonymized/`

## File Placement Rules

### 🚫 **DO NOT PLACE** source code in root directory
**Wrong:**
```
ABC-Application/
├── discord_agents.py          # ❌ Should be in src/
├── live_workflow_orchestrator.py  # ❌ Should be in src/
└── src/
```

**Correct:**
```
ABC-Application/
├── src/
│   ├── agents/
│   │   ├── discord_agents.py
│   │   └── live_workflow_orchestrator.py
│   └── main.py
└── tools/                     # For utility scripts
```

### 📁 **Source Code** → `src/` directory only
```
src/
├── main.py                    # Application entry point
├── agents/                    # Agent implementations
│   ├── base.py               # BaseAgent class
│   ├── macro.py              # MacroAgent
│   ├── data.py               # DataAgent
│   ├── strategy.py           # StrategyAgent
│   ├── risk.py               # RiskAgent
│   ├── reflection.py         # ReflectionAgent
│   ├── execution.py          # ExecutionAgent
│   ├── learning.py           # LearningAgent
│   ├── discord_agents.py     # Discord integration agents
│   └── live_workflow_orchestrator.py # Live workflow management
├── utils/                    # Utility modules
│   ├── tools.py              # Main tools aggregator
│   ├── validation.py         # Input validation
│   ├── financial_tools.py    # Financial calculations
│   ├── news_tools.py         # News APIs
│   ├── market_data_tools.py  # Market data
│   ├── backtesting_tools.py  # Backtesting
│   ├── social_media_tools.py # Social sentiment
│   ├── agent_tools.py        # Agent coordination
│   └── a2a_protocol.py       # Agent-to-agent communication
├── workflows/                # Workflow implementations
│   └── iterative_reasoning_workflow.py # Iterative reasoning
└── monitoring/               # Monitoring and health checks
    └── api_health_dashboard.py # API monitoring tools
```

### 📚 **Documentation** → `docs/` directory
```
docs/
├── README.md                  # Documentation index
├── architecture.md            # System architecture
├── AGENTS/                    # Agent documentation
├── FRAMEWORKS/                # Framework guides
├── IMPLEMENTATION/            # Implementation guides
└── REFERENCE/                 # Reference materials
```

### ⚙️ **Configuration** → `config/` directory
```
config/
├── risk-constraints.yaml
├── profitability-targets.yaml
├── environments/
└── defaults/
```

### 🧪 **Tests** → Appropriate test directories
```
unit-tests/                    # Unit tests
├── test_agents.py
├── test_utils.py
└── test_data.py

integration-tests/            # Integration tests
├── comprehensive_test.py
├── discord_integration_test.py
└── workflow_integration_test.py
```

### 🛠️ **Tools/Utilities** → `tools/` directory
```
tools/
├── check_bot_status.py       # Bot status checking
├── debug_channels.py         # Discord channel debugging
├── quick_workflow_test.py    # Workflow testing utilities
├── setup_discord.py          # Discord setup utilities
├── start_live_workflow.py    # Live workflow starters
├── test_grok.py              # Grok API testing
├── workflow_status_tracker.py # Workflow monitoring
└── monitoring/               # Monitoring tools
    └── deployment_scripts/   # Deployment utilities
```

## Quality Assurance

### File Organization Audit:
- [ ] No source code in root directory
- [ ] All Python modules in appropriate `src/` subdirectories
- [ ] Documentation properly organized in `docs/`
- [ ] Tests in correct test directories
- [ ] Tools/utilities in `tools/` directory
- [ ] Configuration files in `config/`

### Import Audit:
- [ ] All imports use correct paths after file moves
- [ ] No broken imports in the codebase
- [ ] Relative imports work correctly
- [ ] External dependencies properly declared

### Documentation Audit:
- [ ] All file references updated after moves
- [ ] Code examples work with new paths
- [ ] Architecture docs match current implementation
- [ ] Setup instructions accurate

## Benefits of Proper Organization

1. **Maintainability**: Clear separation of concerns
2. **Scalability**: Easy to add new components
3. **Collaboration**: Team members know where to find/look for files
4. **Deployment**: Clear structure for packaging and distribution
5. **Testing**: Isolated test environments
6. **Documentation**: Coherent docs that match implementation

## Quick Reference

### Adding New Components:

**New Agent:**
```
# Implementation: src/agents/new_agent.py
# Documentation: docs/AGENTS/new-agent.md
# Tests: unit-tests/test_new_agent.py
# Config: config/new_agent.yaml (if needed)
```

**New Utility:**
```
# Implementation: src/utils/new_utility.py
# Documentation: docs/FRAMEWORKS/new-utility.md
# Tests: unit-tests/test_new_utility.py
```

**New Tool:**
```
# Implementation: tools/new_tool.py
# Documentation: docs/REFERENCE/new-tool.md
```

**New Simulation:**
```
# Implementation: simulations/new_simulation.py
# Documentation: docs/IMPLEMENTATION/simulations.md (add section)
```

This organization ensures clean, maintainable code that scales well and maintains coherence between documentation and implementation.