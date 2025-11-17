---
applyTo: '*ABC-Application*'
---
# AI Development Instructions for ABC-Application

## Model: Grok Code Fast 1
**Context**: You are an expert AI programming assistant working on a sophisticated multi-agent AI portfolio management system called ABC-Application.

## Core Principles

### 1. **Documentation-First Development**
- **ALWAYS** update relevant `.md` files when implementing new code
- Maintain coherence between documentation and implementation
- Use documentation as the source of truth for system architecture

### 2. **File Organization Standards**
- **Source Code**: `src/` directory only
- **Documentation**: `docs/` directory with subdirectories
- **Configuration**: `config/` directory
- **Tests**: `unit-tests/` and `integration-tests/` directories
- **Tools/Utilities**: `tools/` directory
- **Simulations**: `simulations/` directory
- **Data/Logs**: `data/` and `logs/` directories

### 3. **Code Quality Standards**
- **Type Hints**: Use comprehensive type annotations
- **Docstrings**: Google-style docstrings for all functions/classes
- **Error Handling**: Comprehensive try/except blocks with specific exceptions
- **Logging**: Structured logging with appropriate levels
- **Async/Await**: Proper async patterns for Discord bots and API calls

### 4. **Agent Architecture Compliance**
- **A2A Protocol**: All agent-to-agent communication must use the A2A protocol
- **Memory Systems**: Shared memory coordinator for cross-agent learning
- **LLM Integration**: Consistent ChatXAI_grok4 integration across all agents
- **Error Recovery**: Graceful degradation and circuit breakers

## File Organization Guide

### Source Code Structure (`src/`)
```
src/
├── main.py                 # Application entry point
├── agents/                 # Agent implementations
│   ├── base.py            # BaseAgent class
│   ├── macro.py           # MacroAgent
│   ├── data.py            # DataAgent
│   ├── strategy.py        # StrategyAgent
│   ├── risk.py            # RiskAgent
│   ├── reflection.py      # ReflectionAgent
│   ├── execution.py       # ExecutionAgent
│   ├── learning.py        # LearningAgent
│   ├── discord_agents.py  # Discord integration agents
│   └── live_workflow_orchestrator.py # Live workflow management
├── utils/                 # Utility modules
│   ├── tools.py           # Main tools aggregator
│   ├── validation.py      # Input validation
│   ├── financial_tools.py # Financial calculations
│   ├── news_tools.py      # News APIs
│   ├── market_data_tools.py # Market data
│   ├── backtesting_tools.py # Backtesting
│   ├── social_media_tools.py # Social sentiment
│   ├── agent_tools.py     # Agent coordination
│   └── a2a_protocol.py    # Agent-to-agent communication
├── workflows/             # Workflow implementations
│   └── iterative_reasoning_workflow.py # Iterative reasoning
└── monitoring/            # Monitoring and health checks
    └── api_health_dashboard.py # API monitoring tools
```

### Documentation Structure (`docs/`)
```
docs/
├── README.md              # Documentation navigation
├── architecture.md        # System architecture
├── ai-reasoning-agent-collaboration.md # AI reasoning framework
├── macro-micro-analysis-framework.md # Analysis methodology
├── production_readiness_checklist.md # Deployment checklist
├── security_hardening_guide.md # Security measures
├── AGENTS/                # Agent documentation
│   ├── index.md          # Agent overview
│   ├── macro-agent.md    # Macro agent specs
│   ├── data-agent.md     # Data agent specs
│   └── [other agents]
├── FRAMEWORKS/            # Framework documentation
│   ├── langchain-integration.md
│   ├── a2a-protocol.md
│   ├── memory-systems.md
│   └── workflows/        # Workflow documentation
├── IMPLEMENTATION/        # Implementation guides
│   ├── setup.md
│   ├── configuration.md
│   ├── discord-setup.md
│   ├── ibkr-deployment.md
│   └── vultr-deployment.md
└── REFERENCE/             # Reference materials
    ├── api-monitoring.md
    └── troubleshooting.md
```

### Configuration Structure (`config/`)
```
config/
├── risk-constraints.yaml     # Risk management rules
├── profitability-targets.yaml # Performance targets
├── environments/            # Environment-specific configs
├── defaults/               # Default configurations
└── ibkr_config.ini         # IBKR integration settings
```

## Development Workflow

### 1. **Planning Phase**
- Review relevant `.md` documentation first
- Understand current system architecture
- Identify integration points with existing agents

### 2. **Implementation Phase**
- **ALWAYS** update documentation before/after code changes
- Follow established patterns from existing agents
- Use proper async patterns for Discord integration
- Implement comprehensive error handling

### 3. **Testing Phase**
- Unit tests in `unit-tests/` directory
- Integration tests in `integration-tests/` directory
- Update test documentation as needed

### 4. **Documentation Update Phase**
- Update relevant `.md` files immediately after code changes
- Ensure coherence between docs and code
- Update architecture diagrams if needed
- Maintain API documentation

## Code Implementation Standards

### Agent Implementation Template
```python
"""
[Agent Name] Agent
[Brief description of agent's role and capabilities]
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.agents.base import BaseAgent
from src.utils.a2a_protocol import A2AProtocol

logger = logging.getLogger(__name__)

class [AgentName]Agent(BaseAgent):
    """
    [Detailed description of agent capabilities]
    """

    def __init__(self, a2a_protocol: A2AProtocol):
        super().__init__(
            name="[agent_name]",
            role="[agent_role]",
            a2a_protocol=a2a_protocol
        )
        self.capabilities = [
            "[capability_1]",
            "[capability_2]",
            # ...
        ]

    async def async_initialize_llm(self) -> None:
        """Initialize LLM with proper error handling"""
        try:
            # LLM initialization code
            logger.info(f"Successfully initialized LLM for {self.name} agent")
        except Exception as e:
            logger.error(f"Failed to initialize LLM for {self.name}: {e}")
            raise

    async def analyze(self, query: str) -> Dict[str, Any]:
        """
        Main analysis method for [agent role]

        Args:
            query: Analysis request

        Returns:
            Dict containing analysis results
        """
        try:
            # Analysis implementation
            result = {
                "llm_analysis": "Analysis result",
                "confidence": 0.85,
                "timestamp": datetime.now().isoformat()
            }
            return result
        except Exception as e:
            logger.error(f"Analysis failed for {self.name}: {e}")
            return {
                "error": str(e),
                "fallback_analysis": "Basic analysis due to error"
            }
```

### Documentation Update Template
When implementing new features, update these files:

1. **Architecture Documentation** (`docs/architecture.md`)
   - Update component diagrams
   - Add new integration points
   - Document data flow changes

2. **Agent Documentation** (`docs/AGENTS/[agent].md`)
   - Update capabilities list
   - Document new methods
   - Update configuration requirements

3. **Framework Documentation** (`docs/FRAMEWORKS/`)
   - Update integration guides
   - Document protocol changes
   - Add implementation examples

4. **Implementation Guide** (`docs/IMPLEMENTATION/`)
   - Update setup instructions
   - Document configuration changes
   - Add troubleshooting guides

## Quality Assurance Checklist

### Before Code Submission:
- [ ] Documentation updated and coherent with code
- [ ] Type hints added to all functions
- [ ] Comprehensive error handling implemented
- [ ] Logging statements added appropriately
- [ ] Unit tests created/updated
- [ ] Integration tests pass
- [ ] Code follows established patterns

### Before Documentation Updates:
- [ ] Code implementation complete and tested
- [ ] All new features documented
- [ ] Architecture diagrams updated if needed
- [ ] Cross-references between docs verified
- [ ] Examples and code snippets accurate

## Communication Standards

### Commit Messages:
```
feat: Add [feature] to [component]
- Update docs/[file].md with new capabilities
- Add unit tests for [feature]
- Update architecture.md diagram

fix: Resolve [issue] in [component]
- Update docs/[file].md with fix details
- Add regression test

docs: Update [documentation] for [feature]
- Ensure coherence with code implementation
- Add examples and usage instructions
```

### Code Comments:
- Use descriptive variable names
- Add comments for complex logic
- Document assumptions and edge cases
- Reference related documentation

## Performance and Reliability

### Error Handling:
- Use specific exception types
- Implement circuit breakers for external APIs
- Provide meaningful error messages
- Log errors with appropriate context

### Performance:
- Use async/await for I/O operations
- Implement caching where appropriate
- Monitor memory usage
- Profile performance-critical sections

### Reliability:
- Implement health checks
- Use timeouts for external calls
- Handle rate limiting gracefully
- Provide fallback mechanisms

This instructions file ensures consistent, high-quality development that maintains coherence between documentation and implementation while following established patterns and best practices.