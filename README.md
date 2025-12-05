# ABC-Application: AI Portfolio Manager

## Table of Contents
- [Project Overview](#project-overview)
- [Core Innovation](#core-innovation-ai-reasoning-through-agent-collaboration)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Architecture](#architecture)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Project Overview

ABC-Application is a sophisticated multi-agent AI system for quantitative portfolio management that integrates Grok-powered reasoning with Interactive Brokers (IBKR) for professional-grade trading execution. The system operates on a macro-to-micro analysis hierarchy, enabling systematic market scanning that informs strategy proposals, risk assessments, and precise trade executions.

### Key Objectives
- **Performance Target**: Achieve 10-20% monthly returns with <5% maximum drawdown
- **Risk Management**: Probability-of-profit (POP) evaluations with stochastic batching
- **Continuous Learning**: Reflection loops for iterative system improvement
- **Compliance**: Full audit trails and SEC-compliant decision logging

### Technology Stack
- **AI/ML**: Grok API (xAI), LangChain, LangGraph for agent orchestration
- **Trading**: Interactive Brokers API, professional market data feeds
- **Infrastructure**: Python asyncio, Redis caching, collaborative memory systems
- **Monitoring**: API health monitoring, circuit breakers, Discord integration

## Core Innovation: AI Reasoning Through Agent Collaboration

The fundamental breakthrough of ABC-Application is its **8-agent collaborative reasoning architecture**. Rather than relying on a single AI model for decision-making, the system creates a sophisticated debate and deliberation environment where specialized AI agents collaboratively reason through complex investment decisions.

### Agent Framework
- **8 Specialized Agents**: Each agent represents a domain of financial expertise working in orchestrated reasoning loops
- **Collaborative Intelligence**: Agents debate and consensus-build through structured deliberation
- **Autonomous Operation**: Agents make decisions using LLM reasoning and tool interactions
- **Memory Integration**: Shared memory systems enable cross-agent learning and adaptation

### Single Iteration Workflow
1. **Macro Agent**: Establishes market context and identifies top opportunities
2. **Data Agent**: Multi-source data validation and sentiment analysis
3. **Strategy Agent**: Options, flow, and ML strategy generation with debate
4. **Risk Agent**: Probability-of-profit evaluations and risk assessments
5. **Execution Agent**: Trade execution with real-time monitoring
6. **Reflection Agent**: Decision validation and continuous improvement (supreme oversight)
7. **Learning Agent**: Performance analysis and model refinement

## Quick Start

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)
- Interactive Brokers account (for live trading)

### Installation
```bash
# Clone the repository
git clone https://github.com/nvickers04/ABC-Application.git
cd ABC-Application

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Unix/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your API keys and settings
```

### Running the System
```bash
# Start the main system (automatically starts required services like Redis)
python src/main.py

# Run paper trading workflow
python src/agents/unified_workflow_orchestrator.py --mode hybrid --symbols SPY QQQ

# Monitor API health
python tools/api_health_dashboard.py
```

**Note**: Required services (Redis, etc.) are started automatically. If services fail to start, the system will fall back to in-memory alternatives with reduced performance.

## Documentation

### 📚 Documentation Structure
```
docs/
├── architecture.md              # System design and data flows
├── AGENTS/                      # Agent documentation
│   ├── index.md                # Agent inventory and coordination
│   └── [agent-specific].md     # Individual agent docs
├── FRAMEWORKS/                 # Technical frameworks
│   ├── macro-micro-analysis.md # Analysis methodology
│   ├── langchain-integration.md # LLM integration
│   └── a2a-protocol.md         # Agent communication
├── IMPLEMENTATION/             # Setup and deployment
│   ├── setup-and-development.md # Setup and development guide
│   ├── configuration.md       # Config management
└── REFERENCE/                  # Operational docs
    ├── api-monitoring.md      # Health monitoring
    ├── performance.md         # Performance optimization
    ├── security.md            # Security hardening
    └── troubleshooting.md     # Common issues
```

### 🔗 Key Documentation Links
- **[System Architecture](./docs/architecture.md)**: Complete system design and data flows
- **[Agent Framework](./docs/AGENTS/index.md)**: All 8 agents with capabilities and coordination
- **[Macro-Micro Analysis](./docs/FRAMEWORKS/macro-micro-analysis-framework.md)**: Analysis methodology
- **[A2A Protocol](./docs/FRAMEWORKS/a2a-protocol.md)**: Agent-to-agent communication
- **[Performance Optimization](./docs/REFERENCE/performance.md)**: System performance tuning
- **[Security Hardening](./docs/security_hardening_guide.md)**: Security best practices
- **[API Monitoring](./docs/REFERENCE/api-monitoring.md)**: Health monitoring and alerting
- **[Production Deployment](./docs/production_readiness_checklist.md)**: Deployment and readiness checklist

## Architecture

### System Design
ABC-Application follows a **macro-to-micro analysis framework** with layered intelligence:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Macro Layer   │ -> │  Micro Layer    │ -> │ Execution Layer │
│                 │    │                 │    │                 │
│ • Market Regime │    │ • Asset Analysis│    │ • Trade Orders  │
│ • Sector Scan   │    │ • Strategy Gen  │    │ • Risk Mgmt     │
│ • Opportunity ID│    │ • Backtesting   │    │ • Position Mgmt │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Agent Orchestration
The system uses **LangGraph-powered orchestration** for agent coordination:

- **StateGraph Workflow**: Deterministic agent execution flow with conditional branching
- **A2A Protocol**: Asynchronous agent-to-agent communication with message queuing
- **Memory Integration**: Shared Redis-backed memory for cross-agent learning
- **Discord Monitoring**: Real-time workflow monitoring and human intervention

### Data Pipeline
```
Market Data → Validation → Analysis → Strategy → Risk → Execution → Learning
     ↓            ↓          ↓         ↓        ↓         ↓          ↓
   APIs       Circuit      Agents    Models   Checks   Orders   Feedback
```

## Development

### 🛠️ Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/ tests/
black src/ tests/

# Run type checking
mypy src/
```

### 📁 Project Structure
```
ABC-Application/
├── src/                    # Source code
│   ├── agents/            # Agent implementations
│   │   ├── base.py       # Base agent classes
│   │   ├── data.py       # Data collection agents
│   │   ├── strategy.py   # Strategy agents
│   │   └── ...
│   ├── utils/            # Utility modules
│   │   ├── a2a_protocol.py    # Agent communication
│   │   ├── redis_cache.py     # Caching layer
│   │   ├── langfuse_client.py # Tracing
│   │   └── ...
│   └── main.py           # Application entry point
├── docs/                 # Documentation
├── tests/               # Test suites
├── tools/               # Development tools
├── config/              # Configuration files
├── examples/            # Usage examples
└── simulations/         # Backtesting scripts
```

### 🔧 Key Components
- **Agent Framework**: Modular agent system with inheritance and composition
- **Memory Systems**: Redis-backed collaborative memory with persistence
- **API Integration**: Circuit breaker pattern for external API resilience
- **Monitoring**: Comprehensive health monitoring and alerting
- **Configuration**: YAML-based configuration with environment overrides

## Testing

### 🧪 Test Structure
```
tests/
├── unit/                 # Unit tests
│   ├── test_agents.py   # Agent functionality
│   ├── test_utils.py    # Utility functions
│   └── test_memory.py   # Memory systems
├── integration/         # Integration tests
│   ├── test_workflow.py # Full workflow tests
│   └── test_api.py      # External API tests
└── fixtures/            # Test data and mocks
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/        # Unit tests only
pytest tests/integration/ # Integration tests only

# Fast incremental testing
pytest --testmon -n auto --ff -ra -q
```

### Test Coverage
- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: End-to-end workflow validation
- **API Tests**: External service integration testing
- **Performance Tests**: Benchmarking and optimization validation

## Deployment

### 🚀 Production Deployment
```bash
# Production setup
cp .env.template .env.production
# Configure production environment variables

# Build and deploy
docker build -t abc-application .
docker run -d --env-file .env.production abc-application

# Or use the deployment script
./scripts/deploy.sh
```

### 📊 Monitoring & Observability
- **API Health Dashboard**: Real-time monitoring of all external APIs
- **Discord Integration**: Workflow monitoring and alert notifications
- **Langfuse Tracing**: Comprehensive request tracing and analytics
- **Performance Metrics**: System performance and latency monitoring

### 🔒 Security Considerations
- **API Key Management**: Secure credential storage and rotation
- **Input Validation**: Comprehensive sanitization and circuit breakers
- **Audit Logging**: Complete decision traceability for compliance
- **Access Control**: Role-based permissions for system operations

## Contributing

### 🤝 Development Guidelines
1. **Code Style**: Follow PEP 8 with Black formatting
2. **Testing**: Write tests for all new functionality
3. **Documentation**: Update docs for any API changes
4. **Commits**: Use conventional commit messages

### 📋 Pull Request Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 🐛 Issue Reporting
- Use GitHub Issues for bug reports and feature requests
- Include detailed reproduction steps for bugs
- Specify your environment (OS, Python version, etc.)

### 📖 Documentation Standards
- Keep README.md updated with any API changes
- Add docstrings to all public functions and classes
- Update architecture docs for significant changes
- Include examples for new features

---

## Historical Simulations & Backtesting

ABC Application supports comprehensive historical portfolio simulations using multiple data sources:

### Data Sources
- **yfinance**: Free historical data (default for quick testing)
- **IBKR Historical Data**: Professional-grade market data via IBKR API (recommended for production backtesting)
- **MarketDataApp**: Premium institutional-grade market data with quotes and historical data endpoints

### Running Simulations
```bash
# Standard simulation (yfinance data)
python simulations/comprehensive_historical_simulation.py

# IBKR-powered simulation (professional data)
python simulations/comprehensive_ibkr_simulation.py

# Compare data sources
python optimizations/compare_data_sources.py
```

---

*ABC-Application: Building the future of AI-powered quantitative portfolio management. Every decision logged, every trade audited, every return maximized.*

