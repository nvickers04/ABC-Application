# ABC Application Documentation

## Overview

This documentation provides comprehensive guidance for the ABC Application AI-driven trading system, a multi-agent framework for quantitative portfolio management integrating Grok agents with Interactive Brokers (IBKR) for real-time data ingestion, analysis, and trade execution.

## Documentation Structure

### [ARCHITECTURE.md](ARCHITECTURE.md)
System design, data flow, and technical architecture overview.

### [AGENTS/](AGENTS/)
Agent documentation and implementation guides.

- [index.md](AGENTS/index.md) - Agent overview and inventory
- [main-agents/](AGENTS/main-agents/) - Documentation for 8 main agents
- [subagents/](AGENTS/subagents/) - Documentation for 10 data and 4 strategy subagents

### [FRAMEWORKS/](FRAMEWORKS/)
Core frameworks and protocols.

- [macro-micro-analysis.md](FRAMEWORKS/macro-micro-analysis.md) - Macro-to-micro analysis framework
- [langchain-integration.md](FRAMEWORKS/langchain-integration.md) - LangChain integration details
- [a2a-protocol.md](FRAMEWORKS/a2a-protocol.md) - Agent-to-agent communication protocol

### [IMPLEMENTATION/](IMPLEMENTATION/)
Setup, configuration, and development guides.

- [setup.md](IMPLEMENTATION/setup.md) - Installation and setup instructions
- [configuration.md](IMPLEMENTATION/configuration.md) - Configuration files and parameters
- [testing.md](IMPLEMENTATION/testing.md) - Testing framework and procedures

### [REFERENCE/](REFERENCE/)
Reference materials and appendices.

- [api-health-monitoring.md](REFERENCE/api-health-monitoring.md) - API health monitoring system
- [changelog.md](REFERENCE/changelog.md) - System changelog and version history

## Quick Start

1. **Setup**: Follow [IMPLEMENTATION/setup.md](IMPLEMENTATION/setup.md) for installation
2. **Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md) for system overview
3. **Agents**: Review [AGENTS/index.md](AGENTS/index.md) for agent inventory
4. **Configuration**: Configure using [IMPLEMENTATION/configuration.md](IMPLEMENTATION/configuration.md)

## Key Features

- **22-Agent Collaborative Framework**: Specialized AI agents for data collection, strategy generation, risk management, execution, learning, and reflection
- **Macro-to-Micro Analysis**: Systematic scanning of 39+ sectors/assets followed by deep micro-level analysis
- **LangChain Integration**: Modern LLM orchestration with tool-based agent interactions
- **IBKR Integration**: Professional-grade trading interface with real-time execution
- **API Health Monitoring**: Automated monitoring and circuit breaker systems
- **Memory Systems**: Advanced collaborative memory for cross-agent intelligence sharing

## System Goals

- Achieve 10-20% monthly returns with <5% maximum drawdown
- Maintain traceable, ethical wealth management with full audit logging
- Enable continuous learning and adaptation through reflection loops

## Contact & Support

For questions or contributions, refer to the main project README.md in the repository root.

---

*Last updated: November 10, 2025*
*Documentation version: 2.0 (Clean restart)*