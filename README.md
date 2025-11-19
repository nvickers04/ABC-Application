# ABC-Application: AI Portfolio Manager

## Project Overview
ABC-Application is a multi-agent AI system for quantitative portfolio management, integrating Grok agents (via xAI API) with Interactive Brokers (IBKR) for real-time data ingestion, analysis, and trade execution. The system operates on a macro-to-micro hierarchy: high-level market scanning informs strategy proposals, risk assessments, and precise executions. Key goals: Achieve 10-20% monthly returns with <5% max drawdown through probability-of-profit (POP) evaluations, stochastic batching, and reflection loops for continuous learning.

### **Core Innovation: AI Reasoning Through Agent Collaboration**
The fundamental breakthrough of ABC-Application is its **22-agent collaborative reasoning architecture with two-iteration framework and supreme oversight**. Rather than relying on a single AI model for decision-making, the system creates a sophisticated debate and deliberation environment where specialized AI agents collaboratively reason through complex investment decisions. This multi-agent approach leverages collective intelligence to produce more robust, well-reasoned investment strategies.

**Two-Iteration Framework**: The MacroAgent first establishes market context and identifies top opportunities, then the system conducts comprehensive multi-agent deliberation (all 22 agents including subagents) followed by executive-level strategic oversight (main 8 agents only), ensuring both analytical depth and strategic judgment.

**Supreme Oversight**: The ReflectionAgent serves as the system's final arbiter with veto authority and the power to trigger additional iterations based on crisis indicators, preventing catastrophic decisions.

**Why 22 Agents?** Each agent represents a specialized domain of financial expertise (data analysis, strategy generation, risk management, execution, learning, reflection, etc.) working together in orchestrated reasoning loops. The agents debate proposals, challenge assumptions, validate conclusions, and reach consensus through structured deliberation - mimicking institutional investment committee processes but with AI precision and speed.

**Proven Results:** The system achieved profitability even without LLM reasoning, demonstrating the power of the agent collaboration framework. **Now with advanced grok-4-fast-reasoning model integration, returns are projected to be off the charts** as each agent's reasoning capabilities are exponentially enhanced.

- **Core Philosophy**: Traceable, ethical wealth management for legacy building—every decision logged with reasoning for audits (e.g., SEC compliance). Agents collaborate via Agent-to-Agent (A2A) protocols, emphasizing profitability alignment and iterative improvements based on execution vs. backtest deltas.
- **Tech Stack**: Python asyncio for asynchronous processing, Langchain with Grok API for LLM reasoning, collaborative memory systems, YAML configurations, and IBKR API integration. Code is fully implemented with enhanced subagents featuring AI-driven analysis.

## Documentation

For comprehensive documentation of the ABC-Application system, see the [`docs/`](./docs/) directory:

- **[System Overview](./docs/README.md)**: Complete documentation navigation and quick start guide
- **[Architecture](./docs/ARCHITECTURE.md)**: System design, components, and data flow
- **[Agent Inventory](./docs/AGENTS/index.md)**: All 22 agents with capabilities and coordination
- **[Frameworks](./docs/FRAMEWORKS/)**: Macro-micro analysis, LangChain integration, A2A protocols
- **[Implementation](./docs/IMPLEMENTATION/)**: Setup, configuration, testing, and deployment
- **[Reference](./docs/REFERENCE/)**: API monitoring, health checks, and changelog

## Key Features
- **Multi-Agent Architecture**: Specialized agents for data collection, strategy development, risk management, execution, learning, and reflection.
- **LLM Integration**: Grok API-powered reasoning for market analysis, strategy optimization, and risk assessment.
- **Collaborative Memory**: Shared memory coordinator enabling cross-agent insight sharing and temporary research sessions.
- **Enhanced Analyzers**: OptionsStrategyAnalyzer, FlowStrategyAnalyzer, and MLStrategyAnalyzer with advanced capabilities including Greeks calculations, institutional flow analysis, and predictive modeling.
- **IBKR Integration**: Unified interface for IBKR connectivity with LangChain tool integration for seamless trading operations
- **IBKR Historical Data**: Professional-grade historical market data for accurate backtesting and simulation, with direct integration to IBKR's comprehensive data feeds.
- **API Health Monitoring**: Automated monitoring system tracking response times, success rates, and circuit breaker status for all data APIs (MarketDataApp, Kalshi, yFinance, NewsAPI, FRED, Currents, Twitter, Whale Wisdom, Grok) with real-time dashboard and alerts.
- **MarketDataApp Integration**: Premium institutional-grade market data with 7 data endpoints (quotes, trades, orderbook, options, darkpool, microstructure, flow) now fully activated with LLM-powered data exploration and circuit breaker protection.
- **Modular Tools Architecture**: Organized utility functions into specialized modules (validation, financial, news, market data, backtesting, social media, agent tools) for better maintainability and code organization.
- **Robust Error Handling**: Comprehensive input validation, circuit breakers, and graceful degradation with backup data sources and redundant systems.
- **A2A Protocols**: Agent-to-agent communication for coordinated decision-making.
- **Backtesting & Simulation**: Integrated backtesting framework with stochastic batching and reflection loops, now enhanced with IBKR data sources.

## Quick Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment: Ensure Python 3.11+ and activate virtual environment if needed.
3. Run the system: `python src/main.py` (starts agent orchestration loop with automatic health monitoring)
- Monitor API health: `python tools/api_health_dashboard.py` (interactive dashboard for real-time API status)
- Test agents individually: `python src/agents/data.py` or run `python integration-tests/comprehensive_test.py` for subagent validation

## File Structure
- `README.md`: This file.
- `requirements.txt`: Python dependencies.
- `src/`: Core implementation code.
  - `main.py`: Main application entry point.
  - `agents/`: Agent implementations and base classes.
  - `utils/`: Utility modules and tools.
    - `tools.py`: Main tools aggregator (imports from specialized modules).
    - `validation.py`: Input validation, sanitization, and circuit breakers.
    - `financial_tools.py`: Stock data, sentiment analysis, risk calculation.
    - `news_tools.py`: News APIs and economic data tools.
    - `market_data_tools.py`: Market data APIs and WebSocket connections.
    - `backtesting_tools.py`: Performance metrics and backtesting frameworks.
    - `social_media_tools.py`: Social media sentiment and monitoring.
    - `agent_tools.py`: Agent coordination and collaborative decision tools.
- `agents/`: Agent documentation and prompts (.md files).
- `config/`: YAML configurations (e.g., risk-constraints.yaml).
- `integrations/`: IBKR connector, NautilusIBKRBridge, and external integrations.
- `simulations/`: All simulation and backtesting scripts.
- `integration-tests/`: Integration and system testing scripts.
- `unit-tests/`: Unit tests for individual components.
- `tools/`: Operational tools and utilities.
- `optimizations/`: Performance optimization and analysis tools.
- `setup/`: Installation files and setup utilities.
- `results/`: JSON output files from simulations and tests.
- `docs/`: Documentation and configuration templates.
- `examples/`: Simulation and demo scripts.
- `data/`: Data files, logs, cache, and models.

## Development Resources

### 📋 **Development Guidelines**
- **[AI Development Instructions](./AI_DEVELOPMENT_INSTRUCTIONS.md)**: Comprehensive guide for getting better output from Grok Code Fast 1
- **[File Organization Guide](./FILE_ORGANIZATION_GUIDE.md)**: Standards for file placement and project structure
- **[Documentation Coherence Guide](./DOCUMENTATION_COHERENCE_GUIDE.md)**: Maintaining alignment between .md files and code

### 🗂️ **Project Organization**
- **Source Code**: `src/` directory (agents, utils, workflows)
- **Documentation**: `docs/` directory (architecture, agents, frameworks, implementation)
- **Configuration**: `config/` directory (YAML configs, environment settings)
- **Tests**: `unit-tests/` and `integration-tests/` directories
- **Tools**: `tools/` directory (utilities, monitoring, deployment scripts)
- **Simulations**: `simulations/` directory (backtesting and analysis scripts)

### 📚 **Key Documentation Files**
- **[System Architecture](./docs/architecture.md)**: Complete system design and data flows
- **[Agent Framework](./docs/ai-reasoning-agent-collaboration.md)**: Multi-agent reasoning architecture
- **[Macro-Micro Framework](./docs/macro-micro-analysis-framework.md)**: Analysis methodology
- **[Production Checklist](./docs/production_readiness_checklist.md)**: Deployment and security requirements

## Current Status
- **Implemented**: Full agent framework with LLM integration, collaborative memory systems, and enhanced strategy subagents. IBKR integration with LangChain tools for unified trading interface. IBKR historical data provider for professional-grade backtesting. API health monitoring system with automated checks, circuit breaker integration, and real-time dashboard. Modular tools architecture with specialized modules for better code organization and maintainability. Comprehensive error handling with input validation, circuit breakers, and graceful degradation. **MarketDataApp premium data source fully activated** with LLM-powered exploration of institutional-grade market data.
- **Testing**: Comprehensive test suite for subagents, memory systems, bridge integration, API health monitoring, and historical data providers. System robustness validated through comprehensive audit and fixes.
- **Integration**: A2A protocols, shared memory coordinator, base agent inheritance, unified IBKR trading interface, automated API health monitoring, professional historical data feeds, and modular utility architecture.

## Recent Improvements (v2.1)
- **MarketDataApp Activation**: Fully activated premium institutional data source with 7 data endpoints (quotes, trades, orderbook, options, darkpool, microstructure, flow) and LLM-powered intelligent exploration capabilities.
- **Code Modularization**: Split monolithic 296KB `tools.py` file into 7 specialized modules for better maintainability
- **System Robustness**: Implemented comprehensive error handling, circuit breakers, and graceful degradation
- **Input Validation**: Enhanced data sanitization and validation with HTML/script removal and anomaly detection
- **Dependency Management**: Updated requirements.txt and resolved missing package installations
- **Redis Integration**: Configured Redis server for memory persistence and caching
- **Audit Compliance**: Addressed all identified issues from comprehensive project audit

## Next Steps & Roadmap
- **Short-Term**: Complete data subagent implementations, integrate IBKR API for live trading.
- **Medium-Term**: Add real-time market data feeds, expand backtesting capabilities, implement live trading safeguards.
- **Long-Term**: Achieve 10-20% monthly ROI with <5% drawdown; full audit logging and SEC compliance features.

## Historical Simulations & Backtesting

ABC Application supports comprehensive historical portfolio simulations using multiple data sources:

### Data Sources
- **yfinance**: Free historical data (default for quick testing)
- **IBKR Historical Data**: Professional-grade market data via IBKR API (recommended for production backtesting)
- **MarketDataApp**: Premium institutional-grade real-time market data with 7 data endpoints (quotes, trades, orderbook, options, darkpool, microstructure, flow)

### Running Simulations

#### Standard Simulation (yfinance data)
```bash
python simulations/comprehensive_historical_simulation.py
```

#### IBKR-Powered Simulation (Professional data)
```bash
python simulations/comprehensive_ibkr_simulation.py
```

#### Compare Data Sources
```bash
python optimizations/compare_data_sources.py
```

### IBKR Data Benefits
- **Professional Quality**: Institutional-grade bar data with enhanced accuracy
- **Comprehensive Coverage**: More historical data points than free sources
- **Real-time Access**: Direct integration with live trading platform
- **Enhanced Features**: Support for options, futures, and advanced order types
- **Regulatory Compliance**: Same data used by professional traders

### Simulation Features
- Multi-asset portfolio backtesting
- Rebalancing strategies (daily/weekly/monthly/quarterly)
- Transaction costs and slippage modeling
- Risk metrics (Sharpe ratio, max drawdown, VaR)
- Performance analytics and reporting

For contributions or issues, ensure all changes tie back to profitability and traceability (e.g., update CHANGELOG.md with diffs). This project honors long-term wealth stewardship—do your absolute best.