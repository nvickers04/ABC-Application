# Master Index for ABC Application AI Portfolio Manager Project

## Quick Start
- **Overview**: Multi-agent AI trading system with Grok/IBKR. Start with `README.md`.
- **Setup**: Install deps (`pip install -r requirements.txt`), run `python src/main.py`.
- **Key Files**: `src/main.py` (orchestration), `agents/` (agent prompts/notes), `config/` (YAML settings).
- **AI Guidance**: See `.github/copilot-instructions.md`.

## How to Navigate This Index (Simple Instructions for Any LLM)
This index is easy to follow. It lists all files in the project. Each section has a table with files. To find something:
1. Scroll to the section name (e.g., "Agents" for agent files).
2. Look at the table: It shows File Path, Description, and more.
3. For Agents: Each agent has its own sub-section with a table. The table lists the notes file (e.g., agents/data-agent-notes.md) and prompt file (e.g., agents/data-agent-prompt.md).
4. Click links if on GitHub (relative paths work as hyperlinks).
5. Search the page: Use Ctrl+F to find words like "data-agent-prompt.md" or "Strategy Agent".
6. If lost: Start at "Root Files", then "Agents" for specifics.
This is made simple: No fancy stuff, just tables and clear paths. If a file is missing, check CHANGELOG.md.

## Overview
This file is a simple list of all project documents. Each entry includes:
- **File Path**: Where the file is (e.g., agents/data-agent-complete.md).
- **Description**: What the file is about.
- **Dependencies/Cross-Refs**: Other files it links to.
- **Last Updated**: When it was last changed.
- **Critique/Notes**: Quick tips.

Sections:
- Root Files: Big picture files.
- Core Utilities: Shared tools and logs.
- Documentation: Plans and examples.
- Configuration: Settings files.
- Agents: Files for each agent, including subagents.

This index helps find things fast. Supports traceable audits for funding. Last Reviewed: 2025-11-04.

## Root Files
| File Path | Description | Dependencies/Cross-Refs | Last Updated | Critique/Notes |
|-----------|-------------|--------------------------|--------------|----------------|
| README.md | Main project info: What it is, goals, file list, next steps. Start here. | Links to core/, docs/, config/, agents. | 2025-11-04 | Updated with current implementation status and features. Ties to making money (10-20% monthly returns). |
| CHANGELOG.md | List of changes with dates and diffs. | Links to all files. | 2025-11-04 | Updated with recent subagent implementations and LLM integration. Helps track updates. |
| code-skeleton.md | Plan for code: Classes, functions, implementation details. | Links to base_prompt.txt, agent prompts, YAMLs. | 2025-11-04 | Updated to reflect implemented code structure. Ties to agents. |

## Core Utilities
| File Path | Description | Dependencies/Cross-Refs | Last Updated | Critique/Notes |
|-----------|-------------|--------------------------|--------------|----------------|
| sim-training-log.txt | Logs of sim tests: Dates, results, ROI ties. | Links to risk-constraints.yaml, learning-agent-complete.md. | 2025-11-04 | Updated with recent simulation results. Helps learn without real trades. |
| portfolio-dashboard.txt | Plan for dashboard: Costs, metrics, alerts. | Links to agent-behavior-guidelines.md, risk-constraints.yaml. | 2025-11-04 | Updated with current metrics and alerts. Ties to saving alpha (0.2-0.5%). |
| learning-data-changelog.md | Weekly changes for learning: Adjustments, metrics, diffs. | Links to risk-agent-complete.md, learning-agent-complete.md. | 2025-11-04 | Updated with recent learning data changes. Ties to better strategies. |

## Documentation
| File Path | Description | Dependencies/Cross-Refs | Last Updated | Critique/Notes |
|-----------|-------------|--------------------------|--------------|----------------|
| docs/architecture.md | How the system works: Flows, loops, agents. | Links to agents/a2a-protocol.txt, agents/agent-behavior-guidelines.md. | 2025-11-04 | Updated with current agent architecture and subagents. Ties to risk control. |
| docs/resource-mapping-and-evaluation.md | List of tools/resources: Ranks, tables, fits to agents. | Links to awesome-quant, config/langchain-integration.md. | 2025-11-04 | Updated with latest tools and evaluations. Helps pick best tools. |
| docs/closed-loop-example.txt | Examples of cycles: Bearish/low-vol tests. | Links to docs/architecture.md, config/profitability-targets.yaml. | 2025-11-04 | Updated with new examples. Ties to alpha lifts (+3%). |
| config/profit-projections.md | ROI plans: Sims and real examples. | Links to config/profitability-targets.yaml, sim-training-log.txt. | 2025-11-04 | Updated with current projections. Ties to 20% goals. |
| agents/memory-management.md | How memory works: Types, tools, xAI links. | Links to config/langchain-integration.md, agents/a2a-protocol.txt. | 2025-11-04 | Updated with collaborative memory implementation. Ties to smart agents. |
| config/langchain-integration.md | All about Langchain: Agents, tools, memory. | Links to base_prompt.txt, agent prompts. | 2025-11-04 | Updated with current integration details. Ties to auto agents. |
| core/backtesting/outline.txt | Backtesting plan: Tools, uses in agents. | Links to docs/architecture.md. | 2025-11-04 | Updated with implementation notes. Ties to safe learning. |

## Configuration
| File Path | Description | Dependencies/Cross-Refs | Last Updated | Critique/Notes |
|-----------|-------------|--------------------------|--------------|----------------|
| config/risk-constraints.yaml | Limits for risk: Sizes, drawdowns, pyramiding. | Links to all agents, portfolio-dashboard.txt. | 2025-11-04 | Updated with current risk parameters. Ties to <5% loss. |
| config/profitability-targets.yaml | Goals for money: Targets, bonuses. | Links to config/risk-constraints.yaml, config/profit-projections.md. | 2025-11-04 | Updated with current targets. Ties to 20% monthly. |
| config/ibkr-integration.txt | How to link IBKR: Ideas, tools. | Links to agents/execution-agent-complete.md. | 2025-11-04 | Updated with integration details. Ties to making money. |

## Agents
## Agents
### Data Agent (Go here for Data notes and prompt)
| File Path | Description | Dependencies/Cross-Refs | Last Updated | Critique/Notes |
|-----------|-------------|--------------------------|--------------|----------------|
| agents/data-agent-complete.md | Complete Data Agent documentation: Sources, processing, subagents. | Links to docs/resource-mapping-and-evaluation.md, subagent docs. | 2025-11-04 | Comprehensive guide for data collection and processing. Ties to good data. |
| agents/marketdatasub.md | Market Data Subagent: Real-time and historical market data handling. | Links to agents/data-agent-complete.md. | 2025-11-04 | Details market data integration. |
| agents/economicdatasub.md | Economic Data Subagent: Economic indicators and analysis. | Links to agents/data-agent-complete.md. | 2025-11-04 | Economic data processing and insights. |
| agents/sentimentsub.md | Sentiment Data Subagent: Market sentiment analysis. | Links to agents/data-agent-complete.md. | 2025-11-04 | Sentiment data collection and interpretation. |
| agents/optionsdatasub.md | Options Data Subagent: Options market data and analytics. | Links to agents/data-agent-complete.md. | 2025-11-04 | Options-specific data handling. |

### Strategy Agent (Go here for Strategy notes and prompt)
| File Path | Description | Dependencies/Cross-Refs | Last Updated | Critique/Notes |
|-----------|-------------|--------------------------|--------------|----------------|
| agents/strategy-agent-complete.md | Complete Strategy Agent documentation: Strategy development and subagents. | Links to agents/data-agent-complete.md, agents/risk-agent-complete.md. | 2025-11-04 | Comprehensive strategy framework. Ties to more money. |
| agents/optionsstrategysub.md | Options Strategy Subagent: Options trading strategies with LLM analysis. | Links to agents/strategy-agent-complete.md. | 2025-11-04 | Advanced options strategies. |
| agents/flowstrategysub.md | Flow Strategy Subagent: Institutional flow analysis and strategies. | Links to agents/strategy-agent-complete.md. | 2025-11-04 | Flow-based trading strategies. |
| agents/mlstrategysub.md | ML Strategy Subagent: Machine learning-driven strategies. | Links to agents/strategy-agent-complete.md. | 2025-11-04 | Predictive modeling for strategies. |

### Risk Agent (Go here for Risk notes and prompt)
| File Path | Description | Dependencies/Cross-Refs | Last Updated | Critique/Notes |
|-----------|-------------|--------------------------|--------------|----------------|
| agents/risk-agent-complete.md | Complete Risk Agent documentation: Risk assessment and management. | Links to config/risk-constraints.yaml. | 2025-11-04 | Comprehensive risk management framework. Ties to safe money. |

### Reflection Agent (Go here for Reflection notes and prompt)
| File Path | Description | Dependencies/Cross-Refs | Last Updated | Critique/Notes |
|-----------|-------------|--------------------------|--------------|----------------|
| agents/reflection-agent-complete.md | Complete Reflection Agent documentation: Performance review and learning. | Links to config/profitability-targets.yaml. | 2025-11-04 | Reflection and continuous improvement framework. Ties to better learning. |

### Learning Agent (Go here for Learning notes and prompt)
| File Path | Description | Dependencies/Cross-Refs | Last Updated | Critique/Notes |
|-----------|-------------|--------------------------|--------------|----------------|
| agents/learning-agent-complete.md | Complete Learning Agent documentation: Model training and adaptation. | Links to sim-training-log.txt. | 2025-11-04 | Learning and adaptation framework. Ties to smart changes. |

### General Agent Documentation
| File Path | Description | Dependencies/Cross-Refs | Last Updated | Critique/Notes |
|-----------|-------------|--------------------------|--------------|----------------|
| agents/agent-behavior-guidlines.md | Guidelines for agent behavior and collaboration. | Links to all agent docs. | 2025-11-04 | Behavioral standards for agents. |
| agents/agent-parameters-overview.md | Overview of agent parameters and configurations. | Links to config files. | 2025-11-04 | Parameter definitions and usage. |
| agents/a2a-protocol.txt | Agent-to-Agent communication protocols. | Links to docs/architecture.md. | 2025-11-04 | A2A messaging standards. |

## Integrations
| File Path | Description | Dependencies/Cross-Refs | Last Updated | Critique/Notes |
|-----------|-------------|--------------------------|--------------|----------------|
| integrations/ibkr_connector.py | IBKR API connector for live trading and market data. | Links to src/agents/execution_tools.py. | 2025-11-04 | Core IBKR integration for trading. |
| integrations/nautilus_ibkr_bridge.py | NautilusIBKRBridge for unified IBKR integration with nautilus_trader compatibility. | Links to integrations/ibkr_connector.py, src/agents/execution_tools.py. | 2025-11-04 | Advanced trading bridge with risk management and position sizing. |
| integrations/test_nautilus_bridge.py | Test suite for NautilusIBKRBridge functionality. | Links to integrations/nautilus_ibkr_bridge.py. | 2025-11-04 | Bridge integration testing and validation. |
| integrations/agent-model-mapping.md | Mapping of agents to AI models and capabilities. | Links to agents/ directory. | 2025-11-04 | Agent-model integration specifications. |

## Critique/Insights for Master Index
Updated to reflect current project structure with implemented code, LLM integration, and comprehensive agent documentation including subagents. Ties to easy access for code build and maintenance. Updated for complete .md files and subagent organization.