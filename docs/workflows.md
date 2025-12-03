# Workflows Documentation

## Overview

The AI Portfolio Manager system uses structured workflows to orchestrate collaborative reasoning among multiple AI agents. These workflows ensure systematic analysis, decision-making, and learning in trading scenarios. The core of the system is built around LangGraph's StateGraph, which manages the flow between agents while maintaining a shared state.

All workflows are managed through the A2AProtocol class in `src/utils/a2a_protocol.py`, which handles agent registration, message passing, and Discord integration for monitoring and control. The `live_workflow_orchestrator.py` handles scheduling and continuous operation.

## Main Workflow: AI Trading Workflow

This is the primary workflow that runs the complete 8-step collaborative reasoning process for trading decisions. It follows a sequential path with conditional branching, ensuring thorough analysis from macroeconomic overview to post-execution learning.

### Workflow Steps
1. **🌍 Macro Agent**: Analyzes macroeconomic conditions and identifies top sectors for micro analysis. Outputs selected sectors and market regime.
   
2. **📊 Data Agent**: Collects and validates market data for the selected sectors/symbols. Focuses on multi-source data integration.

3. **🔍 Strategy Agent**: Generates trading strategies based on data and macro context. Incorporates learning directives from previous cycles.

4. **⚖️ Risk Agent**: Assesses risks and approves/rejects strategies. Uses conditional branching: approved strategies proceed to execution; rejected ones go directly to reflection.

5. **🚀 Execution Agent**: Plans and validates trade execution for approved strategies.

6. **🧠 Reflection Agent**: Provides oversight, potentially vetoing decisions. Analyzes the entire process for coherence.

7. **📚 Learning Agent**: Analyzes performance, generates improvements, and creates directives for future cycles.

### Key Features
- **State Management**: Uses `AgentState` (Pydantic model) to track data across steps.
- **Conditional Logic**: Risk approval determines if execution occurs.
- **Discord Monitoring**: Each step logs structured updates to Discord with emojis, colors, and fields for readability.
- **Summary**: A comprehensive summary is logged upon completion, showing outcomes from all steps.

### How to Run
- Via Discord: Use `!start_workflow` in the monitoring channel.
- Programmatically: Call `a2a.run_orchestration(initial_data)` where `initial_data` is a dict with symbols and other parameters.
- Scheduled: Runs automatically via APScheduler in `live_workflow_orchestrator.py` at predefined times (see Scheduled Workflows below).

## Scheduled Workflows

The system supports 24/6 continuous operation with market-aware scheduling, aligned with NYSE trading hours (Eastern Time - ET). Schedules are managed by APScheduler in `live_workflow_orchestrator.py` and use the NYSE calendar for trading day validation. Workflows only run on valid trading days (excluding weekends and holidays).

### Trading Day Schedule (Mon-Fri, ET)
- **5:30 AM ET**: Early Monday Prep - Extra early Monday market regime assessment (only on Mondays).
- **6:00 AM ET**: Pre-Market Prep - Early pre-market analysis and data collection.
- **7:30 AM ET**: Market Open Prep - Final pre-open analysis and position setup (2+ hours before 9:30 AM open).
- **12:00 PM ET**: Midday Check - Intraday performance and adjustment analysis.
- **4:30 PM ET**: Market Close Review - End-of-day performance review.
- **5:00 PM ET**: Post-Market Review - Post-market analysis and next-day preparation.

### Emergency Triggers
- **VIX > 30**: High volatility analysis.
- **Market Move > ±2%**: Significant market movement response.
- **System Health Issues**: Automated health checks and alerts.

These scheduled workflows leverage the main AI Trading Workflow phases but can be customized for specific times (e.g., pre-market focuses on readiness assessment).

## Other Workflow Types

### Analysis-Only Workflow
- Triggered via `!analyze <query>` in Discord.
- Routes the query to the most relevant agent (e.g., "macro" for market regime questions).
- Returns focused analysis without running the full trading workflow.
- Useful for quick insights or human-directed research.

### Custom Workflows
- The system supports extension through LangGraph:
  - Add new nodes/edges in `_build_graph()` for custom flows.
  - Example: Add parallel branches for concurrent data collection and macro analysis.
- Currently, no predefined custom workflows exist, but the framework allows easy addition (see `add_langgraph_edge()` stub).

## Workflow Control Commands (Discord)
- `!start_workflow`: Initiate the main AI Trading Workflow.
- `!start_premarket_analysis`: Manually trigger premarket analysis workflow (early market preparation).
- `!start_market_open_execution`: Fast-track execution at market open (leverages premarket analysis).
- `!start_trade_monitoring`: Start trade monitoring for active positions.
- `!pause_workflow`: Pause an active workflow.
- `!resume_workflow`: Resume a paused workflow.
- `!stop_workflow`: Stop the current workflow.
- `!workflow_status`: Check current workflow status.
- `!status`: Get system health status.
- `!analyze <query>`: Run analysis-only workflow.

## Monitoring and Logging
- All workflows log to a configured Discord channel with structured embeds.
- Logs include status emojis (✅ for success, ❌ for failure), key metrics, and progress indicators (e.g., "Step 3/8").
- Final summary provides a complete overview of outcomes.
- **Trade Alerts**: Dedicated channel for real-time trade notifications (executions, risks, errors) with automatic retries and fallbacks.
- **Ranked Trade Proposals**: Dedicated channel for prioritized trade proposals, sorted by confidence and expected return.

## Extensibility
- **Adding Agents**: Use `register_agent(role, agent_instance, langchain_agent)` to integrate new agents.
- **LangChain Integration**: Agents can wrap LangChain executors for enhanced capabilities.
- **Error Handling**: Built-in retries and conditional routing for resilience.

For implementation details, see `src/utils/a2a_protocol.py` and `src/agents/live_workflow_orchestrator.py`. For Discord integration, refer to `docs/discord-agent-integration.md`. For 24/6 setup, see `docs/IMPLEMENTATION/24_6_CONTINUOUS_OPERATION.md`.

## Consensus Workflow Polling

The Consensus Workflow Polling system enables agents to request and achieve consensus on critical decisions, particularly for risk and strategy assessments. This ensures collaborative decision-making with proper oversight and Discord visibility.

### How It Works
1. **Agent Request**: Risk or Strategy agents can request consensus via `orchestrator.request_consensus(question, target_agents)`
2. **Poll Creation**: Orchestrator creates a poll with configurable timeout and targets specific agents
3. **Voting Phase**: Target agents are polled asynchronously until consensus or timeout
4. **Consensus Check**: Majority vote (>50%) with sufficient confidence (>60%) achieves consensus
5. **Discord Updates**: Real-time status updates and final results posted to Discord
6. **Persistence**: All polls saved to JSON with metrics tracking

### Key Features
- **Agent-Initiated**: Risk/Strategy agents can trigger polls for position sizing, trade approvals, etc.
- **Configurable**: Timeout, confidence thresholds, and polling intervals via `config/consensus_config.yaml`
- **Discord Integration**: Slash commands (`/consensus_status`, `/poll_consensus`) and status embeds
- **Metrics & Alerts**: Success rates, response times, and automated alerts for consensus events
- **Persistence**: Survives system restarts with full poll state recovery

### Usage Examples
```python
# Agent requesting consensus
poll_id = await orchestrator.request_consensus(
    "Is this 5% position size appropriate for current volatility?",
    "risk_agent",
    ["strategy_agent", "execution_agent"]
)

# Discord slash commands
/consensus_status    # View active/completed polls
/poll_consensus "Should we exit this position?" risk_agent strategy_agent
```

### Configuration
See `config/consensus_config.yaml` for polling intervals, timeouts, agent permissions, and alert settings.

### Implementation
- **Core**: `src/workflows/consensus_poller.py` - Main polling logic and state management
- **Integration**: `src/agents/live_workflow_orchestrator.py` - Discord commands and agent requests
- **Tests**: `unit-tests/test_consensus_poller.py` and `integration-tests/test_consensus_integration.py`
