# Unified Workflow Orchestrator Agent

## Overview
The UnifiedWorkflowOrchestrator is the central coordination system that consolidates Live Workflow, Continuous Trading, and 24/6 operations into a single bullet-proof workflow system with multiple operating modes.

## Capabilities

### Workflow Orchestration
- **Multi-Mode Operation**: ANALYSIS, EXECUTION, HYBRID, and BACKTEST modes
- **Agent Coordination**: Manages 8-agent collaborative reasoning processes
- **Scheduled Operations**: Market-aware scheduling with health monitoring

### Trade Proposal Ranking
- **Confidence-Based Ranking**: Primary sort by confidence score (descending)
- **Expected Return Tiebreaker**: Secondary sort by expected return for equal confidence
- **Real-time Processing**: Live ranking during analysis cycles

### Discord Integration
- **Alerts Channel**: Trade execution and system status alerts
- **Ranked Trades Channel**: Dedicated channel for ranked trade proposals
- **Real-time Updates**: Live workflow progress and decision updates

### Risk Management Integration
- **Position Sizing**: Automated risk-based position calculations
- **Circuit Breakers**: Emergency stop mechanisms for adverse conditions
- **Health Monitoring**: API and system health continuous monitoring

## Operating Modes

### ANALYSIS Mode
Full collaborative analysis with human intervention capabilities.

### EXECUTION Mode
Automated trading execution with minimal human oversight.

### HYBRID Mode
Combined analysis and automated execution with human monitoring.

### BACKTEST Mode
Historical simulation and validation of strategies.

## Key Methods

### rank_trade_proposals(proposals)
Ranks trade proposals by confidence and expected return.

**Parameters:**
- `proposals`: List of trade proposal dictionaries

**Returns:**
- Ranked list of proposals

### send_trade_alert(message, alert_type)
Sends alerts to the Discord alerts channel.

### send_ranked_trade_info(proposals_info, info_type)
Sends ranked trade information to the dedicated ranked trades channel.

### _extract_trade_alert_info(response_data)
Extracts and formats trade alert information from agent responses.

## Integration Points

### Agent Communication
- A2A Protocol for inter-agent communication
- Strategy Agent for trade proposal generation
- Risk Agent for position sizing and validation
- Execution Agent for trade implementation

### External Systems
- Interactive Brokers (IBKR) for trade execution
- Discord for real-time monitoring and alerts
- Redis for caching and memory management
- Langfuse for LLM operation tracking

## Configuration

### Scheduling
- Analysis cycles: 5-minute intervals
- Execution cycles: Market hours only (9:30 AM - 4:00 PM ET)
- Health checks: 1-minute intervals

### Risk Parameters
- Maximum drawdown limits
- Position sizing algorithms
- Volatility-based adjustments

## Monitoring and Health Checks

### System Health
- API connectivity monitoring
- Agent responsiveness checks
- Memory usage tracking
- Error rate monitoring

### Performance Metrics
- Execution cycle completion times
- Trade success rates
- System uptime statistics
- Alert delivery confirmation