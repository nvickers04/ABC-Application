# Tools Folder

This folder contains operational tools and utilities for running and monitoring the ABC Application system.

## Operational Tools

### Trading Tools
- `start_unified_workflow.py` - Unified workflow orchestrator starter script
- `start_unified_workflow.bat` - Windows launcher for unified workflow
- `unified_workflow_orchestrator.py` - Core unified workflow implementation

### Monitoring Tools
- `api_health_dashboard.py` - Real-time API health monitoring dashboard

## Usage

### Start Unified Workflow
```bash
# Using the batch file (recommended)
tools/start_unified_workflow.bat --mode hybrid --symbols SPY,QQQ

# Or directly with Python
python tools/start_unified_workflow.py --mode hybrid --symbols SPY,QQQ

# Analysis-only mode
python tools/start_unified_workflow.py --mode analysis --symbols AAPL,MSFT

# Execution-only mode (automated trading)
python tools/start_unified_workflow.py --mode execution --symbols SPY
```

### Monitor API Health
```bash
python tools/api_health_dashboard.py
```

## Features

### Unified Workflow Orchestrator
- **Multiple Operating Modes**: Analysis, Execution, Hybrid
- **Market-Aware Scheduling**: Respects trading hours and extended sessions
- **Multi-Agent Collaboration**: Full 22-agent system for analysis modes
- **Automated Execution**: Simple Strategy → Risk → Execution pipeline
- **Production Scheduling**: 24/6 operation with health monitoring
- **Discord Integration**: Real-time alerts and human oversight
- **Health Monitoring**: Automatic system health checks and recovery

### API Health Monitoring
- Real-time health status for all APIs
- Circuit breaker integration
- Response time tracking
- Success rate monitoring
- Alert system for degraded services