# 🤖 Live Iterative Reasoning Workflow - Real-Time Orchestration

This directory contains tools to implement the **22-agent collaborative reasoning framework** through **live, interactive Discord orchestration**. Watch the workflow unfold in real-time and intervene with questions during the process!

## 🎯 What's Different Now

**BEFORE:** Manual workflows where you send commands one-by-one
**NOW:** Live orchestrator that automatically runs the complete workflow while you watch and participate in Discord!

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `live_workflow_orchestrator.py` | **NEW!** Live Discord orchestrator with real-time interaction |
| `start_live_workflow.py` | **NEW!** Easy launcher for the live orchestrator |
| `iterative_reasoning_workflow.py` | Background automated workflow (legacy) |
| `manual_workflow_guide.md` | Manual workflow reference |
| `quick_workflow_test.py` | Test individual phases |
| `workflow_status_tracker.py` | Track workflow progress |
| `WORKFLOW_README.md` | This documentation |

## 🚀 Quick Start - Live Orchestration

### Step 1: Start the Live Orchestrator
```bash
python start_live_workflow.py
```
Or directly:
```bash
python live_workflow_orchestrator.py
```

### Step 2: Go to Discord and Control the Workflow
Once the orchestrator connects, use these commands in your Discord server:

```
!start_workflow   - Begin the complete iterative reasoning process
!pause_workflow   - Pause mid-workflow (to ask questions)
!resume_workflow  - Continue after pausing
!stop_workflow    - End workflow early
!workflow_status  - Check current progress
```

### Step 3: Watch and Participate!
- **Watch** the workflow unfold automatically in real-time
- **Ask questions** anytime: "Why did the risk agent say that?" or "Can you explain this strategy?"
- **Intervene** with concerns or additional context
- **See responses** from all agents as they happen

## 🎭 How Live Orchestration Works

### 🤖 Automatic Execution
- Runs through all 10 workflow phases automatically
- Sends commands to agents with appropriate timing
- Collects and logs all responses

### 👤 Human Participation
- **Ask questions** during any phase - the orchestrator acknowledges and pauses briefly
- **Get clarification** from agents in real-time
- **Provide additional context** that agents can incorporate
- **Challenge assumptions** and see agent responses

### 📊 Live Monitoring
- Real-time status updates
- Response counting per phase
- Intervention logging
- Progress tracking

## 📋 Workflow Phases (Fully Automated)

| Phase | Duration | What Happens |
|-------|----------|--------------|
| **Macro Foundation** | 15s | Market regime assessment & opportunity identification |
| **Intelligence Gathering** | 48s | Multi-source data collection & validation |
| **Strategy Development** | 48s | Collaborative strategy formation |
| **Multi-Agent Debate** | 25s | Cross-domain challenge & refinement |
| **Risk Assessment** | 36s | Probabilistic analysis & constraints |
| **Consensus Building** | 15s | Conflict mediation & agreement |
| **Execution Validation** | 20s | Practical feasibility checks |
| **Learning Integration** | 10s | Continuous improvement |
| **Executive Review** | 60s | Elevated strategic oversight |
| **Supreme Oversight** | 100s | Final audit & veto authority |

**Total Time: ~50-90 minutes** (depending on responses and interventions)

## 📊 Workflow Structure

### Phase 0: Macro Foundation (5-10 min)
- **MacroAgent** establishes market regime context
- Identifies top 5 opportunities for analysis
- Sets baseline risk parameters

### Iteration 1: Comprehensive Deliberation (20-30 min)
7 phases with all agents participating:
1. **Intelligence Gathering** - Multi-source data collection
2. **Strategy Development** - Collaborative strategy formation
3. **Multi-Agent Debate** - Cross-domain challenge and refinement
4. **Risk Assessment** - Probabilistic analysis and constraints
5. **Consensus Building** - Conflict mediation and agreement
6. **Execution Validation** - Practical feasibility checks
7. **Learning Integration** - Continuous improvement

### Iteration 2: Executive Oversight (15-20 min)
Executive-level review with elevated risk sensitivity and strategic focus.

### Supreme Oversight (10-15 min)
**ReflectionAgent** conducts final audit with veto authority and crisis detection.

## 🛠️ Tools Usage

### Status Tracker
```bash
python workflow_status_tracker.py
```
- Track workflow progress
- Get recommended next commands
- Record insights, decisions, and warnings
- Monitor completion status

### Key Features
- **Progress Tracking**: Visual status of all 10 workflow phases
- **Command Recommendations**: Get next suggested commands based on current phase
- **Insight Capture**: Record key learnings and decisions
- **Warning System**: Track concerns and risk signals
- **Completion Metrics**: Duration tracking and summary statistics

## 📈 Workflow Metrics

The system tracks:
- Commands sent per phase
- Agent responses received
- Key insights captured
- Decisions made
- Warnings/concerns raised
- Total workflow duration

## 🎯 Best Practices

### 1. Sequential Execution
- Wait for agent responses before proceeding
- Read all responses carefully
- Note disagreements and concerns

### 2. Cross-Agent Validation
- Compare data agent consistency
- Respect risk agent challenges
- Consider execution agent practicality

### 3. Reflection Agent Authority
- **Veto Power**: Reflection agent can override any decision
- **Extra Iterations**: Can mandate additional analysis if concerning patterns emerge
- **Crisis Detection**: Monitors for "canary in the coal mine" indicators

### 4. Learning Integration
- Teach agents from each workflow: `!l learn [insight]`
- Build institutional knowledge over time
- Improve future workflows based on past outcomes

## 🔧 Customization

### Modifying Commands
Edit the command sequences in:
- `iterative_reasoning_workflow.py` - Automated workflow
- `manual_workflow_guide.md` - Manual guide
- `workflow_status_tracker.py` - Recommended commands

### Adding Phases
Extend the workflow by adding phases to the status tracker and implementing corresponding command sequences.

### Custom Debate Topics
Modify debate commands for specific scenarios:
```
!m debate "Should we overweight [sector] given [conditions]?" [agents]
```

## 📊 Monitoring & Validation

### Agent Status Checks
```
!m status  !d status  !s status  !r status
!ref status  !e status  !l status
```

### Memory Review
```
!m memory  !s memory  !r memory  !ref memory
```

### Learning Validation
```
!l analyze Show me what you've learned from recent workflows
```

## 🚨 Troubleshooting

### Agents Not Responding
1. Check bot status: Use status commands above
2. Verify command syntax: Must start with agent prefix (!m, !d, etc.)
3. Wait between commands: Allow 5-10 seconds processing time

### Incomplete Analysis
1. Request clarification: `!ref analyze Can you elaborate on [point]?`
2. Additional analysis: `![agent] analyze [follow-up question]`
3. Deeper discussion: `!m debate "[issue]" [relevant agents]`

### Disagreement with Conclusions
1. Present counter-arguments: `!ref analyze I disagree because [reason]`
2. Request re-evaluation: `!ref analyze Please reconsider [concern]`
3. Challenge assumptions: `!r analyze What if [alternative scenario]?`

## 📈 Expected Outcomes

### Quality Improvements
- **Institutional-grade decisions** through collaborative reasoning
- **Risk mitigation** via multi-agent validation
- **Crisis prevention** through reflection agent oversight
- **Continuous learning** and process improvement

### Performance Metrics
- **Decision confidence** increases with each iteration
- **Risk awareness** amplifies in executive oversight
- **Implementation feasibility** validated before execution
- **Historical patterns** incorporated for better predictions

## 🎉 Success Metrics

A successful workflow completion shows:
- ✅ All 10 phases completed
- 📊 22+ agent responses collected
- 🎯 Clear decisions with risk assessments
- 🧠 Learning insights captured
- ⚠️ Warnings appropriately addressed
- ⏱️ Reasonable completion time (50-90 minutes)

## 🔄 Continuous Improvement

### Post-Workflow Analysis
1. Review workflow results in `workflow_results.json`
2. Identify bottlenecks and improvement areas
3. Teach agents new patterns: `!l learn [pattern]`
4. Refine command sequences for future workflows

### System Evolution
- Add new agents as capabilities expand
- Refine debate topics based on successful patterns
- Enhance risk detection algorithms
- Improve execution validation processes

---

**This workflow transforms your Discord bot system into a sophisticated collaborative reasoning platform that replicates institutional investment committee processes with AI precision and scalability!** 🚀

For questions or issues, check the individual file documentation or run the test scripts to validate functionality.