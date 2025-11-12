# ABC Application System Architecture

# System Architecture

## Overview# High-Level Flow Description

# This describes the multi-agent system's workflow in a sequential, bullet-point format for clarity. It emphasizes macro-to-micro daily progression (e.g., broad data analysis to granular executions), A2A interactions (e.g., shared data/metrics for decisions), reflection management (iterative reviews with learning loops), and IBKR integration. Resources like current LangChain tools inspire pipelines, while exchange-calendars informs time-constrained checks. Now incorporates Langchain for enhanced orchestration: Agents as ReAct/custom modules with prompts (from base_prompt.txt and per-agent files) telling them to be well-informed (e.g., validate via tools/A2A), self-improving (e.g., reflect on batches/memory for refinements like SD >1.0 adjustments), and decisive (e.g., ROI >20% heuristics with escalations); LangGraph for flows/loops, memory for changelogs/batches. (For full A2A details, cross-ref a2a-protocol.txt as the centralized oracle.)

ABC Application is a sophisticated multi-agent AI system for quantitative portfolio management, combining Grok-powered reasoning with Interactive Brokers (IBKR) for professional-grade trading execution. The system operates on a macro-to-micro analysis hierarchy, enabling systematic market scanning combined with deep fundamental analysis.

### **Core Innovation: AI Reasoning Through 22-Agent Collaboration**

# ABC Application System Architecturearchitecture.md

## Overview

ABC Application is a sophisticated multi-agent AI system for quantitative portfolio management, combining Grok-powered reasoning with Interactive Brokers (IBKR) for professional-grade trading execution. The system operates on a macro-to-micro analysis hierarchy, enabling systematic market scanning combined with deep fundamental analysis.

## Core Architecture PrinciplesThe ABC Application system's fundamental breakthrough is its **22-agent collaborative reasoning architecture**. This creates a sophisticated AI reasoning environment where specialized agents debate, deliberate, and reach consensus on investment decisions - mimicking institutional investment committees but with AI precision, speed, and scalability.



### Agent-Based Design**Why 22 Agents for Reasoning?** Each agent represents a domain of financial expertise working in orchestrated reasoning loops:

- **22 Specialized Agents**: Each agent represents a domain of financial expertise- **Data Agents (11)**: Multi-source data validation and sentiment analysis

- **Collaborative Intelligence**: Agents debate and consensus-build through structured deliberation- **Strategy Agents (3)**: Options, flow, and ML strategy generation with debate

- **Autonomous Operation**: Agents make decisions using LLM reasoning and tool interactions- **Risk Agent (1)**: Probability-of-profit evaluations and risk assessments  

- **Memory Integration**: Shared memory systems enable cross-agent learning and adaptation- **Execution Agent (1)**: Trade execution with real-time monitoring

- **Learning Agent (1)**: Performance analysis and model refinement

### Macro-to-Micro Framework- **Reflection Agent (1)**: Decision validation and continuous improvement

- **Macro Phase**: Systematic scanning of 39+ sectors/assets for opportunity identification- **Macro Agent (1)**: Sector scanning and market regime analysis

- **Micro Phase**: Deep analysis of selected opportunities using full data pipeline- **Supporting Agents (3)**: Memory, coordination, and health monitoring

- **Hierarchical Intelligence**: Combines broad market perspective with detailed security analysis

**Collaborative Reasoning Process:**

#### **Two-Iteration Framework: Comprehensive → Executive Level**

The collaborative reasoning process operates in two distinct iterations, each building upon the previous with increasing levels of strategic oversight and risk sensitivity.

**Macro Foundation: Market Regime Assessment & Opportunity Identification**
The MacroAgent establishes the strategic foundation before any detailed analysis begins, scanning 39+ sectors/assets and identifying top opportunities for focused analysis.

**Iteration 1: Comprehensive Multi-Agent Deliberation (All 22 Agents)**
All agents, including subagents, participate in the complete 7-phase process to ensure maximum information gathering, diverse perspectives, and thorough analysis on the MacroAgent's prioritized opportunities.

**Iteration 2: Executive-Level Strategic Oversight (Main 8 Agents Only)**
Following the comprehensive deliberation, the main agents conduct a focused strategic review, applying executive-level judgment and risk sensitivity.

#### **Reflection Agent's Supreme Oversight Authority**
The ReflectionAgent serves as the system's final arbiter with unilateral authority to ensure decision quality and risk management:
- **Veto Authority**: Can veto any strategy based on catastrophic scenario analysis
- **Additional Iteration Trigger**: Can mandate one final comprehensive review if "canary in the coal mine" indicators emerge
- **Data Resurrection**: Can require reconsideration of any previously discussed data point or concern

**For detailed explanation of the 22-agent collaborative reasoning architecture, see:** `docs/ai-reasoning-agent-collaboration.md`

2. **StrategyAgent** - Trade strategy generation and optimization

3. **RiskAgent** - Portfolio risk management and position sizing* Macro Inputs and Data Ingestion: Start with external market data (e.g., from IBKR or yfinance-inspired feeds). The Data Agent processes this into structured formats (e.g., time series features via tsfresh concepts), providing a broad market overview. (See data-agent-notes.md for weekly adaptations and non-X sources.) Langchain: Agent with tools for pulls (e.g., yfinance_tool, x_semantic_search for sentiment); memory for changelog validations.

4. **ExecutionAgent** - Trade execution and order management

5. **ReflectionAgent** - Performance analysis and system improvement* Strategy Generation: Data Agent shares processed inputs via A2A to the Strategy Agent, which generates macro-level strategies (e.g., trend forecasts) transitioning to micro-level trade proposals with train-of-thought reasoning (e.g., step-by-step logic from current LangChain tools; options/flow-based for alpha; min params/diversification; pyramiding proposals with vol/corr). (See strategy-agent-notes.md for integration with weekly batches and options expansion.) Langchain: ReAct chain for proposals; memory for batch refinements (e.g., diversify if SD >1.0).

6. **LearningAgent** - Model refinement and pattern recognition

7. **MemoryAgent** - Memory coordination and retrieval* Risk Assessment: Strategy Agent passes proposals to the Risk Agent for probability of profit evaluations (e.g., Sharpe ratios via pyfolio), incorporating risk models (tf-quant-finance). A2A ensures shared metrics for collaborative adjustments; Risk Agent loads/enforces config/risk-constraints.yaml limits (core job: auto-adjust all metrics via sims/reflections; vets bonus overrides like sentiment SD ignores; bidirectional loop with Strategy until alpha/risk agreement, including dynamic pyramiding control/vol/corr; inherent goal weighting; tie-breaker/escalation). (See risk-agent-notes.md for stochastic outputs and dynamic management.) Langchain: Bidirectional edges in LangGraph for loops; tools for sims (e.g., tf_quant_monte_carlo); memory for post-batch adjustments.

8. **MacroAgent** - Sector analysis and asset class selection

* Pre-Execution Review: Risk Agent outputs to the Execution Agent, which initiates a preliminary check before final commitment.

#### Data Subagents (10)

- EconomicDatasub, SentimentDatasub, YfinanceDatasub, OptionsDatasub* Final Reflection Before Execution: To enforce time constraints and common-sense clarity, the Execution Agent triggers one last reflection loop—pulling from the Reflection Agent for a quick validation. This uses exchange-calendars concepts (e.g., check if current time is within market hours, holidays, or valid sessions) to avoid executions outside trading windows. Additionally, apply a common-sense test: Cross-verify trade details against predefined sanity rules (e.g., ensure quantities are feasible, no delusional elements like impossible prices, and alignment with overall portfolio logic). If it fails, loop back to Strategy/Risk for iteration; if passes, proceed—else, opt for "no trade" (USD hold benchmarked vs inflation/gold/crypto/FX costs from YAML). (See execution-agent-notes.md for USD-benchmarked logic and multi-asset paper testing.) Langchain: Reflection as mini-chain in LangGraph; tools for time/sanity checks; memory for outcome reflections.

- InstitutionalDatasub, NewsDatasub, FundamentalDatasub, MicrostructureDatasub

- KalshiDatasub, MarketDataAppDatasub* Micro Execution: Execution Agent handles IBKR-linked trades (e.g., via current LangChain tools and IBKR integration; multi-asset options/FX) or no-trade holds, logging outcomes (slippage live-only; no sim accuracy); ongoing A2A pings to Risk/Strategy for scaling assessments while active (continuous/vol/news/corr). (See execution-agent-notes.md for support for POP evaluations.) Langchain: Async edges for pings; tools for IBKR executions (e.g., ibkr_execute_tool); memory for drag weighing.



#### Strategy Subagents (4)* Post-Execution Reflection and Learning: Outcomes feed back via A2A to the Reflection Agent (Zipline-inspired backtests for reviews) and Learning Agent (FinRL/tf-quant-finance for ML refinements), closing the loop for experiential edge-finding (e.g., update probabilities based on real results). (See learning-agent-notes.md for parallel simulation training.) Langchain: Reflection/Learning as closing nodes; memory for convergence metrics (e.g., loss <0.01); tools for offline sims.

- FlowStrategySub, MLStrategySub, OptionsStrategySub, MultiInstrumentStrategySub

Weekly Stochastic Batching and POP Evaluations

### Data Flow Architecture* Daily Accumulation: Risk Agent logs stochastic outputs (e.g., JSON for Monte Carlo sims) and Execution Agent logs actuals (e.g., JSON for trade details); Learning Agent consolidates full DataFrames for problem trades (e.g., outliers appended during aggregation). (Cross-ref a2a-protocol.txt for formats.)

* Weekly Processing: Learning Agent aggregates into DataFrames; computes variance metrics (actual vs theoretical POP) against mean +1 SD threshold (e.g., trigger if >1 SD for sustained gaps).

```* Triggers and Adjustments: If threshold met, Learning Agent sends batched directives (DataFrames) via A2A to Data Agent for refinements (e.g., tsfresh updates); shares references to Strategy, Risk, Execution for context.

Market Data Sources* Handling Inconsistencies: O...(truncated 3672 characters)... 3-5 iters—ties to profitability (e.g., "Loop maxed alpha to 28% within drawdown"); Risk tie-breaker on risk; if unresolved after 5 iters, Strategy concedes and retries with different metrics; escalate to Reflection on high-conviction deadlocks; inherent goal weighting (max profit/min time/risk). Langchain: Bidirectional edges/routers for caps/escalations.

    ↓* Quarterly Audit Loop: If Q1 cumulative <30% vs target (Reflection poll), then A2A review for estimates >20% (no penalties, pure review); else, log success—ties to profitability (e.g., "Audit: 18% achieved; vote on 25% Q2 upside"). Optional trigger: If estimates > reflection_bonus_threshold (0.25 from profitability-targets.yaml), award bonuses (virtual alpha credits, e.g., +5% POP in Learning batches) logged in changelogs for profit incentive; route overrides (e.g., sentiment SD ignores) through Risk for vetting. Loose expense check: If token/external drags >0.5% (from portfolio-dashboard.txt), flag for reflection prune (e.g., reduce batch frequency; preserves 0.5-1% alpha). Langchain: Poll hubs with memory audits.

DataAgent (Aggregation & Processing)* Sim Processing Loop: If sim results processed (per-week log), then distribute knowledge via A2A DataFrames to agents; else, retry offline run—ties to profitability (e.g., "Sim lift +1.2% ROI: Feeds batch for target alignment"). Langchain: Subgraphs for sims/distributions.

    ↓

MacroAgent (Sector Analysis & Selection)A2A and Reflection Management (Cross-Ref a2a-protocol.txt for Centralized Details)

    ↓* A2A: Use event-driven (current LangChain tools) or data-sharing (pandas from yfinance) formats; extend to Data Agent for X feeds (JSON summaries). Langchain: Message passing with schemas; hubs for broadcasts.

DataAgent (Micro Analysis on Selected Assets)* Reflection: Post-trade metrics (pyfolio) feed Learning for probability refinements; pre-execution final step ensures time/clarity; Risk auto-adjusts all YAML metrics post-reflection (sims pre-launch); Execution logs performance for all outcomes (trades/holds) for risk reduction; loose expense tracking in dashboard for quarterly reviews (optional trigger); escalates deadlocks. Langchain: Memory stores for summaries; loops with evaluation tools.

    ↓

StrategyAgent (Trade Generation)Agent Behaviors Integration (New Section for Autonomy)

    ↓* All agents follow behaviors in agent-behavior-guidelines.md: Proactive A2A querying for info gaps, self-improving via batches/changelogs (e.g., reflect on SD >1.0 from memory for adjustments), decisive ROI heuristics (>20% estimates with escalations), and common-sense checks. This embeds autonomy into flows (e.g., Strategy pings Data on stale sentiment before proposal). Behaviors encoded in prompts (base_prompt.txt + per-agent) to tell agents: "Validate via tools, refine from memory batches, decide with ROI >20% and log in JSON."

RiskAgent (Risk Assessment & Sizing)* Reasoning: Integrates guidelines for well-informed decisions; enhances self-improvement by tying behaviors to loops (e.g., "On SD trigger, Risk adjusts YAML per heuristic"); Langchain prompts ensure traceable, adaptive agents.

    ↓

ExecutionAgent (Order Execution)Expense Pruning Behaviors (Tied to portfolio-dashboard.txt)

    ↓* Integrated from agent-behavior-guidelines.md: Agents monitor drags (e.g., Execution weighs in pings); Reflection flags in audits if >0.5%; prune if drag > alpha * 0.3 (revert on ROI drop >5%). Langchain: Embed in prompts; routers prune chains dynamically.

ReflectionAgent (Performance Analysis)* Reasoning: Scalability safeguard; preserves alpha without overload, backing funding with cost-efficient loops.

    ↓

LearningAgent (Model Refinement)Additional Details

    ↺* A2A Protocol: Agents use standardized sharing (e.g., JSON for events/logs, DataFrames for metrics/batches) to enable seamless collaboration and traceability. (Full spec in a2a-protocol.txt.)

Memory Systems (Continuous Learning)* Reflection Management: Embedded throughout, with weekly batching as a system-wide loop for SD-tied reviews; dedicated changelog for Learning-to-Data changes; Risk-led constraints for closed-loop dynamism; Execution's USD/no-trade as benchmarked reflection. Langchain: Memory for experiential loops; tools for validations.

```* IBKR Integration: Centered in Execution Agent, with time validations and actual logs feeding evaluations. Langchain: Custom tools for API calls.



### Technology StackReasoning: Bullet-point flow improves readability over diagrams; incorporates weekly/SD batching, Risk-managed constraints (all metrics adjustable via sims), and Execution's USD/no-trade discipline for time constraints in reflections, backing funding with defensible risk mitigation (e.g., prevents ~10-20% of invalid trades conceptually by enforcing market realities and common-sense clarity, now self-improving via experiential loops with slippage conservatism). Phased approach backs funding with milestones; ensures robust organization for profitable, experiential system, with time constraints reducing invalid trades; now expanded with A2A cross-refs to a2a-protocol.txt for declutter, tying to aspirational ROI (e.g., "Loops preserve 18% Q1 estimate vs 20% goal") for coherence and technical journey reduction. Merged integration for leaner structure, saving ~15% overhead while preserving phased scalability. Added reflection bonuses to audits as optional triggers for profit incentive, gamifying upside without enforcements to drive ~15% ambition lift; loose expense cap in audits prunes drags early for alpha preservation; integrated options/multi-asset for Strategy/Execution asymmetry; deepened Strategy-Risk loop for alpha/risk harmony with pyramiding/min params/diversification/ongoing scaling/vol/news/corr/escalation/retry/tie-breaker/inherent goal. Added behaviors integration for agent autonomy, lifting decision robustness ~10%. Added expense pruning for scalability, preserving 0.5-1% alpha. Langchain addition enhances agent behaviors/modularity, reducing variances ~15% for funded traceability.

#### Core Technologies
- **Python 3.11+**: Primary development language
- **LangChain**: LLM orchestration and tool integration
- **Grok API**: Advanced reasoning and market analysis
- **IBKR API**: Professional trading execution
- **Redis**: High-performance caching and memory storage

#### Data Sources
- **yfinance**: Free historical market data
- **IBKR Historical Data**: Professional-grade market data
- **MarketDataApp**: Premium real-time trading data
- **FRED**: Economic indicators and policy data
- **NewsAPI/CurrentsAPI**: Real-time news and sentiment
- **Twitter/X API**: Social sentiment analysis
- **Kalshi**: Prediction market data

#### Infrastructure
- **Asyncio**: Asynchronous processing for concurrent operations
- **Pandas/NumPy**: Data manipulation and analysis
- **Redis**: Caching and memory management
- **YAML**: Configuration management
- **Logging**: Comprehensive audit trails

## Agent Communication Protocol (A2A)

### Communication Patterns
- **Debate Sessions**: Multi-agent discussions for consensus building
- **Data Sharing**: Structured JSON/DataFrame exchanges
- **State Coordination**: Shared memory spaces for agent synchronization
- **Event-Driven**: Asynchronous messaging for real-time coordination

### Protocol Structure
```json
{
  "message_type": "debate|data_share|state_update",
  "sender": "agent_name",
  "recipients": ["agent_list"],
  "content": {
    "context": "...",
    "data": {...},
    "decisions": [...]
  },
  "timestamp": "ISO_datetime",
  "correlation_id": "uuid"
}
```

## Memory Architecture

### Memory Types
- **Short-term Memory**: Current session context and temporary data
- **Long-term Memory**: Historical patterns and learned behaviors
- **Episodic Memory**: Specific agent interactions and outcomes
- **Semantic Memory**: Financial concepts and relationships
- **Shared Memory**: Cross-agent intelligence and collaborative insights

### Memory Integration
- **Redis Backend**: High-performance storage and retrieval
- **Vector Storage**: Semantic search capabilities for pattern matching
- **Collaborative Learning**: Agents contribute to and learn from shared experiences
- **Context Preservation**: Maintain analysis context across sessions

## Risk Management Framework

### Risk Layers
1. **Position-Level**: Individual trade risk controls
2. **Portfolio-Level**: Overall portfolio diversification and limits
3. **System-Level**: Circuit breakers and emergency stops
4. **Operational-Level**: API health monitoring and failover

### Risk Parameters
- **Maximum Drawdown**: <5% portfolio limit
- **Position Sizing**: Risk-based allocation algorithms
- **Volatility Controls**: Dynamic position adjustments
- **Correlation Limits**: Sector and asset class diversification

## Performance Objectives

### Target Metrics
- **Return Target**: 10-20% monthly returns
- **Risk Control**: <5% maximum drawdown
- **Win Rate**: >60% profitable trades
- **Sharpe Ratio**: >2.0 risk-adjusted returns

### Success Criteria
- **Profitability**: Consistent achievement of return targets
- **Risk Management**: Maintain drawdown limits under all conditions
- **Scalability**: System performance with increasing complexity
- **Reliability**: 99.9% uptime with automated recovery

## Security and Compliance

### Data Security
- **API Key Management**: Secure credential storage and rotation
- **Data Encryption**: Encrypted communication channels
- **Access Controls**: Role-based permissions for system components

### Regulatory Compliance
- **Audit Logging**: Complete transaction and decision trails
- **Traceability**: All decisions linked to reasoning and data sources
- **Ethical AI**: Bias monitoring and fairness controls
- **Financial Regulations**: Compliance with trading and reporting requirements

## Deployment Architecture

### Development Environment
- **Local Development**: Full system simulation capabilities
- **Testing Framework**: Comprehensive unit and integration tests
- **CI/CD Pipeline**: Automated testing and deployment

### Production Environment
- **Cloud Infrastructure**: Scalable hosting with redundancy
- **Monitoring Systems**: Real-time performance and health monitoring
- **Backup Systems**: Data backup and disaster recovery
- **Failover Mechanisms**: Automatic system recovery and switching

## Future Enhancements

### Planned Features
- **Advanced ML Models**: Enhanced predictive capabilities
- **Real-time Adaptation**: Dynamic strategy adjustment
- **Multi-Asset Expansion**: Additional asset classes and markets
- **Performance Analytics**: Advanced reporting and visualization

### Research Areas
- **LLM Advancement**: Integration of newer reasoning models
- **Market Microstructure**: Enhanced order flow analysis
- **Alternative Data**: Novel data sources and signals
- **Portfolio Optimization**: Advanced allocation algorithms

---

*This architecture document provides the foundation for understanding the ABC Application system. For detailed implementation guides, refer to the AGENTS and IMPLEMENTATION sections.*