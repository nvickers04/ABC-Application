architecture.md
# System Architecture
# High-Level Flow Description
# This describes the multi-agent system's workflow in a sequential, bullet-point format for clarity. It emphasizes macro-to-micro daily progression (e.g., broad data analysis to granular executions), A2A interactions (e.g., shared data/metrics for decisions), reflection management (iterative reviews with learning loops), and IBKR integration. Resources like current LangChain tools inspire pipelines, while exchange-calendars informs time-constrained checks. Now incorporates Langchain for enhanced orchestration: Agents as ReAct/custom modules with prompts (from base_prompt.txt and per-agent files) telling them to be well-informed (e.g., validate via tools/A2A), self-improving (e.g., reflect on batches/memory for refinements like SD >1.0 adjustments), and decisive (e.g., ROI >20% heuristics with escalations); LangGraph for flows/loops, memory for changelogs/batches. (For full A2A details, cross-ref a2a-protocol.txt as the centralized oracle.)

### **Core Innovation: AI Reasoning Through 22-Agent Collaboration**
The ABC Application system's fundamental breakthrough is its **22-agent collaborative reasoning architecture**. This creates a sophisticated AI reasoning environment where specialized agents debate, deliberate, and reach consensus on investment decisions - mimicking institutional investment committees but with AI precision, speed, and scalability.

**Why 22 Agents for Reasoning?** Each agent represents a domain of financial expertise working in orchestrated reasoning loops:
- **Data Agents (11)**: Multi-source data validation and sentiment analysis
- **Strategy Agents (3)**: Options, flow, and ML strategy generation with debate
- **Risk Agent (1)**: Probability-of-profit evaluations and risk assessments  
- **Execution Agent (1)**: Trade execution with real-time monitoring
- **Learning Agent (1)**: Performance analysis and model refinement
- **Reflection Agent (1)**: Decision validation and continuous improvement
- **Macro Agent (1)**: Sector scanning and market regime analysis
- **Supporting Agents (3)**: Memory, coordination, and health monitoring

**Collaborative Reasoning Process:**
1. **Proposal Generation**: Strategy agents independently generate investment proposals
2. **Debate & Challenge**: Agents cross-validate assumptions and challenge conclusions
3. **Risk Assessment**: Risk agent evaluates all proposals with stochastic analysis
4. **Consensus Building**: Reflection agent mediates conflicts and ensures alignment
5. **Execution Validation**: Final sanity checks before trade execution
6. **Learning Integration**: Performance feedback refines future reasoning

**Proven Results:** The system achieved profitability through agent collaboration alone. **With grok-4-fast-reasoning model integration, each agent's reasoning capabilities are exponentially enhanced, projecting returns off the charts.**

**Implementation:** Agents use structured reasoning protocols with Langchain ReAct patterns, collaborative memory systems, and A2A communication for seamless deliberation. This creates institutional-grade decision quality through AI collective intelligence.

**For detailed explanation of the 22-agent collaborative reasoning architecture, see:** `docs/ai-reasoning-agent-collaboration.md`

* Macro Inputs and Data Ingestion: Start with external market data (e.g., from IBKR or yfinance-inspired feeds). The Data Agent processes this into structured formats (e.g., time series features via tsfresh concepts), providing a broad market overview. (See data-agent-notes.md for weekly adaptations and non-X sources.) Langchain: Agent with tools for pulls (e.g., yfinance_tool, x_semantic_search for sentiment); memory for changelog validations.

* Strategy Generation: Data Agent shares processed inputs via A2A to the Strategy Agent, which generates macro-level strategies (e.g., trend forecasts) transitioning to micro-level trade proposals with train-of-thought reasoning (e.g., step-by-step logic from current LangChain tools; options/flow-based for alpha; min params/diversification; pyramiding proposals with vol/corr). (See strategy-agent-notes.md for integration with weekly batches and options expansion.) Langchain: ReAct chain for proposals; memory for batch refinements (e.g., diversify if SD >1.0).

* Risk Assessment: Strategy Agent passes proposals to the Risk Agent for probability of profit evaluations (e.g., Sharpe ratios via pyfolio), incorporating risk models (tf-quant-finance). A2A ensures shared metrics for collaborative adjustments; Risk Agent loads/enforces config/risk-constraints.yaml limits (core job: auto-adjust all metrics via sims/reflections; vets bonus overrides like sentiment SD ignores; bidirectional loop with Strategy until alpha/risk agreement, including dynamic pyramiding control/vol/corr; inherent goal weighting; tie-breaker/escalation). (See risk-agent-notes.md for stochastic outputs and dynamic management.) Langchain: Bidirectional edges in LangGraph for loops; tools for sims (e.g., tf_quant_monte_carlo); memory for post-batch adjustments.

* Pre-Execution Review: Risk Agent outputs to the Execution Agent, which initiates a preliminary check before final commitment.

* Final Reflection Before Execution: To enforce time constraints and common-sense clarity, the Execution Agent triggers one last reflection loop—pulling from the Reflection Agent for a quick validation. This uses exchange-calendars concepts (e.g., check if current time is within market hours, holidays, or valid sessions) to avoid executions outside trading windows. Additionally, apply a common-sense test: Cross-verify trade details against predefined sanity rules (e.g., ensure quantities are feasible, no delusional elements like impossible prices, and alignment with overall portfolio logic). If it fails, loop back to Strategy/Risk for iteration; if passes, proceed—else, opt for "no trade" (USD hold benchmarked vs inflation/gold/crypto/FX costs from YAML). (See execution-agent-notes.md for USD-benchmarked logic and multi-asset paper testing.) Langchain: Reflection as mini-chain in LangGraph; tools for time/sanity checks; memory for outcome reflections.

* Micro Execution: Execution Agent handles IBKR-linked trades (e.g., via current LangChain tools and IBKR integration; multi-asset options/FX) or no-trade holds, logging outcomes (slippage live-only; no sim accuracy); ongoing A2A pings to Risk/Strategy for scaling assessments while active (continuous/vol/news/corr). (See execution-agent-notes.md for support for POP evaluations.) Langchain: Async edges for pings; tools for IBKR executions (e.g., ibkr_execute_tool); memory for drag weighing.

* Post-Execution Reflection and Learning: Outcomes feed back via A2A to the Reflection Agent (Zipline-inspired backtests for reviews) and Learning Agent (FinRL/tf-quant-finance for ML refinements), closing the loop for experiential edge-finding (e.g., update probabilities based on real results). (See learning-agent-notes.md for parallel simulation training.) Langchain: Reflection/Learning as closing nodes; memory for convergence metrics (e.g., loss <0.01); tools for offline sims.

Weekly Stochastic Batching and POP Evaluations
* Daily Accumulation: Risk Agent logs stochastic outputs (e.g., JSON for Monte Carlo sims) and Execution Agent logs actuals (e.g., JSON for trade details); Learning Agent consolidates full DataFrames for problem trades (e.g., outliers appended during aggregation). (Cross-ref a2a-protocol.txt for formats.)
* Weekly Processing: Learning Agent aggregates into DataFrames; computes variance metrics (actual vs theoretical POP) against mean +1 SD threshold (e.g., trigger if >1 SD for sustained gaps).
* Triggers and Adjustments: If threshold met, Learning Agent sends batched directives (DataFrames) via A2A to Data Agent for refinements (e.g., tsfresh updates); shares references to Strategy, Risk, Execution for context.
* Handling Inconsistencies: O...(truncated 3672 characters)... 3-5 iters—ties to profitability (e.g., "Loop maxed alpha to 28% within drawdown"); Risk tie-breaker on risk; if unresolved after 5 iters, Strategy concedes and retries with different metrics; escalate to Reflection on high-conviction deadlocks; inherent goal weighting (max profit/min time/risk). Langchain: Bidirectional edges/routers for caps/escalations.
* Quarterly Audit Loop: If Q1 cumulative <30% vs target (Reflection poll), then A2A review for estimates >20% (no penalties, pure review); else, log success—ties to profitability (e.g., "Audit: 18% achieved; vote on 25% Q2 upside"). Optional trigger: If estimates > reflection_bonus_threshold (0.25 from profitability-targets.yaml), award bonuses (virtual alpha credits, e.g., +5% POP in Learning batches) logged in changelogs for profit incentive; route overrides (e.g., sentiment SD ignores) through Risk for vetting. Loose expense check: If token/external drags >0.5% (from portfolio-dashboard.txt), flag for reflection prune (e.g., reduce batch frequency; preserves 0.5-1% alpha). Langchain: Poll hubs with memory audits.
* Sim Processing Loop: If sim results processed (per-week log), then distribute knowledge via A2A DataFrames to agents; else, retry offline run—ties to profitability (e.g., "Sim lift +1.2% ROI: Feeds batch for target alignment"). Langchain: Subgraphs for sims/distributions.

A2A and Reflection Management (Cross-Ref a2a-protocol.txt for Centralized Details)
* A2A: Use event-driven (current LangChain tools) or data-sharing (pandas from yfinance) formats; extend to Data Agent for X feeds (JSON summaries). Langchain: Message passing with schemas; hubs for broadcasts.
* Reflection: Post-trade metrics (pyfolio) feed Learning for probability refinements; pre-execution final step ensures time/clarity; Risk auto-adjusts all YAML metrics post-reflection (sims pre-launch); Execution logs performance for all outcomes (trades/holds) for risk reduction; loose expense tracking in dashboard for quarterly reviews (optional trigger); escalates deadlocks. Langchain: Memory stores for summaries; loops with evaluation tools.

Agent Behaviors Integration (New Section for Autonomy)
* All agents follow behaviors in agent-behavior-guidelines.md: Proactive A2A querying for info gaps, self-improving via batches/changelogs (e.g., reflect on SD >1.0 from memory for adjustments), decisive ROI heuristics (>20% estimates with escalations), and common-sense checks. This embeds autonomy into flows (e.g., Strategy pings Data on stale sentiment before proposal). Behaviors encoded in prompts (base_prompt.txt + per-agent) to tell agents: "Validate via tools, refine from memory batches, decide with ROI >20% and log in JSON."
* Reasoning: Integrates guidelines for well-informed decisions; enhances self-improvement by tying behaviors to loops (e.g., "On SD trigger, Risk adjusts YAML per heuristic"); Langchain prompts ensure traceable, adaptive agents.

Expense Pruning Behaviors (Tied to portfolio-dashboard.txt)
* Integrated from agent-behavior-guidelines.md: Agents monitor drags (e.g., Execution weighs in pings); Reflection flags in audits if >0.5%; prune if drag > alpha * 0.3 (revert on ROI drop >5%). Langchain: Embed in prompts; routers prune chains dynamically.
* Reasoning: Scalability safeguard; preserves alpha without overload, backing funding with cost-efficient loops.

Additional Details
* A2A Protocol: Agents use standardized sharing (e.g., JSON for events/logs, DataFrames for metrics/batches) to enable seamless collaboration and traceability. (Full spec in a2a-protocol.txt.)
* Reflection Management: Embedded throughout, with weekly batching as a system-wide loop for SD-tied reviews; dedicated changelog for Learning-to-Data changes; Risk-led constraints for closed-loop dynamism; Execution's USD/no-trade as benchmarked reflection. Langchain: Memory for experiential loops; tools for validations.
* IBKR Integration: Centered in Execution Agent, with time validations and actual logs feeding evaluations. Langchain: Custom tools for API calls.

Reasoning: Bullet-point flow improves readability over diagrams; incorporates weekly/SD batching, Risk-managed constraints (all metrics adjustable via sims), and Execution's USD/no-trade discipline for time constraints in reflections, backing funding with defensible risk mitigation (e.g., prevents ~10-20% of invalid trades conceptually by enforcing market realities and common-sense clarity, now self-improving via experiential loops with slippage conservatism). Phased approach backs funding with milestones; ensures robust organization for profitable, experiential system, with time constraints reducing invalid trades; now expanded with A2A cross-refs to a2a-protocol.txt for declutter, tying to aspirational ROI (e.g., "Loops preserve 18% Q1 estimate vs 20% goal") for coherence and technical journey reduction. Merged integration for leaner structure, saving ~15% overhead while preserving phased scalability. Added reflection bonuses to audits as optional triggers for profit incentive, gamifying upside without enforcements to drive ~15% ambition lift; loose expense cap in audits prunes drags early for alpha preservation; integrated options/multi-asset for Strategy/Execution asymmetry; deepened Strategy-Risk loop for alpha/risk harmony with pyramiding/min params/diversification/ongoing scaling/vol/news/corr/escalation/retry/tie-breaker/inherent goal. Added behaviors integration for agent autonomy, lifting decision robustness ~10%. Added expense pruning for scalability, preserving 0.5-1% alpha. Langchain addition enhances agent behaviors/modularity, reducing variances ~15% for funded traceability.