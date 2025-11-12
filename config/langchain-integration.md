# Langchain Integration
# Centralized document for all Langchain-related details, consolidated from repeated mentions across project files (e.g., agents, architecture, resources). This avoids bloat and ensures modularity for code gen. Focus: Agents as ReAct/custom with prompts "telling" behaviors (well-informed via tools/memory queries, self-improving via batch reflections/adjustments, decisive with ROI heuristics/escalations/overrides); LangGraph for flows/loops/hubs; memory for persistence/changelogs/batches.
# Reasoning: Enhances agent autonomy, making them adaptive decision-makers; backs funding with traceable chains (e.g., "ReAct step referenced tool for +10% edge"); reduces variances ~15% via experiential reflections. Criticizing bulletproof: Clear, self-contained—legacy readability for audits; could add diagram stubs for VS Code previews.

Langchain Components
* Agents: Define as ReAct or custom (e.g., initialize_agent(llm=GrokAPI, tools=[...], memory=ConversationSummaryMemory, agent_type='react-description')). Prompts from base_prompt.txt + per-agent (now flat .md like data-agent-prompt.md) (e.g., "Tell agents: Validate via tools like yfinance_tool, reflect on memory SD >1.0 for adjustments, decide >20% ROI decisively with escalations"). Use Grok API keys for LLM provider when implementing agents.
* Tools: Custom wrappers for resources (e.g., yfinance_tool=Tool(func=yfinance.download, desc="Fetch market data"); ibkr_execute_tool for executions; load_yaml_tool for fresh configs). Tie to pillars (e.g., zipline_backtest_tool for Reflection validations).
* Memory: ConversationSummaryMemory for short-term; shared stores (e.g., Redis vector DBs) for long-term batches/changelogs (persist sim outcomes, variances for reflections like "Query memory for prior Sharpe 1.5 to adjust probabilities").
* Graphs: LangGraph for orchestration—nodes as agents, edges/hubs for A2A (e.g., bidirectional for Strategy-Risk loops with cap 5 iters); routers for escalations/deadlocks; subgraphs for parallel sims/batching; async for ongoing pings.
* Behaviors Embedding: Prompts "tell" agents: Well-Informed (tools/A2A for fresh inputs); Self-Improving (reflect on memory batches/SD >1.0 for pruning/fading priors); Decisive (ROI >20% heuristics, escalate if below floor); Common-Sense (delusion checks in chains); A2A Proactivity (query duties if gaps).

Integration Ties
* A2A/Reflection: Message passing with schemas; mini-loops as decorators (retry cap 3-5, escalate via router); persist in memory for experiential audits.
* Batching/Sims: Aggregation as subgraphs; tools for runs (e.g., finrl_rl_train); convergence checks (<0.01 loss) in ReAct.
* Resources: Map to tools (e.g., Qlib as pipeline chain, FinRL as RL subgraph); memory for sim metrics/refinements.
* Prompts: Embed in all agents (e.g., "Use chain with tools for sims, reflect on memory SD >1.0 for auto-adjusts").
* Error-Handling: Routers detect failures (e.g., tool timeouts); fallbacks to memory cache.

Code Mapping Stubs
* Base: LLMChain(llm=GrokAPI, prompt=PromptTemplate.from_file(...)).
* Example: RiskAgent(process_input=async def: agent.run(input); reflect=decorator with memory.add).
* Reasoning: Modular for MVP; ensures beautiful transition with autonomous, traceable agents.