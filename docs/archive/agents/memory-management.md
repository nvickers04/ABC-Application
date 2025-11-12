# Memory Management for AI Portfolio Manager

## Overview
This document outlines the memory management strategy for the AI Portfolio Manager, a multi-agent system using Grok agents via xAI API. Memory ensures agents retain critical context for decision-making, personalization, and profitability (target: 15-20% annual returns with <5% drawdown). It addresses short-term (session-specific) and long-term (cross-session/user-specific) needs, enabling agents to "remember" market analyses, user preferences, trade histories, and inter-agent communications.

Key Principles:
- **Persistence & Retrieval**: All memories must be auditable, with logs for compliance (e.g., SEC regulations).
- **Scalability**: Handle high-frequency data (e.g., real-time IBKR feeds) without exceeding LLM context windows.
- **Security**: Encrypt sensitive data (e.g., user portfolios); use decay for non-essential info.
- **Hybrid Approach**: Leverage xAI's built-in Grok memory for basic statefulness (cross-ref https://x.ai/api for API-specific limits and endpoints, e.g., response IDs for state recall), augmented by external tools for multi-agent complexity.
- **Critiques & Risks**: Avoid single points of failure; test for memory overload in volatile markets. Ensure ethical handling—no unsolicited data retention.

## Memory Types & Handling
Inspired by human cognition, adapted for financial agents:

1. **Short-Term Memory (Working/Thread-Scoped)**:
   - **Purpose**: Retain immediate context (e.g., current market query, ongoing trade simulation).
   - **Tools/Implementation**:
     - xAI Grok Memory: Server-side, tied to response IDs for seamless recall in single-agent threads (API details: https://x.ai/api; limits ~10-20 interactions per thread).
     - LangGraph Checkpoints: Persist state per session; prune via summarization to fit context limits.
     - Redis: In-memory storage for fast access; use eviction policies for temporary data.
   - **Capacity**: Limit to 5-10 recent interactions; summarize older ones.
   - **Examples**: Agent analyzing intraday stock volatility recalls recent IBKR data fetches.

2. **Long-Term Memory (Cross-Thread/Persistent)**:
   - **Sub-Types**:
     - **Semantic**: Facts/preferences (e.g., user risk tolerance: conservative, 60/40 stock/bond allocation).
     - **Episodic**: Past events (e.g., 2024 Q3 trade outcomes, lessons from drawdowns).
     - **Procedural**: Rules/instructions (e.g., rebalancing algorithm: sell if >10% over target).
   - **Tools/Implementation**:
     - Mem0: Universal layer for self-improving recall; integrate with CrewAI for multi-agent sharing. Use vector search for efficient retrieval.
     - LangGraph Stores: Hierarchical namespaces (e.g., user_id/portfolio_id); support semantic similarity queries.
     - Redis: Vectorization + graph storage for interconnected memories (e.g., link trades to market events).
     - Zep/Letta (Optional): For temporal graphs if scalability demands grow.
   - **Storage**: JSON docs in Redis/DB; embeddings for search (e.g., via FAISS if needed).
   - **Update Mechanism**: Background async writes; LLM-driven reflection for refinements (e.g., update procedural memory post-trade review).
   - **Decay/Pruning**: Timestamp-based expiration (e.g., non-critical data after 30 days); manual overrides for compliance.

3. **Multi-Agent Memory Sharing**:
   - **Purpose**: Enable coordination (e.g., Analysis Agent shares insights with Execution Agent).
   - **Implementation**: Shared namespaces in LangGraph/Redis; Mem0 for cross-agent personalization.
   - **Protocol**: Use A2A (Agent-to-Agent) messaging with memory embeds; log all shares for traceability.
   - **Examples**: Risk Agent recalls episodic memory from Trade Agent to veto high-volatility positions.

4. **Open Positions Memory**:
   - **Purpose**: Track live portfolio positions across agents for real-time risk monitoring and P&L calculation.
   - **Implementation**: Dedicated position tracking in AdvancedMemoryManager with methods for track_open_position(), update_open_position(), close_open_position(), and get_open_positions().
   - **Storage**: Positions stored in shared namespace "portfolio" with status tracking (open/closed).
   - **Integration**: Execution Agent automatically tracks positions on trade execution; Dashboard displays live positions; Risk Agent monitors for exposure limits.
   - **Features**: Supports partial closes, P&L tracking, position metadata (entry price, timestamp, source).
   - **Examples**: Strategy Agent checks open positions before proposing new trades; Risk Agent calculates portfolio exposure in real-time.

## Integration with System Components
- **xAI Grok Agents**: Use API for native memory (redirect to https://x.ai/api for details; e.g., integrate with Grok 4 endpoints for persistent state across sessions). Augment with wrappers for external tools.
- **IBKR Brokerage Link**: Store trade confirmations as episodic memories; retrieve for audits.
- **Profitability Alignment**: Memories inform decisions toward targets (e.g., recall past strategies yielding >15% returns).
- **Workflow Mapping**:
  - On Init: Load user semantic memory.
  - During Task: Use short-term for real-time; query long-term via vector search.
  - Post-Task: Write new memories asynchronously; reflect for improvements.

## Testing & Validation
- **Metrics**: Recall accuracy (>95%), latency (<500ms), token efficiency (reduce by 70-80%).
- **Scenarios**: Simulate market crashes (e.g., recall 2008 patterns); test multi-agent handoffs.
- **Edge Cases**: Memory conflicts (resolve via timestamps); data loss (use backups).
- **Auditing**: Full logs; exportable for funding reviews.

## Next Steps for Code Transition
- Map to Classes: e.g., `MemoryManager` base class with subclasses for types/tools.
- Dependencies: List pip installs (e.g., mem0ai, langgraph, redis).
- Refinements: Iterate on this doc before prototyping—focus on hybrid to mitigate xAI limitations (API cross-ref added for specifics).

This structure ensures robust, defensible reasoning for your legacy project.