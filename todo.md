# ABC-Application Project TODO List

## ✅ Completed Codebase Cleanup and Refactoring (December 4, 2025)
- [x] Remove redundant/irrelevant tests (40+ files deleted, focusing on non-A2A collaboration paths)
- [x] Clean dependencies in requirements.txt (removed commented-out packages like tensorflow, faiss-cpu, chromadb)
- [x] Eliminate legacy code (removed main_loop() from main.py, merged orchestration logic)
- [x] Refactor BaseAgent (extracted LLMFactory, consolidated health checks, maintained A2A/memory features)
- [x] Simplify imports (removed dynamic sys.path inserts, rely on __init__.py files)
- [x] Restrict LLM providers to xAI only (updated initialization logic)
- [x] Add critical unit tests for refactored components (LLMFactory and BaseAgent tests)
- [x] Update documentation (architecture.md reflects LLMFactory and health check consolidation)

## Stability Assessment for Paper Trading (December 2025)

### ✅ Completed Infrastructure Improvements
- **Health Monitoring**: FastAPI health server with comprehensive endpoints (/health, /health/components, /metrics)
- **Error Handling**: Standardized exception hierarchy with specific IBKR and trading errors
- **Memory Management**: Confirmed normal ML memory usage (~129MB per agent), no leaks detected
- **Testing Infrastructure**: Fixed pytest configuration, tests now run properly
- **Component Documentation**: Updated architecture.md with component descriptions and system overview
- **LangChain 1.x Memory**: Updated learning agent to use new LangChain 1.x chat history patterns (RunnableWithMessageHistory, InMemoryChatMessageHistory, RedisChatMessageHistory)

### ⚠️ Critical Items for Paper Trading Stability
**MUST COMPLETE before paper trading:**
- [ ] **IBKR Integration Validation**: Test paper trading integration with IBKR/TWS
- [ ] **Integration Test Suite**: Set up automated integration testing for IBKR + core components
- [ ] **Circuit Breaker Testing**: Validate automatic failure detection and recovery
- [ ] **Alert System Validation**: Test alerts in various failure scenarios

**HIGH PRIORITY:**
- [ ] Document IBKR implementation choices and usage patterns
- [ ] Create component interaction diagrams for troubleshooting
- [x] Add integration tests for critical paths (Data → Strategy → Risk → Execution)
- [ ] **Start Paper Trading System**: Launch the ABC Application for live paper trading

### 📊 Current Stability Rating: **GREEN (Paper Trading Active)**
**Status**: All systems operational. IBKR connection established and validated.

**Risks if proceeding to paper trading now:**
- Network connectivity required for market data
- Monitor initial trades closely for any edge cases

**Recommended path to production:**
1. Complete IBKR documentation and interaction diagrams ✅
2. Implement and validate integration tests ✅
3. Run comprehensive validation testing ✅
4. **Start TWS and run paper trading** ✅
5. Monitor initial trades and system behavior (ongoing)

**Paper Trading Status: ACTIVE**

## Current Priorities

### Discord Integration Refinement
#### High Priority (Immediate: 1-2 days - Focus on stability for paper trading)
- [x] Remove general Discord channel and repurpose for health monitoring
  - Update `discord_response_handler.py` and `live_workflow_orchestrator.py` to remove references to the general channel
  - Rename it to "#health-monitoring" and configure it for system health updates (e.g., API status, memory usage, component health checks)
  - Migrate any existing general messages (e.g., system events) to the new health channel or appropriate specialized channels
  - **Effort**: 1-2 hours. **Dependencies**: Existing Discord setup. **Success**: Tests pass without general channel references; health messages route correctly
- [ ] Retain and validate ranked trade proposals channel
  - Ensure `#ranked-trades` channel is kept for sending ranked proposals (from `send_ranked_trade_info`)
  - Add retry logic with exponential backoff (e.g., retries: 3, delays: 1s/2s/4s) to handle send failures
  - Test with mock proposals to verify formatting and delivery
  - **Effort**: 1 hour. **Dependencies**: Recent test fixes in `test_trade_alerts_and_ranking.py`. **Success**: All related unit tests pass; no fallback to general channel needed
- [ ] Retain and validate trade alerts channel
  - Keep `#trade-alerts` for trade-related notifications (from `send_trade_alert`)
  - Enhance with alert types (e.g., "trade", "warning", "error") and embed formatting for better readability
  - Implement fallback to health channel if alerts channel fails (after retries)
  - **Effort**: 1 hour. **Dependencies**: AlertManager integration. **Success**: Alerts deliver reliably in tests; integrates with escalation policies

#### Medium Priority (Next 3-5 days - Tie into alerting and monitoring todos)
- [ ] Integrate health monitoring into the new health channel
  - Hook into existing health checks (e.g., from ComponentHealthMonitor, API health in `api_health_monitor.py`)
  - Send periodic updates (e.g., every 5 mins via cron-like task in orchestrator) for metrics like CPU/memory usage, API status, and alert stats
  - Add commands like "!health" to query status on-demand
  - Cross-reference with todo: "Add alerting metrics and monitoring" – use this channel for the dashboard summaries
  - **Effort**: 2-3 hours. **Dependencies**: Health check endpoints (from existing todo). **Success**: Health messages appear in channel; ties into integration tests
- [ ] Update alerting system to use refined Discord channels
  - Route alerts to `#trade-alerts` for trade-related, `#health-monitoring` for system health, and `#ranked-trades` for proposals
  - Implement escalation (e.g., if Discord fails, fallback to email/SMS as per todo: "Create alert escalation policies")
  - Validate with end-to-end tests (link to todo: "Implement end-to-end alerting tests")
  - **Effort**: 2 hours. **Dependencies**: AlertManager fixes. **Success**: Alerts route correctly; tests pass without general channel usage
- [ ] Document updated Discord setup and maintenance
  - Update `alert-manager.md` with new channel structure, usage, and troubleshooting
  - Add to todo: "Document alerting system maintenance procedures" – include channel management and health monitoring guides
  - **Effort**: 1 hour. **Dependencies**: Implementation of above tasks. **Success**: Docs reflect changes; easy to maintain

#### Low Priority (1 week+ - Optimization and cleanup)
- [ ] Clean up code and tests for removed general channel
  - Audit and remove general channel references in orchestrator and handlers
  - Update unit tests (e.g., `test_trade_alerts_and_ranking.py`) to reflect new structure (no general fallback assertions)
  - Link to todo: "Review Import Dependencies" – clean imports while doing this
  - **Effort**: 1 hour. **Dependencies**: High-priority removals. **Success**: Codebase has no general channel mentions; tests pass
- [x] Test hybrid monitoring with potential Langfuse addition
  - If pivoting to Langfuse (as discussed previously), integrate traces to feed into health channel (e.g., summary metrics)
  - Run integration tests with alerting enabled (link to todo: "Run comprehensive system integration test")
  - **Effort**: 3-4 hours. **Dependencies**: Langfuse setup (optional). **Success**: Monitoring data flows to health channel; no conflicts with Discord

### Langfuse Integration for Agent Monitoring
#### Phase 1: Environment Setup (1-2 hours)
- [x] Install Langfuse Python SDK: Add `langfuse` to `requirements.txt` and run `pip install langfuse`. (**Effort**: 10min; **Deps**: None; **Success**: `import langfuse` works.)
- [x] Create Langfuse account: Sign up at langfuse.com and get API keys (public_key, secret_key). (**Effort**: 15min; **Deps**: None; **Success**: Keys obtained.)
- [x] Set up local/cloud instance: For development, use cloud; for production, consider self-hosted Docker. (**Effort**: 20min; **Deps**: Docker if self-hosting; **Success**: Dashboard accessible.)
- [x] Create config file: Add `config/langfuse_config.yaml` with keys (public_key, secret_key, host="https://cloud.langfuse.com" or local URL). (**Effort**: 10min; **Deps**: None; **Success**: File loads in Python.)
- [x] Test connection: Create temp script `test_langfuse.py` with client init and basic trace; run it. (**Effort**: 15min; **Deps**: SDK; **Success**: Traces appear in dashboard.)

#### Phase 2: Integrate Langfuse into Base Agent (2-3 hours)
- [x] Update `src/agents/base.py`: Import LangfuseCallbackHandler; initialize in `__init__` with config; add to LLM calls. (**Effort**: 45min; **Deps**: Phase 1; **Success**: Basic traces logged for LLM interactions.)
- [x] Add span decorators: Wrap key methods like `_process_input` with `@langfuse.span(name="AgentProcess")` for detailed tracing. (**Effort**: 30min; **Deps**: Base integration; **Success**: Method-level traces visible.)
- [x] Enhance tracing: Add metadata (agent role, input size, processing time) to spans. (**Effort**: 30min; **Deps**: Span decorators; **Success**: Rich trace data in dashboard.)
- [x] Integrate with memory ops: Trace memory reads/writes in `advanced_memory.py`. (**Effort**: 30min; **Deps**: Base tracing; **Success**: Memory operations tracked.)

#### Phase 3: Multi-Agent Tracing and Monitoring (2-4 hours)
- [x] Add A2A protocol tracing: In `a2a_protocol.py`, trace message sends/receives with sender/receiver metadata. (**Effort**: 45min; **Deps**: Phase 2; **Success**: Inter-agent communication traced.)
- [x] Workflow orchestration tracing: In `live_workflow_orchestrator.py`, trace phase executions and agent responses. (**Effort**: 45min; **Deps**: A2A tracing; **Success**: Full workflow traces.)
- [x] Consensus polling traces: In `consensus_poller.py`, trace poll creation, voting, and resolution. (**Effort**: 30min; **Deps**: Workflow tracing; **Success**: Consensus decisions logged.)
- [x] Error and alert integration: Link Langfuse traces to AlertManager alerts for correlation. (**Effort**: 30min; **Deps**: Alert system; **Success**: Traces include alert context.)

#### Phase 4: Dashboard and Analytics (1-2 hours)
- [x] Set up custom metrics: Track agent performance (response time, success rate, token usage). (**Effort**: 30min; **Deps**: Phase 3; **Success**: Metrics dashboard populated.)
- [x] Create monitoring views: Use Langfuse dashboard for agent health, error patterns, and optimization opportunities. (**Effort**: 30min; **Deps**: Metrics; **Success**: Visual insights available.)
- [x] Integrate with health channel: Send summary metrics from Langfuse to Discord health channel periodically. (**Effort**: 30min; **Deps**: Discord health monitoring; **Success**: Hybrid monitoring active.)

#### Phase 5: Testing and Validation (1-2 hours)
- [x] Unit tests: Mock Langfuse client in agent tests; verify traces are sent. (**Effort**: 30min; **Deps**: Phase 2; **Success**: Tests pass with tracing.)
- [x] Integration tests: Run workflows and verify end-to-end traces in dashboard. (**Effort**: 45min; **Deps**: Phase 3; **Success**: Complete traces for full workflows.)
- [x] Performance validation: Ensure tracing doesn't impact agent response times significantly. (**Effort**: 30min; **Deps**: Integration tests; **Success**: <5% performance overhead.)

#### Phase 6: Rollout and Monitoring (Ongoing)
- [ ] Deploy to production: Update deployment scripts with Langfuse config. (**Effort**: 30min; **Deps**: All phases; **Success**: Production traces active.)
- [ ] Monitor and iterate: Weekly review traces for agent improvements; alert on anomalies. (**Effort**: Ongoing; **Success**: Continuous optimization.)

### Code Quality Improvements
- [ ] Implement consistent error logging
- [ ] Update API documentation for all components
- [ ] Add Integration Test Suite
- [ ] Set up automated integration testing
- [ ] Create test environments mirroring production
- [ ] Implement continuous integration for integration tests
- [ ] Review Import Dependencies
- [ ] Audit all import statements for consistency
- [ ] Remove unused imports and update deprecated patterns
- [ ] Implement import linting rules

### 📊 Current Stability Rating: **GREEN (Paper Trading Active)**
**Status**: All systems operational. IBKR connection established and validated.

**Risks if proceeding to paper trading now:**
- Network connectivity required for market data
- Monitor initial trades closely for any edge cases

**Recommended path to production:**
1. Complete IBKR documentation and interaction diagrams ✅
2. Implement and validate integration tests ✅
3. Run comprehensive validation testing ✅
4. **Start TWS and run paper trading** ✅
5. Monitor initial trades and system behavior (ongoing)

**Paper Trading Status: ACTIVE**


### Consensus Workflow Polling Implementation (High Priority)
**Goal:** Implement polling for consensus workflow with Discord visibility.

#### Discord Features
- [ ] Reactions for actions

### Code Quality Improvements
- [ ] Implement consistent error logging
- [ ] Update API documentation for all components
- [ ] Add Integration Test Suite
- [ ] Set up automated integration testing
- [ ] Create test environments mirroring production
- [ ] Implement continuous integration for integration tests
- [ ] Review Import Dependencies
- [ ] Audit all import statements for consistency
- [ ] Remove unused imports and update deprecated patterns
- [ ] Implement import linting rules

### Testing & Validation
- [ ] Test paper trading integration with IBKR/TWS
- [ ] Add test coverage for error scenarios and edge cases

## Integration & Architecture Issues

### IBKR Implementation Consistency
- [ ] Test migration in staging environment before production rollout
- [ ] Add import path validation to CI/CD pipeline
- [ ] Evaluate Bridge vs Direct Connector
- [ ] Compare performance benchmarks between implementations
- [ ] Assess maintenance overhead and complexity
- [ ] Make architectural decision with clear rationale documented

### Testing & Integration Gaps
- [ ] Implement proper test fixtures and cleanup
- [ ] Add test coverage for error scenarios and edge cases



## Organizational & Practical Recommendations

### Code Organization & Architecture
- [ ] Create Component Ownership Guidelines
- [ ] Implement API Versioning Strategy
- [ ] Standardize Configuration Management
- [ ] Establish Code Review Checklist

### Development Workflow Improvements
- [ ] Set Up Automated Dependency Updates
- [ ] Create Development Environment Setup Script
- [ ] Implement Feature Flags
- [ ] Establish Performance Benchmarks

### Testing & Quality Assurance
- [ ] Implement Test Data Management
- [ ] Add Integration Test Automation
- [ ] Create Chaos Engineering Tests
- [ ] Establish Code Coverage Requirements

### Monitoring & Observability
- [ ] Implement Structured Logging
- [ ] Add Business Metrics Tracking
- [ ] Create Alert Escalation Policies
- [ ] Set Up Log Aggregation

### Security & Compliance
- [ ] Implement Secret Rotation
- [ ] Add Security Headers
- [ ] Create Security Testing Pipeline
- [ ] Establish Incident Response Plan

### Deployment & Operations
- [ ] Implement Blue-Green Deployments
- [ ] Create Runbook Documentation
- [ ] Set Up Automated Backups
- [ ] Establish Disaster Recovery Plan

### Team Productivity & Collaboration
- [ ] Create Onboarding Documentation
- [ ] Implement Knowledge Base
- [ ] Set Up Regular Architecture Reviews
- [ ] Establish Tech Debt Budget

### Performance & Scalability
- [ ] Implement Caching Strategy
- [ ] Set Up Horizontal Scaling
- [ ] Optimize Database Queries
- [ ] Add Rate Limiting

### Maintenance & Sustainability
- [ ] Create Component Health Checks
- [ ] Set Up Automated Cleanup
- [ ] Establish Upgrade Path
- [ ] Implement Feature Usage Tracking

## Priority & Implementation Timeline

### High Priority - Paper Trading Preparation
- [ ] Validate circuit breaker and alert systems in failure scenarios
- [ ] Document component interactions and create troubleshooting diagrams

### Medium Priority (1-3 months)
- [ ] Resolve IBKR implementation architecture decision
- [ ] Implement comprehensive testing suite
- [ ] Add performance monitoring and optimization
- [ ] Create deployment automation and CI/CD improvements

### Low Priority (3-6 months)
- [ ] Implement advanced features (real-time streaming, ML enhancements)
- [ ] Security hardening and compliance improvements
- [ ] Scalability enhancements and cloud migration
- [ ] Advanced monitoring and analytics

### Ongoing Maintenance (Monthly)
- [ ] Regular security updates and dependency management
- [ ] Performance monitoring and optimization
- [ ] Documentation updates and knowledge base maintenance
- [ ] Code quality reviews and technical debt reduction

## Future Development Roadmap

### Phase 1: Paper Trading Readiness (Current Focus - December 2025)
- [ ] Complete IBKR integration validation and testing
- [ ] Implement comprehensive integration test suite
- [ ] Validate monitoring and alerting in realistic scenarios
- [ ] Document all component interactions and failure modes
- [ ] Establish paper trading environment and monitoring

### Phase 2: Performance & Scale (Q1 2026)
- [ ] Implement real-time data streaming from multiple sources
- [ ] Add advanced caching and performance optimization
- [ ] Implement horizontal scaling capabilities
- [ ] Add advanced analytics and reporting

### Phase 3: Intelligence & Automation (Q2 2026)
- [ ] Integrate machine learning models for trade prediction
- [ ] Implement automated trading strategies
- [ ] Add natural language processing for market analysis
- [ ] Create predictive maintenance for system components

### Phase 4: Enterprise Features (Q3-Q4 2026)
- [ ] Multi-asset class support (crypto, forex, commodities)
- [ ] Advanced risk management and portfolio optimization
- [ ] Regulatory compliance automation
- [ ] Enterprise integration APIs and webhooks

### Research & Innovation
- [ ] Explore blockchain integration for trade settlement
- [ ] Investigate quantum computing applications for optimization
- [ ] Research advanced AI techniques (reinforcement learning, GANs)
- [ ] Explore decentralized trading protocols and DeFi integration

### Community & Ecosystem
- [ ] Open source component contributions
- [ ] API marketplace for third-party integrations
- [ ] Educational content and developer resources
- [ ] Partnership development with financial institutions

## Learning Agent and Acontext Integration

### Phase 1: Environment Setup (1-2 hours)
- [ ] Install Acontext CLI: Run `curl -fsSL https://install.acontext.io | sh` in terminal. (**Effort**: 10min; **Deps**: None; **Success**: CLI available via `acontext --help`.)
- [ ] Create dedicated project dir: `mkdir acontext_learning && cd acontext_learning`. (**Effort**: 5min; **Deps**: None; **Success**: Dir exists.)
- [ ] Start Acontext backend: Run `acontext docker up` (ensure Docker running; set OpenAI key in `.env`). (**Effort**: 20min; **Deps**: Docker, OpenAI key; **Success**: API pings at `http://localhost:8029/api/v1`; Dashboard at `http://localhost:3000` loads.)
- [x] Install Python SDK: Add `acontext` to `requirements.txt` and run `pip install acontext`. (**Effort**: 10min; **Deps**: Python env; **Success**: `import acontext` works.)
- [ ] Test client init: Create temp script `test_client.py` with client init and `client.ping()`; run it. (**Effort**: 15min; **Deps**: SDK; **Success**: No errors, ping returns True.)
- [x] Create config file: Add `config/acontext_config.yaml` with keys (base_url, api_key, space_name="Trading-Learning-SOPs"). (**Effort**: 10min; **Deps**: None; **Success**: File loads in Python via yaml.safe_load.)
- [ ] Backup codebase: `git add . && git commit -m "Pre-Acontext integration" && git checkout -b feature/acontext-learning`. (**Effort**: 5min; **Deps**: Git; **Success**: Branch created.)

### Phase 2: Integrate Acontext into Learning Agent (1-2 days)
- [x] Update `__init__` in `src/agents/learning.py`: Import AcontextClient; init client from config; create Space (`self.learning_space = client.spaces.create(...)`); store `self.learning_space_id`. Add try/except for fallback. (**Effort**: 30min; **Deps**: Phase 1; **Success**: Init runs without errors; Space created visible in Dashboard.)
- [x] Add session logging in `_process_input`: After processing logs, create session (`client.sessions.create(space_id=...)`); send each log as message (`send_message(..., format="openai")`); call `flush(session.id)`. Wrap flush in async if needed. (**Effort**: 1hr; **Deps**: Client init; **Success**: Logs appear in session via Dashboard; tasks extracted.)
- [x] Enhance SOP storage in `_generate_combined_directives`: After directives, check convergence; if met, build SOP dict (use_when, preferences, tool_sops from tools used); create block (`client.spaces.blocks.create(..., path=f"/optimizations/{use_when}")`). (**Effort**: 1hr; **Deps**: Session logging; **Success**: SOP block appears in Space; queryable.)
- [x] Add artifact upload for ML/backtests: In `run_backtest_simulation` and `train_strategy_predictor`, after results, create Disk (`client.disks.create()`); upsert artifact (`client.disks.artifacts.upsert(..., FileUpload(filename="results.json", content=json.dumps(results)))`); reference ID in SOP. (**Effort**: 45min; **Deps**: SOP storage; **Success**: Artifacts downloadable from Dashboard; ID stored in memory.)
- [x] Integrate SOP query: In `_generate_combined_directives` (before LLM), build query from convergence/sd_variance; search (`client.spaces.experience_search(..., mode="agentic")`); if results, enrich directives (e.g., multiply value by sop.efficiency_multiplier; add 'sop_enhanced': True). Cache top 5 in `self.memory['sop_cache']`. (**Effort**: 1hr; **Deps**: All above; **Success**: Directives include SOP data; fallback if no results.)
- [x] Add health check: In `_generate_combined_directives`, `if hasattr(self, 'acontext_client') and self.acontext_client.ping():` proceed; else fallback. Log errors to `self.memory['acontext_errors']`. (**Effort**: 20min; **Deps**: Query; **Success**: Graceful fallback on API down.)
- [x] Refactor realtime: In `process_realtime_data`, after insights, log to session (batch 3-5 calls before flush). (**Effort**: 30min; **Deps**: Session logging; **Success**: Realtime logs in sessions.)

### Phase 3: Enhance Propagation via A2A (1 day)
- [x] Enrich directives: In `_generate_combined_directives`, add to each: `'sop_id': search_results[0].id if results else None, 'applies_to': self._get_applies_to(directive['refinement']), 'source': 'acontext_learned' if sop_enhanced else 'internal'`. Define `_get_applies_to` using agent_scopes (e.g., 'sizing_lift' → ['strategy', 'execution']). (**Effort**: 45min; **Deps**: Phase 2; **Success**: Directives have metadata.)
- [x] Update `distribute_realtime_insights`: In loop, filter `if recipient in directive['applies_to']:`; append to `a2a_message['content']['directives']`; send via `self.a2a_protocol.send_message`. Prioritize high-confidence (e.g., if confidence >0.8, immediate). (**Effort**: 1hr; **Deps**: Enrich; **Success**: Mock send logs filtered recipients.)
- [x] Add `apply_directive` to `src/agents/base.py`: Parse directive; if source=='acontext_learned' and role in applies_to, validate (e.g., `self.validate_directive(directive)` → check value < threshold); apply (e.g., `self.configs[refinement] = value`); return True/False. (**Effort**: 45min; **Deps**: None; **Success**: Base test: apply mock directive updates config.)
- [x] Per-agent receivers: In strategy.py/risk.py/execution.py/reflection.py, in `process_input` or A2A handler: `for directive in message['content']['directives']: self.apply_directive(directive)`. Override `validate_directive` (e.g., risk: if 'risk' in refinement and value >1.2: return False). (**Effort**: 1hr total, 15min/agent; **Deps**: Base apply; **Success**: Each applies relevant directives.)
- [x] Queue low-priority: In distribution, if priority=='low', add to `self.low_priority_queue`; add `process_queued_insights` call in orchestrator loop. (**Effort**: 30min; **Deps**: Update distribution; **Success**: Queued items processed on timer.)

### Phase 4: Testing and Validation (1-2 days)
- [x] Unit tests for learning.py: In `unit-tests/test_learning_agent.py`, mock AcontextClient (patch methods); test init (Space created), logging (messages sent), query (enrich directives), fallback (no client → baseline). (**Effort**: 2hr; **Deps**: Phase 2; **Success**: 90% coverage; pytest passes.)
- [x] Integration tests: New `integration-tests/test_acontext_learning.py` – Local Acontext up; simulate logs → verify session/SOP → enriched directive → mock A2A send. Test realtime with fake data. (**Effort**: 3hr; **Deps**: Docker; **Success**: End-to-end: SOP created, queried, propagated.)
- [x] Cross-agent tests: Mock A2A in `test_live_workflow_orchestrator.py`; assert strategy configs updated from learning directive. Test veto (risk rejects high value). (**Effort**: 2hr; **Deps**: Phase 3; **Success**: Orchestrator loop applies without errors.)
- [x] Edge case tests: No internet (fallback), timeout (retry flush 3x), invalid SOP (ignore, log). Stress: 100 logs → no crash. (**Effort**: 1hr; **Deps**: Unit; **Success**: Handles failures gracefully.)
- [ ] Manual validation: Run local workflow; check Dashboard (sessions have tasks, Space has SOPs); verify propagation (logs show applies_to filtering). (**Effort**: 1hr; **Deps**: All tests; **Success**: Manual run: Directive from Acontext improves mock Sharpe.)

### Phase 5: Rollout and Monitoring (Ongoing, 4-6 hours initial)
- [ ] Deploy config: Update `setup/setup_live_trading.py` to docker-compose Acontext (with env vars for keys). (**Effort**: 30min; **Deps**: Phase 1; **Success**: Prod start includes Acontext.)
- [ ] Merge and deploy: `git merge feature/acontext-learning`; update `deploy-to-vultr.ps1` for Vultr (Docker image). (**Effort**: 20min; **Deps**: Tests; **Success**: Deploys without errors.)
- [ ] Add monitoring: In `src/utils/alert_manager.py`, alert on Acontext errors (e.g., ping fail); track metrics in learning (`self.memory['acontext_metrics']['hit_rate']`). Integrate with reflection for post-apply eval. (**Effort**: 1hr; **Deps**: Phase 2; **Success**: Alerts fire on mock failure.)
- [ ] Initial rollout test: Paper trading mode; monitor 1-2 days (SOP creation rate, propagation success). (**Effort**: 2hr + monitoring; **Deps**: Deploy; **Success**: No regressions; >50% hit rate.)
- [ ] Iteration: Weekly review logs/Dashboard; refine (e.g., add multi-modal if needed). (Ongoing; **Success**: Sustained improvements, e.g., 10% better directives.)

## Hierarchical Agent Architecture (Future Implementation - On Hold)

### Phase 1: Base Infrastructure Setup (2-3 hours)
- [ ] Add `HIERARCHICAL` mode to `WorkflowMode` enum in `src/agents/unified_workflow_orchestrator.py` with proposal-based workflow. (**Effort**: 30min; **Deps**: None; **Success**: New mode selectable without breaking existing modes.)
- [ ] Create `src/agents/hierarchical_base.py` with `ProposalManager` class for specialist-to-leader proposal routing and `ConfigManager` for leader config modifications. (**Effort**: 1hr; **Deps**: None; **Success**: Classes import without errors; basic proposal routing works.)
- [ ] Extend A2A protocol message types in `src/utils/a2a_protocol.py` with `trade_proposal`, `config_update`, and `performance_feedback` messages. (**Effort**: 45min; **Deps**: None; **Success**: New message types handled by protocol without breaking existing communication.)

### Phase 2: Domain and Leader Group Implementation (3-4 hours)
- [ ] Create `src/agents/domain_groups.py` with base `DomainGroup` class managing 5 sector groups (tech, healthcare, commodities, finance, energy) using existing agent classes. (**Effort**: 1.5hr; **Deps**: Phase 1; **Success**: Domain groups initialize with existing agents; sector-specific data filtering works.)
- [ ] Add `src/agents/leader_group.py` with `LeaderGroup` class managing 4 leader agents with config modification capabilities. (**Effort**: 1.5hr; **Deps**: Phase 1; **Success**: Leader group initializes; config modification methods work without affecting core configs.)

### Phase 3: Configuration and Performance Tracking (2-3 hours)
- [ ] Create `config/hierarchical_config.yaml` with sector definitions, minimum allocations (5%), and performance tracking settings. (**Effort**: 45min; **Deps**: None; **Success**: Config loads properly; validation passes.)
- [ ] Add `performance_history` field to track experiential learning for future specialization. (**Effort**: 30min; **Deps**: Phase 2; **Success**: Performance data stored and retrievable.)
- [ ] Implement proposal-based communication flow (specialists propose → leaders review → approval/rejection). (**Effort**: 1hr; **Deps**: Phase 1; **Success**: Proposals route correctly through hierarchy.)

### Phase 4: Testing and Validation (2-3 hours)
- [ ] Add simple proposal testing in `integration-tests/test_hierarchical_proposals.py` validating proposal submission and leader review without full execution. (**Effort**: 1hr; **Deps**: Phase 3; **Success**: Proposal flow tests pass; no conflicts with existing system.)
- [ ] Validate no conflicts with current unified workflow orchestrator and agent scopes. (**Effort**: 45min; **Deps**: All phases; **Success**: Existing workflows continue functioning; hierarchical mode isolated.)
- [ ] Test config management isolation (leader modifications don't affect core system configs). (**Effort**: 45min; **Deps**: Phase 2; **Success**: Config changes contained to hierarchical components.)

### Phase 5: Rollout Preparation (1-2 hours - On Hold)
- [ ] Document hierarchical architecture integration points and extension patterns. (**Effort**: 45min; **Deps**: All phases; **Success**: Clear documentation for future implementation.)
- [ ] Create migration path from current unified orchestrator to hierarchical system. (**Effort**: 45min; **Deps**: All phases; **Success**: Clear upgrade strategy documented.)
