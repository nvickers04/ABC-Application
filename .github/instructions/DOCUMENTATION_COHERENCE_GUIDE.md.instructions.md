---
applyTo: '*ABC-Application*'
---
# Documentation Coherence Guide

## Maintaining Coherence Between .md Files and Code

### Philosophy
**Documentation is the source of truth** - Code implements the documented architecture, not the other way around. When documentation and code diverge, update the documentation first, then align the code.

### Development Workflow

#### 1. **Planning Phase** (Documentation-Driven)
- **Read relevant `.md` files first** to understand intended architecture
- Understand agent roles, data flows, and integration points from docs
- Identify integration points with existing agents
- Review existing instruction files for consistency

#### 2. **Implementation Phase** (Documentation-Aligned)
- Implement code following documented patterns and standards
- Use established naming conventions and structures from docs
- Follow async patterns, error handling, and logging standards
- Maintain file organization rules from FILE_ORGANIZATION_GUIDE

#### 3. **Testing Phase** (Verification)
- Unit tests in `unit-tests/` directory
- Integration tests in `integration-tests/` directory
- Verify code matches documented behavior
- Test documentation examples work

#### 4. **Documentation Update Phase** (Immediate Updates)
- **Update `.md` files immediately** after code implementation
- Ensure all new features are documented
- Update cross-references and examples
- Verify coherence between docs and code

## Documentation Update Triggers

### When Implementing New Code:
- [ ] **Architecture docs** updated (`docs/architecture.md`)
- [ ] **Agent capabilities** documented (`docs/AGENTS/[agent].md`)
- [ ] **Integration points** documented (`docs/FRAMEWORKS/`)
- [ ] **Setup instructions** updated (`docs/IMPLEMENTATION/`)

### When Modifying Existing Code:
- [ ] **API changes** documented
- [ ] **Configuration changes** reflected in docs
- [ ] **Breaking changes** clearly marked
- [ ] **Migration guides** provided if needed

### When Adding New Features:
- [ ] **Performance Impact**: Document any performance changes or optimizations
- [ ] **Privacy Considerations**: Include GDPR compliance notes for data handling

### Performance Documentation
- Update `docs/architecture.md` with performance benchmarks
- Document async patterns and caching strategies
- Include profiling results in implementation docs

### Privacy Documentation
- Add data flow diagrams showing personal data handling
- Document consent mechanisms and retention policies
- Include breach response procedures in security guides
- [ ] **Feature overview** added to relevant docs
- [ ] **Usage examples** provided
- [ ] **Configuration options** documented
- [ ] **Troubleshooting** section updated

## Documentation Standards

### File Structure
```
docs/
├── README.md              # Navigation and overview
├── architecture.md        # System design and components
├── ai-reasoning-agent-collaboration.md  # AI framework
├── macro-micro-analysis-framework.md    # Analysis methodology
├── production_readiness_checklist.md    # Deployment checklist
├── security_hardening_guide.md          # Security measures
├── AGENTS/                # Agent-specific documentation
├── FRAMEWORKS/            # Framework and integration guides
│   ├── langchain-integration.md
│   ├── a2a-protocol.md
│   ├── memory-systems.md
│   └── workflows/        # Workflow documentation
├── IMPLEMENTATION/        # Setup, configuration, deployment
│   ├── setup-and-development.md
│   ├── configuration.md
│   ├── discord-setup.md
│   ├── ibkr-deployment.md
│   └── vultr-deployment.md
└── REFERENCE/             # API docs, troubleshooting, examples
    ├── api-monitoring.md
    └── troubleshooting.md
```

### Content Standards

#### Code Examples
- **Must be executable** - Test all code snippets
- **Use correct imports** - Update after file reorganization
- **Include error handling** - Show proper exception handling
- **Be version-specific** - Note Python/asyncio versions

#### File References
- **Use absolute paths** from project root
- **Update after reorganization** - Fix all path references
- **Include file purposes** - Explain what each file does

#### Architecture Diagrams
- **Keep updated** with code changes
- **Use consistent notation** (UML, flowcharts, etc.)
- **Include data flows** and integration points

## Code-to-Documentation Mapping

### Agent Implementation
```python
# Code: src/agents/macro_agent.py
class MacroAgent(BaseAgent):
    def analyze_economy(self):
        # Implementation

# Documentation: docs/AGENTS/macro-agent.md
## Macro Agent
### Capabilities
- Economic analysis
- Market regime identification
### Methods
- `analyze_economy()` - Performs macroeconomic analysis
```

### Framework Integration
```python
# Code: src/utils/a2a_protocol.py
class A2AProtocol:
    def send_message(self, agent_id, message):
        # Implementation

# Documentation: docs/FRAMEWORKS/a2a-protocol.md
## Agent-to-Agent Protocol
### Purpose
Enables communication between agents
### Methods
- `send_message(agent_id, message)` - Sends message to agent
```

### Configuration
```yaml
# Config: config/risk-constraints.yaml
max_drawdown: 0.05
var_confidence: 0.95

# Documentation: docs/IMPLEMENTATION/configuration.md
## Risk Configuration
### max_drawdown
Maximum allowed portfolio drawdown (0.05 = 5%)
### var_confidence
Confidence level for VaR calculations (0.95 = 95%)
```

## Quality Assurance Checklist

### Documentation Completeness
- [ ] All public APIs documented
- [ ] Configuration options explained
- [ ] Setup instructions accurate
- [ ] Troubleshooting guides current
- [ ] Code examples executable

### Documentation Accuracy
- [ ] File paths correct after reorganization
- [ ] Code snippets work
- [ ] Architecture diagrams match code
- [ ] Cross-references valid
- [ ] Version information current

### Documentation Consistency
- [ ] Naming conventions consistent
- [ ] Terminology standardized
- [ ] Formatting uniform
- [ ] Style guide followed

## Maintenance Schedule

### Daily
- Update docs for any code changes
- Verify documentation examples work
- Check for broken links/references

### Weekly
- Review documentation completeness
- Update architecture diagrams
- Audit cross-references

### Monthly
- Full documentation audit
- Update setup instructions
- Review and update examples

## Tools for Coherence

### Documentation Linting
```bash
# Check for broken links
find docs/ -name "*.md" -exec grep -l "\[.*\](\." {} \;

# Validate code examples
python tools/validate_docs.py
```

### Code-Doc Synchronization
```bash
# Generate API docs from code
python tools/generate_api_docs.py

# Check doc-code alignment
python tools/verify_coherence.py
```

### Automated Checks
- Pre-commit hooks for documentation updates
- CI/CD checks for documentation completeness
- Automated testing of documentation examples

## Common Issues and Solutions

### Issue: Documentation lags behind code
**Solution:** Make documentation updates part of the development process, not an afterthought

### Issue: Code examples don't work
**Solution:** Test all code snippets before committing documentation

### Issue: File reorganization breaks references
**Solution:** Update all documentation references immediately after moving files

### Issue: Architecture docs don't match implementation
**Solution:** Treat documentation as the design specification, update it first

## Benefits of Coherence

1. **Onboarding**: New developers can understand the system from docs
2. **Maintenance**: Documentation serves as implementation guide
3. **Quality**: Code reviews can verify against documented architecture
4. **Collaboration**: Team alignment on system design and patterns
5. **Evolution**: Documentation guides system evolution and refactoring

## Quick Reference

### Adding New Agent
1. **Document first**: Write `docs/AGENTS/new-agent.md`
2. **Implement code**: Create `src/agents/new_agent.py`
3. **Update architecture**: Add to `docs/architecture.md`
4. **Add tests**: Create `unit-tests/test_new_agent.py`
5. **Update setup**: Add to `docs/IMPLEMENTATION/setup-and-development.md`

### Modifying Existing Agent
1. **Check docs**: Read current documentation
2. **Update docs**: Document planned changes
3. **Implement code**: Make changes following docs
4. **Update examples**: Ensure code snippets work
5. **Verify coherence**: Check docs match code

This guide ensures that documentation and code evolve together, maintaining a coherent and maintainable system.