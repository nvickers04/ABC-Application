---
applyTo: '*ABC-Application*'
---
# Chat Development Guide - File Coherence & Development Workflow

## Overview
This guide governs our chat-based development interactions to ensure file coherence, consistent development practices, and alignment with existing instruction files. It focuses on development workflow and file consistency rather than mixing with application runtime components.

## Core Principles

### 1. **File Coherence First**
- **ALWAYS** check existing instruction files before implementing anything
- Maintain consistency across all `.github/instructions/*.md.instructions.md` files
- Reference existing guidelines from AI_DEVELOPMENT_INSTRUCTIONS, DOCUMENTATION_COHERENCE_GUIDE, and FILE_ORGANIZATION_GUIDE
- Update relevant `.md` files immediately when creating new code

### 2. **Development Workflow Standards**
- **Read docs first** to understand intended architecture before coding
- **Follow established patterns** from existing codebase
- **Update documentation immediately** after code changes
- **Verify coherence** between all related files

### 3. **Chat-Driven Development Standards**
- Use chat context to understand user intent before implementing
- Ensure all implementations align with documented architecture
- Maintain separation between development tools and application runtime

## File Consistency Verification

### Instruction File Cross-Reference
Before implementing any changes, verify consistency across all instruction files:

**AI_DEVELOPMENT_INSTRUCTIONS.md.instructions.md**
- [ ] Code quality standards followed
- [ ] Agent architecture compliance maintained
- [ ] File organization rules respected

**DOCUMENTATION_COHERENCE_GUIDE.md.instructions.md**
- [ ] Documentation updated for all code changes
- [ ] Code examples in docs are executable
- [ ] Cross-references between docs are valid

**FILE_ORGANIZATION_GUIDE.md.instructions.md**
- [ ] Files placed in correct directories
- [ ] No source code in root directory
- [ ] Import paths updated after moves

**CHAT_DEVELOPMENT_GUIDE.md.instructions.md** (this file)
- [ ] Development workflow followed
- [ ] File coherence maintained
- [ ] No conflicts with existing files

### Consistency Check Commands
```bash
# Check for misplaced Python files
find . -name "*.py" -not -path "./src/*" -not -path "./tools/*" -not -path "./unit-tests/*" -not -path "./integration-tests/*" -not -path "./simulations/*" -not -path "./myenv/*" | head -10

# Verify documentation file references
grep -r "\.\./\.\./" docs/ | grep -v node_modules | head -5

# Check for broken imports after file moves
python -c "import sys; sys.path.insert(0, 'src'); import ast; [print(f'BROKEN: {f}') for f in ['src/agents/*.py', 'src/utils/*.py'] if not ast.parse(open(f).read(), f)]"
```

## Development Workflow Standards

### Pre-Implementation Checklist
1. **Read All Instruction Files**: Review AI_DEVELOPMENT_INSTRUCTIONS, DOCUMENTATION_COHERENCE_GUIDE, FILE_ORGANIZATION_GUIDE
2. **Check Existing Code**: Search for similar functionality in current codebase
3. **Verify File Placement**: Confirm correct directory per FILE_ORGANIZATION_GUIDE
4. **Plan Documentation Updates**: Identify which `.md` files need updating

### Implementation Standards
- **Follow Established Patterns**: Use existing code as templates
- **Maintain Code Quality**: Apply standards from AI_DEVELOPMENT_INSTRUCTIONS
- **Update Documentation Immediately**: Modify docs alongside code per DOCUMENTATION_COHERENCE_GUIDE
- **Test Thoroughly**: Ensure changes work and don't break existing functionality

### Post-Implementation Verification
- **Run Consistency Checks**: Use commands above to verify file organization
- **Update Cross-References**: Ensure all docs reference correct file paths
- **Verify Examples**: Test any code examples in documentation
- **Check Imports**: Confirm all import statements work after changes

## Chat Interaction Standards

### Development Session Structure
1. **Context Gathering**: Review existing instruction files and codebase
2. **Intent Clarification**: Confirm user requirements align with existing architecture
3. **Implementation Planning**: Reference existing docs and instruction files
4. **Code Implementation**: Follow AI_DEVELOPMENT_INSTRUCTIONS standards
5. **Documentation Update**: Update docs per DOCUMENTATION_COHERENCE_GUIDE
6. **Verification**: Run consistency checks to ensure coherence

### File Consistency Verification
```bash
# Check for root-level Python files (should be none)
find . -maxdepth 1 -name "*.py" | grep -v __pycache__

# Verify all Python files are in correct directories
find src/ tools/ unit-tests/ integration-tests/ simulations/ -name "*.py" | wc -l

# Check documentation file references are current
grep -r "src/" docs/ | head -5
```

### When to Create vs Modify Files
- **Create New**: Only when functionality doesn't exist and follows FILE_ORGANIZATION_GUIDE
- **Modify Existing**: When extending current features or fixing bugs
- **Update Docs**: Always update relevant `.md` files immediately
- **Check Consistency**: Run verification commands after changes

## Integration with Existing Instructions

### AI_DEVELOPMENT_INSTRUCTIONS Compliance
- [ ] **Documentation-First**: Update `.md` files before/after code changes
- [ ] **File Organization**: Follow established directory structure
- [ ] **Code Quality**: Use type hints, docstrings, error handling
- [ ] **Agent Architecture**: Maintain A2A protocol and memory systems

### DOCUMENTATION_COHERENCE_GUIDE Compliance
- [ ] **Read Docs First**: Always check existing documentation
- [ ] **Update Immediately**: Modify docs alongside code changes
- [ ] **Verify Examples**: Test all code snippets in documentation
- [ ] **Maintain Cross-References**: Update all related documentation

### FILE_ORGANIZATION_GUIDE Compliance
- [ ] **No Root Code**: All Python files in appropriate subdirectories
- [ ] **Proper Placement**: Source in `src/`, docs in `docs/`, tools in `tools/`
- [ ] **Import Updates**: Fix import paths after any file moves
- [ ] **Documentation References**: Update file path references in docs

### Consistency Verification
- [ ] **Cross-File References**: All instruction files reference each other correctly
- [ ] **No Duplication**: Each guide covers unique aspects without overlap
- [ ] **Version Alignment**: All guides reflect current project structure
- [ ] **Path Accuracy**: All file paths in examples are correct

## Quality Assurance for Chat Sessions

### Pre-Implementation Checklist
- [ ] **Instruction Review**: Read all relevant instruction files
- [ ] **Documentation Review**: Check existing documentation for similar features
- [ ] **File Organization**: Verify correct directory placement per FILE_ORGANIZATION_GUIDE
- [ ] **Conflict Check**: Ensure no conflicts with existing files or functionality

### Post-Implementation Checklist
- [ ] **Documentation Updated**: All relevant `.md` files modified per DOCUMENTATION_COHERENCE_GUIDE
- [ ] **Code Quality**: Follows AI_DEVELOPMENT_INSTRUCTIONS standards
- [ ] **File Organization**: Maintains FILE_ORGANIZATION_GUIDE structure
- [ ] **Cross-References**: Update all documentation links and examples
- [ ] **Consistency Verified**: Run file consistency checks

## Common Chat Scenarios

### Scenario: User Wants New Feature
```
User: "Add feature X"
Response:
1. Check existing docs for similar features
2. Review existing instruction files for standards
3. Plan implementation following FILE_ORGANIZATION_GUIDE
4. Update docs immediately after code changes
5. Run consistency checks
```

### Scenario: User Reports Issues
```
User: "Something's not working"
Response:
1. Review error messages and logs
2. Check existing documentation for troubleshooting
3. Verify file organization and imports
4. Fix issues following development standards
5. Update docs if needed
```

### Scenario: File Organization Question
```
User: "Where should I put this file?"
Response:
1. Reference FILE_ORGANIZATION_GUIDE
2. Check existing similar files for patterns
3. Verify no root directory violations
4. Update documentation references
```

## File Consistency Maintenance

### Regular Verification Tasks
- **Daily**: Check for new files in incorrect locations
- **Weekly**: Verify all documentation references are current
- **Monthly**: Audit all instruction files for consistency

### Automated Checks
```bash
# Check for Python files outside approved directories
find . -name "*.py" -not -path "./src/*" -not -path "./tools/*" -not -path "./unit-tests/*" -not -path "./integration-tests/*" -not -path "./simulations/*" -not -path "./myenv/*" | grep -v __pycache__

# Verify documentation file references
grep -r "src/" docs/ | head -10

# Check for broken relative imports
python -c "
import os
import ast
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            try:
                ast.parse(open(path).read())
            except SyntaxError as e:
                print(f'SYNTAX ERROR: {path} - {e}')
"
```

## Quick Reference Commands

### File Organization Verification
```bash
# Check for misplaced files
find . -name "*.py" -not -path "./src/*" -not -path "./tools/*" -not -path "./unit-tests/*" -not -path "./integration-tests/*" -not -path "./simulations/*" -not -path "./myenv/*"

# Count files in correct directories
find src/ tools/ unit-tests/ integration-tests/ simulations/ -name "*.py" | wc -l

# Check documentation coherence
grep -r "\.\./" docs/ | grep -v node_modules | head -5
```

### Development Standards Check
- [ ] AI_DEVELOPMENT_INSTRUCTIONS followed
- [ ] DOCUMENTATION_COHERENCE_GUIDE maintained
- [ ] FILE_ORGANIZATION_GUIDE respected
- [ ] File consistency verified

## Code Labeling System for Enhanced Searchability

### Standardized Code Labels
To improve searchability across the messy codebase, implement these structured labels:

#### File Header Labels (Top of every Python file):
```python
# [LABEL:COMPONENT] [LABEL:AGENT] [LABEL:FRAMEWORK] [LABEL:FUNCTION]
# [LABEL:AUTHOR] [LABEL:UPDATED] [LABEL:REVIEWED]
#
# Purpose: [Brief description]
# Dependencies: [Key imports/modules]
# Related: [docs/file.md, other components]
```

#### Documentation Header Labels (Top of every .md file):
```markdown
---
[LABEL:DOC:category] [LABEL:DOC:topic] [LABEL:DOC:audience]
[LABEL:AUTHOR] [LABEL:UPDATED] [LABEL:REVIEWED]
---

# Document Title

## Purpose
[Brief description of document purpose]

## Related Files
- Code: [src/file.py]
- Config: [config/file.yaml]
- Tests: [unit-tests/test_file.py]
```

#### Function/Method Labels:
```python
def function_name(params):
    """
    [LABEL:AGENT:macro] [LABEL:FUNCTION:data_processing]
    [Brief description]
    """
```

#### Class Labels:
```python
class ClassName:
    """
    [LABEL:COMPONENT:a2a_protocol] [LABEL:AGENT:coordinator]
    [Description]
    """
```

### Label Categories

#### Component Labels:
- `[LABEL:COMPONENT:main_orchestrator]` - Main application entry
- `[LABEL:COMPONENT:a2a_protocol]` - Agent communication
- `[LABEL:COMPONENT:memory_system]` - Memory management
- `[LABEL:COMPONENT:data_pipeline]` - Data processing
- `[LABEL:COMPONENT:risk_engine]` - Risk calculations
- `[LABEL:COMPONENT:execution_engine]` - Trade execution

#### Agent Labels:
- `[LABEL:AGENT:macro]` - MacroAgent
- `[LABEL:AGENT:data]` - DataAgent  
- `[LABEL:AGENT:strategy]` - StrategyAgent
- `[LABEL:AGENT:risk]` - RiskAgent
- `[LABEL:AGENT:execution]` - ExecutionAgent
- `[LABEL:AGENT:reflection]` - ReflectionAgent
- `[LABEL:AGENT:learning]` - LearningAgent
- `[LABEL:AGENT:memory]` - MemoryAgent

#### Framework Labels:
- `[LABEL:FRAMEWORK:langchain]` - LangChain integration
- `[LABEL:FRAMEWORK:langgraph]` - StateGraph orchestration
- `[LABEL:FRAMEWORK:asyncio]` - Async operations
- `[LABEL:FRAMEWORK:redis]` - Caching/memory
- `[LABEL:FRAMEWORK:pydantic]` - Data validation
- `[LABEL:FRAMEWORK:ibkr]` - Trading API

#### Function Labels:
- `[LABEL:FUNCTION:data_ingestion]` - Data collection
- `[LABEL:FUNCTION:data_processing]` - Data transformation
- `[LABEL:FUNCTION:risk_calculation]` - Risk assessment
- `[LABEL:FUNCTION:strategy_generation]` - Strategy creation
- `[LABEL:FUNCTION:trade_execution]` - Order execution
- `[LABEL:FUNCTION:performance_analysis]` - Results evaluation
- `[LABEL:FUNCTION:memory_operations]` - Memory management

#### Documentation Labels:
- `[LABEL:DOC:architecture]` - System design and structure
- `[LABEL:DOC:agent_guide]` - Agent-specific documentation
- `[LABEL:DOC:framework]` - Framework and protocol guides
- `[LABEL:DOC:implementation]` - Setup and deployment guides
- `[LABEL:DOC:reference]` - API docs and troubleshooting
- `[LABEL:DOC:development]` - Development instructions and standards

#### Documentation Topic Labels:
- `[LABEL:DOC:topic:macro_micro]` - Macro-to-micro analysis framework
- `[LABEL:DOC:topic:a2a_protocol]` - Agent-to-agent communication
- `[LABEL:DOC:topic:memory_system]` - Memory management and coordination
- `[LABEL:DOC:topic:api_monitoring]` - API health and monitoring
- `[LABEL:DOC:topic:deployment]` - Deployment and production setup
- `[LABEL:DOC:topic:security]` - Security and compliance

#### Documentation Audience Labels:
- `[LABEL:DOC:audience:developer]` - For developers implementing features
- `[LABEL:DOC:audience:operator]` - For system operators and administrators
- `[LABEL:DOC:audience:architect]` - For system architects and designers
- `[LABEL:DOC:audience:user]` - For end users and stakeholders

### Implementation Strategy

#### Phase 1: Core Files (High Priority)
Apply labels to:
- `src/main.py` - Main orchestrator
- `src/utils/a2a_protocol.py` - Communication framework
- `src/agents/*.py` - All agent implementations
- `src/utils/tools.py` - Core utilities

#### Phase 2: Documentation Files (High Priority)
Apply labels to:
- `docs/architecture.md` - System architecture overview
- `docs/ai-reasoning-agent-collaboration.md` - AI framework guide
- `docs/macro-micro-analysis-framework.md` - Analysis methodology
- `docs/AGENTS/index.md` - Agent documentation index
- `docs/FRAMEWORKS/a2a-protocol.md` - Communication protocol
- `docs/IMPLEMENTATION/setup.md` - Setup and deployment

#### Phase 3: Supporting Files ✅ COMPLETED
Apply to:
- Configuration files ✅ COMPLETED
- Test files ✅ COMPLETED  
- Additional documentation files ✅ COMPLETED
- Tool scripts ✅ COMPLETED

#### Phase 4: Validation
- Run semantic searches with labels on both code and docs ✅ COMPLETED
- Verify label consistency across code and documentation ✅ COMPLETED
- Update search documentation ✅ COMPLETED

### Phase 3 Implementation Summary ✅

**Configuration Files Labeled:**
- `config/ibkr_config.ini` - [LABEL:CONFIG:trading_platform] [LABEL:CONFIG:ibkr] [LABEL:CONFIG:paper_trading]
- `config/risk-constraints.yaml` - [LABEL:CONFIG:risk_management] [LABEL:CONFIG:constraints] [LABEL:FRAMEWORK:yaml]
- `config/profitability-targets.yaml` - [LABEL:CONFIG:profitability] [LABEL:CONFIG:targets] [LABEL:FRAMEWORK:yaml]
- `config/langchain-integration.md` - [LABEL:DOC:framework] [LABEL:DOC:topic:langchain] [LABEL:DOC:audience:developer]

**Test Files Labeled:**
- `unit-tests/test_a2a_protocol.py` - [LABEL:TEST:a2a_protocol] [LABEL:TEST:unit] [LABEL:FRAMEWORK:pytest] [LABEL:FRAMEWORK:asyncio]
- `unit-tests/conftest.py` - [LABEL:TEST:config] [LABEL:FRAMEWORK:pytest] [LABEL:FRAMEWORK:pytest_asyncio]
- `pytest.ini` - [LABEL:TEST:config] [LABEL:FRAMEWORK:pytest] [LABEL:CONFIG:testing]

**Documentation Files Labeled:**
- `docs/ai-reasoning-agent-collaboration.md` - [LABEL:DOC:framework] [LABEL:DOC:topic:ai_reasoning] [LABEL:DOC:audience:architect]
- `docs/macro-micro-analysis-framework.md` - [LABEL:DOC:framework] [LABEL:DOC:topic:macro_micro] [LABEL:DOC:audience:developer]
- `docs/production_readiness_checklist.md` - [LABEL:DOC:deployment] [LABEL:DOC:topic:production] [LABEL:DOC:audience:architect]
- `docs/security_hardening_guide.md` - [LABEL:DOC:security] [LABEL:DOC:topic:security] [LABEL:DOC:audience:administrator]
- `docs/IMPLEMENTATION/setup.md` - [LABEL:DOC:implementation] [LABEL:DOC:topic:deployment] [LABEL:DOC:audience:administrator]

**Tool Scripts Labeled:**
- `tools/continuous_trading.py` - [LABEL:TOOL:trading] [LABEL:TOOL:continuous] [LABEL:FRAMEWORK:asyncio]
- `tools/start_live_workflow.py` - [LABEL:TOOL:workflow] [LABEL:TOOL:launcher] [LABEL:FRAMEWORK:discord]

### Search Enhancement Examples

#### Before Labels:
```
semantic_search("agent communication protocol")
```
Returns scattered results across multiple files

#### After Labels (Code):
```
semantic_search("[LABEL:COMPONENT:a2a_protocol]")
```
Returns precisely targeted code results

#### After Labels (Documentation):
```
semantic_search("[LABEL:DOC:topic:a2a_protocol]")
```
Returns precisely targeted documentation results

#### Complex Queries:
```
semantic_search("[LABEL:AGENT:strategy] [LABEL:FUNCTION:risk_calculation]")
```
Finds strategy agent risk calculation functions

```
semantic_search("[LABEL:DOC:framework] [LABEL:DOC:topic:memory_system]")
```
Finds framework documentation about memory systems

### Benefits for Messy Code

1. **Precision**: Find exactly what you need, not 50 scattered results
2. **Discovery**: Easy to explore related functionality across code and docs
3. **Maintenance**: Clear ownership and dependencies between code and documentation
4. **Documentation**: Self-documenting code and documentation structure
5. **Collaboration**: Team can quickly understand both code purpose and documentation
6. **Cross-References**: Easy to find code implementations from docs and vice versa

### Integration with Existing Tools

Labels work with all search tools:
- `semantic_search`: Natural language + label filtering (code and docs)
- `grep_search`: Pattern match on labels in any file type
- `list_code_usages`: Find usages of labeled components
- `file_search`: Find files by label patterns
- `read_file`: Read labeled files with context

This labeling system transforms messy code into a searchable, maintainable codebase.

This guide ensures our chat-based development maintains file coherence, prevents conflicts, and aligns with all existing instruction files while maintaining code quality and proper organization.</content>
<parameter name="filePath">c:\Users\nvick\ABC-Application\.github\instructions\CHAT_DEVELOPMENT_GUIDE.md.instructions.md