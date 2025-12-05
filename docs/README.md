# ABC-Application Documentation

## 📚 Documentation Overview

This directory contains comprehensive documentation for the ABC-Application AI Portfolio Manager system. The documentation is organized into logical sections for easy navigation and maintenance.

## 📁 Documentation Structure

```
docs/
├── README.md                    # This file - documentation navigation
├── architecture.md             # System architecture and design
├── AGENTS/                     # Agent documentation
│   ├── index.md               # Agent inventory and coordination
│   └── [individual agents]    # Agent-specific documentation
├── FRAMEWORKS/                # Technical frameworks
│   ├── macro-micro-analysis-framework.md
│   ├── langchain-integration.md
│   ├── a2a-protocol.md
│   └── memory-systems.md
├── IMPLEMENTATION/            # Setup and deployment
│   ├── setup-and-development.md # Setup and development guide
│   ├── configuration.md      # Configuration management
├── REFERENCE/                 # Operational reference
│   ├── api-monitoring.md     # API health monitoring
│   ├── performance.md        # Performance optimization
│   └── troubleshooting.md    # Common issues and solutions
├── security_hardening_guide.md    # Security hardening guide
├── production_readiness_checklist.md # Production deployment checklist
└── workflows.md               # Workflow documentation
```

## 🚀 Quick Start

### For New Developers
1. **[Setup and Development Guide](./IMPLEMENTATION/setup-and-development.md)**: Installation, configuration, and development workflow
2. **[Architecture Overview](./architecture.md)**: Understanding the system design
3. **[Agent Framework](./AGENTS/index.md)**: How agents work together

### For Contributors
1. **[Setup and Development Guide](./IMPLEMENTATION/setup-and-development.md)**: Development processes and standards
2. **[Configuration](./IMPLEMENTATION/configuration.md)**: Configuration management
3. **[Testing](./IMPLEMENTATION/testing.md)**: Testing strategies and practices

### For Operators
1. **[Production Deployment](./production_readiness_checklist.md)**: Production deployment and readiness
2. **[Security Hardening](./security_hardening_guide.md)**: Security best practices
3. **[Monitoring](./REFERENCE/api-monitoring.md)**: System monitoring and health checks
4. **[Troubleshooting](./REFERENCE/troubleshooting.md)**: Common issues and solutions

## 📖 Key Topics

### System Architecture
- **[Macro-Micro Framework](./FRAMEWORKS/macro-micro-analysis-framework.md)**: Analysis methodology
- **[A2A Protocol](./FRAMEWORKS/a2a-protocol.md)**: Agent communication
- **[Memory Systems](./FRAMEWORKS/memory-systems.md)**: Data persistence and sharing

### Agent System
- **[Agent Coordination](./AGENTS/index.md)**: How agents collaborate
- **[Base Agent](./AGENTS/base.md)**: Agent architecture and interfaces
- **[Specialized Agents](./AGENTS/)**: Individual agent capabilities

### Operations
- **[API Health Monitoring](./REFERENCE/api-monitoring.md)**: External service monitoring
- **[Security](./REFERENCE/security.md)**: Security best practices
- **[Performance](./REFERENCE/performance.md)**: Optimization techniques

## 🔧 Development Standards

### Documentation Conventions
- Use consistent heading hierarchy (H1 → H2 → H3)
- Include code examples with syntax highlighting
- Cross-reference related documentation
- Keep examples up-to-date with current APIs

### File Organization
- Group related documents in subdirectories
- Use descriptive filenames with kebab-case
- Include table of contents in longer documents
- Update navigation when adding new documents

### Content Guidelines
- Write for the target audience (developers/operators/users)
- Include practical examples and code snippets
- Document limitations and known issues
- Keep information current with code changes

## 🤝 Contributing to Documentation

### Adding New Documentation
1. Choose the appropriate subdirectory based on content type
2. Follow naming conventions and include front matter
3. Add cross-references to related documents
4. Update this README and any relevant navigation

### Updating Existing Documentation
1. Check for outdated information or examples
2. Update code examples to match current APIs
3. Add new sections for new features
4. Review and update cross-references

### Documentation Reviews
- Technical accuracy
- Clarity and readability
- Completeness of examples
- Navigation and cross-references

## 📋 Documentation Status

| Section | Status | Coverage |
|---------|--------|----------|
| Architecture | ✅ Complete | System design, data flows, frameworks |
| Agents | 🟡 Partial | Base framework documented, individual agents need updates |
| Implementation | 🟡 Partial | Setup and development covered, deployment needs expansion |
| Reference | 🟡 Partial | API monitoring documented, security and troubleshooting partial |
| Workflows | ❌ Missing | Workflow documentation needs creation |

## 🔗 External Resources

- [Main README](../README.md): Project overview and quick start
- [API Documentation](../src/): Inline code documentation
- [GitHub Issues](https://github.com/nvickers04/ABC-Application/issues): Bug reports and feature requests
- [GitHub Wiki](https://github.com/nvickers04/ABC-Application/wiki): Additional documentation

---

*For questions about documentation or suggestions for improvement, please create an issue or pull request.*