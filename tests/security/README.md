# Security Testing Suite

This directory contains comprehensive security testing and vulnerability assessment scripts for the ABC trading system.

## Overview

The security tests cover:
- Encryption key handling and data protection
- Injection attack prevention
- Authentication failure handling
- Automated vulnerability scanning
- CI/CD security integration

## Test Categories

### Encryption Security Tests
- `test_encryption_security.py` - Encryption key management, data protection, and cryptographic security

### Injection & Authentication Tests
- `test_injection_auth.py` - SQL injection, command injection, XSS, CSRF, and authentication failure prevention

### Automated Scanning
- `run_security_scan.py` - Comprehensive security scanner script
- `bandit_config.yaml` - Bandit security scanning configuration

## Running Security Tests

### Run All Security Tests
```bash
python -m pytest tests/security/ -v
```

### Run Specific Security Test
```bash
python -m pytest tests/security/test_encryption_security.py -v
```

### Run Automated Security Scan
```bash
python tests/security/run_security_scan.py
```

### Run with Security Scanning Tools
```bash
# Bandit static security analysis
bandit -c tests/security/bandit_config.yaml -r src/

# Safety dependency vulnerability check
safety check

# Combined security scan
python tests/security/run_security_scan.py --output security_report_$(date +%Y%m%d_%H%M%S).json
```

## Security Scanning Configuration

### Bandit Configuration (`bandit_config.yaml`)
- Excludes test directories and common safe patterns
- Configures severity levels for different vulnerability types
- Skips known safe code patterns

### Scan Results
The automated scanner produces:
- Bandit static analysis results
- Dependency vulnerability reports
- File permission checks
- Hardcoded secrets detection
- Unit test security validation

## Security Test Coverage

### Encryption & Data Protection
- Key generation and validation
- Secure key storage and rotation
- Data encryption/decryption
- Tampering detection
- Memory security

### Injection Attack Prevention
- SQL injection prevention
- Command injection prevention
- Path traversal prevention
- Cross-site scripting (XSS) prevention
- Malformed JSON injection handling

### Authentication & Authorization
- Authentication failure handling
- Rate limiting bypass prevention
- Session fixation prevention
- CSRF token validation
- API key exposure prevention

### System Security
- File permission validation
- Hardcoded secrets detection
- Buffer overflow prevention
- Resource exhaustion handling

## CI/CD Integration

### Automated Security Pipeline
```yaml
- name: Security Tests
  run: |
    # Run security unit tests
    python -m pytest tests/security/ --tb=short

    # Run automated security scan
    python tests/security/run_security_scan.py --output security_results.json

    # Check for critical vulnerabilities
    if [ $(jq '.results.bandit.high_severity' security_results.json) -gt 0 ]; then
      echo "Critical security issues found!"
      exit 1
    fi
```

### Security Gates
- **High Severity**: Block deployment (SQL injection, hardcoded secrets, etc.)
- **Medium Severity**: Warning (weak encryption, unsafe patterns)
- **Low Severity**: Monitoring (code quality issues)

## Security Best Practices

### Development Guidelines
1. **Never hardcode secrets** - Use environment variables or secure vaults
2. **Validate all inputs** - Implement proper input sanitization
3. **Use parameterized queries** - Prevent SQL injection
4. **Implement proper error handling** - Don't expose sensitive information
5. **Regular security updates** - Keep dependencies updated

### Testing Guidelines
1. **Test for common vulnerabilities** - OWASP Top 10 coverage
2. **Include security in CI/CD** - Automated security scanning
3. **Regular penetration testing** - External security assessments
4. **Monitor for new threats** - Stay updated on security advisories

## Troubleshooting

### Common Security Test Issues
- **Bandit false positives**: Update `bandit_config.yaml` skips
- **Dependency vulnerabilities**: Update requirements.txt
- **File permission errors**: Fix file permissions in development environment
- **Test timeouts**: Increase timeout values for complex cryptographic tests

### Debugging Security Tests
```bash
# Run with detailed output
python -m pytest tests/security/ -v -s

# Run specific failing test
python -m pytest tests/security/test_encryption_security.py::TestEncryptionSecurity::test_encryption_key_generation -v

# Debug security scanner
python tests/security/run_security_scan.py --debug
```

## Compliance & Standards

The security tests ensure compliance with:
- **OWASP Top 10** - Web application security standards
- **NIST Cybersecurity Framework** - Security best practices
- **ISO 27001** - Information security management
- **PCI DSS** (if handling payment data) - Payment card industry standards

## Contributing

When adding new security tests:
1. Follow existing naming conventions: `test_*.py`
2. Include comprehensive test coverage for new features
3. Update this README with new test categories
4. Ensure tests can run in CI environment
5. Document any new security scanning rules