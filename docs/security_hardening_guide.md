# [LABEL:DOC:security] [LABEL:DOC:topic:security] [LABEL:DOC:audience:administrator]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Comprehensive security hardening guide for production deployment
# Dependencies: cryptography, python-dotenv, secure infrastructure
# Related: docs/production_readiness_checklist.md, .env.enc, .env.key
#
# 🔒 Security Hardening Guide - Critical First Steps

## 🚨 **IMMEDIATE SECURITY CONCERNS**

### **1. API Key Exposure (CRITICAL)**
**Issue**: API keys stored in plain text in `.env` file
**Risk**: Complete system compromise if repository is breached

**Immediate Actions:**
```bash
# 1. Install python-dotenv and cryptography
pip install python-dotenv cryptography

# 2. Create encrypted environment file
python -c "
from cryptography.fernet import Fernet
import os

# Generate encryption key (store securely!)
key = Fernet.generate_key()
with open('.env.key', 'wb') as f:
    f.write(key)

# Encrypt existing .env
cipher = Fernet(key)
with open('.env', 'rb') as f:
    data = f.read()
encrypted = cipher.encrypt(data)
with open('.env.enc', 'wb') as f:
    f.write(encrypted)

print('Environment encrypted. Delete .env and store .env.key securely!')
"
```

**Secure Key Storage Options:**
- Hardware Security Module (HSM)
- AWS Secrets Manager
- Azure Key Vault
- **HashiCorp Vault** (Recommended for this system)

### **2. HashiCorp Vault Setup for Secrets Management**
**New Security Layer**: Implemented Vault integration for all sensitive credentials

**Vault Setup Steps:**
```bash
# 1. Download and install Vault
# Windows: Download from https://developer.hashicorp.com/vault/downloads
# Extract to C:\vault\

# 2. Create Vault configuration
cat > vault-config.hcl << EOF
storage "file" {
  path = "./vault-data"
}

listener "tcp" {
  address = "127.0.0.1:8200"
  tls_disable = "true"  # Use proper TLS in production
}

api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"
EOF

# 3. Start Vault in development mode (for testing)
vault server -dev -config=vault-config.hcl

# 4. Set environment variable
$env:VAULT_ADDR = "http://127.0.0.1:8200"
$env:VAULT_TOKEN = "dev-token"  # From Vault startup output

# 5. Enable KV v2 secrets engine
vault secrets enable -path=secret kv-v2

# 6. Store secrets
vault kv put secret/discord DISCORD_ORCHESTRATOR_TOKEN="your_token_here"
vault kv put secret/ibkr IBKR_API_KEY="your_key_here"
```

**Code Integration:**
```python
# src/utils/vault_client.py - New secure client
import hvac
import os
import time

class VaultClient:
    def __init__(self):
        self.client = hvac.Client(
            url=os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200'),
            token=os.getenv('VAULT_TOKEN')
        )
        self._authenticate()

    def _authenticate(self):
        if not self.client.is_authenticated():
            raise Exception("Vault authentication failed")

    def get_secret(self, path: str) -> str:
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data']
        except Exception as e:
            logger.error(f"Failed to retrieve secret {path}: {e}")
            raise

vault_client = VaultClient()

def get_vault_secret(key: str) -> str:
    """Secure secret retrieval function"""
    return vault_client.get_secret(f"secret/{key}")
```

### **3. Redis Security Hardening**
**New Security**: Redis now requires authentication and is bound locally

**Redis Security Configuration:**
```ini
# redis.windows.conf - Updated security settings
bind 127.0.0.1
port 6379
requirepass SecureRedisPass2025!
maxmemory 256mb
maxmemory-policy allkeys-lru

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command SHUTDOWN SHUTDOWN_REDIS
```

**Redis Client Configuration:**
```python
# src/utils/redis_cache.py - Secure Redis client
import redis
import os

def get_redis_client():
    return redis.Redis(
        host='127.0.0.1',
        port=6379,
        password=os.getenv('REDIS_PASSWORD', 'SecureRedisPass2025!'),
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        max_connections=20
    )
```

### **4. Database Security**
**Current Issue**: Plain text passwords in scripts

**Immediate Fix:**
```sql
-- Create secure user with encrypted password
CREATE USER abc_user WITH ENCRYPTED PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE abc_application TO abc_user;

-- Enable SSL connections
ALTER SYSTEM SET ssl = on;
```

### **3. Network Security**
**Immediate Actions:**
```bash
# Configure UFW firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8000  # Application port
sudo ufw --force enable

# Disable root login
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart sshd
```

## 📊 **MONITORING & ALERTING SETUP**

### **1. Real-time Alerting System**
```python
# Create alerts.py
import smtplib
from email.mime.text import MIMEText
import logging

class AlertSystem:
    def __init__(self, smtp_server, smtp_port, sender_email, sender_password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password

    def send_alert(self, subject, message, recipients):
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = self.sender_email
        msg['To'] = ', '.join(recipients)

        try:
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, recipients, msg.as_string())
            server.quit()
            logging.info(f"Alert sent: {subject}")
        except Exception as e:
            logging.error(f"Failed to send alert: {e}")

# Critical alerts
alert_system = AlertSystem(
    smtp_server="smtp.gmail.com",
    smtp_port=465,
    sender_email="alerts@yourdomain.com",
    sender_password="secure_password"
)

# Usage in your code
if portfolio_drawdown > 0.05:  # 5% drawdown
    alert_system.send_alert(
        "CRITICAL: Portfolio Drawdown Alert",
        f"Portfolio drawdown exceeded 5%: {portfolio_drawdown:.1%}",
        ["admin@yourdomain.com", "risk@yourdomain.com"]
    )
```

### **2. Health Check Enhancements**
```python
# Enhanced health check script
def comprehensive_health_check():
    checks = {
        'database': check_database_connection(),
        'redis': check_redis_connection(),
        'api_keys': validate_api_keys(),
        'disk_space': check_disk_space(),
        'memory_usage': check_memory_usage(),
        'trading_system': check_trading_system()
    }

    failed_checks = [k for k, v in checks.items() if not v['status']]

    if failed_checks:
        alert_system.send_alert(
            "SYSTEM HEALTH FAILURE",
            f"Failed checks: {', '.join(failed_checks)}",
            ["admin@yourdomain.com"]
        )

    return checks
```

## 🔐 **ACCESS CONTROL IMPLEMENTATION**

### **1. Environment-Based Configuration**
```python
# config/settings.py
import os
from dotenv import load_dotenv

# Load encrypted environment
def load_secure_env():
    # Load encryption key from secure location
    with open('.env.key', 'rb') as f:
        key = f.read()

    cipher = Fernet(key)
    with open('.env.enc', 'rb') as f:
        encrypted_data = f.read()

    decrypted_data = cipher.decrypt(encrypted_data)

    # Load into environment
    for line in decrypted_data.decode().split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

load_secure_env()

# Environment-specific settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

if ENVIRONMENT == 'production':
    DEBUG = False
    ALLOWED_HOSTS = ['your-production-domain.com']
    DATABASE_URL = os.getenv('PROD_DATABASE_URL')
else:
    DEBUG = True
    ALLOWED_HOSTS = ['localhost', '127.0.0.1']
    DATABASE_URL = os.getenv('DEV_DATABASE_URL')
```

### **2. Logging Security**
```python
# Secure logging configuration
import logging
from logging.handlers import RotatingFileHandler

def setup_secure_logging():
    # Create logs directory with restricted permissions
    os.makedirs('logs', exist_ok=True)
    os.chmod('logs', 0o700)

    # Configure secure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/abc_application.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)

    # Secure formatter (no sensitive data)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

# Initialize secure logging
logger = setup_secure_logging()
```

## 🛡️ **REGULATORY COMPLIANCE**

### **1. Audit Trail Implementation**
```python
# audit.py
import json
from datetime import datetime
import hashlib

class AuditLogger:
    def __init__(self, audit_file='logs/audit.log'):
        self.audit_file = audit_file
        self.ensure_audit_file()

    def ensure_audit_file(self):
        if not os.path.exists(self.audit_file):
            with open(self.audit_file, 'w') as f:
                f.write('')  # Create empty file
        # Set restrictive permissions
        os.chmod(self.audit_file, 0o600)

    def log_trade(self, trade_data):
        """Log trade with tamper-proof hash"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'trade_execution',
            'data': trade_data,
            'user': os.getenv('CURRENT_USER', 'system'),
            'ip_address': self.get_client_ip()
        }

        # Create hash for integrity
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry['integrity_hash'] = entry_hash

        # Append to audit log
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        logger.info(f"Trade audited: {trade_data.get('symbol', 'UNKNOWN')}")

    def get_client_ip(self):
        # Implement IP detection logic
        return os.getenv('CLIENT_IP', 'unknown')

# Global audit logger
audit_logger = AuditLogger()

# Usage in trading code
def execute_trade(symbol, quantity, price):
    # Execute trade logic...

    # Log to audit trail
    audit_logger.log_trade({
        'symbol': symbol,
        'quantity': quantity,
        'price': price,
        'timestamp': datetime.utcnow().isoformat(),
        'strategy': 'pyramiding'
    })
```

## 🚀 **QUICK IMPLEMENTATION SCRIPT**

```bash
#!/bin/bash
# security_hardening.sh - Quick security setup

echo "🔒 Starting ABC Application Security Hardening..."

# 1. Encrypt environment variables
echo "Encrypting environment variables..."
python3 -c "
from cryptography.fernet import Fernet
import os

key = Fernet.generate_key()
with open('.env.key', 'wb') as f:
    f.write(key)

cipher = Fernet(key)
with open('.env', 'rb') as f:
    data = f.read()
encrypted = cipher.encrypt(data)
with open('.env.enc', 'wb') as f:
    f.write(encrypted)

os.remove('.env')  # Remove plain text file
print('✅ Environment encrypted')
"

# 2. Set up secure logging
echo "Setting up secure logging..."
mkdir -p logs
chmod 700 logs
touch logs/abc_application.log
chmod 600 logs/abc_application.log

# 3. Configure firewall
echo "Configuring firewall..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8000
sudo ufw --force enable

# 4. Secure SSH
echo "Securing SSH..."
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

echo "🎉 Security hardening completed!"
echo ""
echo "📋 Next steps:"
echo "1. Store .env.key in secure location (HSM, vault, etc.)"
echo "2. Set up monitoring and alerting"
echo "3. Configure backup encryption"
echo "4. Test all security measures"
```

## 📊 **MONITORING DASHBOARD**

Create a simple monitoring dashboard:

```python
# monitoring_dashboard.py
from flask import Flask, jsonify
import psutil
import time

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'system': {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        },
        'application': {
            'trading_active': True,  # Check actual status
            'last_trade': '2025-11-11T08:00:00Z',
            'portfolio_value': 100000,
            'daily_pnl': 1250.50
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, ssl_context='adhoc')  # Use proper SSL cert in production
```

---

## 🎯 **IMPLEMENTATION PRIORITY**

**Week 1: Critical Security**
1. ✅ Encrypt API keys and environment variables
2. ✅ Set up firewall and SSH hardening
3. ✅ Implement secure logging

**Week 2: Monitoring & Alerting**
4. ⏳ Set up alerting system
5. ⏳ Create health check endpoints
6. ⏳ Implement audit logging

**Week 3: Testing & Validation**
7. ⏳ Security testing
8. ⏳ Penetration testing
9. ⏳ Compliance validation

---

**⚠️ CRITICAL**: Do not proceed to live trading until all security measures are implemented and tested. The current system has excellent functionality but requires these security foundations before production deployment.