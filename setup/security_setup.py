#!/usr/bin/env python3
"""
Security Setup Script - Critical Production Security Implementation
Run this script to implement essential security measures before production deployment.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from cryptography.fernet import Fernet

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecuritySetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / '.env'
        self.env_key_file = self.project_root / '.env.key'
        self.env_encrypted_file = self.project_root / '.env.enc'

    def encrypt_environment_variables(self):
        """Encrypt the .env file with sensitive credentials"""
        logger.info("🔐 Encrypting environment variables...")

        if not self.env_file.exists():
            logger.error("❌ .env file not found!")
            return False

        try:
            # Generate encryption key
            key = Fernet.generate_key()

            # Save key securely (in production, store in HSM/vault)
            with open(self.env_key_file, 'wb') as f:
                f.write(key)
            os.chmod(self.env_key_file, 0o600)  # Restrictive permissions

            # Encrypt .env file
            cipher = Fernet(key)
            with open(self.env_file, 'rb') as f:
                data = f.read()

            encrypted_data = cipher.encrypt(data)
            with open(self.env_encrypted_file, 'wb') as f:
                f.write(encrypted_data)

            # Remove plain text file
            os.remove(self.env_file)

            logger.info("✅ Environment variables encrypted successfully")
            logger.warning("⚠️  IMPORTANT: Store .env.key securely (HSM, vault, etc.)")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to encrypt environment: {e}")
            return False

    def setup_secure_logging(self):
        """Set up secure logging with proper permissions"""
        logger.info("📝 Setting up secure logging...")

        try:
            logs_dir = self.project_root / 'logs'
            logs_dir.mkdir(exist_ok=True)
            os.chmod(logs_dir, 0o700)  # Owner read/write/execute only

            log_file = logs_dir / 'abc_application.log'
            if not log_file.exists():
                log_file.touch()

            os.chmod(log_file, 0o600)  # Owner read/write only

            logger.info("✅ Secure logging configured")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup secure logging: {e}")
            return False

    def configure_firewall(self):
        """Configure basic firewall rules"""
        logger.info("🔥 Configuring firewall...")

        try:
            # Check if running on Linux with UFW
            if sys.platform.startswith('linux'):
                commands = [
                    ['sudo', 'ufw', 'default', 'deny', 'incoming'],
                    ['sudo', 'ufw', 'default', 'allow', 'outgoing'],
                    ['sudo', 'ufw', 'allow', 'ssh'],
                    ['sudo', 'ufw', 'allow', '8000'],  # Application port
                    ['sudo', 'ufw', '--force', 'enable']
                ]

                for cmd in commands:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.warning(f"Firewall command failed: {' '.join(cmd)}")
                        logger.warning(f"Output: {result.stderr}")

                logger.info("✅ Firewall configured")
            else:
                logger.info("⏭️  Skipping firewall setup (not Linux)")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to configure firewall: {e}")
            return False

    def secure_ssh_config(self):
        """Secure SSH configuration"""
        logger.info("🔑 Securing SSH configuration...")

        try:
            if sys.platform.startswith('linux'):
                sshd_config = Path('/etc/ssh/sshd_config')
                if sshd_config.exists():
                    # Read current config
                    with open(sshd_config, 'r') as f:
                        content = f.read()

                    # Disable root login if not already disabled
                    if 'PermitRootLogin yes' in content:
                        content = content.replace('PermitRootLogin yes', 'PermitRootLogin no')

                    # Write back
                    with open(sshd_config, 'w') as f:
                        f.write(content)

                    # Restart SSH service
                    subprocess.run(['sudo', 'systemctl', 'restart', 'sshd'], check=True)

                    logger.info("✅ SSH configuration secured")
                else:
                    logger.warning("⚠️  SSH config file not found")
            else:
                logger.info("⏭️  Skipping SSH config (not Linux)")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to secure SSH: {e}")
            return False

    def create_secure_config_loader(self):
        """Create a secure configuration loader"""
        logger.info("⚙️  Creating secure configuration loader...")

        config_loader = '''
# src/utils/secure_config.py
import os
from pathlib import Path
from cryptography.fernet import Fernet

def load_secure_environment():
    """Load encrypted environment variables"""
    project_root = Path(__file__).parent.parent.parent
    key_file = project_root / '.env.key'
    encrypted_file = project_root / '.env.enc'

    if not key_file.exists() or not encrypted_file.exists():
        raise FileNotFoundError("Encrypted environment files not found. Run security_setup.py first.")

    # Load encryption key
    with open(key_file, 'rb') as f:
        key = f.read()

    # Decrypt environment
    cipher = Fernet(key)
    with open(encrypted_file, 'rb') as f:
        encrypted_data = f.read()

    decrypted_data = cipher.decrypt(encrypted_data).decode()

    # Load into environment
    for line in decrypted_data.split('\\n'):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

# Load secure environment on import
load_secure_environment()
'''

        config_file = self.project_root / 'src' / 'utils' / 'secure_config.py'
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            f.write(config_loader)

        logger.info("✅ Secure configuration loader created")
        return True

    def create_audit_logger(self):
        """Create audit logging system"""
        logger.info("📊 Creating audit logging system...")

        audit_logger = '''
# src/utils/audit_logger.py
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
import os

class AuditLogger:
    def __init__(self, audit_file='logs/audit.log'):
        self.audit_file = Path(audit_file)
        self.setup_audit_file()

    def setup_audit_file(self):
        """Ensure audit file exists with secure permissions"""
        self.audit_file.parent.mkdir(exist_ok=True)
        if not self.audit_file.exists():
            self.audit_file.touch()
        os.chmod(self.audit_file, 0o600)  # Owner read/write only

    def log_event(self, event_type, data, user='system'):
        """Log auditable event with integrity hash"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'data': data,
            'user': user,
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        }

        # Create integrity hash
        entry_str = json.dumps(entry, sort_keys=True)
        integrity_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry['integrity_hash'] = integrity_hash

        # Append to audit log
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(entry) + '\\n')

        logging.info(f"Audit logged: {event_type}")

    def log_trade(self, trade_data):
        """Log trade execution"""
        self.log_event('trade_execution', trade_data)

    def log_system_event(self, event_data):
        """Log system events"""
        self.log_event('system_event', event_data)

# Global audit logger instance
audit_logger = AuditLogger()
'''

        audit_file = self.project_root / 'src' / 'utils' / 'audit_logger.py'
        audit_file.parent.mkdir(parents=True, exist_ok=True)

        with open(audit_file, 'w') as f:
            f.write(audit_logger)

        logger.info("✅ Audit logging system created")
        return True

    def run_all_security_measures(self):
        """Run all security hardening measures"""
        logger.info("🚀 Starting comprehensive security hardening...")

        results = []

        # Critical security measures
        results.append(("Environment Encryption", self.encrypt_environment_variables()))
        results.append(("Secure Logging", self.setup_secure_logging()))
        results.append(("Firewall Configuration", self.configure_firewall()))
        results.append(("SSH Security", self.secure_ssh_config()))

        # Supporting systems
        results.append(("Secure Config Loader", self.create_secure_config_loader()))
        results.append(("Audit Logger", self.create_audit_logger()))

        # Summary
        print("\\n" + "="*60)
        print("🔒 SECURITY HARDENING RESULTS")
        print("="*60)

        successful = 0
        for measure, result in results:
            status = "✅" if result else "❌"
            print(f"{status} {measure}")
            if result:
                successful += 1

        print(f"\\n📊 Summary: {successful}/{len(results)} measures implemented successfully")

        if successful == len(results):
            print("\\n🎉 All security measures implemented!")
            print("\\n📋 NEXT STEPS:")
            print("1. Store .env.key in a secure location (HSM/vault)")
            print("2. Test encrypted environment loading")
            print("3. Set up monitoring and alerting")
            print("4. Run security testing")
            print("5. Prepare for compliance audit")
        else:
            print("\\n⚠️  Some security measures failed. Review logs and fix issues before production.")

        return successful == len(results)

def main():
    """Main security setup function"""
    print("🔒 ABC Application Security Hardening Script")
    print("=" * 50)

    # Check if running as appropriate user (Linux only)
    try:
        if os.geteuid() == 0:
            print("⚠️  Running as root - some operations may not work as expected")
            print("   Consider running as regular user with sudo access\n")
    except AttributeError:
        # Windows doesn't have geteuid
        pass

    setup = SecuritySetup()
    success = setup.run_all_security_measures()

    if success:
        print("\\n✅ Security hardening completed successfully!")
        return 0
    else:
        print("\\n❌ Security hardening completed with errors!")
        return 1

if __name__ == "__main__":
    sys.exit(main())